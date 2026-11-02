from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Paired final decoder-hidden cosine / native eager attention on fixed 300 prompts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

SOURCE = ROOT
sys.path.insert(0, str(SOURCE))
from input_pipeline import encode_sample, QueryCapture, write_json
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class HiddenCapture(QueryCapture):
    def __init__(self, model):
        # Split independent heads only: the native BF16 eager computation is preserved.
        super().__init__(model, full_eager_limit=4800, chunk_size=4)
        self.native = self.original
        self.original = self.record_native
        self.handles = [layer.register_forward_hook(self.hook(i))
                        for i, layer in enumerate(model.model.language_model.layers)]

    def configure(self, info, device):
        super().configure(info, device)
        self.cosine = [None] * len(self.layers)
        self.token_cosine = [None] * len(self.layers)
        self.token_attention = [None] * len(self.layers)
        self.norm_min = []

    def forward(self, module, *args, **kwargs):
        self.active_layer = self.layers.get(id(module))
        return super().forward(module, *args, **kwargs)

    def record_native(self, module, *args, **kwargs):
        out, weights = self.native(module, *args, **kwargs)
        if id(module) in self.layers:
            # Mean heads and query rows, preserving every individual video token.
            v = weights[0].index_select(1, self.qpos).index_select(2, self.vpos).float()
            value = (v.sum(0).mean(0) / 32).cpu().numpy()
            layer = self.layers[id(module)]
            if self.token_attention[layer] is None:
                self.token_attention[layer] = value
            else:
                self.token_attention[layer] += value
        return out, weights

    def hook(self, layer):
        def capture(module, args, output):
            # Block output, after attention residual, MLP, and MLP residual.
            # Before external DeepStack addition (L0..2) and final model RMSNorm.
            q = output[0].index_select(0, self.qpos).float()
            v = output[0].index_select(0, self.vpos).float()
            qnorm, vnorm = q.norm(dim=-1), v.norm(dim=-1)
            assert (qnorm > 0).all() and (vnorm > 0).all()
            score = F.normalize(q, dim=-1) @ F.normalize(v, dim=-1).T
            assert torch.isfinite(score).all() and score.abs().max() <= 1.00001
            t, hw = self.info['video_grid_thw'][0], self.info['spatial_tokens_per_bin']
            self.cosine[layer] = score.reshape(len(self.qpos), t, hw).mean(-1).cpu().numpy()
            self.token_cosine[layer] = score.mean(0).cpu().numpy()
            self.norm_min.append([float(qnorm.min()), float(vnorm.min())])
        return capture

    def restore(self):
        self.modeling.eager_attention_forward = self.native
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260930)
    samples = json.loads((relocate_path(GENERATION / 'samples.json')).read_text())
    results = HIDDEN
    if (relocate_path(results/'samples.json')).exists():
        assert json.loads((relocate_path(results/'samples.json')).read_text())==samples, 'Existing sample manifest must not change'
    selected=[s for s in samples if not args.only or s['sample_key'] in args.only]
    if all((relocate_path(results/s['dataset']/s['sample_key']/'validation.json')).exists() for s in selected):
        print('All selected samples already complete; no model load required',flush=True)
        return
    write_json(relocate_path(results / 'samples.json'), samples)
    write_json(relocate_path(results / 'run_config.json'), dict(
        checkpoint='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        source=str(SOURCE), n_samples=len(samples), layer_band=[8,21], dtype='bfloat16',
        output='decoder block output after both residuals; before external DeepStack L0..2 and model final RMSNorm',
        cosine='FP32 pairwise cosine(query output, video-token output), mean spatial tokens per temporal bin',
        attention='native eager; <=4800 tokens full heads, longer inputs 4-head chunks; all causal keys in softmax, then video selection',
        aggregation='mean cosine over L8..21 and query rows, softmax across time; mean attention over same layers/heads/rows, normalize video mass',
        temperatures=[0.01,0.02,0.05,0.1,0.2,1.0], primary_temperature=0.1,
        torch=torch.__version__, inference_seed=20260930,
        sample_manifest_reused=True, sample_selection_seed=20260929,
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [relocate_path(Path(__file__)), relocate_path(ROOT/'_lib/input_pipeline.py'), relocate_path(GENERATION / 'samples.json')]}))
    proc = AutoProcessor.from_pretrained(relocate_path(SOURCE/'model'), local_files_only=True)
    print('Loading model, native eager / full heads up to 4800 tokens, otherwise 4-head chunks', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        relocate_path(SOURCE/'model'), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', device_map='cuda:0').eval()
    capture = HiddenCapture(model)
    try:
        for sample in samples:
            if args.only and sample['sample_key'] not in args.only:
                continue
            folder = relocate_path(results/sample['dataset']/sample['sample_key'])
            if (relocate_path(folder/'validation.json')).exists():
                print('Complete', sample['sample_key'], flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            start = time.time()
            inputs, info, expanded = encode_sample(sample, proc)
            previous_folder = relocate_path(GENERATION/sample['dataset']/sample['sample_key'])
            old_info = json.loads((relocate_path(previous_folder/'metadata.json')).read_text())
            for field in ['query_positions','video_positions','sampled_timestamps','video_grid_thw','sequence_length']:
                assert info[field] == old_info[field], field
            assert expanded == (relocate_path(previous_folder/'expanded_input.txt')).read_text()
            write_json(relocate_path(folder/'metadata.json'), info)
            inputs = inputs.to('cuda:0')
            assert inputs['attention_mask'].all()
            capture.configure(info, 'cuda:0')
            torch.cuda.reset_peak_memory_stats()
            print('Running',sample['sample_key'],info['sequence_length'],flush=True)
            with torch.inference_mode():
                output = model.model(**inputs, use_cache=False, return_dict=True)
            del output, inputs
            attention = np.stack(capture.temporal)
            cosine = np.stack(capture.cosine)
            token_attention = np.stack(capture.token_attention)
            token_cosine = np.stack(capture.token_cosine)
            previous = np.load(relocate_path(previous_folder/'attention.npz'))['query_temporal']
            assert attention.shape == previous.shape
            assert np.array_equal(attention,previous), 'Same prompt/native eager must reproduce prior attention exactly'
            t, hw = info['video_grid_thw'][0], info['spatial_tokens_per_bin']
            np.testing.assert_allclose(token_attention.reshape(36,t,hw).sum(-1),attention.mean((1,2)),atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(token_cosine.reshape(36,t,hw).mean(-1),cosine.mean(1),atol=2e-7,rtol=2e-6)
            # Attention head/query axes retained in prior bit-identical artifacts.
            np.savez_compressed(relocate_path(folder/'distributions.npz'), cosine_query_frame=cosine,
                cosine_video_token=token_cosine, attention_query_frame=attention.mean(1),
                attention_video_token=token_attention, sampled_timestamps=info['sampled_timestamps'])
            write_json(relocate_path(folder/'validation.json'), dict(
                prior_attention_bit_identical=True, input_positions_and_text_identical=True,
                max_abs_attention_difference=float(np.max(np.abs(attention-previous))),
                cosine_shape=list(cosine.shape), token_shape=list(token_cosine.shape),
                output_norm_min_by_layer=capture.norm_min, attention_validation=capture.validation,
                elapsed_seconds=time.time()-start, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            print('Saved',sample['sample_key'],'seconds',round(time.time()-start,2),flush=True)
            torch.cuda.empty_cache()
    finally:
        capture.restore()


if __name__ == '__main__':
    main()
