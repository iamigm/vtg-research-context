"""Official TimeLens input preparation, shared without attention instrumentation."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from types import SimpleNamespace
import copy,json,os,time,hashlib,threading
MODEL=relocate_path(Path('/home/irteam/backbones/models/Qwen3-VL-8B-Instruct'))
ORDERS=['video_query','query_video']
DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens']

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    write_result_text(tmp, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



class OrderProcessor:
    def __init__(self, processor, reverse=False):
        self.processor = processor
        self.reverse = reverse

    def apply_chat_template(self, messages, **kwargs):
        if self.reverse:
            messages[0]['content'].reverse()
        self.messages = copy.deepcopy(messages)
        return self.processor.apply_chat_template(messages, **kwargs)

    def __call__(self, *args, **kwargs):
        self.metadata = copy.deepcopy(kwargs['video_metadata'][0])
        return self.processor(*args, **kwargs)


def prepare_pair(sample, processor, fps=2):
    import torch
    import evaluation.utils as upstream
    args = SimpleNamespace(model_path=str(MODEL), min_tokens=64, total_tokens=14336, fps=fps)
    original_vision = upstream.process_vision_info
    cache = []

    def vision(messages, **kwargs):
        video = next(x for x in messages[0]['content'] if x['type'] == 'video')
        if not cache:
            cache.append((copy.deepcopy(video), original_vision(messages, **kwargs)))
        assert video == cache[0][0]
        images, videos, extra = cache[0][1]
        # Reuse the decoded frames but give each processor an independent metadata dict.
        return images, [(tensor, copy.deepcopy(meta)) for tensor, meta in videos], copy.deepcopy(extra)

    upstream.process_vision_info = vision
    try:
        proxies = [OrderProcessor(processor, order == 'query_video') for order in ORDERS]
        pair = [upstream.GroundingDataset([sample], proxy, args)[0]['inputs'] for proxy in proxies]
    finally:
        upstream.process_vision_info = original_vision
    a, b = pair
    for key in a:
        if key != 'input_ids':
            assert torch.equal(a[key], b[key]), key
    prompt = upstream.GROUNDER_PROMPT.format(sample['query'])
    tokens = processor.tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = [x['input_ids'][0].tolist() for x in pair]

    def locate(seq):
        matches = [i for i in range(len(seq)-len(tokens)+1) if seq[i:i+len(tokens)] == tokens]
        assert len(matches) == 1
        return matches[0]

    qa, qb = map(locate, ids)
    assert ids[1] == ids[0][:qb] + tokens + ids[0][qb:qa] + ids[0][qa+len(tokens):]
    video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    positions = [[i for i, token in enumerate(row) if token == video_id] for row in ids]
    assert max(positions[0]) < qa and qb+len(tokens) <= min(positions[1])
    meta = proxies[0].metadata
    timestamps = processor._calculate_timestamps(copy.deepcopy(meta['frames_indices']), meta['fps'], processor.video_processor.merge_size)
    grid = a['video_grid_thw'][0].tolist()
    assert len(timestamps) == grid[0]
    return pair, dict(prompt=prompt, video_grid_thw=grid, timestamps=timestamps,
                     duration_video=meta['total_num_frames']/meta['fps'],
                     decoder_metadata=meta, video_positions=dict(zip(ORDERS, positions)),
                     sequence_length=len(ids[0]), non_text_inputs_bitwise_identical=True,
                     exact_token_block_swap=True, decoded_video_reused=True)


