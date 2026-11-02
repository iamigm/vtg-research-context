from result_layout import relocate_path, write_result_text, result_glob
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Validate all generated answers, attention alignment, plots and source preservation."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from comparison import ROOT, RESULTS, DATASETS
sys.path.insert(0,str(relocate_path(ROOT/"python_deps")))
from pypdf import PdfReader
from query_analysis import verify_provenance


def main():
    samples=json.loads((relocate_path(RESULTS/"samples.json")).read_text())
    previous=ROOT
    assert samples==json.loads((relocate_path(QUERY / 'samples.json')).read_text())
    counts={};cap=0;unparsed=[];max_error=0.;decode_layers=0;max_memory=0.
    for sample in samples:
        folder=relocate_path(RESULTS/ sample["dataset"] / sample["sample_key"])
        answer=json.loads((relocate_path(folder/"answer.json")).read_text())
        meta=json.loads((relocate_path(folder/"metadata.json")).read_text())
        v=json.loads((relocate_path(folder/"validation.json")).read_text())
        n=len(answer["token_ids"]);q=len(meta["query_positions"]);t=meta["video_grid_thw"][0]
        with np.load(relocate_path(folder/"attention.npz")) as data:
            assert data["query_temporal"].shape==(36,32,q,t)
            assert data["answer_temporal"].shape==data["predictive_temporal"].shape==(36,32,n,t)
            for scope in ["query","answer","predictive"]:
                temp=data[f"{scope}_temporal"];mass=data[f"{scope}_key_mass"]
                assert np.isfinite(temp).all() and (temp>=0).all()
                assert np.allclose(temp.sum(-1),mass[...,0],atol=2e-6)
                assert np.max(np.abs(mass.sum(-1)-1),initial=0)<.0041
            with np.load(relocate_path(QUERY/ sample["dataset"] / sample["sample_key"]/"attention.npz")) as old:
                assert np.array_equal(data["query_temporal"],old["temporal_attention"])
            # Prediction of token j+1 uses the same row that consumes token j.
            assert np.array_equal(data["predictive_temporal"][:,:,1:],data["answer_temporal"][:,:,:-1])
        p=meta["sequence_length"]
        assert answer["capture_positions"]==list(range(p-1,p+n))
        assert v["query_matches_previous_bitwise"] and v["capture_steps"]==n+1
        assert max(answer["timestamp_token_indices"],default=-1)<n
        assert answer["generated_tokens_count"]<=128
        assert answer["generated_tokens_count"]==n+int(answer["stopped_on_eos"])
        if answer["truncated"]:assert n==128;cap+=1
        if not answer["parsed_spans"]:unparsed.append(dict(sample_key=sample["sample_key"],answer=answer["text"]))
        for name in ["comparison.png","predictive.png"]:
            with Image.open(relocate_path(folder/name)) as image:image.verify()
        assert len(PdfReader(relocate_path(folder/"comparison.pdf")).pages)==1
        assert (relocate_path(folder/"PLOTTED")).exists()
        assert (relocate_path(folder/"expanded_input.txt")).read_bytes()==(relocate_path(QUERY/ sample["dataset"] / sample["sample_key"]/"expanded_input.txt")).read_bytes()
        counts[sample["dataset"]]=counts.get(sample["dataset"],0)+1
        max_error=max(max_error,v["attention_row_sum_max_error"])
        max_memory=max(max_memory,v["peak_gpu_allocated_gib"])
        decode_layers+=v["capture_steps"]*36
    assert counts==dict.fromkeys(DATASETS,100)
    pdf_counts={}
    for dataset in DATASETS:
        for suffix,n in [("detail",100),("overview",17)]:
            name=f"{dataset}_{suffix}.pdf";pdf_counts[name]=len(PdfReader(relocate_path(RESULTS/name)).pages);assert pdf_counts[name]==n
    pdf_counts["comparison_report.pdf"]=len(PdfReader(relocate_path(RESULTS/"comparison_report.pdf")).pages)
    assert pdf_counts["comparison_report.pdf"]==300
    for name in ["summary","layer_comparison"]:
        with Image.open(relocate_path(RESULTS/f"{name}.png")) as image:image.verify()
        assert len(PdfReader(relocate_path(RESULTS/f"{name}.pdf")).pages)==1
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in {"href","src"} and v and not v.startswith(("http","#")):
                    assert (relocate_path(RESULTS/v)).is_file(),v
    page=(relocate_path(RESULTS/"report.html")).read_text();Links().feed(page);assert page.count("<article ")==300
    provenance=verify_provenance(samples)
    result=dict(provenance,samples_per_dataset=counts,query_matches_previous_bitwise_all_300=True,
        predictive_consumed_row_alignment_all_300=True,validated_layer_steps=decode_layers,
        max_attention_row_sum_error=max_error,peak_gpu_allocated_gib=max_memory,
        truncated_answers=cap,answers_without_valid_span=unparsed,pdf_page_counts=pdf_counts,
        all_plots_valid=True,source_measurements_verified=True)
    write_result_text((relocate_path(RESULTS/"final_validation.json")), json.dumps(result,indent=2)+"\n")
    files=list(result_glob(ROOT, "*.py"))+list(result_glob(ROOT, "*.md"))+[p for p in RESULTS.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_result_text((relocate_path(GENERATION / 'artifact_sha256.json')), json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
