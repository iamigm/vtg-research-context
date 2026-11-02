from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Common temporal curves and paired metrics for live query/answer attention."""
from pathlib import Path
import json
import numpy as np
from query_analysis import merged_gt, overlap_fractions

RESULTS = GENERATION
DATASETS = ["Charades-STA", "QVHighlights", "ActivityNet-Captions"]
LABELS = {"query": "Input query", "answer": "Answer: all tokens", "coordinates": "Answer: timestamp digits",
          "first_coordinate": "Answer: first coordinate", "predictive": "Predicting answer tokens",
          "predictive_coordinates": "Predicting timestamp digits"}
COLORS = {"query":"#2474b5", "answer":"#d46a28", "coordinates":"#8a4caf", "first_coordinate":"#aa6677",
          "predictive":"#65717e", "predictive_coordinates":"#bb8787"}


def load_sample(sample):
    folder = relocate_path(RESULTS / sample["dataset"] / sample["sample_key"])
    meta = json.loads((relocate_path(folder / "metadata.json")).read_text())
    answer = json.loads((relocate_path(folder / "answer.json")).read_text())
    with np.load(relocate_path(folder / "attention.npz")) as data:
        arrays = {key:data[key] for key in data.files}
    return folder, meta, answer, arrays


def curves_for(answer, arrays):
    curves = {"query":arrays["query_temporal"][8:22].mean((0,1,2))}
    if arrays["answer_temporal"].shape[2]:
        curves["answer"] = arrays["answer_temporal"][8:22].mean((0,1,2))
        curves["predictive"] = arrays["predictive_temporal"][8:22].mean((0,1,2))
    nums = answer["timestamp_token_indices"]
    first = answer["first_coordinate_indices"]
    if nums:
        curves["coordinates"] = arrays["answer_temporal"][8:22][:,:,nums].mean((0,1,2))
        curves["predictive_coordinates"] = arrays["predictive_temporal"][8:22][:,:,nums].mean((0,1,2))
    if first:
        curves["first_coordinate"] = arrays["answer_temporal"][8:22][:,:,first].mean((0,1,2))
    return {key:value.astype(float) for key,value in curves.items()}


def tiou(a,b):
    inter = max(0, min(a[1],b[1])-max(a[0],b[0]))
    union = max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union > 0 else 0.0


def evaluate_sample(sample):
    folder, meta, answer, arrays = load_sample(sample)
    curves = curves_for(answer, arrays)
    t = np.array(meta["sampled_timestamps"])
    duration = meta["duration_video"]
    edges = np.r_[0, (t[:-1]+t[1:])/2, duration]
    gt = merged_gt(meta["gt_windows"],duration)
    fractions = overlap_fractions(edges,gt)
    coverage = sum(e-s for s,e in gt)/duration
    raw_pred = answer["parsed_spans"]
    # Same single-span selection as MTLA for Charades and its ActivityNet extension.
    pred = raw_pred if sample["dataset"] == "QVHighlights" else raw_pred[:1]
    pred_valid = merged_gt(pred,duration)
    pred_fraction = overlap_fractions(edges,pred_valid)
    pred_coverage = sum(e-s for s,e in pred_valid)/duration
    row = dict(dataset=sample["dataset"], sample_key=sample["sample_key"], qid=sample["qid"], query=sample["query"],
               response=answer["text"], generated_spans=raw_pred, protocol_predictions=pred,
               gt_windows=meta["gt_windows"], gt_time_coverage=coverage,
               parsed=bool(raw_pred), truncated=answer["truncated"], answer_tokens=len(answer["token_ids"]),
               timestamp_tokens=len(answer["timestamp_token_indices"]),
               gt_outside_video=any(e>duration for s,e in meta["gt_windows"]),
               prediction_outside_video=any(s<0 or e>duration for s,e in pred),
               best_pair_tiou=max([tiou(p,g) for p in pred for g in meta["gt_windows"]],default=0.0),
               first_span_best_tiou=max([tiou(pred[0],g) for g in meta["gt_windows"]],default=0.0) if pred else 0.0,
               gt_recall_at_05=float(np.mean([max([tiou(p,g) for p in pred],default=0)>=.5 for g in meta["gt_windows"]])),
               scopes={})
    for scope, raw in curves.items():
        p=raw/raw.sum()
        peak=int(raw.argmax())
        gt_mass=float(p@fractions)
        row["scopes"][scope]=dict(total_video_attention=float(raw.sum()), gt_mass=gt_mass,
            gt_mass_excess=gt_mass-coverage,
            peak_time=float(t[peak]), peak_in_gt=any(s<=t[peak]<=e for s,e in gt),
            peak_in_first_10_percent=bool(t[peak]<duration*.1),
            predicted_interval_mass=float(p@pred_fraction), predicted_interval_excess=float(p@pred_fraction)-pred_coverage,
            peak_in_prediction=any(s<=t[peak]<=e for s,e in pred_valid))
    for scope in ["answer","coordinates","first_coordinate","predictive","predictive_coordinates"]:
        if scope in row["scopes"]:
            p=curves[scope]/curves[scope].sum(); q=curves["query"]/curves["query"].sum()
            row["scopes"][scope]["gt_mass_gain_over_query"]=row["scopes"][scope]["gt_mass"]-row["scopes"]["query"]["gt_mass"]
            row["scopes"][scope]["total_variation_from_query"]=float(np.abs(p-q).sum()/2)
    # Fixed band per-token heatmap values, and layer-by-layer paired alignment.
    layer_curves={"query": arrays["query_temporal"].mean((1,2))}
    if "answer" in curves:
        layer_curves["answer"]=arrays["answer_temporal"].mean((1,2))
    if "coordinates" in curves:
        layer_curves["coordinates"]=arrays["answer_temporal"][:,:,answer["timestamp_token_indices"]].mean((1,2))
    layers={}
    for key,a in layer_curves.items():
        prob=a/a.sum(-1,keepdims=True)
        peaks=t[a.argmax(-1)]
        layers[key]=dict(gt_mass=(prob@fractions).tolist(),
                         peak_in_gt=[any(s<=v<=e for s,e in gt) for v in peaks])
    row["layers"]=layers
    return row
