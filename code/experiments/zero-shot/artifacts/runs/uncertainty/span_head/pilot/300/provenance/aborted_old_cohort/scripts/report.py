"""Compare the exact saved pilot with restricted-head outputs, without clipping."""
import collections
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/irteam/analysis/failure_mode_timestamp/failure_cases')
sys.path.insert(0, str(SOURCE / 'scripts'))
from metrics import measure
sys.path.insert(0, '/home/irteam/experiments/mtla_attention/vendor/TimeLens')
from timelens.utils import extract_time, iou


def summarize(rows):
    return dict(n=len(rows), mIoU=100*float(np.mean([r['iou'] for r in rows])), **{f'R1@{t}':100*float(np.mean([r['iou'] >= t for r in rows])) for t in [.3,.5,.7]}, parse_failures=sum(r['parse_failed'] for r in rows), invalid_intervals=sum(r['failure_type']=='invalid_interval' for r in rows), multiple_spans=sum(len(r['timestamps'])>1 for r in rows), no_overlap=sum(r['failure_type']=='no_overlap' for r in rows), out_of_bounds=sum(r['out_of_bounds'] for r in rows), mean_output_tokens=float(np.mean([r['output_tokens'] for r in rows])), token_limit=sum(r.get('hit_token_limit',False) for r in rows))


def bootstrap(rows):
    # Resample videos, keeping queries from the same video together.
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['video_id']].append(r)
    groups = list(grouped.values())
    rng = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        draw = [r for i in rng.integers(0, len(groups), len(groups)) for r in groups[i]]
        values.append([100*np.mean([r['delta_iou'] for r in draw]),100*np.mean([r['delta_success_05'] for r in draw])])
    low, high = np.percentile(values, [2.5,97.5], axis=0)
    return dict(mIoU=[float(low[0]),float(high[0])], R1_05=[float(low[1]),float(high[1])])


def main():
    dest = ROOT / 'reports'
    dest.mkdir(exist_ok=True)
    source_file = SOURCE / 'reports/pilot_100/setting_case_metrics.json'
    old = json.loads(source_file.read_text())
    manifest = json.loads((SOURCE / 'data/samples_100.json').read_text())
    expected = {r['sample_id'] for r in manifest}
    paired = []
    summary = {}
    validations = []
    table = []
    complete = True
    for model in ['qwen25','qwen3']:
        baseline = {r['sample_id']:r for r in old if r['setting']==model+'_native'}
        rows = [json.loads(line) for path in sorted((ROOT/'results'/model).glob('shard*/predictions.jsonl')) for line in path.read_text().splitlines()]
        assert len(rows) == len({r['sample_id'] for r in rows}), 'Duplicate predictions'
        complete &= {r['sample_id'] for r in rows} == expected
        cfgs = [json.loads(p.read_text()) for p in (ROOT/'results'/model).glob('shard*/config.json')]
        if not cfgs:
            continue
        allowed = {t['id'] for t in cfgs[0]['allowed_tokens']}
        for path in (ROOT/'results'/model).glob('shard*/baseline_reproduction.jsonl'):
            validations.extend(json.loads(line) | {'model':model} for line in path.read_text().splitlines())
        for r in rows:
            assert set(r['generated_ids']) <= allowed
            parsed = extract_time(r['answer'])
            assert r['timestamps'] == [[round(s),round(e)] for s,e in parsed]
            r.update(measure(r))
            b = baseline[r['sample_id']]
            b.update(measure(b))
            assert r['input_tokens']==b['input_tokens'] and r['video_grid_thw']==b['video_grid_thw']
            p = r['timestamps'][0] if r['timestamps'] else None
            if p and p[1]>p[0]:
                assert abs(iou(r['span'][0],p)-r['iou'])<1e-12
            paired.append(dict(model=model,dataset=r['dataset'],sample_id=r['sample_id'],video_id=r['video_id'],query=r['query'],gt=r['span'][0],baseline_answer=b['answer'],restricted_answer=r['answer'],baseline_timestamps=b['timestamps'],restricted_timestamps=r['timestamps'],baseline_iou=b['iou'],restricted_iou=r['iou'],delta_iou=r['iou']-b['iou'],delta_success_05=int(r['iou']>=.5)-int(b['iou']>=.5),baseline_failure=b['failure_type'],restricted_failure=r['failure_type'],input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],hit_token_limit=r['hit_token_limit']))
        for dataset in ['charades-timelens','qvhighlights-timelens']:
            new = [r for r in rows if r['dataset']==dataset]
            if not new:
                continue
            base = [baseline[r['sample_id']] for r in new]
            pairs = [r for r in paired if r['model']==model and r['dataset']==dataset]
            bs, ns = summarize(base), summarize(new)
            delta = {k:ns[k]-bs[k] for k in ['mIoU','R1@0.3','R1@0.5','R1@0.7']}
            stats = dict(baseline=bs,restricted=ns,delta_percentage_points=delta,success_to_failure=sum(r['delta_success_05']==-1 for r in pairs),failure_to_success=sum(r['delta_success_05']==1 for r in pairs),identical_spans=sum(r['baseline_timestamps']==r['restricted_timestamps'] for r in pairs),bootstrap_95ci_percentage_points=bootstrap(pairs))
            summary[model+'/'+dataset] = stats
            for label, s in [('Original zero-shot', bs),('Restricted head', ns)]:
                table.append([model,dataset,label,s['n'],*[f'{s[k]:.2f}' for k in ['R1@0.3','R1@0.5','R1@0.7','mIoU']],s['parse_failures'],s['invalid_intervals'],s['mean_output_tokens']])
    result = dict(complete=complete,groups=summary,comparison='Saved original zero-shot vs same prompt and visual input with alphabet-restricted output head; no training or grammar enforcement',baseline_source=str(source_file),baseline_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),baseline_reproduction=validations,validation=dict(unique_sample_ids=True,all_output_tokens_allowed=True,input_token_counts_and_video_grids_match=True,official_parser_rounding_and_iou_match=True),speed_comparison='Historical timings are not comparable to concurrent runs; only output token counts are compared.')
    (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (dest/'paired.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2)+'\n')
    if paired:
        with (dest/'paired.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    headers=['Model','Dataset','Setting','n','R@1@0.3','R@1@0.5','R@1@0.7','mIoU','Parse failure','Invalid interval','Mean output tokens']
    def cells(vals,tag='td'):
        return '<tr>'+''.join(f'<{tag}>'+html.escape(str(v))+f'</{tag}>' for v in vals)+'</tr>'
    page='<!doctype html><meta charset="utf-8"><title>Restricted LM head pilot</title><style>body{font:15px system-ui;margin:32px;max-width:1500px}table{border-collapse:collapse;width:100%;margin-bottom:28px}td,th{border:1px solid #ccc;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap}input{padding:10px;width:70%}</style><h1>Span-only LM head vs original zero-shot</h1>'
    page+='<p>Same 100 queries per dataset; Qwen2.5-VL-7B and Qwen3-VL-8B native inputs. No finetuning. Original prompts, SDPA, BF16, Transformers 4.57.6, greedy generation, 512-token cap. Output head projects only token rows composed of digits, decimal point, comma, hyphen, brackets, colon, space/newline, plus EOS. No grammar, range correction, or clipping.</p>'
    page+='<p>Original baseline directory is unavailable. Original responses and metrics are recovered from the preserved setting_case_metrics.json report. Baseline reproduction checks are recorded below. Historical latency is not a controlled speed benchmark.</p>'
    page+=f'<p>Complete: {complete}. <a href="summary.json">Metrics and validation</a> · <a href="paired.csv">Paired CSV</a> · <a href="paired.json">Paired JSON</a></p><table>'+cells(headers,'th')+''.join(cells(x) for x in table)+'</table>'
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        x = np.arange(2)
        for i, model in enumerate(['qwen25','qwen3']):
            for j, metric in enumerate(['R1@0.5','mIoU']):
                ax = axes[i,j]
                for offset, setting, color in [(-.18,'baseline','#778899'),(.18,'restricted','#2874a6')]:
                    vals=[summary[model+'/'+ds][setting][metric] for ds in ['charades-timelens','qvhighlights-timelens']]
                    bars=ax.bar(x+offset,vals,.36,label=setting,color=color)
                    ax.bar_label(bars,fmt='%.1f')
                ax.set_xticks(x,['Charades-TimeLens','QVHighlights-TimeLens'])
                ax.set_ylim(0,100)
                ax.set_title(model+' · '+metric+' (%)')
                ax.legend()
        fig.suptitle('526-token LM head vs original zero-shot | same 100 queries per dataset')
        fig.tight_layout()
        fig.savefig(dest/'comparison.png',dpi=160)
        fig.savefig(dest/'comparison.pdf')
        plt.close(fig)
        page+='<p><img src="comparison.png" style="max-width:100%"></p>'
    page+='<h2>Paired changes</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>Original baseline reproduction</h2><pre>'+html.escape(json.dumps(validations,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>Every query</h2><input id="search" placeholder="Search model, query, sample"><table id="cases">'+cells(['Model','Sample','Query','GT','Original','Restricted','Original IoU','Restricted IoU'],'th')
    page+=''.join(cells([r['model'],r['sample_id'],r['query'],r['gt'],r['baseline_answer'],r['restricted_answer'],round(r['baseline_iou'],4),round(r['restricted_iou'],4)]) for r in paired)+'</table><script>search.oninput=()=>document.querySelectorAll("#cases tr:has(td)").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(search.value.toLowerCase()))</script>'
    (dest/'index.html').write_text(page)
    print(json.dumps(dict(complete=complete,groups=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
