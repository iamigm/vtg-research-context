"""Validate full sample coverage and compare our scores to upstream IoU."""
from result_layout import relocate_path, write_result_text
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=relocate_path(Path(__file__)).resolve().parents[1];sys.path.insert(0,str(relocate_path(ROOT/'vendor/TimeLens')))
from timelens.utils import iou,extract_time
from metrics import tiou

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='pilot_100');ap.add_argument('--manifest',default='data/samples_100.json');ap.add_argument('--allow-incomplete',action='store_true');a=ap.parse_args()
    expected={r['sample_id']:r for r in json.loads((relocate_path(ROOT/a.manifest)).read_text())};report={}
    for model in ['qwen25']:
        path=relocate_path(ROOT/'results'/a.run/model/'predictions.jsonl')
        if not path.exists():
            report[model]={'complete':False,'count':0,'missing':len(expected)};continue
        rows=[json.loads(s) for s in path.read_text().splitlines() if s];ids=[r['sample_id'] for r in rows]
        assert len(ids)==len(set(ids)),'Duplicate sample ID';assert set(ids)<=expected.keys()
        deltas=[];exports={};failed_parse=0
        for r in rows:
            e=expected[r['sample_id']]
            for k in ['query','span','duration','video_path']:assert r[k]==e[k],(r['sample_id'],k)
            raw=extract_time(r['answer']);assert [list(v) for v in raw]==r['parsed_timestamps']
            assert [[round(s),round(e)] for s,e in raw]==r['timestamps']
            pred=r['timestamps'][0] if r['timestamps'] else [-100,-100]
            deltas.append(abs(tiou(r['span'][0],pred)-iou(r['span'][0],pred)))
            failed_parse+=not bool(raw)
            key=f'{relocate_path(Path(r["video_path"])).name}>>>{r["query"]}>>>{r["span"][0]}'
            assert key not in exports
            exports[key]={'timestamps':r['timestamps'],'answers':r['answer'],'duration':r['duration']}
        target=relocate_path(path.parent/'official_predictions.json');write_result_text(target, json.dumps(exports,indent=2,ensure_ascii=False))
        assert max(deltas,default=0)<1e-12
        report[model]={'complete':set(ids)==expected.keys(),'count':len(ids),'missing':len(expected)-len(ids),'max_upstream_iou_difference':max(deltas,default=0),'parse_failures_included':failed_parse,'metadata_and_parser_verified':True}
        result=subprocess.run([sys.executable,str(relocate_path(ROOT/'vendor/TimeLens/evaluation/compute_metrics.py')),'-f',str(target)],cwd=relocate_path(ROOT/'vendor/TimeLens'),env={**__import__('os').environ,'PYTHONPATH':str(relocate_path(ROOT/'vendor/TimeLens'))},capture_output=True,text=True,check=True)
        write_result_text((relocate_path(path.parent/'official_metric_output.txt')), result.stdout)
    complete=all(r['complete'] for r in report.values());report={'complete':complete,'models':report}
    dest=relocate_path(ROOT/'reports'/a.run);dest.mkdir(parents=True,exist_ok=True);write_result_text((relocate_path(dest/'validation.json')), json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not complete and not a.allow_incomplete:raise SystemExit('Incomplete coverage')
if __name__=='__main__':main()
