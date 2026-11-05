"""CPU-only paired evaluation of interval clipping on fixed rollout/reconsideration outputs."""
from pathlib import Path
import collections
import argparse
import re
import csv
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/runs/rollout/deterministic/pilot/900'
COHORT=ROOT/'artifacts/common/cohorts/nested_pilot_v1/pilot_900/samples.json'
BUDGETS=[1,4,8,16,32]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def interval(candidate):
    p=candidate.get('parsed')
    return [p['start'],p['end']] if candidate.get('stopped_on_eos') and p and p.get('positive') else None


def clipped(span,duration):
    if span is None:return None
    start=max(0.,min(duration,span[0]));end=max(0.,min(duration,span[1]))
    return [start,end] if start<end else None


def intersection(a,b):return max(0.,min(a[1],b[1])-max(a[0],b[0])) if a and b else 0.
def iou(a,b):return intersection(a,b)/max(1e-12,max(a[1],b[1])-min(a[0],b[0])) if a and b else 0.
def mean(xs):return statistics.mean(xs) if xs else None
def fmt(x,digits=4):return '—' if x is None else f'{x:.{digits}f}'


def relation(a,b):
    if not a or not b:return 'invalid'
    if a==b:return 'identical'
    return 'overlapping' if intersection(a,b)>0 else 'disjoint'


def record(row,source,method,budget,raw,after,before_index=None,after_index=None):
    a=iou(raw,row['span'][0]);b=iou(after,row['span'][0])
    return dict(key=row['sample_key'],dataset=row['dataset'],video=row['video_path'],duration=row['duration'],
        gt_ratio=(row['span'][0][1]-row['span'][0][0])/row['duration'],source=source,method=method,budget=budget,
        original_iou=a,clipped_iou=b,delta=b-a,selected_index_before=before_index,selected_index_after=after_index,
        changed=raw is not None and clipped(raw,row['duration'])!=raw,
        emptied=raw is not None and clipped(raw,row['duration']) is None,
        original_invalid=raw is None,clipped_invalid=after is None,
        original_out_of_duration=raw is not None and (raw[0]<0 or raw[1]>row['duration']))


def summarize(rows):
    n=len(rows)
    return dict(n=n,original_miou=mean([r['original_iou'] for r in rows]),clipped_miou=mean([r['clipped_iou']for r in rows]),
        delta_miou=mean([r['delta']for r in rows]),
        original_r={str(t):mean([r['original_iou']>=t for r in rows])for t in [.3,.5,.7]},
        clipped_r={str(t):mean([r['clipped_iou']>=t for r in rows])for t in [.3,.5,.7]},
        changed=sum(r['changed']for r in rows),emptied=sum(r['emptied']for r in rows),
        improved=sum(r['delta']>1e-12 for r in rows),decreased=sum(r['delta']< -1e-12 for r in rows),
        selection_changed=sum(r['selected_index_before']!=r['selected_index_after']for r in rows),
        original_invalid=sum(r['original_invalid']for r in rows),clipped_invalid=sum(r['clipped_invalid']for r in rows))


def condition(row):return f"{row['source']}/{row['method']}@{row['budget']}"


def metric_folder(n):
    return ROOT/f'artifacts/runs/rollout/deterministic/pilot/{n}/metrics/boundary_clipping'


def exception_note(n,kind):
    path=metric_folder(n)/'summary.json'
    if not path.exists():return ''
    values=read(path)
    policy='**범위 초과 예측:** 예측 start/end를 `[0, 영상 길이]`로 clipping하고 길이0이면 IoU=0으로 처리하는 예외를 사후 확인했다. GT는 유지한다. '
    if kind=='deterministic':
        a=values['deterministic/greedy@1'];b=values['deterministic/probability_weighted_medoid@32']
        return policy+f"greedy mIoU는 {a['original_miou']:.4f}→{a['clipped_miou']:.4f}, probability-weighted medoid(32개)는 {b['original_miou']:.4f}→{b['clipped_miou']:.4f}다. 기존 표는 clipping 전 기준이다."
    if kind=='reconsideration':
        deltas=[v['delta_miou']for key,v in values.items()if key.startswith('reconsideration/')]
        return policy+f"재검토 {len(deltas)}개 조건의 mIoU 변화는 {min(deltas):+.4f}~{max(deltas):+.4f}다. 모델 추론·후보 선택을 다시 실행하지 않았으며 기존 표는 clipping 전 기준이다."
    cohort=read(COHORT)[:n];keys={x['sample_key']for x in cohort}
    rows=[r for r in read(metric_folder(900)/'oracle_overlap.json')if r['key']in keys]
    before=[r for r in rows if r['policy']=='original'];after=[r for r in rows if r['policy']=='oracle_reselected_clipped']
    raw=sum(r['relation']=='disjoint'for r in before);clip=sum(r['relation']=='disjoint'for r in after);invalid=sum(r['relation']=='invalid'for r in after)
    v=values['deterministic/oracle_after_clipping@32']
    return policy+f"후보 clipping 후 oracle을 다시 고르면 mIoU는 {v['original_miou']:.4f}→{v['clipped_miou']:.4f}, greedy와 비겹침인 oracle은 {raw}→{clip} query다. clipping 후 길이0인 응답 때문에 {invalid} query는 구간 비교가 불가능하며, 기존 표는 clipping 전 기준이다."


def integrate_text(text,n,kind):
    text=re.sub(r'\n?<!-- boundary-clipping-reference -->.*?<!-- /boundary-clipping-reference -->\n?','\n',text,flags=re.S)
    text=re.sub(r'\n## Boundary clipping\n\n\[예측을 영상 길이 안으로 clipping한 사후 성능 비교\]\(\.\./boundary_clipping/REPORT\.md\)\.[^\n]*\n?','\n',text)
    text=re.sub(r'\n?<!-- clipping-exception:begin -->.*?<!-- clipping-exception:end -->\n?','\n',text,flags=re.S)
    note=exception_note(n,kind)
    if not note:return text
    block='\n\n<!-- clipping-exception:begin -->\n\n'+note+'\n\n<!-- clipping-exception:end -->\n\n'
    headings={'deterministic':'## 32개 후보 선택의 오차·형식 진단','reconsideration':'## 지표와 비교 기준','oracle':'## Oracle performance'}
    heading=headings[kind]
    if heading in text:
        position=text.index(heading)+len(heading)
        return text[:position]+block+text[position:].lstrip('\n')
    return text.rstrip()+block


def integrate_existing():
    for n in [300,600,900]:
        scope=ROOT/f'results/rollout/deterministic/pilot/{n}'
        for relative,kind in [('REPORT.md','deterministic'),('reconsideration/REPORT.md','reconsideration'),('oracle_diversity/REPORT.md','oracle')]:
            p=scope/relative
            if p.exists():p.write_text(integrate_text(p.read_text(),n,kind))


def publish(rows,cohort):
    for n in [300,600,900]:
        assert read(ROOT/f'artifacts/common/cohorts/nested_pilot_v1/pilot_{n}/samples.json')==cohort[:n]
        keys={s['sample_key']for s in cohort[:n]};subset=[r for r in rows if r['key']in keys]
        groups={name:summarize([r for r in subset if condition(r)==name])for name in sorted({condition(r)for r in subset})}
        save(metric_folder(n)/'summary.json',groups)
    integrate_existing()


def main():
    cohort=read(COHORT);assert len(cohort)==900
    selections=collections.defaultdict(list)
    for r in csv.DictReader((SOURCE/'metrics/per_query.csv').open()):
        if r['method']=='greedy'and int(r['budget'])!=1:continue
        selections[r['sample_key']].append(r)
    rec_scores={(r['key'],r['name']):r['iou']for r in read(SOURCE/'reconsideration/metrics/per_query.json')}
    rows=[];overlaps=[];hashes={};negative=0
    for sample in cohort:
        key=sample['sample_key'];duration=sample['duration'];gt=sample['span'][0]
        rawpath=SOURCE/'rollouts'/(key+'.json');recpath=SOURCE/'reconsideration/responses'/(key+'.json')
        hashes[str(rawpath)]=sha(rawpath);hashes[str(recpath)]=sha(recpath)
        cs=read(rawpath)['result']['candidates'];spans=[interval(c)for c in cs];clipspans=[clipped(s,duration)for s in spans]
        negative+=sum(c.get('parsed')is not None and c['parsed']['start']<0 for c in cs)
        for r in selections[key]:
            j=int(r['selected_index'])-1;a=spans[j];b=clipspans[j]
            result=record(sample,'deterministic',r['method'],int(r['budget']),a,b,j+1,j+1)
            assert abs(result['original_iou']-float(r['iou']))<1e-12
            rows.append(result)
        bests={}
        for k in BUDGETS:
            valid=[j for j in range(min(k,len(cs)))if spans[j]]
            rawj=max(valid,key=lambda j:(iou(spans[j],gt),-j))if valid else 0
            valid_clip=[j for j in range(min(k,len(cs)))if clipspans[j]]
            newj=max(valid_clip,key=lambda j:(iou(clipspans[j],gt),-j))if valid_clip else 0
            rows.append(record(sample,'deterministic','oracle_after_clipping',k,spans[rawj],clipspans[newj],rawj+1,newj+1))
            assert iou(clipspans[newj],gt)>=iou(clipspans[rawj],gt)-1e-12
            bests[k]=(rawj,newj)
        rawj,newj=bests[32]
        for name,pool,j in [('original',spans,rawj),('fixed_selection_clipped',clipspans,rawj),('oracle_reselected_clipped',clipspans,newj)]:
            branches=[x for x in range(1,len(cs))if x!=j]
            overlaps.append(dict(key=key,policy=name,relation=relation(pool[0],pool[j]),other_branches=len(branches),overlap=sum(intersection(pool[x],pool[j])>0 for x in branches)))
        for name,v in read(recpath)['variants'].items():
            a=interval(v['answer']);b=clipped(a,duration)
            result=record(sample,'reconsideration',name,v['pool'],a,b)
            assert abs(result['original_iou']-rec_scores[key,name])<1e-12
            rows.append(result)
    valid_gt={r['sample_key']for r in cohort if 0<=r['span'][0][0]<r['span'][0][1]<=r['duration']}
    assert all(r['clipped_iou']>=r['original_iou']-1e-12 for r in rows if r['key']in valid_gt)
    out=SOURCE/'metrics/boundary_clipping'
    save(out/'per_query.json',rows);save(out/'oracle_overlap.json',overlaps)
    publish(rows,cohort)
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    save(SOURCE/'validation/boundary_clipping.json',dict(passed=True,queries=900,original_response_hashes=hashes,source_outputs_unchanged=True,
        existing_scores_reproduced=True,clipping_non_decreasing_when_gt_in_duration=True,GT_outside_duration=900-len(valid_gt),
        negative_parsed_deterministic_candidates=negative,cohort_sha256=sha(COHORT),source_sha256=sha(Path(__file__)),rows=len(rows)))
    from artifact_store import register
    register('deterministic.boundary_clipping.pilot900','analyses',out,'Paired evaluation-only clipping of completed deterministic and reconsideration outputs; fixed900 with300/600 prefixes; GT unchanged.',producer='rollout_boundary_clipping.py',cohort=str(COHORT))
    for name in ['deterministic/greedy@1','deterministic/probability_weighted_medoid@32','deterministic/oracle@32','deterministic/oracle_after_clipping@32']+[c for c in sorted({condition(r)for r in rows})if c.startswith('reconsideration/')]:
        print(json.dumps({'condition':name,**summarize([r for r in rows if condition(r)==name])},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--integrate-only',action='store_true');args=parser.parse_args()
    integrate_existing() if args.integrate_only else main()
