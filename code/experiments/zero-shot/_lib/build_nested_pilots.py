"""Freeze nested pilot300/600/900 without inspecting predictions or outcomes."""
from pathlib import Path
import hashlib,collections,math
from adaptive_temporal_sampling import ROOT,read,immutable,fileinfo,stamp,need
from artifact_store import register
BASE=ROOT/'artifacts/common/predictions/legacy';OUT=ROOT/'artifacts/common/cohorts/nested_pilot_v1'
TB=[0,30,60,120,180,float('inf')];QB=[0,.05,.1,.2,.4,1.000001]
def cell(s):
    T=s['duration'];L=s['span'][0][1]-s['span'][0][0]
    return (next(i for i in range(5) if TB[i]<=T<TB[i+1]),next(i for i in range(5) if QB[i]<=L/T<QB[i+1]))
def rank(s):return hashlib.sha256(('nested_pilot_v1|42|'+s['sample_key']).encode()).hexdigest()
def main():
    original=read(BASE/'cohorts/pilot_300/samples.json');pool=read(BASE/'qwen3/samples.json');by={s['sample_key']:s for s in pool}
    need(len(original)==300 and len({s['sample_key'] for s in original})==300,'Original cohort invalid')
    need(all(by[s['sample_key']]==s for s in original),'Original cohort differs from source pool')
    selected=original[:];seen={s['sample_key'] for s in selected};usedvideos={s['video_path'] for s in selected};excluded=[];eligible=[]
    for s in pool:
        if s['sample_key'] in seen:continue
        T=s['duration'];span=s['span'];meta=BASE/'qwen3'/s['dataset']/s['sample_key']/'metadata.json'
        reason=None
        if len(span)!=1 or not all(math.isfinite(float(v)) for v in [T,*span[0]]) or not 0<=span[0][0]<span[0][1]<=T:reason='invalid or out-of-annotation-duration single GT'
        elif not Path(s['video_path']).is_file() or not meta.is_file():reason='missing video or legacy input metadata'
        if reason:excluded.append(dict(sample_key=s['sample_key'],reason=reason));continue
        eligible.append(s)
    stages={300:original[:]};selection=[]
    datasets=sorted({s['dataset'] for s in original})
    for size in [600,900]:
        for ds in datasets:
            current=[s for s in selected if s['dataset']==ds];counts=collections.Counter(cell(s) for s in current)
            while len(current)<size//3:
                remaining=[s for s in eligible if s['dataset']==ds and s['sample_key'] not in seen]
                groups={}
                for s in remaining:groups.setdefault(cell(s),[]).append(s)
                need(bool(groups),'Insufficient candidates')
                group=min(groups,key=lambda c:(counts[c],c));candidates=groups[group];novel=[s for s in candidates if s['video_path'] not in usedvideos]
                s=min(novel or candidates,key=rank);metadata=read(BASE/'qwen3'/ds/s['sample_key']/'metadata.json')
                need(metadata['sample']==s,'Source metadata mismatch')
                F=len(metadata['decoder_metadata']['frames_indices']);need(F>=2 and F%2==0,'Unsupported frame budget')
                selection.append(dict(sample_key=s['sample_key'],dataset=ds,added_at_size=size,duration_bin=group[0],GT_fraction_bin=group[1],new_video=s['video_path'] not in usedvideos,selection_hash=rank(s)))
                selected.append(s);current.append(s);seen.add(s['sample_key']);usedvideos.add(s['video_path']);counts[group]+=1
        stages[size]=selected[:]
    for size,samples in stages.items():
        p=OUT/f'pilot_{size}';immutable(p/'samples.json',samples)
        hist={ds:{str(c):n for c,n in sorted(collections.Counter(cell(s) for s in samples if s['dataset']==ds).items())} for ds in datasets}
        immutable(p/'selection.json',dict(size=size,per_dataset=size//3,original_300_unchanged=True,selection_order=[x for x in selection if x['added_at_size']<=size],counts=hist,unique_videos={ds:len({s['video_path'] for s in samples if s['dataset']==ds}) for ds in datasets}))
        register(f'nested_pilot_v1.{size}.cohort','manifests',p,'Nested pilot; original300 preserved; duration/GT-fraction cell-balanced extension','_lib/build_nested_pilots.py',f'pilot_{size}','complete',['pilot_300.cohort'])
    immutable(OUT/'manifest.json',dict(created_at=stamp(),sources=[fileinfo(BASE/'cohorts/pilot_300/samples.json'),fileinfo(BASE/'qwen3/samples.json')],files=[fileinfo(OUT/f'pilot_{n}/samples.json') for n in stages],policy='Retain original sample objects/order; add100 per dataset per stage; select least-populated available joint cell then prefer unused video then SHA256(nested_pilot_v1|42|sample_key); no response/performance used',duration_edges=[0,30,60,120,180,None],GT_fraction_edges=QB,excluded=excluded,scope='diagnostic stratified expansion, not representative random benchmark average or independent test'))
    print({n:dict(queries=len(s),per_dataset=collections.Counter(x['dataset'] for x in s),videos=len({x['video_path'] for x in s})) for n,s in stages.items()})
if __name__=='__main__':main()
