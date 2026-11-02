"""Post-run migration only. Retain full timestamp/hash metadata, compact RGB to planned IDs."""
import os,fcntl,hashlib,time,json
from pathlib import Path
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,sha,fileinfo,need,stamp
from shared_selected_frames import COMMON,load_selected
RUN='selected_frame_cache_migration_v1';OUT=ROOT/'artifacts/runs'/RUN
TARGETS=['boundary_oracle_budget_pilot_v1','temporal_sampling_pilot300_v2','gt_interval_density_pilot300_v1','gt_extent_boundary_density_pilot900_v1']
WAIT_FOR=['gt_extent_boundary_density_pilot900_v1','gt_actual_density_pilot900_v1','gt_density_allocation_comparison_pilot900_v1']

def state(stage,**kw):
    row=dict(state='building',stage=stage,timestamp=stamp(),**kw);write(OUT/'metadata/status.json',row);print(row,flush=True)
def install_readers():
    # Called after all inference and reporting have finished. Preserve old sources.
    for name in ['boundary_oracle_inputs.py','pilot900_inputs.py']:
        p=ROOT/'_lib'/name;s=p.read_text()
        if 'from shared_selected_frames import load_selected' in s:continue
        write(OUT/'metadata'/('source_before_'+name+'.json'),fileinfo(p));(OUT/'code'/name).write_bytes(p.read_bytes())
        needle="selected=np.array(np.load(cache['path'],mmap_mode='r')[[loc[i] for i in ids]])"
        need(needle in s,'Reader patch target changed: '+name)
        s=s.replace('import os,time,copy,json,threading,traceback','import os,time,copy,json,threading,traceback\nfrom shared_selected_frames import load_selected,available_metadata')
        s=s.replace(needle,'selected=load_selected(cache,ids)')
        needle="    record=group[0];sample=record['sample'];video=sample['video_path'];started=time.perf_counter()"
        need(needle in s,'Preparation patch target changed')
        s=s.replace(needle,needle+"\n    existing=available_metadata(video,[record['grid'][1]*16,record['grid'][2]*16])\n    if existing is not None:return [(r['sample']['sample_key'],existing) for r in group]")
        compile(s,str(p),'exec');tmp=p.with_suffix('.migration.tmp');tmp.write_text(s);os.replace(tmp,p)

def compact():
    for r in WAIT_FOR:need(read(ROOT/'artifacts/runs'/r/'metadata/status.json')['state']=='complete','Run not complete: '+r)
    for cat in ['metadata','logs','code']:(OUT/cat).mkdir(parents=True,exist_ok=True)
    COMMON.mkdir(parents=True,exist_ok=True);(COMMON/'frames').mkdir(exist_ok=True)
    with (OUT/'metadata/migration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metas={};groups={};required={};references=[]
        for target in TARGETS:
            for p in (ROOT/'artifacts/runs'/target/'frames').glob('*.json'):
                m=read(p)
                if not all(k in m for k in ['cache_key','video_path','grid_hw','ids','resized_rgb_hashes']):continue
                metas[str(p.resolve())]=m;groups.setdefault(m['cache_key'],[]).append((p,m));required.setdefault(m['cache_key'],set())
        # Every saved plan in every run that points at these caches must remain replayable.
        for indexpath in (ROOT/'artifacts/runs').glob('*/metadata/frame_cache_index.json'):
            index=read(indexpath);run=indexpath.parent.parent;keymap={}
            for sample,ref in index.items():
                m=metas.get(str(Path(ref).resolve()))
                if m:keymap[sample]=m['cache_key']
            if not keymap:continue
            references.append(fileinfo(indexpath))
            for category in ['plans','inputs']:
                for p in (run/category).glob('*/*.json'):
                    key=p.stem
                    if key not in keymap:continue
                    obj=read(p);ids=obj.get('frame_ids',[]);required[keymap[key]].update(ids)
            state('collecting_references',run=run.name,cache_groups=len(groups))
        # No deletion for arrays with no known planned use; retain rather than guess.
        video_index=read(COMMON/'video_index.json') if (COMMON/'video_index.json').exists() else {};records=[]
        install_readers()
        for key,entries in groups.items():
            ids=sorted(required[key])
            if not ids:continue
            p,m=entries[0];source=Path(m['path']);mp=COMMON/'frames'/(key+'.json');dest=COMMON/'frames'/(key+'.npy')
            state('compacting',cache_key=key,completed=len(records),selected=len(ids),original=len(m['ids']))
            if not mp.exists():
                need(source.exists(),'Source missing before migration');need(sha(source)==m['sha256'],'Source checksum mismatch')
                arr=np.load(source,mmap_mode='r');loc={i:j for j,i in enumerate(m['ids'])};need(set(ids)<=set(loc),'Plan IDs absent from source')
                tmp=dest.with_suffix('.tmp.npy');result=np.lib.format.open_memmap(tmp,mode='w+',dtype=arr.dtype,shape=(len(ids),*arr.shape[1:]));hashes=[]
                for j,i in enumerate(ids):
                    a=arr[loc[i]];h=hashlib.sha256(a.tobytes()).hexdigest();need(h==m['resized_rgb_hashes'][loc[i]],'Original pixel hash mismatch');result[j]=a;hashes.append(h)
                    if j%256==0:state('copying_selected_frames',cache_key=key,frames_completed=j,frames_total=len(ids))
                result.flush();del result,arr;os.replace(tmp,dest)
                write(mp,dict(cache_key=key,path=str(dest),ids=ids,rgb_hashes=hashes,sha256=sha(dest),metadata_sources=[str(p) for p,_ in entries],source_video_hash=m['source_video_hash'],created_at=stamp()))
            saved=read(mp);need(set(ids)<=set(saved['ids']),'Existing compact cache missing plans');need(sha(dest)==saved['sha256'],'Compact cache checksum mismatch')
            # Read every retained frame through the new loader before retiring originals.
            for start in range(0,len(ids),32):load_selected(m,ids[start:start+32])
            lookup=str(Path(m['video_path']).resolve())+'|'+','.join(map(str,m['grid_hw']));video_index[lookup]=[str(p) for p,_ in entries];write(COMMON/'video_index.json',video_index)
            retired=[]
            for meta_path,meta in entries:
                old=Path(meta['path'])
                if old.resolve()==dest.resolve() or not old.exists():continue
                need(sha(old)==meta['sha256'],'Duplicate source changed')
                rec=dict(path=str(old),sha256=meta['sha256'],bytes=old.stat().st_size,metadata=fileinfo(meta_path));retired.append(rec)
            journal=dict(cache_key=key,selected=len(ids),original=len(m['ids']),new=fileinfo(dest),retired=retired,metadata_preserved=True)
            write(OUT/'metadata'/f'{key}.json',journal)
            for rec in retired:Path(rec['path']).unlink()
            records.append(journal);state('cache_complete',completed=len(records),cache_key=key)
        write(OUT/'metadata/summary.json',dict(records=records,input_indexes=references,bytes_retired=sum(x['bytes'] for r in records for x in r['retired']),bytes_shared=sum(r['new']['bytes'] for r in records),unreferenced_groups_retained=sum(not required[k] for k in groups)))
        from artifact_store import register
        register('selected_frames_v1.inputs','inputs',COMMON,'Shared planned source frames; missing frames decoded on demand','_lib/compact_selected_frames.py','nested_pilot_v1','complete',[])
        report=ROOT/'results'/RUN;report.mkdir(parents=True,exist_ok=True)
        summary=read(OUT/'metadata/summary.json');(report/'REPORT.md').write_text('# 사용 프레임 공용 캐시 정리\n\n두900개 실행과 통합보고 완료 후 정리했습니다. 입력·응답·계획·원본영상·전체PTS와 픽셀hash metadata는 보존했습니다. 모든 저장 계획이 요구하는 프레임을 공용캐시에 복사하고 검증한 뒤 기존 비압축 배열을 제거했습니다. 기존 metadata의 path는 이관 기록으로 연결되며 입력코드가 cache_key로 공용파일을 찾습니다. 새 계획이 추가 프레임을 요청하면 원본에서 디코딩하고 원래 픽셀hash와 대조해 확장합니다.\n\n'+f"기존 배열 제거 {summary['bytes_retired']/2**30:.1f}GiB, 공용 배열 {summary['bytes_shared']/2**30:.1f}GiB. 참조를 확인하지 못한 캐시는 삭제하지 않았습니다.\n")
        write(OUT/'metadata/status.json',dict(state='complete',stage='complete',timestamp=stamp(),report=str(report/'REPORT.md')))
if __name__=='__main__':compact()
