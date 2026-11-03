"""CPU-only per-video decoder audit; does not modify inference inputs or running jobs."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,traceback,concurrent.futures
from datetime import datetime,timezone
from timelens_paths import PILOT_SAMPLES,QWEN3_ZERO
ROOT=relocate_path(Path(__file__)).resolve().parents[1];OUT=relocate_path(ROOT/'results/timelens_top30/mmco_diagnosis');BASE=relocate_path(ROOT/'results/timelens_top30')
FFMPEG='/home1/irteam/envs/dit/bin/ffmpeg'
def read(p):return json.loads(p.read_text())
def write(p,r):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');write_result_text(tmp, json.dumps(r,indent=2)+'\n');tmp.replace(p)
def worker(key,mode):
    import decord
    row=read(relocate_path(OUT/'inputs.json'))[int(key)];path=row['video_path'];indices=row['indices']
    print('PHASE open',file=sys.stderr,flush=True)
    vr=decord.VideoReader(path,num_threads=1)
    print('PHASE '+mode,file=sys.stderr,flush=True)
    hashes={}
    if mode=='sparse':
        frames=vr.get_batch(indices).asnumpy()
        for index,frame in zip(indices,frames):hashes[str(index)]=hashlib.sha256(frame.tobytes()).hexdigest()
    else:
        wanted=set(indices)
        for index in range(len(vr)):
            frame=vr.next()
            if index in wanted:hashes[str(index)]=hashlib.sha256(frame.asnumpy().tobytes()).hexdigest()
    print('PHASE done',file=sys.stderr,flush=True)
    write(relocate_path(OUT/'videos'/str(key)/(mode+'.json')),dict(frames=len(vr),fps=vr.get_avg_fps(),sampled_frames=len(indices),hashes=hashes,decord_version=decord.__version__))
def check(key):
    row=read(relocate_path(OUT/'inputs.json'))[key];folder=relocate_path(OUT/'videos'/str(key));folder.mkdir(parents=True,exist_ok=True)
    result=dict(index=key,video_path=row['video_path'],sample_keys=row['sample_keys'])
    def run(mode):
        with (relocate_path(folder/(mode+'.stderr.log'))).open('w') as f:
            try:r=subprocess.run([sys.executable,__file__,'--worker',str(key),mode],stdout=f,stderr=f,timeout=240);rc=r.returncode
            except subprocess.TimeoutExpired:rc='timeout'
        log=(relocate_path(folder/(mode+'.stderr.log'))).read_text(errors='replace')
        return dict(returncode=rc,mmco_messages=log.count('mmco: unref short failure'),stderr_log=str((relocate_path(folder/(mode+'.stderr.log'))).relative_to(OUT)))
    result['sparse']=run('sparse')
    if result['sparse']['mmco_messages'] or result['sparse']['returncode']!=0:
        result['sequential']=run('sequential')
        with (relocate_path(folder/'ffmpeg.stderr.log')).open('w') as log:
            cmd=[FFMPEG,'-hide_banner','-nostdin','-loglevel','warning','-threads','1','-i',row['video_path'],'-map','0:v:0','-an','-threads','1','-f','null','-']
            try:r=subprocess.run(cmd,stdout=log,stderr=log,timeout=240);rc=r.returncode
            except subprocess.TimeoutExpired:rc='timeout'
        log=(relocate_path(folder/'ffmpeg.stderr.log')).read_text(errors='replace');result['ffmpeg_sequential']=dict(returncode=rc,mmco_messages=log.count('mmco: unref short failure'),command=cmd)
        if result['sparse']['returncode']==result['sequential']['returncode']==0:
            a=read(relocate_path(folder/'sparse.json'))['hashes'];b=read(relocate_path(folder/'sequential.json'))['hashes']
            result['sampled_frame_hash_mismatches']=[int(k) for k,v in a.items() if b.get(k)!=v]
            result['compared_sampled_frames']=len(a)
    write(relocate_path(folder/'result.json'),result);return result

def report(results,done,total):
    affected=[r for r in results if r['sparse']['mmco_messages']];checked=[r for r in affected if 'sequential' in r]
    seek_only=[r for r in checked if r['sequential']['mmco_messages']==0 and r['ffmpeg_sequential']['mmco_messages']==0 and r['sequential']['returncode']==r['ffmpeg_sequential']['returncode']==0]
    mismatch=[r for r in checked if r.get('sampled_frame_hash_mismatches')]
    lines=['# H.264 MMCO 경고 원인 진단','',f'진행: {done}/{total} 영상 ({done/total*100:.1f}%). '+('전체 검사 완료.' if done==total else '검사 중: 아래는 중간 결과.'),'',
        '`mmco: unref short failure`는 H.264가 해제 또는 장기 참조로 전환하려는 단기 참조 프레임을 decoder buffer에서 찾지 못했을 때 발생한다. FFmpeg의 `find_short` 실패 분기에서 출력된다. [FFmpeg 소스](https://ffmpeg.org/doxygen/7.1/h264__refs_8c_source.html).','',
        '현재 실험은 Decord CPU VideoReader(num_threads=1)로 영상을 읽은 뒤 GPU 모델에 전달한다. 이 메시지는 LM head나 CUDA 연산에서 발생하지 않는다. 다중 DataLoader의 사전 읽기 때문에 합쳐진 추론 로그의 직전/직후 sample ID만으로 원인 영상을 특정할 수 없다. 이 검사는 영상별 별도 프로세스와 stderr로 분리한다.','',
        f'- Decord sparse get_batch에서 MMCO 경고가 재현된 영상: {len(affected)}개.',
        f'- 같은 영상의 처음부터 순차 읽기에서 Decord·FFmpeg 모두 MMCO 경고가 사라지고 정상 종료: {len(seek_only)}개.',
        f'- sparse와 순차 Decord의 표본 RGB SHA-256이 다른 영상: {len(mismatch)}개.','',
        '순차 읽기는 정상이고 sparse에서만 발생한다면 해당 파일과 Decord의 프레임 탐색 경로 사이의 상호작용이 원인이라는 근거다. 순차 FFmpeg에서도 발생하면 원본 bitstream/decoder 호환성 문제를 추가 조사해야 하며, 이 경고만으로 원본 손상을 단정하지 않는다. Frame hash 비교는 실제 표본 픽셀 영향 여부를 확인한다.','',
        '원본 영상·실험 decoder·모델 입력은 수정하지 않았다. 검사는 저장된 Qwen3 zero-shot metadata의 실제 frame indices를 사용한다. Qwen2.5의 모든 입력 픽셀이나 모든 가능한 탐색 패턴까지 검증한 것은 아니다. Hash가 같아도 순차 decoder 자체가 원본을 올바르게 복원했는지까지 보증하지 않는다.','',
        '| 영상 | Sparse MMCO | 순차 Decord MMCO | 순차 FFmpeg MMCO | 표본 RGB 불일치 |','|---|---:|---:|---:|---:|']
    for r in checked:lines.append(f"| {relocate_path(Path(r['video_path'])).name} | {r['sparse']['mmco_messages']} | {r['sequential']['mmco_messages']} | {r['ffmpeg_sequential']['mmco_messages']} | {len(r.get('sampled_frame_hash_mismatches',[])) if 'sampled_frame_hash_mismatches' in r else '미검증'} |")
    lines+=['','각 영상의 원본 경로, sample ID, 재현 명령, 종료 코드, 프레임 hash, stderr는 `videos/<index>/`와 `results.json`에 보존한다.','',
        '대응: 기존 zero-shot과의 비교 중에는 decoder나 원본을 교체하지 않는다. 경고 자체는 파일별 stderr에 보존하고 진행 로그와 분리한다. 입력 픽셀 불일치가 확인되면 해당 표본을 명시하고 순차 디코딩을 사용하는 별도 비교 실험이 필요하다.']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n');write(relocate_path(OUT/'results.json'),results)

def main():
    OUT.mkdir(parents=True,exist_ok=True);samples=read(PILOT_SAMPLES);videos={}
    for s in samples:
        p=s['video_path'];meta=read(relocate_path(QWEN3_ZERO/s['dataset']/s['sample_key']/'metadata.json'))
        indices=meta['decoder_metadata']['frames_indices']
        if p not in videos:videos[p]=dict(video_path=p,indices=indices,sample_keys=[])
        videos[p]['sample_keys'].append(s['sample_key'])
    rows=list(videos.values());write(relocate_path(OUT/'inputs.json'),rows);started=time.time();results=[]
    write(relocate_path(OUT/'status.json'),dict(state='running',completed=0,total=len(rows),percent=0))
    report(results,0,len(rows))
    with (relocate_path(OUT/'progress.log')).open('a',buffering=1) as log:
        def progress():
            done=len(results);elapsed=time.time()-started;eta=elapsed*(len(rows)-done)/done if done else None
            state=dict(state='complete' if done==len(rows) else 'running',stage='CPU sparse / sequential decoder comparison',completed=done,total=len(rows),percent=round(done/len(rows)*100,2),elapsed_seconds=round(elapsed),eta_seconds=round(eta) if eta is not None else None,updated_at=datetime.now(timezone.utc).isoformat())
            write(relocate_path(OUT/'status.json'),state);log.write(json.dumps(state)+'\n')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(check,i) for i in range(len(rows))}
            while pending:
                finished,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
                results.extend(f.result() for f in finished);results.sort(key=lambda r:r['index']);progress()
                if finished:report(results,len(results),len(rows))
if __name__=='__main__':
    if '--worker' in sys.argv:worker(sys.argv[2],sys.argv[3])
    else:
        try:main()
        except Exception:write(relocate_path(OUT/'status.json'),dict(state='failed',error=traceback.format_exc()));raise
