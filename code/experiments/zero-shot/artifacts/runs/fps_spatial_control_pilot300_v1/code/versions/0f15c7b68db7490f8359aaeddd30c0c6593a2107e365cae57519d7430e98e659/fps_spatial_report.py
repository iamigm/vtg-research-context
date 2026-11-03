"""Independent annotation-only evaluator and spatial-control integrity checks."""
import csv, math, statistics
from collections import defaultdict
from pathlib import Path
from fps_spatial_control import ROOT, OUT, REPORT, COHORT, CONFIG, FPS, ORDERS, MODELS, read, write, sha, objhash, need
KEYS=['miou','r03','r05','r07']
LABELS=['mIoU','R@.3','R@.5','R@.7']
LENGTH=[0,30,60,120,180,math.inf]
RATIO=[0,.05,.10,.20,.40,math.inf]

def metric(rows):
    return dict(n=len(rows),miou=statistics.mean(r['iou'] for r in rows) if rows else None,
                **{f'r0{t}':statistics.mean(r['iou']>=t/10 for r in rows) if rows else None for t in [3,5,7]})

def savecsv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def interval(x,edges):
    return next(i for i in range(len(edges)-1) if edges[i]<=x<edges[i+1])

def binname(i,edges):
    a,b=edges[i:i+2];return f'[{a:g},{b:g})'

def build():
    from timelens.utils import extract_time,iou
    samples=read(COHORT);bykey={s['sample_key']:s for s in samples};rows=[];control={};pixel_hashes={};input_hashes={};checks=0
    expected=len(samples)*len(FPS)*len(MODELS)*len(ORDERS)
    allowed_sources={objhash(read(p)) for p in [OUT/'metadata/source_manifest.json',OUT/'metadata/source_manifest_initial.json'] if p.exists()}
    need(objhash(read(OUT/'metadata/config.json'))==objhash(CONFIG),'Frozen config changed')
    for model in MODELS:
        for fps in FPS:
            for order in ORDERS:
                for s in samples:
                    relative=Path(model)/str(fps)/order/(s['sample_key']+'.json');p=OUT/'responses'/relative;ip=OUT/'inputs'/relative
                    r=read(p);d=read(ip);key=s['sample_key'];need(r['config_hash']==objhash(CONFIG),'Response config differs')
                    need(r['source_manifest_hash'] in allowed_sources,'Response source manifest is not preserved')
                    need(r['input_detail_sha256']==sha(ip) and r['input_hash']==objhash(d['hashes']),'Response/input hash mismatch')
                    need(r['sample_key']==d['sample_key']==key and r['fps']==d['fps']==fps and r['order']==d['order']==order,'Identity mismatch')
                    need(d['uses_gt'] is False and d['plan']['uses_gt'] is False,'Non-zero-shot input')
                    need(abs(math.prod(d['hw'])/CONFIG['target_pixels']-1)<=.1+1e-10,'Spatial area out of range')
                    signature=dict(hw=d['hw'],source=d['source_sha256'])
                    if key in control:need(control[key]==signature,'Spatial grid differs across FPS/models/orders')
                    control[key]=signature
                    vk=(key,fps);nontext={k:v for k,v in d['hashes'].items() if 'pixel' in k or 'grid' in k}
                    need(bool(nontext),'Missing vision tensor hashes')
                    sig=dict(hashes=nontext,plan=d['plan'])
                    if vk in pixel_hashes:need(pixel_hashes[vk]==sig,'Pixels/PTS differ across models or orders')
                    pixel_hashes[vk]=sig
                    pred=extract_time(r['answer']) or [];score=float(iou(s['span'][0],pred[0])) if pred else 0.
                    need(math.isfinite(score) and 0<=score<=1,'Invalid score')
                    gt=s['span'][0];fraction=(gt[1]-gt[0])/s['duration'];li=interval(s['duration'],LENGTH);ri=interval(fraction,RATIO)
                    rows.append(dict(model=model,fps=fps,order=order,dataset=s['dataset'],sample_key=key,video=s['video_path'],duration=s['duration'],
                                     gt_fraction=fraction,length_bin=li,ratio_bin=ri,iou=score,parse_failed=not pred,eos_reached=r['eos_reached'],generated_tokens=r['generated_tokens'],
                                     h=d['hw'][0],w=d['hw'][1],pixels=math.prod(d['hw']),unique_frames=d['plan']['unique_frames'],padding_frames=d['plan']['padding_frames'],
                                     effective_fps=d['plan']['effective_unique_fps'],visual_tokens=d['visual_tokens'],input_tokens=d['input_tokens']))
                    input_hashes[(model,fps,order,key)]=r['input_hash'];checks+=1
    need(len(rows)==expected and len(list((OUT/'responses').rglob('*.json')))==expected,'Unexpected response count')
    groups=defaultdict(list)
    for r in rows:
        for ds in ['all',r['dataset']]:groups[(r['model'],r['order'],ds,r['fps'])].append(r)
    metrics=[]
    for (model,order,ds,fps),rr in sorted(groups.items()):
        value=metric(rr);base=metric(groups[(model,order,ds,2)])
        metrics.append(dict(model=model,order=order,dataset=ds,fps=fps,**value,**{'delta_'+k:100*(value[k]-base[k]) for k in KEYS}))
    savecsv(OUT/'metrics/per_sample.csv',rows);savecsv(OUT/'metrics/summary.csv',metrics)
    write(OUT/'metrics/summary.json',dict(responses=expected,metrics=metrics))
    REPORT.mkdir(parents=True,exist_ok=True)
    lines=['# 공간 해상도 통제 zero-shot FPS sweep','',f'전체 완료: {expected:,}개 응답. 기존 고정 pilot300 × 2모델 × 2입력 순서 × FPS 2/3/4/6/8.','',
           'mIoU=평균(예측 구간∩GT 구간 길이 / 예측 구간∪GT 구간 길이). R@τ=100×IoU≥τ인 질문 수/N. 네 지표 모두 높을수록 좋다. 표의 점수는 %, 변화는 같은 모델·입력 순서·질문의 2fps 대비 pp다. 최종 답변의 첫 구간을 채점하고 파싱 실패는 0점, 반올림·clipping은 하지 않는다. all은 질문 수 가중 평균이다.','',
           '영상별 목표100,352픽셀 ±10% 안에서 32배수 H/W를 정하고 종횡비 오차를 최소화했다. 같은 영상은 모든 FPS·모델·순서에서 정확히 같은 H/W·resize를 사용했다. 원본 RGB에서 직접 resize했으며 실제 픽셀hash를 검증했다. 전체 영상 토큰 예산과 768프레임 상한을 적용하지 않았다. FPS가 증가하면 프레임 수·시각 토큰·연산량이 증가한다.','',
           '샘플링은 원본 PTS에서 k/FPS에 가장 가까운 프레임(동률은 앞 프레임)을 선택한다. 중복은 제거하고 시간 패치 묶음 때문에 마지막 홀수 프레임만 한 번 복제한다. GT·기존 예측·점수는 계획과 모델 입력에 전달하지 않았다. 학습·few-shot 예시·추가 rollout은 없다. 원본 FPS가 낮으면 실제 unique FPS도 목표보다 낮다.','',
           'Instruct는 greedy/max512, Thinking은 T=1,top_p=.95,top_k=20,seed42/max16384다. 같은 모델 안에서는 모든 FPS에 같은 생성 설정과 seed를 적용했다. 모델 간 차이는 생성 정책도 포함하며 FPS 효과는 모델·순서별로 해석한다. 단일 seed로 seed 변동성을 측정하지 않았다.','',
           '| 모델 | 순서 | 데이터셋 | FPS | N | mIoU | Δpp | R@.3 | Δpp | R@.5 | Δpp | R@.7 | Δpp |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        vals=' | '.join(f"{100*r[k]:.2f} | {r['delta_'+k]:+.2f}" for k in KEYS)
        lines.append(f"| {r['model']} | {r['order']} | {r['dataset']} | {r['fps']} | {r['n']} | {vals} |")
    lines+=['','## 실제 입력과 실패','', '| 모델 | 순서 | FPS | 평균 면적 | 평균 unique frames | 평균 실효 FPS | 평균 visual tokens | 파싱 실패 | EOS 미도달 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for order in ORDERS:
            for fps in FPS:
                rr=groups[(model,order,'all',fps)]
                lines.append(f"| {model} | {order} | {fps} | {statistics.mean(r['pixels'] for r in rr):.1f} | {statistics.mean(r['unique_frames'] for r in rr):.1f} | {statistics.mean(r['effective_fps'] for r in rr):.3f} | {statistics.mean(r['visual_tokens'] for r in rr):.1f} | {sum(r['parse_failed'] for r in rr)} | {sum(not r['eos_reached'] for r in rr)} |")
    lines+=['','[영상 길이·GT 비율·두 축 교차표의 네 지표와 2fps 대비 변화](AXES.md). 고정 pilot300만 평가했으며 600/900 확장 결과는 없다. 이 코호트는 기존 진단 표본이며 벤치마크 전체의 무작위 평균으로 해석하지 않는다.','',
            f'[질문별 CSV](../../artifacts/runs/{CONFIG["run_id"]}/metrics/per_sample.csv) · [요약 CSV](../../artifacts/runs/{CONFIG["run_id"]}/metrics/summary.csv) · [검증](../../artifacts/runs/{CONFIG["run_id"]}/validation/final.json) · [전체 실험 목록](../../../index.html)']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    axes=['# FPS sweep: 영상 길이·GT 비율별 비교','', '[주 보고서](REPORT.md) · [전체 실험 목록](../../../index.html)','',
          'GT 비율=(정답 종료−시작)/영상 길이. GT는 사후 진단에만 사용했다. 빈 칸은 N=0, 10개 미만은 소표본으로 표시한다. 각 점수는 %, 괄호는 같은 칸의 2fps 대비 pp이며 모든 점수는 높을수록 좋다. 프레임이 아닌 질문을 분모로 쓴다.']
    for model in MODELS:
        for order in ORDERS:
            for axis in ['length','ratio','cross']:
                axes+=['',f'## {model} / {order} / {axis}','']
                cells=[(li,ri) for li in range(5) for ri in range(5)] if axis=='cross' else [(i,None) for i in range(5)]
                for metric_key,label in zip(KEYS,LABELS):
                    axes+=['',f'### {label}','','| 구간 | N | 2fps | 3fps (Δpp) | 4fps (Δpp) | 6fps (Δpp) | 8fps (Δpp) |','|---|---:|---:|---:|---:|---:|---:|']
                    for a,b in cells:
                        def select(f):
                            return [r for r in groups[(model,order,'all',f)] if ((r['length_bin']==a and r['ratio_bin']==b) if axis=='cross' else r[axis+'_bin']==a)]
                        base=metric(select(2));n=base['n'];title=(binname(a,LENGTH)+' × '+binname(b,RATIO)) if axis=='cross' else binname(a,LENGTH if axis=='length' else RATIO)
                        title+= ' (빈 칸)' if n==0 else (' (소표본)' if n<10 else '')
                        vals=[]
                        for fps in FPS:
                            m=metric(select(fps));need(m['n']==n,'Unpaired axis bin')
                            vals.append('—' if not n else f"{100*m[metric_key]:.2f}"+(f" ({100*(m[metric_key]-base[metric_key]):+.2f})" if fps!=2 else ''))
                        axes.append(f'| {title} | {n} | '+' | '.join(vals)+' |')
    (REPORT/'AXES.md').write_text('\n'.join(axes)+'\n')
    write(OUT/'validation/final.json',dict(passed=True,responses=checks,paired_samples=300,fixed_spatial_grid=True,matched_pixels_and_timestamps_across_models_orders=True,
                                        gt_blind_input_schema=True,expected_count=expected,stored_input_hashes_revalidated=True,metrics_independently_scored=True))

if __name__=='__main__':build()
