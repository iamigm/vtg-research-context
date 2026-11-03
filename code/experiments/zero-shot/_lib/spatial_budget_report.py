"""Complete thresholds, individual axes and joint cells for spatial-budget pilot."""
from result_layout import write_result_text
from result_layout import relocate_path
import numpy as np
from adaptive_temporal_sampling import ROOT,read,write,sha,objhash,need,stamp
from adaptive_temporal_report import score,csvwrite
from spatial_budget_pilot900 import RUN,PARENT,OUT,REPORT,CONDITIONS,AREAS
DS=[('charades-timelens','Charades'),('activitynet-timelens','ActivityNet'),('qvhighlights-timelens','QVHighlights')]
METRICS=[('miou','mIoU',None),('r03','IoU≥0.3 성공률',.3),('r05','IoU≥0.5 성공률',.5),('r07','IoU≥0.7 성공률',.7)]
TB=[(0,30,'<30'),(30,60,'30–60'),(60,120,'60–120'),(120,180,'120–180'),(180,float('inf'),'≥180')];QB=[(0,.05,'<5%'),(.05,.1,'5–10%'),(.1,.2,'10–20%'),(.2,.4,'20–40%'),(.4,1.000001,'≥40%')]
LABEL={'uniform':'Uniform','gt_whole_r8':'GT 전체8배','random_whole_r8_s42':'Random42','random_whole_r8_s43':'Random43','random_mean':'Random 평균'}
def table(headers,rows):return ['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
def report():
    cohort=read(OUT/'metadata/cohort.json');jobs=read(OUT/'metadata/jobs.json');records=[];costs=[];spatial=[]
    bykey={s['sample_key']:s for s in cohort};parent=relocate_path(ROOT/'artifacts/runs'/PARENT)
    for j in jobs:
        key=j['sample_key'];base=OUT/'responses'/j['level']/j['condition']/(key+'.json');r=read(base);detail=read(OUT/'inputs'/j['level']/j['condition']/(key+'.json'))
        need(r['identity']['input_hash']==objhash(detail['hashes']) and r['input_detail_hash']==objhash(detail),'Input record mismatch');need((r['F'],r['B'],r['grid'])==(j['F'],j['B'],j['grid']),'Spatial budget changed')
        s=bykey[key];p=read(OUT/'plans'/j['condition']/(key+'.json'));need(objhash(p)==j['plan_hash'] and detail['frame_ids']==p['frame_ids'],'Temporal plan changed')
        records.append(dict(sample_key=key,dataset=s['dataset'],video=s['video_path'],level=j['level'],condition=j['condition'],**score(r['answer'],s['span'][0],s['duration'])))
        costs.append(dict(sample_key=key,level=j['level'],condition=j['condition'],generation_seconds=r['generation_seconds'],input_seconds=r['input_prepare_seconds'],input_tokens=detail['total_input_tokens'],output_tokens=r['generated_tokens'],peak_GiB=r['peak_allocated_bytes']/2**30))
        if j['condition']=='uniform':spatial.append(j)
    write(OUT/'metrics/evaluation_records.json',records);csvwrite(OUT/'metrics/compute_cost.csv',costs);csvwrite(OUT/'metrics/spatial_inputs.csv',spatial)
    by={(x['level'],x['sample_key'],x['condition']):x for x in records};plans={s['sample_key']:read(OUT/'plans/uniform'/(s['sample_key']+'.json')) for s in cohort}
    lengths={s['sample_key']:plans[s['sample_key']]['T'] for s in cohort};fractions={s['sample_key']:(s['span'][0][1]-s['span'][0][0])/lengths[s['sample_key']] for s in cohort}
    levels=['low','high'];conditions=CONDITIONS+['random_mean'];metrics=[];cells=[];comparisons=[]
    def values(level,ss,c,threshold):
        members=['random_whole_r8_s42','random_whole_r8_s43'] if c=='random_mean' else [c]
        return np.array([np.mean([by[level,s['sample_key'],cc]['iou'] if threshold is None else float(by[level,s['sample_key'],cc]['iou']>=threshold) for cc in members]) for s in ss])
    for n in [300,600,900]:
        for ds,label in DS:
            ss=[s for s in cohort[:n] if s['dataset']==ds];videos=sorted({s['video_path'] for s in ss});vi={v:i for i,v in enumerate(videos)};membership=np.array([vi[s['video_path']] for s in ss]);counts=np.bincount(membership,minlength=len(videos));rng=np.random.default_rng(42);draws=rng.multinomial(len(videos),np.full(len(videos),1/len(videos)),size=20000);den=draws@counts
            def ci(v):
                totals=np.bincount(membership,weights=v,minlength=len(videos));reps=(draws@totals)/den;return list(np.quantile(reps*100,[.025,.975]))
            for level in levels:
                for m,ml,th in METRICS:
                    v={c:values(level,ss,c,th) for c in conditions}
                    for c in conditions:metrics.append(dict(size=n,dataset=ds,level=level,condition=c,metric=m,N=len(ss),percent=float(v[c].mean()*100),delta_uniform_pp=float((v[c]-v['uniform']).mean()*100)))
                    for a,b in [('gt_whole_r8','uniform'),('gt_whole_r8','random_mean')]:
                        diff=v[a]-v[b];comparisons.append(dict(size=n,dataset=ds,level=level,comparison=a+'-'+b,metric=m,delta_pp=float(diff.mean()*100),ci95=ci(diff)))
                    for axis in ['duration','GT_fraction','joint']:
                        for lo,hi,t in (TB if axis!='GT_fraction' else [(0,float('inf'),'all')]):
                            for qlo,qhi,q in (QB if axis!='duration' else [(0,1.000001,'all')]):
                                mask=np.array([lo<=lengths[s['sample_key']]<hi and qlo<=fractions[s['sample_key']]<qhi for s in ss]);N=int(mask.sum())
                                for c in conditions:
                                    cells.append(dict(size=n,dataset=ds,level=level,axis=axis,time=t,fraction=q,condition=c,metric=m,N=N,percent=float(v[c][mask].mean()*100) if N else None,delta_uniform_pp=float((v[c][mask]-v['uniform'][mask]).mean()*100) if N else None,delta_random_pp=float((v[c][mask]-v['random_mean'][mask]).mean()*100) if N else None))
            for m,ml,th in METRICS:
                for c in conditions:
                    diff=values('high',ss,c,th)-values('low',ss,c,th);comparisons.append(dict(size=n,dataset=ds,level='high-low',comparison=c,metric=m,delta_pp=float(diff.mean()*100),ci95=ci(diff)))
                diff=(values('high',ss,'gt_whole_r8',th)-values('high',ss,'uniform',th))-(values('low',ss,'gt_whole_r8',th)-values('low',ss,'uniform',th));comparisons.append(dict(size=n,dataset=ds,level='high-low',comparison='GT advantage change',metric=m,delta_pp=float(diff.mean()*100),ci95=ci(diff)))
    csvwrite(OUT/'metrics/metrics.csv',metrics);csvwrite(OUT/'metrics/axes.csv',cells);write(OUT/'metrics/comparisons.json',comparisons)
    ix={(x['size'],x['dataset'],x['level'],x['condition'],x['metric']):x for x in metrics};ax={(x['size'],x['dataset'],x['level'],x['axis'],x['time'],x['fraction'],x['condition'],x['metric']):x for x in cells}
    intro=['# 공간 해상도를 맞췄을 때 사건 전체 집중의 효과','',
        '기존900개(데이터셋별300개)의 프레임 선택과 시간표현을 그대로 사용했습니다. 새로 바꾼 것은 원본 RGB를 resize하는 크기입니다. 낮은 목표 면적(low=50,176), 높은 목표 면적(high=100,352)을 비교합니다. 공간 비통제 이전 결과는 사용자 요청에 따라 삭제되어 포함하지 않습니다.','',
        'mIoU는 예측·정답 구간의 교집합 길이/합집합 길이를 질문별로 평균한 값입니다. IoU≥0.3·0.5·0.7 성공률은 해당 기준을 만족한 질문 수/전체 질문 수입니다. 모두 %로 표시하며 높을수록 좋습니다. 파싱 실패·무효 구간도 0점으로 포함합니다. pp는 백분율 차이입니다. Random은 seed42/43 각각의 성공 여부를 계산한 뒤 평균하며 두 응답을 독립 질문으로 세지 않습니다.','',
        '영상 길이 구간과 GT 비율 구간은 왼쪽 포함·오른쪽 제외입니다(마지막 GT 구간은100% 포함). GT 비율은 GT 길이/영상 길이입니다. 교차표는 두 축을 동시에 나누며 각 칸 괄호는 질문 수, †는10개 미만, —는 빈 칸입니다. 개별 축 비교는 다른 축을 고정하지 않습니다.','',
        '95% 구간은 영상 단위 cluster bootstrap 20,000회(seed42)입니다. 같은 영상의 질문을 함께 재표집합니다. 여러 비교를 탐색한 결과이며 다중비교 보정은 없습니다. 교차표에는 신뢰구간을 계산하지 않았습니다. 추가600개는 부족한 교차 칸을 보완한 진단 표본이며 원래 벤치마크 분포의 무작위 평균이 아닙니다.','',
        '## 실제 입력 크기','', '정확히 같은 해상도가 아니라 같은 목표 면적 ±10% 안에서 종횡비를 최대한 보존합니다. 총 영상 토큰 예산은 고정하지 않았습니다. 같은 질문·같은 공간 설정의 네 조건은 프레임 수·grid·시각 토큰 수가 정확히 같습니다.']
    rr=[]
    for ds,label in DS:
        for level in AREAS:
            subset=[j for j in spatial if j['dataset']==ds and j['level']==level];areas=[j['actual_pixels'] for j in subset]
            rr.append([label,level,f'{np.median(areas):.0f}',f'{min(areas)}–{max(areas)}',f"{100*max(abs(j['area_error_fraction']) for j in subset):.2f}%",f"{100*max(abs(j['aspect_ratio_error_fraction']) for j in subset):.2f}%",f"{np.median([j['B'] for j in subset]):.0f}"])
    intro+=table(['데이터셋','설정','픽셀 중앙값','픽셀 범위','면적 최대 오차','종횡비 최대 오차','영상 시각 토큰 중앙값'],rr)
    def build(n):
        lines=[]
        for ds,label in DS:
            lines+=['',f'## {label} — {n//3}개 질문: 실제 점수와 변화']
            for level in levels:
                lines+=['','### '+level]
                lines+=table(['조건']+[ml+' % (Uniform 대비 pp)' for _,ml,_ in METRICS],[[LABEL[c]]+[f"{ix[n,ds,level,c,m]['percent']:.2f} ({ix[n,ds,level,c,m]['delta_uniform_pp']:+.2f})" for m,_,_ in METRICS] for c in conditions])
                subset=[x for x in comparisons if x['size']==n and x['dataset']==ds and x['level']==level]
                lines+=table(['비교','지표','변화 pp [95% 구간]'],[[x['comparison'],dict((m,ml) for m,ml,_ in METRICS)[x['metric']],f"{x['delta_pp']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"] for x in subset])
                for axis,title,bins in [('duration','영상 길이별',TB),('GT_fraction','GT 비율별',QB)]:
                    lines+=['','### '+level+' / '+title,'','각 칸은 GT−Uniform pp / GT−Random 평균 pp입니다.'];rs=[]
                    for _,_,b in bins:
                        t,q=(b,'all') if axis=='duration' else ('all',b);xs=[ax[n,ds,level,axis,t,q,'gt_whole_r8',m] for m,_,_ in METRICS]
                        rs.append([b,xs[0]['N']]+['—' if not x['N'] else f"{x['delta_uniform_pp']:+.2f} / {x['delta_random_pp']:+.2f}" for x in xs])
                    lines+=table(['구간','N']+[ml for _,ml,_ in METRICS],rs)
                for comparison,field in [('GT−Uniform','delta_uniform_pp'),('GT−Random 평균','delta_random_pp')]:
                    for m,ml,_ in METRICS:
                        lines+=['',f'### {level} / 두 축 교차표 / {comparison} / {ml}','', '변화 pp (질문 수)'];rs=[]
                        for _,_,q in QB:
                            xs=[ax[n,ds,level,'joint',t,q,'gt_whole_r8',m] for _,_,t in TB]
                            rs.append([q]+['— (0)' if not x['N'] else f"{x[field]:+.2f} ({x['N']})"+(' †' if x['N']<10 else '') for x in xs])
                        lines+=table(['GT 비율']+[t+'초' for _,_,t in TB],rs)
            lines+=['','### 공간 예산을 높였을 때: high−low','', 'GT advantage change는 (high의 GT−Uniform)−(low의 GT−Uniform)입니다. 양수면 공간 예산이 높을 때 시간 집중의 이득이 더 컸다는 뜻입니다.']
            lines+=table(['조건/비교','지표','차이 pp [95% 구간]'],[[x['comparison'],x['metric'],f"{x['delta_pp']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"] for x in comparisons if x['size']==n and x['dataset']==ds and x['level']=='high-low'])
        return lines
    lines=intro+build(900)+['','## 비용과 검증','',f"새 논리 응답 {len(jobs)}개. GPU 생성 시간 합계 {sum(x['generation_seconds'] for x in costs):.1f}초(배치 공유 비용은 응답별 배분). Batch bridge: {read(OUT/'validation/batch_bridge.json')}",'',
        '기존 프레임의 제거·추가 없이 원본 픽셀에서 새 공간 크기를 만들었습니다. 낮은 원본 화질, padding, 종횡비 오차까지 동일해진 것은 아닙니다. 공간 크기를 바꿨을 때의 실험적 효과이며, 데이터셋 차이의 유일한 원인을 증명하지 않습니다.','',
        '[기존300개 상세표](PILOT_300.md) · [중첩600개 상세표](PILOT_600.md)',f'[모든 점수](../../artifacts/runs/{RUN}/metrics/metrics.csv) · [모든 축 원자료](../../artifacts/runs/{RUN}/metrics/axes.csv) · [실제 입력 크기](../../artifacts/runs/{RUN}/metrics/spatial_inputs.csv)']
    write_result_text(REPORT/'REPORT.md', '\n'.join(lines)+'\n')
    for n in [300,600]:write_result_text(REPORT/f'PILOT_{n}.md', '\n'.join([f'# 중첩 pilot {n}개','', '[지표 정의와900개 보고서](REPORT.md)']+build(n))+'\n')
    for p in REPORT.glob('*.md'):
        width=None
        for line in p.read_text().splitlines():
            if line.startswith('|'):
                w=len(line.split('|'))-2
                if width is None:width=w
                need(w==width,'Malformed Markdown table '+str(p))
            else:width=None
    write(OUT/'validation/final.json',dict(passed=True,logical_responses=len(jobs),queries=900,budget_equality_within_spatial_level=True,temporal_plans_unchanged=True,thresholds=[.3,.5,.7],axes=['duration','GT_fraction','joint'],cohorts=[300,600,900],timestamp=stamp()))
if __name__=='__main__':report()
