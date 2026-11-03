"""Query-level exploratory association of saved full-head entropy with grounding IoU."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import json,re,csv,time
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/pilot/300');OUT=relocate_path(BASE/'entropy_iou')
DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];MODELS=['qwen25','qwen3']
rng=np.random.default_rng(42)
def read(p):return json.loads(p.read_text())
def avg(x):return float(np.mean(x)) if len(x) else None
def ci_mean(x):
    if not x:return [None,None]
    a=np.asarray(x);b=rng.choice(a,(2000,len(a)),replace=True).mean(axis=1)
    return np.quantile(b,[.025,.975]).tolist()
def corr_ci(x,y):
    x=np.asarray(x);y=np.asarray(y);point=float(spearmanr(x,y).statistic);bs=[]
    for _ in range(1000):
        idx=rng.integers(0,len(x),len(x));r=float(spearmanr(x[idx],y[idx]).statistic)
        if np.isfinite(r):bs.append(r)
    return point,np.quantile(bs,[.025,.975]).tolist()
def main():
    OUT.mkdir(exist_ok=True);start=time.time();records=[];groups=[];bins=[]
    manifest=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'))
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for row,model in enumerate(MODELS):
        for col,ds in enumerate(DATASETS):
            rs=[];masses=[]
            for sample in [s for s in manifest if s['dataset']==ds]:
                r=read(relocate_path(BASE/model/(sample['sample_key']+'.json')));assert r['sample']==sample
                steps=r['steps'];num=[t for t in steps if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',t['token'])]
                assert num,sample['sample_key']
                # Sensitivity: only tokens overlapping the integer portions of the first two timestamps.
                text=''.join(t['token'] for t in steps);matches=list(re.finditer(r'(?<!\w)\d+(?:\.\d+)?',text))[:2]
                integer_positions=set()
                for match in matches:
                    integer_positions.update(range(match.start(),match.start()+len(match.group().split('.')[0])))
                pos=0;ints=[]
                for t in steps:
                    if integer_positions.intersection(range(pos,pos+len(t['token']))):ints.append(t)
                    pos+=len(t['token'])
                item=dict(model=model,dataset=ds,sample_key=sample['sample_key'],iou=r['iou'],numeric_entropy=avg([t['raw']['entropy_nats'] for t in num]),integer_entropy=avg([t['raw']['entropy_nats'] for t in ints]),first_numeric_entropy=num[0]['raw']['entropy_nats'],all_entropy=avg([t['raw']['entropy_nats'] for t in steps]),numeric_steps=len(num),all_steps=len(steps),processed_numeric_entropy=avg([t['processed']['entropy_nats'] for t in num]))
                masses.extend(t['raw']['top30_mass'] for t in steps);rs.append(item);records.append(item)
            x=[r['iou'] for r in rs];y=[r['numeric_entropy'] for r in rs];rho,ci=corr_ci(x,y)
            good=[r['numeric_entropy'] for r in rs if r['iou']>=.5];bad=[r['numeric_entropy'] for r in rs if r['iou']<.5]
            delta=avg(good)-avg(bad);delta_boot=rng.choice(good,(2000,len(good))).mean(axis=1)-rng.choice(bad,(2000,len(bad))).mean(axis=1)
            other={key:float(spearmanr([r['iou'] for r in rs if r[key] is not None],[r[key] for r in rs if r[key] is not None]).statistic) for key in ['all_entropy','integer_entropy','first_numeric_entropy','processed_numeric_entropy','numeric_steps']}
            g=dict(model=model,dataset=ds,n=len(rs),spearman_numeric=rho,spearman_numeric_ci95=ci,mean_success_entropy=avg(good),n_success=len(good),mean_failure_entropy=avg(bad),n_failure=len(bad),success_minus_failure=delta,delta_ci95=np.quantile(delta_boot,[.025,.975]).tolist(),sensitivity_spearman=other,min_top30_mass=min(masses),mean_top30_mass=avg(masses));groups.append(g)
            ax=axes[row,col];ax.scatter(x,y,s=20,alpha=.55,color='#2974a6');ax.set_title(f'{model} / {ds.split("-")[0]}\nrho={rho:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]');ax.set_xlim(-.03,1.03);ax.grid(alpha=.2)
            edges=[0,.3,.5,.7,1.000001]
            for lo,hi in zip(edges[:-1],edges[1:]):
                vals=[r['numeric_entropy'] for r in rs if lo<=r['iou']<hi];interval=ci_mean(vals);b=dict(model=model,dataset=ds,lower=lo,upper=min(hi,1),n=len(vals),mean=avg(vals),ci95=interval);bins.append(b)
                if vals:ax.errorbar((lo+min(hi,1))/2,b['mean'],yerr=[[b['mean']-interval[0]],[interval[1]-b['mean']]],fmt='o',color='#bc3c36',capsize=4,markersize=7)
            if row==1:ax.set_xlabel('IoU')
            if col==0:ax.set_ylabel('Mean entropy at numeric-token steps (nats)')
            done=len(groups);elapsed=time.time()-start
            state=dict(state='running',completed=done,total=6,percent=100*done/6,elapsed_seconds=elapsed,eta_seconds=elapsed*(6-done)/done)
            write_result_text((relocate_path(OUT/'status.json')), json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    fig.suptitle('Saved greedy trajectories: full-vocabulary raw entropy vs. IoU\nBlue: queries; red: IoU-bin mean with query-bootstrap 95% CI')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(relocate_path(OUT/'entropy_vs_iou.png'),dpi=160);fig.savefig(relocate_path(OUT/'entropy_vs_iou.svg'));plt.close(fig)
    with (relocate_path(OUT/'per_query.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write_result_text((relocate_path(OUT/'summary.json')), json.dumps(dict(groups=groups,bins=bins,bootstrap_seed=42,correlation_bootstraps=1000,mean_bootstraps=2000),indent=2)+'\n')
    lines=['# Zero-shot entropy와 IoU의 관계','',
        '기존 top-30 실험의 600개 full-head greedy 응답을 재집계했다. GPU 추론을 다시 실행하지 않았다. 각 step의 `raw.entropy_nats`는 top-30 내부 재정규화가 아니라 **전체 어휘 softmax**로 이미 계산·저장된 엔트로피다. H = −Σ p(v) ln p(v), 단위는 nat.','',
        '분석 단위는 query다. 주 분석값은 **실제로 숫자 토큰을 생성한 단계들의 엔트로피 평균**이다. 숫자 토큰은 공백을 제외한 문자열이 숫자/소수와 일치하는 경우이며 소수부도 포함한다. 정형 문구 및 EOS를 포함한 전체 단계 평균은 별도 민감도 분석으로 둔다. 각 query에 같은 가중치를 부여해 긴 응답이 표본 수를 늘리지 않게 했다. 모델·벤치마크를 섞지 않고 각 100개씩 비교한다.','',
        '![IoU와 숫자 단계 엔트로피](entropy_vs_iou.png)','',
        '| 모델 | 벤치마크 | Spearman ρ [95% bootstrap CI] | IoU < .5: n / 평균 H | IoU ≥ .5: n / 평균 H | 성공−실패 ΔH [95% CI] |','|---|---|---|---|---|---|']
    for g in groups:
        ci=g['spearman_numeric_ci95'];di=g['delta_ci95']
        lines.append(f"| {g['model']} | {g['dataset']} | {g['spearman_numeric']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {g['n_failure']} / {g['mean_failure_entropy']:.3f} | {g['n_success']} / {g['mean_success_entropy']:.3f} | {g['success_minus_failure']:+.3f} [{di[0]:+.3f}, {di[1]:+.3f}] |")
    lines+=['','ρ < 0이면 IoU가 높을수록 엔트로피가 낮은 경향이다. ΔH < 0이면 IoU≥0.5 집단에서 평균 엔트로피가 더 낮다. CI가 0을 포함하면 이 pilot에서 방향이 분명하다고 말하기 어렵다. CI는 query를 복원 추출한 비모수 percentile bootstrap이며 다중 비교 보정은 하지 않은 탐색적 분석이다.','']
    negative=[g for g in groups if g['spearman_numeric_ci95'][1]<0];positive=[g for g in groups if g['spearman_numeric_ci95'][0]>0]
    lines.append(f"6개 그룹 중 음의 상관 CI가 0을 제외한 그룹 {len(negative)}개, 양의 상관 CI가 0을 제외한 그룹 {len(positive)}개다. 나머지는 방향이 불확실하다. 그룹별 결과를 확인해야 하며 낮은 entropy를 높은 IoU의 보장으로 해석할 수 없다.")
    lines+=['','| 모델 | 벤치마크 | 전체 step ρ | 정수부 step ρ | 첫 숫자 step ρ | 보정 후 숫자 step ρ | 숫자 step 수와 IoU의 ρ |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        q=g['sensitivity_spearman'];lines.append(f"| {g['model']} | {g['dataset']} | {q['all_entropy']:+.3f} | {q['integer_entropy']:+.3f} | {q['first_numeric_entropy']:+.3f} | {q['processed_numeric_entropy']:+.3f} | {q['numeric_steps']:+.3f} |")
    lines+=['','정수부 step은 답변에서 처음 두 숫자의 정수부 문자 위치와 겹치는 토큰으로 정했다. 첫 숫자 step은 두 시각 전체의 불확실성을 대표하지 않지만, 생성 길이 및 후속 숫자 선택의 영향을 줄여 확인하는 보조값이다. 소수부 자리와 토큰 길이가 평균 H에 영향을 줄 수 있어 주 분석과 같이 본다.','',
        '확률분포는 각 greedy 응답의 실제 이전 토큰열에 조건화되어 있다. 이 값은 영상 자체의 불확실성만을 측정하거나, 모든 가능한 rollout의 분포를 적분한 값이 아니다. 특히 확신 있게 틀린 답은 낮은 H·낮은 IoU로 나타날 수 있다. 정확도 calibration이나 인과관계를 주장하지 않는다.','',
        'Qwen2.5 기본 repetition penalty의 영향은 raw와 processed를 구분했다. 주 분석에는 processor 이전 raw 분포를 사용한다. GPU 수치 오차로 top-30 질량 합이 1을 미세하게 넘는 기록이 있을 수 있으나, 엔트로피를 top-30에서 재계산하지 않았다. 기존 Decord 디코딩 품질 한계도 그대로 유지된다.','',
        '만약 top-30 확률만 있었다면 −Σ(top30) p ln p는 전체 H의 일부일 뿐이며, top-30 안에서 재정규화한 H는 다른 양이다. 이 데이터에는 전체 H가 저장되어 있어 그런 근사 없이 비교할 수 있다.','',
        '[query별 CSV](per_query.csv) · [집계 및 IoU 구간별 결과 JSON](summary.json) · [벡터 그래프 SVG](entropy_vs_iou.svg)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
    write_result_text((relocate_path(OUT/'status.json')), json.dumps(dict(state='complete',completed=6,total=6,percent=100,queries=600,elapsed_seconds=time.time()-start),indent=2)+'\n')
if __name__=='__main__':main()
