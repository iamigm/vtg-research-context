"""Compare full-vocabulary uncertainty at numeric and nonnumeric greedy steps."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from collections import defaultdict,Counter
import json,csv,re,time,traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/timelens_top30');OUT=relocate_path(BASE/'numeric_uncertainty')
MODELS=['qwen25','qwen3'];DATASETS=['charades-timelens','activitynet-timelens','qvhighlights-timelens'];ROLES=['hundreds','tens','ones','tenths'];CATS=['hundreds','tens','ones','tenths','decimal_point','separator','whitespace','text','eos','numeric_other','other']
LABELS={'hundreds':'백의 자리','tens':'십의 자리','ones':'일의 자리','tenths':'소수 첫째 자리','decimal_point':'소수점','separator':'구분자','whitespace':'공백','text':'일반 문구','eos':'EOS','numeric_other':'기타 숫자(MM:SS 등)','other':'기타'}
rng=np.random.default_rng(42)
def read(p):return json.loads(p.read_text())
def write(name,x):write_result_text((relocate_path(OUT/name)), json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvfile(name,rows):
    with (relocate_path(OUT/name)).open('w') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summary(rows):
    by=defaultdict(list)
    for r in rows:by[r['sample_key']].append(r['entropy'])
    a=np.array([np.mean(v) for v in by.values()]);steps=np.array([r['entropy'] for r in rows]);bs=rng.choice(a,(2000,len(a)),replace=True).mean(axis=1)
    return dict(n_queries=len(a),n_steps=len(steps),mean=float(a.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),step_weighted_mean=float(steps.mean()),step_median=float(np.median(steps)))
def main():
    OUT.mkdir(exist_ok=True);started=time.time();samples=read(relocate_path(ROOT/'predictions/cohorts/pilot_300/samples.json'))
    with (relocate_path(BASE/'digit_role_entropy/per_step_roles.csv')).open() as f:role_rows=list(csv.DictReader(f))
    rolemap={}
    for r in role_rows:
        key=(r['model'],r['sample_key'],int(r['ar_step']));assert key not in rolemap;rolemap[key]=r
    records=[];done=0
    for model in MODELS:
        for sample in samples:
            r=read(relocate_path(BASE/model/(sample['sample_key']+'.json')))
            for s in r['steps']:
                token=s['token'];numeric=bool(re.fullmatch(r'\s*\d+\s*',token));role=rolemap.get((model,sample['sample_key'],s['step']));cat=role['role'] if role and role['role'] in ROLES+['decimal_point'] else None
                if numeric and cat is None:cat='numeric_other'
                if cat is None:
                    if token in ['<|im_end|>','<|endoftext|>']:cat='eos'
                    elif token.isspace():cat='whitespace'
                    elif not re.search(r'\w',token):cat='separator'
                    elif re.search(r'[a-zA-Z]',token):cat='text'
                    else:cat='other'
                if cat in ROLES:assert numeric
                records.append(dict(model=model,dataset=sample['dataset'],sample_key=sample['sample_key'],ar_step=s['step'],token=token,numeric=numeric,category=cat,endpoint=role['endpoint'] if role else '',gt_digits=int(role['gt_digits']) if role else None,generated_digits=int(role['generated_digits']) if role else None,entropy=s['raw']['entropy_nats'],processed_entropy=s['processed']['entropy_nats'],raw_selected_probability=s['raw_selected_probability'],iou=r['iou']))
            done+=1
            if done%100==0:
                elapsed=time.time()-started;state=dict(state='running',stage='classifying saved token steps',completed=done,total=600,percent=done/6,elapsed_seconds=elapsed,eta_seconds=elapsed*(600-done)/done);write('status.json',state);print(json.dumps(state),flush=True)
    paired=[];categories=[];endpoints=[]
    for model in MODELS:
        for ds in ['all']+DATASETS:
            rs=[r for r in records if r['model']==model and (ds=='all' or r['dataset']==ds)]
            nums=[r for r in rs if r['numeric']];rest=[r for r in rs if not r['numeric']];n=summary(nums);o=summary(rest);by=defaultdict(lambda:defaultdict(list))
            for r in rs:by[r['sample_key']][r['numeric']].append(r['entropy'])
            delta=np.array([np.mean(v[True])-np.mean(v[False]) for v in by.values()]);boot=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
            paired.append(dict(model=model,dataset=ds,numeric=n,nonnumeric=o,paired_difference=float(delta.mean()),difference_ci95=np.quantile(boot,[.025,.975]).tolist(),numeric_entropy_sum_fraction=sum(r['entropy'] for r in nums)/sum(r['entropy'] for r in rs),numeric_step_fraction=len(nums)/len(rs)))
            for cat in CATS:
                sub=[r for r in rs if r['category']==cat]
                if sub:categories.append(dict(model=model,dataset=ds,category=cat,**summary(sub)))
            for ep in ['start','end']:
                for cat in ROLES+['decimal_point']:
                    sub=[r for r in rs if r['endpoint']==ep and r['category']==cat]
                    if sub:endpoints.append(dict(model=model,dataset=ds,endpoint=ep,category=cat,**summary(sub)))
    buckets=defaultdict(list)
    for r in records:
        if r['category'] in ROLES+['decimal_point'] and r['endpoint']:
            buckets[tuple(r[k] for k in ['model','dataset','endpoint','gt_digits','generated_digits','category'])].append(r)
    stratified=[]
    for key,rs in sorted(buckets.items()):
        model,ds,ep,gd,pd,cat=key;stratified.append(dict(model=model,dataset=ds,endpoint=ep,gt_digits=gd,generated_digits=pd,category=cat,**summary(rs)))
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for mi,m in enumerate(MODELS):
        for di,ds in enumerate(DATASETS):
            ax=axes[mi,di];p=next(p for p in paired if p['model']==m and p['dataset']==ds)
            vals=[p['numeric'],p['nonnumeric']];heights=[s['mean'] for s in vals];ax.bar([0,1],heights,color=['#b95b35','#397b9b'],width=.6)
            ax.errorbar([0,1],heights,yerr=[[s['mean']-s['ci95'][0] for s in vals],[s['ci95'][1]-s['mean'] for s in vals]],fmt='none',color='black',capsize=5)
            for x,s in enumerate(vals):ax.text(x,s['mean']+.035,f"{s['mean']:.4f}\n{ s['n_steps']} steps",ha='center',fontsize=10)
            ax.set_xticks([0,1],['Numeric tokens','All other tokens']);ax.set_title(f'{m} / {ds.split("-")[0]}');ax.grid(axis='y',alpha=.2)
            if di==0:ax.set_ylabel('Query-balanced mean full-vocab entropy (nats)')
    fig.suptitle('Numeric vs. nonnumeric steps: same saved greedy responses\n95% query-bootstrap CI; paired numeric-minus-other differences in report');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(relocate_path(OUT/'numeric_vs_other.png'),dpi=160);plt.close(fig)
    columns=['hundreds','tens','ones','tenths','decimal_point','separator','whitespace','text','eos'];matrix=np.full((6,len(columns)),np.nan);annotations={}
    for mi,m in enumerate(MODELS):
        for di,ds in enumerate(DATASETS):
            row=mi*3+di
            for col,c in enumerate(columns):
                s=next((s for s in categories if s['model']==m and s['dataset']==ds and s['category']==c),None)
                if s:matrix[row,col]=s['mean'];annotations[(row,col)]=f"{s['mean']:.3f}\nN={s['n_queries']}"
    fig,ax=plt.subplots(figsize=(15,7));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=max(2.4,float(np.nanmax(matrix))));ax.set_xticks(range(len(columns)),['100s','10s','1s','0.1s','decimal .','separator','space','phrase','EOS'],rotation=20);ax.set_yticks(range(6),[f'{m} / {ds.split("-")[0]}' for m in MODELS for ds in DATASETS])
    for i in range(6):
        for j in range(len(columns)):ax.text(j,i,annotations.get((i,j),'—'),ha='center',va='center',fontsize=10,color='black' if np.isnan(matrix[i,j]) or matrix[i,j]<1.7 else 'white')
    fig.colorbar(im,ax=ax,label='Query-balanced mean entropy (nats)');ax.set_title('Where uncertainty concentrates: numeric place value vs. other token roles\nN = contributing queries (not token count); start/end pooled here; stratified data provided separately');fig.tight_layout();fig.savefig(relocate_path(OUT/'uncertainty_by_role.png'),dpi=160);fig.savefig(relocate_path(OUT/'uncertainty_by_role.svg'));plt.close(fig)
    csvfile('per_step.csv',records);csvfile('category_summary.csv',categories);csvfile('endpoint_role_summary.csv',endpoints);csvfile('digit_stratified_summary.csv',stratified)
    write('summary.json',dict(paired=paired,categories=categories,endpoint_roles=endpoints,digit_stratified=stratified,total_responses=600,total_steps=len(records),numeric_definition='token stripped of whitespace matches digits only',role_mapped_steps=len(rolemap),unmapped_numeric_steps=sum(r['category']=='numeric_other' for r in records),bootstrap_samples=2000,bootstrap_seed=42))
    lines=['# 숫자 생성과 나머지 토큰의 불확실성','',f'기존 full-head greedy 600개 응답, {len(records):,}개 step의 raw 전체 어휘 엔트로피를 재집계했다. GPU 재추론은 없다. 숫자 토큰은 공백 제거 후 문자열 전체가 숫자로 이루어진 토큰이다. 소수점은 숫자에 포함하지 않고 별도 분류했다.','',
        '주 통계는 query별 해당 유형 step의 평균을 구한 뒤 query 간 평균한 값이다. 같은 query의 숫자/나머지를 paired 비교했다. 95% CI는 query 단위 2,000회 bootstrap이다. CSV에는 step 가중 평균과 step 중앙값도 보존했다.','',
        '![숫자 대 나머지](numeric_vs_other.png)','',
        '| 모델 | 벤치마크 | 숫자 H | 나머지 H | paired ΔH [95% CI] | 숫자의 step 비중 (%) | 숫자의 ΣH 비중 (%) |','|---|---|---:|---:|---|---:|---:|']
    for p in paired:
        ci=p['difference_ci95'];lines.append(f"| {p['model']} | {p['dataset']} | {p['numeric']['mean']:.4f} | {p['nonnumeric']['mean']:.4f} | {p['paired_difference']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {100*p['numeric_step_fraction']:.2f} | {100*p['numeric_entropy_sum_fraction']:.2f} |")
    lines+=['','ΣH 비중은 이 greedy 경로에서 관측한 step 엔트로피 합 중 숫자 step이 차지하는 비율이다. 가능한 모든 응답에 대해 평균한 sequence entropy나 IoU에 대한 설명력으로 해석하지 않는다.','',
        '![역할별 불확실성](uncertainty_by_role.png)','',
        '| 모델 | 토큰 역할 | query N / step N | 평균 H [95% CI] |','|---|---|---|---|']
    for s in categories:
        if s['dataset']=='all':lines.append(f"| {s['model']} | {LABELS[s['category']]} | {s['n_queries']} / {s['n_steps']} | {s['mean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] |")
    lines+=['','역할은 반올림 전 생성된 시각 문자열에서 사후 지정했다. 백·십·일의 자리, 소수 첫째 자리를 나누었고 소수점은 별도다. 역할 분석에 포함되지 않은 MM:SS 등 6개 응답의 숫자도 전체 숫자/나머지 비교에는 포함하되 자릿수 표에서는 기타 숫자로 분리했다. 일반 문구, 공백, 구분자, EOS를 분리해 비교한다.','',
        '## 해석','']
    for m in MODELS:
        p=next(p for p in paired if p['model']==m and p['dataset']=='all');cs=[s for s in categories if s['model']==m and s['dataset']=='all' and s['category'] in ROLES];peak=max(cs,key=lambda s:s['mean'])
        lines.append(f"- {m}: 숫자 평균 H={p['numeric']['mean']:.4f}, 나머지={p['nonnumeric']['mean']:.4f} nat. 숫자 step은 {100*p['numeric_step_fraction']:.1f}%지만 관측 ΣH의 {100*p['numeric_entropy_sum_fraction']:.2f}%를 차지한다. 자릿수별 집계에서 가장 큰 평균은 {LABELS[peak['category']]}의 {peak['mean']:.4f} nat(N={peak['n_queries']} query)다.")
    lines+=['','이 결과는 “시간 숫자가 정형 문구보다 예측하기 어려운가”에 관한 것이다. 엔트로피가 큰 숫자가 반드시 틀리거나, 작은 숫자가 반드시 맞는다는 결론은 아니다. 평균 역할 표는 시작·종료와 자릿수 길이가 섞인 설명용 요약이다. 엄밀한 같은 조건 비교를 위해 모델·벤치마크·시작/종료·GT 자릿수·생성 자릿수·역할로 분리한 CSV를 함께 제공했다. 백의 자리처럼 표본이 적은 경우 N을 함께 보아야 한다.','',
        '[모든 step CSV](per_step.csv) · [역할별 통계 CSV](category_summary.csv) · [시작/종료별 역할 CSV](endpoint_role_summary.csv) · [GT·생성 자릿수 통제 CSV](digit_stratified_summary.csv) · [전체 정량 JSON](summary.json)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n');write('status.json',dict(state='complete',completed=600,total=600,percent=100,total_steps=len(records),elapsed_seconds=time.time()-started))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write('status.json',dict(state='failed',error=traceback.format_exc()));raise
