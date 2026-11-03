"""Three-axis analysis of feasibility-preserving numeric steps; saved logits only."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,re,time,traceback,math,sys,os,hashlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=relocate_path(Path(__file__)).resolve().parents[1];BASE=relocate_path(ROOT/'results/uncertainty/token_analysis/full/9404');SOURCE=relocate_path(BASE/'digit_correctness');OUT=relocate_path(SOURCE/'blue_analysis')
FONT=relocate_path(ROOT/'_lib/entropy_viewer/NotoSansKR.ttf');font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
START=time.time();rng=np.random.default_rng(42)
def dump(name,obj):
    write_result_text((relocate_path(OUT/name)), json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def progress(stage,n,total,counts=None):
    elapsed=time.time()-START;d=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),state='running',stage=stage,completed=n,total=total,percent=100*n/total if total else 100,elapsed_seconds=elapsed,eta_seconds=elapsed*(total-n)/n if n else None,eta_scope=stage,counts=counts or {})
    dump('status.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)
def writecsv(name,rows):
    if rows:
        with (relocate_path(OUT/name)).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def candidates(gt):
    return [str(gt)]+[f'{k//10}.{k%10}' for k in range(max(0,gt-1)*10,(gt+1)*10) if round(float(f'{k//10}.{k%10}'))==gt]
def classify_token(before,token,gt,endpoint):
    # Numeric prefixes use the exact same canonical seconds grammar as source.
    numeric=bool(re.fullmatch(r'[0-9.]+',token))
    if numeric:
        proposed=before+token;kind='number' if re.search('[0-9]',token) else 'format'
        return ('safe' if any(v.startswith(proposed) for v in candidates(gt)) else 'unsafe'),kind
    # Only unambiguous supported number terminators are classified; other
    # textual/special/alternative-format candidates remain unresolved.
    closure=bool(re.fullmatch(r'\s+(?:-\s*)?',token)) if endpoint=='start' else bool(re.fullmatch(r'\s+(?:seconds?\.?)?',token))
    if closure and re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9])?',before):
        return ('safe' if round(float(before))==gt else 'unsafe'),'format'
    return 'unresolved','other'
def dist_info(dist,before,gt,ep):
    items=[];mass=Counter();entropy=Counter()
    for t in dist['top30']:
        state,kind=classify_token(before,t['token'],gt,ep);p=t['probability'];items.append(dict(t,state=state,kind=kind));mass[state]+=p;mass[kind]+=p
        entropy[state]+=-p*math.log(p) if p>0 else 0
    total=sum(t['probability'] for t in items);tail=max(0.,1-total)
    lo=min(1.,mass['safe']);hi=max(lo,min(1.,1-mass['unsafe']))
    observed=mass['safe'];within=(entropy['safe']/observed+math.log(observed)) if observed>0 else None
    band='safe_dominant' if lo>=2/3 else ('unsafe_dominant' if hi<=1/3 else ('competition' if lo>=1/3 and hi<2/3 else 'unresolved_band'))
    return dict(safe_lower=lo,safe_upper=hi,unsafe_mass=min(1.,mass['unsafe']),unresolved_mass=min(1.,mass['unresolved']+tail),tail_mass=tail,top30_mass=total,number_mass=mass['number'],format_mass=mass['format'],other_mass=mass['other'],observed_safe_entropy=within,safe_entropy_contribution=entropy['safe'],unsafe_entropy_contribution=entropy['unsafe'],mass_band=band),items
LABEL={'safe_dominant':'正답 가능 우세 (하한≥2/3)','unsafe_dominant':'오류 후보 우세 (상한≤1/3)','competition':'경쟁 (구간이 1/3–2/3 안)','unresolved_band':'확률 구간이 경계를 걸침','endpoint_correct':'끝까지 유지','next_numeric_failure':'다음 숫자에서 실패','later_numeric_failure':'더 나중 숫자에서 실패','format_failure':'소수점·종료에서 실패'}
LABEL['safe_dominant']='정답 가능 우세 (하한≥2/3)'
TITLES={'candidate_axis':'후보 확률이 어디에 모이는가','future_axis':'이후 언제 정답 가능성을 잃는가','position_axis':'어느 위치의 숫자를 생성하는가'}
FIELDS={'model':'모델','dataset':'벤치마크','endpoint':'예측 대상','digit_stage':'숫자 순서','role':'자릿수','mass_band':'후보 분류','outcome':'이후 결과','high_entropy':'불확실성 수준'}
LABEL.update(start='시작 시각',end='종료 시각',first='첫 숫자',subsequent='후속 숫자',hundreds='백의 자리',tens='십의 자리',ones='일의 자리',tenths='소수 첫째 자리',safe_dominant='정답 우세',unsafe_dominant='오답 우세',competition='경쟁',unresolved_band='미확정')
EXAMPLE='예: 정답 가능 후보의 확률 합이 0.45, 정답 불가능 후보의 확률 합이 0.40이면 A=0.45, B=0.40으로 표시한다. 나머지 0.15는 미확인 확률이며 A나 B에 더하지 않는다. 따라서 A+B는 1보다 작을 수 있다.'

def aggregate(rows,keys):
    bins=defaultdict(list)
    for r in rows:bins[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in sorted(bins.items(),key=lambda t:str(t[0])):
        q=defaultdict(list)
        for r in rs:q[(r['model'],r['sample_key'])].append(r)
        vals=np.array([[np.mean([s['entropy'] for s in group]),np.mean([s['safe_lower'] for s in group]),np.mean([s['safe_upper'] for s in group]),np.mean([s['outcome']!='endpoint_correct' for s in group]),np.mean([s['unsafe_mass'] for s in group])] for group in q.values()])
        boot=rng.choice(vals[:,0],(500,len(vals)),replace=True).mean(axis=1)
        result.append(dict(zip(keys,key),n_queries=len(q),n_steps=len(rs),entropy=float(vals[:,0].mean()),entropy_ci_low=float(np.quantile(boot,.025)),entropy_ci_high=float(np.quantile(boot,.975)),safe_lower=float(vals[:,1].mean()),safe_upper=float(vals[:,2].mean()),unsafe_mass=float(vals[:,4].mean()),future_failure_rate=float(vals[:,3].mean()),small_group=len(q)<10))
    return result

def validate():
    assert classify_token('23.','9',24,'start')[0]=='safe'
    assert classify_token('23.','4',24,'start')[0]=='unsafe'
    assert classify_token('','3',24,'start')[0]=='unsafe'
    assert classify_token('2','3',24,'start')[0]=='safe'
    assert classify_token('23','.',24,'start')[0]=='safe'
    assert classify_token('23',' -',24,'start')[0]=='unsafe'
    assert classify_token('24',' -',24,'start')[0]=='safe'
    assert classify_token('24',':',24,'start')[0]=='unresolved'
    assert classify_token('24.','5',25,'end')[0]=='unsafe'
    d,items=dist_info(dict(top30=[dict(id=1,token='9',probability=.4),dict(id=2,token='4',probability=.3)]),'23.',24,'end')
    assert abs(d['safe_lower']-.4)<1e-8 and abs(d['safe_upper']-.7)<1e-8

def main():
    OUT.mkdir(exist_ok=True);validate();progress('loading',0,600)
    dump('provenance.json',dict(source_csv_sha256=hashlib.sha256((relocate_path(SOURCE/'per_step.csv')).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(relocate_path(Path(__file__)).read_bytes()).hexdigest(),entropy='saved raw full-vocabulary',candidate_probabilities='raw top30 without renormalization',high_entropy_quantile=.75,strata=['model','dataset','endpoint','role'],bootstrap_seed=42,bootstrap_repeats=500))
    source=list(csv.DictReader((relocate_path(SOURCE/'per_step.csv')).open()));groups=defaultdict(list)
    for s in source:groups[(s['model'],s['sample_key'],s['endpoint'])].append(s)
    media=json.loads((relocate_path(BASE/'media_manifest.json')).read_text());rows=[];details=[];counts=Counter();byresponse=defaultdict(list)
    for key,ss in groups.items():byresponse[key[:2]].append((key,ss))
    for i,((model,key),endpoints) in enumerate(byresponse.items()):
        raw=json.loads((relocate_path(BASE/model/(key+'.json'))).read_text());stepmap={s['step']:s for s in raw['steps']}
        for (_,_,ep),ss in endpoints:
            for j,s in enumerate(ss):
                if s['numeric']!='True' or s['prefix_correct']!='True':continue
                assert s['classification_rule']=='rounded_prefix_first_feasibility_loss'
                ar=int(s['ar_step']);step=stepmap[ar];intro=s['introduced_text'];assert step['token']==intro,(key,ar,step['token'],intro)
                before=s['prefix'][:-len(intro)];gt=int(float(s['target']));di,items=dist_info(step['raw'],before,gt,ep);proc,_=dist_info(step['processed'],before,gt,ep)
                assert classify_token(before,intro,gt,ep)[0]=='safe'
                failure=next((x for x in ss[j+1:] if x['first_error']=='True'),None)
                if failure is None:outcome='endpoint_correct';distance=None
                elif failure['numeric']!='True':outcome='format_failure';distance=None
                else:
                    distance=int(failure['digit_ordinal'])-int(s['digit_ordinal']);outcome='next_numeric_failure' if distance==1 else 'later_numeric_failure'
                row=dict(model=model,dataset=s['dataset'],sample_key=key,endpoint=ep,ar_step=ar,role=s['role'],digit_stage='first' if int(s['digit_ordinal'])==1 else 'subsequent',gt_digits=int(s['gt_digits']),generated_digits=int(s['generated_digits']),prefix_before=before,selected=intro,gt=gt,entropy=float(s['entropy']),outcome=outcome,numeric_steps_to_failure=distance,failure_ar_step=int(failure['ar_step']) if failure else None,selected_probability=step['raw_selected_probability'],selected_rank=step['raw_selected_rank'],processed_safe_lower=proc['safe_lower'],processed_safe_upper=proc['safe_upper'],**di)
                assert abs(row['entropy']-step['raw']['entropy_nats'])<1e-7
                rows.append(row);details.append(dict(row,query=raw['sample']['query'],answer=raw['answer'],gt_span=raw['sample']['span'][0],prediction=raw['timestamps'][0],media='../'+media[raw['sample']['video_path']]['media'],candidates=items))
        counts[model+'/'+raw['sample']['dataset']]+=1
        if (i+1)%50==0:progress('candidate analysis',i+1,len(byresponse),dict(counts))
    strata=defaultdict(list)
    for row in rows:strata[tuple(row[k] for k in ['model','dataset','endpoint','role'])].append(row['entropy'])
    for row,detail in zip(rows,details):
        a=strata[tuple(row[k] for k in ['model','dataset','endpoint','role'])];threshold=float(np.quantile(a,.75));more=dict(entropy_q75=threshold,stratum_n=len(a),small_stratum=len(a)<10,high_entropy=row['entropy']>=threshold)
        row.update(more);detail.update(more)
    writecsv('per_blue_step.csv',rows)
    axes={
      'candidate_axis':['model','endpoint','high_entropy','mass_band'],
      'future_axis':['model','endpoint','high_entropy','outcome'],
      'position_axis':['model','dataset','endpoint','digit_stage','role','high_entropy'],
      'three_axis_cross':['model','dataset','endpoint','digit_stage','role','high_entropy','mass_band','outcome'],
      'candidate_future':['model','endpoint','high_entropy','mass_band','outcome'],
      'length_control':['model','dataset','endpoint','gt_digits','generated_digits','role','high_entropy','mass_band']}
    tables={name:aggregate(rows,keys) for name,keys in axes.items()}
    for name,table in tables.items():writecsv(name+'.csv',table)
    dump('summary.json',dict(blue_steps=len(rows),high_entropy_steps=sum(r['high_entropy'] for r in rows),queries=len({(r['model'],r['sample_key']) for r in rows}),outcomes=dict(Counter(r['outcome'] for r in rows)),bands=dict(Counter(r['mass_band'] for r in rows)),tables=tables))
    progress('plots and report',1,3,dict(counts))
    fig,axesplot=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    colors={'endpoint_correct':'#357aa0','next_numeric_failure':'#c65243','later_numeric_failure':'#d39b35','format_failure':'#888888'}
    for ax,(model,ep) in zip(axesplot.flat,[(m,e) for m in ['qwen25','qwen3'] for e in ['start','end']]):
        for outcome,color in colors.items():
            rr=[r for r in rows if r['model']==model and r['endpoint']==ep and r['outcome']==outcome]
            ax.scatter([r['safe_lower'] for r in rr],[r['entropy'] for r in rr],s=13,alpha=.45,color=color,label=LABEL[outcome])
        ax.set_title(model+' · '+LABEL[ep]);ax.set_xlabel('A: 정답 가능 후보 확률 합');ax.set_ylabel('전체 어휘 엔트로피 (nat)');ax.grid(alpha=.15)
    axesplot[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(relocate_path(OUT/'safe_mass_vs_entropy.png'),dpi=150);plt.close(fig)
    lines=['# 정답 가능성을 유지한 숫자 단계: 세 축 분석','',f'전체 벤치마크 9,404개 × 두 모델의 지원 표기 응답 중 파란 숫자 **{len(rows)}개**를 분석했다. 첫 숫자를 포함하며 기존 raw 전체 어휘 엔트로피를 그대로 사용한다. 기존 pilot 추론은 재사용하고 나머지는 같은 조건으로 새로 추론했다.','', '[영상과 후보별 확률 보기](report.html)','',
      '## A와 B의 정의','',
      '후보 토큰을 현재 prefix에 붙였을 때, 기존과 동일한 비음수 정수 또는 소수 한 자리 문법에서 Python round(float(value)) 결과가 GT인 완성이 존재하면 정답 가능 후보다. 정수부 선행 0은 허용하지 않는다. 확실한 숫자·소수점·종료 후보만 판정하고 나머지 형식 또는 텍스트 토큰은 미확인으로 둔다. 이 가능성은 문법상 존재 여부이며 모델이 실제로 그 완성을 생성할 확률은 아니다.','',
      'A는 top-30에서 정답 가능이라고 판정한 후보들의 원확률 합이다. B는 정답 불가능이라고 판정한 후보들의 원확률 합이다. top-30 밖과 판정하지 못한 후보는 둘 다 제외한다. 재정규화하지 않으며 최종 응답 정답률을 의미하지 않는다.','',
      EXAMPLE,'','표의 평균 A와 평균 B는 각 샘플 내부에서 해당 단계들의 값을 평균한 뒤 샘플 간 평균했다. 확률 그룹은 A≥2/3이면 정답 가능 후보 우세, B≥2/3이면 불가능 후보 우세, A≥1/3이고 B>1/3이면 양쪽 후보가 경쟁하는 것으로 분류한다. 나머지는 미확인 확률 때문에 그룹을 확정하지 않는다. 1/3과 2/3은 탐색적으로 정한 기준이다.','',
      '높은 엔트로피는 모델 × 데이터셋 × 시작 또는 종료 × 실제 출력 자릿수 내부 75백분위 이상이다. 동률 때문에 정확히 25%가 아닐 수 있다. 숫자 단계 10개 미만인 비교 집단은 별도 표시하며, 첫 숫자와 후속 숫자와 GT 자릿수와 생성 자릿수도 교차 표로 제공한다.','',
      '후속 결과는 동일한 시작 또는 종료 시각의 끝까지 유지 / 다음 숫자 최초 실패 / 더 나중 숫자 최초 실패 / 소수점·종료 최초 실패로 구분한다. 다른 시각의 실패와 혼합하지 않는다.','',
      '## 세 축의 정량 결과','','표의 확률은 top-30에서 판정한 후보들의 확률 합을 샘플별로 평균한 값이다. 엔트로피 단위는 nat이다. 정답 우세·오답 우세·경쟁·미확정의 기준은 위 정의와 같다.','',
      '아래 표는 불확실성이 높은 단계(비교 집단 내 상위 25%)만 보여 준다. 전체와 나머지 단계는 CSV에 있다. 서로 다른 샘플 수는 해당 행에 기여한 영상·질의 쌍의 수이며 서로 다른 영상 수가 아니다. 평균 계산에 사용한 숫자 토큰 개수는 보조 정보로 상세 CSV에만 남겼다. 같은 샘플이 여러 행에 포함될 수 있으므로 행별 샘플 수를 더하면 전체 샘플 수와 다를 수 있다. 평균은 샘플 안의 해당 단계들을 먼저 평균한 뒤 샘플 간 평균했다. 샘플 수 10개 미만은 탐색적으로 해석한다. 평균 엔트로피의 95% 신뢰구간은 샘플 단위 재추출 500회로 계산해 CSV에 저장했다. 이후 실패율 역시 샘플별 해당 단계 중 나중에 가능성을 잃는 비율을 구한 뒤 샘플 간 평균했다.']
    for name in ['candidate_axis','future_axis','position_axis']:
        groupkeys=[k for k in axes[name] if k not in ['model','high_entropy']]
        headers=['모델']+[FIELDS[k] for k in groupkeys]+['샘플 수','엔트로피','정답 가능 확률','정답 불가 확률','후속 실패율']
        lines+=['',f'### {TITLES[name]}','', '<div style="overflow-x:auto"><table style="white-space:nowrap;width:max-content"><thead><tr>'+''.join('<th style="white-space:nowrap">'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>']
        for t in tables[name]:
            if not t['high_entropy']:continue
            values=[t['model']]+[str(LABEL.get(str(t[k]),t[k])) for k in groupkeys]+[str(t['n_queries']),f"{t['entropy']:.3f}",f"{t['safe_lower']:.3f}",f"{t['unsafe_mass']:.3f}",f"{100*t['future_failure_rate']:.1f}%"]
            lines.append('<tr>'+''.join('<td style="white-space:nowrap">'+html.escape(v)+'</td>' for v in values)+'</tr>')
        lines+=['</tbody></table></div>','']
    lines+=['','## 관측 결과와 해석','']
    for model in ['qwen25','qwen3']:
        rr=[r for r in rows if r['model']==model and r['high_entropy']];counts2=Counter(r['mass_band'] for r in rr)
        lines.append(f"- {model}: 높은 엔트로피의 파란 단계 {len(rr)}개. 정답 가능 우세 {counts2['safe_dominant']}개, 오류 후보 우세 {counts2['unsafe_dominant']}개, 정답 가능 확률이 중간인 단계 {counts2['competition']}개, 그룹 확정 불가 {counts2['unresolved_band']}개.")
        for ep in ['start','end']:
            for high in [False,True]:
                rr2=[r for r in rows if r['model']==model and r['endpoint']==ep and r['high_entropy']==high];qs=defaultdict(list)
                for row in rr2:qs[row['sample_key']].append(row['outcome']!='endpoint_correct')
                if qs:lines.append(f"  - {LABEL[ep]}, {'상위 엔트로피' if high else '나머지'}: 서로 다른 샘플 {len(qs)}개, 샘플 평균 후속 실패율 {100*np.mean([np.mean(v) for v in qs.values()]):.1f}%.")
    lines+=['','정답 가능 질량이 높은 고엔트로피는 허용되는 후보 사이의 분산을, 낮은 경우는 오류 후보와의 경쟁을 구분하는 데 유용하다. 미확인 확률이 남아 있으면 A와 B만으로 우세를 단정하지 않는다. 단계별 비교는 첫 숫자와 후속 자리, 데이터셋, 자릿수 구성에 영향을 받으므로 통합 실패율을 예측 성능이나 인과 효과로 해석하지 않는다. 특히 소수 마지막 자리에는 이후 숫자가 없어 이후 실패 기회 자체가 적다. 같은 위치의 교차 표와 함께 보아야 한다.','',
      'observed_safe_entropy는 top-30 내 확인된 정답 가능 후보만 조건부 정규화한 엔트로피이며 전체 정답 후보 엔트로피가 아니다. 원래 전체 어휘 H와 혼동하지 않는다. 원분포의 정답 가능 후보와 불가능 후보 엔트로피 기여량도 CSV에 저장했다.','',
      '![정답 가능 질량과 불확실성](safe_mass_vs_entropy.png)','',
      '## 원자료','', '[단계별 수치](per_blue_step.csv) · [후보 축](candidate_axis.csv) · [후속 결과 축](future_axis.csv) · [위치 축](position_axis.csv) · [세 축 교차](three_axis_cross.csv) · [후보×후속](candidate_future.csv) · [자릿수 통제](length_control.csv)']
    write_result_text((relocate_path(OUT/'REPORT.md')), '\n'.join(lines)+'\n')
    write_result_text((relocate_path(OUT/'data.js')), 'window.DATA='+json.dumps(details,ensure_ascii=False)+';\n')
    build_html();progress('browser validation',2,3,dict(counts));browser_validate()
    dump('validation.json',dict(passed=True,blue_steps=len(rows),candidate_selected_safe=True,entropy_unchanged=True,probability_bounds_checked=all(0<=r['safe_lower']<=r['safe_upper']<=1 for r in rows),three_axes=list(axes),fixtures=10))
    progress('complete',3,3,dict(counts));status=json.loads((relocate_path(OUT/'status.json')).read_text());status['state']='complete';dump('status.json',status)

def build_html():
    write_result_text((relocate_path(OUT/'report.html')), '''<!doctype html><meta charset="utf-8"><title>정답 가능 단계의 불확실성</title><link rel="stylesheet" href="../style.css"><style>body{padding:20px}select{max-width:100%;margin:6px}td,th{padding:5px 12px;white-space:nowrap}table{width:max-content}body{overflow-x:auto}video{width:650px;max-width:100%}.safe{color:#357aa0}.unsafe{color:#c65243}.unresolved{color:#888}</style>
<h1>정답 가능성을 유지한 숫자: 후보 · 이후 결과 · 생성 위치</h1><a href="REPORT.md">정량 결과·해석 MD</a><p>첫 숫자 포함. 후보 확률은 raw top-30이며 나머지 질량은 미확인입니다.</p>
<details><summary>A와 B, 샘플 수 읽는 법</summary><p>A: top-30 안에서 정답 가능 후보의 확률 합. B: top-30 안에서 정답 불가능 후보의 확률 합. 예를 들어 각각 0.45, 0.40이면 A=0.45, B=0.40입니다. 나머지 0.15는 미확인 확률입니다. A+B가 1보다 작을 수 있으며 재정규화하지 않습니다.</p><p>샘플 수는 서로 다른 영상·질의 쌍의 수입니다. 보고서의 평균 A와 B는 샘플 내부 평균 후 샘플 간 평균한 값입니다. 최종 응답 정답률이 아닙니다.</p><p>확률 그룹의 1/3과 2/3은 탐색적 분류 기준입니다.</p></details><div id="filters"></div><p id="count"></p><label for="sort">정렬</label><select id="sort"><option value="sample">샘플 ID 순</option><option value="entropy_desc">엔트로피 높은 순</option><option value="entropy_asc">엔트로피 낮은 순</option></select><select id="cases"></select><h2 id="query"></h2><video id="video" controls muted preload="metadata"></video><p><button id="play">GT 구간 재생</button></p><p id="answer"></p><p id="info"></p><p id="bounds"></p><table><thead><tr><th>후보 토큰</th><th>원확률</th><th>판정</th><th>종류</th></tr></thead><tbody id="tokens"></tbody></table>
<script src="data.js"></script><script>
const $=x=>document.getElementById(x);let current=null;const labels={endpoint_correct:'끝까지 유지',next_numeric_failure:'다음 숫자 실패',later_numeric_failure:'나중 숫자 실패',format_failure:'서식 실패',safe_dominant:'정답 가능 우세',unsafe_dominant:'오류 후보 우세',competition:'경쟁',unresolved_band:'분류 경계 미확정',first:'첫 숫자',subsequent:'후속 숫자',safe:'정답 가능',unsafe:'정답 불가능',unresolved:'미확인'};
Object.assign(labels,{start:'시작 시각',end:'종료 시각',hundreds:'백의 자리',tens:'십의 자리',ones:'일의 자리',tenths:'소수 첫째 자리',true:'불확실성 상위 25%',false:'그 외 단계',number:'숫자',format:'서식',other:'기타',safe_dominant:'정답 우세',unsafe_dominant:'오답 우세',competition:'경쟁',unresolved_band:'미확정'});const fieldNames={model:'모델',dataset:'벤치마크',endpoint:'예측 대상',role:'자릿수',digit_stage:'숫자 순서',outcome:'이후 결과',mass_band:'후보 확률 그룹',high_entropy:'불확실성 수준'};
DATA.forEach((r,i)=>r._index=i);
const fields=['model','dataset','endpoint','role','digit_stage','outcome','mass_band','high_entropy'];
for(const key of fields){let sel=document.createElement('select');sel.id=key;sel.add(new Option(fieldNames[key]+' 전체',''));for(const v of [...new Set(DATA.map(r=>String(r[key])))])sel.add(new Option(labels[v]||v,v));sel.onchange=update;$('filters').append(sel)}
function update(){const list=DATA.filter(r=>fields.every(k=>!$(k).value||String(r[k])===$(k).value)).sort((a,b)=>{const order=$('sort').value;const entropy=order==='entropy_desc'?b.entropy-a.entropy:order==='entropy_asc'?a.entropy-b.entropy:0;return entropy||a.sample_key.localeCompare(b.sample_key,'en',{numeric:true})||a.model.localeCompare(b.model,'en')||a.ar_step-b.ar_step||a.endpoint.localeCompare(b.endpoint,'en')});$('cases').replaceChildren();for(const r of list){let index=r._index;$('cases').add(new Option(r.model+' · '+r.sample_key+' · '+labels[r.endpoint]+' · '+labels[r.role]+' · 생성 단계 '+r.ar_step+' · H='+r.entropy.toFixed(3),index))}$('count').textContent=new Set(list.map(r=>r.sample_key)).size+'개의 서로 다른 샘플, '+list.length+'개의 숫자 생성 단계 (두 모델 선택 시 동일 샘플의 모델별 단계 포함)';if(list.length)show();else{$('tokens').replaceChildren();$('info').textContent='해당 조건 없음';$('video').pause()}}
function show(){gtPlaybackEnd=null;current=DATA[Number($('cases').value)];const r=current;$('query').textContent=r.query;if($('video').getAttribute('src')!==r.media)$('video').src=r.media;$('answer').textContent=r.answer+' | GT '+JSON.stringify(r.gt_span);$('info').textContent='이전까지 생성한 문자열 '+JSON.stringify(r.prefix_before)+' → '+r.selected+' | H='+r.entropy.toFixed(3)+' nat | '+labels[r.outcome]+' | 정답 가능성 최초 상실 단계 '+(r.failure_ar_step??'없음')+' | 상위 25% 기준 '+r.entropy_q75.toFixed(3)+' · 상위 25% 기준을 계산한 숫자 단계 수: '+r.stratum_n;$('bounds').textContent='정답 가능 확률: '+r.safe_lower.toFixed(4)+' | 정답 불가 확률: '+r.unsafe_mass.toFixed(4);$('tokens').replaceChildren();for(const t of r.candidates){let tr=document.createElement('tr');tr.className=t.state;for(const text of [JSON.stringify(t.token),t.probability.toPrecision(5),labels[t.state],labels[t.kind]]){let td=document.createElement('td');td.textContent=text;tr.append(td)}$('tokens').append(tr)}}
let gtPlaybackEnd=null;
$('sort').onchange=update;$('cases').onchange=show;
$('play').onclick=async()=>{if(current){const video=$('video');gtPlaybackEnd=current.gt_span[1];video.currentTime=current.gt_span[0];try{await video.play()}catch(e){gtPlaybackEnd=null;if(e.name!=='AbortError')$('info').textContent='영상 재생 실패: '+e.message}}};
$('video').onpause=()=>{gtPlaybackEnd=null};
$('video').onseeking=()=>{if(current&&($('video').currentTime<current.gt_span[0]||$('video').currentTime>=current.gt_span[1]))gtPlaybackEnd=null};
$('video').ontimeupdate=()=>{if(gtPlaybackEnd!==null&&$('video').currentTime>=gtPlaybackEnd){gtPlaybackEnd=null;$('video').pause()}};update();
</script>''')
def browser_validate():
    from timelens_paths import RESOURCES
    sys.path.insert(0,str(relocate_path(RESOURCES/'browser_deps')));os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(relocate_path(RESOURCES/'browser_cache'));os.environ['LD_LIBRARY_PATH']=str(relocate_path(RESOURCES/'browser_libs/usr/lib/x86_64-linux-gnu'))+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['FONTCONFIG_FILE']=str(relocate_path(RESOURCES/'browser_libs/fonts.conf'))
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required']);page=browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8774/blue_analysis/');page.wait_for_selector('#tokens tr')
        for model in ['qwen25','qwen3']:
            page.select_option('#model',model)
            for ep in ['start','end']:
                page.select_option('#endpoint',ep);assert page.locator('#tokens tr').count()==30
        page.select_option('#high_entropy','true');assert page.locator('#cases option').count()>0
        page.wait_for_function("document.querySelector('video').readyState>=2",timeout=30000);page.click('#play');page.wait_for_function("!document.querySelector('video').paused && document.querySelector('video').currentTime > current.gt_span[0]+0.15",timeout=10000);page.evaluate("document.querySelector('video').pause()");page.screenshot(path=str(relocate_path(OUT/'preview.png')),full_page=True);browser.close()
    assert not errors,errors
    dump('browser_validation.json',dict(passed=True,models=2,endpoints=2,video_playback=True,candidate_rows=30,high_entropy_filter=True,errors=errors))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);dump('status.json',dict(state='failed',timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),error=traceback.format_exc()));raise
