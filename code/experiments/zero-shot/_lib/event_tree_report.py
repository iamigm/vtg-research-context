"""Reports and shared-view integration for event proposal → boundary search."""
from event_tree_core import *
from rollout_method_docs import formalize, refresh_spec
import collections,csv,itertools,subprocess,sys

LABELS={'likelihood_best_first':'Likelihood best-first','value_best_first':'Value best-first','value_beam':'Value beam','diverse_beam':'Diverse beam','uct':'UCT + progressive widening','puct':'PUCT + progressive widening','neural_controller':'Neural controller','self_refine':'Self-Refine','submodular_coverage':'Submodular coverage','reward_balanced':'Reward-balanced allocation','sequential_halving':'Sequential halving','nucleus_frontier':'Candidate nucleus frontier','typicality_frontier':'Candidate typicality frontier','commit_then_refine':'Commit then refine','token_seed_best_first':'Token-seed best-first'}
LITERATURE=[
 ('Tree of Thoughts','https://arxiv.org/abs/2305.10601','BFS/DFS와 중간 상태 평가·backtracking. 여기서는 temporal event/action 상태와 deterministic operators로 적용한다.'),
 ('Diverse Beam Search','https://arxiv.org/abs/1610.02424','beam의 유사 후보 집중을 diversity objective로 완화. 본 실험은 token Hamming penalty가 아닌 시간 구간의 semantic-action beam adaptation이다.'),
 ('Self-Evaluation Guided Beam Search','https://arxiv.org/abs/2305.00633','중간 단계 self-evaluation으로 탐색을 유도. 논문의 stochastic beam을 재현하지 않으며 frozen VLM critic과 deterministic beam을 사용한다.'),
 ('Language Agent Tree Search','https://arxiv.org/abs/2310.04406','선택·확장·value backup·reflection. 외부 환경의 검증된 reward 대신 VLM의 noisy visual judgement를 사용한다.'),
 ('Self-Refine','https://arxiv.org/abs/2303.17651','추가 학습 없이 같은 모델의 feedback을 다음 generation에 반영한다.'),
 ('Scaling LLM Test-Time Compute Optimally','https://arxiv.org/abs/2408.03314','proposal distribution 수정과 verifier-guided 탐색을 구분하고 전체 비용을 측정한다. 학습한 PRM을 도입하는 재현 실험은 아니다.'),
 ('Stochastic Beams / Gumbel-Top-k','https://proceedings.mlr.press/v97/kool19a.html','sequence sampling without replacement는 독립 sampling의 중복 문제를 직접 다룬다. 본 실행은 이전 사용자 지시에 따라 random sampling을 사용하지 않으며 unique branch enumeration을 유지한다. Gumbel sampling을 실행했다고 부르지 않는다.'),
 ('Nucleus Sampling','https://arxiv.org/abs/1904.09751','원래는 token distribution의 누적확률 support를 정한 뒤 sampling한다. 여기서는 공통 사건 후보의 canonical answer probability를 정규화한 finite-list support를 deterministic하게 탐색한다. 원래 nucleus sampling의 재현이 아니다.'),
 ('Locally Typical Sampling','https://arxiv.org/abs/2202.00666','surprisal과 entropy의 차이로 support를 구성한다. 본 실행은 finite event-list typicality adaptation이며 token-level locally typical sampling과 구분한다.'),
 ('Sequential Monte Carlo Steering','https://arxiv.org/abs/2306.03081','여러 particle과 weight·resampling으로 generation을 제어한다. 본 실행의 reward-balanced allocation은 결정적 resource allocation이며 posterior SMC나 unbiased estimator를 주장하지 않는다.'),
 ('ReSCALE / Sequential Halving','https://arxiv.org/abs/2603.21162','최근 연구가 PUCT scaling 악화와 sequential halving을 비교했다. 본 실행은 Gumbel noise·학습 value head 없이 deterministic lookahead halving을 비교하며 해당 논문 성능을 일반화하지 않는다.'),
 ('Contrastive Decoding','https://arxiv.org/abs/2210.15097','expert–amateur log-likelihood 대비는 또 다른 decoding objective다. 숫자 token 반복을 억제하는 목적과 localization의 목적이 다르며, 이번 비교에서는 backbone 효과를 고정하기 위해 별도 amateur model을 추가하지 않는다.'),
 ('Submodular summarization / facility location','https://aclanthology.org/P11-1052/','집합의 coverage와 quality를 함께 최적화하는 수학적 접근을 사건 후보 선택에 적용한다.'),
 ('Best Algorithms for Approximating the Maximum of a Submodular Set Function','https://pubsonline.informs.org/doi/10.1287/moor.3.3.177','고정된 normalized monotone submodular objective와 cardinality constraint에서 greedy 근사 보장. IoU나 전체 adaptive search의 최적성을 보장하지 않는다.')]

SPEC='''# Event proposal and tree search · pilot900

고정900 query, dataset별300개다. 300⊂600⊂900 고정 prefix를 유지한다. 모델은 공유 Qwen3-VL-8B-Instruct, 기존2 FPS·spatial resolution·frame IDs/PTS·input tensor hash·video_query input order를 유지한다. GT는 생성·proposal·선택·critic에 사용하지 않고 보고 단계에서만 사용한다.

## 구조

기존 deterministic rollout 첫8개 외에 모든 leading digit0–9에서 conditional rollout을 수행하며 raw probability가 작다고 제외하지 않는다. 또한 같은 영상에 대해 최대6개의 다른 가능한 사건과 짧은 시각적 근거를 모델이 새로 제안한다. leading digit 이후에는 start와 end를 함께 새로 생성한다. semantic proposal은 새 context의 모델 출력이며 원래 timestamp 분포와 구분한다. 시간 표기는 초 단위를 요청하고 분:초 예외는 parser에서만 처리한다.

동일 구간의 seed들은 provenance와 raw token path를 보존하면서 exact interval만 중복 제거한다. IoU가 낮은 쌍만 사건 후보로 고르는 기존187/595 규칙을 적용하지 않는다. 근접·겹치는 사건 후보도 보존한다. invalid·out-of-duration proposal은 clipping하지 않고 기록한다. 모델이 사건을 찾지 못하면 임의 사건을 채워 넣지 않는다.

공통 candidate마다 사건 적합성과 핵심 모순을 독립 Yes/No likelihood로 평가한다. P(Yes)/(P(Yes)+P(No))와 raw option mass를 함께 저장한다. 사건 적합성≥.5이고 확인된 핵심 모순이 없는 후보만 boundary refinement를 수행한다. 이 gate 자체가 틀릴 수 있으므로 gate 이전·이후 oracle과 gate가 정답 후보를 제거한 수를 반드시 보고한다.

boundary refinement에서는 새 시각을 greedy로 생성하고, 별도의 시각적 observation과 사건·경계·모순 평가를 수행한다. start가 바뀌면 end는 새 start에 조건화해 다시 생성하며 예전 end suffix를 복사하지 않는다. 핵심 actor/action/object/occurrence 모순은 두 독립 질문에서 P(Yes)≥.8, raw Yes/No mass≥.01일 때만 해당 사건을 blocked로 표시한다. 이는 모델이 예측한 모순이며 검증된 영상 사실이나 외부 verifier reward가 아니다. 다른 사건으로 backtracking하거나 새 사건을 proposal할 수 있다.

각 사건 아래 onset/offset/joint operators를 각각 한 번씩 적용한다. 최대 expansion depth3, node당 최대3 child다. 같은 interval이 다시 생성될 수 있으며 경로는 별도 node로 보존한다. 같은 operation 결과를 다시 가져오는 cache hit를 독립 Monte Carlo 관측으로 세지 않는다. 백업은 새로 평가한 descendant에 대해서만 수행한다.

동일 context의 generated prefix들은 token trie다. 재검토 context가 바뀌면 모델의 조건부 분포도 바뀌므로, 상위 구조는 context hash와 action history를 가진 search tree로 표현한다. 각 node는 parent 하나를 가지며 다른 context의 path probability를 하나의 전역 LM probability로 곱하지 않는다.

## 방법과 수식

모든 방법은 공통 proposal·critic·boundary operators와 최종 model pairwise selector를 사용한다. 원 논문을 그대로 재현하는 결과가 아니라 video localization에 맞춘 아래의 고정 adaptation을 비교한다. 추가 학습·새 가중치·독립 random sampling은 없다.

| method | 선택·확장 규칙 |
|---|---|
| likelihood_best_first | 동일 원래 context에서 canonical timestamp answer의 EOS 포함 log probability가 큰 frontier부터 확장. 이미 확장한 수×.3 penalty |
| value_best_first | visual value−.05×이미 확장한 수로 frontier 순위 |
| value_beam | 한 층의 최대4개 상태를 유지하고 child layer를 평가·축소. 막히면 deferred frontier로 backtracking |
| diverse_beam | value beam에 선택 집합과의 최대 temporal IoU×.25 penalty를 추가 |
| uct | Q+1.4×sqrt(log(Nparent+2)/(Nchild+1)), progressive widening, descendant value backup |
| puct | Q+1.4×Pchild×sqrt(Nparent+1)/(Nchild+1), prior=.9×정규화 사건 적합성+.1×uniform, progressive widening |
| neural_controller | 모델이 현재 frontier·모순을 읽고 다음 node와 onset/offset/joint 또는 new proposal을 결정. 잘못된 action은 기록 후 deterministic fallback |
| self_refine | 현재 사건의 feedback을 받아 sequential refinement. 모순·frontier 소진 시 다른 사건으로 이동 |
| submodular_coverage | 아래의 facility-location objective로 최대4개 사건을 선택하고 선택한 사건들에 visit 균형을 맞춰 refinement |
| reward_balanced | exp(value/.25)/(1+event visits)로 다음 사건에 연산 배분. 결정적 weighted-fair allocation이며 posterior particle sampling은 아님 |
| sequential_halving | 최대8개의 diverse 사건을 먼저 확인하고, visit target1→3→7…마다 descendant value가 낮은 절반을 줄임. noisy IID arm 가정의 보장을 주장하지 않음 |
| nucleus_frontier | finite unique canonical event answer의 q_i∝exp(logP_i)에서 누적 .95 support를 선택하고 q_i/(1+visits)로 확장 |
| typicality_frontier | 같은 finite q에서 H=−Σqlogq, abs(−logq_i−H)가 작은 순서로 .95 support를 구성해 확장 |
| commit_then_refine | 한 사건에 commit한 뒤 같은 사건만 정교화. 다른 사건으로 이동하는 generation은 유효 최종 후보에서 제외. backtracking/reproposal의 ablation |
| token_seed_best_first | 기존 첫8개 token 경로만 초기 후보로 사용하고 새 사건 proposal을 하지 않는 value best-first ablation |

공통 visual value는 event_probability×(1−contradiction_probability)×(.5+.5×boundary_probability)이며 coarse event의 boundary_probability는 .5다. PUCT/UCT의 Q는 새 descendant들의 평균 value, 미방문은 해당 node의 초기 value다. root progressive widening은 min(root count,2+floor(2sqrt(root visits+1)))다. 확장된 최종 candidate를 다시 평가한 cache hit를 새 simulation reward로 넣지 않는다. deterministic greedy completion을 사용하므로 이를 random-rollout MCTS 재현이라고 부르지 않는다.

## Analytic method: submodular coverage

고정된 eligible event universe V에서 q_i는 위 visual value다. w_i=.9 q_i/Σq+.1/|V|, τ=max(2초,.05×영상 길이), κ(i,j)=.5 IoU(i,j)+.5 exp(−|center_i−center_j|/τ)로 정의한다.

F(S)=Σ_i w_i max_{j∈S} κ(i,j)+.2 Σ_{j∈S}q_j, F(∅)=0, |S|≤4.

κ와 w,q가 음수가 아니므로 첫 항은 monotone submodular facility location이고 둘째 항은 nonnegative modular다. 따라서 F는 normalized monotone submodular다. 각 단계에서 F(S∪{j})−F(S)가 가장 큰 j를 더하는 greedy는 고정된 V·q·κ와 cardinality budget에서 F(greedy)≥(1−1/e)F(opt)를 만족한다. 작은 cardinality4의 실제 최적값도 enumeration으로 계산해 검증한다.

이 보장은 F에 관한 것이며 GT IoU, 사건 정체성, unequal token cost의 전체 adaptive search 최적성을 보장하지 않는다. 모순·새 proposal로 universe가 바뀌면 다음 snapshot에서 F를 다시 구성한다. 각 snapshot의 정적 subset 선택에만 보장을 적용한다. 근거 점수가 틀릴 때 수학적 최적화도 틀린 사건을 고를 수 있으므로 실제 성능과 함께 비교한다.

## 예산과 선택

공통 proposal·평가는 query마다 한 번 수행하고 모든 비교 방법의 deployment 비용에 합산한다. token-seed ablation은 실제로 필요한 seed 평가 비용만 별도 합산한다. 과거 seed 생성 비용과 공통 video prefill도 별도 표시한다. cache reuse 때문에 실제 실행 시간이 줄어든 것과 방법 자체의 conceptual cost를 구분한다.

추가 search decoder input token 상한은4,096·8,192다. 공통 video prefill 이후의 새 prompt text, prefix replay, generated token을 모두 센다. 각 model operation은 같은 video root부터 새 text KV로 계산해 비용 정의를 고정한다. 동일 query·동일 context·동일 operation은 memoization하되 각 방법에 원래 cold-cache token/시간 비용을 charge한다. 방법 실행 순서는 query hash로 회전한다.

최종 선택은 별도로 최대2,048 token을 예약하고, value와 diversity로 추린 최대4개 후보를 모델이 두 presentation order에서 pairwise 비교한다. 단순 max probability 후처리가 아니다. 선택 실패는 실패로 남긴다. 각 cap을 fresh logical execution으로 평가하므로 낮은 cap에서 중단된 expansion의 일부를 높은 cap에 이중 charge하지 않는다. 이전 cap의 selector 비용은 높은 cap에 넣지 않는다. 미완료 expansion은 tree에 commit하지 않되 이미 쓴 token은 비용에 포함한다.

공통 proposal만으로 model selection을 수행하는 wide/token-seed controls도 별도로 평가한다. 두 search cap의 합계 예산은 공통 비용+각각 최대6,144/10,240 token이다. 추가 호출 수·generated token·synchronized model time·실제 cache execution time을 모두 기록한다. token 상한이 같은 것이 FLOPs·latency 일치를 뜻하지 않는다.

## 보고 지표

IoU=시간 교집합 길이/합집합 길이. mIoU는 query별 IoU 평균, R@t는 IoU≥t인 query 비율(t=.3/.5/.7)이며 높을수록 좋다. Δ는 같은 query의 기존 greedy 대비, Δpp는 R 차이×100이다. 파싱 실패·EOS 미도달·무효 최종 응답은0점. 원래 greedy는 기존 평가 정의로 보존하고 새 proposal의 out-of-duration은 별도 flag와 실패로 남기며 clipping하지 않는다.

전체900 및 고정300/600 prefix, dataset별, 영상 길이별, GT/영상 길이 비율별, 두 축 교차표를 제공한다. 작은 칸(n<10)·빈 칸을 표시한다. video 단위 paired bootstrap400회(seed42)의95% CI와 cap 증가의 paired Δ를 보고한다. 겹치는 query와 조건을 독립 표본으로 합산하지 않는다.

사건 proposal은 기존8/32와 신규 후보의 unique interval 수, temporal cluster 수(proxy), oracle mIoU/R@.3/.5/.7, unique 후보 수별1/4/8/16/32 oracle, 원래8에서 실패→신규에서 회복된 query 수를 보고한다. 같은 interval의 여러 token path와 실제 사건 수는 다르며 cluster도 semantic event gold가 아니다.

gate 이전·이후 oracle을 비교하고 GT 최고 후보를 gate가 제거한 사례를 표시한다. 최종 선택이 raw oracle에 못 미치는 차이는 selection gap으로 보고한다. 선택한 사건 root의 IoU/endpoint MAE와 refinement 결과를 비교해 사건 선택과 boundary refinement를 분리한다. endpoint MAE=(|start−GTstart|+|end−GTend|)/2초, 낮을수록 좋으며 paired-valid 분모를 명시한다. root와 final의 IoU≥.5 subset도 따로 본다. 원래 greedy IoU<.1→새 IoU≥.5 회복, 원래≥.5→새<.5 손상 비율도 각각 해당 baseline subset을 분모로 보고한다.

confirmed model contradiction 수, 실제 다른 event node로 이동한 수, 그 이후 IoU 변화, commit_then_refine 대비 차이를 보고한다. 모순 판정은 모델의 판단이며 잘못된 prune도 GT를 사용한 사후 분석으로 공개한다. raw token probability, generation step의 누적 log probability, 원래 leading digit fork의 부모·child distributions를 저장한다. semantic/review context의 probability와 원래 LM context의 probability를 혼합하지 않는다.

## 실행·보존

원자료는 artifacts/runs/rollout/event_tree_search/pilot/900에 한 번 저장하고300·600은 참조한다. 신규900 query의 canonical family이며 기존 deterministic/reconsideration 데이터를 덮어쓰지 않는다. 공통 서버8780과 계열 대표 viewer 하나를 사용한다. 고정 첫300/600이 모두 완료되면 MD·interactive viewer·검증·공통 index 갱신을 자동 수행하며900 완료 후 최종 보고와 검증을 자동 수행한다. 실행 상태와 오류는 heartbeat log와 metadata/status.json에 남긴다.
'''

def cost(records,refs):return {name:sum(records[k]['cost'].get(name,0) for k in refs) for name in ('decoder_input_tokens','model_seconds','forward_calls','generated_tokens') }|dict(model_evaluations=len(refs))
def seed_refs(data):
 refs=[]
 for e in data['proposals']['events']:
  if any(p['source']=='token_seed' for p in e['provenance']):refs+=e['assessment']['calls']+([e['prior_call']] if e.get('prior_call') else [])
 return refs
def temporal_group_count(spans,threshold=.5):
 groups=[]
 for span in spans:
  match=next((g for g in groups if all(iou(span,p)>=threshold for p in g)),None)
  if match is None:groups.append([span])
  else:match.append(span)
 return len(groups)
def initialize(cohort):
 spec=SPEC+'\n## 문헌 조사와 적용 범위\n\n'+'\n'.join(f'- [{t}]({u}): {note}' for t,u,note in LITERATURE)+'\n'
 (REPORT/'SPEC.md').write_text(formalize(spec,'event_tree_search',900,spec=True))
 if not (REPORT/'REPORT.md').exists():(REPORT/'REPORT.md').write_text('# Event tree search · pilot900\n\n상태: running. 아직 완료된 성능 결과가 아니다.\n\n[설계·방법·수식·문헌](SPEC.md)\n')
 samples=[]
 for s in cohort:
  baseline=read(SOURCE/'web/samples'/f'{s["sample_key"]}.json');old_report=ROOT/'results/rollout/deterministic/pilot/900/viewer.html';video=(old_report.parent/baseline['video']).resolve()
  assert video.is_file(),video
  samples.append(dict(key=s['sample_key'],dataset=s['dataset'],query=s['query'],duration=s['duration'],gt=s['span'][0],greedy=baseline['sample']['greedy_pair'],greedy_iou=baseline['sample']['greedy_iou'],video=os.path.relpath(video,ROOT.parent.resolve())))
 write(OUT/'web/samples.json',samples);build_view(900)

def build_view(n):
 dest=ROOT/'results'/FAMILY/'pilot'/str(n);dest.mkdir(parents=True,exist_ok=True)
 template=Path(__file__).with_name('event_tree_viewer.html').read_text();boot=dict(store=os.path.relpath(OUT,dest),scope=n,experiments=os.path.relpath(ROOT.parent,dest),labels=LABELS,methods=METHODS)
 template=template.replace('__CONFIG__',json.dumps(boot,ensure_ascii=False)).replace('__THEME__',os.path.relpath(ROOT.parent/'_lib/experiment_theme.css',dest))
 sys.path.insert(0,str(ROOT.parent/'_lib'));from build_index import add_navigation
 (dest/'viewer.html').write_text(add_navigation(dest/'viewer.html',template))
 archive=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n)/'code/reporting';archive.mkdir(parents=True,exist_ok=True);(archive/'event_tree_viewer.html').write_bytes(Path(__file__).with_name('event_tree_viewer.html').read_bytes())

def summary(rows):
 if not rows:return None
 n=len(rows);return dict(n=n,miou=sum(r['iou'] for r in rows)/n,r30=sum(r['iou']>=.3 for r in rows)/n,r50=sum(r['iou']>=.5 for r in rows)/n,r70=sum(r['iou']>=.7 for r in rows)/n,failure=sum(r['span'] is None for r in rows)/n,tokens=sum(r['tokens'] for r in rows)/n,seconds=sum(r['seconds'] for r in rows)/n)
def table(rows,methods):
 baseline=summary([r for r in rows if r['method']=='greedy']);out=['| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | decoder tokens | model seconds | failure |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in methods:
  val=summary([r for r in rows if r['method']==method])
  if not val:continue
  columns=[f"{val['miou']:.4f} ({val['miou']-baseline['miou']:+.4f})"]+[f"{100*val[m]:.2f} ({100*(val[m]-baseline[m]):+.2f})" for m in ('r30','r50','r70')]
  out.append('| '+method+' | '+str(val['n'])+' | '+' | '.join(columns)+f" | {val['tokens']:.1f} | {val['seconds']:.2f} | {100*val['failure']:.2f}% |")
 return out

def analytic_audit(events,duration):
 import numpy as np
 roots=[dict(e,id=i+1,assessment=e['assessment']) for i,e in enumerate(events) if e['assessment']['eligible']]
 if not roots:return dict(n=0,greedy=0.,optimal=0.,ratio=None)
 selected=submodular_select(roots,4,duration);value=coverage_value(roots,selected,duration);q=np.array([r['assessment']['value'] for r in roots]);w=.9*q/(q.sum() or 1)+.1/len(q);tau=max(2,.05*duration)
 kernel=np.array([[.5*iou(a['span'],b['span'])+.5*math.exp(-abs(sum(a['span'])/2-sum(b['span'])/2)/tau) for b in roots] for a in roots]);combos=np.array(list(itertools.combinations(range(len(roots)),min(4,len(roots)))));scores=w@kernel[:,combos].max(axis=2)+.2*q[combos].sum(axis=1);optimal=float(scores.max());ratio=value/optimal if optimal else 1.;assert ratio>=1-1/math.e-1e-9
 return dict(n=len(roots),greedy=value,optimal=optimal,ratio=ratio,selected=[r['id'] for r in selected])

def report(n,refresh=True):
 cohort=read(COHORT)[:n];dest=ROOT/'results'/FAMILY/'pilot'/str(n);out=ROOT/'artifacts/runs'/FAMILY/'pilot'/str(n);dest.mkdir(parents=True,exist_ok=True);rows=[];proposals=[];audits=[];keys=[]
 methods=['greedy','proposal_only_token_seed','proposal_only_wide']+[f'{m}@{cap}' for m in METHODS for cap in CONFIG['search_token_caps']]
 for row in cohort:
  key=row['sample_key'];data=read(OUT/'responses'/f'{key}.json');old=read(SOURCE/'rollouts'/f'{key}.json');assert data['config_hash']==signature(CONFIG);assert data['input_hashes']==old['input_hashes'];assert sha(data['baseline_reference'])==data['baseline_sha256'];assert set(data['methods'])==set(METHODS)
  gt=row['span'][0];greedy=pair(old['result']['candidates'][0]);base=iou(greedy,gt);records=data['model_calls'];events=data['proposals']['events'];common=cost(records,data['proposals']['calls']);seed=cost(records,seed_refs(data));seed_generation=old['result']['candidates'][min(7,len(old['result']['candidates'])-1)]['decoder_tokens'];keys.append(key)
  for record in records.values():
   if record.get('probabilities'):assert abs(sum(record['probabilities'])-1)<1e-5 and 0<=record['option_mass']<=1+1e-5
   for token in record.get('tokens',[]):assert 0<=token['probability']<=1+1e-7 and token['cumulative_log_probability']<=1e-6
  def add(method,span,tokens,seconds,**extra):
   rows.append(dict(key=key,dataset=row['dataset'],video=row['video_path'],duration=row['duration'],ratio=(gt[1]-gt[0])/row['duration'],method=method,span=span,iou=iou(span,gt),baseline_iou=base,tokens=tokens,seconds=seconds,endpoint_mae=(abs(span[0]-gt[0])+abs(span[1]-gt[1]))/2 if span else None,**extra))
  add('greedy',greedy,old['result']['candidates'][0]['decoder_tokens'],old['result']['candidates'][0]['elapsed_seconds'])
  for label,result in data['proposal_only'].items():
   cc=seed if label=='token_seed' else common
   add('proposal_only_'+label,result['span'],seed_generation+cc['decoder_input_tokens']+result['cost']['decoder_input_tokens'],cc['model_seconds']+result['cost']['model_seconds'])
  for method,variants in data['methods'].items():
   for cap,result in variants.items():
    assert result['search_cost']['decoder_input_tokens']<=int(cap);assert result['selection_cost']['decoder_input_tokens']<=CONFIG['selection_token_cap'];nodes=result['nodes'];selected=nodes[result['selected']] if result['selected'] else None;root=nodes[selected['event_id']] if selected else None;cc=seed if method=='token_seed_best_first' else common
    assert all(x['id']==i and (i==0 or x['parent']<i) for i,x in enumerate(nodes));assert len({x['id'] for x in nodes})==len(nodes)
    if selected:assert selected['assessment']['eligible'] and not root['assessment']['blocked']
    before=iou(root['span'],gt) if root else None;raw_oracle=max((iou(x.get('span'),gt) for x in nodes[1:]),default=0)
    blocked=[x for x in nodes[1:] if x['assessment']['blocked']];actions=result['actions'];moves=0
    for i,a in enumerate(actions):
     if a['action']=='contradiction_backtrack' and any(z['action']=='refine' and nodes[z['parent']]['event_id']!=a['event'] for z in actions[i+1:]):moves+=1
    add(method+'@'+cap,result['span'],seed_generation+cc['decoder_input_tokens']+result['search_cost']['decoder_input_tokens']+result['selection_cost']['decoder_input_tokens'],cc['model_seconds']+result['search_cost']['model_seconds']+result['selection_cost']['model_seconds'],proposal_root_iou=before,root_span=root['span'] if root else None,root_mae=(abs(root['span'][0]-gt[0])+abs(root['span'][1]-gt[1]))/2 if root else None,oracle=raw_oracle,selection_gap=raw_oracle-iou(result['span'],gt),new_nodes=sum(x['source_event'] is None for x in nodes[1:]),expansions=result['expansions'],contradictions=len(blocked),actual_backtracks=moves,gate_count=sum(e['assessment']['eligible'] for e in events),search_tokens=result['search_cost']['decoder_input_tokens'],selection_tokens=result['selection_cost']['decoder_input_tokens'])
  orig=[];seen=set()
  for c in old['result']['candidates']:
   p=pair(c)
   if p and tuple(p) not in seen:orig.append(p);seen.add(tuple(p))
  fresh=[e['span'] for e in events if usable(e['span'],row['duration'])];eligible=[e['span'] for e in events if e['assessment']['eligible']];raw=max((iou(p,gt) for p in fresh),default=0);gated=max((iou(p,gt) for p in eligible),default=0);pinfo=dict(key=key,dataset=row['dataset'],old_unique=len(orig),new_unique=len(fresh),eligible=len(eligible),invalid=len(events)-len(fresh),oracle_raw=raw,oracle_gated=gated,gate_lost_best=raw>gated+1e-9,old8=max((iou(pair(c),gt) for c in old['result']['candidates'][:8]),default=0),old32=max((iou(pair(c),gt) for c in old['result']['candidates']),default=0))
  pinfo['old8_unique']=len({tuple(pair(c)) for c in old['result']['candidates'][:8] if pair(c)})
  for k in (1,4,8,16,32):pinfo[f'old_unique_oracle{k}']=max((iou(p,gt) for p in orig[:k]),default=0);pinfo[f'new_unique_oracle{k}']=max((iou(p,gt) for p in fresh[:k]),default=0)
  old8=[pair(c) for c in old['result']['candidates'][:8] if pair(c)]
  for threshold in (.3,.5,.7):pinfo[f'old_clusters{int(threshold*100)}']=temporal_group_count(old8,threshold);pinfo[f'new_clusters{int(threshold*100)}']=temporal_group_count(fresh,threshold)
  proposals.append(pinfo);audits.append(dict(key=key,**analytic_audit(events,row['duration'])))
 for category in ('metrics','metadata','validation'):(out/category).mkdir(parents=True,exist_ok=True)
 write(out/'metadata/source_reference.json',dict(queries=n,cohort=str(COHORT),sha256=sha(COHORT),responses=str(OUT/'responses'),prefix=f'0:{n}'))
 write(out/'metrics/per_query.json',rows);write(out/'metrics/proposals.json',proposals);write(out/'metrics/analytic_audit.json',audits)
 fields=sorted(set().union(*(r.keys() for r in rows)))
 with (out/'metrics/per_query.csv').open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 text=[f'# Event proposal and tree search · pilot{n}','',f'완료: {n} query, dataset별{n//3}. 15개 방법×2 search budget 및 proposal-only controls를 같은 고정 query에서 비교했다. 반복 조건 수를 query 수로 합산하지 않았다.','', '[설계·수식·문헌](SPEC.md)' if n==900 else '[설계·수식·문헌](../900/SPEC.md)', '',
 '## 지표','', 'mIoU=전체 query의 temporal IoU 평균, R@t=IoU≥t인 query 수/전체 query 수(t=.3/.5/.7), 높을수록 좋다. IoU=시간 교집합/합집합. Δ는 기존 greedy 대비이며 R의 변화는pp다. 실패는0점. decoder tokens는 seed 생성·공통 proposal/critic·search·최종 선택을 합산한 conceptual cost이며 exact cache reuse로 줄어든 실행 비용과 다르다. model seconds는 새 model operation의 cold-cache 측정 시간 합이고 공통 video prefill 및 과거 seed 생성 시간은 제외되어 latency로 직접 해석하지 않는다. 같은 token 상한이 같은 FLOPs를 뜻하지 않는다.','', '## 전체 성능','']+table(rows,methods)
 text+=['','## 사건 proposal과 oracle','','duration-valid는 시간 범위 검사를 통과한 뒤 model gate 적용 전이며, gated는 사건 적합성 gate 이후다. oracle은 GT를 보고 가장 좋은 후보를 고른 상한이며 실제 선택 결과가 아니다. new는 실제 생성된 유효 unique interval이며 의미적으로 다른 사건 개수가 아니다. 시간 범위 검사 이전의 oracle은 실패 분석에 별도 표시한다.','','| source | average unique candidates | oracle mIoU | R@.3 | R@.5 | R@.7 |','|---|---:|---:|---:|---:|---:|']
 for label,count_key,value_key in [('old paths first8','old8_unique','old8'),('old paths first32','old_unique','old32'),('new proposals duration-valid','new_unique','oracle_raw'),('new proposals gated','eligible','oracle_gated')]:
  vals=[p[value_key] for p in proposals];count=sum(p[count_key] for p in proposals)/n if count_key else 8
  text.append(f'| {label} | {count:.2f} | {sum(vals)/n:.4f} | '+ ' | '.join(f'{100*sum(x>=t for x in vals)/n:.2f}' for t in (.3,.5,.7))+' |')
 text+=['',f"Gate가 raw 최고 IoU를 낮춘 query: {sum(p['gate_lost_best'] for p in proposals)}/{n}. old8 oracle<.5 → new raw oracle≥.5: {sum(p['old8']<.5 and p['oracle_raw']>=.5 for p in proposals)}/{sum(p['old8']<.5 for p in proposals)}. 이는 gate의 false negative와 proposal coverage를 사후 GT로 분석한 값이다.",f"Temporal complete-link cluster(IoU≥.5)의 query당 평균은 old8 {sum(p['old_clusters50'] for p in proposals)/n:.2f} → new proposals {sum(p['new_clusters50'] for p in proposals)/n:.2f}다. 이는 의미적 사건 정체성 gold가 아니며 threshold .3/.7 원자료도 보존한다.",'','| unique candidate budget | old oracle mIoU | new oracle mIoU | old R@.3 | new R@.3 | old R@.5 | new R@.5 | old R@.7 | new R@.7 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k in (1,4,8,16,32):
  a=[p[f'old_unique_oracle{k}'] for p in proposals];b=[p[f'new_unique_oracle{k}'] for p in proposals];text.append(f'| {k} | {sum(a)/n:.4f} | {sum(b)/n:.4f} | '+' | '.join(f'{100*sum(x>=t for x in values)/n:.2f}' for t in (.3,.5,.7) for values in (a,b))+' |')
 text+=['','## 사건 선택 이후 boundary refinement와 backtracking','','root는 최종 선택한 사건의 coarse interval이다. Δroot는 final IoU−root IoU의 paired-valid 평균이다. MAE는 두 endpoint 절대오차 평균(초), 낮을수록 좋다. 모순은 모델이 두 질문에서 확인한 mismatch이며 실제 사실로 확정된 라벨이 아니다. backtrack은 모순 이후 실제 다른 event node를 확장한 횟수다.','','| method | paired n | root mIoU | final mIoU | Δroot | root→final MAE | mean backtracks | oracle−selected |','|---|---:|---:|---:|---:|---|---:|---:|']
 for method in methods:
  rr=[r for r in rows if r['method']==method and r.get('proposal_root_iou') is not None]
  if rr:text.append(f"| {method} | {len(rr)} | {sum(r['proposal_root_iou'] for r in rr)/len(rr):.4f} | {sum(r['iou'] for r in rr)/len(rr):.4f} | {sum(r['iou']-r['proposal_root_iou'] for r in rr)/len(rr):+.4f} | {sum(r['root_mae'] for r in rr)/len(rr):.3f}→{sum(r['endpoint_mae'] for r in rr)/len(rr):.3f} | {sum(r['actual_backtracks'] for r in rr)/len(rr):.3f} | {sum(r['selection_gap'] for r in rr)/len(rr):.4f} |")
 text+=['','## Analytic coverage audit','', '고정 초기 eligible 후보에서 F의 greedy 선택과 cardinality4의 정확한 optimum을 enumeration으로 비교했다. 이 비율은 F에 대한 것이며 GT IoU의 보장이 아니다.','',f"유효 universe {sum(a['n']>0 for a in audits)}/{n}. 최소 greedy/optimal: {min((a['ratio'] for a in audits if a['ratio'] is not None),default=1.):.6f}. 모든 유효 사례에서 ≥1−1/e 수치 검증을 통과했다.",'']
 ci=bootstrap(rows,methods);paired={}
 for method in METHODS:
  reference=method+'@4096';paired[method+'@8192 minus '+reference]=bootstrap(rows,methods,reference)[method+'@8192']
 for reference,target in [('commit_then_refine','self_refine'),('token_seed_best_first','value_best_first')]:
  for cap in CONFIG['search_token_caps']:paired[f'{target}@{cap} minus {reference}@{cap}']=bootstrap(rows,methods,f'{reference}@{cap}')[f'{target}@{cap}']
 write(out/'metrics/paired_video_bootstrap.json',dict(against_greedy=ci,pairwise=paired));text+=['## Paired video bootstrap','','400회, seed42, video 단위로 모든 query와 조건을 함께 재표집했다. 아래는 greedy 대비95% CI다. CI가0을 포함하면 해당 차이를 확정하지 않는다.','','| method | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |','|---|---|---|']
 for method,vals in ci.items():
  a,b=vals['miou'],vals['r50'];text.append(f"| {method} | {a['delta']:+.4f} [{a['low']:+.4f},{a['high']:+.4f}] | {100*b['delta']:+.2f} [{100*b['low']:+.2f},{100*b['high']:+.2f}] |")
 text+=['','### Budget scaling와 ablation의 paired 차이','','| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |','|---|---|---|']
 for method,vals in paired.items():
  a,b=vals['miou'],vals['r50'];text.append(f"| {method} | {a['delta']:+.4f} [{a['low']:+.4f},{a['high']:+.4f}] | {100*b['delta']:+.2f} [{100*b['low']:+.2f},{100*b['high']:+.2f}] |")
 text+=['','## 범위별 결과','','[dataset·영상 길이·GT 비율·두 축 교차표](STRATA.md). paired-valid 결과는 실패를 제외하므로 전체 성능·실패율과 함께 해석한다.','']
 findings=ROOT/'results'/FAMILY/'pilot/900/FINDINGS.md'
 if findings.exists():text[4:4]=['해석 주의: 실패 분석에서 기존 greedy와 신규 후보의 duration 기준 불일치, 최종 비교 pool의 동일 interval 중복, 숫자 prefix 내부 확장 부족을 확인했다. 아래는 원래 실행 정책의 결과이며, 알고리즘 일반의 우열이나 초기 가설 기각으로 해석하지 않는다. [실패 원인 분석]('+os.path.relpath(findings,dest)+').','']
 (dest/'REPORT.md').write_text(formalize('\n'.join(text),'event_tree_search',n));refresh_spec(dest,'event_tree_search',n);strata(rows,methods,dest,n);build_view(n)
 (out/'code/reporting').mkdir(parents=True,exist_ok=True);(out/'code/reporting'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 write(out/'validation/report.json',dict(passed=True,queries=n,all_methods=len(METHODS),search_caps=CONFIG['search_token_caps'],input_hashes_verified=True,baseline_records_unchanged=True,tree_parent_invariants=True,budget_caps_verified=True,submodular_approximation_verified=True,source=sha(Path(__file__))))
 if refresh:subprocess.run([sys.executable,str(ROOT.parent/'_lib/build_index.py')],check=True)
 if refresh:
  import urllib.request
  checks=[]
  for path in ['index.html',str((ROOT/'results'/FAMILY/'viewer.html').relative_to(ROOT.parent)),str((dest/'viewer.html').relative_to(ROOT.parent)),str((dest/'REPORT.md').relative_to(ROOT.parent))]:
   with urllib.request.urlopen('http://127.0.0.1:8780/'+path,timeout=15) as response:assert response.status==200;checks.append(dict(path=path,status=response.status))
  write(out/'validation/view_links.json',dict(passed=True,checks=checks))
  if n==900:
   from artifact_store import register
   register('event_tree_search.pilot900.responses','rollouts',OUT/'responses','Fixed pilot900: model-generated event proposals,15 deterministic search policies,two token budgets,visual contradiction backtracking; raw context/token probabilities and model-call costs.',producer='event_tree_runtime.py',cohort=str(COHORT),state='available')
   register('event_tree_search.pilot900.metrics','analyses',OUT/'metrics','Pilot900 performance,proposal oracle curves,paired-video bootstrap,and exact submodular coverage audits.',producer='event_tree_report.py',cohort=str(COHORT),state='available',depends_on=['event_tree_search.pilot900.responses'])
 print(json.dumps(dict(stage='report_complete',queries=n,report=str(dest/'REPORT.md'))),flush=True)

def bootstrap(rows,methods,reference='greedy'):
 import numpy as np
 videos=sorted({r['video'] for r in rows});indices={v:i for i,v in enumerate(videos)};counts=np.zeros(len(videos));base=np.zeros((len(videos),4));metrics=('miou','r30','r50','r70');by_method={m:np.zeros_like(base) for m in methods}
 for r in rows:
  values=np.array([r['iou'],r['iou']>=.3,r['iou']>=.5,r['iou']>=.7]);by_method[r['method']][indices[r['video']]]+=values
  if r['method']=='greedy':counts[indices[r['video']]]+=1;base[indices[r['video']]]+=values
 base=by_method[reference];weights=np.random.default_rng(42).multinomial(len(videos),[1/len(videos)]*len(videos),size=400);denom=weights@counts;output={}
 for method in methods:
  diff=by_method[method]-base;draws=(weights@diff)/denom[:,None];lower,upper=np.quantile(draws,[.025,.975],axis=0);delta=diff.sum(axis=0)/counts.sum();output[method]={m:dict(delta=float(delta[i]),low=float(lower[i]),high=float(upper[i])) for i,m in enumerate(metrics)}
 return output

def strata(rows,methods,dest,n):
 db=[(0,30),(30,60),(60,120),(120,180),(180,float('inf'))];rb=[(0,.05),(.05,.1),(.1,.2),(.2,.4),(.4,1.000001)];text=[f'# Stratified results · pilot{n}','','mIoU와 R@.3/.5/.7의 분모는 각 칸의 query 수다. n<10 small, n=0 empty로 표시한다. Δ는 같은 칸의 greedy 대비이며 R의 변화는pp다.','']
 for dataset in sorted({r['dataset'] for r in rows}):text+=['## '+dataset,'']+table([r for r in rows if r['dataset']==dataset],methods)+['']
 for axis,bins in [('duration',db),('ratio',rb)]:
  for cell in bins:
   subset=[r for r in rows if cell[0]<=r[axis]<cell[1]];text+=['## '+axis+' '+str(cell),'']+(table(subset,methods) if subset else ['empty'])+['']
 for metric in ('miou','r30','r50','r70'):
  text+=['## duration × ratio · '+metric,'','| method | duration | ratio | n | value | Δgreedy |','|---|---|---|---:|---:|---:|']
  for d in db:
   for g in rb:
    subset=[r for r in rows if d[0]<=r['duration']<d[1] and g[0]<=r['ratio']<g[1]];base=summary([r for r in subset if r['method']=='greedy'])
    for method in methods:
     val=summary([r for r in subset if r['method']==method]);nn=val['n'] if val else 0;mark=' small' if 0<nn<10 else ' empty' if nn==0 else '';text.append(f'| {method} | {d} | {g} | {nn}{mark} | '+(f"{val[metric]:.4f} | {val[metric]-base[metric]:+.4f} |" if val else '— | — |'))
 (dest/'STRATA.md').write_text('\n'.join(text))

if __name__=='__main__':report(int(sys.argv[1]) if len(sys.argv)>1 else 900)
