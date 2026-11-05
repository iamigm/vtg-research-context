# Event proposal and tree search · pilot900

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

## 문헌 조사와 적용 범위

- [Tree of Thoughts](https://arxiv.org/abs/2305.10601): BFS/DFS와 중간 상태 평가·backtracking. 여기서는 temporal event/action 상태와 deterministic operators로 적용한다.
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424): beam의 유사 후보 집중을 diversity objective로 완화. 본 실험은 token Hamming penalty가 아닌 시간 구간의 semantic-action beam adaptation이다.
- [Self-Evaluation Guided Beam Search](https://arxiv.org/abs/2305.00633): 중간 단계 self-evaluation으로 탐색을 유도. 논문의 stochastic beam을 재현하지 않으며 frozen VLM critic과 deterministic beam을 사용한다.
- [Language Agent Tree Search](https://arxiv.org/abs/2310.04406): 선택·확장·value backup·reflection. 외부 환경의 검증된 reward 대신 VLM의 noisy visual judgement를 사용한다.
- [Self-Refine](https://arxiv.org/abs/2303.17651): 추가 학습 없이 같은 모델의 feedback을 다음 generation에 반영한다.
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): proposal distribution 수정과 verifier-guided 탐색을 구분하고 전체 비용을 측정한다. 학습한 PRM을 도입하는 재현 실험은 아니다.
- [Stochastic Beams / Gumbel-Top-k](https://proceedings.mlr.press/v97/kool19a.html): sequence sampling without replacement는 독립 sampling의 중복 문제를 직접 다룬다. 본 실행은 이전 사용자 지시에 따라 random sampling을 사용하지 않으며 unique branch enumeration을 유지한다. Gumbel sampling을 실행했다고 부르지 않는다.
- [Nucleus Sampling](https://arxiv.org/abs/1904.09751): 원래는 token distribution의 누적확률 support를 정한 뒤 sampling한다. 여기서는 공통 사건 후보의 canonical answer probability를 정규화한 finite-list support를 deterministic하게 탐색한다. 원래 nucleus sampling의 재현이 아니다.
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666): surprisal과 entropy의 차이로 support를 구성한다. 본 실행은 finite event-list typicality adaptation이며 token-level locally typical sampling과 구분한다.
- [Sequential Monte Carlo Steering](https://arxiv.org/abs/2306.03081): 여러 particle과 weight·resampling으로 generation을 제어한다. 본 실행의 reward-balanced allocation은 결정적 resource allocation이며 posterior SMC나 unbiased estimator를 주장하지 않는다.
- [ReSCALE / Sequential Halving](https://arxiv.org/abs/2603.21162): 최근 연구가 PUCT scaling 악화와 sequential halving을 비교했다. 본 실행은 Gumbel noise·학습 value head 없이 deterministic lookahead halving을 비교하며 해당 논문 성능을 일반화하지 않는다.
- [Contrastive Decoding](https://arxiv.org/abs/2210.15097): expert–amateur log-likelihood 대비는 또 다른 decoding objective다. 숫자 token 반복을 억제하는 목적과 localization의 목적이 다르며, 이번 비교에서는 backbone 효과를 고정하기 위해 별도 amateur model을 추가하지 않는다.
- [Submodular summarization / facility location](https://aclanthology.org/P11-1052/): 집합의 coverage와 quality를 함께 최적화하는 수학적 접근을 사건 후보 선택에 적용한다.
- [Best Algorithms for Approximating the Maximum of a Submodular Set Function](https://pubsonline.informs.org/doi/10.1287/moor.3.3.177): 고정된 normalized monotone submodular objective와 cardinality constraint에서 greedy 근사 보장. IoU나 전체 adaptive search의 최적성을 보장하지 않는다.
