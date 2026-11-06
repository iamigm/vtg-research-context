# Temporal mode search · capped rollouts · pilot600

## 범위와 변경

고정 pilot600,세 데이터셋별200 query,seed42·43,probability/token_entropy/temporal_mode 세 방법이다. 이전 token-budget 실행은 중단하고 원자료를 보존했다. 생성 정책이 바뀐 새 조건이므로 이전 결과와 합산하지 않는다. frame IDs·PTS·H×W·tensor hash,모델,0.9% absolute floor,0.01 relative coefficient,90% main nucleus,[0,D] clipping은 유지한다. 0–9 enumeration과 새 interval proposal prompt는 사용하지 않는다.

## Rollout 예산

rollout을 시작해 한 응답을 완성하거나 유효성 실패로 끝낸 시도 한 번을1회로 센다. 중복 sequence·동일 interval도 실제 시도이므로 포함한다. unique sequence·unique interval·temporal mode 수는 별도 기록한다.

| 총 rollout 상한 | Main | Tail |
|---:|---:|---:|
|32|16|16|
|64|32|32|
|128|64|64|
|256|128|128|

각 방법·seed는 main128+tail128을 최대 한 번 실행한다. 작은 예산은 같은 draw stream의 누적 checkpoint다. 3방법×2seed를 합해 query당 최대1,536 rollout 시도다. 원래 약65,500회 규모보다 약43배 작은 횟수 상한이며 시간 단축 배율을 보장하는 값은 아니다. Main65,536/tail131,072 decoder input tokens는 이차 안전 상한으로만 유지한다. 응답 최대512 tokens다. 지원된 branch가 없거나 token guard를 만나면 실제 시도 수가 상한보다 작을 수 있으며 이를 표시한다.

## 초기화와 우선순위

Main root sampling4회,공통 coverage를 합해 초기화는 최대8회다. 새 prefix마다8회를 요구하지 않는다. 실제 main 선택지가2개 이상인 prefix만 main task로 등록한다. 이후 probability는 raw prefix mass/cost와 visit penalty,token_entropy는 추가 raw entropy,temporal_mode는 추가 weighted novelty와 uncertainty bonus를 사용한다. 미조사 edge의 periodic coverage는8회당 최대1회다. Tail도 초기 coverage는최대4회이며 모든 tail entry에32회를 배정하지 않는다.

Checkpoint마다 root/initialization/coverage/adaptive 선택 횟수와 미조사 edge 수를 기록한다. 방법별 adaptive 선택이 예산 내에 실제 발생하는지 테스트한다. 선택 점수가 달라도 모델이 같은 결과를 낼 수 있으며 실제 결과를 인위적으로 다르게 만들지 않는다.

## Tail 격리와 예산 간 무결성

각 checkpoint에서 main 후보를 먼저 고정·판단한 뒤 같은 횟수의 tail을 조사한다. Tail entry는 그 시점까지 main이 방문한 prefix에서만 얻는다. 이후 큰 예산에서 발견할 prefix를 작은 예산의 tail에 미리 사용하지 않는다. Main RNG와 선택 통계는 tail 호출 수 및 중간 judge 호출과 분리한다. 과거 checkpoint는 deep copy로 고정하며 이후 확장으로 바뀌지 않는다. Tail 후보는 모델 판단이나 main 선택에 전달하지 않는다.

## Mode와 모델 판단

부호를 보존해 parsing한 후 endpoint를[0,D]로 clipping하고 start<end를 요구한다. exact interval 통합 뒤 모든 쌍의 IoU≥.5와 endpoint 차이≤max(1초,.1×짧은 interval 길이)를 만족하는 mode로 묶는다. 원래 token IDs·raw interval·path probability는 유지한다. Mode는 semantic event GT가 아닌 시간 가설이다.

관측 main mass 상위6 mode에서 실제 interval medoid와 boundary 대안을 고른다. 같은 frozen VLM이 event match/onset/offset과 짧은 근거를 판단하고,mode 내부 비교 후 mode 대표 전원을 양쪽 A/B 순서로 round robin 비교한다. Tail·token rank·greedy 출처는 제시하지 않는다. 최종 답은 기존 clipped interval이며 새 timestamp를 생성하지 않는다. 각 checkpoint의 judge cap은32,768 tokens이며 동일 prompt는 정확한 context cache로 재사용한다.

## 평가와 비용

IoU=교집합/합집합,mIoU=query별 IoU 평균,R@τ=IoU≥τ 비율(τ=.3/.5/.7)이다. 높을수록 좋다. 실패는0점이며 두 seed를 query 안에서 평균한다. 다른 예산의 값을 한 평균으로 합치지 않는다. 진행 중 표는 해당 방법·예산에서 두 seed가 모두 완료된 query만 집계한다.600 query×3방법×2seed=3,600 route,각4checkpoint=14,400개 예산별 선택이며 독립14,400 query가 아니다.

Main/pool/diagnostic oracle을 구분하고 GT는 모든 선택 이후 평가에만 사용한다. MI는 p/q-weighted 조건부 mode 정보량(bits) 진단이다. Native model mass와 관측 경로 질량 하한을 구분한다. Main/tail token 비용은 해당 checkpoint까지 누적이며 judge는 그 checkpoint만 독립 배치했을 때 필요한 logical 비용이다. 이전 작은 예산의 judge 비용을 큰 예산의 독립 비용에 더하지 않는다. 실제 forward의 CUDA stream 시간은 raw measured_actual_cost와 checkpoint.cost,기존 token·forward·wall 비용은 actual_cost에 기록한다.

## 운영

영상 hash modulo3으로 GPU worker3개에 겹치지 않게 배정한다. worker별 Torch threads2와 CPU input prefetch1개를 사용한다. 고정 code snapshot·조건 및 예산별 atomic checkpoint·20초 heartbeat·최대3회 재개를 사용한다. 완료 후 MD·HTML·raw validation·viewer validation·registry·통합 index를 자동 갱신한다. 동일 계열의 capped_rollouts 조건이며 이전 결과는 별도 선택 항목으로 남긴다.

## 연산 비용과 성능 대비 비용

비용의 분모는 query·method·seed·예산 한 경우다. seed42·43을 query 안에서 먼저 평균한다. 진행 중 비교는 3방법×4예산×2seed가 모두 완료된 같은 query만 사용한다. 다음 두 비용을 구분한다.

**Independent trace work**는 저장된 token ID·분기 prefix·judge prompt를 따라 fresh video KV와 빈 distribution/prompt cache에서 greedy → main K회 → 해당 K의 judge → tail K회만 실행하는 경우의 연산 횟수를 재구성한다. 다른 method·seed·작은 예산 judge의 cache를 빌려 쓰지 않는다. 영상 prefill은 한 번 포함한다. 이는 시간 실측이나 FLOPs 추정이 아니며 실제 native forward 계측과 대조한다. Tail이 중간 main에 영향을 주지 않는 고정 draw stream을 사용한다.

- `decoder inputs`: native decoder forward에 실제 투입하는 token position 수의 합. 재생한 prefix, judge text prefill·근거 생성, 영상 prefill 뒤의 multimodal decoder positions를 포함한다. sampling 횟수나 출력 token 수와 다르다.
- `forwards`: native model forward 호출 수. 하나의 긴 judge text prefill도1회이므로 token 수와 함께 해석한다.
- `attention pairs`: 각 forward의 causal query-key position 쌍 `q(k−q)+q(q+1)/2`의 합. q는 새 input 길이,k는 전체 KV 길이다. layer/head 곱을 생략한 context 길이 workload proxy이며 정확한 FLOPs가 아니다. vision encoder의 별도 연산은 포함하지 않는다.
- `operational`: video prefill+greedy+main+judge. `research total`: operational+진단용 tail. Tail을 후보로 쓰지 않아도 이번 실험의 총비용에서 제외하지 않는다.
- `greedy ×`: 같은 query의 video prefill+greedy 대비 decoder inputs 또는 forward 호출 수 배율. 원래 입력의 hash와 greedy ID가 동일하다. 입력 prefill과 추가 text 비용도 분리한다.

**Observed execution**은 지금 사용하는 worker3개·공유 cache의 nested sweep에서 각 main/tail/judge 구간의 wall seconds,native forward CUDA stream seconds,forward 수,decoder inputs,process별 peak allocated/reserved memory를 기록한다. CUDA events를 각 forward 전후에 배치하며 GPU 동시 실행·queue 대기의 영향을 받을 수 있다. CUDA stream seconds는 GPU 전체 사용시간이나 정확한 kernel active time이 아니다. wall−forward wall은 Python/cache/mode grouping 등 host-side 잔여시간의 근사다. peak memory는 worker별 model/KV를 포함하고 다른 worker의 점유량과 합산한 장치 전체 peak가 아니다.

Observed main/tail은 해당 K까지의 누적,judge는 해당 K의 구간이다. `sweep`은 이전 모든 작은 예산 judge를 포함한 phase 합계이고,`sweep_elapsed_seconds`는 이전 checkpoint 저장과 phase 밖 Python 처리까지 포함한 경과시간(현재 checkpoint 저장 직전)이다. 독립 K 실행시간으로 부르지 않는다. method 실행 순서·이미 저장된 cache·재개에 따라 관측 비용이 달라지므로 method 공정 비교는 independent trace work를 사용한다. 재개 실행의 재탐색·실패 비용은 누적 실행 로그와 raw actual_cost에 보존한다. 계측 도입 이전 checkpoint의 실측은 결측으로 남기며0초로 처리하지 않는다.

효율 지표는 greedy 대비 `ΔmIoU(pp)=100×mean(IoU−IoU_greedy)`,추가1,000 decoder inputs당 향상 `ΔmIoU(pp)/(mean(extra decoder inputs)/1000)`다. 음수는 비용을 써도 성능이 악화됐다는 뜻이다. `marginal Δ`와 `추가 비용`은 같은 query에서 직전 절반 예산 대비 값이다. 같은 query 집합에서 더 낮은 비용·같거나 높은 mIoU의 다른 조건이 있으면 dominated로 표시한다. 평균 개선만으로 연구적 기여를 단정하지 않으며 비용 배율,실측 시간,seed·dataset별 결과와 불확실성을 함께 본다.
