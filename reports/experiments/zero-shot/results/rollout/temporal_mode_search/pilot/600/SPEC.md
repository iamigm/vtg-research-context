# Temporal mode search · pilot600 · 실행 명세

기존 event_tree_search와 구분한 새 실험이다. 고정 pilot600은 세 데이터셋별200 query이며 모델·조건·seed 반복 수와 구분한다. pilot300은 같은 원자료의 고정 prefix를 참조한다. 기존 원자료와 보고서는 덮어쓰지 않는다.

## 입력과 생성

Qwen3-VL-8B-Instruct,2 FPS,spatial target100,352 pixels,video→query,기존 deterministic 입력 tensor hash·frame IDs·PTS·H×W를 유지한다. 원래 TimeLens prompt에서 실제 token prefix를 탐색한다. 0–9 enumeration·새 occurrence proposal prompt·review-conditioned timestamp regeneration은 사용하지 않는다. temperature1,penalty 없음,EOS 포함 최대512 new tokens다. 탐색에는 GT를 전달하지 않는다.

분기 하한은 full-vocabulary raw softmax p≥max(.009,.01×pmax)다. 전체 분포 누적90% nucleus와 하한의 교집합이 core다. 하한을 만족하는 nucleus 밖 token은 지원된 tail이며 나머지는 조사하지 않는다. 목표 질량을 채우기 위해 하한을 낮추지 않는다. 모든 output token을 그대로 유지하며 첫 숫자 이후 정수·소수점·separator·end도 실제 prefix별로 조사한다. token을 문자 단위로 쪼개지 않는다.

Main continuation은 매 step의 core에서 q=p/Σcore p로 sampling한다. Tail은 main-only prefix의 첫 noncore edge를 진입점으로 삼고 이후 지원 집합 전체에서 q=p/Σsupported p로 sampling한다. 한 번 tail이면 끝까지 tail이다. Sampling은 with replacement이며 duplicate draw를 통계에서 지우지 않는다. KV/logits 및 이미 계산한 응답은 재사용한다.

## Clipping과 mode

부호를 보존한 parsing→각 endpoint를[0,D]로 clipping→finite·start<end 검사→exact interval 통합→mode 구성 순서다. D는 재사용 영상 metadata의 duration이다. endpoint를 정렬하지 않고 길이0·역전·형식 실패·EOS 미도달은 실패로 둔다. raw interval·원문·원래 path probability는 보존한다. GT는 바꾸지 않는다.

Mode의 모든 interval 쌍은 IoU≥.5 및 max endpoint 차이≤max(1초,.1×짧은 interval 길이)를 만족한다. 온라인 등록 순서와 동률을 고정하고 기존 mode를 tail로 병합하지 않는다. exact interval 내부 여러 token sequence의 EOS 포함 raw path mass는 sequence별 한 번만 합산한다. 이는 관측 경로의 질량 하한이며 전체 사건 posterior가 아니다. 중앙 후보는 관측 raw path mass 가중 IoU medoid,두 번째는 중앙에서 가장 먼 endpoint 대안이다. 둘 다 실제 main interval이다.

## 탐색과 확률 추정

조건은 probability·token_entropy·temporal_mode,seed42/43/44다. Main root probe16회,주요 분기 suffix 최소8회 후 적응 배정한다. Tail entry는 최소32회 후 추가 조사한다. 방문하지 않은 prefix나 미달 probe는 미조사로 남긴다.

공통 확률 우선순위는 log P(prefix+edge)−log 평균 logical cost−.5 log(1+probe 수)다. entropy 조건은 여기에 log(max(Htoken,1e-6))를 더한다. mode 조건은 log(min(1,새 mode weighted yield/n+sqrt(log(2+draw 수)/(2n))))을 더한다. 새 mode는 해당 조건의 이전 draw에 없던 mode다. 초기 최소 probe를 만족하지 못한 edge는 prefix probability 우선으로 먼저 조사한다.

MI는 exact prefix에서 token×valid-mode joint mass를 추정해 정규화한 mutual information(bits)이다. suffix probe의 importance weight는 Πp/q이며 task별 평균에 원래 edge probability를 곱한다. valid main continuation에 조건부이며 raw full-vocabulary entropy와 구분한다. 표본 부족·미탐색 edge·하한 밖 질량을 함께 기록한다. MI는 진단이며 정답 확률이나 본 실행의 직접 selector가 아니다. S의 novelty estimate와 uncertainty bonus가 mode 조건의 추가 예산을 정한다.

## 모델 판단과 tail 격리

Main mode·후보를 먼저 확정하고 모델 판단까지 끝낸 다음 tail 진단을 수행한다. 후보 hash를 전후 대조한다. Tail에서 발견한 결과는 실제 후보·judge prompt·main 탐색 feedback에 전달하지 않는다.

관측 main path mass 상위 최대6 mode에서 최대2개 boundary 후보를 평가한다. Event match·onset fit·offset fit의 supported/contradicted/uncertain(A/B/C) 확률을 기록하되 곱하거나 mode 전체를 hard-prune하지 않는다. 같은 mode의 두 interval을 양쪽 presentation order로 비교해 대표를 정하고 mode 대표 전원은 round robin으로 비교한다. 비교는 correct action/actor/object/occurrence를 먼저,그다음 onset/offset을 판단한다. token probability·rank·greedy 출처를 judge에게 보여주지 않는다. A/B/C는 단일 token인지 확인한다.

Pair preference=.5×{forward P(A)+.5 forward P(C)+reverse P(B)+.5 reverse P(C)}다. raw option mass<.01은 불확실 tie다. 최종 평균 preference 동률은 observed main mass·작은 endpoint 순이다. 최종 timestamp는 기존 clipped interval이며 재생성하지 않는다. 판단 예산 초과는 첫 mode 중앙 후보 fallback으로 별도 기록한다. 유효 main 후보0개는 실패이며 tail로 대체하지 않는다.

## 예산과 완료

기존 제안보다 분기 예산을 두 배로 올렸다. query·조건·seed마다 main65,536,tail131,072,judge32,768 decoder input tokens다. Prefix replay+생성+새 judge prompt를 logical 독립 실행 비용으로 계수하고 실제 cache 실행 tokens·wall time·video prefill은 별도 기록한다. 생성 도중 cap은 후보로 commit하지 않으며 이미 사용한 비용은 남긴다. Tail 예산 미달 조사 항목은 완료로 표현하지 않는다.

600 query×3조건×3seed=최대5,400개의 최종 선택이다. 이것을 pilot5400으로 부르지 않는다. Main·tail·judge 비용을 분리하고 합계도 보고한다. detached supervisor·20초 heartbeat·조건별 atomic checkpoint·최대3회 자동 재개 후 완료 보고서·검증·registry·index 갱신을 연결한다. 다른 GPU 작업을 중지하지 않는다.

## 평가

IoU=시간 교집합/합집합,mIoU=query별 IoU 평균,R@τ=IoU≥τ인 비율(τ=.3/.5/.7)이다. 높을수록 좋고 실패는0점이다. 같은 clipping을 greedy에도 적용한다. 조건별3seed는 query 안에서 평균하고 query 수와 반복 응답 수를 구분한다. 영상 길이·GT/영상 길이 비율·교차표는 사후 평가에만 사용한다. GT는 후보 생성·mode 구성·선택·중단에 사용하지 않는다.

Main candidate oracle·shortlist oracle·실제 선택·main+tail diagnostic oracle을 분리한다. Tail 포함 oracle은 운영 가능한 성능이 아니다. Mode 수는 관측된 시간 가설 수이며 실제 모든 사건에 대한 coverage 비율을 주장하지 않는다. video 단위 paired bootstrap(seed42,400회)로 greedy 대비 ΔmIoU의95% CI를 보고한다. 같은 query의3seed·조건을 독립 query로 합산하지 않는다.
