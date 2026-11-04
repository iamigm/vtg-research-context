# Branch feedback · pilot600 후속 실험 설계

상태: 설계. 이 문서는 신규 GPU 실행이나 결과를 의미하지 않는다. 기존 rollout reconsideration 실행·응답은 변경하지 않는다.

## 질문과 주 비교

재검토 정보를 받은 모델이 동일한 생성 prefix에서 다른 token을 선택하고, 그 선택에서 새 conditional rollout을 수행하여 시간 구간을 개선하는가?

기존 실험은 완성된 rollout 후보를 평가한 뒤 전체 응답을 다시 생성한다. 후속 실험은 생성 도중의 특정 분기 직전으로 돌아간다. 분기 이후의 token은 새로 생성하며, 기존 완성 응답의 suffix를 복사하지 않는다.

1차 실험은 분기에서의 단일 feedback을 검증한다. 반복적으로 탐색 순서를 바꾸는 2차 실험은 이 결과와 실제 계산 비용을 확인한 뒤 확장한다. 두 실험의 결과를 혼합하지 않는다.

## 입력·cohort·분기 선정

- 기존 고정 pilot600을 유지한다. Charades·ActivityNet·QVHighlights 각200 query다. 모델, video_query input order, frame IDs/PTS, 실제 H×W, input tensor hash를 유지한다.
- 기존 deterministic rollout의 첫8개 경로만 seed 후보로 사용한다. 뒤의9–32번 경로와 GT는 분기 선정·재검토·생성에 사용하지 않는다.
- 경로의 실제 token IDs와 prefix를 사용한다. 초 단위 출력을 분:초로 바꾸거나 자릿수를 다시 포맷하지 않는다. 백의 자리·십의 자리·일의 자리·소수점 이하 자리와 start/end 위치를 기록한다. 예외적인 분:초 출력은 기존 parser로 해석하되 원문·tokenization은 보존한다.
- seed 안의 parent–child 경로 쌍에서 두 종류를 각각 최대1개 고른다. 여기서 parent는 대안 token으로 분기하기 전의 원래 경로이고, child는 대안 token을 선택한 기존 경로다.
  - 사건 후보: 두 구간의 IoU<0.1이고, 중심 시각 차이가 max(2초, 영상 길이의5%)보다 큰 쌍.
  - 경계 후보: 두 구간의 IoU≥0.5이고, 최대 endpoint 변화가 0보다 크고 max(2초, parent 구간 길이의10%) 이하인 쌍.
- 각 종류에서 원래 대안 token probability가 가장 큰 쌍을 고른다. 동률이면 이른 token 위치, 작은 candidate ID 순이다. 이 규칙은 GT 없이 고정한다.
- 같은 query에 두 종류가 있으면 두 분기를 모두 진단하되 독립 query로 세지 않는다. 전체600의 주 성능에는 두 분기 중 대안 token probability가 더 큰 분기를 사용한다. 동률은 token 위치·candidate ID·event/boundary 이름순으로 정한다. GT를 보고 둘 중 좋은 결과를 선택하지 않는다.
- parent가 seed의 greedy와 다를 수 있다. 따라서 분기 실험의 직접 baseline은 해당 parent의 원래 conditional rollout이며, 원래 전체 greedy와의 성능 차이는 별도로 보고한다.

2026-11-04 CPU 확인 결과: 첫8개 경로만으로 사건 후보187 query, 경계 후보595 query가 있고, 두 종류가 함께 있는 query는182개다. 합집합은600 query, 진단 분기는782개다. 이는 GT를 사용하지 않은 대상 수이며 성능 결과가 아니다.

| dataset | 사건 후보 query | 경계 후보 query |
|---|---:|---:|
| Charades | 40 | 198 |
| ActivityNet | 84 | 197 |
| QVHighlights | 63 | 200 |

사건 후보의 분기 위치는 start154/end33, 경계 후보는 start342/end253이다. 시간적으로 다른 구간이라는 정의는 semantic event identity의 gold label이 아니므로 결과에 temporal occurrence proxy임을 명시한다. 두 축은 query 수준에서 겹친다.

Cohort SHA-256: `d7fb4b7cc1c28842be0fa81e9b277c688ccfddf862371a3253a1000db25e4752`.

## 재검토에서 하는 일

세 접근 모두 동일한 분기 prefix, 기존 token과 대안 token, seed에서 얻은 후보 구간, 같은 영상 입력을 받는다. 분기의 두 경로는 반드시 포함하고, 나머지는 unique interval farthest-first로 고른다. 토큰과 interval의 실제 대응을 유지한다.

| 접근 | 추가 판단 | 1차 실험의 상한 |
|---|---|---:|
| agent | 사건 비교 또는 경계 확인을 선택하고 observation을 받아 다음 작업을 결정 | 3 action–observation rounds, 최대6 evaluations |
| verifier | 후보마다 사건 적합성과 경계 적합성을 독립 평가 | 3개 후보×2질문, 최대6 evaluations |
| TTS | 후보 간 pairwise 비교를 양쪽 presentation order로 수행 | 4개 후보 tournament, 최대6 evaluations |

agent는 정해진 분기에서 무엇을 확인할지 선택한다. 1차 실험에서는 분기 위치까지 agent가 선택하게 하지 않는다. 그래야 위치 선택 효과와 분기 판단 수정 효과를 분리할 수 있다.

각 접근이 만든 재검토 기록은 아래의 자유 선택 조건과 기존 token 고정 조건에 똑같이 사용한다. 두 조건에서 review를 다시 수행하지 않는다. 후보 부족·agent early stop으로 실제 evaluation 수가 줄면 실제 값을 기록한다. 평가 횟수의 상한이 같아도 FLOPs가 같다고 주장하지 않는다.

## 동일 분기에서 비교할 조건

`p`는 분기 직전까지의 정확한 token prefix, `a`는 원래 token, `R`은 재검토 기록이다.

| 조건 | 분기에서 하는 일 | 목적 |
|---|---|---|
| 원래 conditional rollout | 원래 context에서 `p`를 재현하고 greedy로 완성 | 원래 parent와의 재현성 및 직접 baseline |
| 후보만 제공 | 후보 목록을 주고 `p`부터 greedy로 새로 완성 | 후보 노출·추가 prompt 자체의 효과 |
| 재검토 후 자유 선택 | 후보와 `R`을 주고 `p`에서 다음 token을 greedy로 선택한 뒤 EOS까지 생성 | 모델 자신의 분기 선택과 새 경로 |
| 재검토 후 기존 token 고정 | 위와 동일한 context에서 `p+a`까지 고정한 뒤 EOS까지 생성 | 동일한 재검토 아래에서 원래 분기를 유지하는 ablation |

재검토 후의 두 조건은 agent/verifier/TTS 각각 수행한다. 자유 선택에서는 특정 대안 token을 강제하지 않는다. 전체 어휘에 대한 모델의 argmax를 사용하며, 원래 후보에 없던 token도 허용한다. 기존 token 고정은 분기 변경의 기여를 진단하기 위한 별도 조건이며 실제 탐색 방법의 성능과 섞지 않는다.

원래 conditional rollout의 재현성 검증을 제외하면 분기당 최대7개의 신규 완성 응답이다: 후보만 제공1개 + 3접근×2조건. 782개 진단 분기에서는 최대5,474개 응답이다. 실제 수는 중복 경로 캐시·대상 부족에 따라 줄 수 있다. 이는600 query에 대한 반복 조건 수이며 pilot5474라고 부르지 않는다.

각 신규 conditional rollout은 최대64개의 새 token을 생성한다. EOS 미도달·파싱 실패·유효하지 않은 구간은 실패로 남긴다. 원래 parent의 남은 suffix가64 token 이내에서 끝나는지는 실행 전 CPU 검증하고, 초과 사례는 제한을 임의 변경하지 않고 명시한다.

## context 구성과 start/end conditioning

원래 완성 응답을 assistant history에 자동으로 붙이는 현재 helper를 그대로 사용하지 않는다. 원래 video/query 입력을 유지한 채 후보와 재검토를 완료된 별도 turn에 넣고, 새 assistant 응답의 시작에 `p`를 정확히 재현한다. 원래 완성 응답은 필요한 경우 다른 후보와 동일한 목록 항목으로만 제시한다. 조건별 message scaffold를 맞추고 전체 context IDs와 hash를 보존한다.

분기 직전까지의 generated prefix는 원래와 같지만 review가 있으면 전체 context는 다르다. 이는 frozen model의 context-conditioned distribution 변화를 측정하는 실험이며, 모델 가중치를 수정하는 실험이 아니다.

- start 분기: start의 남은 token, 구분자, end 전체를 새로 생성한다. end는 새 start prefix에 조건화된다. 원래 end token이나 그 확률을 재사용하지 않는다.
- end 분기: 이미 생성된 start prefix를 유지하고 end의 남은 부분을 생성한다. 이 조건에서 start를 수정했다고 주장하지 않는다.
- 변경할 수 없는 앞쪽 prefix 때문에 오답이 유지될 수 있다. start/end 및 자릿수별로 결과를 따로 보여준다.
- 같은 시각이라도 token prefix가 다르면 별도 경로로 유지한다. 초 단위 통합은 평가·표시에만 사용한다.

재검토 후 자유 선택 경로와 기존 token 고정 경로가 같은 token sequence가 되면 모델 평가를 재사용하되, 두 조건의 정의와 원래 비용을 기록한다. context가 다른 경로 사이에는 KV·logits를 공유하지 않는다.

## 1차 실험의 지표와 판정

정량 지표의 분모는 query, 진단 분기, 적용 가능 subset 중 무엇인지 모두 표시한다. 방향을 반대로 해석하지 않는다.

1. **실제 분기 변경률**: 자유 선택에서 첫 token이 `a`와 달라진 분기 수 / 해당 조건의 전체 진단 분기 수. nonnumeric token·EOS로 바뀐 경우도 변경으로 세고 출력 실패율을 함께 보고한다. token만 바뀌고 같은 시각을 표현한 경우는 구간 변경과 구분한다.
2. **성능**: 각 query의 연속 시간 IoU=교집합 길이/합집합 길이를 계산한다. mIoU는 전체 query의 IoU 평균, R@t는 IoU≥t인 query 수 / 전체 query 수로 t=.3/.5/.7을 보고한다. 높을수록 좋다. 파싱 실패·무효 구간·EOS 미도달은0점이다. clipping·rounding·기존 답 자동 fallback은 하지 않는다. 원래 parent 및 원래 greedy 대비 ΔmIoU와 R의 변화(pp)를 함께 보고한다.
3. **분기 변경의 기여**: 동일 review에서 자유 선택 IoU − 기존 token 고정 IoU. 첫 token이 달라진 subset과 전체 모두 보고한다. 단일 greedy completion의 비교이므로 분기 아래의 모든 suffix에 대한 평균 인과효과로 일반화하지 않는다.
4. **재검토 자체의 기여**: 재검토 후 자유 선택 IoU − 후보만 제공 조건의 IoU. 단순 후보 노출 효과와 구분한다.
5. **token probability**: 전체 어휘에 대해 원래·대안·새로 선택된 token의 raw probability와 rank를 전후로 기록한다. `P(a|video,query,p)`와 `P(a|video,query,candidates,R,p)`는 다른 context의 값임을 명시한다. 후보만 제공 context의 값도 함께 기록한다. 확률 증가 자체는 성능 향상으로 판정하지 않는다.
6. **후속 conditional distribution**: 동일 review에서 `p+a`와 `p+b` 이후 token distribution을 저장한다. 여기서 `b`는 원래 대안 token이다. 자유 선택이 제3의 token `c`를 고르면 `p+c`도 저장한다. 부모와 각 child의 raw probability 및 경로별 누적곱을 보여준다.
7. **path probability**: prefix와 EOS를 포함한 전체 응답에 대해 `P(y|x,R)=∏t P(y_t|x,R,y_<t)`를 계산한다. prefix 이후 suffix만의 conditional path probability도 별도로 기록한다. 계산·저장은 log probability 합으로 하며, 해당 context와 suffix 길이를 표시한다. 다른 review context의 경로들을 하나의 전역 확률 순위로 섞지 않는다.
8. **새 구간 발견**: 새 token 경로 여부와 seed8에 없던 새 구간 여부를 분리한다. oracle은 해당 조건의 seed8과 실제 생성한 신규 경로에서 GT IoU 최대값이다. 운영 가능한 자유 선택 후보와 기존 token 고정 ablation 후보의 oracle을 섞지 않는다. Oracle 증가만으로 실제 선택 성능이 개선됐다고 주장하지 않는다.
9. **사건과 경계**: 사건 후보187개와 경계 후보595개에서 ΔIoU·분기 변경률·실패율을 따로 보고한다. 사건 후보에서는 원래 parent IoU<.1 → 새 IoU≥.5인 수 / 원래 parent IoU<.1인 유효 사례 수를 보고한다. 경계 후보에서는 endpoint MAE=(|start−GTstart|+|end−GTend|)/2와 그 변화(초, 낮을수록 좋음)를 paired-valid subset에서 계산하고 분모를 표시한다. 두 subset은 의미적 사건 gold가 아닌 proxy다.
10. **손상과 비용**: parent IoU≥.5 → 새 IoU<.5인 수 / parent IoU≥.5인 수를 보고한다. model evaluations, 실제 decoder input tokens, generated tokens, prefix replay, forward calls, synchronized model time을 기록한다. 확률 진단 비용은 의사결정 비용과 분리하고 합계도 제시한다. seed 생성 비용도 배포 가정에서는 포함하고 캐시 재사용 비용과 구분한다.

전체600의 주 성능은 사전 고정된 query당1개 분기에 대한 결과다. 사건·경계 진단 분기782개를600과 같은 모수로 표현하거나 두 종류 중 더 높은 IoU를 골라 주 성능으로 보고하지 않는다. dataset별, 영상 길이별, GT/영상 길이 비율별, 두 축 교차표를 함께 보고한다. GT는 지표 계산에만 사용한다. 신뢰구간은 video 단위 paired bootstrap(seed42, 400회)으로 계산하며, 같은 영상·query의 반복 분기와 조건을 함께 재표집한다.

분기 변경 + 후보만 제공 대비 성능 향상 + 기존 token 고정 대비 향상이 함께 관측될 때 분기 수정의 유용성을 지지한다. 분기 변경만 있고 IoU가 그대로면 형식·동일 구간 표현 차이인지 확인한다. 기존 token 고정과 같은 정도로 좋아지면 그 분기의 변경보다는 뒤쪽 token 수정에서 얻은 효과로 해석한다.

## 2차 확장: 반복 feedback을 이용한 실제 탐색

1차 실험은 한 분기에서 실제 새 경로를 생성하는 최소 실험이다. 이후에는 seed8에서 시작해 신규 경로4개마다 재검토하고16·32 경로까지 확장한다. 이때 새로 생성된 prefix와 후보를 다음 재검토에 실제로 사용한다.

같은 review schedule과 최종 선택 절차에서 두 조건을 비교한다.

- 원래 token distribution으로 다음 분기와 conditional rollout을 정하는 조건.
- 재검토 뒤 token distribution으로 다음 분기와 conditional rollout을 정하는 조건.

원래 경로로 돌아가야 하는 경우 기존 prefix를 context별로 다시 평가한다. 갱신된 context의 probability를 과거 context probability와 섞어 heap에 넣지 않는다. context가 갱신되면 남은 frontier를 동일한 새 context에서 다시 평가하거나, 그 비용이 예산을 넘으면 확장을 중지한다.

두 조건의 최종 selector를 같게 두어 탐색 feedback의 효과를 분리한다. 별도로 review 없이 deterministic rollout만 더 수행하는 baseline을 추가하여 review 비용을 탐색에 썼을 때와 비교한다. 최종 선택에도 모델 연산을 사용하며 max probability만으로 대체하지 않는다.

16·32는 경로 수 기준 비교이며 계산량 일치를 의미하지 않는다. 계산량 기준 비교의 query별 budget은 1차 실험에서 측정한 비용만으로, GT 점수와 무관하게 고정한다. 추가 review·frontier 재평가·suffix 생성·최종 선택을 모두 포함한 동일 decoder input token 상한에서 비교하고, latency와 attention-context work도 함께 보고한다. 최종 선택 예산을 먼저 예약하며, budget으로 완료하지 못한 경로는 후보로 세지 않는다. 계산량 sweep의 구체적 상한은 1차 비용 측정 후 별도 config로 동결한다.

## 저장·검증·표시 계획

새 응답·context·분기 정책은 기존 raw data를 덮어쓰지 않고 같은 rollout reconsideration 계열의 의미 있는 조건 `branch_feedback` 아래에 저장한다. 실제 실행이 시작될 때만 대응하는 results/artifacts 경로를 생성한다. 기존 seed 응답과 입력은 경로·hash로 참조한다.

필수 검증은 GT redaction invariance, 원래 parent replay, 정확한 prefix 보존, context별 cache 분리, 자유 선택에서 argmax 일치, start 변경 후 end 재계산, parser 예외 및 EOS 실패 기록, 조건별 비용 합산이다.

실행 시 MD와 interactive detail을 기존 계열 대표 뷰에 통합한다. 분기 직전 prefix, review, 원래/변경 token, 두 child distribution, 새 conditional rollout, GT와 parent 대비 IoU를 한 화면에서 비교한다. 별도 서버는 만들지 않으며 통합8780 index를 사용한다.
