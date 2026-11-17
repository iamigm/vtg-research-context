"""Proposed, unexecuted video-evidence experiments for the entropy appendix.

Primary method sources were checked on 2026-11-17. This module contains no
measurements and does not launch inference. Existing entropy/error results are
reported separately by entropy_error_appendix.py.
"""


def evidence_design_markdown():
    return r"""## 영상 증거와 entropy를 연결하는 다음 실험 — 설계 제안, 미실행

목표를 **“entropy가 낮아지는 경로”보다 “영상 증거 때문에 정답 후보 쪽으로 확신이 모이는 경로”**로 잡는 것이 좋다. 앞의 표는 기존 응답에서 관측한 연관성이다. 아래는 그 연관성을 실제 영상 활용과 연결하기 위한 새 실험 설계이며, 아직 실행하거나 성능 개선을 확인한 결과가 아니다.

### 먼저 구분할 두 가지 변화

현재 viewer의 `Hₜ − Hₜ₋₁`은 서로 다른 prefix에서 계산한 분포의 차이다. 숫자 뒤에 소수점이 올 때와 소수점 뒤에 숫자가 올 때는 다음 token의 종류 자체가 달라진다. 같은 path로 제한해도 이 문법 효과는 남는다. 또한 `Hₜ`는 t번째 token을 고르기 **전** 분포의 entropy다. 그 분포에서 어떤 token을 선택했다고 이미 계산된 `Hₜ`가 낮아지는 것은 아니다. 선택은 다음 prefix와 `Hₜ₊₁`에 영향을 준다.

영상 효과를 보려면 query와 token prefix `y<ₜ`를 완전히 고정하고 영상 입력만 바꿔야 한다. `pₜ⁺ = p(token | query, y<ₜ, V⁺)`, `pₜ⁻ = p(token | query, y<ₜ, V⁻)`라 두자. `V⁺`는 증거가 있는 입력, `V⁻`는 해당 증거를 약화한 입력이다. 이때 `Gₜ = H(pₜ⁻) − H(pₜ⁺)`를 비교한다. 양수면 그 입력 개입에서 증거가 있는 쪽이 더 집중됐다는 뜻이다. 이는 **정해진 입력 개입에 대한 모델의 반응**이며, 영상의 참된 인과 구조나 정답임을 자동으로 입증하지 않는다.

`Gₜ`가 양수라고 정답인 것도, 음수라고 증거가 쓸모없는 것도 아니다. 잘못된 확신을 깨는 증거는 먼저 entropy를 높일 수 있다. 따라서 `Gₜ`와 정답 가능 후보의 probability mass 변화를 함께 본다. 서식 token의 entropy 감소는 숫자 판단 개선과 분리한다.

### 1단계: 같은 prefix에서 증거 제거와 복원을 비교한다

우선 기존 pilot 300개 query에서 저장된 prefix를 그대로 재생해 다음 token 분포만 다시 계산하는 실험을 제안한다. 아직 오류가 나지 않은 numeric prefix 전체를 주 분석 대상으로 삼고, 최초 오류 직전 prefix를 별도 표기한다. 오류 직전 사례만 고르면 어떤 경우에도 오류를 고른 표본이 되므로 정상 선택도 포함한다. 이미 틀린 prefix는 복구 가능성 연구로 분리한다. 기존 full 9,404개 결과와 pilot은 독립 표본으로 합산하지 않는다.

| 조건 | 입력을 어떻게 바꾸나 | 이 비교로 확인하는 것 |
|---|---|---|
| 원본 | 기존 frame IDs·timestamp·H×W·prompt·prefix 그대로 | 재실행 일치와 기준 분포 |
| 관련 증거 제거 | query와 관계된 구간의 frame 내용만 마스킹·약화. frame 수와 timestamp 위치는 유지 | 그 구간에 모델이 실제로 반응하는가 |
| 대조 구간 제거 | 같은 수·길이·처리 강도로 다른 구간을 제거. 여러 고정 seed 사용 | 어떤 영상 손상에도 생기는 효과와 관련 증거의 효과를 구분 |
| 시간 순서 교란 | frame 수·해상도·내용 집합을 유지하고 내용의 시간 순서를 교란. timestamp 슬롯 규칙을 고정·기록 | 정적 장면 정보보다 시간 순서에 민감한가. 인위적 불일치 영향도 포함 |
| query 기반 증거 보강 | coarse context는 남기고 일부 frame 예산을 query 관련 구간과 경계 주변에 배정 | 같은 예산에서 더 유용한 frame을 보여 주면 판단이 바뀌는가 |
| 무작위 보강 대조 | 같은 frame 예산과 같은 처리로 무작위 구간에 재배정 | 추가 계산·샘플링 변화 자체의 효과를 구분 |

**표 해설.** 핵심은 관련 구간을 지웠을 때와 대조 구간을 지웠을 때의 차이다. 화면이 손상되기만 해도 entropy가 오를 수 있으므로 원본 대 제거의 차이 하나로 결론 내리지 않는다. 보강 조건에서는 총 frame 수·각 frame 해상도·visual token 예산을 맞추고, 바뀐 frame timestamp와 전처리 시간을 기록한다. 순서 교란은 별도 진단이며 모든 오류를 시간 증거 부족으로 해석하는 근거가 되지 않는다.

실제 사용할 구간은 **GT를 보지 않는** query 기반 retrieval 또는 사전에 고정한 coarse prediction으로 선택한다. GT 구간을 제거·보강하는 조건은 “정확한 증거 위치를 안다면 효과가 있는가”를 보는 oracle 진단으로만 둔다. oracle 결과와 배포 가능한 선택기의 결과를 같은 성능으로 보고하지 않는다. 원본 전체 영상을 잘라 시간 좌표를 다시 매기면 답 자체가 바뀔 수 있으므로 원래 영상의 시간축과 timestamp 표기를 유지한다.

모든 조건에서 모델 revision, prompt, prefix token IDs, raw logits의 temperature(1), frame 수·실제 H×W, 시간축 병합 설정을 맞춘다. 입력 영상이 달라지면 visual encoding과 그에 의존하는 KV cache를 다시 계산한다. 원본 영상의 cache를 그대로 공유하면 영상 개입을 제대로 평가하지 못할 수 있다. 원본 재실행의 raw top-30과 저장값이 먼저 일치해야 한다.

### 무엇을 수치로 볼 것인가

| 지표 | 정의·분모 | 해석 |
|---|---|---|
| 증거에 따른 entropy 차이 | `Gₜ = H(V⁻, y<ₜ) − H(V⁺, y<ₜ)`; 같은 query·prefix 쌍 | 양수는 증거가 있는 조건에서 분포가 더 집중됨 |
| 관련 구간의 선택적 효과 | `Gₜ(관련 제거) − Gₜ(대조 제거)`; 같은 prefix·동일 손상량 | 양수면 관련 구간 제거의 entropy 효과가 더 큼. 손상 방식 의존성을 함께 보고 |
| 정답 가능 mass 변화 | `ΔSₜ = Σ_{c∈C_safe} pₜ⁺(c) − Σ_{c∈C_safe} pₜ⁻(c)` | 양수면 정답으로 완성할 수 있는 후보 쪽으로 이동. 최종 정답 확률은 아님 |
| 선택 교정·훼손률 | 동일 prefix에서 오답→가능 후보, 가능→오답으로 argmax가 바뀐 쌍의 수 / 평가 prefix 쌍 수 | 분포가 날카로워졌어도 잘못된 선택을 강화했는지 확인 |
| 최종 성능 | 별도의 실제 rollout에서 endpoint 정답률, mIoU, IoU≥0.3·0.5·0.7의 query 비율과 기준 대비 pp | 한 step의 개선이 최종 temporal grounding 개선으로 이어지는지 확인 |
| 비용·개입 비율 | query당 재인코딩 횟수, forward 수, wall time, peak memory, 개입 step 수 / 대상 numeric step 수 | 동일 계산 예산의 대조 정책과 비교 |

**표 해설.** 가장 설득력 있는 패턴은 관련 증거를 보여 줬을 때 entropy가 줄고(`Gₜ>0`), 정답 가능 mass가 늘며(`ΔSₜ>0`), 실제 rollout의 오류도 줄어드는 경우다. `Gₜ>0`인데 `ΔSₜ<0`이면 잘못된 확신을 강화한 반례다. 이 두 방향을 2×2로 나눠 query·prefix 수를 모두 적고, entropy가 높아지며 정답 후보가 살아나는 경우도 보존한다. 정답 가능 후보 집합은 같은 prefix와 같은 completion 규칙으로 평가하며, 판정하지 않은 후보를 오답으로 채우지 않는다.

기존 결과와 연결할 주 지표는 top-30 재정규화 `H₃₀`로 유지하되, 각 조건의 top-30 mass `M₃₀`와 후보 집합 변화를 함께 기록한다. 서로 다른 조건에서 각자 top-30을 뽑으면 비교 집합이 달라지므로 **공통 후보 집합**에서도 검사한다. 예를 들어 양쪽 top-30의 합집합 U를 만들고, 각 조건의 raw logits에서 U의 모든 token 확률을 직접 조회해 합이 1이 되게 정규화한다. U에 없는 token을 0으로 채우면 안 된다. 가능하면 같은 forward에서 full-vocabulary entropy도 보조 지표로 계산하되, 이를 기존 H₃₀와 같은 지표로 섞지 않는다. full-vocabulary safe mass가 불가능하면 U 내부에서 평가한 mass와 U 밖 mass·미판정 mass를 명시한다.

분석은 모델·dataset·start/end·numeric role별로 나누고, prefix를 평균한 뒤 query 또는 video 단위로 요약한다. 같은 video의 여러 query가 있으면 video 단위 paired bootstrap으로 95% CI를 계산한다. 검색·개입 정책과 임계값은 video가 겹치지 않는 개발 분할에서 정하고 평가 분할에서 고정한다. 기존 full 결과는 이미 열람했으므로 완전한 미열람 test라고 부르지 않는다. 특수 필터는 사례 해석용으로 함께 보고하되, 실제 최초 오류 위치나 정답 여부를 runtime trigger로 사용하지 않는다.

### 2단계: 영상에 의해 지지되는 후보를 선택하는 decoding

1단계에서 관련 증거의 효과가 보이면, 우선 numeric step에서만 query 기반 frame 재확인을 적용하는 작은 정책을 비교한다. entropy 상승·높은 entropy는 재확인 시점을 고르는 후보 신호다. 높은 entropy만으로 틀렸다고 판단하거나, 실제 최초 오류 annotation을 사용해 재확인을 시작하지 않는다. 같은 재확인 횟수의 무작위 trigger와 고정 주기 trigger를 대조군으로 둔다.

후보 c의 영상 의존 점수를 `Eₜ(c) = log p(c | V⁺, y<ₜ) − log p(c | V⁻, y<ₜ)`로 정의하고, 예시 선택 점수를 `scoreₜ(c) = log p(c | V⁺, y<ₜ) + αEₜ(c)`로 둘 수 있다. α≥0은 개발 분할에서 결정한다. 원본 또는 증거 입력에서 이미 그럴듯한 후보만 남기는 probability cutoff를 함께 두어, 두 조건 모두 확률이 극히 작은 후보의 큰 비율 차이가 선택을 지배하지 않게 한다. 계산에는 같은 prefix의 raw log-softmax 값을 사용하며, 두 조건의 잘린 top-30만 서로 빼지 않는다.

이는 증거가 사라질 때 지지를 잃는 후보를 우대하는 **검증할 정책 예시**다. `Eₜ(c)>0` 자체는 정답 보장이 아니므로, entropy 변화가 작거나 증거 개입이 불안정한 경우 원래 decoding으로 돌아가는 gate를 개발 분할에서 검증한다. α=0, 재확인만 적용, contrastive score만 적용, entropy만 낮추는 temperature 대조군, 재확인+score를 구분해 평가한다. temperature를 낮추는 것은 영상 증거를 더 활용하지 않아도 분포를 날카롭게 만들 수 있다. 수정된 score 분포의 entropy 감소와 **모델 raw 분포에서 관측한 entropy 감소**도 별도 열로 기록한다.

후보를 선택한 뒤 `Hₜ₊₁`이 가장 작은 branch를 고르는 방법은 첫 실험의 주목표로 삼지 않는 편이 좋다. 확신한 오답이나 소수점·종료 token 같은 쉬운 문법 branch가 유리해질 수 있기 때문이다. branch 비교를 하더라도 동일한 길이·numeric 의사결정 위치에서 영상 증거 점수와 정답 성능을 함께 평가해야 한다. 정상 후보를 오답으로 바꾸는 비율과 종료·서식 분포도 기록한다.

### 이 설계가 참고한 선행 연구

원본과 훼손된 시각 입력의 logits를 대비하는 발상은 [Visual Contrastive Decoding, CVPR 2024 공식 구현](https://github.com/DAMO-NLP-SG/VCD)에 근거한다. 해당 방법은 이미지 object hallucination을 다루며, 여기의 temporal grounding 숫자 정확도나 entropy 감소를 입증한 결과는 아니다.

[Video Contrastive Decoding, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea6d17af54f827336fc8fed27ca0319d-Abstract-Conference.html)는 시간적으로 일관된 영상과 시간 구조를 훼손한 영상의 decoding을 대비하는 접근을 제안한다. 시간 순서 대조군을 두는 데 참고할 수 있다.

[SEASON, CVPR 2026 연구팀 소개](https://research.nvidia.com/labs/twn/publication/cvpr_2026_season/)는 token별 진단과 temporal·spatial negative를 이용한 adaptive contrastive decoding을 제안한다. 모든 token을 똑같이 처리하지 않는 gate 설계에 참고할 수 있으나, 현재 모델·dataset에서의 효과는 별도 검증해야 한다.

현재 단계에서 먼저 할 일은 **고정 prefix + 관련/대조 증거 제거 실험**이다. 이 실험에서 정답 후보를 지지하는 영상 효과가 확인되면 query 기반 재확인과 contrastive decoding으로 확장한다. entropy 감소만을 성공 기준으로 삼지는 않는다.
"""


def evidence_markdown():
    """Alias used by the quantitative appendix generator."""
    return evidence_design_markdown()
