<!-- report-appendix:APPENDIX.md parts=12 -->
<!-- report-part:0:begin -->
# Rollout reconsideration · pilot600

완료: 600 query, dataset별200. 3개 접근과 direct regeneration control을 같은 고정 cohort에서 평가했다. 생성 input hash는 기존 deterministic rollout과 일치하며 GT는 보고 단계에서만 사용했다.

[실험 설계](SPEC.md)

<!-- report-part:0:end -->
<!-- report-part:2:begin -->

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## 지표와 비교 기준

<!-- clipping-exception:begin -->

**범위 초과 예측:** 예측 start/end를 `[0, 영상 길이]`로 clipping하고 길이0이면 IoU=0으로 처리하는 예외를 사후 확인했다. GT는 유지한다. 재검토 10개 조건의 mIoU 변화는 +0.0010~+0.0021다. 모델 추론·후보 선택을 다시 실행하지 않았으며 기존 표는 clipping 전 기준이다.

<!-- clipping-exception:end -->

mIoU는 600 query의 temporal IoU 평균, R@τ는 IoU≥τ query 비율이다. IoU=intersection duration/union duration이며 높을수록 좋다. Δ는 같은 query의 greedy 대비 차이, Δpp는 성공률 차이×100이다. failure는 파싱 실패·invalid interval·EOS 미도달 비율이고0점으로 포함한다. model seconds는 CUDA 동기화로 측정한 의사결정 model wall time이며 공통 video prefill과 사후 probability diagnostic을 제외한다. decoder tokens는 기존 proposal generation 비용과 새 review의 prompt·generated token·KV replay를 합산한다. probability diagnostics 비용은 별도 원자료에 보존한다. 기존 baseline은 추가 review time이0이다. 평가 호출 수를 맞춘 비교가 FLOPs 일치를 뜻하지 않는다.

<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 전체 성능

| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | model evaluations | decoder tokens (proposal+review) | review model seconds | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 600 | 0.5157 (+0.0000) | 0.6783 (+0.00) | 0.5433 (+0.00) | 0.3817 (+0.00) | 0.0 | 16.8 | 0.000 | 0.000 |
| max_probability_pool8 | 600 | 0.5273 (+0.0117) | 0.6950 (+1.67) | 0.5633 (+2.00) | 0.4000 (+1.83) | 0.0 | 84.0 | 0.000 | 0.000 |
| probability_weighted_medoid_pool8 | 600 | 0.5319 (+0.0162) | 0.7000 (+2.17) | 0.5650 (+2.17) | 0.4033 (+2.17) | 0.0 | 84.0 | 0.000 | 0.000 |
| direct_pool8 | 600 | 0.5279 (+0.0122) | 0.6983 (+2.00) | 0.5600 (+1.67) | 0.3917 (+1.00) | 1.0 | 374.4 | 0.791 | 0.000 |
| agent_pool8 | 600 | 0.5200 (+0.0044) | 0.6783 (+0.00) | 0.5483 (+0.50) | 0.3917 (+1.00) | 6.0 | 2498.5 | 6.490 | 0.000 |
| verifier_pool8 | 600 | 0.5221 (+0.0064) | 0.6883 (+1.00) | 0.5483 (+0.50) | 0.3850 (+0.33) | 7.0 | 645.4 | 1.060 | 0.000 |
| tts_pool8_select4 | 600 | 0.5253 (+0.0096) | 0.6950 (+1.67) | 0.5567 (+1.33) | 0.3883 (+0.67) | 7.0 | 779.0 | 1.054 | 0.000 |
| tts_pool8_select8 | 600 | 0.5263 (+0.0106) | 0.6950 (+1.67) | 0.5567 (+1.33) | 0.3917 (+1.00) | 14.8 | 1266.6 | 1.443 | 0.000 |
| direct_pool32 | 600 | 0.5201 (+0.0045) | 0.6833 (+0.50) | 0.5483 (+0.50) | 0.3850 (+0.33) | 1.0 | 611.4 | 0.795 | 0.000 |
| agent_pool32 | 600 | 0.5115 (-0.0041) | 0.6767 (-0.17) | 0.5400 (-0.33) | 0.3717 (-1.00) | 6.5 | 3321.0 | 7.111 | 0.000 |
| verifier_pool32 | 600 | 0.5244 (+0.0087) | 0.6883 (+1.00) | 0.5483 (+0.50) | 0.3867 (+0.50) | 14.9 | 1207.7 | 1.438 | 0.000 |
| tts_pool32_select4 | 600 | 0.5193 (+0.0036) | 0.6900 (+1.17) | 0.5467 (+0.33) | 0.3783 (-0.33) | 7.0 | 1020.0 | 1.073 | 0.000 |
| tts_pool32_select8 | 600 | 0.5145 (-0.0012) | 0.6867 (+0.83) | 0.5367 (-0.67) | 0.3750 (-0.67) | 14.9 | 1520.3 | 1.449 | 0.000 |
| max_probability_pool32 | 600 | 0.5251 (+0.0094) | 0.6917 (+1.33) | 0.5583 (+1.50) | 0.4017 (+2.00) | 0.0 | 291.8 | 0.000 | 0.000 |
| probability_weighted_medoid_pool32 | 600 | 0.5331 (+0.0174) | 0.6967 (+1.83) | 0.5617 (+1.83) | 0.4083 (+2.67) | 0.0 | 291.8 | 0.000 | 0.000 |
| oracle_pool8 | 600 | 0.6697 (+0.1541) | 0.8350 (+15.67) | 0.7067 (+16.33) | 0.5533 (+17.17) | 0.0 | 84.0 | 0.000 | 0.000 |
| oracle_pool32 | 600 | 0.7295 (+0.2139) | 0.8883 (+21.00) | 0.7683 (+22.50) | 0.6417 (+26.00) | 0.0 | 291.8 | 0.000 | 0.000 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## Paired comparison · 95% bootstrap CI

video 단위 paired bootstrap의 방법 간 차이다. CI가0을 포함하면 순위 차이를 확정하지 않는다.

| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| verifier_pool8 minus agent_pool8 | +0.0020 [-0.0060, +0.0106] | +0.0000 [-0.0134, +0.0134] |
| tts_pool8_select4 minus agent_pool8 | +0.0052 [-0.0029, +0.0134] | +0.0083 [-0.0067, +0.0218] |
| tts_pool8_select4 minus verifier_pool8 | +0.0032 [-0.0028, +0.0096] | +0.0083 [-0.0033, +0.0202] |
| verifier_pool32 minus agent_pool32 | +0.0128 [+0.0005, +0.0240] | +0.0083 [-0.0132, +0.0265] |
| tts_pool32_select8 minus agent_pool32 | +0.0030 [-0.0092, +0.0147] | -0.0033 [-0.0251, +0.0167] |
| tts_pool32_select8 minus verifier_pool32 | -0.0099 [-0.0176, -0.0016] | -0.0117 [-0.0268, +0.0034] |
| tts_pool8_select8 minus tts_pool8_select4 | +0.0010 [-0.0010, +0.0030] | +0.0000 [-0.0050, +0.0066] |
| tts_pool32_select4 minus tts_pool8_select4 | -0.0060 [-0.0136, +0.0018] | -0.0100 [-0.0264, +0.0050] |
| tts_pool32_select8 minus tts_pool8_select8 | -0.0118 [-0.0200, -0.0041] | -0.0200 [-0.0366, -0.0067] |
| tts_pool32_select8 minus tts_pool32_select4 | -0.0048 [-0.0090, -0.0008] | -0.0100 [-0.0216, +0.0000] |

<!-- report-part:5:end -->
<!-- report-part:11:begin -->
## Reconsideration 프로세스 탐색

[접근별 프로세스](viewer.html)는 deterministic rollout의 하위 분석이다. 기존 사례 유형·수치 필터와 같은 질문을 사용하며 원래 trajectory 문장·분기 지도·baseline 분포는 원본 뷰 링크에서 확인한다. 이 페이지는 그 화면을 복제하지 않는다.

agent는 실제 controller action과 inspector observation을 시간순으로 표시한다. 명시적 selected candidate가 없으므로 마지막 inspection을 최종 선택으로 간주하지 않는다. verifier는 후보별 event/경계 Yes 확률과 그 곱, 실제 선택 및 final regeneration을 연결한다. test-time scaling은 양쪽 presentation order의 pairwise 점수, round별 승자와 bye를 대진 구조로 표시한다. direct regeneration은 추가 평가 없이 같은 ledger로 생성한 두 pool 조건의 대조다. 각 접근 안에서 예산 조건을 비교한다.

pool은 기존 deterministic 후보의 앞8/32개다. 유효하고 중복되지 않은 구간에서 GT 없이 farthest-first로 다양성을 확보하며, ledger의 후보 목록은 최대8개다. verifier는 별도로 최대3/7개를 평가하고 TTS는 최대4/8개를 대진에 넣는다. 실제 목록·평가 수·원문은 저장된 기록을 사용한다. final answer는 선택 후보와 다를 수 있다. 사후 token distribution 진단은 기본 검토 과정과 분리한다. 원본 응답·영상·baseline token distribution는 복제하지 않는다.
<!-- reconsideration-viewer:end -->

<!-- report-part:11:end -->
## 핵심 지표 정의

IoU=예측·GT 시간 구간의 교집합 길이/합집합 길이, mIoU=평가 query의 IoU 평균, R@τ=IoU≥τ인 query 수/평가 query 수(τ=.3/.5/.7)다. 모두 높을수록 좋으며 % 표시는 100을 곱한다. Δpp=100×(방법−기준)이다. Oracle은 GT로 고른 후보 집합의 사후 상한이며 실제 선택 성능과 구분한다. 반복 응답은 query 안에서 평균한 뒤 query 간 집계한다.

## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
