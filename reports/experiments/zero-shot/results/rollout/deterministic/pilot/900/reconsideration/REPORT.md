<!-- report-appendix:APPENDIX.md parts=12 -->
<!-- report-part:0:begin -->
# Rollout reconsideration · pilot900

완료: 900 query, dataset별300. 3개 접근과 direct regeneration control을 같은 고정 cohort에서 평가했다. 생성 input hash는 기존 deterministic rollout과 일치하며 GT는 보고 단계에서만 사용했다.

[실험 설계](SPEC.md)

<!-- report-part:0:end -->
<!-- report-part:2:begin -->

<!-- report-part:2:end -->
<!-- report-part:3:begin -->
## 지표와 비교 기준

<!-- clipping-exception:begin -->

**범위 초과 예측:** 예측 start/end를 `[0, 영상 길이]`로 clipping하고 길이0이면 IoU=0으로 처리하는 예외를 사후 확인했다. GT는 유지한다. 재검토 10개 조건의 mIoU 변화는 +0.0009~+0.0016다. 모델 추론·후보 선택을 다시 실행하지 않았으며 기존 표는 clipping 전 기준이다.

<!-- clipping-exception:end -->

mIoU는 900 query의 temporal IoU 평균, R@τ는 IoU≥τ query 비율이다. IoU=intersection duration/union duration이며 높을수록 좋다. Δ는 같은 query의 greedy 대비 차이, Δpp는 성공률 차이×100이다. failure는 파싱 실패·invalid interval·EOS 미도달 비율이고0점으로 포함한다. model seconds는 CUDA 동기화로 측정한 의사결정 model wall time이며 공통 video prefill과 사후 probability diagnostic을 제외한다. decoder tokens는 기존 proposal generation 비용과 새 review의 prompt·generated token·KV replay를 합산한다. probability diagnostics 비용은 별도 원자료에 보존한다. 기존 baseline은 추가 review time이0이다. 평가 호출 수를 맞춘 비교가 FLOPs 일치를 뜻하지 않는다.

<!-- report-part:3:end -->
<!-- report-part:4:begin -->
## 전체 성능

| method | n | mIoU (Δ) | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) | model evaluations | decoder tokens (proposal+review) | review model seconds | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 900 | 0.5193 (+0.0000) | 0.6767 (+0.00) | 0.5456 (+0.00) | 0.3967 (+0.00) | 0.0 | 16.8 | 0.000 | 0.000 |
| max_probability_pool8 | 900 | 0.5268 (+0.0075) | 0.6933 (+1.67) | 0.5578 (+1.22) | 0.4022 (+0.56) | 0.0 | 84.2 | 0.000 | 0.000 |
| probability_weighted_medoid_pool8 | 900 | 0.5350 (+0.0157) | 0.7011 (+2.44) | 0.5622 (+1.67) | 0.4156 (+1.89) | 0.0 | 84.2 | 0.000 | 0.000 |
| direct_pool8 | 900 | 0.5333 (+0.0141) | 0.6978 (+2.11) | 0.5644 (+1.89) | 0.4089 (+1.22) | 1.0 | 375.1 | 0.797 | 0.000 |
| agent_pool8 | 900 | 0.5259 (+0.0066) | 0.6811 (+0.44) | 0.5533 (+0.78) | 0.4089 (+1.22) | 6.0 | 2507.0 | 6.570 | 0.000 |
| verifier_pool8 | 900 | 0.5262 (+0.0069) | 0.6856 (+0.89) | 0.5511 (+0.56) | 0.4022 (+0.56) | 7.0 | 646.5 | 1.069 | 0.000 |
| tts_pool8_select4 | 900 | 0.5285 (+0.0093) | 0.6911 (+1.44) | 0.5578 (+1.22) | 0.4044 (+0.78) | 7.0 | 780.6 | 1.064 | 0.000 |
| tts_pool8_select8 | 900 | 0.5298 (+0.0105) | 0.6922 (+1.56) | 0.5578 (+1.22) | 0.4067 (+1.00) | 14.8 | 1268.7 | 1.451 | 0.000 |
| direct_pool32 | 900 | 0.5233 (+0.0040) | 0.6811 (+0.44) | 0.5533 (+0.78) | 0.3967 (+0.00) | 1.0 | 612.0 | 0.800 | 0.000 |
| agent_pool32 | 900 | 0.5146 (-0.0047) | 0.6722 (-0.44) | 0.5422 (-0.33) | 0.3878 (-0.89) | 6.5 | 3404.6 | 7.306 | 0.000 |
| verifier_pool32 | 900 | 0.5254 (+0.0062) | 0.6844 (+0.78) | 0.5500 (+0.44) | 0.3989 (+0.22) | 14.9 | 1209.2 | 1.449 | 0.000 |
| tts_pool32_select4 | 900 | 0.5215 (+0.0022) | 0.6856 (+0.89) | 0.5489 (+0.33) | 0.3878 (-0.89) | 7.0 | 1020.9 | 1.080 | 0.000 |
| tts_pool32_select8 | 900 | 0.5166 (-0.0027) | 0.6800 (+0.33) | 0.5411 (-0.44) | 0.3867 (-1.00) | 14.9 | 1522.5 | 1.458 | 0.000 |
| max_probability_pool32 | 900 | 0.5255 (+0.0063) | 0.6933 (+1.67) | 0.5544 (+0.89) | 0.4011 (+0.44) | 0.0 | 292.3 | 0.000 | 0.000 |
| probability_weighted_medoid_pool32 | 900 | 0.5369 (+0.0176) | 0.6978 (+2.11) | 0.5667 (+2.11) | 0.4200 (+2.33) | 0.0 | 292.3 | 0.000 | 0.000 |
| oracle_pool8 | 900 | 0.6717 (+0.1524) | 0.8278 (+15.11) | 0.7078 (+16.22) | 0.5689 (+17.22) | 0.0 | 84.2 | 0.000 | 0.000 |
| oracle_pool32 | 900 | 0.7347 (+0.2154) | 0.8767 (+20.00) | 0.7778 (+23.22) | 0.6500 (+25.33) | 0.0 | 292.3 | 0.000 | 0.000 |

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
## Paired comparison · 95% bootstrap CI

video 단위 paired bootstrap의 방법 간 차이다. CI가0을 포함하면 순위 차이를 확정하지 않는다.

| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| verifier_pool8 minus agent_pool8 | +0.0003 [-0.0059, +0.0070] | -0.0022 [-0.0132, +0.0101] |
| tts_pool8_select4 minus agent_pool8 | +0.0026 [-0.0031, +0.0095] | +0.0044 [-0.0056, +0.0167] |
| tts_pool8_select4 minus verifier_pool8 | +0.0023 [-0.0030, +0.0075] | +0.0067 [-0.0022, +0.0156] |
| verifier_pool32 minus agent_pool32 | +0.0109 [+0.0035, +0.0195] | +0.0078 [-0.0056, +0.0234] |
| tts_pool32_select8 minus agent_pool32 | +0.0020 [-0.0078, +0.0115] | -0.0011 [-0.0167, +0.0144] |
| tts_pool32_select8 minus verifier_pool32 | -0.0089 [-0.0154, -0.0038] | -0.0089 [-0.0210, +0.0033] |
| tts_pool8_select8 minus tts_pool8_select4 | +0.0012 [-0.0020, +0.0051] | +0.0000 [-0.0066, +0.0067] |
| tts_pool32_select4 minus tts_pool8_select4 | -0.0070 [-0.0138, -0.0003] | -0.0089 [-0.0222, +0.0044] |
| tts_pool32_select8 minus tts_pool8_select8 | -0.0132 [-0.0207, -0.0074] | -0.0167 [-0.0289, -0.0056] |
| tts_pool32_select8 minus tts_pool32_select4 | -0.0049 [-0.0088, -0.0014] | -0.0078 [-0.0156, +0.0000] |

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
