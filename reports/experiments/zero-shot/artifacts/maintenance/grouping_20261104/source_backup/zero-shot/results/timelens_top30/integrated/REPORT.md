# 세 엔트로피 분석 통합: 읽기 쉬운 순서

[통합 영상·토큰 탐색 화면](report.html)

## 먼저 샘플 하나에서 확인하기

1. 모델·벤치마크·샘플을 고르고 영상에서 GT와 예측 구간을 재생한다.
2. 전체 구간 IoU, 시작만 평가한 IoU, 종료만 평가한 IoU를 나란히 확인한다. 시작/종료 단독 평가는 반대쪽 시각을 GT로 고정한다.
3. 시작 또는 종료를 선택하고, 토큰 표의 **전체 구간 판정 / GT 고정 판정** 차이를 본다. 처음 상실한 토큰은 빨간색이며 실제 자리 이름과 H를 함께 표시한다.
4. 토큰 행을 클릭해 아래에서 **같은 모델·벤치마크·시각·GT 자릿수·생성 자릿수·역할**의 유지/최초상실 그룹과 비교한다. N<5는 소표본, 데이터가 없으면 관측 없음으로 표시한다.

색상은 단계의 판정을, 엔트로피는 숫자와 막대 길이로 표현한다. 큰 H와 실패를 같은 색으로 표시하지 않아 “불확실하면 곧 틀린다”는 인상을 피했다. 원래 scatter는 분포 탐색 자료로 남기고 통합 화면의 첫 화면에서는 사용하지 않았다.

## 세 분석이 답하는 질문

| 분석 | 질문 | 통합 화면에서 보는 곳 |
|---|---|---|
| 전체 구간 prefix | 생성 도중 언제 전체 구간의 IoU≥0.5 가능성이 사라졌나? | 전체 구간 카드와 토큰 표의 전체 구간 판정 |
| 반대쪽 GT 고정 | 시작 또는 종료 자체가 틀린 건가, 반대쪽 오차에 가린 건가? | 시작/종료 IoU 카드와 GT 고정 판정 |
| 자릿수·역할 분리 | 같은 자리에서 비교해도 최초 실패 step의 H가 높은가? | 행 선택 후 동일 역할 그룹의 평균 H·N |

## 정량 요약

고정 pilot 300개 × 두 모델=600개 응답이다. 초 단위/소수 한 자리 분석 가능 응답은 594개, endpoint는 1,188개, 역할이 지정된 step은 5,905개다. 나머지 6개는 MM:SS 등의 형식이며, 통합 화면에서도 원본 응답·영상과 제외 사유를 보여준다.

| 모델 | 벤치마크 | 평가 대상 | 분석 N | GT 고정 R@.5 (%) | 공동 판정 실패→단독 성공 N | 최초 가능성 상실 step 수 |
|---|---|---|---:|---:|---:|---:|
| qwen25 | charades-timelens | start | 100 | 73.0 | 31 | 27 |
| qwen25 | activitynet-timelens | start | 100 | 67.0 | 40 | 33 |
| qwen25 | qvhighlights-timelens | start | 100 | 82.0 | 58 | 18 |
| qwen3 | charades-timelens | start | 100 | 82.0 | 33 | 18 |
| qwen3 | activitynet-timelens | start | 96 | 68.8 | 25 | 30 |
| qwen3 | qvhighlights-timelens | start | 98 | 74.5 | 12 | 25 |
| qwen25 | charades-timelens | end | 100 | 56.0 | 14 | 44 |
| qwen25 | activitynet-timelens | end | 100 | 43.0 | 16 | 57 |
| qwen25 | qvhighlights-timelens | end | 100 | 45.0 | 21 | 55 |
| qwen3 | charades-timelens | end | 100 | 63.0 | 14 | 37 |
| qwen3 | activitynet-timelens | end | 96 | 55.2 | 12 | 43 |
| qwen3 | qvhighlights-timelens | end | 98 | 77.6 | 15 | 22 |

공동 구간 IoU만으로 분류하면 한쪽 오류가 다른 쪽의 단계를 가린다. 이를 줄이는 주 분석은 GT 고정 판정이다. 다만 GT 고정은 평가에만 적용했으며, 종료 토큰의 분포는 원래 생성된 시작 시각에 조건화되어 있다.

역할 비교에서는 유지/최초 상실 각 그룹이 5개 이상인 셀을 우선 읽는다. 같은 자릿수/역할에서조차 영상·질의·GT 구간 길이와 이전 prefix는 다르므로 막대 평균 차이는 인과 효과나 오류 탐지 성능이 아니다. 전체 숫자 평균과 여러 자릿수를 섞은 산점도보다 해석 범위가 좁고 구체적이다.

## 일관되게 유지한 기준

- 기준은 IoU≥0.5이며, 원본 CSV에는 0.3/0.7 판정도 보존했다.
- 가능성 유지란 올바른 GT 문자열을 생성했다는 뜻이 아니라 허용한 completion 중 성공 구간이 남아 있다는 뜻이다.
- 초 단위 정수/소수 한 자리, 불필요한 선행 0 없음이라는 표기 범위를 사용했다.
- 예측만 기존 규칙대로 반올림하고 GT는 원 정밀도를 유지했다.
- H는 선택 직전 raw 전체 어휘 분포의 엔트로피(nat)다. 낮은 H는 높은 IoU를 보장하지 않는다.
- 자릿수와 역할은 반올림 전 생성 문자열에서 사후 지정했다. 여러 역할을 포함한 token을 중복 집계하지 않았다.
- 기존 Decord 디코딩 품질 한계와 표본 크기 한계는 유지된다.

## 원본 결과

- [전체 구간 prefix 분석](../../../artifacts/runs/timelens_top30/integrated/sources/prefix_entropy/REPORT.md)
- [GT 고정 endpoint 분석](../../../artifacts/runs/timelens_top30/integrated/sources/endpoint_gt_entropy/REPORT.md)
- [자릿수·역할 분석](../../../artifacts/runs/timelens_top30/integrated/sources/digit_role_entropy/REPORT.md)
- [GT 고정 step CSV](../../../artifacts/runs/timelens_top30/integrated/sources/endpoint_gt_entropy/per_step.csv)
- [역할별 통계 CSV](../../../artifacts/runs/timelens_top30/integrated/sources/digit_role_entropy/role_statistics.csv)

통합 과정에서 모델 추론이나 판정 기준을 바꾸지 않았다. 원본 entropy·label과 통합 값의 일치, 600개 응답 커버리지, 5,905개 역할 step 결합을 검증했다. HTML은 상대경로 자산을 함께 유지하면 file://에서도 사용할 수 있으며 영상 구간 탐색은 로컬 HTTP 서버를 권장한다.
