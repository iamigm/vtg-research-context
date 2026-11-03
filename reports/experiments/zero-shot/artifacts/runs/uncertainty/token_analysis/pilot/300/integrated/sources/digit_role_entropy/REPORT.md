# 자릿수·토큰 역할별 엔트로피: 반대쪽 시각 GT 고정

기존 594개 응답의 시작·종료 1188개 endpoint를 재분석했다. 기존 분석에서 제외한 MM:SS 형식 6개 응답은 그대로 제외한다. 모델 추론 없이 저장된 엔트로피와 endpoint 독립 판정을 재사용했다.

**그룹:** 모델 × 벤치마크 × 시작/종료 × GT 정수부 자릿수 × 실제 생성 정수부 자릿수. GT와 생성 자릿수가 다른 경우를 별도 그림으로 분리했다. 같은 그룹 안에서 백·십·일의 자리, 소수점, 소수 첫째 자리, 종료 구분자를 비교한다.

**점 하나는 해당 토큰을 고르기 직전의 전체 어휘 raw 엔트로피**다. 시간축 평균이나 최초 실패 위치 정렬을 하지 않았다. 파랑은 IoU≥0.5 정답 가능성을 유지한 step, 빨강은 해당 endpoint에서 처음 잃은 step, 회색은 이미 잃은 이후다. 짧은 가로선은 같은 역할/상태의 중앙값이다. 최초 실패가 없는 성공 응답도 파란 점으로 포함한다.

역할은 **최종 생성된 숫자 문자열을 기준으로 사후 지정**한다. 예를 들어 `25.0`의 `2`는 십의 자리, `5`는 일의 자리, `.`은 소수점, `0`은 소수 첫째 자리다. 생성 당시 모델이 숫자의 길이를 미리 확정했다는 뜻은 아니다. 자릿수 분류는 반올림 전 표기를 사용하며, IoU 평가는 기존과 동일하게 예측만 반올림한다. 예: `99.9`는 생성 자릿수 2지만 반올림 후 100이 된다. GT는 원 정밀도로 고정한다.

여러 역할을 한 토큰이 동시에 포함하면 entropy를 역할별로 복제하지 않고 `multi_role`로 분리했다. 해당 step은 0개다. 단일 역할 그림에서는 제외하고 원시 CSV에는 보존했다.

## GT와 생성 자릿수가 같은 경우

### end · 1자리

![end 1 digits](end_gt1_pred1.png)

### end · 2자리

![end 2 digits](end_gt2_pred2.png)

### end · 3자리

![end 3 digits](end_gt3_pred3.png)

### start · 1자리

![start 1 digits](start_gt1_pred1.png)

### start · 2자리

![start 2 digits](start_gt2_pred2.png)

### start · 3자리

![start 3 digits](start_gt3_pred3.png)

## GT와 생성 자릿수가 다른 경우

| 평가 대상 | GT 자릿수 | 생성 자릿수 | endpoint N (두 모델 합산) | 그림 |
|---|---:|---:|---:|---|
| end | 1 | 2 | 29 | [보기](end_gt1_pred2.png) |
| end | 1 | 3 | 2 | [보기](end_gt1_pred3.png) |
| end | 2 | 1 | 18 | [보기](end_gt2_pred1.png) |
| end | 2 | 3 | 15 | [보기](end_gt2_pred3.png) |
| end | 3 | 1 | 4 | [보기](end_gt3_pred1.png) |
| end | 3 | 2 | 28 | [보기](end_gt3_pred2.png) |
| start | 1 | 2 | 28 | [보기](start_gt1_pred2.png) |
| start | 1 | 3 | 4 | [보기](start_gt1_pred3.png) |
| start | 2 | 1 | 14 | [보기](start_gt2_pred1.png) |
| start | 2 | 3 | 16 | [보기](start_gt2_pred3.png) |
| start | 3 | 1 | 2 | [보기](start_gt3_pred1.png) |
| start | 3 | 2 | 9 | [보기](start_gt3_pred2.png) |

## 같은 역할 내 최초 실패 vs 가능성 유지

아래는 두 상태에 각각 최소 5개 관측이 있는 셀이다. 평균은 동일 역할의 서로 다른 query 간 요약이며, 한 응답의 여러 생성 step을 평균한 값이 아니다. 이 수량 기준은 유의성 검정이 아니다. 나머지 희소 셀을 포함한 전체 통계는 CSV에 보존했다.

| 모델 | 벤치마크 | 대상 | GT/생성 자릿수 | 역할 | 유지 N / 평균 H | 최초실패 N / 평균 H | 차이 H |
|---|---|---|---|---|---|---|---:|
| qwen25 | activitynet-timelens | end | 2/2 | 일의 자리 | 35 / 1.428 | 19 / 1.288 | -0.140 |
| qwen25 | activitynet-timelens | end | 2/2 | 십의 자리 | 54 / 0.574 | 12 / 0.360 | -0.214 |
| qwen25 | activitynet-timelens | end | 3/3 | 일의 자리 | 8 / 1.247 | 5 / 1.492 | +0.245 |
| qwen25 | activitynet-timelens | start | 2/2 | 십의 자리 | 43 / 1.305 | 8 / 1.708 | +0.403 |
| qwen25 | charades-timelens | end | 1/2 | 십의 자리 | 7 / 1.175 | 8 / 0.956 | -0.220 |
| qwen25 | charades-timelens | end | 2/2 | 일의 자리 | 42 / 1.505 | 16 / 1.552 | +0.047 |
| qwen25 | charades-timelens | end | 2/2 | 십의 자리 | 58 / 0.606 | 6 / 0.457 | -0.149 |
| qwen25 | charades-timelens | start | 1/1 | 소수 첫째 자리 | 40 / 1.377 | 5 / 1.787 | +0.410 |
| qwen25 | charades-timelens | start | 2/2 | 일의 자리 | 34 / 1.990 | 12 / 2.147 | +0.156 |
| qwen25 | qvhighlights-timelens | end | 2/2 | 일의 자리 | 30 / 1.547 | 6 / 1.755 | +0.209 |
| qwen25 | qvhighlights-timelens | end | 2/2 | 십의 자리 | 36 / 0.574 | 11 / 0.412 | -0.161 |
| qwen25 | qvhighlights-timelens | end | 3/3 | 일의 자리 | 14 / 1.541 | 11 / 1.678 | +0.137 |
| qwen25 | qvhighlights-timelens | end | 3/3 | 십의 자리 | 25 / 0.308 | 8 / 0.314 | +0.006 |
| qwen25 | qvhighlights-timelens | start | 2/2 | 십의 자리 | 40 / 0.828 | 5 / 1.462 | +0.635 |
| qwen3 | activitynet-timelens | end | 2/2 | 일의 자리 | 38 / 1.232 | 13 / 1.665 | +0.433 |
| qwen3 | activitynet-timelens | end | 2/2 | 십의 자리 | 51 / 0.435 | 9 / 0.498 | +0.063 |
| qwen3 | activitynet-timelens | start | 2/2 | 십의 자리 | 41 / 0.555 | 5 / 0.754 | +0.199 |
| qwen3 | charades-timelens | end | 1/2 | 십의 자리 | 5 / 0.815 | 5 / 0.688 | -0.126 |
| qwen3 | charades-timelens | end | 2/2 | 일의 자리 | 42 / 1.252 | 14 / 1.493 | +0.241 |

두 상태 각각 5개 이상인 단일 역할 셀은 19개다. 자릿수·역할을 분리하면 셀당 관측이 크게 줄어든다. 빨간 점이 파란 점보다 항상 높다고 전제하지 말고, 같은 패널·같은 역할의 겹침과 표본 수를 확인해야 한다.

이 분석은 위치/자릿수 혼합을 줄였지만 GT 구간 길이·영상 내용·이미 생성된 prefix까지 통제한 인과 실험은 아니다. 종료 분포는 여전히 실제 생성한 시작 시각에 조건화되어 있고, GT 고정은 평가에만 적용했다. 기존의 초 단위/소수 한 자리 prefix 가능성 판정 범위와 Decord 한계도 유지된다.

[모든 step·역할 CSV](per_step_roles.csv) · [모든 역할별 통계 CSV](role_statistics.csv) · [유지/최초실패 비교 CSV](first_loss_vs_retained.csv) · [endpoint 자릿수 CSV](per_endpoint_digits.csv) · [집계 및 그림 목록 JSON](summary.json)
