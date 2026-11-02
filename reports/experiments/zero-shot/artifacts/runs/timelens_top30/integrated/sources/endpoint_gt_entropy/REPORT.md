# 반대쪽 시각을 GT로 고정한 endpoint별 엔트로피

동일 pilot 300개 × 두 모델의 저장된 greedy 분포를 사용했다. 초 단위/소수 한 자리 범위의 594개 응답에서 시작·종료를 각각 평가했다. MM:SS 등 6개 응답은 명시적으로 제외했다. GPU 재추론은 없다.

- 시작 평가: 예측 시작 시각 + **GT 종료 시각**.
- 종료 평가: **GT 시작 시각** + 예측 종료 시각.

GT는 원래 정밀도 그대로 고정한다. 예측 숫자만 기존 평가와 동일하게 Python round한다. 각 endpoint의 prefix에서 완성 가능한 최대 IoU를 계산하고, 0.5 아래로 처음 내려가는 토큰을 독립적으로 표시한다. 0.3/0.7 기준도 CSV에 포함했다. 반대쪽 시각이 이미 틀렸더라도 이 endpoint가 정답 가능한 상태이면 오류 이후로 분류하지 않는다.

**생성된 시작 시각을 GT로 바꾸어 모델을 다시 실행한 실험은 아니다.** 종료 시각의 엔트로피는 여전히 원래 생성한 시작 시각에 조건화되어 있다. GT 고정은 평가에만 적용하므로 실제 토큰 분포를 그대로 재사용한다.

![시작 시각](start_step_entropy.png)

![종료 시각](end_step_entropy.png)

가로축 0은 해당 endpoint의 최초 가능성 상실 step이고, 점 하나는 평균하지 않은 단일 step의 선택 직전 엔트로피다. 각 그래프에는 그 endpoint에서 실패한 응답만 포함한다. 반대쪽 endpoint의 step은 섞지 않았다. 원은 숫자, ×는 소수점/구분자다.

| 모델 | 벤치마크 | 평가 대상 | N | GT 고정 R@.5 (%) | 평균 절대오차 (초) | 기존 구간 실패→이 endpoint 성공 N | 최초 오류 N (숫자/경계) | 최초 오류 H |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen25 | charades-timelens | start | 100 | 73.0 | 2.830 | 31 | 27 (27/0) | 1.840 |
| qwen25 | activitynet-timelens | start | 100 | 67.0 | 11.830 | 40 | 33 (33/0) | 1.875 |
| qwen25 | qvhighlights-timelens | start | 100 | 82.0 | 8.510 | 58 | 18 (18/0) | 1.698 |
| qwen3 | charades-timelens | start | 100 | 82.0 | 2.230 | 33 | 18 (18/0) | 1.322 |
| qwen3 | activitynet-timelens | start | 96 | 68.8 | 16.990 | 25 | 30 (30/0) | 1.218 |
| qwen3 | qvhighlights-timelens | start | 98 | 74.5 | 13.306 | 12 | 25 (23/2) | 0.967 |
| qwen25 | charades-timelens | end | 100 | 56.0 | 4.420 | 14 | 44 (43/1) | 1.359 |
| qwen25 | activitynet-timelens | end | 100 | 43.0 | 19.440 | 16 | 57 (56/1) | 0.968 |
| qwen25 | qvhighlights-timelens | end | 100 | 45.0 | 19.560 | 21 | 55 (54/1) | 0.999 |
| qwen3 | charades-timelens | end | 100 | 63.0 | 4.230 | 14 | 37 (34/3) | 1.316 |
| qwen3 | activitynet-timelens | end | 96 | 55.2 | 21.427 | 12 | 43 (41/2) | 1.057 |
| qwen3 | qvhighlights-timelens | end | 98 | 77.6 | 13.469 | 15 | 22 (19/3) | 0.850 |

`기존 구간 실패→이 endpoint 성공`은 원래 전체 구간 IoU<0.5이지만 반대쪽을 GT로 고정하면 해당 endpoint가 IoU≥0.5인 응답 수다. 이는 기존 공동 판정에서 가려졌던 endpoint 성능을 보여준다.

실패 step의 H에서 직전 정답 가능 상태의 **마지막 숫자 step** H를 뺀 paired 차이를 summary.json에 저장했다. 직전 숫자 step이 없는 응답은 이 차이에서 제외한다. Bootstrap CI는 query 단위 2,000회 재표집한 탐색적 구간이며, 숫자의 자리·token 종류가 다를 수 있어 인과적 오류 탐지 효과로 해석하지 않는다.

- start: 원래 구간은 실패했지만 반대쪽을 GT로 고정하면 성공하는 응답은 총 199개(두 모델 합산)다.
- end: 원래 구간은 실패했지만 반대쪽을 GT로 고정하면 성공하는 응답은 총 92개(두 모델 합산)다.

판정은 음이 아닌 초 단위 숫자, 정수/소수 한 자리, 선행 0 없음이라는 표기 범위 내에서만 유효하다. 다른 문법, 추가 소수 자리, 수정 문장을 허용하면 회복 가능성이 달라질 수 있다. 표본의 분포는 고정된 실제 greedy 경로의 조건부 분포이며, 가능한 completion이 실제 모델에서 선택될 확률까지 계산한 것은 아니다. 기존 Decord 디코딩 한계도 유지된다.

[endpoint별 CSV](per_endpoint.csv) · [step별 CSV](per_step.csv) · [전체 경로 JSON](trajectories.json) · [집계·검증 JSON](summary.json) · [제외 목록](excluded.csv)
