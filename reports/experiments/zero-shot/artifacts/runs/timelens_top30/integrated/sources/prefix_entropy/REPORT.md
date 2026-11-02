# 정답 가능성을 처음 잃는 생성 단계의 엔트로피

기존 600개 full-head greedy 응답 중 594개를 분석했다. 6개는 초 단위 표기 범위 밖(MM:SS 등)이어서 별도 제외 목록에 기록했다. 재추론은 하지 않았다.

주 기준은 **IoU≥0.5를 만족하는 시간 구간으로 현재 prefix를 완성할 수 있는가**이다. 0.3/0.7 기준도 ../../../artifacts/runs/timelens_top30/prefix_entropy/metrics/per_step.csv와 ../../../artifacts/runs/timelens_top30/prefix_entropy/metadata/trajectories.json에 함께 기록했다. 올바른 GT 숫자 문자열과 일치하는지만 검사하지 않는다. IoU를 만족하는 다른 구간도 정답으로 인정한다.

각 step의 엔트로피는 해당 토큰을 선택하기 **직전**의 raw 전체 어휘 분포다. 생성된 토큰을 prefix에 붙인 뒤 가능한 최대 IoU를 계산한다. 최대 IoU가 처음 0.5 아래로 내려간 토큰을 first loss로 표시한다. 그 이전은 “맞게 생성한 토큰”이 아니라 **정답 가능성을 유지한 토큰**, 그 이후는 **이미 정답 가능성을 잃은 상태**다.

**판정 범위:** 음이 아닌 초 단위 숫자, 정수 또는 소수 한 자리, 불필요한 선행 0 없음, 시작−끝 두 구간 형식. delimiter가 나오면 그 숫자를 확정한다. 기존 평가처럼 Python round 후 첫 구간 IoU를 계산한다. GT 길이와 최저 threshold 0.3에서 도출한 충분히 큰 상한까지 열거하며 영상 duration으로 예측을 잘라내지 않는다. 두 끝점이 모두 확정된 경우 저장된 IoU와 일치함을 전 샘플에서 검증했다.

**이것은 이 표기 범위 안에서의 prefix 가능성 판정이지 토큰의 실제 인과 효과 추정이 아니다.** 이후 더 많은 소수 자리, 다른 시간 형식, 수정 문장을 허용하면 회복 가능성이 달라질 수 있다. 또한 가능한 completion이 존재해도 모델이 실제로 그 completion을 생성할 확률을 추정한 것은 아니다.

![평균하지 않은 step별 엔트로피](../../../artifacts/runs/timelens_top30/prefix_entropy/figures/step_entropy_aligned.png)

가로축 0은 최초 가능성 상실 step, 음수/양수는 그 전/후의 실제 생성 step 거리다. 점 하나는 단일 step 엔트로피이며 시간축으로 평균하지 않았다. 원은 숫자, ×는 구두점/경계이며 가는 선은 같은 query를 잇는다. 아래 표와 [구간 평균 보조 그림](../../../artifacts/runs/timelens_top30/prefix_entropy/figures/prefix_entropy.png)은 요약용이다.

| 모델 | 벤치마크 | 분석 N | first loss N (숫자/경계) | 이전 숫자 H | first loss H | 이후 숫자 H | 성공 응답 숫자 H |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen25 | charades-timelens | 100 | 58 (57/1) | 1.258 | 1.583 | 1.640 | 1.490 |
| qwen25 | activitynet-timelens | 100 | 73 (72/1) | 1.551 | 1.462 | 1.555 | 1.470 |
| qwen25 | qvhighlights-timelens | 100 | 76 (75/1) | 1.264 | 1.331 | 1.884 | 1.449 |
| qwen3 | charades-timelens | 100 | 51 (49/2) | 0.974 | 1.231 | 1.597 | 1.167 |
| qwen3 | activitynet-timelens | 96 | 55 (54/1) | 1.177 | 1.174 | 1.537 | 1.046 |
| qwen3 | qvhighlights-timelens | 98 | 37 (33/4) | 0.858 | 0.938 | 1.418 | 0.996 |

first loss는 샘플당 해당 단일 step의 H다. 이전/이후/성공 그룹은 해당 구간 숫자 step만 query 내부 평균한 보조 요약이며 원래 전체 숫자 단계 평균과 다르다. 숫자가 끝났음을 확정하는 구분자 때문에 가능성이 사라질 수 있어 경계 step도 포함하고 숫자/경계 수를 분리했다. 서로 다른 token 종류·자리수의 엔트로피 수준이 다르므로 평균 차이만으로 오류 탐지 성능을 주장하지 않는다. 원시 step별 결과를 함께 제공한다.

다음 예시는 first loss가 발생한 토큰 주변을 보여준다. 가능한 최대 IoU는 oracle이며 실제 출력 IoU와 구분해야 한다.

### qwen25 · charades-timelens_00027

GT=[29, 34], 최종 IoU=0.357; 답변: `The event happens in 21.9 - 35.7 seconds.`

| Step | Token | H (nat) | 최대 IoU 전→후 | 구분 |
|---:|---|---:|---|---|
| 5 | `␠` | 0.000 | 1.000 → 1.000 | retains_feasibility |
| 6 | `2` | 0.825 | 1.000 → 1.000 | retains_feasibility |
| 7 | `1` | 2.261 | 1.000 → 0.417 | first_loss |
| 8 | `.` | 0.003 | 0.417 → 0.417 | after_loss |
| 9 | `9` | 2.301 | 0.417 → 0.417 | after_loss |
### qwen25 · charades-timelens_00103

GT=[28, 29], 최종 IoU=0.200; 답변: `The event happens in 25.7 - 31.4 seconds.`

| Step | Token | H (nat) | 최대 IoU 전→후 | 구분 |
|---:|---|---:|---|---|
| 5 | `␠` | 0.000 | 1.000 → 1.000 | retains_feasibility |
| 6 | `2` | 0.477 | 1.000 → 1.000 | retains_feasibility |
| 7 | `5` | 2.127 | 1.000 → 0.333 | first_loss |
| 8 | `.` | 0.000 | 0.333 → 0.333 | after_loss |
| 9 | `7` | 2.300 | 0.333 → 0.333 | after_loss |
### qwen3 · charades-timelens_00027

GT=[29, 34], 최종 IoU=0.455; 답변: `The event happens in 25.0 - 36.0 seconds.`

| Step | Token | H (nat) | 최대 IoU 전→후 | 구분 |
|---:|---|---:|---|---|
| 11 | `␠` | 0.000 | 0.556 → 0.556 | retains_feasibility |
| 12 | `3` | 0.033 | 0.556 → 0.556 | retains_feasibility |
| 13 | `6` | 1.726 | 0.556 → 0.455 | first_loss |
| 14 | `.` | 0.000 | 0.455 → 0.455 | after_loss |
| 15 | `0` | 0.001 | 0.455 → 0.455 | after_loss |
### qwen3 · charades-timelens_00103

GT=[28, 29], 최종 IoU=0.250; 답변: `The event happens in 26.0 - 30.0 seconds.`

| Step | Token | H (nat) | 최대 IoU 전→후 | 구분 |
|---:|---|---:|---|---|
| 7 | `6` | 1.559 | 1.000 → 0.500 | retains_feasibility |
| 8 | `.` | 0.000 | 0.500 → 0.500 | retains_feasibility |
| 9 | `0` | 2.292 | 0.500 → 0.333 | first_loss |
| 10 | `␠-` | 0.000 | 0.333 → 0.333 | after_loss |
| 11 | `␠` | 0.000 | 0.333 → 0.333 | after_loss |

[모든 step CSV](../../../artifacts/runs/timelens_top30/prefix_entropy/metrics/per_step.csv) · [모든 생성 경로 JSON](../../../artifacts/runs/timelens_top30/prefix_entropy/metadata/trajectories.json) · [집계/threshold/검증 JSON](../../../artifacts/runs/timelens_top30/prefix_entropy/metrics/summary.json) · [제외 목록](../../../artifacts/runs/timelens_top30/prefix_entropy/metrics/excluded.csv)
