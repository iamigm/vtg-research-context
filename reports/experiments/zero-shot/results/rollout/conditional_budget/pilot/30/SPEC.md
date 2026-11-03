# 시작 prefix 조건부 종료 탐색 예산 측정

사용자 요청에 따라 시작과 종료를 독립으로 취급하지 않고, 정확한 시작 토큰 prefix에 조건부인 종료 분포를 결정적으로 탐색한다. stochastic sampling은 사용하지 않는다.

## 실행 범위

- 기존 고정 pilot300 순서에서 데이터셋별 첫10개, 총30개 질문의 진단. 코호트 원본은 변경하지 않는다. 전체 벤치마크 평균으로 일반화하지 않는다.
- 로컬 Qwen3-VL-8B-Instruct, video_query, 2 FPS. 기존 fps_spatial_control_pilot300_v1의 목표100,352픽셀·고정 H/W 입력과 프레임 PTS를 재사용한다. 모든 분기에서 입력은 동일하다.
- 기존 FPS 실행이 state/stage 모두 complete이고 GPU 여유60GiB 이상일 때 모델을 로드한다. 다른 작업은 중단하지 않는다. 실패를 완료로 간주하지 않는다. 2026-11-03 사용자가 GPU를 비우고 즉시 실행을 요청하여 resource_release.json에 운영 예외를 기록했다. 이 경우 선행 실행 완료 조건만 면제하고 GPU 여유 조건은 유지한다.
- greedy로 출력한 preamble의 첫 숫자 직전까지를 anchor로 고정한다. 보고하는 확률은 이 anchor에 조건부이며, preamble 생성의 전체 확률까지 포함하는 것은 아니다.

## 확률과 예산

시작 시각을 확정한 정확한 토큰 prefix를 z_i, 그 이후 종료 출력의 확률을 P(e|z_i,x)라 하면 구간 경로 질량은 P(z_i|x)P(e|z_i,x)이다. 같은 숫자 시작이라도 표기·토큰 prefix가 다르면 별도 조건부 분포로 유지한다. 토큰이 delimiter와 종료 숫자 일부를 함께 담으면 boundary_spanning으로 표시한다.

숫자는 음이 아닌 정수 또는 소수 한 자리, 형식은 `S - E seconds`로 제한한다. 허용 정수 자릿수는 영상 길이의 자릿수 이상·최소4자리이고 suffix는 최대32토큰이다. 문법 밖 확률은 원래 full-vocabulary softmax에서 측정해 별도 rejected mass로 보존하며 재정규화하지 않는다. 어휘의 실제 토큰 조각을 이어 붙여 검사한다. 미탐색 subtree, 문법 밖 질량, 토큰 길이 상한 질량을 구분한다.

- joint: 미탐색 prefix 중 raw 경로확률이 가장 큰 것을 먼저 확장한다.
- balanced: 같은 시작 아래 반복 확장을 완화하기 위해 경로확률/(해당 시작의 기존 종료 확장수+1)을 우선순위로 사용한다. 모든 시작의 확장을 보장한다는 뜻은 아니다.
- 두 정책 각각 고유 prefix 확장8/16/32/64/128/256/512개에서 기록한다. 이전 확장과 동일한 prefix를 중복 확장하지 않는다.
- 한 정책당 영상 prefill은 한 번이다. KV cache를 공통 prefix까지 되돌리고 필요한 suffix만 재계산한다. 각 정책은 별도 evaluator로 비용을 측정한다.
- 예산은 시작 확장수, 종료 확장수, 실제 decoder 재계산 토큰 수, forward 호출 수, prefill 시간, 경과시간으로 나눈다.

확보한 disjoint 시작-prefix 질량 합 S, 종료까지 완성한 구간 경로 질량 합 J에 대해 `1−J=(1−S)+(S−J)`를 보고한다. 1−S는 시작-prefix로 확보하지 못한 질량(문법 밖 포함), S−J는 확보한 시작 아래 미완성 종료 질량이다. 전체 트리의 `완성+frontier+문법밖+길이상한=1`을 검증한다. 부분 subtree의 대표 결과를 완성 구간 질량으로 계산하지 않는다.

## 조건부 분포 변화

각 시작별 관측 종료 질량과 90%를 덮는 관측 후보 수를 기록한다. 미탐색 꼬리는 제거하지 않는다. 관측 질량이90% 미만이면 K90은 unresolved이고, 도달해도 최소 후보 수 확정치가 아닌 상한이다.

서로 다른 숫자 시작 중 확률이 가장 큰 두 prefix의 종료 분포를 TV 거리로 비교한다. `TV=0.5 Σ|p−q|`, 클수록 분포 차이가 크다. 관측한 부분분포의 L1 거리를 d, 양쪽 미탐색 질량의 합을 r이라 하면 `[max(0,(d−r)/2), min(1,(d+r)/2)]`를 보수적 범위로 보고한다. 이는 모델의 조건부 생성 차이이며 실제 사건 구분 성공을 뜻하지 않는다.

## 검증과 완료 후 처리

- CPU 합성 분포에서 서로 다른 시작에 대한 종료 조건부 분포, 확률 보존, 무작위 호출 부재를 검증했다.
- GPU 추론 첫 질문에서 서로 다른 숫자 분기로 cache rollback한 분포를 새 multimodal prefill 뒤 동일한 단일토큰 replay 결과와 비교한다. top1 일치·확률 L1 차이0.001 미만을 요구하고 실패하면 실행을 중지한다. suffix는 항상 한 토큰씩 재계산해 탐색 순서에 따른 BF16 batch-shape 차이를 통제한다. 전체 텍스트를 한 번에 prefill한 계산과의 BF16 확률 차이도 별도 기록한다.
- 완료 후 REPORT.md, budget.csv, conditional_end.json, start_end_dependence.json을 자동 생성하고 질량 보존을 다시 검증한다. 통합 목록 생성기도 실행한다.
- oracle IoU는 후보 생성의 진단 상한이다. GT는 후보 확장·우선순위·입력 구성에 사용하지 않는다. mIoU 및 IoU≥.3/.5/.7 성공률과 형식/시간 범위 유효성을 구분한다.
- 최대 예산에서 J≥.90/.95를 달성한 질문 수와 도달 비용을 보고한다. 미달이면 충분한 예산을 확보했다고 표현하지 않는다. 이 30개 결과로 전체 pilot 예산을 확정하지 않는다.

## 경로

- [상태](../../../../../artifacts/runs/rollout/conditional_budget/pilot/30/metadata/status.json)
- [실행 로그](../../../../../artifacts/runs/rollout/conditional_budget/pilot/30/logs/runner.log)
- [CPU 검증](../../../../../artifacts/runs/rollout/conditional_budget/pilot/30/validation/synthetic_joint_tree.json)
- [완료 후 보고서](REPORT.md)
- [실행 코드](../../../../../_lib/conditional_rollout_budget.py)

프로세스는 detached session으로 실행한다. 채팅 종료 뒤 계속 실행되지만 환경 전체 재시작 후 자동 복구까지 보장하지 않는다.
