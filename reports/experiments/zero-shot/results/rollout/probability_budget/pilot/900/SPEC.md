# Probability budget · pilot900 · 실행 명세

고정 cohort는 기존 nested pilot900의 정확한 prefix다. Pilot300 전체의 추론·보고서·viewer·원자료 검증을 마친 뒤 누적900으로 진행한다. 이 실행은 candidate proposal 비교이며 independent verifier의 최종 답 선택 실험이 아니다. 기존 3-query 시간 제한 진단과 별도 계열이다.

모델은 Qwen3-VL-8B-Instruct, native temperature1, sampling 없음, seed 반복 없음이다. 입력은 기존 deterministic의2 FPS·video_query·frame IDs·PTS·H×W·tensor hash를 그대로 대조한다. 영상 FPS·frame 수·해상도를 가속 목적으로 줄이지 않는다.

매 step에서 native top-p=0.99 nucleus와 native 확률≥1%의 교집합을 만든다. 1% 미만은 nucleus를 채우기 위해 추가하지 않는다. 정수부 숫자와 소수점 기호에서만 대안을 만들고 일반 문구·소수 자릿수는 greedy다. 소수 자릿수의 branch entropy는 계산하지 않는다. 원자 token이 소수점을 가로지르면 정수부/점의 효과가 같은 fractional-only 대안은 최고 native 확률 token만 유지한다. 전체 vocabulary 확률을 sibling 사이에서 재정규화하지 않는다.

Depth는 이미 생성한 답변 token 수이며, branch 횟수나 미래 답변 길이가 아니다. Greedy 문구·소수 자릿수도 depth에는 포함한다. Video/query 입력 token은 제외한다. Prefix마다 native log probability를 누적한다.

- **levin**: 전역 frontier의 log(depth)−log(native path probability)가 최소인 node를 선택한다. Greedy 위치에서도 다음 prefix를 다른 frontier와 다시 비교한다. 원래 LevinTS의 첫 목표 종료를 다중 후보 열거로 바꾼 적용이다. 목표 tree는 이번 token/분기 정책을 통과한 tree다.
- **native_mass**: 첫 실제 numeric fork의 자식들로 서로 겹치지 않는 고정 subtree 그룹을 만든다. 그룹별 미탐색 native 질량 U에 비례해 실제 decoder 입력 연산을 배분한다. Credit은 소비한 실제 연산으로 갱신하고 quantum32에서 다음 그룹을 선택한다. 그룹 내에서는 최고 native prefix를 골라 greedy continuation을 완성한다. 초기 fork 이전의 비용도 전체 예산에 포함한다.
- **numeric_novelty**: 같은 scheduler에 w_i∝U_i(1+0.1q_i)를 적용한다. q_i는 그 그룹에서 완성한 유효 구간들의 native-mass 가중 평균 novelty다. 각 novelty=1−다른 그룹의 완성 구간과의 최대 temporal IoU다. 자기 그룹과 비교하지 않는다. 비교 근거가 없으면 보정0이며, unknown을 무효로 취급하지 않는다. 목표 weight의 상대 odds 보정은 최대1.1배다. 실제 연산 배분은 job·cache 단위여서 그 비율과 정확히 일치하지 않을 수 있다. 이 방법은 DORA의 배분 틀을 참고한 numeric variant이며 원래 PRM·embedding 기반 DORA의 재현이나 최적성 보장을 주장하지 않는다.

같은 interval에 도달한 서로 다른 sequence의 native 질량은 합친다. 후보 개수를 예산 가중치로 쓰거나 진짜 확률 질량을 중복 개수로 나누지 않는다. 다른 token prefix는 같은 interval이어도 모델 상태를 합치지 않는다. Semantic embedding·GT·신경망 reward는 배분에 사용하지 않는다. Temporal novelty는 사건 의미나 정답 확률이 아니다.

각 query×방법의 decoder 입력 상한은2,048이며512·1,024·2,048 checkpoint를 한 누적 search에서 기록한다. 샘플·방법 사이로 잔여 예산을 이월하지 않는다. Prefix replay도 상한에 포함하며 model forward 직전에 검사하므로 미완성 prefix는 pending으로 남긴다. 새 video/query prefill은 검색 상한 밖의 공유 비용으로 별도 기록한다. Greedy reference도 별도 연산으로 측정한다. 시간 제한으로 후보를 자르지 않는다.

V=완성된 유효 경로의 native 질량, U=아직 pending인 frontier 질량+context capacity 때문에 미완성인 질량이다. U/(V+U)≤0.01이면 조기 종료한다. 이 상한은 허용된 분기 정책 안의 최종 유효 분포에 관한 것이며 full model 분포 전체의99%를 보장하지 않는다. 필터 제외·정책 제외·무효 질량을 각각 보존한다. Context limit 질량을 유효/무효로 확정하지 않는다.

영상/query KV는 query당 한 번 공유한다. 계산한 answer prefix의 text-only KV suffix를 최대8GiB에서 LRU 재사용하며 다른 prefix의 상태를 근사하지 않는다. Smoke에서 복원 결과와 fresh prefill의 확률·admitted token을 대조한다. Cache의 저장·복원 비용과 eviction·재실행 token도 측정한다. Node expansion 보장은 GPU 비용 보장과 다르다.

완성 interval은[0,actual metadata duration]으로 clipping하고 start<end와 EOS를 요구한다. 뒤집힌 구간을 정렬하지 않는다. Interval entropy=−Σp log₂p는 exact clipped (start,end) atom의 관측 유효 질량 조건부 분포에서 계산한다. 추가 entropy binning은 없다. 시작·끝 covariance는 같은 가중 분포의 population covariance이며 단위는s²다. 미탐색 부분이 있는 통계를 전체 분포의 정확한 통계로 부르지 않는다.

[LevinTS](https://proceedings.neurips.cc/paper/2018/file/52c5189391854c93e8a0e1326e56c14f-Paper.pdf), [PHS의 sub-probability 정의](https://arxiv.org/html/2103.11505), [DORA](https://arxiv.org/html/2506.15707v1)가 방법 참고다.
