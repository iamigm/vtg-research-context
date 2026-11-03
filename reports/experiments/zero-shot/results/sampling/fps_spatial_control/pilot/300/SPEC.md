# 공간 해상도 통제 zero-shot FPS sweep · pilot300 (300질문)

기존 고정 pilot300(Charades/ActivityNet/QVHighlights 각100) × Qwen3-VL-8B Instruct/Thinking × 영상→질문/질문→영상 × FPS2/3/4/6/8 =6,000개 응답이다. 2fps도 새 입력으로 다시 생성한다.

영상마다 목표100,352픽셀 ±10% 범위에서 원본 종횡비 오차가 가장 작은32배수 H/W를 정한다. 동률은 면적오차,H,W 순이다. 모든 FPS·모델·순서에서 해당영상의 H/W를 고정한다. 영상 간 종횡비와 면적은 허용범위에서 달라도 된다. 원본 RGB에서 직접 bicubic antialias resize하며 전체영상 토큰 예산으로 자동 축소하지 않는다. 해상도를FPS별로 바꾸거나 context에 맞추어 프레임을 줄이지 않는다. context초과/OOM은 명시적 실패로 기록한다.

원본 PTS에서0부터k/FPS 시각에 가장 가까운프레임을 선택하고 동률은 앞프레임을 택한다. 중복을 제거하며 전체영상을 사용한다. 768프레임 상한은 없다. 시간패치2 묶음에 필요한 경우 마지막프레임만 복제하고 별도로 기록한다. 원본FPS가 낮을 때의 실효FPS도 보고한다. GT/예측/성능은 샘플링과 공간크기 결정 및 모델입력에 사용하지 않는다. 모델에는 영상과질문만 주며 학습·추가예시·추가rollout은 없다.

기존 모델별 생성설정을 유지한다. Instruct greedy/max512, Thinking sampling T1/top_p.95/top_k20/seed42/max16384, repetition_penalty1, KVcache, BF16/FlashAttention2. 모델 간 비교는 생성정책 차이를 포함하며 FPS 효과는 같은 모델·입력순서 안에서 비교한다. 각응답 seed42를 재설정한다. 체크포인트·processor는 기존 공유저장소를 로컬로 읽고 해시를 기록한다.

기존 spatial_sampling_pilot900_v2가 정상complete이고 스모크가passed일 때 대기supervisor가 자동실행한다. 실패한 선행작업을 완료로 간주하지 않는다. 실제추론은 영상SHA256 mod2로 겹침없는 두프로세스에 배정하고 각batch1, CPU전처리각1/모델Torch각2로 병행한다. CPUquota12·메모리160GiB 내에서 운영한다. 환경전체재시작을 견딘다고 주장하지 않는다.

완료후 입력·픽셀·PTS·H/W·예측출처를 검증하고 mIoU/R@.3/.5/.7 실제점수와2fps대비pp, 벤치마크별/영상길이별/GT비율별/두축교차표 및 실패율을 MD로 자동생성한다. 개수는질문 기준이며 코호트는기존 진단표본이다. 600/900확장은이번범위에 포함하지 않는다. 스모크의최대2생성토큰 응답은실험결과에포함하지 않는다.

[기존 해상도 변화 감사](SPATIAL_AUDIT.md) · [상태](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/metadata/status.json) · [로그](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/logs/supervisor.log) · [전체 실험 목록](../../../../../../index.html)

## 2026-11-03 재개 수정

RAM 감시는 회수 가능한 clean file cache와 slab cache를 실제 상주 메모리 압박에서 분리한다. 익명 메모리110GiB 초과 또는 회수 불가능 메모리의 한도92% 초과가15초 지속될 때 해당 실행만 중지한다. 전체RAM 사용량도 함께 기록한다.

292개 영상의 PTS를 모두 검사했다. ActivityNet10개 영상에서 같은PTS가 한쌍씩 있으나 시각 역전은 없다. 동일PTS에서는 원본index가 앞선 프레임을 선택하며 타임스탬프를 변경하거나 프레임을 임의의 다른 시각에 배치하지 않는다. 모든FPS에 같은 규칙을 쓴다. 이미 준비된97개 영상의 샘플링 계획과 저장된580개 응답·입력은 변경하지 않는다. 원래 코드와 수정 코드의 출처를 각각 보존한다.

[전체 계획·기존 응답 검증](../../../../../artifacts/runs/sampling/fps_spatial_control/pilot/300/validation/recovery_20261103.json)
