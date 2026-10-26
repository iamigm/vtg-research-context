# 생성 중 attention 추출 속도 비교

완료된 trial: 8/8

동일한 모델·입력·seed·생성 한도·FP32 저장 설정. 원본 4개 작업과 GPU를 공유하므로 독점 GPU 속도나 전체 실험 ETA를 확정하지 않는다. replay/live/live/replay 순서로 시간 변화의 영향을 줄인다. 전처리와 모델 로드 시간은 제외하고 생성·attention 계산·파일 저장·검증을 포함한다.

| 입력 순서 | replay 평균(초) | live 평균(초) | 배속 |
|---|---:|---:|---:|
| video_query | 25.07 | 16.20 | 1.548× |
| query_video | 94.72 | 55.36 | 1.711× |

video_query는 최대 128토큰, query_video는 최대 512토큰으로 제한한 부분 trace 벤치마크다. 전체 답변 길이 실험과 구분한다. 모든 방식은 실제 sampling을 수행하며 원본 token IDs 동일성과 상세/요약 attention 오차를 확인한다.
