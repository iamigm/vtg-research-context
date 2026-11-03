# Thinking pilot 300 × 입력 순서 2종 — 최종 요약 · pilot300 (300질문)

600/600회 완료. 종료: 2026-10-25T19:54:21+0900. 최초 정상 supervisor 시작부터 약 5.56시간(전환·검증·공유 GPU 벤치마크 포함).

Qwen3-VL-8B-Thinking. 각 데이터셋 100개를 동일한 두 입력 순서로 비교했다. seed=42, temperature=1, top_p=.95, top_k=20, 최대 16,384 생성 토큰.

## 성능과 reasoning 길이

| Dataset | 영상→질문 mIoU(%) | 질문→영상 mIoU(%) | 변화(pp) | 영상→질문 reasoning 평균 | 질문→영상 reasoning 평균 |
|---|---:|---:|---:|---:|---:|
| charades-timelens | 42.36 | 44.04 | +1.68 | 379.9 | 1374.3 |
| activitynet-timelens | 42.15 | 38.98 | -3.17 | 394.8 | 1910.7 |
| qvhighlights-timelens | 61.35 | 60.72 | -0.63 | 400.9 | 1010.6 |
| all | 48.62 | 47.91 | -0.71 | 391.9 | 1431.9 |

질문→영상의 reasoning은 평균 3.65배 길지만, 전체 mIoU는 -0.71pp다. paired bootstrap 95% CI는 [-3.39, 1.91]pp로 입력 순서의 확실한 성능 우열을 주장하기 어렵다. 긴 reasoning이 더 좋은 답을 보장하지 않는다는 관측이며 인과 결론은 아니다.

mIoU는 저장된 최종 답변에서 파싱한 첫 구간과 manifest의 단일 GT의 IoU 평균이다. 파싱 실패는 0점으로 포함했다. 공식 leaderboard 평가와는 구분한다.

## 종료 및 저장 검증

- 600개 기록 모두 기존 저장 검증 통과. 이번 요약에서는 cohort/order 중복·누락, 완료 표시, validation 결과 및 attention 파일 크기를 다시 확인했다. 전체 약 199 GiB의 attention SHA-256을 다시 읽어 검증하지는 않았다.
- QVHighlights 영상→질문 1건(qvhighlights-timelens_00147)은 reasoning 16,384토큰 한도에 도달해 최종 답변이 없다. 나머지 599건은 EOS 및 thinking 종료가 확인됐다.
- 최종 구간 파싱 실패는 총 2건: 위 길이 제한 사례와 Charades 영상→질문 charades-timelens_01482.

- Attention 실제 저장량: 영상→질문 43.41 GiB, 질문→영상 155.65 GiB, 합계 199.06 GiB. HDF5 FP32 + LZF 기준이며 기타 JSON/텍스트·벤치마크 파일은 제외한다.
- 측정 방식: {'live': 415, 'replay': 184, 'replay_saved_generation': 1}. 전환 이전 결과를 덮어쓰지 않았다.

## Attention 산출물

- 상세 분포: L0·7·14·21·28·35 각각의 32개 head 평균. 영상·질문·이전 생성 토큰을 포함한 모든 key를 보존.
- 보조 요약: 전체 36개 레이어 × 32개 head × 7개 key 범주의 attention mass.
- Query는 모든 생성 토큰을 예측하는 직전 위치. 종료 태그와 EOS 행도 포함하며 tokens.json으로 구분한다.
- 현재 보고서는 결과·길이·저장 검증 요약이다. 어느 레이어/head가 영상 근거를 더 참조했는지에 대한 attention 내용 분석은 별도 작업이다.

## 실행 속도

별도 고정 입력 벤치마크에서는 live 추출이 1.55–1.71배 빨랐다. 본 실험의 동일 데이터셋·입력 순서별 비대응 표본 비교에서는 약 1.19–1.24배였다. 표본 길이와 동시 GPU 부하가 달라 두 수치를 혼동하지 않는다. 속도 측정 보고서 원문은 현재 결과 트리에 남아 있지 않다.

## 해석 범위

- Single seed=42 and one rollout per order; no repeated-seed uncertainty.
- Scores reuse the current parser and first predicted span versus single GT; not an official leaderboard evaluation.
- Paired bootstrap is over the fixed pilot pairs; it does not account for sampling-seed variability.
- Attention patterns have been captured and validated, but this summary does not infer visual grounding or causal reliance from them.
