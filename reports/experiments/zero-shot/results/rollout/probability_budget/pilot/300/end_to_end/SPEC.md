# End-to-end completion · pilot300 · 실행 명세

기존 probability budget의 후보를 그대로 사용한다. 후보 생성의 512·1,024·2,048 checkpoint와 LevinTS·native mass·numeric novelty를 구분한다. 고정 pilot300을 모두 처리하고 보고서·viewer까지 검증한 뒤 나머지600의 후속 계산을 수행한다. 기존 탐색은 재실행하거나 GT로 수정하지 않는다.

## 독립 구간 검증과 최종 생성

Qwen2.5-VL-7B-Instruct가 모든 고유 후보 구간을 독립적으로 다시 본다. 제안 모델인 Qwen3-VL-8B-Instruct와 가중치가 다른 모델이다. 검증 입력은 해당 crop과 query뿐이다. 다른 후보·원래 path probability·GT를 넣지 않는다. Native next-token의 0=다른 사건/없음, 1=부분적·경계 부정확, 2=완전한 사건·적절한 경계 확률을 측정한다. score=(p1+2p2)/2이며 p는 세 label 안의 조건부 확률이다. 원래 세 label의 전체 vocabulary 질량도 저장한다. 점수가 높을수록 이 prompt에서 모델이 긍정적으로 평가했다는 뜻이며 정답 확률의 보정값이 아니다.

각 조건에 대해 검증 점수 최대 후보, 후보만 제공한 Qwen3-VL의 최종 재생성(proposal_only), 검증 점수도 제공한 최종 재생성(reviewed)을 평가한다. 원래 full video/query 입력·tensor hash·greedy 답변은 동일하게 공유하고 매 prompt는 저장한다. Semantic cosine은 어느 선택에도 쓰지 않는다. Final answer는 후보의 경계를 수정할 수 있으므로 candidate oracle은 final regeneration의 엄밀한 상한이 아니다. EOS 없는 생성·parsing 실패는 유효 결과로 바꾸지 않고 IoU0으로 평가한다. 생성은 greedy·최대64 token이며 [0,실제 영상 길이] clipping을 적용한다.

## Crop과 연산 재사용

원래2 FPS frame sampling에서 구간 내부의 프레임을 모두 사용한다. 내부 프레임이 없으면 원본 영상에서 구간 중간 시점에 실제 표시되는 프레임을 추출한다. 한 프레임은 Qwen의 시간축 병합을 위해 두 번 넣으며 이 경우를 기록한다. Crop timestamp는 첫 프레임부터의 상대 시각으로 바꾼다. 같은 frame IDs·상대 timestamp를 갖는 입력은 query 안에서 한 번 계산한다. 이는 서로 다른 사건이라는 판단이나 decoding tree pruning이 아니다. Crop의 실제 시간 해상도 때문에 서로 다른 숫자가 같은 시각 입력이 될 수 있다.

Qwen3의 원래 H×W를 보존한다. Qwen2.5 verifier에는 원래 면적과 가까운28 배수 H×W로 resize한다. 프레임 수를 줄이지 않는다. Batch의 최대 padded 입력은16,384 token·최대8 clip이다. 단일 긴 clip은 자르지 않고 단독 처리한다. 첫 실제 batch는 serial 결과와 대조한다. 검증 확률의 최대 절대 차이가1%p 이상이거나 semantic cosine이0.999 이하이면 해당 모델은 serial native 실행으로 전환하고, 검사와 버린 batch의 계산 비용도 기록한다. 같은 최종 prompt는 해시로 한 번 생성한다. 본 실험 안의 검증 통과 여부만 남기고 별도 smoke-test 실험은 만들지 않는다.

## Query-free semantic 측정

Qwen3-VL-8B-Instruct의 마지막 layer에서 video token만 mean pooling하고 L2 정규화한다. 공간 격자 크기가 일정하므로 시간별 격자에 같은 가중치를 주는 frame-balanced mean과 같다. 마지막 nonpadding token pooling도 같은 forward에서 보조 측정한다. Qwen3-VL-Embedding-8B는 공식 구현과 같은 generic instruction·chat template·마지막 nonpadding token pooling을 사용한다. 이 tokenizer에서 마지막 token이 문자 그대로 EOS인지는 따로 가정하지 않는다. 임의의 EOS를 추가하거나8192 token으로 자르지 않는다. 두 모델 모두 query·GT 없이 crop만 받는다.

실제 벡터는 query 처리 중 RAM에만 두고 폐기한다. 모델별 scalar 통계를 계산한 즉시 checkpoint로 저장하며 재개할 때 끝난 모델은 건너뛴다. 후보가0개면 비교를 만들지 않고,1개면 서로 다른 두 구간의 cosine 통계는 n=0·미정의로 남긴다. 이를 cosine0으로 대체하지 않는다. 동일 모델 안의 cosine에 대해 native-mass 가중 mean·std·10/50/90% quantile, temporal IoU 구간별 통계, 시간 그룹 내부/사이 통계, greedy와의 유사도, 가장 다른12쌍을 저장한다. 서로 다른 모델의 벡터를 직접 비교하지 않는다. Cosine이 높다는 사실만으로 같은 사건이라고 확정하지 않는다.

## 시간 분포와 분기별 불확실성

Entropy와 covariance는 원래 exact clipped(start,end) atom에서 계산한다. KDE를 위한 시간 격자나 entropy 재binning을 추가하지 않는다. 시간 그룹은 endpoint 공간의 Gaussian KDE(h=1초)와8-nearest support graph(거리≤4h)로 구한다. Edge saddle은 두 끝과 중간의 밀도 중 최솟값이다. Saddle÷두 peak 중 낮은 밀도≥0.5이면 합친다. 관측 support graph로 근사한 통계적 mode이며 별개 semantic 사건의 GT가 아니다. h=0.5·2초에서 그룹 수의 민감도도 함께 기록한다.

각 실제 numeric fork에서 완성된 유효 descendant의 질량을 모아 최종 interval entropy(bits), start/end population covariance(s²), 시간 그룹 entropy를 계산한다. 큰 interval entropy는 끝까지 이어간 유효 후보가 여러 숫자 구간에 퍼졌다는 뜻이다. 분기 token과 시간 그룹의 관측 MI=H(C)−ΣP(a|관측 valid)H(C|a)도 저장한다. 미탐색·무효 부분을 채운 전체 분포의 정확한 entropy/MI라고 부르지 않는다. 소수 자릿수를 새로운 branch로 만들거나 fractional branch entropy를 측정하지 않는다.

## 평가와 비용

IoU=교집합 길이÷합집합 길이, mIoU=query별 IoU 평균, R@τ=IoU≥τ 비율이며 높을수록 좋다. Δpp=100×(해당 결과−같은 query의 greedy). Changed는 clipped span이 정확히 달라진 query 비율이며 개선을 뜻하지 않는다. Invalid final answer는 IoU0으로 포함한다. GT는 이 평가에서만 읽는다.

검증·Instruct semantic·Embedding semantic·최종 생성·CPU 통계의 비용을 분리한다. 실제 unique batch forward의 nonpadding/padded tokens·CUDA·wall을 기록한다. Clip별 amortized 시간은 batch 시간을 입력 token 비율로 나눈 회계값이며 clip 단독 latency나 FLOPs가 아니다. 조건별 final generation 비용은 고정 실행 순서에서 실제 소비한 incremental 연산이며 동일 prompt 재사용을 독립 실행처럼 합산하지 않는다. 같은 GPU에서 기존 proposal search와 실행될 수 있으므로 wall·CUDA event 시간에는 자원 경합이 포함된다.

## 참고한 공개 구현

- [VideoMind](https://github.com/yeliudev/VideoMind): localize, verify isolated moments, synthesize; no trained LoRA or paper performance reproduced.
- [VideoAgent](https://github.com/wxh1996/VideoAgent): visual evidence inspection before answering; no API, caption pipeline, or reported results reproduced.
- [Qwen3-VL-Embedding official implementation](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py): query-free instruction and last-nonpadding pooling; same local model and processor, no silent truncation.
