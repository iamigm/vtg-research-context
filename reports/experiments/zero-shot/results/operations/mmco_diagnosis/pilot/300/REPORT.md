# H.264 MMCO 경고 원인 진단 · pilot300 (300질문)

진행: 292/292 영상 (100.0%). 전체 검사 완료.

`mmco: unref short failure`는 H.264가 해제 또는 장기 참조로 전환하려는 단기 참조 프레임을 decoder buffer에서 찾지 못했을 때 발생한다. FFmpeg의 `find_short` 실패 분기에서 출력된다. [FFmpeg 소스](https://ffmpeg.org/doxygen/7.1/h264__refs_8c_source.html).

현재 실험은 Decord CPU VideoReader(num_threads=1)로 영상을 읽은 뒤 GPU 모델에 전달한다. 이 메시지는 LM head나 CUDA 연산에서 발생하지 않는다. 다중 DataLoader의 사전 읽기 때문에 합쳐진 추론 로그의 직전/직후 sample ID만으로 원인 영상을 특정할 수 없다. 이 검사는 영상별 별도 프로세스와 stderr로 분리한다.

- Decord sparse get_batch에서 MMCO 경고가 재현된 영상: 55개.
- 같은 영상의 처음부터 순차 읽기에서 Decord·FFmpeg 모두 MMCO 경고가 사라지고 정상 종료: 55개.
- sparse와 순차 Decord의 표본 RGB SHA-256이 다른 영상: 9개.

순차 읽기는 정상이고 sparse에서만 발생한다면 해당 파일과 Decord의 프레임 탐색 경로 사이의 상호작용이 원인이라는 근거다. 순차 FFmpeg에서도 발생하면 원본 bitstream/decoder 호환성 문제를 추가 조사해야 하며, 이 경고만으로 원본 손상을 단정하지 않는다. Frame hash 비교는 실제 표본 픽셀 영향 여부를 확인한다.

원본 영상·실험 decoder·모델 입력은 수정하지 않았다. 검사는 저장된 Qwen3 zero-shot metadata의 실제 frame indices를 사용한다. Qwen2.5의 모든 입력 픽셀이나 모든 가능한 탐색 패턴까지 검증한 것은 아니다. Hash가 같아도 순차 decoder 자체가 원본을 올바르게 복원했는지까지 보증하지 않는다.

| 영상 | Sparse MMCO | 순차 Decord MMCO | 순차 FFmpeg MMCO | 표본 RGB 불일치 |
|---|---:|---:|---:|---:|
| v_lCX7y_KAihU.mp4 | 1 | 0 | 0 | 1 |
| v_n2wq_9TeNYM.mp4 | 2 | 0 | 0 | 0 |
| v_0PmrImNqA2w.mp4 | 1 | 0 | 0 | 0 |
| v_Lf3oTCD4d08.mp4 | 2 | 0 | 0 | 2 |
| v_KbEoaYhMZ6c.mp4 | 2 | 0 | 0 | 0 |
| v_tCQiu-qY9XA.mp4 | 2 | 0 | 0 | 0 |
| v_dDN37ufNu84.mp4 | 1 | 0 | 0 | 0 |
| v_aOTtBZynDOQ.mp4 | 1 | 0 | 0 | 0 |
| v_-_gDSRlC1kg.mp4 | 2 | 0 | 0 | 0 |
| v_VFqkLp5mzBM.mp4 | 4 | 0 | 0 | 0 |
| v_aS0wGPhD48o.mp4 | 2 | 0 | 0 | 0 |
| v_mFDC1CLt6B4.mp4 | 3 | 0 | 0 | 4 |
| v_BfLrltipDDU.mp4 | 4 | 0 | 0 | 0 |
| v_yPA6klGWEsc.mp4 | 5 | 0 | 0 | 0 |
| v_aDWrPrNFdR0.mp4 | 6 | 0 | 0 | 0 |
| v_yFPxSn69pcc.mp4 | 3 | 0 | 0 | 0 |
| v_U_ia-tINzpw.mp4 | 7 | 0 | 0 | 0 |
| v_mHVmDOxtVt0.mp4 | 1 | 0 | 0 | 0 |
| v_U0d68z5HTwE.mp4 | 4 | 0 | 0 | 0 |
| v_U36rsW_WhUA.mp4 | 1 | 0 | 0 | 0 |
| v_vy91mJTl7rQ.mp4 | 1 | 0 | 0 | 0 |
| v_LIJBolW8k5o.mp4 | 4 | 0 | 0 | 0 |
| v_XyQSmMYbP6o.mp4 | 5 | 0 | 0 | 0 |
| v_s5y4xXcphcc.mp4 | 7 | 0 | 0 | 12 |
| v_Cu-p0FZOqi8.mp4 | 4 | 0 | 0 | 0 |
| v_P_zz379qSuo.mp4 | 2 | 0 | 0 | 0 |
| v_VwmYoF9Rh_8.mp4 | 1 | 0 | 0 | 0 |
| v_rQZIJBinOsw.mp4 | 3 | 0 | 0 | 0 |
| v_aNE5ZWD5E34.mp4 | 7 | 0 | 0 | 0 |
| v_kbe4iowYMqM.mp4 | 3 | 0 | 0 | 0 |
| v_yDH9iAn82Q8.mp4 | 5 | 0 | 0 | 0 |
| v_l7gWFOa7FnI.mp4 | 3 | 0 | 0 | 0 |
| v_9qVcdqGeAzE.mp4 | 5 | 0 | 0 | 0 |
| v__QTQEw1b_-U.mp4 | 2 | 0 | 0 | 0 |
| v_WPVb8fYLFUM.mp4 | 2 | 0 | 0 | 1 |
| v_3TxZTZEEg44.mp4 | 1 | 0 | 0 | 0 |
| v_VLjfzOpn-AQ.mp4 | 6 | 0 | 0 | 0 |
| v_IjwOh2YmT9U.mp4 | 5 | 0 | 0 | 1 |
| v_K0MzjnMzbj4.mp4 | 2 | 0 | 0 | 0 |
| v_7-_Nur_xiV4.mp4 | 5 | 0 | 0 | 0 |
| v_GI8tylrKKlA.mp4 | 5 | 0 | 0 | 38 |
| v_immCYvN8pwQ.mp4 | 4 | 0 | 0 | 0 |
| v_wLKePf07V14.mp4 | 4 | 0 | 0 | 0 |
| v_5_M10vevgJg.mp4 | 4 | 0 | 0 | 0 |
| v_3j2d27w3x5Q.mp4 | 4 | 0 | 0 | 0 |
| v_No5ZwqHdEQU.mp4 | 4 | 0 | 0 | 0 |
| v_9PP5_HGpu4c.mp4 | 4 | 0 | 0 | 3 |
| v_rzIaKwWJDZI.mp4 | 6 | 0 | 0 | 0 |
| v_dRAn_gsx9Wc.mp4 | 2 | 0 | 0 | 0 |
| v_45gAK3x_0ds.mp4 | 2 | 0 | 0 | 0 |
| v_xbQQhK7wQZQ.mp4 | 8 | 0 | 0 | 0 |
| v_AimG8xzchfI.mp4 | 9 | 0 | 0 | 4 |
| v_HEw5wIWVpWE.mp4 | 6 | 0 | 0 | 0 |
| v_qY7LG7r_IA8.mp4 | 2 | 0 | 0 | 0 |
| v_6EWzgWd72Cs.mp4 | 13 | 0 | 0 | 0 |

각 영상의 원본 경로, sample ID, 재현 명령, 종료 코드, 프레임 hash, stderr는 `../../../../../artifacts/runs/operations/mmco_diagnosis/pilot/300/data/videos/<index>/`와 `../../../../../artifacts/runs/operations/mmco_diagnosis/pilot/300/metrics/results.json`에 보존한다.

대응: 기존 zero-shot과의 비교 중에는 decoder나 원본을 교체하지 않는다. 경고 자체는 파일별 stderr에 보존하고 진행 로그와 분리한다. 입력 픽셀 불일치가 확인되면 해당 표본을 명시하고 순차 디코딩을 사용하는 별도 비교 실험이 필요하다.
