# Interval distribution · pilot300 · 실행 명세

Top-p=0.99는 원래 전체 vocabulary softmax에서 누적 임계값을 넘는 token까지 포함한다. 그 집합 중 원래 확률1% 미만 token은 제외한다. 1%는 포함하며 하한 때문에99%를 채우지 못해도 추가하지 않는다. Sampling·고정 node/rollout/token 예산·uncertainty pruning은 없다. EOS로 끝나는 허용 경로를 추적하며 실제 model context capacity에 도달하면 미완성 질량으로 남기고 전체 완료로 주장하지 않는다.

입력은 원래 Qwen3-VL-8B-Instruct,2 FPS,video_query,고정 H×W/frame IDs/PTS/tensor hash다. 무작위 반복 seed는 없다. 고정900 cohort의 정확한 prefix300을 decoding·두 모델 semantic·독립4B 검증·MD/HTML·검증까지 마친 뒤 나머지600으로 확장한다. 원자료는 canonical900에 한 번 저장하고300은 부분집합으로 참조한다.

경로 질량은 원래 token 확률의 곱이다. 분기별 raw token entropy와 다음 token entropy 변화,유효 완성 interval의 entropy·시작/종료 covariance·분기 내부/분기 사이 covariance를 저장한다. Interval entropy는 exact clipped (start,end) 쌍으로 계산하며 추가 binning은 없다. Clipping은[0,metadata duration],start<end,EOS 필수다. 조건부 최종 통계는 유효 잎 질량으로 정규화하고 분기 가중치는 p(token)×그 아래 유효 질량이다. 원래 확률 질량과 조건부 가중치를 분리한다. 필터 제외·무효·미완성 질량도 표시한다.

시간 가설은0.5초 grid에 probability-weighted density를 만들고 bandwidth1초로 smoothing한 뒤 watershed 봉우리·골짜기로 나눈다. Saddle가 낮은 쪽 peak의50%보다 높으면 합치며0.5/2초 bandwidth에서 partition 안정성을 검사한다. 이 grid는 clustering용이며 entropy를 다시 binning하지 않는다. 구간 자체를 잘라 비중첩으로 만들지 않는다. Group은 의미 사건 GT가 아니다.

Semantic은 원래 query 없이 각 clip을 독립 encoding한다. Instruct 한 forward에서 마지막 layer의 visual token 공간·시간 평균,4개 시간 bin 연결,공식 embedding과 같은 마지막 readout을 얻는다. Qwen3-VL-Embedding-8B는 같은 clip·고정 instruction·readout token의 normalized hidden state다. Temporal patch당 실제 시간 중심을 이용해 가중하고4개 bin이 비면 해당 조건은 결측이다. 선택된 프레임이 없으면 임의로 채우지 않고 semantic coverage 결측으로 남긴다. 원본 interval이 달라도 실제 frame/PTS/processor 입력이 같으면 재사용한다. 전체 hidden state는 보존하지 않고 normalized vector는 영상·모델 단위 임시 SQLite에서 통계 계산 후 삭제한다.

Unit vector의 가중 평균μ에서 dispersion=1−||μ||²,그룹 A/B의 가중 평균 cosine=μ_A·μ_B다. Self pair를 제외한 구간 평균 cosine도 따로 표시한다. Centroid cosine은두μ를 다시 정규화한 값이며 평균 pair cosine과 다르다. 모든 interval pair를 영구 저장하지 않고 group/branch 요약과 대표 interval IDs를 보존한다. 다른 clustering으로 재분석하려면 clip을 다시 encoding할 수 있다.

독립 검증 모델은 제안8B와 다른 checkpoint인 Qwen3-VL-4B-Instruct다. 각 유효 후보를 앞뒤1초 context와 함께 다시 보고,event/onset/offset의 A=supported,B=contradicted,C=uncertain 확률을 얻는다. 공통 video/query prefix KV를 한 번 계산한 뒤 세 짧은 criterion suffix를 평가한다. Score=P(A)+0.5P(C),선택지 원래 확률 합<1%이면0.5로 판단 보류다. Event-only와 세 criterion score의 기하평균 두 방식을 비교한다. Zero-shot baseline도 같은 방식으로 다시 보고,동점이면 baseline 우선·그다음 원래 경로 질량순이다. 최종 답은 기존 후보 또는 baseline이며 새 timestamp는 생성하지 않는다. GT는 이 모든 과정 이후 채점에만 쓴다. 모든 후보를 평가하되 동일 입력의 점수는 재사용한다. 성능 향상은 가설이며 결과로 검증한다.

[VideoMind 공식 구현](https://github.com/yeliudev/VideoMind/blob/6721d2564eeb9ea8ab6c0ab8eaa39bd231ee8927/videomind/dataset/wrappers/verifying.py)은 예측 구간 crop과 yes/no 검증을, [VideoHV-Agent 공식 구현](https://github.com/Haorane/VideoHV-Agent/blob/ddc160bd05e2d64922ca717d1834500fa7d8f492/VideoHV-Agent/video_hv/verifier.py)은 가설을 영상 근거로 검증하는 절차를 참고했다. 이번 pointwise4B 방법은 이 아이디어를 적용한 별도 조건이며 논문 전체의 재현이 아니다.

비용은 decoding/각 semantic 조건/독립 검증을 나눠 native forward 수,decoder inputs,CUDA stream seconds,wall seconds,peak allocated memory를 기록한다. 동일한 Instruct forward에서 얻는 세 pooling의 GPU encoding 비용은 공유 비용이지 세 번 수행한 것으로 합산하지 않는다. Event-only는 공통 prefill+event suffix,세 criterion은공통 prefill+세 suffix 비용이다. Native decoder inputs는 모델에 실제 넣은 token position 수이며 output token 수나 FLOPs가 아니다. CUDA stream seconds는 kernel active time과 같지 않다. Model load·입력 준비·report 시간도 운영 비용과 구분한다. 실측 share 비용을 방법별 독립 실행시간으로 주장하지 않는다.
