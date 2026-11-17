# Token uncertainty · decoding 경로별 비교 · pilot · 900 queries
<!-- token-unified-report -->

## 범위와 사용 방법

고정 pilot 900 queries에 저장된 Qwen3 deterministic rollout의 27,460개 경로를 확인한다. 왼쪽에서 sample을 선택하고 decoding 경로를 바꾸면 전체 응답의 step별 entropy 곡선·token distribution·영상 예측 구간이 함께 바뀐다. 각 token을 눌러 해당 step의 분포를 읽고, 부모 또는 같은 실행의 greedy 경로를 겹쳐 비교한다.

pilot300·600·900은 각자 별도의 sample viewer를 갖는다. 앞선 고정 cohort가 다음 cohort의 정확한 prefix이며 원자료는 canonical pilot900에서 참조한다. full9404에는 pilot 경로를 연결하거나 pilot 결과를 합산하지 않는다. 서로 겹치는 pilot 단계를 독립 표본처럼 더하지 않는다.

이 scope는 deterministic 경로만 제공하며, 기존 Token uncertainty greedy는 pilot300 및 full9404의 원래 scope에 보존한다. deterministic 입력은 Qwen3-VL-8B-Instruct, 2 FPS, spatial target 100,352 pixels, video_query다. 기존 Token uncertainty 입력과 다르므로 같은 분포로 합치지 않는다.

## 화면과 비교 기준

| 자료 | 읽는 방법 |
|---|---|
| step별 entropy 곡선 | 가로축에 실제 token을 표시한다. 점·token을 누르면 선택 step의 분포로 이동한다. |
| 부모 / greedy 곡선 | 같은 decoding 순번을 맞춰 겹친다. 분기 이후에는 서로 다른 prefix의 조건부 분포이므로 동일 입력에서 한 token만 바꾼 인과 비교가 아니다. |
| 강제 token / conditioning prefix / free suffix | 강제로 선택한 token과 이미 고정된 prefix를 이후 모델의 자유 생성과 구분한다. 강제 token에서 생긴 오류를 모델의 자유 선택 오류로 세지 않는다. |
| 좌측 특수·정량 필터 | 이 scope의 deterministic greedy(candidate #1)가 기준이다. 경로를 바꿔도 sample 필터의 기준은 유지하며 본문의 곡선·분포·요약은 선택 경로를 따른다. |
| 전체 경로 표 | 경로별 parent·분기 위치·예측 구간·IoU·numeric step 평균 entropy를 비교한다. |

**표 해설.** 먼저 같은 sample 안에서 오류가 생기는 위치와 entropy 변화를 확인한다. 다음으로 부모 곡선과 비교해 변화가 강제 분기 자체에서 생겼는지, 이후 free suffix에서 이어지는지 살핀다. 낮은 entropy가 정답이나 영상 증거 활용을 보장하지는 않는다.

## 지표 정의

원본 top-30 확률 합 M=Σpᵢ, 재정규화 qᵢ=pᵢ/M, H₃₀=−Σqᵢlnqᵢ로 계산한다. 단위는 nat, 범위는 0–ln(30)≈3.401이다. 값이 클수록 top-30 내부 확률이 고르다. top-30 밖의 mass는 포함하지 않는다. 선택 token이 top-30 밖이면 observed 원래 확률을 별도로 보여주며 entropy에 31번째 항으로 추가하지 않는다.

Δprevious=H₃₀(최초 numeric 오류)−H₃₀(같은 endpoint의 직전 정답 가능 numeric step)이며 소수점·서식을 건너뛴다. Δtoken=H₃₀(k)−H₃₀(k−1)은 최초 numeric 또는 format 오류 k의 바로 앞 decoding token을 사용한다. 양수는 상승, 음수는 감소이며 비교 상대가 없는 경우는 미정이다. 소수 첫째 자리 오류와 소수점 자체의 format 오류는 분리한다.

같은 필터 그룹의 복수 선택은 OR, 그룹 사이와 정량 범위는 AND다. 낮은 entropy ≤0.5 nat, 높은 entropy ≥1.5 nat은 탐색용 기준이며 검증된 오류 탐지 임계값이 아니다. IoU는 예측·GT 시간 구간의 교집합 길이/합집합 길이이고 높을수록 좋다. numeric mean H는 경로의 숫자 token entropy 평균이다.

지원하는 endpoint 판정은 음이 아닌 정수 또는 소수 한 자리의 초 표기다. clock 등 다른 표기도 전체 entropy와 token distribution은 볼 수 있으나 자릿수 최초 오류 판정은 붙이지 않는다. 저장 경로에는 raw 분포만 사용하며 기존 greedy의 processed 분포·후보 정답 가능성 label을 재사용하지 않는다.

## 결과와 appendix

[pilot sample viewer](report.html) · [자료·정량 appendix](APPENDIX.md)
