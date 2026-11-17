# Token uncertainty · within-path 비교 · pilot · 300 queries
<!-- token-unified-report -->

## 무엇을 먼저 보는가

한 sample의 같은 decoding 경로에서 step별 entropy가 어떻게 변하는지가 주 분석이다. 결과 탭에서 왼쪽 sample을 고르면 응답 전체의 entropy 곡선과 최초 오류를 본다. step을 누르면 해당 token distribution과 정답 가능성을 보조 패널에서 확인한다. start/end는 별도로 paired 비교하며, 회색의 오류 이후 step은 그 비교에서 제외한다.

샘플 안의 **decoding 경로 선택**에서 기존 greedy와 저장된 deterministic rollout 경로를 전환한다. 현재 scope에서는 300개 query에 Qwen3의 다른 경로가 연결되어 있다. 좌측 경로 보유 필터로 해당 샘플만 고를 수 있다. 선택 경로의 step별 raw top-30 entropy와 token distribution을 같은 패널에서 확인하고, 부모 또는 같은 deterministic 실행의 greedy 경로를 겹쳐 비교한다. 가로축은 각각의 생성 token이며 공통 prefix 이후에는 다른 prefix의 분포다.

deterministic 경로는 pilot 900의 기존 분포를 직접 참조한다. pilot300·600·900은 각각의 고정 cohort만 보여주는 별도 화면이며 full과 합치지 않는다. 기존 Token uncertainty greedy와 입력 조건이 다르므로 source를 구분하며 통계나 probability를 합치지 않는다. 강제 token·고정된 prefix·이후 free suffix를 표시하고, 강제로 생긴 오류를 모델의 자유 선택 오류와 구분한다. 왼쪽 오류·정량 필터는 기존 greedy sample 기준이며 경로를 바꿔도 그 모집단은 유지한다. 아래 표도 기존 greedy의 집계다.

pilot/300의 실제 query 수는 300개다. 두 모델 중 지원 표기의 594개 응답을 within-path 분석에 사용했다. 지원 표기 밖 응답도 영상·응답·전체 token distribution은 확인할 수 있다.

## 지표와 필터 기준

모든 entropy와 후보 확률은 top-30 재정규화 기준이다. 각 step의 top-30 원래 확률 합을 M이라 두고 qᵢ=pᵢ/M, H₃₀=−Σqᵢlnqᵢ로 계산한다. 단위는 nat이며 0–ln(30)≈3.401이다. entropy가 높으면 top-30 내부 후보 확률이 더 고르다. top-30 밖의 probability mass는 포함하지 않는다. Δprevious=H₃₀(최초 numeric 오류)−H₃₀(직전 정답 가능 numeric step)이다. 양수면 오류 때 더 높았다. 정답 가능성은 허용 문법 아래 반올림 endpoint 정답으로 완성할 수 있는지의 판정이며 최종 정답 확률이 아니다.

| 특수 필터 기준 | 계산·분류 | 포함 범위 |
|---|---|---|
| 직전 숫자 대비 상승·감소 (기존) | Δprevious = H(최초 numeric 오류) − H(직전 정답 가능 numeric step) | 같은 endpoint에서 이전 숫자가 있는 paired 경로. 중간 소수점·서식은 건너뜀 |
| 직전 token 대비 상승·감소 | Δtoken = H(k) − H(k−1), k는 최초 오류(numeric 또는 format)의 decoding step | 바로 앞 decoding token. 소수점·공백·endpoint 밖 token도 포함. 첫 숫자·서식 오류도 직전 token이 있으면 포함 |
| 소수부 최초 오류 | 최초 numeric 오류의 role이 tenths | 허용된 소수 한 자리 표기의 소수 첫째 자리. 소수점 자체의 오류는 format으로 분리 |
| 정수부 / 서식 / 오류 없음 | 최초 오류의 숫자 role / format 판정 / 최초 오류 없음 | 같은 버튼 그룹 안에서는 OR, 서로 다른 그룹 및 정량 범위는 AND로 결합 |

**표 해설.** 두 상승·감소 기준은 비교 상대가 다르다. 예를 들어 정수 숫자 H=1.2, 소수점 H=0.01, 최초 소수부 오류 H=0.8이면 기존 숫자 기준은 −0.4 nat으로 감소지만 바로 직전 token 기준은 +0.79 nat으로 상승이다. 이는 계산 설명용 가상 예시다. 양수만 상승, 음수만 감소이며 0과 비교 상대가 없는 경로는 둘 다 아니다. 모든 H는 raw top-30 재정규화 entropy다. 아래 기존 정량 표는 직전 숫자 기준을 유지한다.

같은 필터 그룹 안에서 여러 버튼을 선택하면 하나라도 만족하는 경로를 포함한다(OR). 서로 다른 그룹과 정량 범위는 모두 만족해야 한다(AND). 선택한 버튼을 다시 누르면 해제하며 전체는 해당 그룹의 제한을 없앤다. 낮은 entropy 필터와 최초 오류 entropy 범위·정렬은 numeric·format 최초 오류 모두에 적용한다. 좌측 빠른 필터의 낮은 entropy는 ≤0.5 nat, 높은 entropy는 ≥1.5 nat인 탐색용 고정 기준이다. 이는 검증된 오류 탐지 임계값이 아니다. 수치 범위는 선택한 모델·endpoint의 sample별 지표에 AND로 적용한다. 오류 전후 차이가 없는 경로는 해당 범위를 지정했을 때 제외하며, 필터 후 숫자는 고유 query 수다.

## 같은 경로의 오류 전후 차이

| 모델 | dataset | endpoint | paired query 수 | 오류 − 직전 숫자 (nat) [95% CI] |
|---|---|---|---:|---:|
| qwen25 | activitynet-timelens | start | 43 | 0.5316 [0.3487, 0.7160] |
| qwen25 | activitynet-timelens | end | 46 | 0.8065 [0.6464, 0.9780] |
| qwen25 | charades-timelens | start | 44 | 0.9331 [0.7337, 1.1169] |
| qwen25 | charades-timelens | end | 31 | 0.8437 [0.6187, 1.0577] |
| qwen25 | qvhighlights-timelens | start | 54 | 0.8810 [0.7280, 1.0273] |
| qwen25 | qvhighlights-timelens | end | 54 | 0.8193 [0.6760, 0.9607] |
| qwen3 | activitynet-timelens | start | 36 | 0.7237 [0.5295, 0.9140] |
| qwen3 | activitynet-timelens | end | 49 | 1.0041 [0.8626, 1.1507] |
| qwen3 | charades-timelens | start | 35 | 0.9360 [0.7510, 1.1171] |
| qwen3 | charades-timelens | end | 36 | 0.9876 [0.8191, 1.1482] |
| qwen3 | qvhighlights-timelens | start | 34 | 0.8146 [0.6281, 1.0106] |
| qwen3 | qvhighlights-timelens | end | 52 | 0.8853 [0.7404, 1.0289] |

**표 해설.** 각 sample에서 오류 entropy와 직전 숫자의 entropy를 뺀 후 query 사이에서 평균한 값이다. 12/12개 그룹의 CI가 0 위에 있다. 이는 그룹 평균의 상승이며 모든 sample에서 상승했다는 뜻은 아니다. CI는 query 단위 bootstrap 2,000회로 구했다. 자릿수·순서를 통제한 결과와 제외 경로는 within-path 상세 appendix에서 함께 읽는다.

## 결과와 appendix

[통합 sample viewer](report.html) · [전체 appendix 목록](APPENDIX.md)

sample 기준 수준만 제거해도 prefix·자릿수·후보 집합의 차이는 남는다. 저장된 greedy 경로의 사후 비교이므로 인과 효과나 온라인 오류 탐지 성능을 뜻하지 않는다. pilot과 full은 별도 범위이며 독립 표본처럼 합산하지 않는다.

[Entropy 상승·최초 오류의 정량 비교와 영상 증거 실험안](entropy_error/REPORT.md)은 appendix에서 확인한다. 특수 필터별 기술통계와 같은 경로의 유지 step을 대조한 결과, 아직 실행하지 않은 영상 개입 설계를 구분했다.

다른 저장 자료의 재집계: [다른 decoding 경로 · deterministic pilot 900](../../../../rollout/deterministic/viewer.html?view=pilot%2F900%2FAPPENDIX_ENTROPY.md) · [고정 prefix 영상 perturbation · pilot 60](../../../../rollout/oracle_switch_controls/viewer.html?view=pilot%2F60%2FAPPENDIX_ENTROPY.md)
