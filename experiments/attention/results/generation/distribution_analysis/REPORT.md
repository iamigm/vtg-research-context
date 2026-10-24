# Query–generated answer attention 분포 비교

**시간별 attention의 상대적 순위는 상당 부분 유지되지만, 확률 질량 배분과 최고점은 달라진다. 답변에서는 영상 초반 집중이 더 강해진다.** 세 데이터셋의 TV 중앙값은 0.160–0.225, Spearman 중앙값은 0.653–0.749다. 초반 10%를 제외해도 Spearman 0.600–0.691이 남는다. 이는 평균 곡선의 관찰이며, 개별 토큰의 동작이나 인과관계를 뜻하지 않는다.

기존 실제 greedy 생성 300개(데이터셋당 100개)의 저장된 곡선을 재계산했다. 레이어 8–21, 모든 head 및 해당 토큰을 평균한 attention이다. 주 분석은 답변 토큰을 입력으로 처리하는 행, EOS 제외 답변 전체다. 영상 내 합을 1로 맞춘 뒤 같은 샘플끼리 비교한다. 각 샘플을 동일 가중치로 집계한다. 모델 재실행은 하지 않았다.

## 분포 차이와 경향성

아래 연속형 수치는 샘플별 지표의 중앙값이다. Peak 일치는 샘플 비율이다.

| Dataset | TV | JS divergence (bits) | Cosine | Spearman | Top-20% overlap | Same peak | Wasserstein / duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Charades-STA | 0.1600 | 0.0285 | 0.9035 | 0.7486 | 66.7% | 34.0% | 4.7% |
| QVHighlights | 0.2125 | 0.0494 | 0.8202 | 0.7181 | 66.7% | 39.0% | 4.7% |
| ActivityNet-Captions | 0.2246 | 0.0596 | 0.8094 | 0.6532 | 62.5% | 44.0% | 5.0% |

- TV = ½Σ|p−q|. 0은 동일, 1은 질량이 완전히 분리됨. 1−TV는 두 분포가 공유하는 확률 질량이다.
- JS는 base-2 Jensen–Shannon **divergence**, 범위 0–1. JS distance의 제곱이다.
- Spearman은 시간 bin별 attention 크기의 순위 상관(-1–1)이다. Pearson 계수는 계산하지 않았다.
- Top-20%는 k=ceil(0.2×bin 수)인 상위 bin 두 집합의 교집합/k. 독립 무작위 선택 기대치는 k/bin 수(약 20%)다. 시계열 자기상관을 고려한 유의성 검정은 아니다.
- Wasserstein은 timestamp 지점에 질량을 놓았을 때 필요한 평균 이동 거리를 영상 길이로 나눈 값이다.
- 동일 peak는 정확히 같은 bin에서 최댓값을 갖는 비율이다. 단순 영상 초반 쏠림도 유사도를 높일 수 있다.

## 영상 초반 10%를 제외한 민감도 분석

timestamp가 영상 길이의 10% 미만인 bin을 제거하고 남은 질량을 각각 다시 정규화했다. 경계에 걸친 bin을 분할하지 않는다.

| Dataset | Full → cropped Spearman | Full → cropped Top-20% overlap | Cropped TV | Positive rank correlation | Full rho ≥ 0.5 | Full rho ≥ 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Charades-STA | 0.749 → 0.691 | 66.7% → 60.0% | 0.115 | 96/100 | 88/100 | 64/100 |
| QVHighlights | 0.718 → 0.683 | 66.7% → 59.3% | 0.175 | 100/100 | 87/100 | 57/100 |
| ActivityNet-Captions | 0.653 → 0.600 | 62.5% → 54.2% | 0.178 | 95/100 | 77/100 | 39/100 |

## 변화 방향

영상 총량과 early peak 비율은 평균, 나머지는 paired 변화량의 중앙값이다. 첫 10% 질량은 timestamp 중점 bin 안에서 균일한 질량을 가정해 경계 중첩을 반영한다.

| Dataset | Query → answer video mass | Query → answer early peak | First 10% mass gain | Centroid shift / duration | Adjacent rise/fall agreement | GT mass gain |
|---|---:|---:|---:|---:|---:|---:|
| Charades-STA | 24.6% → 7.6% | 59.0% → 95.0% | +6.72%p | -3.45% | 72.0% | -0.61%p |
| QVHighlights | 23.4% → 7.3% | 57.0% → 97.0% | +9.43%p | -2.66% | 70.3% | -1.88%p |
| ActivityNet-Captions | 22.4% → 5.8% | 61.0% → 84.0% | +9.38%p | -2.56% | 68.9% | -0.71%p |

음수 중심 이동은 영상 앞쪽으로 이동함을 뜻한다. 인접 bin 상승/하강 일치는 0 변화가 있는 쌍을 제외한 부호 일치율이다. 무작위 기대치는 두 곡선의 상승/하강 비율의 곱으로 계산해 JSON에 저장했다.

## 불확실성과 토큰 선택 민감도

샘플 단위 bootstrap 4,000회, seed=20260929. 각 데이터셋의 100개 샘플은 영상 ID가 모두 다르다. 아래는 중앙값의 95% percentile 구간이다. 샘플 변동성을 나타내며, 모델·프롬프트·다른 seed로의 일반화를 보장하지 않는다.

| Dataset | TV median 95% CI | Spearman median 95% CI | Predictive TV / rho | Timestamp digits n; TV / rho | Non-truncated n; TV / rho |
|---|---|---|---|---|---|
| Charades-STA | 0.147–0.168 | 0.713–0.776 | 0.160 / 0.745 | 99; 0.158 / 0.734 | 100; 0.160 / 0.749 |
| QVHighlights | 0.201–0.219 | 0.688–0.762 | 0.214 / 0.717 | 95; 0.216 / 0.706 | 98; 0.213 / 0.718 |
| ActivityNet-Captions | 0.206–0.244 | 0.632–0.693 | 0.224 / 0.650 | 99; 0.229 / 0.642 | 100; 0.225 / 0.653 |

Predictive는 각 답변 토큰을 생성하기 직전 행이다. Timestamp digits는 구간 파싱 성공 샘플만 포함한다. 두 경우 모두 동일 샘플의 query와 비교했다. 레이어/head/답변 토큰을 평균한 곡선에 대한 결과이므로 각 개별 토큰·head까지 동일 경향이라고 해석할 수 없다. 영상 attention 총량 감소에는 답변이 볼 수 있는 key 집합 증가가 포함되므로 시각 정보 사용량의 인과적 감소라고 단정하지 않는다.

## 검증과 산출물

모든 샘플·토큰 그룹에서 재계산한 TV와 영상 총량이 기존 sample_metrics.json과 atol=1e-10으로 일치했다. timestamp와 정규화된 곡선도 검증했다. 원본 CSV 및 분석 스크립트 SHA256을 저장했다.

- [샘플 분포 그림](distribution_summary.png) / [PDF](distribution_summary.pdf)
- [상대 시간 평균 분포](relative_time_profiles.png) / [PDF](relative_time_profiles.pdf)
- [샘플별 수치](sample_metrics.csv) / [집계 및 CI](summary.json)
- 재실행: 실험 폴더에서 `.venv/bin/python analyze_distribution.py`
