# 데이터셋별 100개 attention–GT 분석

Charades-STA test, QVHighlights val, ActivityNet-Captions val_1에서 각각 서로 다른 영상 100개씩, 총 300개를 분석했다. 기존 실험과 같은 seed 20260929, 백본, query 정의, FPS/pixel budget 및 레이어 8–21 집계를 유지했다. 모든 샘플의 초 단위 attention 곡선과 별도 GT 트랙을 저장했다. Pearson correlation은 계산하지 않았다.

[데이터셋 요약 그림](results/summary.png) · [300개 검색/열람](results/index.html) · [분위수별 예시](results/representative_examples.png)

**GT 구간으로 attention이 모이는 경향과 영상 초반에 집중되는 경향이 함께 나타난다.** 세 데이터셋 모두 GT 질량 초과의 중앙값이 양수지만, 전체 최고점이 GT 안에 있는 경우는 약 절반이다. 따라서 GT와의 대응을 볼 때 최고점 하나와 분포 전체를 구분해야 한다.

## GT와의 대응

최고 attention peak가 GT 안에 있는 빈도와, GT의 길이 때문에 자연히 들어갈 수 있는 attention을 구분해 집계했다. 아래 ‘GT 질량 초과’는 영상 내부 attention을 합계 1로 정규화한 후, GT attention 질량에서 GT 시간 비율을 뺀 값의 중앙값이다.

| 데이터셋 | 최고 peak가 GT 안 | 균등 bin 선택 기준 | 항상 첫 bin 선택 기준 | GT 질량 초과 중앙값 | 시간 균등 기준보다 GT 질량이 큰 샘플 |
|---|---:|---:|---:|---:|---:|
| Charades-STA | 50/100 | 27.2% | 22.0% | +7.42%p | 68/100 |
| QVHighlights | 49/100 | 24.2% | 15.0% | +10.55%p | 83/100 |
| ActivityNet-Captions | 50/100 | 31.0% | 27.0% | +5.11%p | 67/100 |

GT 질량 초과가 양수이면 GT의 길이만으로 설명되는 것보다 attention이 GT에 더 모인 것이다. 이는 입력 attention의 기술 통계이며, 모델이 생성한 구간의 정확도나 MTLA re-ranking 성능을 뜻하지 않는다.

## 영상 시작 부분의 집중

| 데이터셋 | 첫 temporal bin이 최고점 | 영상 초반 10%에 최고점 | 첫 bin 제외 후 최고점이 GT 안 |
|---|---:|---:|---:|
| Charades-STA | 46.0% | 59.0% | 56.0% |
| QVHighlights | 19.0% | 57.0% | 52.0% |
| ActivityNet-Captions | 28.0% | 61.0% | 54.0% |

첫 bin 제외 결과는 같은 저장 attention의 사후 집계다. query의 의미 때문에 attention이 달라졌다는 인과적 검증은 아니다. GT가 영상 시작에 놓이는 데이터셋의 위치 편향을 확인할 수 있도록 항상 첫 bin을 선택하는 기준과 상대 시간별 GT occupancy도 함께 제시했다.

[상대 시간 요약 그림](results/relative_time_summary.png)에서는 원래 초 단위 시간을 영상 길이로 나누어 비교한다. 이는 보조 그림이고, 개별 300개 그림의 가로축은 모두 초 단위다.

레이어별로 보면 중간 레이어에서 GT peak 정렬이 높다. 관측 최고값은 Charades-STA의 15번 레이어 55%, QVHighlights의 15번 레이어 65%, ActivityNet의 16번 레이어 57%다. 이는 동일 100개에서 살펴본 사후 관찰이며, 기본 집계 범위인 레이어 8–21은 변경하지 않았다.

## 전체 그림과 예시 선택

예시는 GT 질량 초과의 25·50·75 분위수에 가장 가까운 순위의 샘플을 각 데이터셋에서 1개씩 선택했다. attention이 잘 맞는 사례만 골라 보여주지 않도록 선택 규칙과 ID를 공개한다. 대표 그림의 각 행은 데이터셋, 각 열은 해당 분위수다.

| 데이터셋 | 분위수 | query ID | 그림 |
|---|---:|---|---|
| Charades-STA | 25% | test_370 | [상세](results/Charades-STA/Charades-STA_38/attention.png) |
| Charades-STA | 50% | test_485 | [상세](results/Charades-STA/Charades-STA_26/attention.png) |
| Charades-STA | 75% | test_210 | [상세](results/Charades-STA/Charades-STA_83/attention.png) |
| QVHighlights | 25% | 10050 | [상세](results/QVHighlights/QVHighlights_56/attention.png) |
| QVHighlights | 50% | 3748 | [상세](results/QVHighlights/QVHighlights_51/attention.png) |
| QVHighlights | 75% | 5762 | [상세](results/QVHighlights/QVHighlights_32/attention.png) |
| ActivityNet-Captions | 25% | v_iUMDlxU14bM_4 | [상세](results/ActivityNet-Captions/ActivityNet-Captions_40/attention.png) |
| ActivityNet-Captions | 50% | v_sG3JpMuXFnU_4 | [상세](results/ActivityNet-Captions/ActivityNet-Captions_02/attention.png) |
| ActivityNet-Captions | 75% | v_CMTiL1ctmDs_0 | [상세](results/ActivityNet-Captions/ActivityNet-Captions_09/attention.png) |

- [Charades-STA 100개 전체 곡선](results/Charades-STA_overview.pdf) / [상세](results/Charades-STA_detail.pdf)
- [QVHighlights 100개 전체 곡선](results/QVHighlights_overview.pdf) / [상세](results/QVHighlights_detail.pdf)
- [ActivityNet 100개 전체 곡선](results/ActivityNet-Captions_overview.pdf) / [상세](results/ActivityNet-Captions_detail.pdf)

## Annotation 범위와 검증

GT가 실제 영상 끝을 넘는 경우에도 원본 GT와 샘플 선택을 유지했다. 그림에는 원본 GT 전체와 영상 밖의 회색 영역을 표시했고, 수치 요약만 실제 영상과 겹치는 구간을 사용했다. 아주 작은 길이 차이도 포함한 개수와 1초를 넘는 경우를 나누어 기록한다.

| 데이터셋 | GT가 영상 끝을 넘는 샘플 | 그중 1초 초과 | 유효 GT가 영상 안에 존재 |
|---|---:|---:|---:|
| Charades-STA | 13 | 8 | 100/100 |
| QVHighlights | 1 | 0 | 100/100 |
| ActivityNet-Captions | 0 | 0 | 100/100 |

GT 범위가 영상 안에 완전히 들어오는 샘플만 별도로 확인해도 GT 질량 초과의 중앙값은 양수였다. Charades-STA 87개에서 +6.27%p, QVHighlights 99개에서 +10.39%p다. 기본 결과에서는 원래 선정한 100개씩을 모두 유지했다.

300개 모두 공식 원본 annotation과 대조했고, 총 10,800개 레이어의 attention 검증을 완료했다. 원본 MTLA 72개 파일은 SHA-256이 동일하다. 긴 입력의 head 분할 eager 연산은 기존 전체 eager 결과 두 샘플과 비트 단위로 동일함을 확인했다.

이 결과는 고정 시드로 선정한 100개씩에 대한 관찰이며, 전체 벤치마크 성능이나 통계적 유의성을 주장하지 않는다. [방법·지표 정의·재실행](README.md), [샘플별 수치](results/sample_metrics.csv), [검증 기록](results/final_validation.json).
