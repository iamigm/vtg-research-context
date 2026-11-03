# 같은 100개 샘플: 입력 query와 실제 생성 답변 attention 비교 · pilot300 (300질문)

Charades-STA test, QVHighlights val, ActivityNet-Captions val_1에서 이전과 동일한 100개씩, 총 300개에 대해 실제 답변을 생성했다. 프롬프트와 영상 전처리는 유지했고, 각 샘플에서 greedy 답변 한 개를 생성했다. 아래 attention은 실제 KV-cache 생성 실행에서 직접 읽었으며 GT를 모델에 입력하지 않았다.

[전체 결과](report.html) · [요약 그림](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/summary.png) · [레이어별 비교](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/layer_comparison.png)

## 실제 답변 및 GT와의 겹침

모델의 raw 답변과 파싱한 `[s, e]`를 모두 저장했다. 원래 single-span 지시문은 `s - e seconds` 또는 설명 문장으로도 응답한다. 이를 사후 파싱했으며 답변을 GT로 교체하거나 GT에 맞게 수정하지 않았다. Charades/ActivityNet은 MTLA 규칙대로 파싱된 첫 구간, QVHighlights는 모든 구간을 사용했다.

| 데이터셋 | 답변 생성 | 구간 파싱 성공 | 128-token 제한 도달 | 예측–GT 중 tIoU≥0.5인 쌍이 있는 샘플 |
|---|---:|---:|---:|---:|
| Charades-STA | 100 | 99 | 0 | 42/100 |
| QVHighlights | 100 | 95 | 2 | 41/100 |
| ActivityNet-Captions | 100 | 99 | 0 | 43/100 |

위 tIoU 집계는 이 100개 샘플의 겹침 확인용이다. 특히 QVHighlights는 여러 GT 중 하나만 맞아도 해당하므로 공식 mAP나 모든 구간의 recall과 같지 않다. 쿼리별 GT recall 및 원본 예측은 JSON/CSV에 별도로 기록했다.

## Query와 답변의 attention

주 비교는 query 이벤트 문장 전체, 답변 전체(EOS 제외), 파싱된 timestamp의 숫자 토큰이다. 레이어 8–21과 모든 head를 평균했다. ‘영상 attention 총량’은 원래 softmax attention 중 영상에 간 probability mass이다. ‘GT 비중 변화’는 영상 안에서 합계 1로 정규화한 분포의 GT 질량을 같은 샘플의 query와 비교한 값이며, 두 개념을 구분해야 한다.

| 데이터셋 | 토큰 그룹 | 영상 attention 총량 평균 | 최고 peak가 GT 안 | Query 대비 GT 비중 변화 중앙값 | GT 비중이 Query보다 커진 샘플 |
|---|---|---:|---:|---:|---:|
| Charades-STA | Input query | 24.6% | 50/100 | — | — |
| Charades-STA | Answer: all tokens | 7.6% | 29/100 | -0.61%p | 46/100 |
| Charades-STA | Answer: timestamp digits | 7.6% | 29/99 | +1.09%p | 56/99 |
| QVHighlights | Input query | 23.4% | 49/100 | — | — |
| QVHighlights | Answer: all tokens | 7.3% | 22/100 | -1.88%p | 42/100 |
| QVHighlights | Answer: timestamp digits | 7.1% | 22/95 | -0.46%p | 45/95 |
| ActivityNet-Captions | Input query | 22.4% | 50/100 | — | — |
| ActivityNet-Captions | Answer: all tokens | 5.8% | 35/100 | -0.71%p | 43/100 |
| ActivityNet-Captions | Answer: timestamp digits | 5.9% | 33/99 | -0.55%p | 40/99 |

숫자/timestamp 비교의 분모는 유효 구간을 파싱할 수 있었던 샘플 수다. 답변 전체 비교는 구간 파싱 여부와 무관하게 유지하며, GT 비중 변화는 항상 같은 샘플끼리 계산한다. 답변은 input query와 앞선 답변 토큰도 key로 보기 때문에 영상 attention 총량 감소만으로 시각 정보의 인과적 사용이 줄었다고 단정하지 않는다.

### 분포 변화의 해석

**세 데이터셋 모두 답변 전체의 attention peak가 GT 안에 들어간 비율은 query보다 낮았고, 영상 앞부분에 peak가 몰리는 경향은 더 강했다.** 이 조건에서는 답변 attention의 최고점만으로 GT 위치를 찾는 방식이 query보다 좋은 결과를 보이지 않았다.

GT peak는 가장 높은 한 지점만 보며, GT 비중은 구간 전체의 질량을 본다. 두 지표의 변화 방향이 다르면 ‘답변 attention이 GT와 더 잘 맞는다’는 단일 결론으로 합치지 않는다. 아래 수치는 같은 샘플의 실제 생성 결과를 비교한 관찰이다.

- **Charades-STA**: query → 답변 전체의 GT peak는 50/100 → 29/100, 영상 앞 10%에 peak가 있는 비율은 59.0% → 95.0%다. GT 비중 변화 중앙값은 답변 전체 -0.61%p, 시간 숫자 토큰 +1.09%p다.
- **QVHighlights**: query → 답변 전체의 GT peak는 49/100 → 22/100, 영상 앞 10%에 peak가 있는 비율은 57.0% → 97.0%다. GT 비중 변화 중앙값은 답변 전체 -1.88%p, 시간 숫자 토큰 -0.46%p다.
- **ActivityNet-Captions**: query → 답변 전체의 GT peak는 50/100 → 35/100, 영상 앞 10%에 peak가 있는 비율은 61.0% → 84.0%다. GT 비중 변화 중앙값은 답변 전체 -0.71%p, 시간 숫자 토큰 -0.55%p다.

128-token 제한에 도달한 답변을 제외한 민감도 확인:

| 데이터셋 | 토큰 그룹 | 남은 샘플 | GT peak | Query 대비 GT 비중 변화 중앙값 |
|---|---|---:|---:|---:|
| Charades-STA | Answer: all tokens | 100 | 29/100 | -0.61%p |
| Charades-STA | Answer: timestamp digits | 99 | 29/99 | +1.09%p |
| QVHighlights | Answer: all tokens | 98 | 22/98 | -1.88%p |
| QVHighlights | Answer: timestamp digits | 94 | 22/94 | -0.30%p |
| ActivityNet-Captions | Answer: all tokens | 100 | 35/100 | -0.71%p |
| ActivityNet-Captions | Answer: timestamp digits | 99 | 33/99 | -0.55%p |

## 토큰 위치 해석

주 그래프의 답변 attention은 **생성된 토큰을 입력으로 처리할 때의 행**이다. 예를 들어 답변의 `14` 숫자 토큰 자체가 영상에 보내는 attention이며 MTLA의 응답 토큰 위치 관례를 따른다. 그 토큰을 **생성하기 직전의** attention은 한 행 앞이므로 `predictive_temporal` 및 각 샘플의 `predictive.png`로 따로 저장했다. 두 관례가 섞이지 않도록 별도로 집계했다.

| 데이터셋 | 답변 전체: 토큰 처리 행 GT peak | 답변 전체: 생성 직전 행 GT peak | 숫자: 토큰 처리 행 GT peak | 숫자: 생성 직전 행 GT peak |
|---|---:|---:|---:|---:|
| Charades-STA | 29.0% | 29.0% | 29.3% | 29.3% |
| QVHighlights | 22.0% | 22.0% | 23.2% | 23.2% |
| ActivityNet-Captions | 35.0% | 34.0% | 33.3% | 33.3% |

## 전체 그림

각 그림의 시간축 아래 **GT(초록)**와 **모델 답변 구간(빨강)**을 별도로 표시했다. 상단은 원래 attention score, 두 번째 곡선은 영상 내부 정규화 분포, 아래 히트맵은 query/답변 각각의 토큰별 분포다.

- [Charades-STA 100개 비교](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/Charades-STA_overview.pdf) / [상세](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/Charades-STA_detail.pdf)
- [QVHighlights 100개 비교](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/QVHighlights_overview.pdf) / [상세](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/QVHighlights_detail.pdf)
- [ActivityNet 100개 비교](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/ActivityNet-Captions_overview.pdf) / [상세](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/figures/ActivityNet-Captions_detail.pdf)

## 검증과 해석 범위

입력 text, 영상 토큰 위치, 실제 timestamp가 이전 실험과 같은지 확인했다. 생성 중 읽은 query attention은 300개 모두 이전 배열과 비트 단위로 일치한다. GT가 영상 밖을 넘는 경우는 원본을 보존하고, attention 질량 계산에만 실제 영상과의 교집합을 사용했다. Pearson correlation은 계산하지 않았다. 이 결과는 한 번의 greedy 답변과 고정된 100개씩에 대한 관찰이며, MTLA의 16-rollout re-ranking이나 전체 벤치마크 평가 결과가 아니다.

[방법과 재실행](../../../../../../README.md) · [샘플별 수치](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/metrics/sample_metrics.csv) · [집계 JSON](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/metrics/summary.json)
