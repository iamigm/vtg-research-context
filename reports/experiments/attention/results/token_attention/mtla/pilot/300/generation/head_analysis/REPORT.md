<!-- report-appendix:APPENDIX.md parts=9 -->
<!-- report-part:0:begin -->
# 레이어·head별 영상 초반 편향 · pilot300 (300질문)

실제 생성 300개(각 벤치마크 100개), 36개 레이어 × 32개 head를 분석했다. **L/H는 모두 0-based**이며, 서로 다른 레이어의 같은 head 번호는 별개의 head다. Query 이벤트 토큰과 EOS 제외 답변 전체 토큰을 각각 평균했다. 답변은 생성된 토큰을 입력으로 처리하는 행이다.

주 편향 점수는 **각 샘플의 영상 내 attention 중 앞 10%에 배정된 질량을 계산한 뒤 샘플 평균**한 값이다. 시간 bin 경계는 실제 timestamp의 중점이고, 경계와 겹치는 bin 질량을 시간 비율로 배분했다. 균일한 시간 분포의 기준은 10%. 값이 클수록 앞부분 집중이 강하다. 영상에 보내는 attention 총량도 함께 봐야 한다.

[인터랙티브 32-head 분포](report.html) · [36개 레이어 전체 PDF](../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/figures/all_layers_heads.pdf) · [전체 순위 CSV](../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/metrics/rankings.csv) · [전체 heatmap](../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/figures/bias_heatmap.png)

<!-- report-part:0:end -->
<!-- report-part:5:begin -->
## 데이터셋별 1위

| 데이터셋 | Query 1위 | 질량 | Answer 1위 | 질량 |
|---|---|---:|---|---:|
| Charades-STA | L27 H22 | 93.69% | L27 H21 | 93.78% |
| QVHighlights | L27 H22 | 98.18% | L27 H23 | 97.99% |
| ActivityNet-Captions | L20 H24 | 79.85% | L33 H0 | 79.90% |

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
## 원래 평균 곡선의 초반 질량에 가장 많이 기여하는 head

조건부 집중률 순위와 별개로, 레이어 8–21에서 실제 softmax의 초반 영상 질량을 많이 보내는 head를 정렬했다. 기여율은 모든 샘플·선택 head의 raw 초반 질량 합에서 해당 head가 차지하는 비율이다. 개별 샘플의 기여율을 평균한 값과는 다르다.

| Scope | Layer / Head | Raw 초반 질량 | 선택 head 내 기여율 | 조건부 초반 질량 |
|---|---|---:|---:|---:|
| query | L12 H18 | 28.589% | 1.42% | 41.96% |
| query | L15 H11 | 22.949% | 1.14% | 33.27% |
| query | L13 H11 | 20.570% | 1.02% | 40.69% |
| query | L11 H14 | 19.002% | 0.94% | 28.20% |
| query | L21 H8 | 18.024% | 0.89% | 31.16% |
| answer | L21 H7 | 11.551% | 1.35% | 87.71% |
| answer | L21 H8 | 11.265% | 1.32% | 32.45% |
| answer | L12 H18 | 10.322% | 1.21% | 42.66% |
| answer | L20 H30 | 8.975% | 1.05% | 31.73% |
| answer | L9 H14 | 8.417% | 0.99% | 34.33% |

<!-- report-part:6:end -->
<!-- report-part:7:begin -->
## 기준 변경과 해석 범위

- query, first_5pct: 1위 L27 H22; 앞 10% 순위와 Spearman 0.997.
- query, first_20pct: 1위 L27 H22; 앞 10% 순위와 Spearman 0.994.
- answer, first_5pct: 1위 L27 H23; 앞 10% 순위와 Spearman 0.997.
- answer, first_20pct: 1위 L27 H21; 앞 10% 순위와 Spearman 0.998.

Bootstrap은 샘플 단위 4,000회(seed=20260929)로 각 고정 head의 평균 CI를 구했다. 데이터로 상위 head를 선택했으므로 CI는 1위의 통계적 우월성이나 다중비교 보정을 뜻하지 않는다. 고정 모델·프롬프트·샘플에 대한 기술적 분석이다. 초반 편향이 위치 효과인지 영상 내용 때문인지는 이 관측만으로 분리할 수 없다.

정규화 후 높은 초반 비율이라도 전체 영상 attention이 작을 수 있다. `../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/metrics/rankings.csv`에 전체 영상 질량, raw 초반 질량, uniform-video-key 기준, 질량 중심, 첫 bin 질량을 함께 저장했다. 매우 짧은 영상은 하나의 bin이 앞 10%보다 넓으므로 시간 중첩 비율을 적용한 값의 상한이 낮아질 수 있다.

각 head 분포는 영상 내 합을 1로 정규화한 뒤 동일 가중치로 평균했다. 영상 attention 총량으로 가중된 기존 전체 평균 곡선과는 집계 순서가 다르다. 전 레이어에서 같은 head 번호를 합산해 특정 head라고 부르지 않았다.

<!-- report-part:7:end -->
<!-- report-part:8:begin -->
## 파일

- `../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/data/layers/`: 36개 레이어별 32-head 그림.
- `<dataset>/bias_heatmap.png`, `<dataset>/top_heads.png`: 벤치마크별 비교.
- `<dataset>/<sample_key>/head_attention.npz`: 토큰 평균 후 head별 원래 시간 곡선 및 지표.
- `../../../../../../../artifacts/runs/token_attention/mtla/pilot/300/generation/head_analysis/arrays/aggregate_profiles.npz`: 50개 상대 시간 bin에 질량을 보존해 재배분한 head별 평균 곡선.
- 재실행: `.venv/bin/python analyze_heads.py`. 기존 원본 데이터와 생성 결과는 수정하지 않는다.
<!-- report-part:8:end -->
## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
