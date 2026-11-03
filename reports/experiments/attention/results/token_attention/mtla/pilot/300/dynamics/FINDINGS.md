# 실제 답변 생성의 토큰별 attention 변화 · pilot300 (300질문)

이전과 동일한 300개 샘플의 실제 greedy 생성 기록을 재사용했다. 새 답변을 생성하지 않았으며, EOS를 제외한 **4,139개 토큰**을 분석했다. 주 분석은 토큰을 선택하는 forward 행(predictive), 보조 분석은 그 토큰을 입력으로 처리하는 다음 행(consumed)이다. 레이어 8–21과 32개 head를 평균했다.

[토큰 슬라이더·자동 재생](report.html) · [요약 그림](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/figures/summary.png) · [전체 토큰 CSV](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/metrics/all_token_metrics.csv)

## 토큰마다 분포가 변하는가?

**분포는 토큰마다 변하지만, attention 최고점은 많은 샘플에서 같은 시간 bin에 머문다.** 아래 TV는 영상 내부 합계가 1인 분포 사이의 total variation, 즉 `0.5 × Σ|p_j − p_(j−1)|`이다. 0은 같은 분포이고, 1은 질량이 완전히 겹치지 않는 분포다. 상관계수나 p-value가 아니다.

| 데이터셋 | 샘플 / 생성 토큰 | 인접 단계 TV¹ | 답변 내내 최고점이 같은 샘플 | 영상 앞 10%에 최고점² |
|---|---:|---:|---:|---:|
| Charades-STA | 100 / 1228 | 0.118 | 56/100 | 90.3% |
| QVHighlights | 100 / 1449 | 0.144 | 50/100 | 95.1% |
| ActivityNet-Captions | 100 / 1462 | 0.141 | 60/100 | 82.4% |

¹ 각 샘플에서 인접 단계 TV를 평균한 후, 100개 샘플의 중앙값을 계산했다. 긴 답변이 더 큰 가중치를 받지 않는다. ² 샘플별 토큰 비율을 먼저 구한 후 100개 샘플을 동일 가중 평균했다. 이전 보고서의 답변 전체 평균 곡선 peak 비율과는 다른 지표다.

최고점이 고정되어도 주변 시간 구간의 질량과 GT 비중은 변할 수 있다. 특히 앞부분에 강한 peak가 유지되는 경우, 최고점 이동만 측정하면 생성 중의 변화를 놓친다.

## 생성 진행에 따른 변화

| 데이터셋 | 첫→마지막 단계 영상 attention 변화³ | 첫→마지막 단계 GT 비중 변화³ | 시간 숫자 vs 나머지 토큰의 GT 비중 차이⁴ |
|---|---:|---:|---:|
| Charades-STA | -1.49%p | +4.06%p | +0.44%p (n=99) |
| QVHighlights | -0.27%p | +4.09%p | +1.41%p (n=95) |
| ActivityNet-Captions | -0.96%p | +3.08%p | +1.50%p (n=99) |

³ 같은 샘플의 마지막−첫 단계 차이의 중앙값이다. 영상 attention은 전체 causal key 중 영상 토큰에 할당된 원래 probability mass이며, GT 비중은 영상 내부에서 재정규화한 질량이다. **첫 단계는 assistant prefix의 마지막 행이고, 마지막 단계는 다른 종류의 토큰을 선택하므로 이 차이를 단조 증가나 생성하면서 위치 추정 능력이 개선됐다는 증거로 해석하지 않는다.**

⁴ 같은 답변에서 유효하게 파싱된 timestamp 숫자 토큰 단계와 나머지 단계를 각각 평균한 뒤 차이의 중앙값을 구했다. 토큰 종류별 관찰 비교이며 인과 효과는 아니다.

요약 그림은 답변의 상대 진행도를 10개 구간으로 나누고, 구간 안의 토큰을 샘플별로 평균한 뒤 샘플 간 중앙값과 IQR을 그렸다. 각 구간에 실제 토큰이 있는 샘플만 사용하며, 기여한 샘플 수는 `../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/metrics/summary.json`의 `progress_bins`에 기록했다. 개별 그림과 CSV는 상대 진행도로 축약하지 않은 모든 단계를 제공한다.

## 제공한 그림과 수치

- 인터랙티브 뷰어: 현재 토큰·직전 단계·입력 query의 시간별 곡선, GT/답변 구간, 슬라이더와 자동 재생. 원래 점수와 영상 내부 정규화 분포를 함께 표시한다.
- 샘플별 그림: 토큰×영상 시간 히트맵, 영상/input query/이전 답변/기타 prefix의 attention 비중, GT 비중, peak/centroid 이동, 인접 단계 TV.
- CSV: 모든 토큰의 실제 ID·문자열·forward 위치·raw GT 질량·GT 비중·peak·centroid·entropy·인접 단계 TV·peak 이동량. 첫 단계의 이전 단계 지표는 결측값으로 보존했다.
- [Charades-STA 100개 PDF](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/figures/Charades-STA_token_dynamics.pdf)
- [QVHighlights 100개 PDF](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/figures/QVHighlights_token_dynamics.pdf)
- [ActivityNet-Captions 100개 PDF](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/figures/ActivityNet-Captions_token_dynamics.pdf)
- [전체 300개 PDF](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/figures/all_token_dynamics.pdf)

## 검증과 범위

300개 모두 기존 생성 답변과 토큰 ID를 유지했다. 예측 행과 토큰 처리 행의 한 칸 대응, 시간별 attention 합, GT 질량과 TV 계산, 원본 MTLA 및 이전 실험 파일의 해시를 확인한다. GT는 모델에 입력하지 않았다. Pearson correlation은 계산하지 않았다. 128토큰 제한에 도달한 QVHighlights 2개도 포함하며, 유효 구간이 없는 답변도 제외하지 않는다. EOS 선택 단계 자체는 이 분석 범위에서 제외했다.

[방법과 재실행](../../../../../../README.md) · [검증 결과](../../../../../../artifacts/runs/token_attention/mtla/pilot/300/dynamics/validation/validation.json)
