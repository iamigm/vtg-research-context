<!-- report-appendix:APPENDIX.md parts=73 -->
<!-- report-part:0:begin -->
# 중첩 pilot 300개 · pilot300 (300질문)

[지표 정의와900개 보고서](../900/REPORT.md)

<!-- report-part:0:end -->
## 핵심 지표 정의

IoU=예측·GT 시간 구간의 교집합 길이/합집합 길이, mIoU=평가 query의 IoU 평균, R@τ=IoU≥τ인 query 수/평가 query 수(τ=.3/.5/.7)다. 모두 높을수록 좋으며 % 표시는 100을 곱한다. Δpp=100×(방법−기준)이다. Oracle은 GT로 고른 후보 집합의 사후 상한이며 실제 선택 성능과 구분한다. 반복 응답은 query 안에서 평균한 뒤 query 간 집계한다.

## 데이터셋·공간 조건별 핵심 결과

| Dataset | Spatial target | 조건 | mIoU % (Δpp) | R@.3 % (Δpp) | R@.5 % (Δpp) | R@.7 % (Δpp) |
|---|---|---|---:|---:|---:|---:|
| Charades | low | Uniform | 44.52 (+0.00) | 60.00 (+0.00) | 48.00 (+0.00) | 25.00 (+0.00) |
| Charades | low | GT 전체8배 | 44.91 (+0.38) | 62.00 (+2.00) | 46.00 (-2.00) | 30.00 (+5.00) |
| Charades | low | Random 평균 | 39.34 (-5.19) | 53.50 (-6.50) | 42.50 (-5.50) | 18.00 (-7.00) |
| Charades | high | Uniform | 45.18 (+0.00) | 64.00 (+0.00) | 48.00 (+0.00) | 27.00 (+0.00) |
| Charades | high | GT 전체8배 | 46.64 (+1.46) | 64.00 (+0.00) | 47.00 (-1.00) | 31.00 (+4.00) |
| Charades | high | Random 평균 | 40.54 (-4.64) | 58.50 (-5.50) | 44.50 (-3.50) | 21.50 (-5.50) |
| ActivityNet | low | Uniform | 40.15 (+0.00) | 51.00 (+0.00) | 40.00 (+0.00) | 30.00 (+0.00) |
| ActivityNet | low | GT 전체8배 | 49.83 (+9.68) | 64.00 (+13.00) | 54.00 (+14.00) | 37.00 (+7.00) |
| ActivityNet | low | Random 평균 | 36.65 (-3.50) | 49.00 (-2.00) | 36.50 (-3.50) | 24.50 (-5.50) |
| ActivityNet | high | Uniform | 44.33 (+0.00) | 57.00 (+0.00) | 46.00 (+0.00) | 30.00 (+0.00) |
| ActivityNet | high | GT 전체8배 | 54.27 (+9.93) | 74.00 (+17.00) | 57.00 (+11.00) | 39.00 (+9.00) |
| ActivityNet | high | Random 평균 | 41.12 (-3.22) | 56.00 (-1.00) | 45.50 (-0.50) | 28.50 (-1.50) |
| QVHighlights | low | Uniform | 58.56 (+0.00) | 74.00 (+0.00) | 62.00 (+0.00) | 47.00 (+0.00) |
| QVHighlights | low | GT 전체8배 | 61.93 (+3.36) | 81.00 (+7.00) | 69.00 (+7.00) | 47.00 (+0.00) |
| QVHighlights | low | Random 평균 | 58.80 (+0.24) | 77.00 (+3.00) | 64.50 (+2.50) | 46.00 (-1.00) |
| QVHighlights | high | Uniform | 58.16 (+0.00) | 77.00 (+0.00) | 57.00 (+0.00) | 48.00 (+0.00) |
| QVHighlights | high | GT 전체8배 | 65.49 (+7.33) | 83.00 (+6.00) | 72.00 (+15.00) | 53.00 (+5.00) |
| QVHighlights | high | Random 평균 | 59.25 (+1.09) | 77.50 (+0.50) | 64.00 (+7.00) | 48.00 (+0.00) |

## mIoU 변화의 95% 구간

| Dataset | Spatial target | 비교 | 지표 | 변화 pp [95% 구간] |
|---|---|---|---|---:|
| Charades | low | gt_whole_r8-uniform | mIoU | +0.38 [-3.96, +4.75] |
| Charades | low | gt_whole_r8-random_mean | mIoU | +5.57 [+1.76, +9.42] |
| Charades | high | gt_whole_r8-uniform | mIoU | +1.46 [-2.04, +5.07] |
| Charades | high | gt_whole_r8-random_mean | mIoU | +6.10 [+2.08, +10.24] |
| ActivityNet | low | gt_whole_r8-uniform | mIoU | +9.68 [+3.03, +16.51] |
| ActivityNet | low | gt_whole_r8-random_mean | mIoU | +13.18 [+8.08, +18.60] |
| ActivityNet | high | gt_whole_r8-uniform | mIoU | +9.93 [+4.57, +15.37] |
| ActivityNet | high | gt_whole_r8-random_mean | mIoU | +13.15 [+8.30, +18.16] |
| QVHighlights | low | gt_whole_r8-uniform | mIoU | +3.36 [-2.95, +9.78] |
| QVHighlights | low | gt_whole_r8-random_mean | mIoU | +3.12 [-2.43, +8.57] |
| QVHighlights | high | gt_whole_r8-uniform | mIoU | +7.33 [+2.37, +12.72] |
| QVHighlights | high | gt_whole_r8-random_mean | mIoU | +6.24 [+1.93, +11.02] |

## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
