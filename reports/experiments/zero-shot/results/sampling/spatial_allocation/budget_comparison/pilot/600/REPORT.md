<!-- report-appendix:APPENDIX.md parts=73 -->
<!-- report-part:0:begin -->
# 중첩 pilot 600개 · pilot600 (600질문)

[지표 정의와900개 보고서](../900/REPORT.md)

<!-- report-part:0:end -->
## 핵심 지표 정의

IoU=예측·GT 시간 구간의 교집합 길이/합집합 길이, mIoU=평가 query의 IoU 평균, R@τ=IoU≥τ인 query 수/평가 query 수(τ=.3/.5/.7)다. 모두 높을수록 좋으며 % 표시는 100을 곱한다. Δpp=100×(방법−기준)이다. Oracle은 GT로 고른 후보 집합의 사후 상한이며 실제 선택 성능과 구분한다. 반복 응답은 query 안에서 평균한 뒤 query 간 집계한다.

## 데이터셋·공간 조건별 핵심 결과

| Dataset | Spatial target | 조건 | mIoU % (Δpp) | R@.3 % (Δpp) | R@.5 % (Δpp) | R@.7 % (Δpp) |
|---|---|---|---:|---:|---:|---:|
| Charades | low | Uniform | 44.59 (+0.00) | 61.50 (+0.00) | 44.00 (+0.00) | 27.00 (+0.00) |
| Charades | low | GT 전체8배 | 47.72 (+3.12) | 66.00 (+4.50) | 46.50 (+2.50) | 31.50 (+4.50) |
| Charades | low | Random 평균 | 40.93 (-3.67) | 54.50 (-7.00) | 40.25 (-3.75) | 21.75 (-5.25) |
| Charades | high | Uniform | 46.88 (+0.00) | 64.00 (+0.00) | 48.50 (+0.00) | 30.00 (+0.00) |
| Charades | high | GT 전체8배 | 47.97 (+1.09) | 65.00 (+1.00) | 48.50 (+0.00) | 32.00 (+2.00) |
| Charades | high | Random 평균 | 42.78 (-4.11) | 58.75 (-5.25) | 43.50 (-5.00) | 24.00 (-6.00) |
| ActivityNet | low | Uniform | 44.88 (+0.00) | 58.00 (+0.00) | 45.50 (+0.00) | 33.00 (+0.00) |
| ActivityNet | low | GT 전체8배 | 52.06 (+7.18) | 67.50 (+9.50) | 55.50 (+10.00) | 39.50 (+6.50) |
| ActivityNet | low | Random 평균 | 42.27 (-2.62) | 57.50 (-0.50) | 42.25 (-3.25) | 29.25 (-3.75) |
| ActivityNet | high | Uniform | 48.64 (+0.00) | 64.50 (+0.00) | 50.50 (+0.00) | 35.00 (+0.00) |
| ActivityNet | high | GT 전체8배 | 54.21 (+5.58) | 72.50 (+8.00) | 57.00 (+6.50) | 39.50 (+4.50) |
| ActivityNet | high | Random 평균 | 45.84 (-2.80) | 61.75 (-2.75) | 48.25 (-2.25) | 33.50 (-1.50) |
| QVHighlights | low | Uniform | 58.48 (+0.00) | 74.00 (+0.00) | 61.00 (+0.00) | 47.00 (+0.00) |
| QVHighlights | low | GT 전체8배 | 62.92 (+4.44) | 80.50 (+6.50) | 69.50 (+8.50) | 51.00 (+4.00) |
| QVHighlights | low | Random 평균 | 57.57 (-0.92) | 74.75 (+0.75) | 62.50 (+1.50) | 45.25 (-1.75) |
| QVHighlights | high | Uniform | 60.05 (+0.00) | 78.50 (+0.00) | 61.50 (+0.00) | 49.50 (+0.00) |
| QVHighlights | high | GT 전체8배 | 64.73 (+4.68) | 82.00 (+3.50) | 71.50 (+10.00) | 53.50 (+4.00) |
| QVHighlights | high | Random 평균 | 58.53 (-1.52) | 76.50 (-2.00) | 62.00 (+0.50) | 47.75 (-1.75) |

## mIoU 변화의 95% 구간

| Dataset | Spatial target | 비교 | 지표 | 변화 pp [95% 구간] |
|---|---|---|---|---:|
| Charades | low | gt_whole_r8-uniform | mIoU | +3.12 [+0.26, +5.97] |
| Charades | low | gt_whole_r8-random_mean | mIoU | +6.79 [+4.35, +9.29] |
| Charades | high | gt_whole_r8-uniform | mIoU | +1.09 [-1.34, +3.52] |
| Charades | high | gt_whole_r8-random_mean | mIoU | +5.20 [+2.74, +7.75] |
| ActivityNet | low | gt_whole_r8-uniform | mIoU | +7.18 [+2.60, +11.76] |
| ActivityNet | low | gt_whole_r8-random_mean | mIoU | +9.80 [+6.15, +13.57] |
| ActivityNet | high | gt_whole_r8-uniform | mIoU | +5.58 [+1.46, +9.66] |
| ActivityNet | high | gt_whole_r8-random_mean | mIoU | +8.37 [+4.72, +12.09] |
| QVHighlights | low | gt_whole_r8-uniform | mIoU | +4.44 [+0.02, +8.85] |
| QVHighlights | low | gt_whole_r8-random_mean | mIoU | +5.35 [+1.61, +9.15] |
| QVHighlights | high | gt_whole_r8-uniform | mIoU | +4.68 [+1.18, +8.33] |
| QVHighlights | high | gt_whole_r8-random_mean | mIoU | +6.20 [+2.78, +9.72] |

## Appendix

[상세 방법·조건별 결과·검증](APPENDIX.md)는 상단 `appendix` 메뉴에서도 열 수 있다.
