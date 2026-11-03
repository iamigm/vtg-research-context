# 종료 시각의 소수점 분기 사례

종료 시각의 소수점 기호 `.` 자체를 다른 토큰으로 바꾼 분기다. 내부 코드 `end.seconds.decimal_separator`는 end=종료 시각, seconds=초 필드, decimal_separator=소수점 자체라는 뜻이다. 소수점 아래 숫자를 조정한 분기와 다르다.

900질문 중 84질문에서 243개 분기가 있었다. 최초 greedy의 종료가1.x초인데 GT 종료가100∼199초인 질문은 2개다. 탐색 중 생긴 부모 후보까지 포함하면 4개다. 이 두 범위를 혼동하지 않는다.

[모든 샘플 HTML](viewer.html) · [종료 소수점 전체](viewer.html#preset=dot) · [최초 greedy부터 1초대였던 경우](viewer.html#preset=greedy_short) · [탐색 부모까지 포함한 경우](viewer.html#preset=parent_short)

## GT 종료는100초대인데 부모가1초대로 답한 사례

아래 각 질문에서는 이 조건에 해당하는 분기 중 부모 대비 IoU 개선이 가장 큰 분기를 표시했다. GT와의 IoU가 낮다고 다른 사건이라고 단정하지 않는다. 모델의 내부 의도는 알 수 없지만, `1.`에서 정수 부분을 끝내어 실제 출력이1초대가 된 것은 확인할 수 있다.

| 샘플 | 최초 응답도 해당? | GT(초) | 최초 greedy | 분기 직전 부모 | 분기 후 후보 | 부모 대비 ΔIoU |
|---|---|---|---|---|---|---:|
| [qvhighlights-timelens_00759](viewer.html#sample=qvhighlights-timelens_00759&candidate=30&preset=parent_short) | 아니오, 탐색 부모만 | [0, 117] | The event happens in 0.0 - 48.0 seconds. | The event happens in 0.0 - 1.0 seconds. | The event happens in 0.0 - 10.0 seconds. | +0.0769 |
| [activitynet-timelens_01723](viewer.html#sample=activitynet-timelens_01723&candidate=3&preset=parent_short) | 예 | [0, 106] | The event happens in 0.0 - 1.3 seconds. | The event happens in 0.0 - 1.3 seconds. | The event happens in 0.0 - 107.3 seconds. | +0.9756 |
| [qvhighlights-timelens_00841](viewer.html#sample=qvhighlights-timelens_00841&candidate=16&preset=parent_short) | 아니오, 탐색 부모만 | [80, 115] | The event happens in 1.0 - 116.0 seconds. | The event happens in 1.2 - 1.7 seconds seconds. | The event happens in 1.2 - 116.8 seconds. | +0.3028 |
| [qvhighlights-timelens_00507](viewer.html#sample=qvhighlights-timelens_00507&candidate=18&preset=parent_short) | 예 | [115, 146] | The event happens in 0.0 - 1.3 seconds seconds. | The event happens in 0.0 - 1.3 seconds seconds. | The event happens in 0.0 - 10.0 seconds. | +0.0000 |

## 종료 소수점 분기가 있었던 전체84개 질문

각 질문을 클릭하면 해당 후보로 이동한다. 위 조건에 해당하면 해당 조건 분기를, 그 외에는 부모 대비 개선이 가장 큰 소수점 분기를 표시한다. 평균이 아닌 개별 사례 목록이다. Δ부모=후보 IoU−분기 직전 부모 IoU, Δgreedy=후보 IoU−최초 응답 IoU. 두 값은 다를 수 있다.

| 샘플 | GT(초) | 분기 수 | 비교할 부모 → 후보 | Δ부모 | Δgreedy |
|---|---|---:|---|---:|---:|
| [charades-timelens_01003](viewer.html#sample=charades-timelens_01003&candidate=17&preset=dot) | [2, 3] | 1 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.1000 | -0.2333 |
| [charades-timelens_01186](viewer.html#sample=charades-timelens_01186&candidate=28&preset=dot) | [7, 13] | 3 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 12.0 seconds. | +0.3846 | +0.3846 |
| [charades-timelens_02466](viewer.html#sample=charades-timelens_02466&candidate=16&preset=dot) | [2, 19] | 1 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 32.3 seconds. | +0.4737 | +0.4737 |
| [charades-timelens_02575](viewer.html#sample=charades-timelens_02575&candidate=25&preset=dot) | [0, 28] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 12.0 seconds. | +0.3929 | +0.3929 |
| [charades-timelens_02709](viewer.html#sample=charades-timelens_02709&candidate=30&preset=dot) | [0, 27] | 2 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 29.8 seconds. | +0.8320 | +0.2764 |
| [charades-timelens_02886](viewer.html#sample=charades-timelens_02886&candidate=7&preset=dot) | [0, 20] | 3 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 19.0 seconds. | +0.8900 | +0.8900 |
| [charades-timelens_02987](viewer.html#sample=charades-timelens_02987&candidate=28&preset=dot) | [18, 19] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 20.8 seconds. | +0.0481 | -0.2460 |
| [activitynet-timelens_00745](viewer.html#sample=activitynet-timelens_00745&candidate=32&preset=dot) | [3, 4] | 1 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.1000 | +0.1000 |
| [activitynet-timelens_01629](viewer.html#sample=activitynet-timelens_01629&candidate=19&preset=dot) | [44, 45] | 1 | The event happens in 0.0 - 1.2 seconds seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [activitynet-timelens_01717](viewer.html#sample=activitynet-timelens_01717&candidate=29&preset=dot) | [0, 106] | 4 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.0 seconds. | +0.2547 | -0.7142 |
| [activitynet-timelens_02578](viewer.html#sample=activitynet-timelens_02578&candidate=25&preset=dot) | [0, 16] | 9 | The event happens in 0.0 - 10.0 seconds. → The event happens in 0.0 - 107.7 seconds. | -0.4764 | -0.5389 |
| [qvhighlights-timelens_00408](viewer.html#sample=qvhighlights-timelens_00408&candidate=24&preset=dot) | [20, 150] | 9 | The event happens in 0.0 - 7.0 seconds. → The event happens in 0.0 - 78.0 seconds. | +0.3867 | -0.4788 |
| [qvhighlights-timelens_00665](viewer.html#sample=qvhighlights-timelens_00665&candidate=11&preset=dot) | [1, 150] | 3 | The event happens in 0.0 - 22.8 seconds. → The event happens in 0.0 - 227.8 seconds. | +0.5087 | +0.5087 |
| [qvhighlights-timelens_00748](viewer.html#sample=qvhighlights-timelens_00748&candidate=20&preset=dot) | [5, 18] | 2 | The event happens in 0.0 - 3.5 seconds. → The event happens in 0.0 - 34.7 seconds. | +0.3746 | -0.3024 |
| [qvhighlights-timelens_00759](viewer.html#sample=qvhighlights-timelens_00759&candidate=30&preset=dot) | [0, 117] | 4 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0769 | -0.3248 |
| [qvhighlights-timelens_00778](viewer.html#sample=qvhighlights-timelens_00778&candidate=12&preset=dot) | [48, 80] | 3 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 112.0 seconds. | +0.2857 | +0.2857 |
| [qvhighlights-timelens_01183](viewer.html#sample=qvhighlights-timelens_01183&candidate=14&preset=dot) | [0, 84] | 3 | The event happens in 0.0 - 1.7 seconds. → The event happens in 0.0 - 17.5 seconds. | +0.1881 | -0.7857 |
| [qvhighlights-timelens_01355](viewer.html#sample=qvhighlights-timelens_01355&candidate=31&preset=dot) | [0, 150] | 3 | The event happens in 0.0 - 3.8 seconds. → The event happens in 0.0 - 37.0 seconds. | +0.2213 | +0.1500 |
| [qvhighlights-timelens_01509](viewer.html#sample=qvhighlights-timelens_01509&candidate=20&preset=dot) | [127, 150] | 1 | The event happens in 0.0 - 14.9 seconds. → The event happens in 0.0 - 149.7 seconds. | +0.1513 | -0.0017 |
| [activitynet-timelens_03675](viewer.html#sample=activitynet-timelens_03675&candidate=27&preset=dot) | [2, 91] | 4 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 32.0 seconds. | +0.3187 | -0.1455 |
| [activitynet-timelens_02821](viewer.html#sample=activitynet-timelens_02821&candidate=12&preset=dot) | [12, 85] | 1 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 25.0 seconds. | +0.1529 | +0.1129 |
| [activitynet-timelens_02885](viewer.html#sample=activitynet-timelens_02885&candidate=10&preset=dot) | [6, 150] | 1 | The event happens in 0.0 - 2.3 seconds. → The event happens in 0.0 - 23.5 seconds. | +0.1167 | -0.0639 |
| [activitynet-timelens_03353](viewer.html#sample=activitynet-timelens_03353&candidate=22&preset=dot) | [3, 25] | 3 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.0 seconds. | +0.7333 | +0.6240 |
| [activitynet-timelens_03166](viewer.html#sample=activitynet-timelens_03166&candidate=14&preset=dot) | [0, 194] | 3 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 32.0 seconds. | +0.1495 | +0.1134 |
| [activitynet-timelens_01723](viewer.html#sample=activitynet-timelens_01723&candidate=3&preset=dot) | [0, 106] | 3 | The event happens in 0.0 - 1.3 seconds. → The event happens in 0.0 - 107.3 seconds. | +0.9756 | +0.9756 |
| [activitynet-timelens_03638](viewer.html#sample=activitynet-timelens_03638&candidate=7&preset=dot) | [0, 175] | 1 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 25.0 seconds. | +0.1314 | -0.8486 |
| [activitynet-timelens_00154](viewer.html#sample=activitynet-timelens_00154&candidate=18&preset=dot) | [0, 12] | 2 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.8 seconds. | +0.1396 | -0.4555 |
| [activitynet-timelens_01668](viewer.html#sample=activitynet-timelens_01668&candidate=29&preset=dot) | [82, 89] | 1 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [activitynet-timelens_01347](viewer.html#sample=activitynet-timelens_01347&candidate=23&preset=dot) | [1, 45] | 6 | The event happens in 0.0 - 4.0 seconds. → The event happens in 0.0 - 45.0 seconds. | +0.9111 | +0.5715 |
| [activitynet-timelens_02810](viewer.html#sample=activitynet-timelens_02810&candidate=17&preset=dot) | [0, 154] | 4 | The event happens in 0.0 - 14.0 seconds. → The event happens in 0.0 - 149.0 seconds. | +0.8766 | -0.0312 |
| [activitynet-timelens_04235](viewer.html#sample=activitynet-timelens_04235&candidate=16&preset=dot) | [0, 219] | 10 | The event happens in 0.0 - 12.0 seconds. → The event happens in 0.0 - 122.0 seconds. | +0.5023 | +0.5114 |
| [activitynet-timelens_00477](viewer.html#sample=activitynet-timelens_00477&candidate=10&preset=dot) | [0, 41] | 3 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 35.0 seconds. | +0.7805 | +0.7805 |
| [activitynet-timelens_00096](viewer.html#sample=activitynet-timelens_00096&candidate=15&preset=dot) | [0, 42] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 14.0 seconds. | +0.3095 | -0.6212 |
| [activitynet-timelens_03004](viewer.html#sample=activitynet-timelens_03004&candidate=11&preset=dot) | [78, 141] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 24.0 seconds. | +0.0000 | +0.0000 |
| [charades-timelens_03285](viewer.html#sample=charades-timelens_03285&candidate=27&preset=dot) | [1, 30] | 2 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.6 seconds. | +0.8810 | +0.6077 |
| [charades-timelens_03338](viewer.html#sample=charades-timelens_03338&candidate=21&preset=dot) | [2, 28] | 2 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 11.0 seconds. | +0.3214 | -0.5973 |
| [charades-timelens_02725](viewer.html#sample=charades-timelens_02725&candidate=20&preset=dot) | [0, 19] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 20.0 seconds. | +0.8447 | +0.1237 |
| [charades-timelens_03095](viewer.html#sample=charades-timelens_03095&candidate=12&preset=dot) | [0, 9] | 2 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.7889 | +0.0111 |
| [charades-timelens_01803](viewer.html#sample=charades-timelens_01803&candidate=30&preset=dot) | [10, 13] | 1 | The event happens in 0.0 - 1.3 seconds. → The event happens in 0.0 - 12.0 seconds. | +0.1538 | +0.1538 |
| [charades-timelens_01611](viewer.html#sample=charades-timelens_01611&candidate=29&preset=dot) | [0, 16] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 20.0 seconds. | +0.6750 | -0.1143 |
| [charades-timelens_00462](viewer.html#sample=charades-timelens_00462&candidate=17&preset=dot) | [0, 3] | 3 | The event happens in 0.0 - 1.3 seconds. → The event happens in 0.0 - 13.7 seconds. | -0.2144 | -0.7810 |
| [charades-timelens_00621](viewer.html#sample=charades-timelens_00621&candidate=24&preset=dot) | [10, 30] | 3 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 12.0 seconds. | +0.0667 | -0.1333 |
| [charades-timelens_00206](viewer.html#sample=charades-timelens_00206&candidate=31&preset=dot) | [1, 4] | 2 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.3000 | -0.2000 |
| [charades-timelens_02866](viewer.html#sample=charades-timelens_02866&candidate=10&preset=dot) | [8, 9] | 2 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.1000 | +0.1000 |
| [qvhighlights-timelens_00517](viewer.html#sample=qvhighlights-timelens_00517&candidate=21&preset=dot) | [0, 150] | 4 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 38.0 seconds. | +0.2333 | -0.6133 |
| [qvhighlights-timelens_00841](viewer.html#sample=qvhighlights-timelens_00841&candidate=16&preset=dot) | [80, 115] | 4 | The event happens in 1.2 - 1.7 seconds seconds. → The event happens in 1.2 - 116.8 seconds. | +0.3028 | -0.0016 |
| [qvhighlights-timelens_01083](viewer.html#sample=qvhighlights-timelens_01083&candidate=21&preset=dot) | [0, 150] | 1 | The event happens in 0.0 - 2.2 seconds. → The event happens in 0.0 - 22.0 seconds. | +0.1320 | -0.8320 |
| [qvhighlights-timelens_00791](viewer.html#sample=qvhighlights-timelens_00791&candidate=19&preset=dot) | [0, 133] | 2 | The event happens in 0.0 - 15.0 seconds. → The event happens in 0.0 - 150.0 seconds. | +0.7739 | -0.1000 |
| [qvhighlights-timelens_00589](viewer.html#sample=qvhighlights-timelens_00589&candidate=15&preset=dot) | [9, 21] | 9 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.5 seconds. | +0.5333 | +0.5333 |
| [qvhighlights-timelens_00163](viewer.html#sample=qvhighlights-timelens_00163&candidate=31&preset=dot) | [0, 45] | 3 | The event happens in 0.0 - 13.0 seconds. → The event happens in 0.0 - 132.0 seconds. | +0.0520 | -0.5869 |
| [qvhighlights-timelens_01428](viewer.html#sample=qvhighlights-timelens_01428&candidate=15&preset=dot) | [0, 123] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.0 seconds. | +0.1626 | -0.7746 |
| [qvhighlights-timelens_00592](viewer.html#sample=qvhighlights-timelens_00592&candidate=31&preset=dot) | [44, 72] | 1 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [qvhighlights-timelens_00680](viewer.html#sample=qvhighlights-timelens_00680&candidate=10&preset=dot) | [10, 150] | 2 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.8 seconds. | +0.0853 | -0.0075 |
| [qvhighlights-timelens_00459](viewer.html#sample=qvhighlights-timelens_00459&candidate=30&preset=dot) | [2, 147] | 1 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.0 seconds. | +0.1361 | +0.0905 |
| [qvhighlights-timelens_01185](viewer.html#sample=qvhighlights-timelens_01185&candidate=28&preset=dot) | [64, 80] | 1 | The event happens in 0.0 - 1.3 seconds seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | -0.1875 |
| [qvhighlights-timelens_00829](viewer.html#sample=qvhighlights-timelens_00829&candidate=30&preset=dot) | [0, 43] | 2 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 39.5 seconds. | +0.8488 | +0.3884 |
| [qvhighlights-timelens_01112](viewer.html#sample=qvhighlights-timelens_01112&candidate=22&preset=dot) | [0, 150] | 3 | The event happens in 0.0 - 22.9 seconds. → The event happens in 0.0 - 227.0 seconds. | +0.5081 | -0.3379 |
| [qvhighlights-timelens_00876](viewer.html#sample=qvhighlights-timelens_00876&candidate=29&preset=dot) | [0, 150] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.0 seconds. | +0.1333 | -0.2600 |
| [qvhighlights-timelens_01397](viewer.html#sample=qvhighlights-timelens_01397&candidate=6&preset=dot) | [0, 43] | 2 | The event happens in 0.0 - 5.0 seconds. → The event happens in 0.0 - 54.5 seconds. | +0.6727 | -0.1561 |
| [activitynet-timelens_00452](viewer.html#sample=activitynet-timelens_00452&candidate=14&preset=dot) | [7, 8] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.1000 | -0.2333 |
| [activitynet-timelens_00875](viewer.html#sample=activitynet-timelens_00875&candidate=17&preset=dot) | [64, 73] | 2 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [activitynet-timelens_00726](viewer.html#sample=activitynet-timelens_00726&candidate=15&preset=dot) | [0, 65] | 6 | The event happens in 0.0 - 4.0 seconds. → The event happens in 0.0 - 47.0 seconds. | +0.6615 | -0.2723 |
| [activitynet-timelens_00038](viewer.html#sample=activitynet-timelens_00038&candidate=21&preset=dot) | [10, 56] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 12.0 seconds. | +0.0357 | +0.0357 |
| [activitynet-timelens_02018](viewer.html#sample=activitynet-timelens_02018&candidate=32&preset=dot) | [0, 132] | 3 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 22.0 seconds. | +0.1515 | -0.8273 |
| [activitynet-timelens_00889](viewer.html#sample=activitynet-timelens_00889&candidate=14&preset=dot) | [0, 64] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 14.0 seconds. | +0.2031 | -0.7781 |
| [activitynet-timelens_03720](viewer.html#sample=activitynet-timelens_03720&candidate=26&preset=dot) | [0, 21] | 3 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.0 seconds. | +0.5571 | +0.5963 |
| [activitynet-timelens_00719](viewer.html#sample=activitynet-timelens_00719&candidate=29&preset=dot) | [0, 66] | 6 | The event happens in 0.0 - 4.0 seconds. → The event happens in 0.0 - 42.0 seconds. | +0.5758 | -0.3591 |
| [activitynet-timelens_00468](viewer.html#sample=activitynet-timelens_00468&candidate=18&preset=dot) | [43, 44] | 2 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [activitynet-timelens_00455](viewer.html#sample=activitynet-timelens_00455&candidate=5&preset=dot) | [47, 56] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [charades-timelens_01888](viewer.html#sample=charades-timelens_01888&candidate=17&preset=dot) | [0, 10] | 2 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.8800 | +0.8800 |
| [charades-timelens_00707](viewer.html#sample=charades-timelens_00707&candidate=12&preset=dot) | [1, 7] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 11.2 seconds. | +0.5357 | +0.5357 |
| [charades-timelens_01270](viewer.html#sample=charades-timelens_01270&candidate=20&preset=dot) | [20, 21] | 1 | The event happens in 0.0 - 2.0 seconds. → The event happens in 0.0 - 23.6 seconds. | +0.0424 | -0.1902 |
| [charades-timelens_00164](viewer.html#sample=charades-timelens_00164&candidate=23&preset=dot) | [14, 15] | 1 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [charades-timelens_00774](viewer.html#sample=charades-timelens_00774&candidate=31&preset=dot) | [4, 7] | 1 | The event happens in 0.0 - 1.3 seconds. → The event happens in 0.0 - 15.5 seconds. | +0.1935 | -0.1398 |
| [charades-timelens_02259](viewer.html#sample=charades-timelens_02259&candidate=9&preset=dot) | [0, 25] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 18.0 seconds. | +0.6800 | -0.2681 |
| [charades-timelens_01491](viewer.html#sample=charades-timelens_01491&candidate=23&preset=dot) | [30, 36] | 2 | The event happens in 0.0 - 1.2 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | -0.6667 |
| [charades-timelens_02996](viewer.html#sample=charades-timelens_02996&candidate=7&preset=dot) | [25, 30] | 2 | The event happens in 0.0 - 3.0 seconds. → The event happens in 0.0 - 30.3 seconds. | +0.1650 | +0.0717 |
| [charades-timelens_00115](viewer.html#sample=charades-timelens_00115&candidate=18&preset=dot) | [6, 8] | 3 | The event happens in 0.0 - 1.0 seconds. → The event happens in 0.0 - 10.0 seconds. | +0.2000 | +0.2000 |
| [qvhighlights-timelens_01313](viewer.html#sample=qvhighlights-timelens_01313&candidate=18&preset=dot) | [0, 150] | 3 | The event happens in 0.0 - 7.0 seconds. → The event happens in 0.0 - 78.0 seconds. | +0.4733 | -0.4787 |
| [qvhighlights-timelens_00010](viewer.html#sample=qvhighlights-timelens_00010&candidate=26&preset=dot) | [0, 142] | 4 | The event happens in 0.0 - 4.0 seconds. → The event happens in 0.0 - 40.0 seconds. | +0.2535 | +0.1796 |
| [qvhighlights-timelens_00507](viewer.html#sample=qvhighlights-timelens_00507&candidate=18&preset=dot) | [115, 146] | 4 | The event happens in 0.0 - 1.3 seconds seconds. → The event happens in 0.0 - 10.0 seconds. | +0.0000 | +0.0000 |
| [qvhighlights-timelens_00223](viewer.html#sample=qvhighlights-timelens_00223&candidate=20&preset=dot) | [24, 51] | 2 | The event happens in 0.0 - 2.5 seconds seconds. → The event happens in 0.0 - 23.8 seconds seconds. | +0.0000 | +0.0000 |
| [qvhighlights-timelens_01049](viewer.html#sample=qvhighlights-timelens_01049&candidate=23&preset=dot) | [0, 150] | 4 | The event happens in 0.0 - 22.9 seconds. → The event happens in 0.0 - 229.7 seconds. | +0.5004 | -0.3450 |
| [qvhighlights-timelens_00444](viewer.html#sample=qvhighlights-timelens_00444&candidate=19&preset=dot) | [17, 21] | 3 | The event happens in 0.0 - 2.0 seconds seconds. → The event happens in 0.0 - 24.0 seconds. | +0.1667 | -0.2333 |
