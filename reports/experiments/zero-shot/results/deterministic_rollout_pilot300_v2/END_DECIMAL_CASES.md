# 종료 시각의 소수점 분기 사례

종료 시각의 소수점 기호 `.` 자체를 다른 토큰으로 바꾼 분기다. 내부 코드 `end.seconds.decimal_separator`는 end=종료 시각, seconds=초 필드, decimal_separator=소수점 자체라는 뜻이다. 소수점 아래 숫자를 조정한 분기와 다르다.

300질문 중 19질문에서 59개 분기가 있었다. 최초 greedy의 종료가1.x초인데 GT 종료가100∼199초인 질문은 0개다. 탐색 중 생긴 부모 후보까지 포함하면 1개다. 이 두 범위를 혼동하지 않는다.

[모든 샘플 HTML](viewer.html) · [종료 소수점 전체](viewer.html#preset=dot) · [최초 greedy부터 1초대였던 경우](viewer.html#preset=greedy_short) · [탐색 부모까지 포함한 경우](viewer.html#preset=parent_short)

## GT 종료는100초대인데 부모가1초대로 답한 사례

아래 각 질문에서는 이 조건에 해당하는 분기 중 부모 대비 IoU 개선이 가장 큰 분기를 표시했다. GT와의 IoU가 낮다고 다른 사건이라고 단정하지 않는다. 모델의 내부 의도는 알 수 없지만, `1.`에서 정수 부분을 끝내어 실제 출력이1초대가 된 것은 확인할 수 있다.

| 샘플 | 최초 응답도 해당? | GT(초) | 최초 greedy | 분기 직전 부모 | 분기 후 후보 | 부모 대비 ΔIoU |
|---|---|---|---|---|---|---:|
| [qvhighlights-timelens_00759](viewer.html#sample=qvhighlights-timelens_00759&candidate=30&preset=parent_short) | 아니오, 탐색 부모만 | [0, 117] | The event happens in 0.0 - 48.0 seconds. | The event happens in 0.0 - 1.0 seconds. | The event happens in 0.0 - 10.0 seconds. | +0.0769 |

## 종료 소수점 분기가 있었던 전체19개 질문

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
