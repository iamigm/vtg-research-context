# Appendix · 세 세팅의 zero-shot failure 분석 · pilot200 (200질문)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:4:begin -->
## 대표 영상 검토

[동영상 갤러리](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html)는 GT, Qwen2.5 기본, Qwen2.5 TimeLens-7B식 입력, Qwen3 기본의 네 타임라인을 보여준다. 모델 예측은 세 개다.

<!-- report-part:4:end -->
<!-- report-part:5:begin -->
### charades-timelens/PRTAJ/3

Qwen2.5에서 GT [15,28]초에 대해 native [17,27]초(IoU 0.769)가 변환 후 [38,41]초(IoU 0)로 이동했다. 검토 프레임에서 15초 전후 검은 겉옷을 집어 들고 17–26초에 걸쳐 상체에 걸치는 모습이 보인다. 38–41초에는 이미 옷을 입은 상태다.

해석 후보: 옷을 입는 동작과 입은 뒤의 상태를 시간적으로 구별하지 못한 선택일 수 있다.

검토 한계: 20개 균일 프레임을 통한 시각 검토이며 어두운 장면과 중간 동작 생략 때문에 세부 경계를 확정하지 않는다. 입력 변환의 어느 요소가 원인인지는 이 비교로 분리할 수 없다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#charades-timelens%2FPRTAJ%2F3) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__PRTAJ__3.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__PRTAJ__3.json)

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
### charades-timelens/PZ0ND/0

Qwen2.5에서 GT [5,11]초에 대해 native [0,5]초(IoU 0)가 변환 후 [0,11]초(IoU 0.545)로 바뀌어 성공 기준을 넘었다. 초반에는 기존 상의를 벗는 모습, 약 5–11초에는 새 흰 상의를 집어 입는 모습이 보인다. 변환 결과는 앞선 동작까지 포함하는 넓은 구간이다.

해석 후보: 동작 시작점을 정확히 찾기보다는 구간을 넓혀 GT를 포함하면서 점수가 회복된 사례다.

검토 한계: 회복은 IoU 0.5 기준이며 시작점의 의미적 정확성이 해결됐다는 뜻은 아니다. 프레임 표본으로 연속 동작의 모든 경계를 확인하지 않았다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#charades-timelens%2FPZ0ND%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__PZ0ND__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__PZ0ND__0.json)

<!-- report-part:6:end -->
<!-- report-part:7:begin -->
### charades-timelens/TCJWE/0

Qwen2.5에서 GT [14,19]초에 대해 native [14,20]초(IoU 0.833)가 변환 후 [29,34]초(IoU 0)로 이동했다. 디코딩한 영상 길이는 약 30.7초이므로 예측 끝은 영상 밖이다. 동일한 주방과 인물들이 영상 전반에 반복해서 등장한다.

해석 후보: 비슷한 장면들 가운데 잘못된 시점을 선택하고 종료 시간을 영상 범위 안으로 제한하지 못한 사례다.

검토 한계: 작은 조미료 물체는 균일 검토 프레임만으로 명확히 식별하기 어렵다. 시간 변환 수식이 틀렸다는 증거로 해석하지 않는다. 실제 입력 시간은 원본 프레임 인덱스/FPS와 일치한다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#charades-timelens%2FTCJWE%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__TCJWE__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__TCJWE__0.json)

<!-- report-part:7:end -->
<!-- report-part:8:begin -->
### qvhighlights-timelens/71Fa81Wr-dE_360.0_510.0/0

Qwen2.5는 native에서 GT와 같은 [49,53]초를 찾았지만, 변환 후 원래 응답은 [145.5,146.5]초였다. 공식 Python round 규칙을 적용하면 [146,146]초가 되어 invalid_interval로 분류된다. 원래 소수 구간도 GT와 겹치지 않으므로 반올림만의 실패는 아니다. 49–53초를 별도로 추출해 확인하면 침대에 누워 눈을 감은 아이가 보이지만, 145–147초에는 분홍 외투를 입고 트램펄린 위에서 움직이는 아이가 보인다.

해석 후보: 단순한 끝점 오차를 넘어 다른 행동이 있는 장면을 선택한 경우다. 반올림이 추가로 구간을 0초 길이로 만들었다.

검토 한계: 20개 균일 프레임에는 짧은 GT가 빠져 있어 GT와 예측 주변을 별도로 추출했다. 소수 출력이 어느 시각 표현에 연결됐는지 내부 attention은 검증하지 않았다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#qvhighlights-timelens%2F71Fa81Wr-dE_360.0_510.0%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__71Fa81Wr-dE_360.0_510.0__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__71Fa81Wr-dE_360.0_510.0__0.json)

[GT·예측 주변 확대](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__71Fa81Wr-dE_360.0_510.0__0_detail.jpg)

<!-- report-part:8:end -->
<!-- report-part:9:begin -->
### qvhighlights-timelens/hm4l6e_V-hQ_360.0_510.0/0

크루즈 여행의 경험과 감정을 공유하는 장면이라는 query의 GT는 [0,150]초다. Qwen2.5 native [0,150]초(IoU 1.0)가 변환 후 [0,7]초(IoU 0.047)로 축소됐다. 검토 프레임에는 초반 갑판의 인물 설명, 중간 실내 설명과 이동, 후반 식사 장면이 이어진다.

해석 후보: 넓은 여행 에피소드를 묻는 query를 초반에 직접 일치하는 짧은 장면으로 좁혔을 수 있다.

검토 한계: 시각 입력만 사용했으며 경험·감정의 실제 발화 내용은 확인하지 않았다. GT의 서사적 범위에 대한 annotation 판단과 모델의 짧은 시각 장면 선택을 구분해 해석해야 한다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#qvhighlights-timelens%2Fhm4l6e_V-hQ_360.0_510.0%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__hm4l6e_V-hQ_360.0_510.0__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__hm4l6e_V-hQ_360.0_510.0__0.json)

<!-- report-part:9:end -->
<!-- report-part:10:begin -->
### charades-timelens/TIWRY/0

컵을 내려놓는 query의 GT는 [0,1]초다. Qwen3 기본 입력은 [0,2]초(IoU 0.5)를 답했다. Qwen2.5는 기본 [0,4]초에서 TimeLens-7B식 입력 변환 후 [21,26]초로 이동했다. 후반 20–26초에는 싱크대 주변에서 손을 움직이는 장면이 보인다. 확대 프레임에서는 1.0초에도 컵을 들고 있고 약 1.3–1.7초에 조리대로 손을 뻗어, GT의 세부 경계에도 검토 여지가 있다.

해석 후보: Qwen2.5 변환 결과는 비슷한 장소에서 이어지는 다른 손 동작 구간을 선택했을 수 있다. 짧은 초기 동작의 경계 정보 감소도 가능한 설명이다.

검토 한계: GT가 1초로 매우 짧다. GT 안에 선택 프레임이 있다는 사실만으로 동작의 전후 변화까지 충분히 보존됐다고 할 수 없다. 컵과 손의 미세한 접촉을 균일 프레임만으로 확정하지 않는다. GT는 수정하지 않았다. 이 경계 불확실성이 20초 이후를 선택한 오류를 설명하지는 않는다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#charades-timelens%2FTIWRY%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__TIWRY__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__TIWRY__0.json)

[GT·예측 주변 확대](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__TIWRY__0_detail.jpg)

<!-- report-part:10:end -->
<!-- report-part:11:begin -->
### charades-timelens/81VSN/0

TV 옆을 떠나는 query의 GT는 [19,24]초다. Qwen3 기본 [22,25]초(IoU 0.333)는 동작 후반만 덮는다. Qwen2.5는 기본 [21,27]초에서 변환 후 [39,44]초로 이동한다. 약 20.7초에는 TV 옆을 떠나고 22.8초에는 출구 쪽으로 이동하며, 37–39초에는 다시 방으로 들어오는 모습이 보인다.

해석 후보: Qwen2.5는 떠남과 복귀의 다른 구간을 혼동했을 수 있다. Qwen3 기본 결과는 시작 경계가 늦다.

검토 한계: 균일 검토 프레임을 통한 해석이며 동작의 정확한 시작점을 확정하지 않는다. 사례는 전체 오류 빈도를 대표하지 않는다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#charades-timelens%2F81VSN%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__81VSN__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/charades-timelens__81VSN__0.json)

<!-- report-part:11:end -->
<!-- report-part:12:begin -->
### qvhighlights-timelens/trcIoufeEgM_210.0_360.0/0

나무 옆의 노란 개라는 query의 GT는 [75,96]초다. Qwen3 기본은 [1,3]초(IoU 0)를 답했다. Qwen2.5는 기본 [89,92]초에서 변환 후 [150,152]초로 이동했다. 초반은 숲에서 인물이 설명하는 장면이고, GT 부근은 나무를 다루는 장면이며 94.7초 검토 프레임에 개가 보인다.

해석 후보: Qwen3 기본은 다른 장면을 선택했고, Qwen2.5 기본은 GT의 짧은 일부만 덮었다. Qwen2.5 변환 후에는 영상 밖 구간을 답했다.

검토 한계: 균일 프레임만으로 개가 GT의 모든 순간에 보인다고 판단하지 않는다. 가려짐과 카메라 구도 변화가 있다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#qvhighlights-timelens%2FtrcIoufeEgM_210.0_360.0%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__trcIoufeEgM_210.0_360.0__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__trcIoufeEgM_210.0_360.0__0.json)

<!-- report-part:12:end -->
<!-- report-part:13:begin -->
### qvhighlights-timelens/A_MFAuOwK5k_360.0_510.0/0

스파게티를 만들기 위해 기계를 돌리는 query의 GT는 [23,36]초다. Qwen3 기본은 [17,35]초(IoU 0.632)를 답했다. Qwen2.5는 기본 [20,25]초, 변환 후 [20,24]초로 GT의 초반 일부만 덮었다. 15.8초에는 접시의 반죽, 23.7초에는 기계에 반죽을 넣는 모습, 31.6초에는 기계 옆 가닥 면이 보인다.

해석 후보: Qwen2.5는 넓게 이어지는 기계 조작 구간을 너무 짧게 잡았을 수 있다.

검토 한계: 모든 손잡이 회전을 연속 확인하지 않았다. 준비와 기계 동작의 세부 경계는 추가 검토 여지가 있다.

[동영상 재생](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html#qvhighlights-timelens%2FA_MFAuOwK5k_360.0_510.0%2F0) · [프레임 요약](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__A_MFAuOwK5k_360.0_510.0__0.jpg) · [metadata](../../../../../artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/qvhighlights-timelens__A_MFAuOwK5k_360.0_510.0__0.json)

<!-- report-part:13:end -->
