# 폐기한 2fps→8fps 실험의 실제 공간 해상도 감사

동일한 고정 pilot300의 Instruct 영상→질문 입력과 기존2fps metadata를 짝지었다. 8fps 두 모델·두 순서의 grid1,200개가 같음도 확인했다. 프레임 면적=video_grid_thw[1]×video_grid_thw[2]×16²픽셀. 평균은 질문별 동일 가중치이며 같은 영상의 다른 질문도 각각 센다. 면적 감소율=100×(1−8fps 평균면적/2fps 평균면적), 샘플별 면적비 중앙값은 median(각8fps면적/각2fps면적)이다. 감소는 프레임당 공간 정보가 줄어드는 방향이다.

| 데이터셋 | N | 해상도 변경 | 평균 면적 2fps | 평균 면적 8fps | 면적 감소 | 샘플별 면적비 중앙값 | 768프레임 도달 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 300 | 282 | 204,049 | 82,169 | 59.7% | 0.659 | 168 |
| activitynet-timelens | 100 | 86 | 123,269 | 63,222 | 48.7% | 0.642 | 68 |
| charades-timelens | 100 | 96 | 395,653 | 121,815 | 69.2% | 0.264 | 0 |
| qvhighlights-timelens | 100 | 100 | 93,225 | 61,471 | 34.1% | 0.659 | 100 |

Charades 예시: charades-timelens_00123은1024×576→512×288, 589,824→147,456픽셀로 면적이75% 감소했다. 전체300개 중282개(94%)가 변경됐다. QVHighlights100개와 ActivityNet68개가768프레임에 도달했으므로 요청8fps와 실제샘플링률도 일치하지 않을 수 있다.

이 실험의 예측·trace·성능 보고서는 사용자 요청으로 삭제했다. 해상도 통제가 필요하다는 근거인 위 진단 수치와 입력 출처 hash만 보존한다. 새FPS sweep의 성능 기준선으로 재사용하지 않는다.

[진단 원자료](../../artifacts/runs/fps_spatial_control_pilot300_v1/diagnostics/discarded_2fps_8fps_spatial_audit.json) · [새 실험 사양](SPEC.md) · [전체 실험 목록](../../../index.html)
