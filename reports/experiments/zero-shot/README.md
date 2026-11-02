# Zero-shot 평가와 생성 실험

입력·생성 원자료와 파생 분석을 분리한다. 재사용 자료의 분류, 읽기 API, 신규 등록 절차는 **[산출물 저장소 안내](artifacts/README.md)**에서 관리한다. 문서에 응답 개수를 복사하지 않고 각 실행의 상태·manifest를 참조한다.

| 분석 | 보고서 |
|---|---|
| 기존 zero-shot 전체 집계: 실행 버전·전체/pilot·fps·입력 순서 | [통합 보고서](results/all_zeroshot_v1/REPORT.md) · [지표 CSV](artifacts/runs/all_zeroshot_v1/metrics/metrics.csv) |
| 공간 해상도 통제 FPS2/3/4/6/8 zero-shot (선행 실험 완료 후 자동 실행) | [사양](results/fps_spatial_control_pilot300_v1/SPEC.md) · [기존 공간 변화 감사](results/fps_spatial_control_pilot300_v1/SPATIAL_AUDIT.md) · [상태](artifacts/runs/fps_spatial_control_pilot300_v1/metadata/status.json) |
| Thinking vs Instruct 동일 pilot 비교 | [성능·생성량·조건 차이](results/thinking_vs_instruct_v1/REPORT.md) |
| 전체 벤치마크 자릿수별 정답 가능성과 불확실성 | [전체 digit_correctness](results/timelens_top30_full/digit_correctness/REPORT.md) |
| 전체 벤치마크 정답 가능 상태의 세 축 분석 | [전체 blue_analysis](results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md) |
| pilot 자릿수별 정답 가능성과 불확실성 | [pilot digit_correctness](results/timelens_top30/digit_correctness/REPORT.md) |
| pilot 정답 가능 상태의 세 축 분석 | [pilot blue_analysis](results/timelens_top30/digit_correctness/blue_analysis/REPORT.md) |
| 입력 순서와 모델별 zero-shot 성능 | [입력 순서](results/input_order/REPORT.md) |
| 입력 순서별 예측 일관성과 성공 여부 | [예측 일관성](results/input_order/consistency_v1/REPORT.md) · [영상 비교 뷰어](results/input_order/consistency_v1/report.html) |
| 숫자·구간 전용 head 비교 | [제한 head](results/timelens_span_head/REPORT.md) |
| full-head 토큰 확률 | [top-30](results/timelens_top30/REPORT.md) |
| stochastic rollout | [설정·결과](results/timelens_rollout/REPORT.md) |
| 기존 엔트로피 분석 통합 | [통합 보고서](results/timelens_top30/integrated/REPORT.md) |
| 숫자와 다른 토큰의 불확실성 | [숫자 불확실성](results/timelens_top30/numeric_uncertainty/REPORT.md) |

전체 실행 중에는 완료 후 생성되는 보고서 링크가 아직 없을 수 있다. [전체 실행 상태](artifacts/runs/timelens_top30_full/metadata/status.json) · [로그](artifacts/runs/timelens_top30_full/logs/progress.log) · [별도 보관 상태](artifacts/common/status.json)

HTML은 각 결과 폴더에서 영상·토큰 상호작용에 사용한다. 정량 수치와 해석의 원본은 MD다. 전체 뷰어는 localhost:8774, pilot 자릿수 뷰어는 localhost:8773이다.

입력 전처리와 Python 환경은 `../_shared/timelens/`를 공유한다. [Attention 실험](../attention/README.md)은 독립된 소비자다. [기존 predictions 경로 안내](artifacts/common/predictions/legacy/README.md)는 과거 loader 호환성만 설명한다.

최상위의 실제 폴더는 `_lib/`, `results/`, `artifacts/`로 구분한다. 과거 예측은 `artifacts/common/predictions/legacy/`, 캐시는 `artifacts/common/cache/`, 정리 출처는 `artifacts/common/provenance/`에 둔다. 기존 `predictions` 링크는 실행 중인 작업을 위한 임시 경로다.
