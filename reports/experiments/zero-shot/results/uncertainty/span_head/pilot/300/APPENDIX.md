# Appendix · Span 전용 LM head와 zero-shot 비교 · pilot300 (300질문)

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:4:begin -->
## 출력 안정성

각 값은 **zero-shot → 제한 head**다. 평균 토큰 수에는 EOS가 포함된다. 구간 오류는 반올림 후 시작 ≥ 끝인 경우이며, 범위 초과는 첫 구간을 annotation duration과 비교한 값이다. 범위 초과 집계에는 파싱 실패에 부여한 fallback도 포함된다. 512토큰 도달 자체를 점수 0으로 처리하지는 않으며, 파싱 가능한 첫 구간은 그대로 평가한다.

| 모델 | 벤치마크 | 평균 출력 토큰 | 512토큰 도달 | 파싱 실패 | 구간 오류 | 다중 구간 | 범위 초과 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Charades-TimeLens | 17.30 → 12.78 | 0 → 0 | 0 → 1 | 0 → 0 | 0 → 2 | 11 → 17 |
| Qwen2.5-VL-7B | ActivityNet-TimeLens | 18.13 → 13.37 | 0 → 0 | 0 → 0 | 0 → 1 | 0 → 1 | 6 → 6 |
| Qwen2.5-VL-7B | QVHighlights-TimeLens | 18.48 → 13.67 | 0 → 0 | 0 → 0 | 0 → 2 | 0 → 2 | 0 → 2 |
| Qwen3-VL-8B | Charades-TimeLens | 17.14 → 184.30 | 0 → 33 | 0 → 0 | 0 → 0 | 0 → 0 | 16 → 13 |
| Qwen3-VL-8B | ActivityNet-TimeLens | 18.22 → 109.33 | 0 → 18 | 0 → 0 | 0 → 0 | 0 → 0 | 9 → 10 |
| Qwen3-VL-8B | QVHighlights-TimeLens | 18.48 → 43.97 | 0 → 5 | 0 → 0 | 0 → 1 | 0 → 0 | 0 → 1 |

<!-- report-part:4:end -->
<!-- report-part:6:begin -->
## 검증과 재현

- Qwen3 기준선 재현 기록 17건에서 저장된 생성 토큰과의 일치를 확인했다. 재시작 및 smoke 검사를 포함하므로 고유 샘플 수는 아니다.
- Qwen3의 저장된 입력 문자열·프레임 sampling metadata를 대조했고, Qwen2.5의 두 비교군은 같은 입력 텐서와 input-ID 해시를 사용했다.
- 모든 생성 토큰이 허용 목록에 속하는지, sample ID 중복이 없는지, 원래 예측의 해시가 유지되는지 확인했다.
- 공식 parser·반올림·첫 구간 IoU를 적용했고, 공식 evaluator CLI의 12개 평가군과 집계값이 일치했다.
- GT는 평가에만 사용했다. 이전 경로의 잘못된 cohort로 중단한 실행은 `../../../../../artifacts/runs/uncertainty/span_head/pilot/300/provenance/aborted_old_cohort`에 보존했고 본 집계에서 제외했다.

[정량 JSON](../../../../../artifacts/runs/uncertainty/span_head/pilot/300/metrics/summary.json) · [샘플별 CSV](../../../../../artifacts/runs/uncertainty/span_head/pilot/300/metrics/paired.csv) · [샘플별 JSON](../../../../../artifacts/runs/uncertainty/span_head/pilot/300/metrics/paired.json) · [고정 샘플](../../../../../artifacts/common/predictions/legacy/cohorts/pilot_300/samples.json) · [그림 PDF](../../../../../artifacts/runs/uncertainty/span_head/pilot/300/figures/comparison.pdf) · [완료 상태](../../../../../artifacts/runs/uncertainty/span_head/pilot/300/metadata/status.json)

```bash
cd /home1/irteam/experiments/zero-shot
# 저장된 측정값으로 Markdown만 재생성; 추론·평가 재실행 없음
.venv/bin/python _lib/timelens_span_report.py --render-only
# 예측부터 집계·공식 평가 검증·그림·Markdown을 재생성
.venv/bin/python run.py span-head report
```

정량 결과와 해석은 Markdown으로 제공한다. HTML은 실제 영상 재생 등 상호작용이 필요한 분석에만 사용한다.
<!-- report-part:6:end -->
