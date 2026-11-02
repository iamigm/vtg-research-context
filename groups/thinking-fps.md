# Thinking/Instruct와 FPS

고정pilot300 ×두 입력순서. 2fps Thinking600회, 8fps 두모델1,200응답·trace1,200개 완료 보고. 8fps mIoU Instruct VQ48.49/QV36.96%, Thinking49.24/46.43%; 2fps대비 +.27/+1.08/+.62/−1.48pp. Decoding·토큰예산·공간해상도·768frame cap이 함께 영향을 준다.

## 실행별 보고서·관련 코드

### attention/thinking_pilot_orders_v1

[FINAL_REPORT.md](../reports/experiments/attention/results/thinking_pilot_orders_v1/FINAL_REPORT.md) · [REPORT.md](../reports/experiments/attention/results/thinking_pilot_orders_v1/REPORT.md) · [SCHEMA.md](../reports/experiments/attention/results/thinking_pilot_orders_v1/SCHEMA.md)

현재 관련 코드 13개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/thinking_vs_instruct_8fps_v1

[REPORT.md](../reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/REPORT.md) · [TOKEN_TRACE_REPORT.md](../reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/TOKEN_TRACE_REPORT.md) · [REPORT.md](../reports/experiments/zero-shot/results/thinking_vs_instruct_8fps_v1/performance/REPORT.md)

현재 관련 코드 13개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

### zero-shot/thinking_vs_instruct_v1

[REPORT.md](../reports/experiments/zero-shot/results/thinking_vs_instruct_v1/REPORT.md)

현재 관련 코드 13개. 정확한 파일목록과 설정·상태 evidence는 [실행별 manifest](../EXPERIMENTS.json)에 연결했다.

## 코드 백업

아래는 현재 구현의 연결이다. 역사적 실행 revision의 증명은 아니다. 공용 dependency는 [shared](shared.md)를 참조한다.

- [zero-shot/_lib/compare_thinking_instruct.py](../code/experiments/zero-shot/_lib/compare_thinking_instruct.py)
- [zero-shot/_lib/pilot_8fps.py](../code/experiments/zero-shot/_lib/pilot_8fps.py)
- [zero-shot/_lib/pilot_8fps_traces.py](../code/experiments/zero-shot/_lib/pilot_8fps_traces.py)
- [zero-shot/_lib/report_pilot_8fps.py](../code/experiments/zero-shot/_lib/report_pilot_8fps.py)
- [zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/code/response_only_runner.py](../code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/code/response_only_runner.py)
- [zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps.py](../code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps.py)
- [zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps_traces.py](../code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/before_pilot_8fps_traces.py)
- [zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/benchmark.py](../code/experiments/zero-shot/artifacts/runs/thinking_vs_instruct_8fps_v1/performance/code/benchmark.py)
- [attention/_lib/summarize_thinking_pilot.py](../code/experiments/attention/_lib/summarize_thinking_pilot.py)
- [attention/_lib/thinking_live_benchmark.py](../code/experiments/attention/_lib/thinking_live_benchmark.py)
- [attention/_lib/thinking_pilot.py](../code/experiments/attention/_lib/thinking_pilot.py)
- [attention/_lib/thinking_pilot_live.py](../code/experiments/attention/_lib/thinking_pilot_live.py)
- [attention/artifacts/runs/thinking_pilot_orders_v1/code/source.py](../code/experiments/attention/artifacts/runs/thinking_pilot_orders_v1/code/source.py)
