# 8fps 수집 방식 및 성능 개선

새 응답의 token trace는 생성과 동시에 수집한다. trace 수집을 켜기 전에 완료된 응답만 저장된 생성 토큰을 고정하여 replay로 보완한다. 기존 trace가 있는 응답은 건너뛴다. Attention 행렬을 추출하는 추가 replay는 없다. Rollout은 수행하지 않는다.

매 토큰의 raw/processed top30 및 통계를 GPU의 작은 tensor로 유지하다가 응답 종료 시 한 번에 CPU로 옮긴다. Generation processor가 없는 greedy에서는 동일 raw/processed 분포를 한 번만 계산한다. 모든 값과 schema를 유지하고 JSON 공백만 줄였다.

각 샘플의 두 입력 순서는 기존 prepare_pair에서 비텍스트 입력이 bitwise 동일함을 검사한다. 이 쌍 안에서만 영상 인코더 출력을 재사용하며 샘플·모델이 바뀌면 cache를 비운다. 텍스트 decoder와 입력 순서별 조건부 생성은 각각 실행한다.

한 개 8fps 입력에서 기존/개선 수집기를 비교했다. Greedy와 seed42 sampling 모두 생성 토큰, raw/processed top30, 선택 확률·순위, entropy·질량 통계가 정확히 일치했다. 수집 개선은 각각 약 3.6% 가속이었다. 개선 수집기를 사용한 두 순서 쌍에서 vision cache 미사용 2.616초, 사용 2.393초로 약 9.3% 추가 가속이었다. 두 순서 모두 생성 토큰과 전체 trace가 정확히 일치했다.

단일 Charades 샘플 및 짧은 응답의 측정이므로 전체 cohort나 긴 Thinking 응답의 속도 향상률로 해석하지 않는다. 입력 대기·생성 및 GPU 수집·CPU materialization·trace 저장 시간과 vision cache hit를 응답마다 performance.json으로 기록한다.

[검증 및 측정 원자료](../../../artifacts/runs/thinking_vs_instruct_8fps_v1/performance/metadata/benchmark.json) · [실행 상태](../../../artifacts/runs/thinking_vs_instruct_8fps_v1/metadata/status.json)
