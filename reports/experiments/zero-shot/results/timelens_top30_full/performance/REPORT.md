# 전체 벤치마크 추론 병목 조사

병목은 CPU 영상 디코딩·크기 조정 및 입력 공급이다. 모델 생성 설정은 유지하고 모델별 DataLoader worker를 2개에서 4개로 늘렸다. 저장된 완성 응답은 재사용하고 진행 중인 미완성 샘플만 다시 준비한다.

## 변경 전 관측

| 항목 | Qwen2.5 | Qwen3 |
|---|---:|---:|
| 최근 측정 응답 수 | 119 | 119 |
| 벽시계 시간(초) | 594.5 | 773.0 |
| 모델 생성 구간 합계(초) | 171.3 | 198.2 |
| 생성 구간 비중 | 28.8% | 25.6% |
| 생성 시간 중앙값(초) | 1.24 | 1.34 |

벽시계 시간과 생성 시간의 차이는 입력 공급 대기, 전송, 파일 저장 등을 포함한다. 이것만으로 원인을 단정하지 않고 아래 관측·프로파일을 함께 확인했다.

- GPU 45회 관측: 평균 사용률 52.5%, 19회는 10% 이하. VRAM은 약 42GB로 용량 부족이 아니다.
- DataLoader worker 4개는 각각 평균 약 1.14~1.17 CPU 코어를 사용했다. 모델 프로세스는 45회 중 각각 20회·33회 futex 대기 상태였다. 대기 상태만으로 원인을 단정하지 않았다.
- cgroup CPU 할당은 12코어, 관측 평균 사용량은 약 6.74코어였다. 제한 발생은 453구간 중 5구간으로 CPU 할당 전체의 포화가 주원인은 아니었다.
- 별도 CPU 프로파일: Qwen2.5 두 샘플의 준비 시간 15.59초 중 영상 읽기 8.76초, resize 5.05초. Qwen3 두 샘플은 준비 46.28초 중 영상 읽기 28.47초. 프로파일 샘플은 제한적이며 전체 평균 처리 시간과 동일시하지 않는다.
- 프로파일한 네 샘플은 기존 저장 응답의 입력 token-ID SHA-256과 일치했다. Qwen3는 기존 전처리 코드의 metadata 검증도 유지했다.

## 적용 범위

전처리 worker 수만 변경했다. 영상 읽기 알고리즘, 프레임 추출, resize, prompt, full-head greedy 설정과 확률 기록 방식은 그대로다. CPU 여유를 쓰기 위해 동시 준비하는 샘플 수만 늘렸다. 같은 GPU에 모델을 더 올리거나 생성 batch를 바꾸지 않았다.

변경 후 각 모델에서 새 응답 60개 이상을 수집해 처리량을 별도로 기록한다. GPU 작업은 백그라운드로 계속 진행된다.

[45회 관측 원자료](../../../artifacts/runs/timelens_top30_full/performance/metadata/observation.json) · [입력 준비 검증](../../../artifacts/runs/timelens_top30_full/performance/metadata/preparation.json) · [Qwen2.5 CPU 프로파일](../../../artifacts/runs/timelens_top30_full/performance/metadata/qwen25_cpu_profile.txt) · [Qwen3 CPU 프로파일](../../../artifacts/runs/timelens_top30_full/performance/metadata/qwen3_cpu_profile.txt) · [변경 기록](../../../artifacts/runs/timelens_top30_full/performance/metadata/tuning.json)

## 변경 후 관측

전후 샘플이 달라 영상 길이와 디코딩 난이도 차이가 포함된다. 아래 비율은 동일 입력의 통제 실험이 아닌 처리량 관측치다.

| 모델 | 측정 응답 수 | 변경 전 초/응답 | 변경 후 초/응답 | 관측 속도 비율 |
|---|---:|---:|---:|---:|
| qwen25 | 67 | 5.00 | 3.58 | 1.40 |
| qwen3 | 59 | 6.50 | 3.72 | 1.75 |

변경 후 측정 구간 GPU 평균 사용률: 75.8%.
