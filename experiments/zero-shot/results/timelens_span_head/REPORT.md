# Span 전용 LM head와 zero-shot 비교

**Qwen2.5는 QVHighlights에서 가장 크게 개선됐고, Qwen3는 Charades에서 뚜렷하게 악화됐다.** Qwen2.5의 ActivityNet도 평균 성능이 올랐지만, Charades에서는 지표별 방향이 달랐다. Qwen3는 세 벤치마크 모두 평균 성능이 낮아졌으며, ActivityNet·QVHighlights의 변화 구간은 0을 포함한다.

완료 상태: **600/600 모델·query 쌍**. 모델 2종 × 벤치마크 3종 × 100 query이며 고유 query는 300개다. 아래 정량 결과는 저장된 예측으로 계산했고, 공식 TimeLens evaluator의 12개 평가군과 일치한다.

## 비교 대상과 실험 조건

**표본 선정:** 별도의 기존 TimeLens pilot manifest를 찾지 못해, 통합된 9,404개 zero-shot 결과에서 벤치마크별 `random.Random(42).sample(..., 100)`으로 추출했다. 각 벤치마크에서 seed를 다시 42로 설정했다. 기존 일반 벤치마크 300개 목록과는 다르다. 이 새 고정 목록을 모든 비교군에 동일하게 적용했다. [샘플 목록](../../predictions/cohorts/pilot_300/samples.json)과 [선정 기록](../../predictions/cohorts/pilot_300/selection.json)에 정확한 ID·출처 해시를 남겼다.

| 항목 | 설정 |
|---|---|
| Qwen3 기준선 | 기존 `experiments/zero-shot/predictions/qwen3`의 `video_query` zero-shot 응답 |
| Qwen2.5 기준선 | 같은 300개에서 새로 실행한 native video 입력의 full-head zero-shot |
| 변경한 부분 | LM head에서 허용 토큰에 해당하는 기존 가중치 행만 projection |
| 허용 토큰 | 숫자, 온점, 쉼표, 하이픈, 대괄호, 콜론, 공백·줄바꿈 및 EOS: 총 526개 |
| 가중치·프롬프트 | 추가 학습 없음, 모델별 baseline과 동일한 프롬프트 |
| 시각 입력 | native video → query, 2 FPS, min_tokens 64, total_tokens 14336 |
| 추론 환경 | BF16, FlashAttention 2, Transformers 4.57.1, PyTorch 2.6.0+cu124 |
| 생성 | greedy, 최대 512토큰, seed 42 |
| Repetition penalty | Qwen2.5 1.05, Qwen3 1.0; 모델 내 두 비교군에 동일 적용 |
| 별도 출력 제약 | 문법 강제, 시간 범위 제한, 사후 clipping 없음 |

526개는 단일 문자 수가 아니라 허용 문자만으로 이루어진 tokenizer 토큰과 EOS의 수다. 전체 출력 차원은 Qwen2.5 152,064개, Qwen3 151,936개다. 허용 행만 계산한 점수를 원래 토큰 ID 위치로 돌려놓고 나머지 점수는 음의 무한대로 처리한다. 기존 full head는 기준선 실행을 위해 메모리에 보존하므로 모델 전체 메모리가 그 비율만큼 줄어든 실험은 아니다.

기존 지시문은 다음과 같다. 제한 head에서는 지시문의 자연어 답변 문구를 생성할 수 없지만, 출력 형식을 숫자 전용으로 바꾸는 추가 프롬프트는 사용하지 않았다.

```text
Please find the visual event described by the sentence '{query}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'.
```

**평가:** 공식 `extract_time`으로 파싱한 뒤 Python `round`로 정수 초로 반올림하고 첫 구간을 평가한다. 파싱 실패에는 기존 코드와 같은 영상 길이 밖의 fallback 구간을 부여한다. GT나 예측을 보정하지 않는다. R@1@t는 첫 예측 구간의 IoU가 t 이상인 query 비율이며, mIoU는 평균 IoU다.

## 정량 결과

모든 성능 값은 %. 각 행은 100 query다. 기준선과 제한 head를 같은 샘플에서 비교했다.

| 모델 | 벤치마크 | 설정 | R@1@0.3 | R@1@0.5 | R@1@0.7 | mIoU |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Charades-TimeLens | Zero-shot | 61.00 | 42.00 | 17.00 | 40.65 |
| Qwen2.5-VL-7B | Charades-TimeLens | 제한 head | 59.00 | 44.00 | 16.00 | 40.04 |
| Qwen2.5-VL-7B | ActivityNet-TimeLens | Zero-shot | 41.00 | 27.00 | 15.00 | 29.96 |
| Qwen2.5-VL-7B | ActivityNet-TimeLens | 제한 head | 48.00 | 35.00 | 21.00 | 35.17 |
| Qwen2.5-VL-7B | QVHighlights-TimeLens | Zero-shot | 50.00 | 24.00 | 8.00 | 30.83 |
| Qwen2.5-VL-7B | QVHighlights-TimeLens | 제한 head | 65.00 | 46.00 | 32.00 | 45.63 |
| Qwen3-VL-8B | Charades-TimeLens | Zero-shot | 65.00 | 49.00 | 31.00 | 47.53 |
| Qwen3-VL-8B | Charades-TimeLens | 제한 head | 55.00 | 39.00 | 26.00 | 37.59 |
| Qwen3-VL-8B | ActivityNet-TimeLens | Zero-shot | 55.00 | 43.00 | 28.00 | 40.77 |
| Qwen3-VL-8B | ActivityNet-TimeLens | 제한 head | 52.00 | 39.00 | 24.00 | 36.70 |
| Qwen3-VL-8B | QVHighlights-TimeLens | Zero-shot | 73.00 | 62.00 | 45.00 | 57.13 |
| Qwen3-VL-8B | QVHighlights-TimeLens | 제한 head | 71.00 | 60.00 | 41.00 | 55.29 |

![모델·벤치마크별 성능 비교](comparison.png)

## Paired 변화와 불확실성

변화량은 제한 head − zero-shot이며 단위는 %p다. 95% 구간은 영상 단위 paired bootstrap 2,000회, seed 42로 계산했다. 같은 영상의 여러 query를 함께 재표집한다. 아래 구간은 각 비교의 탐색적 추정이며 다중비교 보정은 하지 않았다. 실패↔성공은 IoU 0.5 기준이다.

| 모델 | 벤치마크 | ΔR@1@0.5 [95% 구간] | ΔmIoU [95% 구간] | 실패→성공 | 성공→실패 | 예측 구간 동일 |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Charades-TimeLens | +2.00 [-3.06, +7.85] | -0.60 [-3.42, +2.39] | 5 | 3 | 46/100 |
| Qwen2.5-VL-7B | ActivityNet-TimeLens | +8.00 [-0.99, +17.00] | +5.21 [+0.02, +10.97] | 14 | 6 | 26/100 |
| Qwen2.5-VL-7B | QVHighlights-TimeLens | +22.00 [+12.61, +32.32] | +14.80 [+9.65, +20.10] | 26 | 4 | 15/100 |
| Qwen3-VL-8B | Charades-TimeLens | -10.00 [-18.18, -2.02] | -9.94 [-14.75, -5.28] | 4 | 14 | 41/100 |
| Qwen3-VL-8B | ActivityNet-TimeLens | -4.00 [-11.11, +3.03] | -4.07 [-9.07, +1.19] | 5 | 9 | 38/100 |
| Qwen3-VL-8B | QVHighlights-TimeLens | -2.00 [-9.90, +5.05] | -1.84 [-7.46, +3.36] | 6 | 8 | 54/100 |

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

## 해석

**Qwen2.5의 QVHighlights 개선이 가장 뚜렷하다.** R@1@0.5는 +22.00%p, mIoU는 +14.80%p다. 실패→성공 26개가 성공→실패 4개보다 많고 두 지표의 bootstrap 구간도 0보다 높다. R@1@0.7도 8%에서 32%로 올라 단순히 낮은 IoU 기준만 넘긴 개선은 아니다. 다만 전수평가 결과로 일반화하려면 추가 확인이 필요하다.

**Qwen2.5의 나머지 결과는 구분해야 한다.** ActivityNet은 R@1@0.5 +8.00%p, mIoU +5.21%p지만 R@1@0.5 구간은 0을 포함하며 mIoU 하한도 약 +0.02%p로 작다. Charades는 R@1@0.5가 +2.00%p인 반면 mIoU는 −0.60%p이고, 두 구간 모두 0을 포함해 일관된 개선 근거가 약하다.

**Qwen3에는 동일한 제한을 일반적으로 유리하다고 볼 근거가 없다.** Charades는 R@1@0.5 −10.00%p, mIoU −9.94%p이며 두 구간이 모두 음수다. ActivityNet·QVHighlights도 평균 값은 낮아졌지만 두 지표의 구간이 0을 포함하므로 그 두 데이터셋의 하락을 확정적인 효과로 해석하지 않는다.

**어휘 제한은 출력 문법과 종료를 보장하지 않는다.** Qwen3는 56/300개가 512토큰에 도달했고 평균 출력 토큰도 크게 늘었다. EOS는 허용 목록에 포함되어 있으므로 종료 토큰을 제거한 결과는 아니다. Qwen2.5는 출력이 짧아졌지만 파싱 실패·역전 구간·다중 구간이 일부 생겼다. 자연어 답변 경로 차단과 기존 문장형 프롬프트의 상호작용은 원인 후보지만, 이번 기록만으로 반복 생성이 정확도 하락을 유발했다는 인과관계까지 확인한 것은 아니다.

**계산량 감소와 실제 속도 향상은 구분해야 한다.** head의 projection 대상은 약 15만 개에서 526개로 줄었지만 decoder와 영상 처리 계산은 남고, Qwen3의 생성 길이는 늘었다. 기존 Qwen3 기준선 시간에는 attention 계측이 포함되며 이번 실행은 GPU 동시 사용 조건도 달라 저장된 시간만으로 head 제한의 end-to-end 속도 향상률을 산출하지 않는다. H200에서 CUDA MPS와 최대 4개 복제본을 사용해 실행했다.

후속 실험에서는 숫자 전용 출력 프롬프트 변경과 span 문법·종료 제약을 각각 별도 비교군으로 분리하는 것이 적절하다. 이번 보고서는 그런 추가 변경을 적용한 결과가 아니다.

## 검증과 재현

- Qwen3 기준선 재현 기록 17건에서 저장된 생성 토큰과의 일치를 확인했다. 재시작 및 smoke 검사를 포함하므로 고유 샘플 수는 아니다.
- Qwen3의 저장된 입력 문자열·프레임 sampling metadata를 대조했고, Qwen2.5의 두 비교군은 같은 입력 텐서와 input-ID 해시를 사용했다.
- 모든 생성 토큰이 허용 목록에 속하는지, sample ID 중복이 없는지, 원래 예측의 해시가 유지되는지 확인했다.
- 공식 parser·반올림·첫 구간 IoU를 적용했고, 공식 evaluator CLI의 12개 평가군과 집계값이 일치했다.
- GT는 평가에만 사용했다. 이전 경로의 잘못된 cohort로 중단한 실행은 `provenance/aborted_old_cohort`에 보존했고 본 집계에서 제외했다.

[정량 JSON](summary.json) · [샘플별 CSV](paired.csv) · [샘플별 JSON](paired.json) · [고정 샘플](../../predictions/cohorts/pilot_300/samples.json) · [그림 PDF](comparison.pdf) · [완료 상태](status.json)

```bash
cd /home1/irteam/experiments/zero-shot
# 저장된 측정값으로 Markdown만 재생성; 추론·평가 재실행 없음
.venv/bin/python _lib/timelens_span_report.py --render-only
# 예측부터 집계·공식 평가 검증·그림·Markdown을 재생성
.venv/bin/python run.py span-head report
```

정량 결과와 해석은 Markdown으로 제공한다. HTML은 실제 영상 재생 등 상호작용이 필요한 분석에만 사용한다.
