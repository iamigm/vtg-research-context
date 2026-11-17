# 저장된 다른 decoding 경로의 entropy와 오류 · pilot · 900 queries

기존 deterministic rollout의 Qwen3-VL-8B-Instruct 분포를 다시 읽은 **사후 분석**이다. 모델 재추론·새 영상 개입 없이 저장된 top-30을 재정규화했다. Token uncertainty의 greedy full 결과와 별도 입력·후보 수집 조건이므로 두 표본을 독립 표본처럼 합산하지 않는다.

원래 실행은 2 FPS, spatial target 100,352 pixels, video_query input order, 공식 TimeLens prompt를 사용했다. 이번 집계는 저장된 입력 hash와 분포 검증에 연결하며 입력이나 generation 정책을 바꾸지 않는다.

고정 pilot 900 queries(각 dataset 300), 고유 영상 892개, 후보 응답 27,460개, 최초 greedy 900개와 분기 26,560개, 고유 prefix 199,189개를 확인했다. 같은 prefix의 분포는 index에서 한 번 읽고 전체 vocabulary 배열은 다시 읽지 않았다.

## 먼저 읽을 결과

대체 경로의 free suffix에서도 직전 token 대비 상승만으로 오류를 가려내기는 어렵다. 정수부의 민감도/오경보율은 97.9%/92.3%, 소수부는 100.0%/99.9%다. 정답 가능성을 유지한 숫자에서도 상승이 매우 흔하다.

같은 prefix에서 다른 token을 택하기 **전** entropy는 부모와 자식이 동일하다(26,560쌍 검증). 따라서 대체 token을 고르기 전의 entropy만으로 그 후보가 더 좋은지 알 수 없다. 분기 뒤 같은 endpoint의 다음 숫자 위치를 맞춘 비교는 11,810쌍, 891queries에서 가능했다.

그중 자식의 다음 숫자 entropy가 더 낮은 7,205쌍에서도 최종 IoU 개선 2,340쌍, 악화 3,915쌍, 동일 950쌍이 함께 있었다. query 안에서 평균한 IoU 변화는 -0.016 [-0.022, -0.010]다. entropy가 낮은 branch를 고르는 규칙만으로 정확도 개선이 보장되지 않는다.

## 비교 정의와 분모

Hₜ=−Σqᵢlnqᵢ, qᵢ=pᵢ/Σtop30 pᵢ인 raw top-30 재정규화 entropy를 사용한다. 단위는 nat, 범위는 0–ln30다. 기존 index의 entropy_nats는 full-vocabulary 값이므로 주 분석에 사용하지 않았다.

첫 후보는 greedy baseline이다. 대체 후보는 마지막 강제 prefix 길이 L 이후 j≥L인 **free suffix**만 model의 선택으로 센다(여기서 j는 0부터 시작). 강제 token과 그 앞의 조건화 prefix를 모델의 새로운 오류로 세지 않는다. 각 endpoint의 rounded 정답으로 완성할 가능성이 이전 prefix까지 남은 risk step만 포함하며 최초 오류 이후는 제외한다. 원래 자릿수별 정답과 같은 canonical 비음수 정수 또는 소수 한 자리 seconds 문법을 사용한다. clock·여러 소수 자리 등은 이 오류 분석에서 제외한다.

Δtokenₜ=Hₜ−Hₜ₋₁은 바로 앞 decoding step을 비교한다. Δnumericₜ=Hₜ−Hⱼ는 같은 endpoint의 직전 정답 가능 숫자 j를 비교하며 현재도 숫자여야 한다. 비교 상대는 이미 고정된 conditioning prefix 안에 있을 수 있다. Δ>0을 상승으로 정의한다.

step 비율은 query 안의 (endpoint, prefix hash, 실제 선택 token)을 중복 제거해 공통 prefix가 후보 수만큼 반복되지 않게 했다. greedy와 alternative-free는 별도 집계다. 민감도=오류 중 상승 비율, 오경보율=safe 중 상승 비율, precision=상승 중 오류 비율이다. 강제 선택을 제외한 탐색 후보의 사후 표본이므로 자연 sampling 확률이나 online detector 성능은 아니다.

Cpath=최초 free 오류 Δ−앞선 free safe Δ 평균이다. 둘 다 있는 candidate×endpoint만 포함하고, query 안에서 candidate×endpoint를 먼저 평균한 뒤 query를 같은 가중치로 평균한다. 공통 prefix 중복은 이 경로별 비교에는 남을 수 있지만 query를 복제 독립 표본으로 세지 않는다. CI는 query bootstrap 1,000회(seed42), video 의존성과 다중 비교는 보정하지 않았다.

## 저장 범위와 제외

| 구분 | greedy | alternative free |
|---|---|---|
| 후보 수 | 900 | 26560 |
| 지원 문법 후보 수 | 879 | 26034 |
| free risk step 발생 수 | 4615 | 37493 |
| 공통 risk step 중복 제거 수 | 0 | 0 |
| 오류 이후 free step 제외 수 | 3995 | 78419 |

**표 해설.** 후보 수는 query 수와 다르다. 동일 prefix의 여러 후보 참조를 step 비율의 독립 표본으로 세지 않는다. 강제·조건화 prefix의 risk step 발생 89737개와 그 안의 최초 오류 29933개도 model free 선택 통계에서 제외했다. 지원하지 않는 후보는 greedy 21개, alternative 526개다.

## 다른 decoding step에서도 상승이 오류를 구분하는가

| 경로 | 기준 | 현재 위치 | query | 오류 / safe | 민감도 | 오경보율 | precision |
|---|---|---|---|---|---|---|---|
| greedy | token | all | 879 | 1301 / 3314 | 97.3% | 63.2% | 37.7% |
| greedy | token | integer | 879 | 1100 / 1762 | 97.8% | 93.0% | 39.6% |
| greedy | token | fractional | 524 | 187 / 454 | 100.0% | 100.0% | 29.2% |
| greedy | token | format | 535 | 14 / 1098 | 21.4% | 0.0% | 100.0% |
| greedy | numeric | all | 753 | 714 / 1031 | 93.4% | 74.4% | 46.5% |
| greedy | numeric | integer | 619 | 527 / 577 | 95.4% | 78.7% | 52.6% |
| greedy | numeric | fractional | 524 | 187 / 454 | 87.7% | 68.9% | 34.4% |
| alternative_free | token | all | 854 | 11329 / 26164 | 97.6% | 55.8% | 43.1% |
| alternative_free | token | integer | 777 | 9106 / 12358 | 97.9% | 92.3% | 43.9% |
| alternative_free | token | fractional | 741 | 2119 / 3199 | 100.0% | 99.9% | 39.9% |
| alternative_free | token | format | 768 | 104 / 10607 | 22.1% | 0.1% | 71.9% |
| alternative_free | numeric | all | 800 | 7388 / 7942 | 96.1% | 85.2% | 51.2% |
| alternative_free | numeric | integer | 644 | 5269 / 4743 | 96.4% | 79.8% | 57.3% |
| alternative_free | numeric | fractional | 741 | 2119 / 3199 | 95.2% | 93.2% | 40.4% |

**표 해설.** 같은 숫자 위치에서 민감도와 오경보율이 비슷하면 상승 여부가 오류를 잘 구분하지 못한다. format을 함께 묶은 all은 숫자와 다른 전이 빈도에 영향을 받으므로 integer·fractional 행을 우선 확인한다. numeric 기준은 이전 safe 숫자가 존재하는 일부 step만 포함한다.

## 같은 경로의 평소 변화와 최초 free 오류 변화

| dataset | 경로 | 기준 | candidate×endpoint 수 | query 수 | Cpath nat [95% CI] |
|---|---|---|---|---|---|
| all | greedy | token | 728 | 557 | 0.846 [0.751, 0.934] |
| all | greedy | numeric | 202 | 180 | 0.413 [0.281, 0.539] |
| all | alternative_free | token | 7057 | 694 | 1.747 [1.659, 1.837] |
| all | alternative_free | numeric | 1786 | 291 | 0.167 [0.054, 0.285] |
| activitynet-timelens | greedy | token | 254 | 186 | 0.729 [0.575, 0.862] |
| activitynet-timelens | greedy | numeric | 64 | 53 | 0.545 [0.321, 0.769] |
| activitynet-timelens | alternative_free | token | 2665 | 235 | 1.563 [1.417, 1.709] |
| activitynet-timelens | alternative_free | numeric | 691 | 98 | 0.083 [-0.134, 0.315] |
| charades-timelens | greedy | token | 241 | 191 | 1.118 [0.969, 1.261] |
| charades-timelens | greedy | numeric | 42 | 40 | -0.055 [-0.342, 0.202] |
| charades-timelens | alternative_free | token | 2106 | 227 | 2.100 [1.944, 2.257] |
| charades-timelens | alternative_free | numeric | 272 | 60 | -0.077 [-0.315, 0.174] |
| qvhighlights-timelens | greedy | token | 233 | 180 | 0.680 [0.524, 0.822] |
| qvhighlights-timelens | greedy | numeric | 96 | 87 | 0.548 [0.385, 0.712] |
| qvhighlights-timelens | alternative_free | token | 2286 | 232 | 1.589 [1.450, 1.722] |
| qvhighlights-timelens | alternative_free | numeric | 823 | 133 | 0.339 [0.192, 0.490] |

**표 해설.** Cpath가 양수면 같은 경로의 앞선 free safe step보다 최초 free 오류 때 entropy 변화가 더 컸다. 첫 free step부터 오류이면 이 비교에 들어가지 않는다. 현재 role·숫자 순서를 맞춘 회귀가 아니므로 문법 전이가 남아 있으며, 기존 greedy full 분석의 통제 회귀와 동일한 추정량이 아니다.

## 같은 부모에서 분기한 다음 숫자 entropy와 최종 성능

분기 위치 직후 같은 endpoint에서 처음 나타나는 numeric step을 부모·자식에서 찾고 role과 numeric ordinal이 같은 경우만 비교한다. 서로 다른 token prefix에 조건화된 분포이며 영상은 동일하다. 마지막 강제 위치 이후 자식은 greedy로 완성되므로 추가 강제 선택이 없는 해당 child의 직접 continuation을 사용한다. 후손 후보를 parent의 직접 continuation으로 혼합하지 않는다. 정답 여부와 무관하게 모든 비교 가능한 branch를 포함한다.

ΔH=H(child 다음 숫자)−H(parent 다음 숫자), ΔIoU=IoU(child)−IoU(parent)다. IoU는 원래 deterministic 평가대로 연속 초·clipping/반올림 없이 계산하고 파싱 실패·역전·EOS 미도달은 0이다. 다음 숫자의 entropy는 실제 suffix를 관측한 값이므로 즉시 선택 가능한 비용 없는 점수가 아니다.

| dataset | 다음 숫자 H | 비교 쌍 / query | IoU 개선/악화/동일 | query 평균 ΔIoU [95% CI] |
|---|---|---|---|---|
| all | lower | 7205 / 863 | 2340/3915/950 | -0.016 [-0.022, -0.010] |
| all | higher | 4605 / 820 | 2060/1797/748 | -0.011 [-0.019, -0.003] |
| all | equal | 0 / 0 | 0/0/0 | — [—, —] |
| activitynet-timelens | lower | 2323 / 283 | 720/1221/382 | -0.012 [-0.022, -0.001] |
| activitynet-timelens | higher | 1686 / 279 | 708/653/325 | -0.022 [-0.034, -0.010] |
| activitynet-timelens | equal | 0 / 0 | 0/0/0 | — [—, —] |
| charades-timelens | lower | 2518 / 290 | 787/1454/277 | -0.024 [-0.037, -0.012] |
| charades-timelens | higher | 1329 / 264 | 720/416/193 | 0.007 [-0.009, 0.022] |
| charades-timelens | equal | 0 / 0 | 0/0/0 | — [—, —] |
| qvhighlights-timelens | lower | 2364 / 290 | 833/1240/291 | -0.012 [-0.020, -0.003] |
| qvhighlights-timelens | higher | 1590 / 277 | 632/728/230 | -0.016 [-0.029, -0.004] |
| qvhighlights-timelens | equal | 0 / 0 | 0/0/0 | — [—, —] |

**표 해설.** 낮은 entropy에서 개선·악화가 함께 나오면 entropy 최소화만으로 좋은 branch를 고를 수 없다. 쌍의 개수 비율과 query 평균은 가중치가 다르며 query별 branch 수가 결과를 지배하지 않게 평균 변화는 query 안에서 먼저 평균했다. 같은 query가 lower/higher 두 그룹에 모두 들어갈 수 있어 행을 독립 비교하지 않는다.

| 분기 token 위치 | 다음 숫자 H | 비교 쌍 / query | IoU 개선/악화/동일 | query 평균 ΔIoU [95% CI] |
|---|---|---|---|---|
| integer | lower | 7205 / 863 | 2340/3915/950 | -0.016 [-0.022, -0.010] |
| integer | higher | 4605 / 820 | 2060/1797/748 | -0.011 [-0.019, -0.003] |
| integer | equal | 0 / 0 | 0/0/0 | — [—, —] |

**표 해설.** 분기한 token의 정수부·소수부·서식을 분리했다. 같은 endpoint에 뒤따르는 숫자가 없으면 비교 자체가 불가능하므로 마지막 소수 자리 분기는 대체로 제외된다. 이 빈 표본을 효과 없음으로 해석하지 않는다.

| branch 비교 제외 사유 | 쌍 수 |
|---|---|
| branch_no_later_numeric_same_endpoint | 12276 |
| branch_later_role_mismatch | 1881 |
| branch_unsupported_endpoint | 593 |

**표 해설.** 26,560개의 전체 branch 중 위 제외 조건을 통과한 쌍만 다음 숫자 비교에 들어간다. 마지막 숫자 분기와 표기 변경을 제거한 선택된 부분집합이다.

![다른 경로의 entropy 진단](../../../../../artifacts/runs/rollout/deterministic/pilot/900/figures/entropy_saved_paths.svg)

**그래프 해설.** 왼쪽은 오류와 safe 숫자에서 직전 token 대비 entropy 상승이 얼마나 자주 나오는지 비교한다. 두 막대가 비슷하면 상승만으로 오류를 구분하기 어렵다. 오른쪽은 다음 숫자 entropy가 낮아진 branch의 query 평균 ΔIoU(pp)와 bootstrap 95% CI다. 0보다 낮으면 그 조건에서는 평균적으로 부모보다 나빠졌다는 뜻이다.

## 영상 증거에 대해 아직 말할 수 없는 것

여기서는 영상 입력을 바꾸지 않고 같은 영상에서 여러 prefix·선택 경로를 비교했다. 따라서 분기 후 entropy 감소를 영상 증거 활용의 인과 효과로 부를 수 없다. 이 저장 자료는 “어떤 후보를 선택하면 이후 분포가 더 집중되는가”를 검사하는 데 유용하며, 실제 영상 증거 효과는 같은 prefix에서 관련/대조 구간 제거를 별도로 비교해야 한다. 낮은 entropy가 정답을 강화했는지 최종 IoU·rounded endpoint와 함께 확인해야 한다.

## 재현과 원자료

[집계 JSON](../../../../../artifacts/runs/rollout/deterministic/pilot/900/metrics/entropy_summary.json) · [출처 SHA-256](../../../../../artifacts/runs/rollout/deterministic/pilot/900/metadata/entropy_sources.json) · [검증](../../../../../artifacts/runs/rollout/deterministic/pilot/900/validation/entropy_checks.json)

재생성: `.venv/bin/python _lib/rollout_entropy_appendix.py`. 입력은 canonical pilot900의 `metadata/source_records.json`, `rollouts/<query>.json`, `token_traces/<query>/index.json`이다. 원래 응답·분포·정답 판정을 덮어쓰지 않으며 새 추론은 실행하지 않는다.

