# Within-path 상세 결과 · full · 9,404 queries

[쉬운 해설과 핵심 결과로 돌아가기](REPORT.md)

## paired 비교 포함·제외 상세

이 보고서의 entropy와 후보 확률은 top-30 재정규화 기준이다. 각 step에서 M = Σ(top-30) pᵢ, qᵢ = pᵢ/M, H₃₀ = −Σ qᵢ ln qᵢ로 계산한다. 단위는 nat, 범위는 0–ln(30) ≈ 3.401197이며 높을수록 top-30 내부 분포가 고르다. raw와 processed는 각각 자신의 top-30을 사용한다. top-30 밖의 질량은 조건부 분포에서 제외한다. 정답 가능성은 이 분석의 허용 문법·반올림 endpoint 일치 판정을 뜻하며 모델의 최종 정답 확률과 다르다. 원본 응답·판정·전체 vocabulary 분포 계측값은 보존하고 보고서의 파생 지표만 재계산했다.

| 모델 | dataset | endpoint | 경로 수 | paired | 오류 없음 | 서식 최초 오류 | numeric 오류·이전 숫자 없음 |
|---|---|---|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | start | 4500 | 1824 | 1137 | 6 | 1533 |
| qwen25 | activitynet-timelens | end | 4500 | 2031 | 360 | 45 | 2064 |
| qwen25 | charades-timelens | start | 3363 | 1262 | 810 | 2 | 1289 |
| qwen25 | charades-timelens | end | 3363 | 1435 | 361 | 2 | 1565 |
| qwen25 | qvhighlights-timelens | start | 1541 | 692 | 463 | 2 | 384 |
| qwen25 | qvhighlights-timelens | end | 1541 | 720 | 105 | 20 | 696 |
| qwen3 | activitynet-timelens | start | 4334 | 1444 | 1520 | 18 | 1352 |
| qwen3 | activitynet-timelens | end | 4334 | 2052 | 648 | 29 | 1605 |
| qwen3 | charades-timelens | start | 3363 | 1250 | 1228 | 9 | 876 |
| qwen3 | charades-timelens | end | 3363 | 1540 | 429 | 28 | 1366 |
| qwen3 | qvhighlights-timelens | start | 1490 | 444 | 665 | 5 | 376 |
| qwen3 | qvhighlights-timelens | end | 1490 | 754 | 332 | 12 | 392 |

**표 해설.** 위 표는 모델·dataset·start/end별 포함·제외 내역이다. 각 행에서 paired, 오류 없음, 서식 최초 오류, 이전 숫자 없음의 합이 경로 수다. paired만 앞의 오류 전후 평균에 들어간다. start와 end는 같은 query에서 나온 두 경로일 수 있으므로 행의 수를 독립 query 수처럼 더하지 않는다.


## 경로 간·경로 내부 분산

여기서 경로 간 분산의 묶음은 endpoint가 아니라 **model × query 응답**이다. 같은 응답의 start/end risk numeric step을 합쳐 평균과 분산을 구한다. between은 응답 평균들이 서로 다른 정도, within은 한 응답 안에서 step들이 다른 정도다. 각 응답에 같은 가중치를 주며, 이는 모든 step을 같은 가중치로 둔 분산과 다를 수 있다.

response별 risk step 평균 μᵢ와 분산 σ²ᵢ를 구한다. 경로에 같은 가중치를 부여하면 총 분산은 Varᵢ(μᵢ) + meanᵢ(σ²ᵢ)다. 아래 between 비중은 앞 항을 총 분산으로 나눈 값이다. 경로 간 항에는 sample뿐 아니라 숫자·길이·prefix 구성 차이가 섞여 있으므로 intrinsic uncertainty 비중이라고 부르지 않는다.

| 모델 | dataset | query 수 | between variance (nat²) | within variance (nat²) | between 비중 |
|---|---|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | 4500 | 0.1286 | 0.4789 | 0.2116 |
| qwen25 | charades-timelens | 3363 | 0.1133 | 0.4052 | 0.2186 |
| qwen25 | qvhighlights-timelens | 1541 | 0.1233 | 0.4925 | 0.2002 |
| qwen3 | activitynet-timelens | 4334 | 0.1468 | 0.3948 | 0.2711 |
| qwen3 | charades-timelens | 3363 | 0.1278 | 0.4808 | 0.2100 |
| qwen3 | qvhighlights-timelens | 1490 | 0.1115 | 0.3858 | 0.2243 |

**표 해설.** 위 표의 between 비중은 “응답 평균이 서로 다른 정도”를 전체 분산으로 나눈 값이다. 현재 범위는 20.0–27.1%다. within은 같은 응답 안에서 step별 entropy가 달라지는 정도다. between에 출력 길이와 숫자 구성 차이도 섞이므로 sample 고유 불확실성의 비율로 읽지 않는다.


## Processed 및 경로 내부 순위

processed Δprevious는 저장된 processed 분포에서 별도로 top-30을 재정규화한 값이다. raw와 후보 집합이 같다고 가정하지 않는다. rank는 이전 숫자들과의 비교 비율이며 정답 확률이나 오류 탐지 정확도가 아니다. 예를 들어 이전 숫자 세 개보다 오류 entropy가 각각 높음·같음·낮음이면 rank = (1 + 0.5 + 0)/3 = 0.5다. `all` 행은 start/end에서 얻은 값을 먼저 query 안에서 평균한 뒤 query 사이에서 평균한다.

rank는 최초 오류 H가 이전 safe numeric H보다 큰 쌍의 비율이며 동률은 0.5로 센다. 0.5 초과면 오류가 이전 숫자보다 대체로 높다. 이 값도 query 내부 평균 후 query 간 평균한다.

| 모델 | dataset | endpoint | processed Δprevious [95% CI] | raw rank [95% CI] |
|---|---|---|---:|---:|
| qwen25 | activitynet-timelens | start | 0.6815 [0.6520, 0.7097] | 0.9263 [0.9147, 0.9381] |
| qwen25 | activitynet-timelens | end | 0.7763 [0.7515, 0.8016] | 0.9319 [0.9209, 0.9424] |
| qwen25 | activitynet-timelens | all | 0.7180 [0.6975, 0.7386] | 0.9249 [0.9157, 0.9339] |
| qwen25 | charades-timelens | start | 0.7541 [0.7202, 0.7867] | 0.8930 [0.8772, 0.9093] |
| qwen25 | charades-timelens | end | 0.9216 [0.8888, 0.9522] | 0.9178 [0.9031, 0.9317] |
| qwen25 | charades-timelens | all | 0.8375 [0.8129, 0.8608] | 0.9028 [0.8904, 0.9150] |
| qwen25 | qvhighlights-timelens | start | 0.8552 [0.8147, 0.8956] | 0.9713 [0.9586, 0.9834] |
| qwen25 | qvhighlights-timelens | end | 0.7735 [0.7351, 0.8123] | 0.9646 [0.9507, 0.9764] |
| qwen25 | qvhighlights-timelens | all | 0.7949 [0.7658, 0.8259] | 0.9599 [0.9486, 0.9710] |
| qwen3 | activitynet-timelens | start | 0.8183 [0.7874, 0.8478] | 0.9036 [0.8882, 0.9181] |
| qwen3 | activitynet-timelens | end | 0.8662 [0.8427, 0.8900] | 0.9448 [0.9349, 0.9535] |
| qwen3 | activitynet-timelens | all | 0.8382 [0.8175, 0.8593] | 0.9229 [0.9137, 0.9321] |
| qwen3 | charades-timelens | start | 0.8984 [0.8656, 0.9302] | 0.8940 [0.8772, 0.9100] |
| qwen3 | charades-timelens | end | 1.0181 [0.9962, 1.0395] | 0.9808 [0.9737, 0.9870] |
| qwen3 | charades-timelens | all | 0.9505 [0.9292, 0.9712] | 0.9331 [0.9227, 0.9426] |
| qwen3 | qvhighlights-timelens | start | 0.8108 [0.7608, 0.8608] | 0.9088 [0.8818, 0.9347] |
| qwen3 | qvhighlights-timelens | end | 0.8538 [0.8176, 0.8937] | 0.9509 [0.9368, 0.9649] |
| qwen3 | qvhighlights-timelens | all | 0.8258 [0.7953, 0.8579] | 0.9350 [0.9217, 0.9484] |

**표 해설.** 위 표의 왼쪽은 processed 분포에서도 오류 전후 차이가 나타나는지 확인하는 보조 결과다. 오른쪽 rank는 최초 오류 entropy가 이전 정답 가능 숫자보다 높은 비교의 비율이다(동률은 0.5). 예를 들어 0.8이면 비교의 80%에 해당하는 가중 비율이지, 정답률 80%나 오류 탐지 정확도 80%가 아니다. all은 같은 query의 start/end 값을 먼저 평균한 집계다.


## 경로별 기준 수준과 생성 위치를 통제한 비교

risk set은 numeric이며 선택 직전까지 정답 가능했던 step이다. 유지 step과 최초 numeric 오류 step만 포함한다. Hᵢₜ = αᵢ + β·first_errorᵢₜ + γ·roleᵢₜ + δ·digit_ordinalᵢₜ + εᵢₜ로 비교한다. response FE는 query·model별 평균을 빼고 start/end indicator를 통제한다. endpoint FE는 query·model·endpoint별 평균을 뺀다. 위치 통제 모형은 endpoint FE에 실제 출력 자릿수 role과 numeric 순서의 categorical 효과를 더한다. pooled는 response를 같은 가중치로 두고 endpoint indicator만 통제한다. 각 FE 모형은 해당 경로 안의 risk step 가중치를 1/nᵢ로 둔다. 경로 안에서 오류 여부가 변하지 않는 경우 단순 endpoint FE의 β에 직접 기여하지 않는다. role·순서 효과는 다른 경로와 함께 추정하므로 완전히 같은 자릿수의 matched comparison은 아니다.

β의 단위는 nat이며 양수면 최초 오류 step의 entropy가 더 높다. CI는 query-cluster sandwich covariance에 G/(G−1) 보정과 t(G−1) 임계값을 사용했다. β가 별도로 식별되지 않으면 NA다. 같은 관측 risk set을 사용하지만 모형별 경로 가중치와 통제 조건은 다르다.

| 모델 | dataset | 모형 | risk step 수 | query 수 | β (nat) [95% CI] |
|---|---|---|---:|---:|---:|
| qwen25 | activitynet-timelens | pooled | 16782 | 4500 | 0.4263 [0.4041, 0.4484] |
| qwen25 | activitynet-timelens | response | 16782 | 4500 | 0.7357 [0.7129, 0.7585] |
| qwen25 | activitynet-timelens | endpoint | 16782 | 4500 | 0.8608 [0.8396, 0.8819] |
| qwen25 | activitynet-timelens | endpoint_role_position | 16782 | 4500 | 0.1015 [0.0750, 0.1281] |
| qwen25 | charades-timelens | pooled | 11574 | 3363 | 0.4865 [0.4603, 0.5128] |
| qwen25 | charades-timelens | response | 11574 | 3363 | 0.6580 [0.6297, 0.6862] |
| qwen25 | charades-timelens | endpoint | 11574 | 3363 | 0.9180 [0.8942, 0.9419] |
| qwen25 | charades-timelens | endpoint_role_position | 11574 | 3363 | 0.1021 [0.0682, 0.1361] |
| qwen25 | qvhighlights-timelens | pooled | 6083 | 1541 | 0.4324 [0.3928, 0.4721] |
| qwen25 | qvhighlights-timelens | response | 6083 | 1541 | 0.7720 [0.7341, 0.8100] |
| qwen25 | qvhighlights-timelens | endpoint | 6083 | 1541 | 0.9622 [0.9321, 0.9923] |
| qwen25 | qvhighlights-timelens | endpoint_role_position | 6083 | 1541 | 0.1487 [0.1100, 0.1875] |
| qwen3 | activitynet-timelens | pooled | 17266 | 4334 | 0.3901 [0.3663, 0.4138] |
| qwen3 | activitynet-timelens | response | 17266 | 4334 | 0.6665 [0.6420, 0.6909] |
| qwen3 | activitynet-timelens | endpoint | 17266 | 4334 | 0.9099 [0.8893, 0.9305] |
| qwen3 | activitynet-timelens | endpoint_role_position | 17266 | 4334 | 0.2583 [0.2324, 0.2841] |
| qwen3 | charades-timelens | pooled | 12487 | 3363 | 0.4614 [0.4332, 0.4897] |
| qwen3 | charades-timelens | response | 12487 | 3363 | 0.6612 [0.6301, 0.6922] |
| qwen3 | charades-timelens | endpoint | 12487 | 3363 | 1.0436 [1.0235, 1.0637] |
| qwen3 | charades-timelens | endpoint_role_position | 12487 | 3363 | 0.1318 [0.1028, 0.1608] |
| qwen3 | qvhighlights-timelens | pooled | 6787 | 1490 | 0.3822 [0.3456, 0.4188] |
| qwen3 | qvhighlights-timelens | response | 6787 | 1490 | 0.6275 [0.5890, 0.6661] |
| qwen3 | qvhighlights-timelens | endpoint | 6787 | 1490 | 0.9168 [0.8838, 0.9498] |
| qwen3 | qvhighlights-timelens | endpoint_role_position | 6787 | 1490 | 0.3335 [0.2939, 0.3732] |

**표 해설.** 위 표는 모델·dataset마다 네 가지 비교 방법을 모두 나열한 상세표다. risk step 수는 오류 전까지의 숫자와 최초 numeric 오류를 합한 관측 수이고 query 수는 그 출처 질문 수다. β가 양수면 해당 조건에서 오류 step의 entropy가 높았다는 뜻이다. CI가 0을 가로지르면 방향이 뚜렷하지 않다. 단순 paired 평균과 가중치·통제 조건이 달라 β를 Δprevious와 같은 수치로 기대하면 안 된다.


