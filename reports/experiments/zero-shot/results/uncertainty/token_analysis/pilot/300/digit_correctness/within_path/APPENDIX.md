# Within-path 상세 결과 · pilot · 300 queries

[쉬운 해설과 핵심 결과로 돌아가기](REPORT.md)

## paired 비교 포함·제외 상세

이 보고서의 entropy와 후보 확률은 top-30 재정규화 기준이다. 각 step에서 M = Σ(top-30) pᵢ, qᵢ = pᵢ/M, H₃₀ = −Σ qᵢ ln qᵢ로 계산한다. 단위는 nat, 범위는 0–ln(30) ≈ 3.401197이며 높을수록 top-30 내부 분포가 고르다. raw와 processed는 각각 자신의 top-30을 사용한다. top-30 밖의 질량은 조건부 분포에서 제외한다. 정답 가능성은 이 분석의 허용 문법·반올림 endpoint 일치 판정을 뜻하며 모델의 최종 정답 확률과 다르다. 원본 응답·판정·전체 vocabulary 분포 계측값은 보존하고 보고서의 파생 지표만 재계산했다.

| 모델 | dataset | endpoint | 경로 수 | paired | 오류 없음 | 서식 최초 오류 | numeric 오류·이전 숫자 없음 |
|---|---|---|---:|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | start | 100 | 43 | 17 | 0 | 40 |
| qwen25 | activitynet-timelens | end | 100 | 46 | 7 | 0 | 47 |
| qwen25 | charades-timelens | start | 100 | 44 | 26 | 0 | 30 |
| qwen25 | charades-timelens | end | 100 | 31 | 11 | 0 | 58 |
| qwen25 | qvhighlights-timelens | start | 100 | 54 | 24 | 0 | 22 |
| qwen25 | qvhighlights-timelens | end | 100 | 54 | 7 | 0 | 39 |
| qwen3 | activitynet-timelens | start | 96 | 36 | 25 | 0 | 35 |
| qwen3 | activitynet-timelens | end | 96 | 49 | 9 | 0 | 38 |
| qwen3 | charades-timelens | start | 100 | 35 | 43 | 0 | 22 |
| qwen3 | charades-timelens | end | 100 | 36 | 11 | 2 | 51 |
| qwen3 | qvhighlights-timelens | start | 98 | 34 | 39 | 0 | 25 |
| qwen3 | qvhighlights-timelens | end | 98 | 52 | 23 | 1 | 22 |

**표 해설.** 위 표는 모델·dataset·start/end별 포함·제외 내역이다. 각 행에서 paired, 오류 없음, 서식 최초 오류, 이전 숫자 없음의 합이 경로 수다. paired만 앞의 오류 전후 평균에 들어간다. start와 end는 같은 query에서 나온 두 경로일 수 있으므로 행의 수를 독립 query 수처럼 더하지 않는다.


## 경로 간·경로 내부 분산

여기서 경로 간 분산의 묶음은 endpoint가 아니라 **model × query 응답**이다. 같은 응답의 start/end risk numeric step을 합쳐 평균과 분산을 구한다. between은 응답 평균들이 서로 다른 정도, within은 한 응답 안에서 step들이 다른 정도다. 각 응답에 같은 가중치를 주며, 이는 모든 step을 같은 가중치로 둔 분산과 다를 수 있다.

response별 risk step 평균 μᵢ와 분산 σ²ᵢ를 구한다. 경로에 같은 가중치를 부여하면 총 분산은 Varᵢ(μᵢ) + meanᵢ(σ²ᵢ)다. 아래 between 비중은 앞 항을 총 분산으로 나눈 값이다. 경로 간 항에는 sample뿐 아니라 숫자·길이·prefix 구성 차이가 섞여 있으므로 intrinsic uncertainty 비중이라고 부르지 않는다.

| 모델 | dataset | query 수 | between variance (nat²) | within variance (nat²) | between 비중 |
|---|---|---:|---:|---:|---:|
| qwen25 | activitynet-timelens | 100 | 0.1137 | 0.5166 | 0.1803 |
| qwen25 | charades-timelens | 100 | 0.1105 | 0.4121 | 0.2114 |
| qwen25 | qvhighlights-timelens | 100 | 0.1352 | 0.5259 | 0.2045 |
| qwen3 | activitynet-timelens | 96 | 0.1652 | 0.3523 | 0.3193 |
| qwen3 | charades-timelens | 100 | 0.1394 | 0.4884 | 0.2221 |
| qwen3 | qvhighlights-timelens | 98 | 0.0978 | 0.4163 | 0.1902 |

**표 해설.** 위 표의 between 비중은 “응답 평균이 서로 다른 정도”를 전체 분산으로 나눈 값이다. 현재 범위는 18.0–31.9%다. within은 같은 응답 안에서 step별 entropy가 달라지는 정도다. between에 출력 길이와 숫자 구성 차이도 섞이므로 sample 고유 불확실성의 비율로 읽지 않는다.


## Processed 및 경로 내부 순위

processed Δprevious는 저장된 processed 분포에서 별도로 top-30을 재정규화한 값이다. raw와 후보 집합이 같다고 가정하지 않는다. rank는 이전 숫자들과의 비교 비율이며 정답 확률이나 오류 탐지 정확도가 아니다. 예를 들어 이전 숫자 세 개보다 오류 entropy가 각각 높음·같음·낮음이면 rank = (1 + 0.5 + 0)/3 = 0.5다. `all` 행은 start/end에서 얻은 값을 먼저 query 안에서 평균한 뒤 query 사이에서 평균한다.

rank는 최초 오류 H가 이전 safe numeric H보다 큰 쌍의 비율이며 동률은 0.5로 센다. 0.5 초과면 오류가 이전 숫자보다 대체로 높다. 이 값도 query 내부 평균 후 query 간 평균한다.

| 모델 | dataset | endpoint | processed Δprevious [95% CI] | raw rank [95% CI] |
|---|---|---|---:|---:|
| qwen25 | activitynet-timelens | start | 0.5056 [0.3197, 0.6896] | 0.9302 [0.8605, 0.9884] |
| qwen25 | activitynet-timelens | end | 0.7807 [0.6196, 0.9402] | 0.8913 [0.8043, 0.9783] |
| qwen25 | activitynet-timelens | all | 0.6270 [0.5020, 0.7449] | 0.9216 [0.8582, 0.9701] |
| qwen25 | charades-timelens | start | 0.9888 [0.8345, 1.1436] | 0.9318 [0.8409, 1.0000] |
| qwen25 | charades-timelens | end | 0.8194 [0.6126, 1.0327] | 0.9355 [0.8387, 1.0000] |
| qwen25 | charades-timelens | all | 0.8826 [0.7308, 1.0282] | 0.9167 [0.8333, 0.9833] |
| qwen25 | qvhighlights-timelens | start | 0.8508 [0.7093, 0.9888] | 0.9815 [0.9444, 1.0000] |
| qwen25 | qvhighlights-timelens | end | 0.7796 [0.6461, 0.9148] | 0.9815 [0.9444, 1.0000] |
| qwen25 | qvhighlights-timelens | all | 0.8164 [0.7062, 0.9285] | 0.9795 [0.9452, 1.0000] |
| qwen3 | activitynet-timelens | start | 0.7237 [0.5400, 0.9112] | 0.8611 [0.7500, 0.9722] |
| qwen3 | activitynet-timelens | end | 1.0041 [0.8626, 1.1533] | 0.9592 [0.9082, 1.0000] |
| qwen3 | activitynet-timelens | all | 0.8862 [0.7534, 1.0091] | 0.9073 [0.8386, 0.9677] |
| qwen3 | charades-timelens | start | 0.9360 [0.7552, 1.1090] | 0.8857 [0.7714, 0.9714] |
| qwen3 | charades-timelens | end | 0.9876 [0.8256, 1.1356] | 0.9722 [0.9167, 1.0000] |
| qwen3 | charades-timelens | all | 0.9354 [0.7984, 1.0681] | 0.9167 [0.8426, 0.9815] |
| qwen3 | qvhighlights-timelens | start | 0.8146 [0.6131, 1.0066] | 0.8382 [0.7059, 0.9412] |
| qwen3 | qvhighlights-timelens | end | 0.8853 [0.7394, 1.0246] | 0.9615 [0.9038, 1.0000] |
| qwen3 | qvhighlights-timelens | all | 0.8214 [0.7087, 0.9311] | 0.9104 [0.8433, 0.9701] |

**표 해설.** 위 표의 왼쪽은 processed 분포에서도 오류 전후 차이가 나타나는지 확인하는 보조 결과다. 오른쪽 rank는 최초 오류 entropy가 이전 정답 가능 숫자보다 높은 비교의 비율이다(동률은 0.5). 예를 들어 0.8이면 비교의 80%에 해당하는 가중 비율이지, 정답률 80%나 오류 탐지 정확도 80%가 아니다. all은 같은 query의 start/end 값을 먼저 평균한 집계다.


## 경로별 기준 수준과 생성 위치를 통제한 비교

risk set은 numeric이며 선택 직전까지 정답 가능했던 step이다. 유지 step과 최초 numeric 오류 step만 포함한다. Hᵢₜ = αᵢ + β·first_errorᵢₜ + γ·roleᵢₜ + δ·digit_ordinalᵢₜ + εᵢₜ로 비교한다. response FE는 query·model별 평균을 빼고 start/end indicator를 통제한다. endpoint FE는 query·model·endpoint별 평균을 뺀다. 위치 통제 모형은 endpoint FE에 실제 출력 자릿수 role과 numeric 순서의 categorical 효과를 더한다. pooled는 response를 같은 가중치로 두고 endpoint indicator만 통제한다. 각 FE 모형은 해당 경로 안의 risk step 가중치를 1/nᵢ로 둔다. 경로 안에서 오류 여부가 변하지 않는 경우 단순 endpoint FE의 β에 직접 기여하지 않는다. role·순서 효과는 다른 경로와 함께 추정하므로 완전히 같은 자릿수의 matched comparison은 아니다.

β의 단위는 nat이며 양수면 최초 오류 step의 entropy가 더 높다. CI는 query-cluster sandwich covariance에 G/(G−1) 보정과 t(G−1) 임계값을 사용했다. β가 별도로 식별되지 않으면 NA다. 같은 관측 risk set을 사용하지만 모형별 경로 가중치와 통제 조건은 다르다.

| 모델 | dataset | 모형 | risk step 수 | query 수 | β (nat) [95% CI] |
|---|---|---|---:|---:|---:|
| qwen25 | activitynet-timelens | pooled | 361 | 100 | 0.3910 [0.2556, 0.5264] |
| qwen25 | activitynet-timelens | response | 361 | 100 | 0.5997 [0.4494, 0.7499] |
| qwen25 | activitynet-timelens | endpoint | 361 | 100 | 0.7667 [0.6260, 0.9073] |
| qwen25 | activitynet-timelens | endpoint_role_position | 361 | 100 | -0.1635 [-0.3770, 0.0499] |
| qwen25 | charades-timelens | pooled | 339 | 100 | 0.3974 [0.2336, 0.5613] |
| qwen25 | charades-timelens | response | 339 | 100 | 0.6146 [0.4340, 0.7953] |
| qwen25 | charades-timelens | endpoint | 339 | 100 | 0.9556 [0.8080, 1.1032] |
| qwen25 | charades-timelens | endpoint_role_position | 339 | 100 | 0.1114 [-0.0850, 0.3078] |
| qwen25 | qvhighlights-timelens | pooled | 408 | 100 | 0.4506 [0.2928, 0.6083] |
| qwen25 | qvhighlights-timelens | response | 408 | 100 | 0.7643 [0.6276, 0.9009] |
| qwen25 | qvhighlights-timelens | endpoint | 408 | 100 | 0.9668 [0.8555, 1.0781] |
| qwen25 | qvhighlights-timelens | endpoint_role_position | 408 | 100 | 0.1061 [-0.0377, 0.2499] |
| qwen3 | activitynet-timelens | pooled | 358 | 96 | 0.3725 [0.2131, 0.5318] |
| qwen3 | activitynet-timelens | response | 358 | 96 | 0.5985 [0.4506, 0.7465] |
| qwen3 | activitynet-timelens | endpoint | 358 | 96 | 0.9150 [0.7939, 1.0361] |
| qwen3 | activitynet-timelens | endpoint_role_position | 358 | 96 | 0.3257 [0.1432, 0.5081] |
| qwen3 | charades-timelens | pooled | 359 | 100 | 0.4121 [0.2167, 0.6075] |
| qwen3 | charades-timelens | response | 359 | 100 | 0.5870 [0.3419, 0.8322] |
| qwen3 | charades-timelens | endpoint | 359 | 100 | 1.0255 [0.8951, 1.1559] |
| qwen3 | charades-timelens | endpoint_role_position | 359 | 100 | 0.0140 [-0.1570, 0.1849] |
| qwen3 | qvhighlights-timelens | pooled | 471 | 98 | 0.3108 [0.1667, 0.4550] |
| qwen3 | qvhighlights-timelens | response | 471 | 98 | 0.5745 [0.4165, 0.7325] |
| qwen3 | qvhighlights-timelens | endpoint | 471 | 98 | 0.9512 [0.8259, 1.0766] |
| qwen3 | qvhighlights-timelens | endpoint_role_position | 471 | 98 | 0.2701 [0.1359, 0.4043] |

**표 해설.** 위 표는 모델·dataset마다 네 가지 비교 방법을 모두 나열한 상세표다. risk step 수는 오류 전까지의 숫자와 최초 numeric 오류를 합한 관측 수이고 query 수는 그 출처 질문 수다. β가 양수면 해당 조건에서 오류 step의 entropy가 높았다는 뜻이다. CI가 0을 가로지르면 방향이 뚜렷하지 않다. 단순 paired 평균과 가중치·통제 조건이 달라 β를 Δprevious와 같은 수치로 기대하면 안 된다.


