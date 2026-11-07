# Temporal mode search: 설계 원리와 기대 효과

이 실험은 **모델이 이미 고려하고 있는 시간 구간의 대안들을 찾아, 영상을 다시 보고 그중 하나를 고르게 하면 zero-shot보다 정확해지는가**를 확인한다. 동시에 **그 개선에 드는 추가 연산이 감당할 만한가**를 평가한다.

예를 들어 “사람이 컵을 집어 드는 순간”을 찾는 질문에 모델이 `10–15초`를 답했다고 하자. 실제로는 처음 컵을 만지는 `9–12초`와 나중에 다시 집는 `40–44초` 사이에서 망설였을 수 있다. 이 실험은 정답 구간인 GT를 보지 않고, 모델의 token distribution과 이어지는 응답을 조사해 이런 대안을 찾는다. 예시의 숫자와 분포는 설명용이며 실제 측정값이 아니다.

설명 대상은 **capped_rollouts 조건**이다. 고정 pilot300·600·900, seed42·43, 세 가지 탐색 방법과 네 가지 rollout 예산을 사용한다. 아래의 기대 효과는 검증할 가설이며, 이미 입증된 성능으로 읽지 않는다.

## 1. 무엇을 바꾸고, 무엇을 고정하는가

Zero-shot 기준은 Qwen3-VL-8B-Instruct가 원래 영상과 질문을 보고 greedy로 생성한 단일 응답이다. 탐색 실험도 같은 모델과 영상 입력에서 출발한다. 모델을 추가로 학습하지 않는다. 현재 입력은2 FPS,목표 프레임 면적100,352 pixels,`video_query` 순서이며,기준 실행의 frame IDs·timestamp·H×W·tensor hash를 유지한다.

| 설계 원칙 | 쉽게 말하면 | 필요한 이유 |
|---|---|---|
| 모델의 확률 지지 안에서 탐색 | 모델이 어느 정도 가능하다고 본 token만 따라간다. | 임의의 숫자 나열과 모델의 불확실성 탐색을 구분한다. |
| GT 없이 후보 생성·선택 | 정답은 마지막 채점 때만 연다. | 실제 추론에서도 쓸 수 있는 방법을 평가한다. |
| 응답 표현과 시간 가설을 구분 | 다른 문장이 같은 구간을 가리키면 같은 후보로 관리한다. | 표현만 다양한 것을 사건 탐색 성과로 세지 않는다. |
| Tail은 조사만 수행 | Tail에서 발견한 구간은 최종 후보로 주지 않는다. | 주력 탐색 바깥에 무엇이 남았는지 별도로 진단한다. |
| 같은 입력과 예산으로 비교 | 프레임·해상도·질문을 유지하고 탐색 우선순위만 비교한다. | 입력 개선이나 연산 증가를 방법 자체의 효과와 혼동하지 않는다. |
| 정확도와 비용을 함께 평가 | 성능 향상 옆에 추가 연산을 적는다. | 오래 계산했다는 사실만으로 연구적 가치를 주장하지 않는다. |

첫 숫자를 0–9로 모두 바꾸는 forcing과, “다른 구간을 여러 개 제안하라”는 후보 생성 prompt는 사용하지 않는다. 다만 **발견한 후보를 판단하도록 질문하는 단계는 있다.** 후보 생성과 후보 판단은 역할이 다르다.

## 2. 전체 과정: 대안을 찾고, 묶고, 비교한다

한 query는 다음 과정을 거친다.

1. **기준 응답 확인:** 같은 입력의 zero-shot greedy 응답을 재현하고 원본과 token IDs가 일치하는지 확인한다.
2. **Main 탐색:** 확률적으로 응답을 생성하면서, 지나온 prefix에서 다른 유력 token을 선택하면 어디로 이어지는지 조사한다.
3. **Mode 구성:** 완성된 유효 interval을 비슷한 시간 구간끼리 묶는다.
4. **모델 판단:** main의 대표 interval들을 영상 근거로 비교하고 최종 interval 하나를 고른다.
5. **Tail 진단:** 그 checkpoint까지 main이 방문한 prefix에서 tail로 들어가는 경로를 조사한다. 결과는 후보 선택에 전달하지 않는다.
6. **사후 평가:** 선택을 모두 마친 뒤 GT로 정확도를 채점하고 탐색 비용을 집계한다.

작은 예산과 큰 예산은 별도 응답 묶음이 아니다. 같은 탐색을 이어 가며 중간 checkpoint를 저장한다. 각 checkpoint의 main 후보와 판단은 해당 시점의 결과로 고정한다.

## 3. 어디까지를 모델이 지지하는 분기로 보는가

### Token 하나를 허용하는 기준

각 prefix에서 모델의 전체 vocabulary softmax 확률을 먼저 계산한다. 그다음 다음 조건을 만족하는 token만 탐색에 허용한다.

`token 확률 ≥ max(0.009, 0.01 × 가장 높은 token 확률)`

즉, absolute floor는 **0.9%**, relative coefficient는 **0.01**이다. 가장 유력한 token의 확률이 60%라면 하한은 0.9%이고, 95%라면 하한은 0.95%다. 이 확률은 필터를 통과한 token끼리 다시 정규화하기 **전**의 값이다.

이 조건은 모든 위치에 적용한다. 첫 번째 token만의 제한이 아니며, 숫자뿐 아니라 구분자와 EOS도 같은 확률 기준을 따른다. Token은 tokenizer의 원래 단위로 취급한다. 숫자 한 자리와 token 하나가 항상 같다고 가정하지 않는다.

### Main과 tail은 완성 경로의 구분이다

먼저 각 prefix의 token을 확률순으로 정렬해 누적 확률이 90%에 도달하는 최소 집합을 **core**로 표시한다. 그중 위 하한을 통과한 token이 main 탐색에 쓰인다. 하한은 통과했지만 core 바깥에 있는 token은 tail 진입점이 된다.

다음은 한 prefix의 예시다.

| 다음 token | 원래 확률 | 분류 |
|---|---:|---|
| A | 86% | Core, 탐색 허용 |
| B | 6% | Core, 탐색 허용. 여기까지 누적92% |
| C | 4% | 탐색 허용, core 바깥 |
| D | 3% | 탐색 허용, core 바깥 |
| E | 0.8% | 하한 미달, 탐색 제외 |
| 나머지 합계 | 0.2% | 각 token이 하한 미달, 탐색 제외 |

**Main sequence**는 모든 위치에서 core token만 선택한 완성 경로다. **Tail sequence**는 허용된 non-core token을 한 번이라도 거친 경로다. Tail에 들어간 뒤의 token은 core일 수도 있고 non-core일 수도 있다. Tail에서 다시 main으로 분류되지는 않는다.

따라서 첫 token이 A여도 뒤에서 non-core token을 선택하면 tail이다. 반대로 긴 main sequence의 전체 path probability는 매우 작을 수 있다. **Main은 “완성 문장의 확률이 일정 값보다 크다”는 뜻이 아니다.**

90%는 각 위치에서의 구분 기준이다. 길이가 여러 token인 응답 전체에서 main이 모델 확률의 정확히90%를 차지한다는 뜻도 아니다. 하한 때문에 제외되는 질량도 별도로 기록한다.

## 4. 왜 서로 다른 응답을 temporal mode로 묶는가

다음 응답들은 표기는 달라도 같은 시간 가설에 가깝다.

- `10–15초`
- `10.0–15.0초`
- `10.2–15.1초`

반면 `40–44초`는 다른 발생 시점을 가리킬 가능성이 크다. 이 차이를 관리하는 단위가 **temporal mode**다.

여기서 mode는 시간 구간으로 정의한 가설이다. **다른 mode가 반드시 다른 의미의 사건임을 보장하지는 않는다.** 같은 사건의 경계를 크게 다르게 잡으면 mode가 나뉠 수 있고, 시간적으로 겹치는 다른 행동은 같은 mode로 묶일 수 있다. GT 없이 사용할 수 있는 temporal proxy라는 한계가 있다.

### 먼저 유효한 interval인지 확인한다

응답은 EOS로 끝나고, 해석 가능한 interval이 정확히 하나 있어야 한다. 시작과 끝의 부호를 보존해 parsing한 뒤, 각각을 실제 영상 metadata의 `[0, D]`로 clipping한다. Clipping 후 `start < end`여야 유효하다.

예를 들어 길이60초 영상에서 `−2–8초`는 `[0,8]`이 된다. `70–80초`는 `[60,60]`이 되어 무효다. `15–10초`의 순서를 바꿔 유효하게 만들지는 않는다. 원래 응답과 raw interval도 보존한다. 이 clipping은 예측에 적용하며 GT를 고치는 규칙이 아니다.

### 중복은 세 수준으로 관리한다

| 수준 | 처리 | 무엇을 구분하기 위한가 |
|---|---|---|
| 같은 token sequence | 저장된 동일 sequence를 재사용한다. 반복 draw는 유지한다. | 같은 응답을 여러 번 뽑은 사실과 고유 응답 수를 구분한다. |
| 같은 clipped interval | 하나의 interval로 통합한다. 서로 다른 sequence의 확률 질량은 합친다. | 숫자 표기나 표현 차이를 후보 증가로 세지 않는다. |
| 가까운 interval | 조건을 만족하면 같은 mode에 넣는다. | 작은 경계 차이와 다른 시간 가설을 구분한다. |

### 같은 mode로 묶는 정확한 기준

**Clipping 후의 두 interval이 충분히 겹치고, 시작과 끝도 모두 충분히 가까워야 한다. 새 interval은 mode 안의 모든 기존 interval과 이 조건을 만족해야 한다.** 영상의 의미를 모델에게 물어 같은 사건인지 판정하는 규칙은 아니다.

두 interval을 `A=[s_A,e_A]`, `B=[s_B,e_B]`라고 하자. 모든 시간의 단위는 **초**다.

| 기호 | 계산 | 의미 |
|---|---|---|
| `L_A`, `L_B` | `e_A−s_A`, `e_B−s_B` | 각 interval의 길이 |
| `I` | `max(0, min(e_A,e_B)−max(s_A,s_B))` | 겹치는 길이 |
| `U` | `L_A+L_B−I` | 합집합 길이 |
| `IoU(A,B)` | `I/U` | 두 interval의 겹침 비율 |
| `d(A,B)` | `max(abs(s_A−s_B), abs(e_A−e_B))` | 시작 차이와 끝 차이 중 큰 값 |
| `τ(A,B)` | `max(1초, 0.1×min(L_A,L_B))` | 이 두 interval에 허용하는 경계 차이 |

두 interval의 **pairwise compatibility**는 아래 두 조건을 **동시에** 만족하는 것이다.

```text
IoU(A,B) ≥ 0.5
그리고
d(A,B) ≤ τ(A,B)
```

`d≤τ`는 시작과 끝의 차이가 **각각** 허용 범위 안에 있어야 한다는 뜻이다. 둘의 평균 차이를 쓰지 않는다. `τ`는 영상 전체 길이가 아니라 **비교하는 두 interval 중 짧은 것의 길이**로 정한다. 짧은 interval 길이가10초 이하이면1초,10초를 넘으면 그 길이의10%다. 경계값0.5와τ는 포함한다.

비교에는 원래 timestamp나 화면에 반올림해 표시한 숫자가 아니라 **clipping 후 저장된 수치**를 사용한다. 별도의 반올림으로 경계값을 통과시키지 않는다. `[0,D]`로 clipping하기 때문에 raw interval이 달라도 clipped interval이 완전히 같아질 수 있다.

### 숫자로 보는 같은 mode / 다른 mode

아래 표의 “같음”은 **두 interval만 있을 때 같은 mode에 들어갈 수 있다**는 뜻이다. 이미 다른 interval이 들어 있는 mode에 합류하려면 그 interval들과도 모두 통과해야 한다.

| A | B | IoU | 경계 차이 d | 허용 차이 τ | Pairwise 판정 |
|---|---|---:|---:|---:|---|
| [10,20] | [10,20] | 1.0000 | 0초 | 1초 | 같음. Exact interval 중복으로 통합 |
| [10,20] | [11,21] | 0.8182 | 1초 | 1초 | 같음. 허용 경계값 포함 |
| [10,20] | [11.1,21.1] | 0.8018 | 1.1초 | 1초 | 다름. 많이 겹쳐도 경계 차이 초과 |
| [10,11] | [10.5,11.5] | 0.3333 | 0.5초 | 1초 | 다름. 경계는 가까워도 겹침 부족 |
| [10,13] | [11,14] | 0.5000 | 1초 | 1초 | 같음. 두 조건 모두 경계값 통과 |
| [10,50] | [13,53] | 0.8605 | 3초 | 4초 | 같음. 짧은 길이40초의10% 허용 |
| [10,50] | [14,49] | 0.8750 | 4초 | 3.5초 | 다름. 짧은 길이35초를 기준으로 제한 |

**IoU만 높다고 같은 mode가 되는 것도, 시작 시각만 가깝다고 같은 mode가 되는 것도 아니다.** 두 조건을 함께 써서, 긴 구간의 높은 겹침이 큰 경계 차이를 가리거나 짧은 구간끼리 거의 겹치지 않는 문제를 줄이려는 설계다. 이 threshold의 최적성을 입증했다는 뜻은 아니다.

### 여러 interval을 하나의 mode에 넣는 절차

실제 구현은 아래 순서로 **발견된 응답을 한 개씩** 처리한다.

1. 같은 token sequence를 이미 기록했으면 mode와 질량을 다시 추가하지 않는다. 반복 draw의 비용·횟수 기록은 유지한다.
2. 유효한 clipped interval이 없으면 어느 mode에도 넣지 않는다.
3. 새 interval과 **그룹 안의 모든 interval**이 pairwise compatibility를 만족하는 mode들을 찾는다. 대표 interval이나 medoid 하나와만 비교하지 않는다.
4. 가능한 mode가 없으면 새 mode를 만든다. 하나면 그 mode에 넣는다.
5. 가능한 mode가 여러 개면, 각 mode의 기존 interval 중 새 interval과 **가장 가까운 interval의 경계 차이 d**를 비교해 가장 작은 mode를 고른다. 이 값도 같으면 **먼저 만들어진 mode**,즉 더 작은 mode ID를 고른다.
6. 새 interval의 좌표가 이미 그 mode에 있으면 interval 목록을 늘리지 않는다. 새로운 token sequence라면 고유 경로와 그 모델 확률 질량은 해당 mode에 추가한다.

기존 mode들을 나중에 합치거나, 이미 배정한 interval을 다른 mode로 옮기거나, 전체 자료로 재clustering하지 않는다. 확률 질량·GT·query의 의미·모델의 judge 응답은 이 **그룹 배정 기준**에 사용하지 않는다. 확률 질량은 이후 대표 선택과 shortlist 정렬에 별도로 사용한다.

### “모두와 가까워야 한다”와 발견 순서의 영향

예를 들어 `A=[10,20]`, `B=[11,21]`, `C=[12,22]`를 생각하자.

- A와 B는 같은 기준을 통과한다. B와 C도 통과한다.
- 하지만 A와 C는 경계 차이2초가 허용1초를 넘어 통과하지 못한다.
- **A → B → C** 순서로 발견하면 `{A,B}`와 `{C}`가 된다. C는 B와 가까워도 A와 멀어서 첫 mode에 들어가지 못한다.
- **B → C → A** 순서로 발견하면 `{B,C}`와 `{A}`가 된다.
- **A → C → B** 순서에서는 먼저 `{A}`,`{C}`가 생긴다. B는 두 mode 모두에 들어갈 수 있고 거리도 같으므로 먼저 만들어진 A의 mode에 들어간다. 두 기존 mode를 합치지는 않는다.

따라서 pairwise compatibility는 수학적인 동치관계가 아니다. A≈B와 B≈C가 A≈C를 보장하지 않는다. 현재 mode는 **모든 쌍이 조건을 만족하도록 유지하는 online group**이며 발견 순서에 따라 최종 묶음이 달라질 수 있다. “의미적으로 같은 사건이면 언제나 같은 mode가 된다”는 보장은 없다.

### Main과 tail,mode ID의 범위

Main과 tail은 **서로 별도의 mode registry**에서 같은 규칙으로 묶는다. 같은 interval이 양쪽에서 발견되어도 후보 registry를 합치지 않으며,tail mode를 main 후보로 승격하지 않는다. Viewer의 `main mode 0`과 `tail mode 0`은 서로 다른 식별자다.

Mode ID는 **query·방법·seed·main/tail 안에서 생성 순서대로 붙이는 번호**다. 다른 query나 방법·seed의 같은 번호가 같은 사건을 뜻하지 않는다. 하나의 탐색 stream을 예산별로 이어 갈 때는 ID와 기존 배정을 유지하되 새 interval과 mode가 추가될 수 있다. Viewer에서 확률 질량순으로 카드가 나열돼도 ID 자체가 확률 순위인 것은 아니다.

### Mode coverage를 따로 관리한다는 뜻

100번 생성했더라도 모두 같은 mode라면 시간 가설은 하나만 확인한 셈이다. 반대로10번 생성해 세 mode를 찾았다면 서로 다른 시간 가설 세 개를 확인했다.

그래서 rollout 시도 수, unique sequence 수, unique interval 수, mode 수를 각각 기록한다. 이를 통해 “많이 생성했는가”와 “다른 시간 가설을 찾았는가”를 구분한다. 다만 GT 없는 **mode coverage는 관측한 mode의 다양성**이다. 가능한 모든 사건 중 몇%를 찾았는지를 안다는 뜻은 아니다.

현재 sampling은 **with replacement**다. 같은 sequence가 다시 나올 수 있고, 중복과 무효 응답도 rollout 예산을 소비한다. 중복 저장과 확률 질량의 이중 계산은 막지만, 반복 시도 자체를 없애지는 않는다. Sampling without replacement는 현재 실행에 적용한 방법이 아니다.

## 5. Token uncertainty와 사건에 관한 uncertainty는 어떻게 다른가

모델이 다음 token을 망설인다고 해서, 서로 다른 시간 가설을 고민하는 것은 아니다. `10`과 `10.0` 사이의 망설임은 최종 interval을 거의 바꾸지 않을 수 있다. 반면 어떤 token 선택은 `10초 부근`과 `40초 부근`으로 결과를 갈라놓을 수 있다.

이를 설명하는 진단값이 **mutual information, MI**다.

`B(r) = I(A; C | r, valid)`

- `r`: 지금까지 생성한 prefix.
- `A`: 그다음 선택한 token.
- `C`: 끝까지 생성했을 때 도착한 temporal mode.
- `valid`: 유효한 interval로 끝난 continuation 안에서 비교한다는 조건.

쉽게 말하면 **“다음 token을 알면, 최종 mode가 어디일지 얼마나 더 잘 알 수 있는가?”**다.

| 상황 | A를 고른 뒤 | B를 고른 뒤 | 해석 |
|---|---|---|---|
| 표현만 달라짐 | Mode1/Mode2가 각각50% | Mode1/Mode2가 각각50% | Token을 알아도 mode 예측이 나아지지 않는다. MI=0bit. |
| 시간 가설이 갈라짐 | 항상 Mode1 | 항상 Mode2 | Token이 최종 mode를 알려준다. A/B가 같은 확률이면 MI=1bit. |

반대로 두 token이 서로 다른 mode로 이어져도, 한 token의 확률이99.9%라면 MI는1bit보다 훨씬 작다. MI에는 정규화된 확률 비중이 반영된다. 다만 **MI 하나만으로 여러 prefix 사이의 예산을 정하기는 부족하다.** 아주 드물게 도착하는 prefix에서도, 그 안의 두 대안은50:50으로 갈려 MI가 높을 수 있기 때문이다. 유효 응답으로 조건화하면 무효 응답이 많은 분기의 손실도 값 하나에 드러나지 않는다.

현재 MI는 방문한 분기의 continuation으로 추정하는 **진단값**이다. 전체 트리를 정확히 알고 계산한 값이 아니며, 이것을 직접 최대화하는 것이 현재 scheduler의 구현은 아니다. 무효 응답과 유효 응답의 mode 분포를 함께 읽어야 한다.

## 6. 실제 예산은 어디에 더 쓰는가

현재 세 방법은 후보 생성·mode 규칙·최종 judge를 공유하고, **다음에 조사할 분기의 우선순위**를 달리한다.

| 방법 | 더 조사하려는 곳 | 확인하려는 가설 |
|---|---|---|
| `probability` | 모델의 경로 확률이 높고 조사 비용이 낮은 곳 | 확률 지지만 따라가도 충분한가? |
| `token_entropy` | 위 조건에 더해, 다음 token 분포가 불확실한 곳 | Token 수준의 망설임이 유용한 단서인가? |
| `temporal_mode` | 위 조건에 더해, 새 mode를 발견할 가능성이 있는 곳 | 표현 차이보다 시간 가설의 다양성을 따라가면 나아지는가? |

모든 방법에 이미 많이 조사한 분기의 우선순위를 낮추는 항이 있다. Tail 조사에는 세 방법 모두 novelty를 고려하는 scheduler를 사용하되, tail 진입점은 각 방법이 main에서 방문한 prefix에 따라 달라진다.

### Score S(r,a)를 말로 읽으면

`temporal_mode`의 우선순위는 다음 생각을 구현한다.

**“모델이 이 prefix와 token을 함께 선택할 가능성이 얼마나 되는가 × 유효한 새 mode를 찾을 전망 / 조사 비용.”** 여기에 반복 배정을 줄이는 항을 곱한다.

예를 들어 어떤 분기의 새 mode 발견 전망이 높아도, 그 prefix에 도착할 모델 확률이 극히 작거나 이어서 생성하는 비용이 크면 우선순위가 내려간다. 반대로 매우 유력한 분기라도 계속 같은 mode만 나오면 다른 곳을 조사할 여지가 생긴다.

현재 구현의 순위를 만드는 식은 아래처럼 쓸 수 있다. 코드에서는 수치 안정성을 위해 log로 계산한다.

`S_probability(r,a) = P_model(r,a) / [max(1, 평균 비용) × sqrt(1+n)]`

`S_temporal_mode(r,a) = S_probability(r,a) × U(r,a)`

여기서 `n`은 그 분기에 쌓인 continuation 관측 수다. 평균 비용은 현재 구현의 logical token 비용이며 GPU seconds 자체가 아니다. `U`는 가중한 새 mode 발견 실적에, 관측이 적은 분기를 탐색하기 위한 bonus를 더하고 최대1로 제한한 값이다. `token_entropy`는 기본 score에 해당 prefix의 전체 token entropy를 곱한다.

**Score는 최종 답의 정답 확률이 아니다.** 다음 연산을 어디에 쓸지 정하는 우선순위다. `U`도 정확히 보정된 확률이나 엄밀한 신뢰상한으로 주장하지 않는다. 새 mode인지 여부는 발견 시점의 registry를 기준으로 하므로 탐색 이력에 의존한다.

Sampling할 때는 허용된 token끼리 확률을 다시 정규화한다. 따라서 기록에는 원래 모델 확률 `p`와 실제 sampling 확률 `q`를 구분한다. MI와 novelty 추정에는 continuation의 `p/q` 가중치를 사용한다. 자주 뽑혔다는 이유만으로 모델이 그만큼 지지한다고 해석하지 않기 위해서다.

## 7. Stochastic sampling이 주요 분기를 놓치면 어떻게 하는가

무작정 완성 응답만 반복 생성하면, 드문 대안은 한 번도 뽑히지 않을 수 있다. 그래서 **분기를 어디서 조사할지는 scheduler가 정하고, 그 분기 이후의 continuation은 stochastic하게 생성**한다.

- Main 시작점에서4회 sampling해 초기 경로를 살핀다.
- 방문한 prefix에 main 선택지가 두 개 이상이면, 다른 허용 token도 조사할 task로 등록한다.
- 초기 main 조사 전체는 최대8회다. 새 prefix가 나올 때마다 초기 예산을 다시 크게 배정하지 않는다.
- 초기 구간 이후에는 최대8회당1회 수준으로 미조사 분기를 우선하는 coverage 배정을 넣는다. 이때도 모델의 경로 확률이 높은 쪽을 우선한다.
- 나머지 배정은 위 score로 결정한다. Novelty 실적이 아직 없는 분기에도 탐색 bonus를 준다.
- Seed42·43으로 sampling에 따른 변동을 확인한다.

이는 임의의 숫자를 강제하는 방법과 다르다. **이미 방문한 prefix에서 확률 하한과 main 규칙을 통과한 대안**을 조건으로 잡고, 모델이 그 뒤를 이어 생성하게 한다.

그래도 주요 분기를 모두 찾는다는 보장은 없다. 아직 방문하지 않은 prefix 안의 대안은 알 수 없고, 예산보다 미조사 분기가 많을 수 있다. 하한 미달 token은 이번 탐색의 범위 밖이다. 그래서 미조사 edge 수와 mode 수, 예산 증가에 따른 변화를 기록한다. Coverage 배정도 모든 edge를 빠짐없이 도는 완전탐색은 아니다.

## 8. 발견한 후보를 모델은 어떻게 판단하는가

Main mode를 **관측한 고유 경로의 원래 확률 질량 합**으로 정렬하고 상위6개까지 판단에 전달한다. 이 질량은 관측한 경로만 더한 값이므로 mode의 전체 확률 질량에 대한 하한이다. 같은 sequence를 반복 sampling했다고 질량을 여러 번 더하지 않는다.

각 mode에서는 실제로 생성된 interval 중 최대2개를 대표로 고른다. 하나는 다른 interval들과의 가중 IoU가 큰 medoid이고, 다른 하나는 그 medoid에서 경계 차이가 가장 큰 대안이다. 평균 timestamp로 새 interval을 만들지 않는다.

Frozen VLM은 같은 영상을 보며 다음을 확인한다.

1. 질문의 행동·인물·대상·발생 시점과 맞는가?
2. 시작 경계가 필요한 행동의 시작을 포함하는가?
3. 끝 경계가 필요한 행동의 끝을 포함하는가?

각 항목에 supported/contradicted/uncertain에 해당하는 선택 확률과 짧은 영상 근거를 기록한다. 이 항목들을 단순히 합산해 최종 답을 정하는 것은 아니다. **최종 선택은 interval 간 pairwise preference**로 한다.

먼저 같은 mode의 두 대표를 비교하고, 이어 mode별 승자들을 round robin으로 비교한다. A/B 순서를 바꿔 두 번 묻고 합쳐 위치 편향을 줄인다. C는 동등하거나 영상 근거가 부족하다는 응답이다. 허용 선택지 A/B/C의 원래 확률 합이1% 미만이면 해당 pairwise 응답을 무리하게 정규화하지 않고 동률로 처리한다. 동률 해소에는 관측 질량과 고정 규칙을 사용한다.

Judge에는 tail 후보나 token rank, greedy 출처를 알려주지 않는다. 최종 답은 전달받은 실제 interval 중 하나이며 새 timestamp를 제안하게 하지 않는다.

| 예외 상황 | 현재 처리 |
|---|---|
| 전부 중복되어 대표 interval이 하나뿐 | 그 interval을 바로 선택한다. 불필요한 비교를 하지 않는다. |
| 하나의 mode에 경계 대안은 둘 이상 | 같은 mode 안의 대표 interval을 비교한다. |
| 유효한 main 후보가 전혀 없음 | 실패로 기록한다. Tail 후보로 메우지 않으며 평가 점수는0이다. |
| Judge token 예산 소진 | 관측 main 질량이 가장 큰 mode의 첫 대표 interval로 fallback하고 상태를 남긴다. |
| 지원되는 main 분기가 없거나 token guard 도달 | 가능한 지점에서 탐색을 종료하고 실제 시도 수와 종료 사유를 남긴다. |

## 9. Tail을 조사하되 후보로 주지 않는 이유

Tail은 **main에서 놓친 시간 가설이 얼마나 남아 있는가**를 보기 위한 진단 장치다. 각 checkpoint의 main 후보와 판단을 먼저 고정한 뒤, 그 시점까지 main이 방문한 prefix의 non-core 진입점을 조사한다.

Tail에서 GT와 가까운 interval이 발견돼도 최종 답에는 쓰지 않는다. 이후 main의 RNG·탐색 통계·mode registry에도 tail 정보를 섞지 않는다. 다만 계산 cache는 같은 연산을 재사용할 수 있으므로 실제 시간 계측에서는 그 효과를 구분한다.

예를 들어 main oracle보다 tail까지 포함한 diagnostic oracle이 훨씬 높다면, 허용된 token 중에서도 core 바깥에 유용한 대안이 남아 있다는 뜻이다. 이 결과는 현재 main 정책의 한계를 설명하지만, 실제 최종 성능 개선으로 세지는 않는다. Tail도 확률 하한 안에서 조사하므로 전체 모델 응답 공간에 대한 완전한 진단은 아니다.

## 10. 예산과 비교 단위

| 총 rollout 상한 / method / seed | Main 상한 | Tail 상한 |
|---:|---:|---:|
| 32 | 16 | 16 |
| 64 | 32 | 32 |
| 128 | 64 | 64 |
| 256 | 128 | 128 |

Rollout1회는 응답 하나를 완성하거나 유효성 실패로 끝나는 생성 시도다. 중복도 포함한다. Judge 호출은 이 횟수에 포함하지 않고 비용으로 따로 기록한다. 최대 길이512 tokens와 main/tail/judge token guard도 유지하므로 실제 rollout 수가 상한보다 작을 수 있다.

한 query에서 세 방법과 두 seed를 모두 최대 예산까지 수행하면 rollout 시도는 최대1,536회다. 작은 예산은 같은 stream의 checkpoint이므로 네 예산의 상한을 독립 실행처럼 더하지 않는다. 다만 여러 checkpoint에서 실제로 수행한 judge 계산은 전체 실험 비용에 남는다.

Pilot300 ⊂ 600 ⊂ 900은 고정 cohort의 정확한 prefix다. Pilot900은 세 데이터셋별300 query이며, seed나 방법에 따른 반복 응답을 query 수에 더하지 않는다. 두 seed는 query 안에서 먼저 평균한다. 이 cohort는 층화 진단 표본이며 전체 benchmark의 무작위 표본 평균으로 일반화하지 않는다.

## 11. 기대 효과를 어떻게 확인할 것인가

### 정확도: Zero-shot보다 실제로 나아지는가

IoU는 예측과 GT interval의 교집합 길이를 합집합 길이로 나눈 값이다. mIoU는 query별 IoU의 평균이고, R@τ는 IoU가τ 이상인 query의 비율이다. τ는0.3·0.5·0.7을 사용한다. 모두 높을수록 좋고 실패는0점이다.

Zero-shot 대비 차이는 `Δpp = 100 × (탐색 성능 − 같은 query의 zero-shot 성능)`으로 적는다. 예를 들어 mIoU가0.50에서0.51로 오르면 +1pp다. 예산·방법 간에는 같은 완료 query 집합을 비교하고 데이터셋·영상 길이·GT/영상 길이 비율별 차이도 확인한다.

### 후보와 선택: 어느 단계에서 좋아지거나 막히는가

Oracle은 GT를 보고 사후에 가장 좋은 후보를 골랐을 때의 상한이다. 실제 모델의 선택 성능이 아니다.

| 평가 항목 | 무엇의 사후 최고 IoU인가? | 알아보려는 것 |
|---|---|---|
| Main oracle | 발견한 모든 main interval | 탐색이 유용한 대안을 찾았는가? |
| Pool oracle | Judge에 전달한 shortlist의 대표 interval | 상위6 mode·대표 선택에서 좋은 후보를 잃었는가? |
| 최종 성능 | 모델이 실제 선택한 interval | 후보 비교가 제대로 작동했는가? |
| Diagnostic oracle | Main과 tail의 모든 interval | 진단용 tail에 추가 기회가 남았는가? |

Main oracle은 오르는데 최종 성능이 그대로라면 selector가 병목일 수 있다. Main oracle과 pool oracle 사이가 크면 shortlist가 병목일 수 있다. 예산만 늘고 mode 수와 oracle이 늘지 않으면 같은 가설을 반복 조사하는 비중이 클 수 있다.

`temporal_mode`가 같은 비용에서 다른 두 방법보다 좋은 후보와 최종 답을 찾는다면, 시간 가설의 novelty를 활용한 배정에 근거가 생긴다. 반대로 차이가 없으면 이 설계의 이점을 입증하지 못한 것이다. 모든 방법이 judge를 공유하므로 zero-shot 대비 개선만으로 scheduler와 judge의 기여를 각각 분리해 주장할 수는 없다.

### 비용: 개선폭에 비해 연산이 너무 크지 않은가

비용은 두 관점으로 본다.

- **Independent trace work:** 각 query·method·seed·예산을 빈 cache에서 단독 실행한다고 할 때의 native decoder inputs, forward 호출 수, attention pairs를 저장된 trace로 재구성한다. Attention pairs는 context 길이를 반영하는 workload proxy이며 정확한 FLOPs가 아니다.
- **Observed execution:** 현재 worker와 cache 환경에서 실제 걸린 wall time, native forward CUDA stream time, cache 저장·복원 시간, memory를 기록한다. CUDA stream time을 GPU 전체 active time으로 부르지 않는다.

Operational 비용은 영상 prefill+greedy+main+judge이고, research total은 여기에 tail을 더한 값이다. **이번 실험에 실제로 쓴 연산을 설명할 때 tail 비용을 빼지 않는다.** 직전 절반 예산보다 얼마나 개선됐고 얼마나 비싸졌는지, 추가1,000 decoder inputs당 ΔmIoU가 얼마인지도 확인한다.

Exact prefix KV cache와 worker3개의 MPS 실행은 중복 계산과 실행 대기를 줄이기 위한 구현이다. 탐색 규칙과 입력을 유지하며, 기존 구현과의 분포·경로·judge 일치를 검증한다. 제한된 calibration이나 microbenchmark의 가속률을 전체 pilot의 가속률로 확대 해석하지 않는다. Cache 재사용이 있는 실제 시간과, 각 방법을 독립 실행하는 비용도 구분한다.

## 12. 이 실험이 답하려는 질문

좋은 결과는 mode 수가 많아지는 것만으로 정해지지 않는다. **GT 없이 찾은 대안이 더 좋은 답으로 이어지고, 그 추가 비용까지 설득력 있어야 한다.**

현재 설계가 확인하려는 것은 세 가지다. 모델의 token uncertainty 중 시간 가설을 바꾸는 부분을 탐색에 활용할 수 있는가, 그렇게 얻은 후보를 같은 모델이 영상 근거로 더 잘 선택할 수 있는가, 그리고 단순히 더 많이 sampling하는 것보다 효율적인가다.

Mode의 의미적 정확성, 제한된 탐색에서 놓친 분기, 같은 모델을 judge로 쓰는 한계, 두 seed와 층화 pilot의 일반화 범위는 결과를 해석할 때 함께 남겨 둔다. 이 설명 문서는 설계의 이유를 다루며, 최신 성능과 완료 여부는 아래 누적 보고서를 따른다.

## 실행 명세·결과·구현

- [Pilot300·600·900 누적 비교](../../pilot/overview/capped_rollouts/REPORT.md)
- [현재 실행 명세: pilot900](../../pilot/900/capped_rollouts/SPEC.md)
- [현재 결과 보고서: pilot900](../../pilot/900/capped_rollouts/REPORT.md)
- [계열 대표 viewer](../../viewer.html)

설명은 다음 동결 구현과 설정을 확인해 작성했다. 실제 식과 예외 처리는 구현을 기준으로 한다.

- [동결 설정](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/metadata/config.json)
- [탐색 우선순위와 checkpoint](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/900/capped_rollouts/code/temporal_mode_capped_core.py)
- [확률 필터·mode·MI·judge](../../../../../artifacts/runs/rollout/temporal_mode_search/pilot/600/code/temporal_mode_core.py)
