# Appendix · Event proposal and tree search · pilot900

[핵심 보고서](REPORT.md)

상세 설계·조건별 결과·검증 내용이다. 실행 범위와 지표 정의는 핵심 보고서와 함께 확인한다.

<!-- report-part:1:begin -->
<!-- rollout-methods:begin -->

## Shared pipeline — 한 query가 최종 답이 되기까지

이 절은 **Event proposal and tree search에서 비교한 15개 방법**의 기본 흐름을 설명한다. 앞선 stochastic rollout·deterministic rollout·reconsideration 실험은 각각 해당 보고서에 설명되어 있다.

한 영상과 query를 주면, 먼저 답이 될 만한 interval을 여러 개 모은다. 그다음 모델에게 각 interval이 요청한 사건을 보여주는지 묻는다. 남은 후보 중 하나를 골라 start와 end를 다시 예측하고, 새 예측을 평가한다. 이 과정을 예산 안에서 반복한 뒤 모델 비교로 최종 interval 하나를 고른다.

**후보 수집 → 사건 적합성 확인 → 재검토할 후보 선택 → start/end 재예측 → 다시 평가·탐색 → 최종 후보 비교** 순서다. GT는 이 과정이 끝난 다음 성능을 계산할 때만 쓴다.

### 1. Initial proposals — 어디를 답 후보로 볼지 넓게 모은다

처음부터 한 interval이 맞다고 정하지 않는다. 다음 세 곳에서 후보를 얻어 합친다. 여기서 **proposal**은 “이 interval이 답일 수도 있다”는 제안이며, 서로 다른 semantic event임이 검증됐다는 뜻은 아니다.

| 후보를 얻는 곳 | 모델에 실제로 시킨 일 | 확보하려는 가능성 |
|---|---|---|
| 기존 deterministic rollout 첫 8개 | 앞선 실험에서 token 분기를 따라 완성한 답을 재사용한다. | 이미 관측한 다른 시간 예측을 출발점으로 삼는다. |
| 첫 숫자의 10개 분기 | greedy 답의 첫 숫자 위치에 `0`부터 `9`까지 하나씩 강제하고, 그 뒤를 각각 greedy decoding한다. | 첫 숫자의 낮은 확률 선택도 한 번은 따라가 본다. |
| 새 model prompt | “영상 전체에서 가능한 occurrence를 최대 6개 찾고 interval과 짧은 시각 증거를 써라”라고 요청한다. | 기존 숫자 분기에서 나오지 않은 interval도 모델이 직접 제안하게 한다. |

이렇게 얻은 답을 start/end로 parsing한다. 처음에는 정확히 같은 interval을 하나로 합치고, 어느 경로에서 나왔는지는 모두 기록한다. 8+10+6개의 **서로 다른 사건**이 생기는 것은 아니다. 중복·생성 실패가 있을 수 있고, 여러 interval이 같은 사건을 가리킬 수도 있다.

첫 숫자 10개를 따라간 뒤에는 suffix를 greedy로 완성했다. 그 뒤 정수의 두 번째·세 번째 자리까지 모두 분기한 실행은 아니다. 새 model prompt에는 JSON array를 요청했고, 그 형식으로 parsing할 수 없는 일반 timestamp 답은 이 proposal 단계에서 채택하지 않았다.

### 2. Critic — 각 후보가 요청한 사건을 담고 있는지 확인한다

각 후보마다 같은 영상·query·interval을 모델에 보여주고 다음 질문을 따로 한다. 이렇게 후보를 평가하는 역할을 **critic**이라고 한다. 별도의 학습된 모델을 쓴 것이 아니라 같은 frozen VLM을 다른 prompt로 다시 호출했다.

| 질문 | 확인하는 내용 | 그 답을 쓰는 곳 |
|---|---|---|
| Event presence | “이 구간에 요청한 action이 맞는 actor·object와 함께 있는가?” | 계속 검토할 후보인지 정한다. |
| Contradiction | “시간 경계를 조금 바꿔도 해결되지 않는 action·actor·object·occurrence 불일치가 있는가?” | 그 후보 자체를 포기해야 할 가능성을 확인한다. |
| Boundary fit | “시작과 끝이 action을 정확히 포함하며 불필요한 부분을 덜어냈는가?” | boundary를 재예측한 뒤 그 결과를 평가한다. |

처음 모은 후보에는 앞의 두 질문만 한다. boundary 질문은 refinement 후에 한다. 각 질문에는 Yes/No를 요청하고, 두 token의 확률을 이용한다. Event presence의 Yes 확률이 0.5 이상이고 contradiction이 확인되지 않은 후보를 탐색 대상으로 남긴다. contradiction을 강하게 지지하면 다른 prompt로 한 번 더 확인한다. 정확한 기준은 아래 수식에 적었다.

이 값들로 후보마다 **value function**, 즉 다음에 더 살펴볼 대상을 고르는 데 쓰는 평가값을 계산한다. 요청한 사건을 담고 있다고 보고, 근본적인 불일치가 적다고 보고, 재검토 후 boundary도 적합하다고 볼수록 value가 높다. 이는 모델 판단을 조합한 값이며 실제 정답 확률로 보정한 값은 아니다.

### 3. Search policy — 다음 모델 연산을 어느 후보에 쓸지 고른다

모든 후보를 끝없이 재검토할 수 없으므로 하나를 선택한다. **여기서 방법별 차이가 생긴다.** value가 높은 후보를 먼저 볼 수도 있고, 아직 적게 검토한 후보에 기회를 줄 수도 있으며, 여러 시간대의 후보를 함께 유지할 수도 있다. 모델이 다음 행동을 직접 고르는 조건도 있다.

아래 방법별 토글의 공식은 주로 이 결정을 설명한다. 새 proposal을 허용하는지, 처음에 어느 후보를 주는지, 최종 선택을 한 occurrence로 제한하는지 같은 차이도 해당 토글에 명시했다.

### 4. Boundary refinement — 선택한 후보의 start와 end를 다시 예측한다

선택한 interval과 앞서 얻은 관찰을 보여주고 “이 occurrence를 다시 보고 시작·종료를 정교하게 해라”라고 요청한다. focus는 `onset`(시작), `offset`(종료), `joint`(둘 모두) 중 하나다. **어느 focus든 모델은 start와 end 전체를 새로 생성한다.** start를 바꾸면 end도 그 새로운 start token 뒤에서 다시 계산된다.

그 뒤 새 interval에서 어떤 시작·종료 증거 또는 근본적 불일치가 보이는지 짧게 설명하게 하고, 2단계의 critic 질문들로 새 답을 다시 평가한다. 이때 Boundary fit도 측정한다. 모델 weight를 학습해서 바꾸는 과정은 아니다. 같은 모델에 새 prompt·interval·history를 주어 새 답과 확률을 계산한다.

**새 답이 나왔다고 이전 interval을 덮어쓰지는 않는다.** 이전 후보와 새 후보를 부모·자식으로 함께 기록한다. 다음은 관계를 설명하기 위한 가상 예시이며 실제 관측 결과가 아니다.

| 상태 | interval | 어디에서 생겼는가 |
|---|---|---|
| A | [18, 24] | 처음 모은 후보 |
| A1 | [17.5, 24] | A의 시작을 재검토한 새 예측 |
| A2 | [18, 23.6] | A의 종료를 재검토한 다른 새 예측 |
| B | [104, 112] | 처음 모은 다른 후보 |
| B1 | [104.2, 111.8] | B를 재검토한 새 예측 |

이 tree에서 node 하나는 interval·관찰·평가값을 담고, edge 하나는 재검토 한 번을 뜻한다. 한 token을 고를 때마다 edge가 생기는 **원래 응답의 token tree와는 단위가 다르다.** 첫 숫자 분기는 1단계의 후보 수집에 사용했고, 그 뒤의 tree search는 새 prompt로 interval을 재예측한 상태들을 탐색한다.

### 5. Backtracking and new proposals — 계속 수정할지, 다른 후보로 갈지 결정한다

새 답과 평가값을 기록한 뒤 3단계로 돌아가 다음 대상을 고른다. 같은 후보를 이어서 수정할 수도 있고 다른 후보로 옮길 수도 있다. 새 child의 value와 해당 경로에 쓴 방문 수를 조상 node에도 누적한다. UCT 같은 방법은 이 기록을 다음 선택에 사용한다.

refinement 중 “이 occurrence는 요청한 사건과 근본적으로 다르다”는 판단이 두 질문에서 기준을 충족하면, 같은 occurrence로 묶인 경로 전체를 후속 탐색과 최종 선택에서 제외한다. 위 가상 예시에서 A1을 보고 이런 판단이 내려지면 A·A1·A2도 제외될 수 있다. 그 뒤에는 B 같은 다른 후보를 검토할 수 있다. **모델의 잘못된 판단이면 좋은 parent까지 버릴 수 있는 구현**이다.

대부분의 조건은 refinement가 4회 누적될 때마다 기존 검토 결과를 보여주고 새로운 occurrence 하나를 추가로 제안하게 한다. 이렇게 새 후보를 얻으면 다시 2단계부터 평가한다. 새 proposal을 하지 않는 ablation 등은 아래 토글에 적었다.

### 6. Final selection — 예산이 끝나면 남은 후보들을 모델로 비교한다

search를 멈춘 뒤에는 마지막으로 생성한 답을 자동 채택하지 않는다. **지금까지 저장한 유효 후보들**에서 최종 비교 대상을 최대 4개 추린다. 초기 후보나 중간 parent도 남아 있다면 여기에 들어갈 수 있다.

이 4개는 critic value가 높으면서 이미 고른 interval과 지나치게 겹치지 않도록 MMR로 순차 선택한다. MMR의 계산식은 아래에 있다. 현재 구현은 이 단계에서 같은 interval의 중복 node를 완전히 제거하지 않으므로, 4개가 모두 다른 사건이라는 보장은 없다.

그다음 두 후보씩 모델에게 “어느 interval이 요청한 사건과 boundary를 더 정확히 찾았는가?”라고 묻는다. A/B 순서를 바꿔 한 번 더 묻고 두 판단을 합친다. 이긴 후보를 다음 round로 올려 하나가 남을 때까지 비교한다. **최종 답은 winner에 저장된 기존 interval**이다. 마지막에 timestamp를 한 번 더 생성하지 않는다.

### 7. Evaluation — 답을 확정한 뒤 GT로 성능을 계산한다

앞의 모든 선택은 GT 없이 수행한다. 최종 답이 정해진 뒤 GT와 비교해 mIoU·R@0.3/0.5/0.7 등을 계산한다. 생성한 후보들 중 GT에 가장 가까운 답을 사후에 고른 값이 oracle이다. oracle은 후보 안에 좋은 답이 있었는지를 보여주고, 실제 선택 성능은 모델이 그 답을 골랐는지를 보여준다.

### Budget — 어디까지 반복하는가

search에는 4,096 또는 8,192 decoder input tokens를 배정하고, 마지막 후보 비교에는 별도 2,048 token cap을 둔다. search cap은 완성 후보 개수나 모델 호출 횟수가 아니다. 새 prompt 처리와 응답 생성에 필요한 text 연산을 세는 예산이다. 같은 단계라도 prompt가 길거나 호출이 많으면 더 빨리 예산을 쓴다.

처음 후보를 모으고 평가한 비용은 search cap 밖에서 공통 비용으로 더한다. 따라서 표의 전체 비용은 “4,096 또는 8,192”보다 클 수 있다. 아래부터는 위 절차의 기호·수식·고정 설정이며, 방법별 선택 공식은 그 뒤 토글에서 확인할 수 있다.

## Autoregressive decoding — 시간 구간 답변은 어떻게 만들어지는가

모델은 start와 end를 한 번에 독립적으로 고르지 않는다. 응답을 앞에서부터 token 하나씩 생성한다. 앞에서 선택한 token들이 다음 token의 확률을 바꾼다. 이 생성 방식을 **autoregressive decoding**이라고 한다.

예를 들어 start의 첫 token이 `1`이어도, 뒤에 `.`이 오면 1초대가 되고 `4`가 오면 14초 또는 그 뒤로 더 이어지는 숫자가 될 수 있다. 그 start를 생성한 뒤에 end를 쓰므로, start를 바꿔 생성한 답의 end도 새 조건에서 계산해야 한다.

각 step에서 확률이 가장 높은 token을 고르면 **greedy decoding**이다. 어떤 위치에서 다른 token을 강제로 선택하고 그 뒤를 greedy로 완성하면 그 선택이 어떤 다른 interval로 이어지는지 볼 수 있다. 이때 같은 interval을 얻어도 실제 token 순서가 다르면 모델이 따라간 경로는 다르다.

아래는 이 설명의 정확한 기호와 계산식이다. 뒤의 방법별 토글도 실제 실행 코드에 대응한다. 논문 알고리즘을 변형한 부분은 해당 토글에 적었으며, snake_case 문자열은 저장된 실험 식별자다.

### Notation — 기호가 가리키는 값

| 기호·용어 | 이 문서에서의 의미 |
|---|---|
| $x=(V,q)$ | 모델에 실제 입력한 영상 frame들과 query. 원본 영상 전체를 모두 보았다는 뜻은 아니다. |
| $D$ | annotation의 영상 길이, 단위 seconds. |
| $c$ | 영상·query·instruction·해당 방법의 review prompt와 history를 포함한 model context. |
| $y_{<t}$ | 현재 응답에서 이미 선택한 정확한 token ID의 순서. 숫자로 변환한 시각만으로 대체하지 않는다. |
| $z_t(a),\ p_\theta(a\mid c,y_{<t})$ | 다음 token $a$의 logit과 전체 vocabulary softmax 확률. 모델 parameter $\theta$는 고정이다. |
| $y=(y_1,\ldots,y_T)$ | 한 token 경로. 완성 응답의 sequence log-likelihood를 비교할 때는 EOS를 포함하며, grammar 종료를 사용하는 conditional budget 실험은 별도로 구분한다. |
| $I_i=[s_i,e_i]$ | 응답에서 parsing한 시작·종료 시각. 동일한 $I_i$라도 서로 다른 token 경로일 수 있다. |
| $G$ | GT interval. 생성·분기·선택 prompt에 넣지 않고 평가와 oracle 계산에만 사용한다. |
| greedy decoding | 매 generation step에서 가장 확률이 높은 token을 선택하는 생성 방법. 완성 응답의 확률을 전역 최대화한다는 보장은 없다. |
| teacher forcing | 계산하려는 token 경로를 입력으로 고정하고 각 위치의 확률을 평가하는 방법. 새 응답을 자유롭게 생성하는 과정과 구분한다. |

확률은 다음과 같이 계산한다. $\mathcal V$는 전체 vocabulary이고, $\ell$은 자연로그를 사용한다.

$$
p_\theta(a\mid c,y_{<t})=\frac{\exp z_t(a)}{\sum_{b\in\mathcal V}\exp z_t(b)},\qquad
P_\theta(y_{1:j}\mid c)=\prod_{t=1}^{j}p_\theta(y_t\mid c,y_{<t}).
$$

$$
\ell_\theta(y\mid c)=\sum_{t=1}^{T}\log p_\theta(y_t\mid c,y_{<t}),\qquad
y_t^{\mathrm{greedy}}=\operatorname*{arg\,max}_{a\in\mathcal V}p_\theta(a\mid c,y_{<t}).
$$

start를 나타내는 token이 달라지면 separator를 거쳐 end를 생성할 때의 prefix도 달라진다. 따라서 여러 start에 같은 end distribution을 재사용하지 않는다. 각 부분이 token 경계에서 나뉘는 표기의 경우 다음 조건부 확률로 쓸 수 있다.

$$
P_\theta(y^{s},y^{\mathrm{sep}},y^{e},y^{\mathrm{tail}}\mid c)
=P_\theta(y^{s}\mid c)\,
P_\theta(y^{\mathrm{sep}},y^{e},y^{\mathrm{tail}}\mid c,y^{s}).
$$

여기서 $y^s$는 start를 나타내는 token들, $y^{\mathrm{sep}}$는 그 뒤 구분자, $y^e$는 end를 나타내는 token들, $y^{\mathrm{tail}}$은 뒤 문구와 EOS다. 예를 들어 같은 `1` 다음에도 `.`·`4`·`1`을 선택하면 서로 다른 시각과 이후 end distribution으로 이어질 수 있다. **백의 자리·십의 자리 같은 decimal place는 완성된 숫자를 해석할 때 정해진다.** 처음 나온 `1`만 보고 백의 자리라고 확정하지 않는다. token 하나가 숫자 여러 자리나 구분자를 함께 포함하면 그 token을 쪼개지 않고 전체 token ID prefix를 조건으로 사용한다.

일반 timestamp 응답에는 seconds 형식을 요청했지만 별도 형식으로 강제 변환한 뒤 token distribution을 측정하지 않았다. parsing에서 `MM:SS`·`HH:MM:SS` 예외를 처리하는 계열도, 분포는 실제로 생성된 원래 token prefix에서 계산한다. 동일 시각의 표기만 같다고 context를 합치지 않는다. JSON proposal은 별도의 prompt를 추가한 model operation이므로 원래 timestamp 응답과 context 및 distribution이 다르다.

### Probability normalization — 어떤 집합 안에서의 확률인가

다음 세 값은 분모가 다르다. 전체 vocabulary에서 한 token이 나올 확률, 두 응답만 비교한 조건부 확률, 유한한 interval 후보 목록 안에서의 조건부 확률을 섞지 않는다.

$$
b_c(\mathrm{Yes})=
\frac{p_\theta(\mathrm{Yes}\mid c)}{p_\theta(\mathrm{Yes}\mid c)+p_\theta(\mathrm{No}\mid c)},\qquad
m_c=p_\theta(\mathrm{Yes}\mid c)+p_\theta(\mathrm{No}\mid c).
$$

$b_c$는 Yes/No 안에서 정규화한 값이고 $m_c$는 그 두 token의 원래 확률 합이다. 이 값이 높다고 사건이 실제로 정답일 확률이 그만큼이라고 보장되지는 않는다. A/B 비교도 같은 방식이다. 각 선택지가 단일 token인 것은 실행 시 확인한다.

후보 목록 $\mathcal C$에서 sequence log-likelihood를 정규화할 때는 다음을 쓴다. 이는 목록 밖의 interval을 제외한 분포이며, 영상의 모든 가능한 사건에 대한 posterior가 아니다.

$$
q_i=\frac{\exp(\ell_i-\ell_{\max})}{\sum_{j\in\mathcal C}\exp(\ell_j-\ell_{\max})},\qquad
\ell_{\max}=\max_{j\in\mathcal C}\ell_j.
$$

### Evaluation and cost — 성능과 계산량

query 수를 $N$, 해당 query의 평가 IoU를 $u_n$이라고 하면 $\mathrm{mIoU}=N^{-1}\sum_n u_n$, $R@\tau=N^{-1}\sum_n\mathbf1[u_n\ge\tau]$이며 $\tau\in\{0.3,0.5,0.7\}$이다. mIoU와 R은 높을수록 좋다. 둘 다 기본 범위는 0–1이며, 기존 표에서 %로 표시한 값은 100을 곱한 값이다. $\Delta R$의 pp는 $100(R_{\mathrm{method}}-R_{\mathrm{reference}})$다. 실패 응답의 처리와 interval 유효성은 각 계열의 실제 정책을 아래에 따로 명시한다.

후보 집합 $\mathcal C_K$에 대한 oracle은 $\max_{i\in\mathcal C_K}\mathrm{IoU}(I_i,G)$다. GT를 보고 고른 상한이므로 실제 selector의 성능과 분리한다. endpoint MAE는 $(\lvert s-s_G\rvert+\lvert e-e_G\rvert)/2$ seconds이며 낮을수록 좋다. 유효 응답만의 조건부 평균이면 그 분모를 따로 표시한다.

후보 수, node expansion 수, model operation 수, decoder input token 수는 서로 다른 예산이다. token 수에는 새 prompt 처리·prefix replay·생성 token의 실제 입력 처리를 구분해 기록하고, 영상 prefill은 별도로 표시한다. 같은 token 상한이 같은 FLOPs나 latency를 보장하지 않는다. exact cache reuse로 실제 실행이 줄어든 경우에도, 각 방법을 독립 배치할 때 필요한 비용과 실제 실행 비용을 구분한다.

## Implementation details — 위 절차의 수식과 고정 설정

### Inputs and contexts — 같은 영상으로 무엇을 다시 계산했는가

모델은 고정한 Qwen3-VL-8B-Instruct이고, 완료 범위는 pilot900이다. pilot300·600은 같은 순서의 prefix subset이다. pilot900에는 세 데이터셋별 300 query가 있다. 2 FPS·spatial target 100,352 pixels·video→query input order를 사용했고 frame ID·timestamp·입력 hash를 기존 deterministic rollout과 대조했다.

공통 video-query KV cache를 재사용한다. 새 inspection prompt는 원래 assistant turn을 빈 응답으로 닫은 뒤 새 user turn으로 구성하며, 원래 greedy 답 전체를 assistant history에 자동으로 붙이지 않는다. 각 호출의 prompt·필요한 ancestor observation·선택한 prefix가 그 호출의 context를 정한다. 같은 모델이어도 이 context가 달라지면 token distribution이 달라진다.

### Initial proposals — 후보 수집의 정확한 범위

$\mathcal C_8$은 기존 첫 8개 token 경로, $j$는 greedy 답의 첫 숫자 token 위치, $\oplus$는 token 순서를 이어 붙이는 연산이다. $\mathcal I_{\mathrm{JSON}}$은 별도 prompt에서 parsing한 최대 6개 interval이다.

$$
\mathcal I_0=\operatorname{unique}_{[s,e]}\left(
\operatorname{parse}(\mathcal C_8)\ \cup\
\{\operatorname{parse}(y_{<j}\oplus d\oplus\operatorname{greedy}(c,y_{<j}\oplus d)):d\in\{0,\ldots,9\}\}\ \cup\
\mathcal I_{\mathrm{JSON}}\right).
$$

첫 숫자는 raw token probability와 관계없이 10개를 모두 실행하고 뒤를 최대 64 token greedy decoding한다. JSON proposal은 최대 320 token이다. array parsing 실패 시 완성된 JSON object는 복구한다. initial proposals에서만 exact interval을 중복 제거하며 출처를 함께 남긴다. JSON prompt는 원래 timestamp 응답과 context가 다르므로, 두 호출의 probability를 같은 분포의 값처럼 합치지 않는다.

### Critic and value function — 후보를 남기는 기준과 평가값

먼저 interval이 $0\le s_i<e_i\le D+10^{-3}$을 만족해야 한다. duration 초과를 clipping하지 않는다. 기존 greedy baseline은 duration 초과를 flag만 했으므로 신규 후보와 유효성 기준이 다르다. 기존 성능 표는 원래 실행 정책을 유지한 값이다.

$E_i,C_i,B_i$는 각각 Event presence·Contradiction·Boundary fit 질문의 Yes/No 안에서 정규화한 Yes 확률이다. 위 Probability normalization 절의 $b_c$를 질문마다 따로 계산한다. 초기 후보에는 Boundary fit을 묻지 않고 $B_i=0.5$로 둔다.

$C_i\ge0.8$이고 raw Yes/No probability 합이 0.01 이상이면 확인 질문을 추가한다. 확인 질문에서도 Yes 확률≥0.8, raw probability 합≥0.01이어야 contradiction을 확인한 것으로 처리한다. 두 prompt는 같은 모델·영상에 대한 판단이므로 통계적으로 독립인 증거는 아니다.

$$
A_i=\mathbf1[0\le s_i<e_i\le D+10^{-3}]\,
\mathbf1[E_i\ge0.5]\,
\mathbf1[\text{contradiction not confirmed}],\qquad
V_i=A_i E_i(1-C_i)(0.5+0.5B_i).
$$

$A_i=1$이면 해당 후보 자체는 계속 사용할 수 있고, 0이면 사용하지 않는다. $V_i$는 search에 쓰는 heuristic value function이다. 예를 들어 사용 가능한 후보에서 $E_i=0.8,C_i=0.1,B_i=0.5$라면 $V_i=0.54$다. 이는 계산을 설명하기 위한 값이며 관측 결과가 아니다.

위 곱은 독립 사건의 결합 확률을 추정한 식이 아니다. 초기에는 $B_i=0.5$를 대입했으므로, refinement에서 **같은 interval을 다시 생성해도** Boundary fit을 실제 측정한 뒤 value가 달라질 수 있다. value 증가만으로 boundary 개선을 판정하지 않는다.

likelihood가 필요한 방법에는 interval을 `The event happens in <start> - <end> seconds.`로 작성하고 EOS를 붙여 원래 context에서 teacher forcing한 $\ell_i$도 준다. 숫자는 Python `:g` 표기다. 이는 canonical answer의 sequence log-likelihood이며 event posterior나 JSON proposal probability가 아니다.

### Tree expansion — 어떤 기록을 남기며 얼마나 깊게 검토하는가

가상 root의 depth는 0, 초기 event node의 depth는 1이다. 각 node에 `onset`·`offset`·`joint`를 한 번씩, 최대 3개 focus로 시도한다. depth≤3인 node를 확장하므로 자식은 depth 4까지 생긴다.

한 expansion은 timestamp regeneration(최대 64 token), 새 interval에 대한 시각 증거·불일치 관찰(최대 48 token), critic 질문들, canonical log-likelihood 평가 순서다. generation prompt에는 ancestor observation 일부를 전달한다. 유효한 timestamp는 EOS 도달·parsing 성공·$0\le s<e$가 필요하며 이어서 위 duration 기준을 적용한다.

부모·자식 interval의 IoU<0.2이면 다른 occurrence로 이동한 것으로 분류한다. 그 외에는 같은 event lineage, 즉 같은 occurrence에서 이어진 것으로 취급하는 node들의 묶음을 유지한다. 이 heuristic은 semantic identity의 GT가 아니다. 같은 lineage의 refinement에서 확인된 contradiction이 나오면 event root를 blocked로 바꾸어 그 묶음 전체를 제외한다.

대부분의 방법은 누적 refinement 4회마다 새 occurrence 하나를 JSON object로 제안한다(최대 96 token). 최근 검토한 interval·불일치·관찰 최대 12개를 보여준다. 확장 가능한 node가 없으면 최대 3회 proposal을 시도한다. `neural_controller`는 proposal을 직접 고를 수 있고, `commit_then_refine`·`token_seed_best_first`는 추가 proposal을 하지 않는다.

예산 부족으로 expansion 전체를 마치지 못하면 tree 변경은 취소하고, 이미 쓴 연산은 비용에 남긴다. 중간에 생성했더라도 commit되지 않은 child는 최종 후보에 넣지 않는다.

### Backup and frontier — 다음 선택을 위한 기록

**frontier** $\mathcal F$는 다음에 더 검토할 수 있는 node들의 집합이다. 해당 node가 eligible이고 같은 event root가 blocked가 아니며 depth≤3, 완료 focus 수 $k_i<3$이어야 한다. 최종 후보에는 남아 있어도 더 확장할 수 없는 node는 frontier에는 없다.

**backup**은 새 child의 평가값을 그 경로의 조상들에게도 반영하는 연산이다. 새 child의 value가 $v$이면 child부터 가상 root까지 $N_i$에 방문 1회를 더하고 $W_i$에 $v$를 더한다. $Q_i$는 누적 value의 평균이며 아직 방문하지 않았으면 자기 value를 쓴다.

$$
N_i\leftarrow N_i+1,\qquad W_i\leftarrow W_i+v,\qquad
Q_i=\begin{cases}W_i/N_i&N_i>0,\\V_i&N_i=0.\end{cases}
$$

이 방문은 독립적인 stochastic simulation이 아니다. generation은 greedy이고 cache가 같은 model operation을 재사용하기도 한다. UCT의 일반적 Monte Carlo 가정과 수렴 보장을 이번 IoU 성능에 그대로 적용하지 않는다. 방법별로 명시하지 않은 argmax 동률은 작은 node ID를 우선한다.

### Final selection — 최대 4개를 추린 뒤 모델로 두 개씩 비교

먼저 eligible이고 event root가 blocked가 아닌 node들을 모은다. 그 안에서 **Maximal Marginal Relevance (MMR)**로 최대 4개를 순차 선택한다. $S$는 이미 고른 집합이고 처음에는 빈 집합이다. $\max\varnothing=0$으로 둔다.

$$
i^*=\operatorname*{arg\,max}_{i\notin S}
\left[V_i-\alpha\max_{j\in S}\mathrm{IoU}(I_i,I_j)\right],\qquad\alpha=0.1.
$$

value가 높은 후보를 우선하되 이미 고른 interval과 비슷하면 penalty를 주는 식이다. 표준 MMR의 $\lambda V-(1-\lambda)\operatorname{similarity}$에서 $\lambda=1/(1+\alpha)$로 두면 같은 선택 순서가 된다. exact interval 중복 제거를 다시 하지 않는 것이 현재 구현의 한계다. [MMR 원 논문](https://doi.org/10.1145/290941.291025).

그다음 **knockout tournament**, 즉 한 쌍의 winner를 다음 round로 올리는 방식으로 비교한다. $b_{i,j}$는 “A=$i$, B=$j$일 때 A가 더 정확한가?”에 대한 Yes/No 정규화 확률이다. 순서를 바꾸어 두 번 묻고 다음처럼 합친다.

$$
\pi_{i,j}=\tfrac12\{b_{i,j}(\mathrm{Yes})+b_{j,i}(\mathrm{No})\},\qquad
\text{winner}=\begin{cases}i&\pi_{i,j}\ge0.5,\\j&\text{otherwise}.\end{cases}
$$

홀수 후보는 다음 round로 바로 올린다. 최후 winner의 기존 interval을 반환하고, 후보가 없거나 final selection 예산을 넘으면 실패다. proposal-only controls는 search expansion 없이 이 selector만 적용한다. `proposal_only_token_seed`는 기존 첫 8개만, `proposal_only_wide`는 전체 초기 proposal을 쓴다.

### Cost accounting — 표의 총 token 수가 search cap보다 큰 이유

각 방법을 초기 상태에서 search cap 4,096 / 8,192로 각각 평가하고, final selection에는 별도 2,048 token cap을 둔다. 큰 cap의 search에 작은 cap의 최종 비교 결과를 이어 쓰지 않는다.

$$
C_{\mathrm{total}}=C_{\mathrm{seed}}+C_{\mathrm{initial\ proposals}}+
C_{\mathrm{search}}+C_{\mathrm{selection}},\qquad
C_{\mathrm{search}}\le b,\quad b\in\{4096,8192\},\quad C_{\mathrm{selection}}\le2048.
$$

$C_{\mathrm{seed}}$는 기존 첫 8개 경로 생성, $C_{\mathrm{initial\ proposals}}$는 새 초기 후보 생성·평가, $C_{\mathrm{search}}$는 반복 재검토, $C_{\mathrm{selection}}$은 마지막 후보 비교의 decoder input token 비용이다. 공통 video prefill은 별도로 표시한다. `token_seed_best_first`와 token-seed control에는 필요한 seed 평가 비용만 부과한다.

query별 정확한 입력·parameter가 같은 model operation은 cache로 재사용한다. 비교 방법을 따로 실행했을 때의 비용에는 매번 필요한 비용을 다시 합산하고, 실제 실행한 token·시간은 별도 기록한다. prompt replay와 teacher forcing도 예산에 포함한다. model call 전에 prompt 길이와 최대 generation 길이로 상한을 확인하므로 남은 예산을 모두 소진하기 전에 멈출 수 있다.

구현 근거는 `_lib/event_tree_core.py`의 `initial_proposals`, `assess`, `Search.select`, `Search.expand`, `Search.final_select`다.

## Search policies — 다음에 무엇을 검토할지 고르는 방법

<details id="method-likelihood-best-first">
<summary>Best-first search — sequence log-likelihood 우선</summary>

실험 식별자: `likelihood_best_first`. 공통 frontier $\mathcal F$ 전체에서 다음 node를 고른다. $k_i$는 해당 node에서 완료한 focus 수다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(\ell_i-0.3k_i).
$$

동률은 depth가 작은 node, 그다음 작은 ID 순서다. 높은 canonical answer likelihood를 우선하되 같은 node에 계속 계산이 쏠리는 것을 0.3 penalty로 완화한다. 한 번 고르면 공통 expansion 한 번을 수행하고 frontier에서 다시 고른다. uniform-cost search나 A*의 admissible heuristic을 구현한 것은 아니며 최적 답을 보장하지 않는다. $\ell_i$는 현재 review history가 아닌 원래 video-query context의 값이다.

</details>

<details id="method-value-best-first">
<summary>Greedy best-first search — value function 우선</summary>

실험 식별자: `value_best_first`. 다음 평가식을 frontier 전체에 적용한다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(V_i-0.05k_i).
$$

critic value가 큰 node를 우선하고 이미 수행한 focus마다 0.05를 뺀다. 새로운 관찰로 tree가 갱신되면 다시 선택한다. $V_i$에는 future return의 학습 추정이나 정답 IoU가 들어 있지 않다. penalty와 value 모두 이 실험의 heuristic이며 논문에서 보장된 bound가 아니다.

</details>

<details id="method-value-beam">
<summary>Beam search — value function으로 width 4 유지</summary>

실험 식별자: `value_beam`. 한 번에 유지할 beam $\mathcal B$의 최대 크기는 4다. 새 beam을 만들 때 layer에서 $V_i$가 높은 4개를 고른다.

$$
\mathcal B\leftarrow\operatorname{Top4}_{i\in\mathcal L}V_i,\qquad
i^*=\operatorname*{arg\,min}_{i\in\mathcal B\cap\mathcal F}(k_i,-V_i,i).
$$

현재 beam에 확장 가능한 node가 남으면 그 beam을 먼저 사용한다. 모두 소진하면 직전 beam의 확장 가능한 child들을 다음 layer로 사용하고, 그것도 없으면 전체 frontier에서 가장 얕은 depth를 사용한다. 같은 beam 안에서는 focus를 적게 수행한 node부터 확장한다. 탈락한 frontier를 영구 삭제하지 않으므로, 매 token step에서 width 밖을 버리는 표준 sequence beam search와 다르다. 여기서 layer는 action tree의 interval 상태이며 generation step이 아니다.

</details>

<details id="method-diverse-beam">
<summary>Diversity-promoting beam search — MMR로 중복 interval 억제</summary>

실험 식별자: `diverse_beam`. `value_beam`과 layer 이동·width 4·focus 순서는 같다. 차이는 beam의 4개 node를 MMR로 고르는 것이다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal L\setminus S}
\left[V_i-0.25\max_{j\in S}\mathrm{IoU}(I_i,I_j)\right],\qquad S\leftarrow S\cup\{i^*\}.
$$

value가 조금 낮아도 기존 선택과 다른 시간대인 node가 들어갈 수 있다. 원래 [Diverse Beam Search](https://arxiv.org/abs/1610.02424)의 grouped sequence decoding을 재현하지 않았다. 이번 구현은 interval IoU를 사용하는 MMR beam selection이며, diversity penalty 0.25가 최종 selector의 0.1과 다르다.

</details>

<details id="method-uct">
<summary>UCT with progressive widening — 평균 value와 방문 불확실성으로 탐색</summary>

실험 식별자: `uct`. UCT는 **Upper Confidence bounds applied to Trees**다. parent $p$에서 고려할 수 있는 child $i$ 중 다음 식이 큰 쪽으로 내려간다.

$$
U_i=Q_i+1.4\sqrt{\frac{\log(N_p+2)}{N_i+1}}.
$$

첫 항은 관측한 descendant value의 평균이고 두 번째 항은 방문이 적은 child를 다시 볼 기회다. 미방문 node는 $Q_i=V_i$로 시작한다. 원래 UCT의 무한 우선순위 초기 방문 대신 분모에 1을 더한 유한 bonus를 쓴다. 동률은 작은 ID 순이다. [UCT 원 논문](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf).

두 방법 모두 root의 child를 MMR($\alpha=0.25$) 순서로 놓고 아래 개수만 고려한다. $r$은 가상 root, $R$은 확장 가능한 descendant가 있는 root child 수다.

$$
L_r=\min\{R,\ 2+\lfloor2\sqrt{N_r+1}\rfloor\}.
$$

선택된 child $i$가 확장 가능하고, 아직 child가 없거나 $k_i<\min\{3,1+\lfloor\sqrt{N_i}\rfloor\}$이면 그 node에 새 focus를 연다. 그렇지 않으면 기존 child로 내려간다. 이것이 **progressive widening**, 즉 방문이 늘 때 고려하는 action 수를 점차 늘리는 절차다. root에서 열지 않은 사건이 자동으로 사라지는 것은 아니다. 같은 node의 동일 focus를 반복 simulation하지 않는다.

이 구현은 deterministic greedy expansion과 같은 VLM의 heuristic critic을 사용한다. 독립 reward sample이나 정확한 simulator를 가정한 UCT의 수렴 속도를 이번 IoU 성능에 대한 보장으로 사용하지 않는다.

</details>

<details id="method-puct">
<summary>PUCT with progressive widening — event probability를 prior로 사용</summary>

실험 식별자: `puct`. tree를 내려가는 방식은 UCT와 같고 bonus에 child prior를 넣는다. $\mathcal H_p$는 현재 고려하는 parent의 child 집합이다.

$$
P_i=0.9\frac{\max(E_i,10^{-6})}{\sum_{j\in\mathcal H_p}\max(E_j,10^{-6})}
+\frac{0.1}{\lvert\mathcal H_p\rvert},\qquad
U_i=Q_i+1.4P_i\frac{\sqrt{N_p+1}}{N_i+1}.
$$

90%는 event presence critic, 10%는 uniform prior로 둔다. 이는 canonical answer likelihood나 학습한 policy network의 prior가 아니다. [AlphaZero의 prior-guided tree search](https://arxiv.org/abs/1712.01815)와 같은 PUCT 형식을 사용하지만 self-play 학습·Dirichlet noise·학습된 별도 value model은 없다.

두 방법 모두 root의 child를 MMR($\alpha=0.25$) 순서로 놓고 아래 개수만 고려한다. $r$은 가상 root, $R$은 확장 가능한 descendant가 있는 root child 수다.

$$
L_r=\min\{R,\ 2+\lfloor2\sqrt{N_r+1}\rfloor\}.
$$

선택된 child $i$가 확장 가능하고, 아직 child가 없거나 $k_i<\min\{3,1+\lfloor\sqrt{N_i}\rfloor\}$이면 그 node에 새 focus를 연다. 그렇지 않으면 기존 child로 내려간다. 이것이 **progressive widening**, 즉 방문이 늘 때 고려하는 action 수를 점차 늘리는 절차다. root에서 열지 않은 사건이 자동으로 사라지는 것은 아니다. 같은 node의 동일 focus를 반복 simulation하지 않는다.

</details>

<details id="method-neural-controller">
<summary>Model-based action selection — 모델이 다음 inspection 결정</summary>

실험 식별자: `neural_controller`. 동일한 frozen VLM을 controller로도 사용한다. $h$는 최근 action 기록과 frontier 요약, $a$는 refine/propose·node ID·focus를 담은 JSON action이다.

$$
a=\operatorname{parse}_{\mathrm{JSON}}\bigl(\operatorname{greedy}_\theta(c,\operatorname{prompt}(h))\bigr).
$$

MMR($\alpha=0.1$)로 추린 frontier 최대 8개와 최근 4개 action을 보여주고 최대 48 token으로 action을 생성한다. `propose`면 새 occurrence를 요청하고, `refine`이면 모델이 고른 유효한 node를 확장한다. node가 유효하지 않으면 value best-first 선택으로 fallback한다. focus가 잘못됐거나 이미 수행됐으면 남은 첫 focus를 사용한다. 새 policy를 학습한 것이 아니며 controller가 critic과 독립인 모델도 아니다. action 계획에 사용한 model 연산도 search 예산에 넣었다. [Language Agent Tree Search](https://arxiv.org/abs/2310.04406)는 참고한 방향이며 해당 논문의 전체 agent 시스템 재현은 아니다.

</details>

<details id="method-self-refine">
<summary>Iterative refinement with feedback — 현재 occurrence를 이어서 수정</summary>

실험 식별자: `self_refine`. 처음에는 value가 가장 높은 node를 고른다. 이후 현재 event lineage의 frontier가 있으면 다음 tuple이 큰 node를 선택한다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F:\,\operatorname{event}(i)=\operatorname{event}(\mathrm{current})}
(\operatorname{depth}_i,V_i,-k_i,-i).
$$

더 깊은 refinement를 먼저 이어가고, 같은 event에 확장 가능한 node가 없으면 전체 frontier에서 최대 value로 옮긴다. 공통 주기에 따라 새 proposal도 수행하고, regeneration이 다른 occurrence로 이동하면 current도 따라 바뀐다. 따라서 엄격히 하나의 answer만 반복 수정하는 방법은 아니다. [Self-Refine](https://arxiv.org/abs/2303.17651)의 feedback→refinement 구조를 적용했지만 여기서는 별도 critic·여러 event·공통 최종 selector를 결합했다.

</details>

<details id="method-submodular-coverage">
<summary>Monotone submodular maximization — facility location objective</summary>

실험 식별자: `submodular_coverage`. 이 방법이 analytic 비교다. 현재 확장 가능한 descendant를 가진 event root의 집합을 $\mathcal U$라 한다. node가 아니라 서로 다른 event root를 먼저 최대 4개 고른다. interval 중심을 $c_i=(s_i+e_i)/2$라 두면 다음과 같다.

$$
w_i=0.9\frac{V_i}{Z}+\frac{0.1}{\lvert\mathcal U\rvert},\qquad
\tau=\max(2,0.05D),\qquad
K(i,j)=0.5\mathrm{IoU}(I_i,I_j)+0.5\exp(-\lvert c_i-c_j\rvert/\tau).
$$

$$
F(S)=\sum_{i\in\mathcal U}w_i\max_{j\in S}K(i,j)+0.2\sum_{j\in S}V_j,\qquad F(\varnothing)=0.
$$

$Z=\sum_{u\in\mathcal U}V_u$이며 합이 0이면 구현대로 $Z=1$로 둔다. 첫 항은 선택된 대표가 각 후보와 얼마나 가까운지 보는 **facility location** objective다. 둘째 항은 높은 value의 후보를 더 선호하는 modular 항이다. $w_i$가 nonnegative이므로 고정된 후보·평가값에서 $F$는 monotone submodular다. 즉 선택을 늘릴수록 objective가 줄지 않고, 이미 선택한 것이 많아질수록 새 후보 하나의 추가 이익은 작아진다.

$$
S\leftarrow S\cup\operatorname*{arg\,max}_{j\in\mathcal U\setminus S}
\{F(S\cup\{j\})-F(S)\},\qquad \lvert S\rvert\le4.
$$

매 step에서 이 greedy marginal selection을 다시 한다. 뽑힌 최대 4개 event 중 방문이 가장 적은 root를 고르고, 그 event 안에서는 $V_i-0.05k_i$가 큰 node를 확장한다. root 동률은 value, 작은 ID 순이다.

cardinality 제약과 고정된 $\mathcal U,w,K,V$에서 greedy는 $F(S_{\mathrm{greedy}})\ge(1-1/e)F(S^*)$를 만족한다. 이 보장은 **그 순간의 $F$에만** 해당한다. 각 expansion의 token 비용이 같다는 가정도 두지 않았으며, GT IoU·최종 selector·시간에 따라 변하는 전체 tree search의 성능을 보장하지 않는다. [Submodular summarization의 facility location](https://aclanthology.org/P11-1052/), [고전 greedy approximation 결과](https://doi.org/10.1007/BF01588971).

</details>

<details id="method-reward-balanced">
<summary>Highest averages method — softmax value weight로 방문 배분</summary>

실험 식별자: `reward_balanced`. 이 이름은 저장된 실험 식별자다. 실제 root 배분식은 fixed weight일 때 **Highest averages method의 D’Hondt rule**과 같은 $w/(n+1)$ 형태다. 이번 weight는 vote가 아니라 critic value의 exponential이다.

$$
w_r=\exp(V_r/0.25),\qquad
r^*=\operatorname*{arg\,max}_{r}\frac{w_r}{N_r+1},\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F:\,\operatorname{event}(i)=r^*}(V_i-0.05k_i).
$$

value가 높은 event를 우선하지만 방문할수록 분모가 커져 다른 event에 기회를 준다. weight가 고정되고 모든 root가 계속 열려 있는 단순 배분 문제와 달리, 이 tree에서는 새 event가 추가되고 root가 blocked·exhausted될 수 있다. 따라서 고정 배분의 최적성이나 best-arm identification 보장을 주장하지 않는다. REBASE의 재현도 아니다. [D’Hondt rule 공식 설명](https://www.europarl.europa.eu/thinktank/en/academic/EPRS_BRI%282024%29762352).

</details>

<details id="method-sequential-halving">
<summary>Successive Halving — event 후보를 단계적으로 절반씩 축소</summary>

실험 식별자: `sequential_halving`. **Successive Halving / Sequential Halving** 계열의 제거 방식을 event root에 적용한 변형이다. 처음 active set $\mathcal A$는 MMR($\alpha=0.15$)로 최대 8개 root를 고른다. 현재 최소 방문 수에 1을 더한 값을 첫 target $t$로 둔다.

$$
M_r=\max_{i\in\operatorname{descendants}(r)}V_i,\qquad
\mathcal A\leftarrow\operatorname{Top}_{\lceil\lvert\mathcal A\rvert/2\rceil}M_r,\qquad
t\leftarrow2t+1.
$$

active root 모두 $N_r\ge t$일 때 $M_r$가 높은 절반만 남긴다. 다음 확장은 방문 수가 가장 작은 active root, 그 안의 $V_i-0.05k_i$ 최대 node다. active root가 모두 사용 불가가 되면 현재 가능한 root에서 set을 다시 만든다. 제거한 event가 최종 후보 집합에서 자동 삭제되는 것은 아니다.

원래 fixed-budget Sequential Halving처럼 총 budget을 round별로 균등 분할하지 않았고, arm reward의 표본 평균 대신 **가장 높은 descendant critic value**를 사용했다. 정확한 논문 알고리즘의 재현이나 regret 보장은 아니다. [Successive Halving 원 연구](https://proceedings.mlr.press/v51/jamieson16.html).

</details>

<details id="method-nucleus-frontier">
<summary>Nucleus truncation — 후보 분포의 cumulative probability 0.95로 제한</summary>

실험 식별자: `nucleus_frontier`. exact interval을 중복 제거한 사용 가능한 event root 목록에서 canonical sequence log-likelihood $\ell_r$를 softmax하여 $q_r$를 만든다. 이 분포는 위 공통 절의 유한 후보 목록 분포다.

$$
q_{(1)}\ge q_{(2)}\ge\cdots,\qquad
m=\min\left\{h:\sum_{j=1}^{h}q_{(j)}\ge0.95\right\},\qquad
\mathcal A=\{(1),\ldots,(m)\}.
$$

$$
r^*=\operatorname*{arg\,max}_{r\in\mathcal A}\frac{q_r}{N_r+1}.
$$

그 event 안의 확장 node는 $V_i-0.05k_i$로 고른다. **sampling은 하지 않는다.** [Nucleus sampling](https://arxiv.org/abs/1904.09751)의 cumulative probability에 따른 truncation만 event 목록에 옮긴 것이다. raw full-vocabulary token top-p, stochastic rollout, 모든 가능한 event의 probability 95% coverage와는 다르다.

</details>

<details id="method-typicality-frontier">
<summary>Locally typical truncation — surprisal과 entropy 차이로 제한</summary>

실험 식별자: `typicality_frontier`. 후보 집합의 $q_r$는 nucleus와 같다. **surprisal**은 $-\log q_r$, **entropy**는 그 기대값 $H(q)$다. 확률이 큰 순서 대신 두 값의 차이가 작은 순서로 후보를 정렬한다.

$$
H(q)=-\sum_rq_r\log q_r,\qquad d_r=\lvert-\log q_r-H(q)\rvert.
$$

$d_r$ 오름차순에서 cumulative probability가 0.95 이상이 되는 최소 집합을 고른 뒤, 그 안의 root를 $q_r/(N_r+1)$로 선택한다. 이후 node 선택은 $V_i-0.05k_i$다. [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)의 entropy 기반 truncation을 유한 event 목록에 적용했으며 이 구현에는 sampling이 없다. 분포가 충분히 집중되면 nucleus와 같은 집합이 될 수 있다. 따라서 이름이 다르다는 것만으로 다른 경로를 탐색했다고 판단하지 않는다.

</details>

<details id="method-commit-then-refine">
<summary>Ablation — 한 occurrence에 고정한 iterative refinement</summary>

실험 식별자: `commit_then_refine`. 공통 critic을 통과한 후보 중 처음 value가 가장 높은 event $r_0$를 고정한다.

$$
\mathcal F_{r_0}=\{i\in\mathcal F:\operatorname{event}(i)=r_0\},\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F_{r_0}}(\operatorname{depth}_i,V_i,-k_i,-i).
$$

새 occurrence proposal을 하지 않고, refinement가 부모와 IoU<0.2인 다른 occurrence로 이동하면 그 child를 사용 불가로 만든다. 최종 selector도 $r_0$ lineage 안으로 제한한다. 해당 event가 blocked되거나 더 확장할 수 없으면 다른 event로 옮기지 않고 중단한다. 사건을 다시 고를 수 있는 것이 도움이 되는지를 보는 ablation이지 별도 논문 알고리즘의 이름이 아니다.

</details>

<details id="method-token-seed-best-first">
<summary>Ablation — 기존 token 경로 8개만 사용하는 best-first search</summary>

실험 식별자: `token_seed_best_first`. 초기 node를 기존 deterministic rollout 첫 8개에서 온 interval로만 만든다. 새 leading-digit 경로와 JSON proposal은 이 방법에 주지 않고, search 중 새 occurrence proposal도 하지 않는다.

$$
\mathcal I_0=\operatorname{unique}_{[s,e]}\operatorname{parse}(\mathcal C_8),\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(V_i-0.05k_i).
$$

경계 regeneration 자체가 다른 occurrence로 이동하는 것은 허용한다. critic·tree depth·final pairwise selector는 공통이다. initial proposal 비용도 seed 평가에 필요한 부분만 부과한다. `value_best_first`와의 차이는 **후보 공급 범위**이며, search selection 공식을 바꾼 비교가 아니다.

</details>

<!-- rollout-methods:end -->
<!-- report-part:1:end -->
<!-- report-part:5:begin -->
## 사건 proposal과 oracle

duration-valid는 시간 범위 검사 통과 후 model gate 이전이며, gated는 사건 적합성 gate 이후다. 시간 범위 검사 이전의 oracle은 실패 분석에 별도 표시한다. oracle은 GT를 보고 가장 좋은 후보를 고른 상한이며 실제 선택 결과가 아니다. new는 실제 생성된 유효 unique interval이며 의미적으로 다른 사건 개수가 아니다.

| source | average unique candidates | oracle mIoU | R@.3 | R@.5 | R@.7 |
|---|---:|---:|---:|---:|---:|
| old paths first8 | 7.90 | 0.6717 | 82.78 | 70.78 | 56.89 |
| old paths first32 | 30.27 | 0.7347 | 87.67 | 77.78 | 65.00 |
| new proposals duration-valid | 16.10 | 0.6983 | 86.00 | 75.00 | 59.11 |
| new proposals gated | 10.96 | 0.6523 | 81.22 | 70.00 | 55.11 |

Gate가 raw 최고 IoU를 낮춘 query: 150/900. old8 oracle<.5 → new raw oracle≥.5: 59/263. 이는 gate의 false negative와 proposal coverage를 사후 GT로 분석한 값이다.
Temporal complete-link cluster(IoU≥.5)의 query당 평균은 old8 2.12 → new proposals 5.72다. 이는 의미적 사건 정체성 gold가 아니며 threshold .3/.7 원자료도 보존한다.

| unique candidate budget | old oracle mIoU | new oracle mIoU | old R@.3 | new R@.3 | old R@.5 | new R@.5 | old R@.7 | new R@.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5193 | 0.4985 | 67.67 | 64.89 | 54.56 | 52.22 | 39.67 | 37.78 |
| 4 | 0.6399 | 0.6187 | 79.89 | 77.33 | 67.56 | 65.00 | 53.78 | 51.44 |
| 8 | 0.6719 | 0.6532 | 82.78 | 80.78 | 70.78 | 68.78 | 56.89 | 54.78 |
| 16 | 0.7063 | 0.6928 | 85.33 | 85.33 | 75.22 | 74.22 | 61.67 | 58.44 |
| 32 | 0.7347 | 0.6983 | 87.67 | 86.00 | 77.78 | 75.00 | 65.00 | 59.11 |

<!-- report-part:5:end -->
<!-- report-part:6:begin -->
## 사건 선택 이후 boundary refinement와 backtracking

root는 최종 선택한 사건의 coarse interval이다. Δroot는 final IoU−root IoU의 paired-valid 평균이다. MAE는 두 endpoint 절대오차 평균(초), 낮을수록 좋다. 모순은 모델이 두 질문에서 확인한 mismatch이며 실제 사실로 확정된 라벨이 아니다. backtrack은 모순 이후 실제 다른 event node를 확장한 횟수다.

| method | paired n | root mIoU | final mIoU | Δroot | root→final MAE | mean backtracks | oracle−selected |
|---|---:|---:|---:|---:|---|---:|---:|
| likelihood_best_first@4096 | 858 | 0.4162 | 0.4220 | +0.0058 | 12.991→12.889 | 0.064 | 0.3087 |
| likelihood_best_first@8192 | 858 | 0.4170 | 0.4223 | +0.0053 | 13.001→12.886 | 0.083 | 0.3094 |
| value_best_first@4096 | 858 | 0.3745 | 0.3978 | +0.0232 | 15.024→14.668 | 0.078 | 0.3354 |
| value_best_first@8192 | 858 | 0.3756 | 0.3957 | +0.0201 | 15.014→14.705 | 0.103 | 0.3386 |
| value_beam@4096 | 858 | 0.4087 | 0.4251 | +0.0165 | 13.689→13.431 | 0.091 | 0.3081 |
| value_beam@8192 | 858 | 0.4035 | 0.4182 | +0.0146 | 14.000→13.787 | 0.103 | 0.3162 |
| diverse_beam@4096 | 858 | 0.4062 | 0.4209 | +0.0148 | 13.746→13.487 | 0.097 | 0.3129 |
| diverse_beam@8192 | 858 | 0.3957 | 0.4097 | +0.0140 | 14.471→14.249 | 0.103 | 0.3252 |
| uct@4096 | 858 | 0.4108 | 0.4254 | +0.0146 | 13.542→13.249 | 0.110 | 0.3094 |
| uct@8192 | 858 | 0.4082 | 0.4210 | +0.0127 | 13.916→13.691 | 0.167 | 0.3168 |
| puct@4096 | 858 | 0.4104 | 0.4211 | +0.0107 | 13.662→13.505 | 0.107 | 0.3127 |
| puct@8192 | 858 | 0.4096 | 0.4196 | +0.0100 | 13.834→13.744 | 0.163 | 0.3173 |
| neural_controller@4096 | 860 | 0.4031 | 0.4170 | +0.0138 | 13.608→13.323 | 0.045 | 0.3140 |
| neural_controller@8192 | 858 | 0.4045 | 0.4215 | +0.0171 | 13.659→13.296 | 0.064 | 0.3099 |
| self_refine@4096 | 858 | 0.3727 | 0.3979 | +0.0252 | 15.088→14.630 | 0.066 | 0.3342 |
| self_refine@8192 | 858 | 0.3725 | 0.3973 | +0.0248 | 15.064→14.615 | 0.082 | 0.3360 |
| submodular_coverage@4096 | 858 | 0.4098 | 0.4278 | +0.0180 | 13.742→13.505 | 0.113 | 0.3067 |
| submodular_coverage@8192 | 858 | 0.4005 | 0.4154 | +0.0150 | 13.957→13.770 | 0.146 | 0.3206 |
| reward_balanced@4096 | 858 | 0.4118 | 0.4287 | +0.0169 | 13.596→13.314 | 0.105 | 0.3061 |
| reward_balanced@8192 | 858 | 0.4058 | 0.4197 | +0.0139 | 13.810→13.583 | 0.166 | 0.3177 |
| sequential_halving@4096 | 858 | 0.4111 | 0.4279 | +0.0169 | 13.416→13.056 | 0.106 | 0.3068 |
| sequential_halving@8192 | 858 | 0.4030 | 0.4151 | +0.0121 | 14.027→13.809 | 0.152 | 0.3215 |
| nucleus_frontier@4096 | 858 | 0.4174 | 0.4231 | +0.0056 | 12.940→12.855 | 0.065 | 0.3077 |
| nucleus_frontier@8192 | 858 | 0.4191 | 0.4243 | +0.0053 | 13.070→12.996 | 0.080 | 0.3077 |
| typicality_frontier@4096 | 858 | 0.4174 | 0.4231 | +0.0056 | 12.940→12.855 | 0.065 | 0.3077 |
| typicality_frontier@8192 | 858 | 0.4191 | 0.4243 | +0.0053 | 13.070→12.996 | 0.080 | 0.3077 |
| commit_then_refine@4096 | 833 | 0.3377 | 0.3830 | +0.0453 | 17.017→16.046 | 0.000 | 0.3449 |
| commit_then_refine@8192 | 833 | 0.3377 | 0.3819 | +0.0442 | 17.017→16.060 | 0.000 | 0.3465 |
| token_seed_best_first@4096 | 807 | 0.4752 | 0.4781 | +0.0029 | 11.555→11.492 | 0.001 | 0.2003 |
| token_seed_best_first@8192 | 807 | 0.4822 | 0.4851 | +0.0029 | 11.360→11.306 | 0.006 | 0.1935 |

<!-- report-part:6:end -->
<!-- report-part:8:begin -->
## Paired video bootstrap

400회, seed42, video 단위로 모든 query와 조건을 함께 재표집했다. 아래는 greedy 대비95% CI다. CI가0을 포함하면 해당 차이를 확정하지 않는다.

| method | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| greedy | +0.0000 [+0.0000,+0.0000] | +0.00 [+0.00,+0.00] |
| proposal_only_token_seed | -0.0529 [-0.0705,-0.0338] | -6.67 [-9.37,-3.75] |
| proposal_only_wide | -0.1240 [-0.1447,-0.1027] | -14.56 [-17.77,-11.43] |
| likelihood_best_first@4096 | -0.1170 [-0.1368,-0.0953] | -13.44 [-16.35,-10.25] |
| likelihood_best_first@8192 | -0.1167 [-0.1366,-0.0956] | -13.33 [-16.20,-10.09] |
| value_best_first@4096 | -0.1401 [-0.1606,-0.1172] | -16.44 [-19.31,-13.11] |
| value_best_first@8192 | -0.1420 [-0.1611,-0.1224] | -16.44 [-19.46,-13.35] |
| value_beam@4096 | -0.1140 [-0.1341,-0.0940] | -13.33 [-16.43,-10.38] |
| value_beam@8192 | -0.1206 [-0.1418,-0.1002] | -14.44 [-17.79,-11.58] |
| diverse_beam@4096 | -0.1180 [-0.1379,-0.0984] | -14.11 [-17.13,-11.02] |
| diverse_beam@8192 | -0.1287 [-0.1491,-0.1083] | -15.78 [-18.95,-12.57] |
| uct@4096 | -0.1137 [-0.1340,-0.0924] | -13.00 [-15.91,-9.77] |
| uct@8192 | -0.1180 [-0.1376,-0.0981] | -13.89 [-17.02,-10.72] |
| puct@4096 | -0.1178 [-0.1377,-0.0982] | -13.44 [-16.58,-10.42] |
| puct@8192 | -0.1192 [-0.1392,-0.0988] | -14.44 [-17.37,-11.31] |
| neural_controller@4096 | -0.1208 [-0.1407,-0.0990] | -13.89 [-17.11,-10.64] |
| neural_controller@8192 | -0.1174 [-0.1362,-0.0964] | -13.67 [-16.65,-10.16] |
| self_refine@4096 | -0.1399 [-0.1587,-0.1170] | -17.00 [-19.80,-13.59] |
| self_refine@8192 | -0.1405 [-0.1599,-0.1177] | -16.89 [-19.69,-13.56] |
| submodular_coverage@4096 | -0.1114 [-0.1308,-0.0921] | -12.56 [-15.80,-9.53] |
| submodular_coverage@8192 | -0.1232 [-0.1446,-0.1039] | -14.44 [-17.43,-11.32] |
| reward_balanced@4096 | -0.1106 [-0.1314,-0.0892] | -13.00 [-16.17,-9.89] |
| reward_balanced@8192 | -0.1192 [-0.1393,-0.0988] | -13.67 [-16.83,-10.58] |
| sequential_halving@4096 | -0.1113 [-0.1315,-0.0908] | -13.11 [-16.37,-10.05] |
| sequential_halving@8192 | -0.1236 [-0.1438,-0.1044] | -14.78 [-17.94,-11.83] |
| nucleus_frontier@4096 | -0.1160 [-0.1361,-0.0943] | -13.44 [-16.21,-10.30] |
| nucleus_frontier@8192 | -0.1147 [-0.1359,-0.0931] | -13.00 [-15.58,-10.09] |
| typicality_frontier@4096 | -0.1160 [-0.1361,-0.0943] | -13.44 [-16.21,-10.30] |
| typicality_frontier@8192 | -0.1147 [-0.1359,-0.0931] | -13.00 [-15.58,-10.09] |
| commit_then_refine@4096 | -0.1648 [-0.1848,-0.1424] | -19.11 [-22.15,-15.90] |
| commit_then_refine@8192 | -0.1658 [-0.1862,-0.1429] | -19.22 [-22.16,-15.81] |
| token_seed_best_first@4096 | -0.0906 [-0.1097,-0.0703] | -10.33 [-13.26,-7.44] |
| token_seed_best_first@8192 | -0.0843 [-0.1031,-0.0640] | -9.33 [-11.97,-6.55] |

<!-- report-part:8:end -->
<!-- report-part:9:begin -->
### Budget scaling와 ablation의 paired 차이

| comparison | ΔmIoU [95% CI] | ΔR@.5 [95% CI] |
|---|---|---|
| likelihood_best_first@8192 minus likelihood_best_first@4096 | +0.0003 [-0.0024,+0.0032] | +0.11 [-0.33,+0.66] |
| value_best_first@8192 minus value_best_first@4096 | -0.0020 [-0.0076,+0.0038] | +0.00 [-1.00,+1.00] |
| value_beam@8192 minus value_beam@4096 | -0.0066 [-0.0132,+0.0005] | -1.11 [-2.21,+0.00] |
| diverse_beam@8192 minus diverse_beam@4096 | -0.0107 [-0.0186,-0.0031] | -1.67 [-2.88,-0.45] |
| uct@8192 minus uct@4096 | -0.0042 [-0.0116,+0.0039] | -0.89 [-2.23,+0.55] |
| puct@8192 minus puct@4096 | -0.0014 [-0.0088,+0.0066] | -1.00 [-2.22,+0.44] |
| neural_controller@8192 minus neural_controller@4096 | +0.0034 [-0.0050,+0.0119] | +0.22 [-1.33,+1.67] |
| self_refine@8192 minus self_refine@4096 | -0.0006 [-0.0023,+0.0010] | +0.11 [-0.22,+0.45] |
| submodular_coverage@8192 minus submodular_coverage@4096 | -0.0118 [-0.0195,-0.0036] | -1.89 [-3.23,-0.44] |
| reward_balanced@8192 minus reward_balanced@4096 | -0.0086 [-0.0167,-0.0006] | -0.67 [-1.66,+0.44] |
| sequential_halving@8192 minus sequential_halving@4096 | -0.0123 [-0.0202,-0.0041] | -1.67 [-3.00,-0.33] |
| nucleus_frontier@8192 minus nucleus_frontier@4096 | +0.0012 [-0.0023,+0.0043] | +0.44 [-0.22,+1.11] |
| typicality_frontier@8192 minus typicality_frontier@4096 | +0.0012 [-0.0023,+0.0043] | +0.44 [-0.22,+1.11] |
| commit_then_refine@8192 minus commit_then_refine@4096 | -0.0010 [-0.0029,+0.0004] | -0.11 [-0.45,+0.33] |
| token_seed_best_first@8192 minus token_seed_best_first@4096 | +0.0063 [-0.0002,+0.0126] | +1.00 [-0.11,+2.01] |
| self_refine@4096 minus commit_then_refine@4096 | +0.0249 [+0.0108,+0.0388] | +2.11 [+0.11,+4.13] |
| self_refine@8192 minus commit_then_refine@8192 | +0.0253 [+0.0117,+0.0394] | +2.33 [+0.33,+4.42] |
| value_best_first@4096 minus token_seed_best_first@4096 | -0.0495 [-0.0666,-0.0319] | -6.11 [-8.58,-3.44] |
| value_best_first@8192 minus token_seed_best_first@8192 | -0.0577 [-0.0738,-0.0407] | -7.11 [-9.67,-4.54] |

<!-- report-part:9:end -->
