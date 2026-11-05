# Deterministic rollout · pilot300 · Methods

<!-- rollout-methods:begin -->

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

## Shared pipeline — Deterministic rollout 후보 생성

Qwen3-VL-8B-Instruct, 2 FPS, spatial target 100,352 pixels, 고정 pilot300→600→900을 사용했다. 각 query의 같은 video-query context에서 greedy 답을 후보 1로 두고, 서로 다른 완성 token 경로를 최대 32개 만든다. 예산 $K\in\{1,4,8,16,32\}$는 **생성 순서에서 앞선 완성 경로 수**다. event 개수·유효 interval 개수·node expansion 수가 아니다.

1. 완성 응답에서 시간 숫자와 `.`·`:`에 해당하는 token 위치를 찾는다. 각 위치의 full-vocabulary 확률 중 숫자·구분자에 해당하는 token만 분기 대상으로 고려한다.
2. 원래 선택을 포함해 위치당 최대 4개, raw probability≥$10^{-4}$, raw probability 합 목표 0.95를 사용한다. 이 제약 때문에 반드시 0.95에 도달하는 것은 아니다. 분기 대상으로 걸렀다고 확률을 다시 정규화하지 않는다.
3. 대체 token 하나를 강제하고 이후 전체 vocabulary에서 greedy decoding한다. 최대 512 token까지 생성하고 start가 바뀌면 end도 새 prefix 뒤에서 다시 생성한다.
4. 얻은 경로에서 마지막 강제 분기 뒤의 위치를 다시 분기할 수 있다. 같은 강제 prefix와 동일한 완성 token 경로의 중복 실행을 막는다. 같은 interval을 다른 token으로 표현한 경로는 별도 후보다.

대기 중인 분기 $(y_{<j},a)$의 우선순위는 다음 prefix log-probability다.

$$
U(y_{<j},a)=\sum_{t<j}\log p_\theta(y_t\mid c,y_{<t})+\log p_\theta(a\mid c,y_{<j}).
$$

높은 $U$부터 완성한다. suffix probability≤1이므로 같은 context에서는 이 prefix probability가 뒤에 이어질 한 경로 확률의 upper bound다. 하지만 분기 token 수·확률 cutoff·greedy suffix로 탐색을 제한하므로 **전체 가능한 응답의 정확한 top-32**라는 보장은 없다. 무작위 sampling도 아니다.

`start.seconds.10^2`는 완성된 start 숫자의 백의 자리, `end.seconds.decimal_separator`는 완성된 end 숫자의 decimal point에서 대체 token을 강제한 위치를 뜻한다. 이 label은 **분기 전 parent 응답**이 완성된 뒤 부여하며, 분기 후 완성 응답의 자리는 별도 `target_roles`에 기록한다. decimal point를 다른 숫자로 바꾸면 자릿수와 시간 규모 자체가 달라질 수 있으므로 그 label만으로 local refinement라고 분류하지 않는다. `MM:SS`·`HH:MM:SS`는 예외 parsing이며 생성 형식을 바꾸는 instruction을 추가하지 않았다.

후보의 유효 조건은 EOS 도달·timestamp parsing 성공·$0\le s<e$다. duration 초과는 flag를 기록하지만 selector에서 제외하지 않았다. invalid는 평가 IoU 0이며 valid 후보가 하나도 없으면 후보 1을 반환한다. canonical string으로 다시 작성한 확률이 아니라 **원래 EOS 포함 token 경로**의 $\ell_i$를 비교한다.

## Candidate selection — 추가 모델 추론 없이 하나를 고르는 기준

$\mathcal C_K^{+}$는 앞선 $K$개 중 유효한 경로들이다. 아래 selector들은 같은 $\mathcal C_K^{+}$를 사용한다. weighted medoid 등은 추가 VLM 호출 없이 기존 likelihood·interval로 계산한다. 이것이 당시 보고서의 실제 선택 기준이다.
<details id="method-greedy">
<summary>Greedy decoding — 후보 1 유지</summary>

실험 식별자: `greedy`. $\hat I_K=I_1$로 모든 예산에서 원래 greedy 응답을 유지한다. 후보를 추가 생성한 효과와 selector의 변경 효과를 비교하는 baseline이다. greedy의 step별 argmax는 sequence 전체의 maximum likelihood와 다르다.

</details>

<details id="method-maximum-likelihood">
<summary>Maximum likelihood selection — 완성 경로 확률이 가장 높은 후보</summary>

실험 식별자: `max_probability`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\ell_i.
$$

EOS를 포함한 sequence log-likelihood를 그대로 사용하며 length normalization을 하지 않는다. 같은 값이면 먼저 생성한 작은 candidate index를 선택한다. 모든 가능한 응답에 대한 전역 maximum likelihood가 아니라 이미 만든 후보 안에서의 최대값이다.

</details>

<details id="method-medoid">
<summary>Medoid selection — 후보들과의 temporal IoU 합이 가장 큰 후보</summary>

실험 식별자: `medoid`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\sum_{j\in\mathcal C_K^{+}}\mathrm{IoU}(I_i,I_j).
$$

각 후보가 다른 후보들과 얼마나 일치하는지를 동일 가중으로 더한다. $j=i$인 self-IoU도 포함한다. $1-\mathrm{IoU}$를 dissimilarity로 둔 medoid, 즉 전체 dissimilarity 합을 최소화하는 실제 후보 선택과 같다. 점수가 같으면 높은 $\ell_i$, 그다음 작은 index를 고른다. 여러 token 경로가 같은 interval로 가면 그 반복도 합에 들어가므로 서로 다른 event의 동일 가중 투표가 아니다.

</details>

<details id="method-weighted-medoid">
<summary>Weighted medoid selection — 경로 확률을 가중치로 사용</summary>

실험 식별자: `probability_weighted_medoid`.

$$
w_j=\exp(\ell_j-\ell_{\max}),\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\sum_{j\in\mathcal C_K^{+}}w_j\mathrm{IoU}(I_i,I_j).
$$

확률이 큰 경로와 일치하는 후보를 더 선호한다. $w_j$를 합이 1이 되도록 나누어도 argmax는 같다. 이는 유한 후보 안의 likelihood 가중치이며 token path likelihood를 정답 확률로 calibration한 것은 아니다. self term과 tie-break는 medoid와 같다. 당시 결과의 주된 likelihood+interval agreement 선택은 이 기준이며, maximum likelihood와 별도 열로 보고했다.

</details>

<details id="method-oracle">
<summary>Oracle — GT로 후보 안의 상한 계산</summary>

실험 식별자: `oracle`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\mathrm{IoU}(I_i,G).
$$

동률은 작은 index를 선택한다. 후보 생성 후 평가에서만 GT를 사용한다. oracle과 실제 selector의 차이는 후보 안에 이미 있는 좋은 interval을 놓친 정도이며, 후보 밖의 더 좋은 사건을 찾아낼 가능성까지 측정하지는 않는다. 예산별 모든 oracle 결과는 기존 성능 표에 그대로 유지했다.

</details>

<!-- rollout-methods:end -->
