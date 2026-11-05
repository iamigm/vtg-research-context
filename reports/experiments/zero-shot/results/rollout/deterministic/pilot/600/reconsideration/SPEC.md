# Rollout reconsideration · pilot600 · Methods

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

## Shared pipeline — 기존 rollout 후보를 재검토하는 실험

고정 pilot300·600·900은 같은 질문 순서의 중첩 cohort로 관리한다. 각 범위의 완료 여부는 보고서 상단의 상태를 따른다. 같은 Qwen3-VL-8B-Instruct와 기존 frame·input hash를 재사용한다. 앞선 deterministic rollout에서 이미 완성한 8개 또는 32개 경로를 읽고, 그 후보를 근거로 **새 model context에서 추가 추론**한다. 원래 token tree를 다시 열어 미탐색 prefix를 확장한 실험은 아니다.

review 대화는 원래 greedy 응답 token과 EOS 뒤에 새 user turn을 붙인다. 따라서 모든 접근이 원래 greedy 답을 볼 수 있다. inspector는 같은 영상과 원래 greedy 응답은 보지만 controller의 review history 전체를 보지는 않는다. 다른 backbone의 독립 verifier를 사용하지 않았다.

후보를 표시하거나 비교할 때는 exact interval 중복을 제거하고 greedy 후보를 먼저 고른다. 이후 이미 선택한 집합 $S$와 가장 다른 후보를 순차 선택한다.

$$
i^*=\operatorname*{arg\,max}_{i\notin S}\left[1-\max_{j\in S}\mathrm{IoU}(I_i,I_j)\right].
$$

동률은 높은 원래 sequence log-likelihood, 작은 candidate index 순이다. 이는 farthest-first selection이다. ledger, 즉 prompt에 넣는 후보 목록은 최대 8개 대표를 보여준다. pool32라도 prompt에서 32개를 모두 나열한 것은 아니다. 각 후보의 실제 interval·원래 log-likelihood와, pool 안에 있는 parent/child 분기 probe의 token 확률을 함께 제공한다.

모든 접근은 마지막에 candidate ledger와 각 접근의 관찰·비교 결과를 받아 greedy timestamp regeneration을 한 번 한다. 최대 64 token이며 기존 후보에서만 고르도록 제한하지 않으므로 **최종 답이 pool 밖의 interval일 수 있다.** grammar-constrained decoding·rounding·clipping은 하지 않았다. seconds를 요청하되 `MM:SS`·`HH:MM:SS` 예외를 parsing하고, EOS 미도달·parsing 실패·$s<0$ 또는 $e\le s$는 0점이다. 실패 시 greedy fallback은 없다. duration 초과는 원래 deterministic 정책처럼 flag한다.

agent·verifier·test-time scaling은 서로 배타적인 이론 분류가 아니다. 여기서는 아래와 같이 **실제 수행 절차와 계산량 배분**으로 구분했다. direct regeneration은 추가 review를 하지 않는 control이다.

## Review procedures — 방법별 수행과 예산
<details id="method-direct-regeneration">
<summary>Direct regeneration — 후보 목록만 주고 답을 다시 생성</summary>

실험 식별자: `direct_pool8`, `direct_pool32`. 후보 목록 $L_K$를 새 user prompt에 넣은 뒤 바로 답을 생성한다.

$$
\hat y=\operatorname{greedy}_\theta(c_{\mathrm{original\ answer}}\oplus L_K),\qquad K\in\{8,32\}.
$$

별도 inspection·binary critic·pairwise comparison이 없다. 추가 model evaluation은 마지막 generation 1회다. 단순히 후보를 다시 보여주고 reasoning 기회를 한 번 준 것의 효과를 비교한다.

</details>

<details id="method-agent-review">
<summary>Agent with tool use — controller와 visual inspection 반복</summary>

실험 식별자: `agent_pool8`은 최대 3회, `agent_pool32`는 최대 7회 action-observation round다. controller·inspector 모두 같은 frozen VLM이다. $h_r$는 지금까지 review history, $a_r$는 inspection action, $o_r$는 inspector 관찰이다.

$$
a_r=\operatorname{greedy}_\theta(c_{\mathrm{controller}},L_K,h_{<r}),\qquad
o_r=\operatorname{greedy}_\theta(c_{\mathrm{inspector}},a_r),\qquad
h_r=h_{r-1}\oplus(a_r,o_r).
$$

controller는 `event(a,b)`로 두 후보가 서로 어떤 occurrence인지 비교하거나 `boundary(a)`로 한 후보의 onset·offset을 검사하거나 `done`을 고른다. 관찰이 한 번 이상 있으면 early stop할 수 있다. 잘못되거나 반복된 action은 아직 안 본 boundary, 그다음 event pair 순서로 fallback한다. 계획과 관찰은 각각 최대 48 token이다.

마지막에는 $L_K,h_r$로 timestamp를 다시 생성한다. model evaluation 상한은 $2r+1$이므로 7회/15회다. 모델이 어디를 더 볼지 고르는 점이 고정 verifier와 다르다. observation은 새 frame을 더 추출하는 도구가 아니라 같은 video input의 VLM 재검토다. 새 policy 학습은 없다. [Language Agent Tree Search](https://arxiv.org/abs/2310.04406)는 참고 방향이다.

</details>

<details id="method-verifier-review">
<summary>Pointwise verification — event와 boundary를 각각 평가</summary>

실험 식별자: `verifier_pool8`은 대표 최대 3개, `verifier_pool32`는 최대 7개를 검증한다. 각 후보에 대해 event가 맞는지와 boundary가 맞는지를 두 별도 Yes/No prompt로 묻는다.

$$
S_i=b_{c_i^{\mathrm{event}}}(\mathrm{Yes})\ b_{c_i^{\mathrm{boundary}}}(\mathrm{Yes}),\qquad
i^*=\operatorname*{arg\,max}_i S_i.
$$

점수가 같으면 작은 candidate index를 고른다. 질문을 따로 했다고 두 판단이 통계적으로 독립인 것은 아니며 위 곱은 heuristic ranking이다. raw Yes/No probability 합도 저장했다. 모든 점수와 preferred candidate를 final regeneration prompt에 전달하므로 최종 답이 반드시 $I_{i^*}$인 것은 아니다.

추가 model evaluations는 후보당 2회+마지막 generation 1회, 최대 7회/15회다. 같은 VLM의 자기 평가를 사용했으며 학습한 verifier·GT likelihood·external ground truth를 사용하지 않았다.

</details>

<details id="method-tts-review">
<summary>Test-time scaling — 모델 비교에 쓰는 연산을 4개에서 8개로 확대</summary>

실험 식별자: `tts_pool8_select4`, `tts_pool8_select8`, `tts_pool32_select4`, `tts_pool32_select8`. $K\in\{8,32\}$는 원래 rollout pool, $m\in\{4,8\}$은 그 안에서 모델로 비교할 farthest-first 대표 수다. 이 2×2 구성이 proposal compute와 selection compute의 효과를 나누어 본다.

후보 쌍 $i,j$에 대해 A/B 중 어느 interval이 더 정확한지를 두 presentation order로 평가한다. $b_{i,j}(A)$는 A=$i$, B=$j$일 때 두 선택지 안에서 정규화한 A의 확률이다.

$$
\pi_{i,j}=\tfrac12\{b_{i,j}(A)+b_{j,i}(B)\},\qquad
i\text{ wins iff }\pi_{i,j}\ge0.5.
$$

knockout tournament를 반복하고 홀수 후보는 다음 round로 바로 올린다. 비교 기록과 winner를 final regeneration에 전달한다. 후보가 충분하면 pairwise model evaluations는 $2(m-1)$회이고 마지막 generation까지 $2m-1$회, 즉 7회/15회다. 마지막 답이 tournament winner와 같도록 강제하지 않는다.

이 TTS는 단순 maximum likelihood 후처리가 아니다. **동일한 rollout pool 안에서도 4→8개로 모델 비교 연산을 늘리는 축**을 갖는다. 모든 pair를 비교한 ranking·여러 무작위 answer의 self-consistency·학습된 reward model의 best-of-N은 아니다. [Test-time compute scaling 연구](https://arxiv.org/abs/2408.03314)는 관련 맥락이며 해당 논문의 학습·추론법 전체를 재현하지 않았다.

</details>

<details id="method-probability-replay">
<summary>Counterfactual probability evaluation — 분기 판단이 바뀌었는지 사후 측정</summary>

최종 review context를 만든 후, 원래 token prefix를 같은 위치까지 teacher forcing하여 바뀐 token distribution을 측정했다. 이 분석은 **위 네 접근의 선택 과정에 다시 넣지 않는 별도 diagnostic**이다. 원래 pool32에서 GT 없이 종류별 최대 1개 probe를 고정했다.

- event probe: parent/alternative의 IoU<0.1이고 interval 중심 차이가 $\max(2,0.05D)$ seconds보다 큰 분기.
- boundary probe: IoU≥0.5, endpoint 최대 이동량이 양수이고 $\max(2,0.1\times\text{parent duration})$ seconds 이하인 분기.

각 종류에서 alternative token의 원래 raw probability가 가장 큰 것을 고른다. low pool에 없는 probe는 review prompt에는 보이지 않고 사후 평가에만 사용했다. 고정 prefix에서 원래 token $a$와 대체 token $b$의 log-odds 변화는 다음과 같다.

$$
\Delta\log\mathrm{odds}=
\log\frac{p_\theta(b\mid c_{\mathrm{review}},y_{<j})}{p_\theta(a\mid c_{\mathrm{review}},y_{<j})}
-\log\frac{p_\theta(b\mid c_{\mathrm{original}},y_{<j})}{p_\theta(a\mid c_{\mathrm{original}},y_{<j})}.
$$

양수면 review 이후 대체 token을 상대적으로 더 선호한다. full-vocabulary Jensen–Shannon divergence는 $M=(P+Q)/2$에서 $\mathrm{JS}(P,Q)=\tfrac12\mathrm{KL}(P\Vert M)+\tfrac12\mathrm{KL}(Q\Vert M)$다. 두 token을 각각 선택한 뒤의 child distribution과 원래 두 완성 경로의 EOS 포함 log-likelihood도 같은 review context에서 다시 측정했다.

확률이 바뀌었다는 사실과 실제 새로운 tree path를 탐색했다는 사실은 다르다. 이 실험은 전자는 측정하지만 그 변화로 원래 token tree를 다시 확장하지 않았다. temporal cluster 간 이동은 occurrence의 proxy이며 semantic identity의 GT가 아니다. clustering은 모든 기존 member와 IoU≥0.5인 group에 순차 배정하는 complete-linkage criterion을 사용하고 threshold 0.3/0.5/0.7 민감도를 함께 보고했다.

</details>

<!-- rollout-methods:end -->
