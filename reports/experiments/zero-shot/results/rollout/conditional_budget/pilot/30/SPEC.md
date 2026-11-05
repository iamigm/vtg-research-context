# Conditional rollout budget · pilot30 · Methods

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

## Shared pipeline — start에 조건화된 end 탐색 비용 진단

이 실험은 고정 pilot300에서 데이터셋별 앞선 10개, 총 30 query를 사용하는 **예산 진단**이다. 모델은 Qwen3-VL-8B-Instruct, 2 FPS, spatial target 100,352 pixels다. start와 end를 독립적인 두 분포로 다루지 않고, start·delimiter를 포함한 정확한 prefix마다 end distribution을 따로 계산했다.

greedy로 생성한 답에서 첫 숫자 직전까지를 고정 anchor로 둔다. 이 절의 모든 probability는 그 anchor가 이미 주어졌다는 조건부 확률이다. anchor 자체를 생성할 확률은 곱하지 않는다.

탐색 grammar는 `number - number seconds`다. number는 음수가 아닌 정수와 선택적인 소수 첫째 자리 1개이며, 숫자 자리 수 상한은 $\max(4,\operatorname{digits}(\lceil D\rceil))$다. `-` 전후 whitespace는 각 최대 2개, suffix는 최대 32 token이다. 이 diagnostic grammar는 seconds만 허용한다. 후속 일반 응답 실험의 `MM:SS` parsing 지원을 이 실행에 소급하지 않는다.

각 prefix에서 원래 full-vocabulary softmax를 계산하고 grammar에 맞는 token을 child로 만든다. **grammar 안의 token들끼리 정규화하지 않는다.** grammar를 벗어난 probability는 별도 합계로 남긴다. terminal은 grammar의 `seconds`까지 완성된 prefix이며 EOS를 요구하지 않는다. 따라서 여기서 terminal probability는 EOS 포함 완성 assistant answer probability와 다르다.

`-`가 처음 포함된 token까지의 정확한 token ID prefix를 start key로 저장한다. token 하나가 delimiter와 end의 일부까지 포함하면 그것도 별도 start key다. 같은 숫자 start라도 다른 whitespace·tokenization이면 end 조건이 달라질 수 있으므로 합치지 않는다. 숫자값으로 합치는 것은 사후 표시뿐이다.

$\pi(u)$는 anchor 이후 prefix $u$의 확률 곱, $\mathcal F$는 아직 확장하지 않은 prefix, $\mathcal L$은 완성 terminal 집합이다. 매 node expansion은 다음-token distribution 하나를 조회하고 그 child들을 만든다. 예산은 $8,16,32,64,128,256,512$회 expansion이며 두 정책에 같은 checkpoint를 사용한다. decoder input token·forward call·시간은 별도로 측정한다.

## Expansion policies — 어떤 prefix를 다음으로 열 것인가
<details id="method-joint-best-first">
<summary>Best-first search — prefix probability 순으로 확장</summary>

실험 식별자: `joint`.

$$
u^*=\operatorname*{arg\,max}_{u\in\mathcal F}\pi(u).
$$

확률이 큰 prefix부터 확장한다. start에 속한 end를 이미 많이 보았는지는 고려하지 않는다. joint start/end 경로의 probability가 높은 부분을 빠르게 보는 장점이 있지만, 높은 확률의 한 start 아래로 계산이 몰릴 수 있다. grammar와 길이 제한 밖으로는 탐색하지 않는다.

</details>

<details id="method-balanced-priority">
<summary>Visit-count penalty — 같은 start 아래 반복 확장에 penalty</summary>

실험 식별자: `balanced`. $k(u)$는 prefix의 exact start key, $n_k$는 그 key 아래 이미 수행한 expansion 수다.

$$
u^*=\operatorname*{arg\,max}_{u\in\mathcal F}
\begin{cases}\pi(u)&k(u)\text{ not yet defined},\\
\pi(u)/(1+n_{k(u)})&\text{otherwise}.\end{cases}
$$

start가 아직 완성되지 않았으면 원래 prefix probability를 쓴다. start가 정해진 뒤에는 같은 조건 아래 end를 많이 확장할수록 우선순위를 낮춘다. 모든 start를 균등하게 탐색한다는 보장은 없고, 낮은 probability의 start는 여전히 보지 못할 수 있다. 이는 이번 실행의 우선순위 heuristic이며 별도의 표준 알고리즘 이름을 붙이지 않는다.

</details>

<details id="method-probability-accounting">
<summary>Probability accounting — 탐색하지 못한 부분까지 구분하는 방법</summary>

중복되지 않는 prefix frontier를 유지하므로 다음 probability 보존 관계를 검사했다. $R_{\mathrm{grammar}}$는 grammar 밖으로 간 경로, $R_{\mathrm{length}}$는 32 token 제한에 걸린 경로의 확률 합이다.

$$
\sum_{l\in\mathcal L}\pi(l)+\sum_{u\in\mathcal F}\pi(u)+R_{\mathrm{grammar}}+R_{\mathrm{length}}=1.
$$

코드 필드 `start_mass`는 지금까지 delimiter에 도달한 exact start key들의 prefix probability 합 $S$, `joint_mass`는 start/end 모두 grammar 끝까지 완성한 terminal probability 합 $J$다. 이 숫자를 정답 비율로 해석하지 않는다.

$$
S=\sum_{k\in\mathcal K_{\mathrm{found}}}\pi(k),\qquad
J=\sum_{l\in\mathcal L}\pi(l),\qquad
1-J=(1-S)+(S-J).
$$

$1-S$는 아직 delimiter까지 찾지 못했거나 grammar 밖으로 간 쪽, $S-J$는 delimiter까지는 도달했지만 그 뒤 end 완성이 아직 안 됐거나 제한으로 버린 쪽의 probability다. 예를 들어 $S$만 높고 $J$가 낮으면 start 값을 몇 개 찾았다는 사실만으로 end까지 충분히 검토했다고 말할 수 없다. 서로 다른 start의 종료 분포를 계산하는 추가 비용이 여기서 드러난다.

30 query의 terminal 성능은 $0\le s<e\le D$인 interval만 유효로 평가했다. 누적 probability 0.9에 도달할 후보 수 $K_{90}$은 발견한 terminal만으로는 계산이 안 될 수 있으며, 부족하면 미해결로 남긴다. 현재 찾은 후보 수를 전체 분포의 최소 필요 예산으로 확정하지 않는다.

두 exact start key의 end distribution 차이는 total variation distance로 분석했다. 현재 관측한 확률들의 L1 차이가 $d$, 두 분포에서 아직 확인하지 못한 확률 합이 $R$이면 다음 범위만 보고한다.

$$
\max\{0,(d-R)/2\}\le\mathrm{TV}\le\min\{1,(d+R)/2\}.
$$

end의 미탐색 확률을 0으로 채워 서로 다른 분포라고 단정하지 않는다. 구현 근거는 `_lib/conditional_rollout_budget.py`의 `grammar`, `tree`, `Evaluator`이며, conditional distribution 진단은 `_lib/conditional_rollout_analysis.py`다.

</details>

<!-- rollout-methods:end -->
