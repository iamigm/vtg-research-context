# TimeLens stochastic rollout: top-k vs top-p

동일 pilot 300개 × 두 모델 × 두 sampling 방식 × 5회 = 6,000개 응답, 107,105개 생성 단계. 기존 zero-shot과 같은 입력 hash를 확인했다.

**두 실험은 top-k만 적용(p=1) / top-p만 적용(k=0)하는 분리 실험이다.** Qwen3 k=20/p=0.8, Qwen2.5 라이브러리 기본 k=50/p=1.0을 사용한다. 모델 카드에서 k와 p를 함께 쓰는 설정 자체의 재현은 아니다. 매 단계에는 truncation 후 유한 score를 가진 후보 전체를 저장했다. 고정 top-30 기록은 하지 않는다.

Qwen3 [모델 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)는 일반 태스크에 temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0, presence_penalty=1.5를 제시한다. Presence penalty는 이미 생성된 토큰에 한 번씩 차감한다. 최대 길이는 zero-shot과 비교하도록 512토큰을 유지했다.

사용자 지시에 따라 Qwen2.5는 Transformers 라이브러리 sampling 기본값 temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0을 명시적으로 적용한다. 모델 배포 파일의 temperature=0.000001/repetition_penalty=1.05와 다르다. Top-p only에서는 top_k=0으로 꺼 p=1.0 전체 어휘 sampling을 수행한다. Qwen2.5 모델 카드가 권장하는 수치라고 주장하지 않는다.

설정 원문과 retrieval 시각/해시는 `../../../../../artifacts/runs/rollout/stochastic/pilot/300/sources/`에 보존했다. 실제 설정은 [../../../../../artifacts/runs/rollout/stochastic/pilot/300/metadata/config.json](../../../../../artifacts/runs/rollout/stochastic/pilot/300/metadata/config.json)에 기록했다. 각 sample/rollout의 seed는 sample ID·rollout 번호·42에서 SHA-256으로 결정하며 두 방식과 모델 사이에 동일하게 대응시켰다. 동일 seed는 서로 다른 분포에서 동일 응답을 보장하지 않는다.

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

## Shared pipeline — 실제 sampling을 사용했던 초기 실험

이 계열은 pilot300 query×모델 2개×sampler 2개×5회, 총 6,000개 응답을 실제 stochastic decoding한 초기 비교다. 후속 deterministic rollout과 다르게 같은 경로가 여러 번 나올 수 있고, 중복을 막거나 without-replacement sampling을 하지 않았다. baseline과 input hash를 대조했으며 이 입력 설정을 후속 fixed spatial resolution 실험과 자동으로 동일시하지 않는다.

먼저 raw LM logit에 presence penalty를 적용하고 temperature로 나눈다. $A_t$는 현재 응답에서 이미 나온 token ID 집합이다. prompt token은 이 presence 집합에 넣지 않는다.

$$
\widetilde z_t(a)=\frac{z_t(a)-\lambda\mathbf1[a\in A_t]}{T},\qquad
\widetilde p_t(a)=\operatorname{softmax}(\widetilde z_t)_a.
$$

repetition penalty는 양 모델 모두 1.0이라 변화가 없다. 그다음 아래 top-k 또는 top-p 하나만 적용하여 남긴 집합 $S_t$ 안에서 재정규화하고 categorical sampling한다.

$$
q_t(a)=\frac{\widetilde p_t(a)\mathbf1[a\in S_t]}{\sum_{b\in S_t}\widetilde p_t(b)},\qquad
y_t\sim\operatorname{Categorical}(q_t),\qquad
P_{\mathrm{sampling}}(y)=\prod_tq_t(y_t).
$$

HTML의 raw probability는 원래 LM-head softmax $p_\theta$, prefilter probability는 penalty·temperature 이후의 $\widetilde p$, sampling probability는 truncation 후의 $q$다. sampling path probability를 표시할 때는 $q$를 곱해야 하며 원래 모델 경로 확률 $\prod_tp_\theta$와 구분한다. 유한 logit을 남긴 후보 수와 FP32 softmax에서 실제 양수인 후보 수도 구분했다.

Qwen3-VL-8B-Instruct는 $T=0.7,\lambda=1.5$, Qwen2.5-VL-7B-Instruct는 $T=1,\lambda=0$이다. 최대 generation 길이는 512 token이다. seed는 `SHA256(sample_key:rollout_index:42)`의 처음 8개 hexadecimal 문자를 정수로 바꾼 값으로, 모델·sampler 사이에 대응시켰다. 같은 seed가 다른 분포에서 같은 응답을 보장하지는 않는다.

원래 evaluation은 TimeLens `extract_time`의 첫 interval을 **정수 seconds로 round한 뒤** IoU를 계산했다. parsing 실패 시 영상 길이 밖의 fallback interval을 기록해 0점 처리했다. 이 rounding은 후속 deterministic/reconsideration의 rounding 없는 evaluation과 다르며, 문서화 과정에서 소급해 바꾸지 않았다. rollout 성능은 query마다 5개 응답을 동일 가중 평균한 뒤 query 평균을 낸다. any-of-5 oracle은 GT로 5개 중 최선을 고른 상한이다.

## Sampling methods — truncation 방법만 분리한 비교
<details id="method-top-k-sampling">
<summary>Top-k sampling — 확률 상위 k개 안에서 추출</summary>

실험 식별자: `top_k`. $\widetilde p$의 상위 $k$개를 남기고 그 안에서 sampling한다. Qwen3는 $k=20$, Qwen2.5는 $k=50$이다. top-p는 1로 두어 별도 truncation을 하지 않았다.

$$
S_t=\operatorname{TopK}_a\widetilde p_t(a),\qquad y_t\sim\operatorname{Categorical}(q_t).
$$

실제 Hugging Face 구현은 k번째 logit과 동률인 token을 함께 남길 수 있어 후보 수가 k보다 많을 수 있다. 이 token도 모두 기록했다. 독립 rollout마다 같은 높은 확률 token을 반복 추출할 수 있다. 후보를 확률 순서대로 중복 없이 검사하는 deterministic 분기와 다르다.

</details>

<details id="method-top-p-sampling">
<summary>Nucleus sampling (top-p) — cumulative probability 기준으로 추출</summary>

실험 식별자: `top_p`. token을 $\widetilde p$ 내림차순으로 정렬하고 다음 최소 prefix를 남긴다.

$$
m=\min\left\{h:\sum_{j=1}^{h}\widetilde p_{(j)}\ge p\right\},\qquad
S_t=\{(1),\ldots,(m)\}.
$$

Qwen3는 $p=0.8$, Qwen2.5는 $p=1.0$이다. top-k를 0으로 끈 분리 실험이다. 따라서 Qwen2.5의 이 조건은 실제로 full-vocabulary sampling이며 truncation 강도가 Qwen3와 같지 않다. 모델 카드에서 top-k와 top-p를 함께 쓰는 설정 자체를 재현한 비교도 아니다. [Nucleus sampling 원 논문](https://arxiv.org/abs/1904.09751).

</details>

<!-- rollout-methods:end -->

| 모델 | 방식 | 벤치마크 | baseline mIoU (%) | rollout 평균 mIoU (%) | baseline R@.5 (%) | rollout 평균 R@.5 (%) | oracle any-of-5 R@.5 (%) | 평균 고유 응답 수 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen25 | top_k | charades-timelens | 40.65 | 32.98 | 42.00 | 30.00 | 56.00 | 4.91 |
| qwen25 | top_k | activitynet-timelens | 29.96 | 20.98 | 27.00 | 17.40 | 38.00 | 4.95 |
| qwen25 | top_k | qvhighlights-timelens | 30.83 | 23.88 | 24.00 | 16.40 | 43.00 | 5.00 |
| qwen25 | top_p | charades-timelens | 40.65 | 32.98 | 42.00 | 30.00 | 56.00 | 4.91 |
| qwen25 | top_p | activitynet-timelens | 29.96 | 20.98 | 27.00 | 17.40 | 38.00 | 4.95 |
| qwen25 | top_p | qvhighlights-timelens | 30.83 | 23.88 | 24.00 | 16.40 | 43.00 | 5.00 |
| qwen3 | top_k | charades-timelens | 47.53 | 43.63 | 49.00 | 47.80 | 63.00 | 4.74 |
| qwen3 | top_k | activitynet-timelens | 40.77 | 38.72 | 43.00 | 40.40 | 57.00 | 4.75 |
| qwen3 | top_k | qvhighlights-timelens | 57.13 | 53.24 | 62.00 | 57.40 | 73.00 | 4.77 |
| qwen3 | top_p | charades-timelens | 47.53 | 44.49 | 49.00 | 49.00 | 59.00 | 4.62 |
| qwen3 | top_p | activitynet-timelens | 40.77 | 39.42 | 43.00 | 40.40 | 51.00 | 4.52 |
| qwen3 | top_p | qvhighlights-timelens | 57.13 | 54.92 | 62.00 | 59.60 | 71.00 | 4.59 |

평균 지표는 query마다 rollout 5개를 동일 가중 평균한 후 100개 query를 집계한 값이다. 500개 rollout을 독립 query로 해석하지 않는다. oracle any-of-5는 GT를 보고 가장 좋은 답을 고르는 상한이며 실제 배포 성능이 아니다.

| 모델 | 방식 | 벤치마크 | 평균 후보 수 | 양의 확률 후보가 1개인 step 비율 (%) |
|---|---|---|---:|---:|
| qwen25 | top_k | charades-timelens | 51.20 | 0.00 |
| qwen25 | top_k | activitynet-timelens | 51.04 | 0.00 |
| qwen25 | top_k | qvhighlights-timelens | 51.02 | 0.00 |
| qwen25 | top_p | charades-timelens | 152064.00 | 0.00 |
| qwen25 | top_p | activitynet-timelens | 152064.00 | 0.00 |
| qwen25 | top_p | qvhighlights-timelens | 152064.00 | 0.00 |
| qwen3 | top_k | charades-timelens | 20.20 | 0.00 |
| qwen3 | top_k | activitynet-timelens | 20.20 | 0.00 |
| qwen3 | top_k | qvhighlights-timelens | 20.18 | 0.00 |
| qwen3 | top_p | charades-timelens | 1.61 | 81.13 |
| qwen3 | top_p | activitynet-timelens | 1.62 | 79.19 |
| qwen3 | top_p | qvhighlights-timelens | 1.53 | 80.90 |

후보 수 통계는 응답 안에서 step 평균을 내고 응답 간 평균했다. top-k 경계의 logit 동률은 Hugging Face 구현에 따라 k개보다 많이 남을 수 있다. HTML은 해당 후보도 모두 표시한다. top-p는 temperature·penalty 적용 후 분포에 적용하며 다른 top-k 제한을 사용하지 않는다.

sampling 확률은 필터 후 분포를 재정규화한 실제 추출 확률이다. 필터 전 확률과 raw LM-head 확률을 함께 표시해 구분한다. 유한 score라도 FP32 softmax underflow로 확률 0이 될 수 있어 유한 후보 수와 양의 확률 후보 수를 따로 기록했다. 토큰 확률을 구간 정답 확률로 해석하면 안 된다.

- qwen25 top_k: 세 벤치마크 평균 mIoU 차이 -7.87 pp, 평균 고유 응답 4.95/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen25 top_p: 세 벤치마크 평균 mIoU 차이 -7.87 pp, 평균 고유 응답 4.95/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen3 top_k: 세 벤치마크 평균 mIoU 차이 -3.28 pp, 평균 고유 응답 4.75/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.
- qwen3 top_p: 세 벤치마크 평균 mIoU 차이 -2.20 pp, 평균 고유 응답 4.58/5. Sampling 설정 변경의 관측 결과이며 후보 집합 크기만의 인과 효과는 아니다.

기존 decoder와 프레임 전처리를 유지했다. 앞선 MMCO 진단에서 sparse Decord와 순차 Decord 사이에 9개 영상의 표본 픽셀 차이가 있었으므로 입력 디코딩 품질에 관한 기존 한계도 유지된다.

[영상·생성 단계별 전체 후보 집합 HTML](report.html) · [정량 JSON](../../../../../artifacts/runs/rollout/stochastic/pilot/300/metrics/summary.json) · [진행 로그](../../../../../artifacts/runs/rollout/stochastic/pilot/300/logs/progress.log)
