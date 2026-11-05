"""Executable-report documentation, derived from the frozen rollout implementations.

This module only formats Markdown. It never loads a model or recomputes scores.
Report generators call formalize() so the rendered report and its source agree.
"""
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- rollout-methods:begin -->'
END = '<!-- rollout-methods:end -->'


def toggle(identifier, title, body):
    return f'<details id="method-{identifier}">\n<summary>{title}</summary>\n\n{body.strip()}\n\n</details>\n'


COMMON = r'''
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
'''


EVENT_EXPLANATION = r'''
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
'''


EVENT_FORMULATION = r'''
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
'''


EVENT_COMMON = EVENT_EXPLANATION + COMMON + EVENT_FORMULATION + '\n## Search policies — 다음에 무엇을 검토할지 고르는 방법\n'


def event_methods():
    parts = [EVENT_COMMON]
    parts.append(toggle('likelihood-best-first', 'Best-first search — sequence log-likelihood 우선', r'''
실험 식별자: `likelihood_best_first`. 공통 frontier $\mathcal F$ 전체에서 다음 node를 고른다. $k_i$는 해당 node에서 완료한 focus 수다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(\ell_i-0.3k_i).
$$

동률은 depth가 작은 node, 그다음 작은 ID 순서다. 높은 canonical answer likelihood를 우선하되 같은 node에 계속 계산이 쏠리는 것을 0.3 penalty로 완화한다. 한 번 고르면 공통 expansion 한 번을 수행하고 frontier에서 다시 고른다. uniform-cost search나 A*의 admissible heuristic을 구현한 것은 아니며 최적 답을 보장하지 않는다. $\ell_i$는 현재 review history가 아닌 원래 video-query context의 값이다.
'''))
    parts.append(toggle('value-best-first', 'Greedy best-first search — value function 우선', r'''
실험 식별자: `value_best_first`. 다음 평가식을 frontier 전체에 적용한다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(V_i-0.05k_i).
$$

critic value가 큰 node를 우선하고 이미 수행한 focus마다 0.05를 뺀다. 새로운 관찰로 tree가 갱신되면 다시 선택한다. $V_i$에는 future return의 학습 추정이나 정답 IoU가 들어 있지 않다. penalty와 value 모두 이 실험의 heuristic이며 논문에서 보장된 bound가 아니다.
'''))
    parts.append(toggle('value-beam', 'Beam search — value function으로 width 4 유지', r'''
실험 식별자: `value_beam`. 한 번에 유지할 beam $\mathcal B$의 최대 크기는 4다. 새 beam을 만들 때 layer에서 $V_i$가 높은 4개를 고른다.

$$
\mathcal B\leftarrow\operatorname{Top4}_{i\in\mathcal L}V_i,\qquad
i^*=\operatorname*{arg\,min}_{i\in\mathcal B\cap\mathcal F}(k_i,-V_i,i).
$$

현재 beam에 확장 가능한 node가 남으면 그 beam을 먼저 사용한다. 모두 소진하면 직전 beam의 확장 가능한 child들을 다음 layer로 사용하고, 그것도 없으면 전체 frontier에서 가장 얕은 depth를 사용한다. 같은 beam 안에서는 focus를 적게 수행한 node부터 확장한다. 탈락한 frontier를 영구 삭제하지 않으므로, 매 token step에서 width 밖을 버리는 표준 sequence beam search와 다르다. 여기서 layer는 action tree의 interval 상태이며 generation step이 아니다.
'''))
    parts.append(toggle('diverse-beam', 'Diversity-promoting beam search — MMR로 중복 interval 억제', r'''
실험 식별자: `diverse_beam`. `value_beam`과 layer 이동·width 4·focus 순서는 같다. 차이는 beam의 4개 node를 MMR로 고르는 것이다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal L\setminus S}
\left[V_i-0.25\max_{j\in S}\mathrm{IoU}(I_i,I_j)\right],\qquad S\leftarrow S\cup\{i^*\}.
$$

value가 조금 낮아도 기존 선택과 다른 시간대인 node가 들어갈 수 있다. 원래 [Diverse Beam Search](https://arxiv.org/abs/1610.02424)의 grouped sequence decoding을 재현하지 않았다. 이번 구현은 interval IoU를 사용하는 MMR beam selection이며, diversity penalty 0.25가 최종 selector의 0.1과 다르다.
'''))
    widening = r'''
두 방법 모두 root의 child를 MMR($\alpha=0.25$) 순서로 놓고 아래 개수만 고려한다. $r$은 가상 root, $R$은 확장 가능한 descendant가 있는 root child 수다.

$$
L_r=\min\{R,\ 2+\lfloor2\sqrt{N_r+1}\rfloor\}.
$$

선택된 child $i$가 확장 가능하고, 아직 child가 없거나 $k_i<\min\{3,1+\lfloor\sqrt{N_i}\rfloor\}$이면 그 node에 새 focus를 연다. 그렇지 않으면 기존 child로 내려간다. 이것이 **progressive widening**, 즉 방문이 늘 때 고려하는 action 수를 점차 늘리는 절차다. root에서 열지 않은 사건이 자동으로 사라지는 것은 아니다. 같은 node의 동일 focus를 반복 simulation하지 않는다.
'''
    parts.append(toggle('uct', 'UCT with progressive widening — 평균 value와 방문 불확실성으로 탐색', r'''
실험 식별자: `uct`. UCT는 **Upper Confidence bounds applied to Trees**다. parent $p$에서 고려할 수 있는 child $i$ 중 다음 식이 큰 쪽으로 내려간다.

$$
U_i=Q_i+1.4\sqrt{\frac{\log(N_p+2)}{N_i+1}}.
$$

첫 항은 관측한 descendant value의 평균이고 두 번째 항은 방문이 적은 child를 다시 볼 기회다. 미방문 node는 $Q_i=V_i$로 시작한다. 원래 UCT의 무한 우선순위 초기 방문 대신 분모에 1을 더한 유한 bonus를 쓴다. 동률은 작은 ID 순이다. [UCT 원 논문](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf).
''' + widening + r'''
이 구현은 deterministic greedy expansion과 같은 VLM의 heuristic critic을 사용한다. 독립 reward sample이나 정확한 simulator를 가정한 UCT의 수렴 속도를 이번 IoU 성능에 대한 보장으로 사용하지 않는다.
'''))
    parts.append(toggle('puct', 'PUCT with progressive widening — event probability를 prior로 사용', r'''
실험 식별자: `puct`. tree를 내려가는 방식은 UCT와 같고 bonus에 child prior를 넣는다. $\mathcal H_p$는 현재 고려하는 parent의 child 집합이다.

$$
P_i=0.9\frac{\max(E_i,10^{-6})}{\sum_{j\in\mathcal H_p}\max(E_j,10^{-6})}
+\frac{0.1}{\lvert\mathcal H_p\rvert},\qquad
U_i=Q_i+1.4P_i\frac{\sqrt{N_p+1}}{N_i+1}.
$$

90%는 event presence critic, 10%는 uniform prior로 둔다. 이는 canonical answer likelihood나 학습한 policy network의 prior가 아니다. [AlphaZero의 prior-guided tree search](https://arxiv.org/abs/1712.01815)와 같은 PUCT 형식을 사용하지만 self-play 학습·Dirichlet noise·학습된 별도 value model은 없다.
''' + widening))
    parts.append(toggle('neural-controller', 'Model-based action selection — 모델이 다음 inspection 결정', r'''
실험 식별자: `neural_controller`. 동일한 frozen VLM을 controller로도 사용한다. $h$는 최근 action 기록과 frontier 요약, $a$는 refine/propose·node ID·focus를 담은 JSON action이다.

$$
a=\operatorname{parse}_{\mathrm{JSON}}\bigl(\operatorname{greedy}_\theta(c,\operatorname{prompt}(h))\bigr).
$$

MMR($\alpha=0.1$)로 추린 frontier 최대 8개와 최근 4개 action을 보여주고 최대 48 token으로 action을 생성한다. `propose`면 새 occurrence를 요청하고, `refine`이면 모델이 고른 유효한 node를 확장한다. node가 유효하지 않으면 value best-first 선택으로 fallback한다. focus가 잘못됐거나 이미 수행됐으면 남은 첫 focus를 사용한다. 새 policy를 학습한 것이 아니며 controller가 critic과 독립인 모델도 아니다. action 계획에 사용한 model 연산도 search 예산에 넣었다. [Language Agent Tree Search](https://arxiv.org/abs/2310.04406)는 참고한 방향이며 해당 논문의 전체 agent 시스템 재현은 아니다.
'''))
    parts.append(toggle('self-refine', 'Iterative refinement with feedback — 현재 occurrence를 이어서 수정', r'''
실험 식별자: `self_refine`. 처음에는 value가 가장 높은 node를 고른다. 이후 현재 event lineage의 frontier가 있으면 다음 tuple이 큰 node를 선택한다.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal F:\,\operatorname{event}(i)=\operatorname{event}(\mathrm{current})}
(\operatorname{depth}_i,V_i,-k_i,-i).
$$

더 깊은 refinement를 먼저 이어가고, 같은 event에 확장 가능한 node가 없으면 전체 frontier에서 최대 value로 옮긴다. 공통 주기에 따라 새 proposal도 수행하고, regeneration이 다른 occurrence로 이동하면 current도 따라 바뀐다. 따라서 엄격히 하나의 answer만 반복 수정하는 방법은 아니다. [Self-Refine](https://arxiv.org/abs/2303.17651)의 feedback→refinement 구조를 적용했지만 여기서는 별도 critic·여러 event·공통 최종 selector를 결합했다.
'''))
    parts.append(toggle('submodular-coverage', 'Monotone submodular maximization — facility location objective', r'''
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
'''))
    parts.append(toggle('reward-balanced', 'Highest averages method — softmax value weight로 방문 배분', r'''
실험 식별자: `reward_balanced`. 이 이름은 저장된 실험 식별자다. 실제 root 배분식은 fixed weight일 때 **Highest averages method의 D’Hondt rule**과 같은 $w/(n+1)$ 형태다. 이번 weight는 vote가 아니라 critic value의 exponential이다.

$$
w_r=\exp(V_r/0.25),\qquad
r^*=\operatorname*{arg\,max}_{r}\frac{w_r}{N_r+1},\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F:\,\operatorname{event}(i)=r^*}(V_i-0.05k_i).
$$

value가 높은 event를 우선하지만 방문할수록 분모가 커져 다른 event에 기회를 준다. weight가 고정되고 모든 root가 계속 열려 있는 단순 배분 문제와 달리, 이 tree에서는 새 event가 추가되고 root가 blocked·exhausted될 수 있다. 따라서 고정 배분의 최적성이나 best-arm identification 보장을 주장하지 않는다. REBASE의 재현도 아니다. [D’Hondt rule 공식 설명](https://www.europarl.europa.eu/thinktank/en/academic/EPRS_BRI%282024%29762352).
'''))
    parts.append(toggle('sequential-halving', 'Successive Halving — event 후보를 단계적으로 절반씩 축소', r'''
실험 식별자: `sequential_halving`. **Successive Halving / Sequential Halving** 계열의 제거 방식을 event root에 적용한 변형이다. 처음 active set $\mathcal A$는 MMR($\alpha=0.15$)로 최대 8개 root를 고른다. 현재 최소 방문 수에 1을 더한 값을 첫 target $t$로 둔다.

$$
M_r=\max_{i\in\operatorname{descendants}(r)}V_i,\qquad
\mathcal A\leftarrow\operatorname{Top}_{\lceil\lvert\mathcal A\rvert/2\rceil}M_r,\qquad
t\leftarrow2t+1.
$$

active root 모두 $N_r\ge t$일 때 $M_r$가 높은 절반만 남긴다. 다음 확장은 방문 수가 가장 작은 active root, 그 안의 $V_i-0.05k_i$ 최대 node다. active root가 모두 사용 불가가 되면 현재 가능한 root에서 set을 다시 만든다. 제거한 event가 최종 후보 집합에서 자동 삭제되는 것은 아니다.

원래 fixed-budget Sequential Halving처럼 총 budget을 round별로 균등 분할하지 않았고, arm reward의 표본 평균 대신 **가장 높은 descendant critic value**를 사용했다. 정확한 논문 알고리즘의 재현이나 regret 보장은 아니다. [Successive Halving 원 연구](https://proceedings.mlr.press/v51/jamieson16.html).
'''))
    parts.append(toggle('nucleus-frontier', 'Nucleus truncation — 후보 분포의 cumulative probability 0.95로 제한', r'''
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
'''))
    parts.append(toggle('typicality-frontier', 'Locally typical truncation — surprisal과 entropy 차이로 제한', r'''
실험 식별자: `typicality_frontier`. 후보 집합의 $q_r$는 nucleus와 같다. **surprisal**은 $-\log q_r$, **entropy**는 그 기대값 $H(q)$다. 확률이 큰 순서 대신 두 값의 차이가 작은 순서로 후보를 정렬한다.

$$
H(q)=-\sum_rq_r\log q_r,\qquad d_r=\lvert-\log q_r-H(q)\rvert.
$$

$d_r$ 오름차순에서 cumulative probability가 0.95 이상이 되는 최소 집합을 고른 뒤, 그 안의 root를 $q_r/(N_r+1)$로 선택한다. 이후 node 선택은 $V_i-0.05k_i$다. [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)의 entropy 기반 truncation을 유한 event 목록에 적용했으며 이 구현에는 sampling이 없다. 분포가 충분히 집중되면 nucleus와 같은 집합이 될 수 있다. 따라서 이름이 다르다는 것만으로 다른 경로를 탐색했다고 판단하지 않는다.
'''))
    parts.append(toggle('commit-then-refine', 'Ablation — 한 occurrence에 고정한 iterative refinement', r'''
실험 식별자: `commit_then_refine`. 공통 critic을 통과한 후보 중 처음 value가 가장 높은 event $r_0$를 고정한다.

$$
\mathcal F_{r_0}=\{i\in\mathcal F:\operatorname{event}(i)=r_0\},\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F_{r_0}}(\operatorname{depth}_i,V_i,-k_i,-i).
$$

새 occurrence proposal을 하지 않고, refinement가 부모와 IoU<0.2인 다른 occurrence로 이동하면 그 child를 사용 불가로 만든다. 최종 selector도 $r_0$ lineage 안으로 제한한다. 해당 event가 blocked되거나 더 확장할 수 없으면 다른 event로 옮기지 않고 중단한다. 사건을 다시 고를 수 있는 것이 도움이 되는지를 보는 ablation이지 별도 논문 알고리즘의 이름이 아니다.
'''))
    parts.append(toggle('token-seed-best-first', 'Ablation — 기존 token 경로 8개만 사용하는 best-first search', r'''
실험 식별자: `token_seed_best_first`. 초기 node를 기존 deterministic rollout 첫 8개에서 온 interval로만 만든다. 새 leading-digit 경로와 JSON proposal은 이 방법에 주지 않고, search 중 새 occurrence proposal도 하지 않는다.

$$
\mathcal I_0=\operatorname{unique}_{[s,e]}\operatorname{parse}(\mathcal C_8),\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal F}(V_i-0.05k_i).
$$

경계 regeneration 자체가 다른 occurrence로 이동하는 것은 허용한다. critic·tree depth·final pairwise selector는 공통이다. initial proposal 비용도 seed 평가에 필요한 부분만 부과한다. `value_best_first`와의 차이는 **후보 공급 범위**이며, search selection 공식을 바꾼 비교가 아니다.
'''))
    return '\n'.join(parts)


DETERMINISTIC = r'''
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
'''


def deterministic_methods():
    return DETERMINISTIC + '\n'.join([
        toggle('greedy', 'Greedy decoding — 후보 1 유지', r'''
실험 식별자: `greedy`. $\hat I_K=I_1$로 모든 예산에서 원래 greedy 응답을 유지한다. 후보를 추가 생성한 효과와 selector의 변경 효과를 비교하는 baseline이다. greedy의 step별 argmax는 sequence 전체의 maximum likelihood와 다르다.
'''),
        toggle('maximum-likelihood', 'Maximum likelihood selection — 완성 경로 확률이 가장 높은 후보', r'''
실험 식별자: `max_probability`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\ell_i.
$$

EOS를 포함한 sequence log-likelihood를 그대로 사용하며 length normalization을 하지 않는다. 같은 값이면 먼저 생성한 작은 candidate index를 선택한다. 모든 가능한 응답에 대한 전역 maximum likelihood가 아니라 이미 만든 후보 안에서의 최대값이다.
'''),
        toggle('medoid', 'Medoid selection — 후보들과의 temporal IoU 합이 가장 큰 후보', r'''
실험 식별자: `medoid`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\sum_{j\in\mathcal C_K^{+}}\mathrm{IoU}(I_i,I_j).
$$

각 후보가 다른 후보들과 얼마나 일치하는지를 동일 가중으로 더한다. $j=i$인 self-IoU도 포함한다. $1-\mathrm{IoU}$를 dissimilarity로 둔 medoid, 즉 전체 dissimilarity 합을 최소화하는 실제 후보 선택과 같다. 점수가 같으면 높은 $\ell_i$, 그다음 작은 index를 고른다. 여러 token 경로가 같은 interval로 가면 그 반복도 합에 들어가므로 서로 다른 event의 동일 가중 투표가 아니다.
'''),
        toggle('weighted-medoid', 'Weighted medoid selection — 경로 확률을 가중치로 사용', r'''
실험 식별자: `probability_weighted_medoid`.

$$
w_j=\exp(\ell_j-\ell_{\max}),\qquad
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\sum_{j\in\mathcal C_K^{+}}w_j\mathrm{IoU}(I_i,I_j).
$$

확률이 큰 경로와 일치하는 후보를 더 선호한다. $w_j$를 합이 1이 되도록 나누어도 argmax는 같다. 이는 유한 후보 안의 likelihood 가중치이며 token path likelihood를 정답 확률로 calibration한 것은 아니다. self term과 tie-break는 medoid와 같다. 당시 결과의 주된 likelihood+interval agreement 선택은 이 기준이며, maximum likelihood와 별도 열로 보고했다.
'''),
        toggle('oracle', 'Oracle — GT로 후보 안의 상한 계산', r'''
실험 식별자: `oracle`.

$$
i^*=\operatorname*{arg\,max}_{i\in\mathcal C_K^{+}}\mathrm{IoU}(I_i,G).
$$

동률은 작은 index를 선택한다. 후보 생성 후 평가에서만 GT를 사용한다. oracle과 실제 selector의 차이는 후보 안에 이미 있는 좋은 interval을 놓친 정도이며, 후보 밖의 더 좋은 사건을 찾아낼 가능성까지 측정하지는 않는다. 예산별 모든 oracle 결과는 기존 성능 표에 그대로 유지했다.
''')])


RECONSIDERATION = r'''
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
'''


def reconsideration_methods():
    return RECONSIDERATION + '\n'.join([
        toggle('direct-regeneration', 'Direct regeneration — 후보 목록만 주고 답을 다시 생성', r'''
실험 식별자: `direct_pool8`, `direct_pool32`. 후보 목록 $L_K$를 새 user prompt에 넣은 뒤 바로 답을 생성한다.

$$
\hat y=\operatorname{greedy}_\theta(c_{\mathrm{original\ answer}}\oplus L_K),\qquad K\in\{8,32\}.
$$

별도 inspection·binary critic·pairwise comparison이 없다. 추가 model evaluation은 마지막 generation 1회다. 단순히 후보를 다시 보여주고 reasoning 기회를 한 번 준 것의 효과를 비교한다.
'''),
        toggle('agent-review', 'Agent with tool use — controller와 visual inspection 반복', r'''
실험 식별자: `agent_pool8`은 최대 3회, `agent_pool32`는 최대 7회 action-observation round다. controller·inspector 모두 같은 frozen VLM이다. $h_r$는 지금까지 review history, $a_r$는 inspection action, $o_r$는 inspector 관찰이다.

$$
a_r=\operatorname{greedy}_\theta(c_{\mathrm{controller}},L_K,h_{<r}),\qquad
o_r=\operatorname{greedy}_\theta(c_{\mathrm{inspector}},a_r),\qquad
h_r=h_{r-1}\oplus(a_r,o_r).
$$

controller는 `event(a,b)`로 두 후보가 서로 어떤 occurrence인지 비교하거나 `boundary(a)`로 한 후보의 onset·offset을 검사하거나 `done`을 고른다. 관찰이 한 번 이상 있으면 early stop할 수 있다. 잘못되거나 반복된 action은 아직 안 본 boundary, 그다음 event pair 순서로 fallback한다. 계획과 관찰은 각각 최대 48 token이다.

마지막에는 $L_K,h_r$로 timestamp를 다시 생성한다. model evaluation 상한은 $2r+1$이므로 7회/15회다. 모델이 어디를 더 볼지 고르는 점이 고정 verifier와 다르다. observation은 새 frame을 더 추출하는 도구가 아니라 같은 video input의 VLM 재검토다. 새 policy 학습은 없다. [Language Agent Tree Search](https://arxiv.org/abs/2310.04406)는 참고 방향이다.
'''),
        toggle('verifier-review', 'Pointwise verification — event와 boundary를 각각 평가', r'''
실험 식별자: `verifier_pool8`은 대표 최대 3개, `verifier_pool32`는 최대 7개를 검증한다. 각 후보에 대해 event가 맞는지와 boundary가 맞는지를 두 별도 Yes/No prompt로 묻는다.

$$
S_i=b_{c_i^{\mathrm{event}}}(\mathrm{Yes})\ b_{c_i^{\mathrm{boundary}}}(\mathrm{Yes}),\qquad
i^*=\operatorname*{arg\,max}_i S_i.
$$

점수가 같으면 작은 candidate index를 고른다. 질문을 따로 했다고 두 판단이 통계적으로 독립인 것은 아니며 위 곱은 heuristic ranking이다. raw Yes/No probability 합도 저장했다. 모든 점수와 preferred candidate를 final regeneration prompt에 전달하므로 최종 답이 반드시 $I_{i^*}$인 것은 아니다.

추가 model evaluations는 후보당 2회+마지막 generation 1회, 최대 7회/15회다. 같은 VLM의 자기 평가를 사용했으며 학습한 verifier·GT likelihood·external ground truth를 사용하지 않았다.
'''),
        toggle('tts-review', 'Test-time scaling — 모델 비교에 쓰는 연산을 4개에서 8개로 확대', r'''
실험 식별자: `tts_pool8_select4`, `tts_pool8_select8`, `tts_pool32_select4`, `tts_pool32_select8`. $K\in\{8,32\}$는 원래 rollout pool, $m\in\{4,8\}$은 그 안에서 모델로 비교할 farthest-first 대표 수다. 이 2×2 구성이 proposal compute와 selection compute의 효과를 나누어 본다.

후보 쌍 $i,j$에 대해 A/B 중 어느 interval이 더 정확한지를 두 presentation order로 평가한다. $b_{i,j}(A)$는 A=$i$, B=$j$일 때 두 선택지 안에서 정규화한 A의 확률이다.

$$
\pi_{i,j}=\tfrac12\{b_{i,j}(A)+b_{j,i}(B)\},\qquad
i\text{ wins iff }\pi_{i,j}\ge0.5.
$$

knockout tournament를 반복하고 홀수 후보는 다음 round로 바로 올린다. 비교 기록과 winner를 final regeneration에 전달한다. 후보가 충분하면 pairwise model evaluations는 $2(m-1)$회이고 마지막 generation까지 $2m-1$회, 즉 7회/15회다. 마지막 답이 tournament winner와 같도록 강제하지 않는다.

이 TTS는 단순 maximum likelihood 후처리가 아니다. **동일한 rollout pool 안에서도 4→8개로 모델 비교 연산을 늘리는 축**을 갖는다. 모든 pair를 비교한 ranking·여러 무작위 answer의 self-consistency·학습된 reward model의 best-of-N은 아니다. [Test-time compute scaling 연구](https://arxiv.org/abs/2408.03314)는 관련 맥락이며 해당 논문의 학습·추론법 전체를 재현하지 않았다.
'''),
        toggle('probability-replay', 'Counterfactual probability evaluation — 분기 판단이 바뀌었는지 사후 측정', r'''
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
''')])


CONDITIONAL = r'''
## Shared pipeline — start에 조건화된 end 탐색 비용 진단

이 실험은 고정 pilot300에서 데이터셋별 앞선 10개, 총 30 query를 사용하는 **예산 진단**이다. 모델은 Qwen3-VL-8B-Instruct, 2 FPS, spatial target 100,352 pixels다. start와 end를 독립적인 두 분포로 다루지 않고, start·delimiter를 포함한 정확한 prefix마다 end distribution을 따로 계산했다.

greedy로 생성한 답에서 첫 숫자 직전까지를 고정 anchor로 둔다. 이 절의 모든 probability는 그 anchor가 이미 주어졌다는 조건부 확률이다. anchor 자체를 생성할 확률은 곱하지 않는다.

탐색 grammar는 `number - number seconds`다. number는 음수가 아닌 정수와 선택적인 소수 첫째 자리 1개이며, 숫자 자리 수 상한은 $\max(4,\operatorname{digits}(\lceil D\rceil))$다. `-` 전후 whitespace는 각 최대 2개, suffix는 최대 32 token이다. 이 diagnostic grammar는 seconds만 허용한다. 후속 일반 응답 실험의 `MM:SS` parsing 지원을 이 실행에 소급하지 않는다.

각 prefix에서 원래 full-vocabulary softmax를 계산하고 grammar에 맞는 token을 child로 만든다. **grammar 안의 token들끼리 정규화하지 않는다.** grammar를 벗어난 probability는 별도 합계로 남긴다. terminal은 grammar의 `seconds`까지 완성된 prefix이며 EOS를 요구하지 않는다. 따라서 여기서 terminal probability는 EOS 포함 완성 assistant answer probability와 다르다.

`-`가 처음 포함된 token까지의 정확한 token ID prefix를 start key로 저장한다. token 하나가 delimiter와 end의 일부까지 포함하면 그것도 별도 start key다. 같은 숫자 start라도 다른 whitespace·tokenization이면 end 조건이 달라질 수 있으므로 합치지 않는다. 숫자값으로 합치는 것은 사후 표시뿐이다.

$\pi(u)$는 anchor 이후 prefix $u$의 확률 곱, $\mathcal F$는 아직 확장하지 않은 prefix, $\mathcal L$은 완성 terminal 집합이다. 매 node expansion은 다음-token distribution 하나를 조회하고 그 child들을 만든다. 예산은 $8,16,32,64,128,256,512$회 expansion이며 두 정책에 같은 checkpoint를 사용한다. decoder input token·forward call·시간은 별도로 측정한다.

## Expansion policies — 어떤 prefix를 다음으로 열 것인가
'''


def conditional_methods():
    return CONDITIONAL + '\n'.join([
        toggle('joint-best-first', 'Best-first search — prefix probability 순으로 확장', r'''
실험 식별자: `joint`.

$$
u^*=\operatorname*{arg\,max}_{u\in\mathcal F}\pi(u).
$$

확률이 큰 prefix부터 확장한다. start에 속한 end를 이미 많이 보았는지는 고려하지 않는다. joint start/end 경로의 probability가 높은 부분을 빠르게 보는 장점이 있지만, 높은 확률의 한 start 아래로 계산이 몰릴 수 있다. grammar와 길이 제한 밖으로는 탐색하지 않는다.
'''),
        toggle('balanced-priority', 'Visit-count penalty — 같은 start 아래 반복 확장에 penalty', r'''
실험 식별자: `balanced`. $k(u)$는 prefix의 exact start key, $n_k$는 그 key 아래 이미 수행한 expansion 수다.

$$
u^*=\operatorname*{arg\,max}_{u\in\mathcal F}
\begin{cases}\pi(u)&k(u)\text{ not yet defined},\\
\pi(u)/(1+n_{k(u)})&\text{otherwise}.\end{cases}
$$

start가 아직 완성되지 않았으면 원래 prefix probability를 쓴다. start가 정해진 뒤에는 같은 조건 아래 end를 많이 확장할수록 우선순위를 낮춘다. 모든 start를 균등하게 탐색한다는 보장은 없고, 낮은 probability의 start는 여전히 보지 못할 수 있다. 이는 이번 실행의 우선순위 heuristic이며 별도의 표준 알고리즘 이름을 붙이지 않는다.
'''),
        toggle('probability-accounting', 'Probability accounting — 탐색하지 못한 부분까지 구분하는 방법', r'''
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
''')])


STOCHASTIC = r'''
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
'''


def stochastic_methods():
    return STOCHASTIC + '\n'.join([
        toggle('top-k-sampling', 'Top-k sampling — 확률 상위 k개 안에서 추출', r'''
실험 식별자: `top_k`. $\widetilde p$의 상위 $k$개를 남기고 그 안에서 sampling한다. Qwen3는 $k=20$, Qwen2.5는 $k=50$이다. top-p는 1로 두어 별도 truncation을 하지 않았다.

$$
S_t=\operatorname{TopK}_a\widetilde p_t(a),\qquad y_t\sim\operatorname{Categorical}(q_t).
$$

실제 Hugging Face 구현은 k번째 logit과 동률인 token을 함께 남길 수 있어 후보 수가 k보다 많을 수 있다. 이 token도 모두 기록했다. 독립 rollout마다 같은 높은 확률 token을 반복 추출할 수 있다. 후보를 확률 순서대로 중복 없이 검사하는 deterministic 분기와 다르다.
'''),
        toggle('top-p-sampling', 'Nucleus sampling (top-p) — cumulative probability 기준으로 추출', r'''
실험 식별자: `top_p`. token을 $\widetilde p$ 내림차순으로 정렬하고 다음 최소 prefix를 남긴다.

$$
m=\min\left\{h:\sum_{j=1}^{h}\widetilde p_{(j)}\ge p\right\},\qquad
S_t=\{(1),\ldots,(m)\}.
$$

Qwen3는 $p=0.8$, Qwen2.5는 $p=1.0$이다. top-k를 0으로 끈 분리 실험이다. 따라서 Qwen2.5의 이 조건은 실제로 full-vocabulary sampling이며 truncation 강도가 Qwen3와 같지 않다. 모델 카드에서 top-k와 top-p를 함께 쓰는 설정 자체를 재현한 비교도 아니다. [Nucleus sampling 원 논문](https://arxiv.org/abs/1904.09751).
''')])


FAMILIES = {
    'event_tree_search': event_methods,
    'deterministic': deterministic_methods,
    'reconsideration': reconsideration_methods,
    'conditional_budget': conditional_methods,
    'stochastic': stochastic_methods,
}


def method_block(family):
    body = FAMILIES[family]().strip()
    if family != 'event_tree_search':
        body = COMMON.strip() + '\n\n' + body
    return BEGIN + '\n\n' + body + '\n\n' + END + '\n'


def formalize(text, family, count=None, spec=False):
    """Insert the same method source into existing MD; leave result tables intact."""
    block = method_block(family)
    if spec:
        title = {'event_tree_search': 'Event proposal and tree search',
                 'deterministic': 'Deterministic rollout',
                 'reconsideration': 'Rollout reconsideration',
                 'conditional_budget': 'Conditional rollout budget',
                 'stochastic': 'Stochastic rollout'}[family]
        return f'# {title} · pilot{count} · Methods\n\n' + block
    pattern = re.compile(re.escape(BEGIN) + r'.*?' + re.escape(END) + r'\n?', re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    # Keep the title, completion statement and interpretation warnings first.
    match = re.search(r'^## ', text, re.M)
    if not match:  # The stochastic report begins with a table, without headings.
        match = re.search(r'^\|', text, re.M)
    position = match.start() if match else len(text)
    return text[:position].rstrip() + '\n\n' + block + '\n' + text[position:]


def refresh_spec(folder, family, count):
    """Refresh a registered SPEC without creating absent cohort documents."""
    path = Path(folder) / 'SPEC.md'
    if path.exists():
        path.write_text(formalize('', family, count, spec=True))


def refresh_existing():
    changed = []
    for family, counts in [('event_tree_search', [300, 600, 900]),
                           ('deterministic', [300, 600, 900]),
                           ('conditional_budget', [30]), ('stochastic', [300])]:
        for count in counts:
            folder = ROOT / 'results/rollout' / family / 'pilot' / str(count)
            for name in ['REPORT.md', 'SPEC.md']:
                path = folder / name
                if not path.exists():
                    continue
                original = path.read_text()
                content = formalize(original, family, count, spec=name == 'SPEC.md')
                if content != original:
                    path.write_text(content)
                    changed.append(str(path.relative_to(ROOT)))
    for count in [300, 600, 900]:
        folder = ROOT / 'results/rollout/deterministic/pilot' / str(count) / 'reconsideration'
        for name in ['REPORT.md', 'SPEC.md']:
            path = folder / name
            if path.exists():
                original = path.read_text()
                content = formalize(original, 'reconsideration', count, spec=name == 'SPEC.md')
                if content != original:
                    path.write_text(content)
                    changed.append(str(path.relative_to(ROOT)))
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-existing', action='store_true')
    args = parser.parse_args()
    if args.refresh_existing:
        for path in refresh_existing():
            print(path)
