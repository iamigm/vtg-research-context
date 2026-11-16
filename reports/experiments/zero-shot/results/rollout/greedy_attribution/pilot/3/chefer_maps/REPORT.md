# Chefer decoder maps · pilot3

**지도 완료 3/3 query · 상태 `complete`**

기존 oracle-switch 극단 사례 3개에서 원래 greedy 분기 token의 logit을 설명한다. 동일 입력·동일 target·동일 video time에서 attention rollout과 final C를 비교한다. 3개는 성능 차이로 선택한 진단 사례이며 전체 benchmark의 평균을 대표하지 않는다.

## 비교하는 두 화면

| 화면 | 정의 | 해석 |
| --- | --- | --- |
| Attention rollout | 원래 attention A의 head 평균에 residual I를 더하고 행 정규화 후 layer를 합성 | Target logit gradient·LRP를 사용하지 않는 baseline |
| Final C | mean_h max((∂z_t/∂A)⊙R_A,0)에 I를 더해 전체 decoder layer 합성 | 선택 target logit의 gradient로 가중한 LRP attention 경로 |

두 지도는 값의 단위와 변환이 달라 같은 색을 같은 절댓값으로 비교하지 않는다.

## 수식과 target

A_l^h는 layer l, head h의 causal attention matrix, R_l^h는 해당 matrix에 대한 LRP relevance, G_l^h=∂z_t/∂A_l^h는 선택한 greedy token logit z_t의 gradient다. [x]₊=max(x,0), E_h는 head의 산술평균이며 positive clipping은 head 평균 **전**에 적용한다. 모든 입력 token을 포함해 계산한 뒤 예측 위치 p의 visual-token 열만 지도에 추출한다.

LRP의 output seed는 선택 token 위치의 one-hot1이다. Linear에는 공식 α=1 규칙, residual Add에는 relevance 재정규화, bilinear 연산에는 두 operand의 절반 분배를 사용한다. Gradient의 objective는 별도로 실제 logit z_t다. 0-based step s에서 predictor 위치는 p=N_prompt+s−1이다.

- Attention baseline: B_l=RowNorm(I+E_h[A_l^h]), map=e_pᵀB₃₅⋯B₀.
- Final C: C_l=I+E_h[[G_l^h⊙R_l^h]₊], map=e_pᵀC₃₅⋯C₀.

Final C는 공식 LRP 구현에 맞춰 layer matrix의 행 정규화를 하지 않는다. Attention baseline에만 RowNorm을 적용하며 분모는 해당 행의 모든 token 열 합이다. 역방향 vector–matrix product로 필요한 target 행을 계산하는 것은 이 행렬곱의 대수적 계산 방식이며 일부 token을 생략한 근사가 아니다. 지도는 확률·GT IoU·자연어 근거의 비율이 아니며 높은 값만으로 causal effect를 증명하지 않는다.

## Qwen decoder에 적용한 범위

이는 Chefer et al.(CVPR2021)의 ViT/BERT 분류 구현을 Qwen의 decoder target logit에 이식한 분석이다. Vision encoder 전체에 LRP를 적용한 pixel attribution으로 부르지 않는다. Decoder visual token 위치에 배치된 contextual representation을 표시한다. DeepStack 경로의 relevance는 별도 분기로 분리한다. RMSNorm·SiLU의 relevance 전달은 identity, RoPE는 고정 linear z-rule, SwiGLU bilinear 연산은 두 operand에 절반씩 전달하는 이식 가정을 사용한다. 원 논문과 architecture가 다르므로 원 논문의 relevance conservation·설명 품질이 이 모델에서 자동 보장된다고 주장하지 않는다. Forward 재현과 수치 검증은 query별 validation 원자료를 확인한다.

## 영상·색상 읽기

각 visual token의 시간 위치는 실제 입력 frame 두 장의 pair에 대응한다. 두 source frame에 같은 pair 지도를 표시하며 두 시점 사이의 새로운 지도를 보간하지 않는다. 재생 중에는 현재 video time에 가장 가까운 저장 input timestamp의 pair를 사용한다. 수동으로 선택한 반복 timestamp의 input slot은 정지 상태에서 보존한다. Frame index는 위, 초 단위 소수 timestamp는 아래에 표시한다.

원본 영상의 실제 가시 영역에 지도 전체 좌표를 맞춘다. 모델 입력은 direct resize를 사용하므로 원영상과 H×W 또는 종횡비가 다를 수 있다. Spatial cell은 contextual visual-token 위치이며 해당 원본 pixel 영역만의 영향이라고 해석하지 않는다.

각 map의 모든 frame·cell 최대값 M을 고정해 color ratio=v/M으로 표시한다. Frame마다 최대값을 다시 맞추지 않는다. M=0이면 모든 값이0임을 명시한다. 다색 palette는 낮은 값에서 높은 값 순서다. 투명도0%는 모든 cell을 완전 불투명하게 표시하며, 투명도100%는 원영상만 표시한다. Overlay의 alpha는 1−투명도/100이며 값의 크기는 alpha가 아닌 색에 반영한다.

## 사례와 원자료

| Query | Target step (0-based) | Token | 지도 |
| --- | --- | --- | --- |
| qvhighlights-timelens_00229 | 5 | `4` | 완료 |
| activitynet-timelens_00812 | 5 | `1` | 완료 |
| charades-timelens_00810 | 5 | `0` | 완료 |

- [선정·원본 입력 참조](../../../../../../artifacts/runs/rollout/greedy_attribution/pilot/3/chefer_maps/metadata/selection.json)
- [실행 설정](../../../../../../artifacts/runs/rollout/greedy_attribution/pilot/3/chefer_maps/metadata/config.json)
- [현재 상태](../../../../../../artifacts/runs/rollout/greedy_attribution/pilot/3/chefer_maps/metadata/status.json)

## 논문과 공식 구현

- [Chefer, Gur, Wolf · Transformer Interpretability Beyond Attention Visualization · CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf)
- [공식 Transformer-Explainability 구현 · commit c3e578f76b95](https://github.com/hila-chefer/Transformer-Explainability/tree/c3e578f76b954e8528afeaaee26de3f07e3fe559)
- [공식 LRP layer 합성과 final C](https://github.com/hila-chefer/Transformer-Explainability/blob/c3e578f76b954e8528afeaaee26de3f07e3fe559/baselines/ViT/ViT_LRP.py)
- [공식 attention rollout baseline](https://github.com/hila-chefer/Transformer-Explainability/blob/c3e578f76b954e8528afeaaee26de3f07e3fe559/baselines/ViT/ViT_explanation_generator.py)
