# Rollout reconsideration · pilot600

고정 pilot600, dataset별200 query다. 기존600개와 순서를 유지하며 GT·예측 성능으로 query를 재선택하지 않는다. Qwen3-VL-8B-Instruct, 2 FPS, 약100,352 pixels의 기존 spatial resolution, video_query input order 및 실제 frame IDs/PTS/input tensor hash를 유지한다. 새로운 학습이나 stochastic sampling을 하지 않는다.

## 가설과 세 접근

재검토 context가 주어졌을 때 token distribution과 path probability가 바뀌는지, 변화가 실제 localization 개선으로 이어지는지 평가한다. model weights는 frozen이며 context-conditioned distribution의 변화다.

| 접근 | 재검토 방식 | low / high |
|---|---|---|
| agent | controller가 event comparison 또는 boundary inspection을 고르고, 같은 VLM의 독립 visual inspector observation을 받아 다음 action을 결정 | 최대3 / 7 action-observation rounds + final regeneration |
| verifier | 서로 다른 candidate3 / 7개의 event alignment와 boundary fit을 독립 Yes/No likelihood로 평가하고 final regeneration | 최대7 / 15 model evaluations |
| test-time scaling | pool8 / pool32에서 candidate4 / 8개를 선택해 model pairwise knockout, 각 pair를 양쪽 presentation order로 평가하고 final regeneration | generation pool × selection compute의2×2 factorial |

세 방법은 같은 candidate pool 및 video evidence를 사용한다. agent의 inspector는 fixed verifier와 같은 backbone을 쓰지만 inspection 순서와 종류가 adaptive다. verifier는 fixed independent pointwise assessments이며, test-time scaling은 fixed-schedule pairwise search다. 이 차이를 operational comparison으로 해석하며 서로 배타적인 이론 범주라고 주장하지 않는다.

TTS는 pool8/select4, pool8/select8, pool32/select4, pool32/select8을 모두 수행한다. 따라서 후보 수를 고정한 selection scaling과 selection 수를 고정한 generation scaling을 구별한다. 선택은 max probability 후처리가 아니라 실제 video-conditioned model inference다. 같은 interval은 중복해서 padding하지 않고, 후보가 부족하면 실제 candidate·evaluation 수를 기록한다.

## 공통 조건과 비용

기존 deterministic rollout의 앞8/32 candidate를 참조하며 새로 복제·sampling하지 않는다. unique interval을 GT 없이 farthest-first로 골라 greedy와 떨어진 temporal hypothesis를 포함한다. pool generation의 원래 decoder cost도 차감하지 않고 합산하며, online 재사용 비용과 conceptual deployment 비용을 구분한다. 기존 greedy, max probability, probability-weighted medoid, oracle과 함께 직접 재응답(direct regeneration, model evaluation1회)을 비교한다.

low/high의 주 비교는 pool8/32와 model evaluation ceiling7/15다. agent early stop과 candidate 부족으로 실제 비용은 달라질 수 있다. model evaluation 수만으로 FLOPs가 같다고 주장하지 않는다. prompt processing·generated token·KV replay를 포함한 decoder input tokens, forward calls, synchronized model wall seconds, attention-context work를 기록하고 성능-비용 Pareto를 비교한다. diagnostic probability replay 비용은 의사결정 비용과 분리해 둘 다 보고한다.

final regeneration은 모든 방법에 같은 timestamp 응답 형식을 요청하며 grammar-constrained decoding을 하지 않는다. 파싱 실패·invalid interval·EOS 미도달은0점이다. main score에서 이전 답으로 자동 fallback하지 않는다. clipping이나 rounding도 하지 않는다.

## 사건 변경과 boundary refinement

GT 없는 complete-link temporal IoU≥0.5 clustering으로 candidate hypotheses를 나눈다. 다른 cluster로 이동했는지와 같은 cluster 안에서 boundary가 달라졌는지를 구분한다. cluster는 temporal occurrence의 proxy이며 semantic event identity의 gold label이 아니다. clustering threshold0.3/0.5/0.7의 민감도도 함께 보고한다.

- event recovery proxy: valid greedy IoU<0.1 → 새 응답 IoU≥0.5이며 temporal cluster도 바뀐 경우. 분모는 valid greedy IoU<0.1인 query다.
- switch rate: valid greedy와 valid final이 다른 cluster인 query / 두 응답이 모두 valid인 query.
- beneficial switch: switch 중 ΔIoU>0.01 비율. harmful revision은 greedy IoU≥0.5 → final IoU<0.5다.
- boundary refinement: greedy IoU≥0.5이고 final도 같은 temporal cluster에 속한 paired-valid query의 endpoint MAE 및 ΔIoU. 해당 분모를 항상 표시한다.
- endpoint MAE=(|start−GTstart|+|end−GTend|)/2, 단위seconds, 낮을수록 좋다. invalid를 제외한 조건부 지표이므로 failure rate와 함께 본다.

주 성능은 mIoU와 R@0.3/0.5/0.7이며, 전체600 query와 dataset별, video duration별, GT/video duration ratio별, 두 축 교차표를 함께 보고한다. paired bootstrap은 video 단위 재표집(seed42)으로 시행한다. GT는 이 보고 단계에서만 쓴다.

## token distribution과 path probability

기존 pool32에서 GT 없이 event proxy(disjoint intervals, center separation>max(2s,0.05×duration))와 local boundary proxy(temporal IoU≥0.5, endpoint shift≤max(2s,0.1×parent length)) 각각 최대1개를 고정한다. 각 종류에서 branch token의 raw probability가 큰 probe를 택한다. low pool에 포함되지 않은 probe는 prompt에 노출하지 않고 사후 counterfactual diagnostic에만 쓴다.

동일한 생성 token prefix를 review context 뒤에 teacher-force하여 원래·대체 token의 확률 변화, log odds 변화, full-vocabulary JS divergence를 측정한다. 두 선택 뒤의 child distribution도 저장한다. 원래 parent/alternative 응답의 EOS 포함 path probability를 같은 review context에서 다시 계산한다. prefix를 고정한 counterfactual probability와 unconstrained final prediction을 구분한다.

qvhighlights-timelens_00507은 고정 pilot600에 없다. 기존 trace를 이용한 reference case audit만 별도로 기록하며600개 통계에 포함하거나 cohort를 교체하지 않는다.

## 문헌과 적용 범위

- [Snell et al., Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): proposal revision과 verifier-guided search를 구분하고, 평가·수정 연산을 포함한 예산을 비교한다. 이 pilot은 학습된 PRM을 재현하지 않으며 frozen VLM judge로 두 축을 시험한다.
- [Qin et al., Pairwise Ranking Prompting](https://arxiv.org/abs/2306.17563): pairwise selection을 이용하고 두 후보의 presentation order를 뒤집어 평가한다. 문서 ranking에서 video interval selection으로 적용한 실험 설계이며 원 논문 성능의 재현이 아니다.
- [Yao et al., ReAct](https://arxiv.org/abs/2210.03629): agent의 action과 observation을 교대로 수행한다. 여기서는 event/boundary visual inspection의 순서를 controller가 고른다.
- [Liu et al., VideoMind](https://arxiv.org/abs/2503.13444): planner·grounder·verifier 역할의 분리를 참고한다. 해당 논문의 LoRA나 학습 가중치를 사용하지 않으므로 zero-shot workflow 비교로 한정한다.
