# VTG 실험 History

관찰: 2026-11-11T20:21:11.660038+09:00

실험 보고서를 원본 근거로 정리했다. 실험 재실행 없이 보고된 수치를 인용하며, 완료·진행중·폐기를 구분한다. 코드 원본경로·SHA-256·파일수정시각·수집시각은 [manifest](EXPERIMENTS.json)에 기록한다. 현재 코드를 역사적 실행 당시 코드와 동일하다고 단정하지 않는다.

## 연구 맥락별 계보

### [Zero-shot 기준선과 숫자 불확실성](groups/baseline-numeric.md)

전체9,404 query/model과 pilot300, 제한 head·full-head top30·자릿수·endpoint·rollout의 계보. Full 18,808응답과 rollout6,000응답은 별도 실행이며 부분집합을 합산하지 않는다.

### [입력 순서와 Attention](groups/order-attention.md)

영상→질문 / 질문→영상 성능·예측 일관성과 query/generation/hidden/spatial/dynamics attention. MTLA pilot과 TimeLens 전체평가를 구분하며 계측량과 성능의 인과를 단정하지 않는다.

### [Thinking/Instruct와 FPS](groups/thinking-fps.md)

고정pilot300 ×두 입력순서. 2fps Thinking600회, 8fps 두모델1,200응답·trace1,200개 완료 보고. 8fps mIoU Instruct VQ48.49/QV36.96%, Thinking49.24/46.43%; 2fps대비 +.27/+1.08/+.62/−1.48pp. Decoding·토큰예산·공간해상도·768frame cap이 함께 영향을 준다.

### [공간 해상도 통제와 Temporal Sampling](groups/spatial-sampling.md)

Nested300⊂600⊂900, 데이터셋별300 query. 공간통제v1은 두공간목표50,176/100,352픽셀 ×4조건=7,200논리응답. 후속v2는 A/B ×두공간목표 ×19조건 ×900=68,400응답 목표. 완료 전 명세·임시보고서를 최종결과로 간주하지 않는다. 11월2일 폐기된 공간 비통제10실험의 결과는 제외한다. 남은 과거 구현은 보존하되 재실행·결과복원하지 않는다. GT oracle 결과와 실제 추론 알고리즘을 구분한다.

### [Frame–Query 유사도와 비용](groups/similarity.md)

CLIP ViT-L/14와 Qwen3-VL-Embedding-8B. 2fps pilot300 query/292 videos →full9,404 query/4,279 videos →16fps pilot300 비용비교. Full mAP .4508/.5103, MRR .5594/.6247. 16fps GPU forward CLIP .7251s/Qwen15.0923s/Instruct7.8314s. 네이티브 전처리·해상도·구조가 달라 파이프라인 비용비교다. 인접frame cosine은 저장embedding의 파생분석이다.

### [입력 변환과 실패 유형](groups/failure-timestamp.md)

Charades·QVHighlights 각100 query. Qwen2.5 native / TimeLens-7B식 입력 / Qwen3 native. Qwen2.5 변환 R@IoU.5 −19/−28pp. Frame복제·fps·modality·시간표현·지시문이 함께 바뀌므로 단일원인 효과로 해석하지 않는다.

### [공용 전처리와 코드 의존성](groups/shared.md)

공용 입력·저장·결과 경로·검증 코드 및 TimeLens/Qwen embedding/MTLA 외부 소스·라이선스. 독립 실험이 아니라 재현을 위한 공통 의존성이다.

## 보관 범위

실행·분석·보고서 생성·검증 코드 및 작은 설정·결과 요약을 함께 보관한다. 외부 vendor 소스와 라이선스도 포함한다. 큰 원자료와 모델 가중치는 서버 원본을 참조한다. 삭제된 결과는 현재 manifest에서 제외하며 Git 과거 이력은 유지한다. 실험 관련 대화의 사용자 요청·최종 보고는 포함한다. 일반 잡담·무관한 주제·도구 출력·내부 reasoning과 수집기 운영 로그는 제외한다.
