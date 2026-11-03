# MTLA attention 통합 실험 결과 · pilot300 (300질문)

같은 Qwen3-VL-8B-Instruct와 같은 300개 입력(데이터셋별 100개)을 사용한 분석을 통합했다. 기존 측정값을 재사용했으며, 통합 과정에서 모델 추론을 새로 수행하지 않았다. 9개 탐색 샘플은 300개 실험의 부분집합이며 모든 저장 배열이 bitwise 동일함을 확인했다.

[통합 뷰어](report.html) · [통합 CSV](../../../../../artifacts/runs/token_attention/mtla/pilot/300/metrics/sample_metrics.csv) · [요약 JSON](../../../../../artifacts/runs/token_attention/mtla/pilot/300/metrics/summary.json)

| 데이터셋 | 샘플 | Query peak∈GT | Answer peak∈GT | Answer 초반 peak | 토큰 간 TV 중앙값 | Hidden–attention TV 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Charades-STA | 100 | 50% | 29% | 95% | 0.118 | 0.197 |
| QVHighlights | 100 | 49% | 22% | 97% | 0.144 | 0.249 |
| ActivityNet-Captions | 100 | 50% | 35% | 84% | 0.141 | 0.223 |

Answer는 생성 토큰을 처리하는 consumed 행이다. 토큰 간 TV는 predictive 행에서 영상 내부 분포를 정규화한 뒤 샘플 내 평균 → 샘플 간 중앙값으로 계산한다. 초반 peak는 영상 앞 10%에 해당한다. Hidden 비교는 cosine을 온도 0.1로 softmax한 분포와 영상 내부 attention 분포 사이의 TV다. 서로 다른 정의의 수치를 하나의 평균으로 섞지 않는다.

Query보다 답변 attention의 최고점이 영상 초반에 더 자주 나타나고, GT 내부 최고점 비율은 낮다. 이 지표만으로 영상 정보의 인과적 사용 여부를 판단할 수는 없다.

## 상세 결과

- [Query attention](query/report.html)
- [Query–answer 비교](generation/report.html)
- [분포 비교](generation/distribution_analysis/REPORT.md)
- [레이어·head 분석](generation/head_analysis/report.html)
- [생성 토큰별 변화](dynamics/report.html)
- [Hidden similarity 비교](hidden/report.html)
- [공간 attention·query 교체 대조](spatial/report.html)

샘플 선택, 전처리, capture 설정은 각 분석의 run_config.json과 원본 메타데이터에 보존했다. 공통 실행 명령은 상위 README.md를 참고한다.
