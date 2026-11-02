# 답변 생성 토큰별 attention 변화

동일한 Charades-STA test / QVHighlights val / ActivityNet-Captions val_1 각 100개 샘플을 분석한다.
입력·프롬프트·전처리·생성 답변은 이전 `mtla_query_vs_generated_answer_attention_100` 그대로다.
이 실험은 실제 생성 중 저장된 attention에 대한 추가 측정이며, 답변을 다시 생성하지 않는다.
원본 MTLA 코드는 이 디렉토리의 `MTLA/`로 복사했고 `vtg_works`는 변경하지 않는다.

- [분석 보고서](FINDINGS.md)
- [인터랙티브 뷰어](results/index.html): 다운로드하거나 브라우저에서 열면 서버·외부 라이브러리 없이 동작한다.
- [전체 토큰 CSV](results/all_token_metrics.csv)
- [요약 그림](results/summary.png)

## 토큰 단계의 정의

입력 길이를 P, 0-based 생성 토큰 번호를 j라고 할 때:

- **predictive (주 분석):** P+j−1 행. j번째 토큰을 선택하는 실제 생성 forward에서 읽은 attention.
  첫 단계는 assistant prefix 마지막 행이다. 이전 답변 key에는 j 이전에 출력된 토큰만 포함된다.
- **consumed (보조):** P+j 행. 생성된 j번째 토큰을 처리하며 다음 토큰을 예측하는 행.
  이 행의 답변 key에는 현재 처리하는 답변 토큰도 포함된다.

뷰어에서 두 관례를 선택할 수 있다. 정적 PDF와 요약은 predictive를 사용한다.
EOS는 제외한다. QVHighlights의 128-token 제한 사례 2개를 포함한다.
consumed 마지막 행은 이전 생성 실험에서 기존 KV cache에 마지막 토큰을 한 번 넣어 확보했다.

## 측정

모든 causal key를 포함한 softmax attention에서 영상 열을 골라 공간 토큰을 시간별로 합한다.
각 생성 토큰별로 레이어 8–21(0-based), 32개 head를 평균하며 토큰끼리 먼저 평균하지 않는다.
영상 시간은 실제 frame index/FPS timestamp를 사용한다. GT는 원래 구간을 그림에 보존하고,
질량 적분에만 영상 범위와의 교집합을 사용한다. time bin은 인접 timestamp의 중점으로 나눈다.

- raw video attention: 전체 key 중 영상에 할당된 확률 질량.
- raw GT attention: 원래 점수에 각 time bin의 GT 중첩 비율을 곱해 합한 값.
- video-conditional GT mass: 시간별 점수 합이 1이 되도록 정규화한 후 GT 중첩 비율로 적분한 값.
- peak: 가장 높은 time bin의 실제 timestamp. 동점은 첫 bin을 선택한다.
- centroid: 영상 내부 정규화 attention으로 가중 평균한 timestamp.
- entropy: 영상 내부 분포 Shannon entropy / log(time bin 개수).
- adjacent total variation: 0.5 × sum(abs(p[j]−p[j−1])). 첫 단계는 null/빈 CSV 셀.
- peak jump / centroid shift: 인접 단계 위치 차이의 절댓값, 초 단위.
- query distance: 각 단계 영상 분포와 입력 query 평균 영상 분포 사이의 TV.

`other_prefix`에는 system/instruction/special token/영상 timestamp 텍스트 등 영상 patch와 query 문장 이외의 입력 토큰이 포함된다.
답변 단계의 분모에는 생성된 이전 답변 key도 추가되므로 영상 attention 총량 감소를 곧 시각 정보의 인과적 사용 감소로 해석하지 않는다.

데이터셋 집계는 샘플에 동일 가중치를 준다. 인접 TV는 샘플 내 평균 후 샘플 간 중앙값이다.
첫→마지막 변화는 같은 샘플의 차이 후 중앙값이며 단조 추세 검정이 아니다.
숫자 그룹은 이전 MTLA 파서가 유효 구간에서 식별한 숫자 토큰을 그대로 사용한다.
Pearson correlation을 계산하지 않는다.

## 실행

`.venv`, `python_deps`는 이전 실험의 환경을 읽기 전용으로 재사용한다.
`browser_deps`, `browser_cache`, `browser_libs`는 오프라인 뷰어의 실제 브라우저 검증을 위한 이 실험 전용 도구다.

```bash
cd /home/irteam/experiments/mtla_answer_token_attention_dynamics_100
.venv/bin/python analyze_dynamics.py
.venv/bin/python render_dynamics.py
.venv/bin/python write_report.py
.venv/bin/python check_browser.py
.venv/bin/python verify_dynamics.py
```

각 샘플의 `dynamics.json`에 토큰별 원래 시간 분포와 두 관례의 지표를 저장한다.
원본 36-layer / 32-head 배열은 이전 실험의 `attention.npz`에 있으며 복제하지 않는다.
실험 결과와 소스의 SHA256은 `artifact_sha256.json`, 검증 결과는 `results/validation.json`에 저장한다.

## 벤치마크별 샘플 폴더

샘플은 `results/<dataset>/<sample_key>/`에 저장한다. 예: `results/Charades-STA/Charades-STA_01/`.
`Charades-STA`, `QVHighlights`, `ActivityNet-Captions` 폴더 아래에 각 샘플의 배열·그림·수치를 모았다. 전체 요약과 `index.html`은 `results/`에 있다.
