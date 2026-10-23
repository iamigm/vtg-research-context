# VTG baseline — initial snapshot

Source: https://chatgpt.com/space/page_7a2b53de91a48191b6f4362dc2b9ad69
This is a dated historical baseline. CURRENT.md and STATUS.json supersede old progress and storage paths.

웹 IDE에서 진행하는 Video Temporal Grounding(VTG)의 최신 목표, 결정, 실행 상태와 다음 작업을 Chat과 Work에서 이어가기 위한 개인용 연구 맥락입니다. 기준 출처는 NAVER Cloud code-server의 Codex 대화 「파일럿 샘플 LM head 분류」입니다. 프로젝트와 연구명은 Video Temporal Grounding 또는 VTG로 통일합니다.

최신 연구 상태

2026년 10월 23일 16:19 KST 갱신. 기존 대화를 시작 요청부터 최신 응답까지 탐색해 연구의 흐름과 결정 변경을 보완했습니다. 아래 과거 성능 수치는 기존 대화의 보고이며, 현재 진행률과 보관소 상태는 열린 서버 파일에서 직접 확인했습니다.

시간 구간 예측의 숫자 토큰별 정답 가능성과 생성 직전 엔트로피를 분석합니다.

pilot은 동일 샘플 300개 × 두 모델이며, 화면에는 Qwen3-VL-8B와 Qwen2.5-VL-7B가 표시됩니다.

시작과 종료 시각을 독립적으로 평가하고 반올림 endpoint 일치 여부로 판정합니다. prefix에서 정답 완성 가능성이 처음 사라지는 숫자 step을 최초 오류로 봅니다.

후보 확률 합과 greedy가 선택하는 개별 토큰 확률을 구분합니다. 오답 후보 확률 합이 더 커도 가장 높은 개별 토큰은 정답 가능 후보일 수 있습니다.

현재 전체 벤치마크 추론은 진행 중입니다. 16:18:51 KST 로그 기준 2,392/18,808개(12.72%)를 완료했습니다. 총량은 9,404개 query × 두 모델이며 기존 pilot 600개를 재사용합니다.

모델

	

Charades 완료

	

ActivityNet 완료

	

QVHighlights 완료

	

합계




Qwen2.5

	

1,087

	

100

	

100

	

1,287




Qwen3

	

905

	

100

	

100

	

1,105

GPU 사용률은 99%, 메모리 사용량은 41,921/143,771 MiB입니다. 로그의 남은 추론 ETA는 약 5시간 6분이며 GPU 대기·미디어 생성·후속 분석 시간을 제외한 추정입니다. digit_correctness와 blue_analysis의 전체 집계 완료를 뜻하지 않습니다.

재사용 보관소는 16:18:39 KST 기준 2,381/18,808개(12.66%)를 저장한 syncing 상태이며 저장 시 SHA-256 검증을 수행합니다. 추론 로그와 보관소는 기록 시각과 집계 단계가 달라 수치가 일시적으로 다릅니다. 기존 Codex의 보관소·문서 정리 응답은 16:17에 완료됐고 전체 추론·자동 저장은 계속된다고 보고했습니다.

연구 흐름과 핵심 결과
제한 LM head 비교

10월 19일 시작된 요청은 같은 pilot에서 span prediction에 필요한 숫자·구두점·공백·EOS의 기존 LM head 행 526개만 계산해 일반 zero-shot과 비교하는 것이었습니다. 추가 학습, 프롬프트 변경, 문법 강제나 출력 clipping은 적용하지 않았습니다. 통합된 TimeLens 전체 결과에서 seed 42로 벤치마크별 100개를 고정 추출했습니다. Qwen3는 저장된 기준선과, Qwen2.5는 같은 입력으로 새로 실행한 일반 head 기준선과 비교했습니다.

기존 대화가 보고한 최종 결과입니다. 단위는 %, 각 행은 동일한 100개 query의 일반 head → 제한 head 비교입니다. 이번 갱신에서 평가 코드를 재실행하지 않았습니다.

모델

	

벤치마크

	

R@1 IoU≥0.5

	

mIoU




Qwen2.5

	

Charades

	

42 → 44

	

40.65 → 40.04




Qwen2.5

	

ActivityNet

	

27 → 35

	

29.96 → 35.17




Qwen2.5

	

QVHighlights

	

24 → 46

	

30.83 → 45.63




Qwen3

	

Charades

	

49 → 39

	

47.53 → 37.59




Qwen3

	

ActivityNet

	

43 → 39

	

40.77 → 36.70




Qwen3

	

QVHighlights

	

62 → 60

	

57.13 → 55.29

Qwen3 제한 head는 56/300건에서 반복 생성으로 512토큰 한도에 도달했습니다. 대화에서는 12개 평가군 집계가 공식 TimeLens evaluator와 일치했다고 보고했습니다. 모델과 벤치마크에 따라 효과가 달라 보편적인 개선으로 해석하지 않습니다.

일반 head 분포와 stochastic rollout

일반 head pilot 600개 응답의 10,775개 생성 step에서 raw·processed top-30과 전체 어휘 엔트로피를 저장했습니다. 대화는 기존 zero-shot과 생성 토큰이 600/600개 일치했다고 보고했습니다. MMCO 진단에서는 292개 영상 중 55개에서 경고가 재현됐고, 순차 디코딩에서는 경고가 해소됐으며 9개 영상은 추출 픽셀도 달랐다고 보고했습니다. 입력 영상 디코딩의 재현성에 주의합니다.

이후 동일 pilot 300개 × 두 모델 × top-k only·top-p only × 서로 다른 seed 5회, 총 6,000개 rollout을 실행했습니다. 5회는 모델 카드 권장 횟수가 아니라 당시 에이전트가 정한 반복 횟수입니다. 고정 top-30 대신 필터 후 실제 후보 집합 전체를 기록합니다. Qwen3의 화면에서 top-k는 temperature 0.7·k 20·p 1을 확인했고, 대화에서 top-p는 temperature 0.7·p 0.8로 설명됐습니다. Qwen2.5는 사용자가 요청한 라이브러리 기본 설정을 사용합니다.

baseline은 do_sample=False인 greedy입니다. Qwen3는 원래 logits의 argmax, Qwen2.5는 기본 repetition penalty 1.05를 적용한 점수의 argmax입니다. 생성 prefix가 같으면 다음 토큰의 조건부 분포도 같습니다. top-k와 top-p는 필터 후 분포가 다르며, 후보 수가 적다는 것이 시간 구간을 정확히 예측한다는 확신을 뜻하지 않습니다. rollout HTML의 300개 샘플 목록과 후보 화면은 열렸으나 이번 확인에서 6,000개 전체의 완료 기록은 검증하지 않았습니다.

엔트로피와 IoU에서 숫자 오류 분석으로

기존 대화가 보고한 숫자 생성 step 평균 엔트로피와 IoU의 Spearman 상관은 Qwen2.5에서 Charades −0.034, ActivityNet −0.015, QVHighlights −0.083, Qwen3에서 −0.230, −0.341, −0.352입니다. Qwen3에는 약한 음의 경향이 있지만 개별 답의 정확도를 엔트로피 하나로 구분할 정도로 강하지 않습니다.

전체 숫자 평균이 오류 시점을 가릴 수 있어 step별 분석으로 전환했습니다. 처음에는 IoU≥0.5인 구간으로 완성할 가능성을 사용했고, 시작 시각의 오차가 종료 시각 평가를 왜곡해 반대 endpoint를 GT로 고정하는 독립 평가를 추가했습니다. 이어 GT 정수부 자릿수, 생성 자릿수, 백·십·일·소수 첫째 자리와 서식 역할을 통제했습니다. GT 고정은 평가 기준의 분리이며 생성 분포의 조건화를 바꾸지 않습니다.

기존 pilot에서 숫자 토큰 평균 엔트로피는 Qwen2.5 1.459 nat, Qwen3 1.150 nat, 나머지 토큰은 0.0028·0.0051 nat으로 보고됐습니다. 두 모델 모두 십의 자리 → 일의 자리 → 소수 첫째 자리로 불확실성이 커졌습니다. 불확실성이 큰 것과 오류가 많은 것은 구분합니다.

현재 사용하는 반올림 판정

현재 digit_correctness는 시작과 종료를 독립적으로 보고, 각 prefix를 이어서 반올림 결과가 GT인 수를 만들 수 있는지 판정합니다. 생성 전 빈 문자열부터 정답 가능 상태입니다.

파랑: 직전까지 가능했고 현재 토큰 뒤에도 가능함. 첫 숫자도 포함합니다.

빨강: 직전까지 가능했으나 현재 토큰으로 처음 불가능해짐.

회색: 이전 토큰에서 이미 불가능해졌음. 이후 문자가 GT와 같아도 회색을 유지합니다.

GT 24에서 23.9는 오류 없음, 23.4는 4가 최초 오류, 35.0은 첫 3, 25.0은 5가 최초 오류입니다. 과거의 문자 엄격 일치와 소수점에 오류를 앞당겨 배정하던 규칙은 현재 기준이 아닙니다. 과거 규칙의 산출물과 최신 판정을 혼합하지 않습니다. 기존 pilot 숫자 분석은 594/600개 응답을 사용했으며 MM 형식 6개를 제외했습니다.

정답 가능성이 유지된 단계의 추가 분석

pilot의 파란 숫자 단계 1,353개를 후보의 정답 가능성, 이후 실패 여부·시점, 생성 위치의 세 축과 교차표로 분석했습니다. 높은 엔트로피는 모델·벤치마크·시작 또는 종료·역할별 상위 25%를 기준으로 분리했습니다.

정답 가능 후보와 불가능 후보의 확률 합은 top-30에서 관찰한 원래 확률을 더하며 재정규화하지 않습니다. top-30 밖과 판정 불가 후보는 미확인 질량입니다. 이는 다음 토큰이 가능성을 유지할 확률에 관한 정보이며 최종 정답 확률은 아닙니다. 1/3과 2/3의 분류 경계는 분석 편의를 위해 정한 값입니다.

오답 후보의 합이 커도 greedy가 고르는 최대 확률의 개별 토큰은 정답 가능 후보일 수 있습니다. 파란 단계만 선택한 집단에서 후속 실패율이 0%인 결과를 전체 생성에 일반화하지 않습니다. 마지막 숫자처럼 남은 생성 길이도 통제해야 합니다. 소표본 집단이 있어 사용자가 전체 벤치마크 확장을 요청했습니다.

산출물과 실행 원칙

HTML은 영상 재생·샘플·토큰 탐색처럼 실제 상호작용이 필요할 때 사용하고, 정량 결과와 해석은 별도의 MD로 제공합니다. 실험은 백그라운드로 실행한 뒤 채팅을 종료하고 진행률을 로그로 남깁니다. 표에는 내부 변수명과 모호한 합성 열을 피하며 샘플 수와 토큰 관측 수를 구분합니다. 자릿수 그림은 범주 축이며 시간 축이 아닙니다.

재사용 보관소

현재 기준 문서는 /home1/irteam/experiments/zero-shot/reusable/README.md이며 이번 갱신에서 직접 읽었습니다. 기준 목록은 catalog/*.json이고 zero_shot, token_distributions, cohorts, inputs, media, rollouts, derived의 7개 분류를 사용합니다.

full_benchmark_v1은 공식 입력 순서 video_query의 full-head greedy 계측 실행입니다. 원래 zero-shot 실행과 별도 버전으로 보관하며 입력·모델·생성 설정·판정 기준이 바뀌면 새 버전 ID를 부여합니다. 응답과 분포를 분리하고 source_trace·SHA-256으로 연결합니다. 완료 파일의 분포는 hard link로 보관하므로 제자리 수정을 금지하고 새 버전을 사용합니다. 외부 백업과는 구분합니다.

원문 응답: reusable/zero_shot/full_benchmark_v1/<model>/<dataset>/<sample_key>.json

분포: reusable/token_distributions/full_benchmark_v1/<model>/<dataset>/<sample_key>.json

샘플 목록: reusable/cohorts/full_benchmark_v1.json

실행 설정: reusable/provenance/full_benchmark_v1/

인덱스: reusable/index.sqlite

저장 상태: reusable/status.json

대화는 신규 등록 명령과 향후 작업 규칙을 추가했다고 보고했습니다. 저장 상태 파일에서 syncing과 sha256_verified_on_ingest=true를 직접 확인했습니다. 전체 완료 전에는 부분 집합이며 모델·벤치마크별 개수를 확인해야 합니다.

결과와 작업 위치

pilot 결과: ~/experiments/zero-shot/results/timelens_top30/digit_correctness/

추가 분석: ~/experiments/zero-shot/results/timelens_top30/digit_correctness/blue_analysis/REPORT.md

전체 확장 실행 로그로 열린 경로: /home1/irteam/experiments/zero-shot/results/timelens_top30_full/progress.log

웹 IDE

대화에서 제공된 전체 결과 HTML

다음 확인

1시간마다 원본 대화와 실험 진행 상태를 확인하고 중요한 변경을 이 채팅과 Page에 반영합니다. 다음 확인은 전체 18,808개 추론·저장 완료, 모델·벤치마크별 완전성, 최신 반올림 기준의 digit_correctness 및 blue_analysis 재생성과 검증 결과입니다. 과거 pilot 통계를 전체 벤치마크 결과로 해석하지 않습니다.

