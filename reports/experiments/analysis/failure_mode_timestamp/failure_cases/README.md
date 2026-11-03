# 세 가지 zero-shot 입력 세팅

남긴 실험은 **Qwen2.5 기본 / Qwen2.5 + TimeLens-7B식 입력 / Qwen3 기본**이다. Qwen3의 별도 변환 실험은 사용자 요청으로 결과·통계·갤러리에서 삭제했다. Qwen3 기본 입력은 이미 textual timestamp를 포함한다.

## 공통 조건

- Charades-TimeLens 100 query, QVHighlights-TimeLens 100 query. 기존 baseline과 같은 manifest, seed=20261007. 전수평가는 아니다.
- 원래 Instruct 가중치 그대로: Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct. TimeLens 학습 가중치는 사용하지 않았다.
- 영상 decoding 2 FPS, min_tokens=64, total_tokens=14336. 세팅 내 각 query의 실제 frame index와 FPS를 사용한다.
- BF16, SDPA, Transformers 4.57.6, greedy generation, max_new_tokens=512, seed=42.
- 하나의 user message에 시각 입력 다음 grounding query를 넣고 한 번에 prefill한다. `<prefill>` / `<query>`라는 별도 토큰이나 영상 전용 KV cache 재사용은 없다.
- Qwen2.5의 기본 system message는 `You are a helpful assistant.`이며 Qwen3에는 별도 system message를 추가하지 않았다.
- 공식 parser의 첫 구간과 Python round에 따른 정수 초를 평가한다. GT와 예측 시간을 사후 보정하지 않는다.

## 1. Qwen2.5 기본

2 FPS로 읽은 서로 다른 두 프레임 `[f0,f1]`, `[f2,f3]`, …를 temporal patch embedding에서 합친다. native video 입력(`pixel_values_videos`, `video_grid_thw`, `second_per_grid_ts`)을 사용한다. 절대 시간 정보는 native temporal MRoPE 위치 좌표로 전달하고, 프레임 블록 앞에 textual timestamp를 별도로 추가하지 않는다.

## 2. Qwen2.5 + TimeLens-7B식 입력

공개 TimeLens-7B processor 구현을 **Qwen2.5 원래 tokenizer·pixel processor·가중치**와 함께 사용한다.

1. baseline과 같은 2 FPS 디코딩 결과 `[f0,f1,f2,f3,…]`를 `[f0,f0,f2,f2,…]`로 바꾼다. 서로 다른 프레임은 약 1 FPS이며, 새로 정확한 1초 간격 sampling을 하는 것은 아니다.
2. 각 복제 쌍을 독립 이미지로 처리한다. `pixel_values`, `image_grid_thw=[[1,h,w],…]`, image token을 사용한다.
3. 각 블록 앞에 선택한 프레임의 `원본 frame index / 원본 FPS`를 소수 한 자리로 표시한다: `12.3s: <image tokens>`.
4. 원래 video 시간 좌표(`second_per_grid_ts`)는 전달하지 않는다. 공간 위치 인코딩은 유지한다.
5. grounding prompt 앞에 아래 timestamp 설명을 추가한다.

```text
You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds).
```

시각 블록 수·공간 grid·시각 토큰 수는 Qwen2.5 baseline과 200개 전체에서 동일하다. 프레임 내용과 텍스트 토큰 수는 달라진다. **프레임 선택/복제, image modality, 시간 표현, 설명 지시문을 함께 바꾼 실험**이다.

## 3. Qwen3 기본

2 FPS의 서로 다른 두 프레임을 temporal patch embedding에서 합친다. 기본 `Qwen3VLProcessor`가 각 시각 블록 앞에 `<12.3 seconds>` 형식의 textual timestamp를 자동으로 넣는다. 대표 시간은 해당 두 프레임의 `(t0+t1)/2`이며 소수 한 자리로 표시한다.

```text
<timestamp of pair 0 seconds> <vision_start> [video tokens for f0,f1] <vision_end>
<timestamp of pair 1 seconds> <vision_start> [video tokens for f2,f3] <vision_end>
...
[grounding query]
```

native video modality와 frame pair 정보를 유지한다. Qwen3의 LLM 위치 좌표 구성은 기본적으로 timestamp별 t=1 그룹을 사용한다. 따라서 Qwen2.5의 절대 시간 MRoPE 방식과 구분해야 하며, Qwen3 기본이 timestamp 없는 입력이라고 부르면 안 된다. 공개 TimeLens-8B evaluator도 이 native Qwen3 입력 경로를 사용한다.

## 공통 grounding query

```text
Please find the visual event described by the sentence '{query}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'.
```

TimeLens-7B식 Qwen2.5 입력만 이 문장 앞에 timestamp 설명 문구를 추가한다. 영상 블록은 이 전체 query text보다 앞에 온다.

## 결과와 검토

- [세 세팅 결과 보고서](results/failure_modes/input_settings/pilot/200/REPORT.md)
- [실제 영상과 GT/예측 구간 비교](artifacts/runs/failure_modes/input_settings/pilot/200/data/cases/report.html)
- [세 세팅 전체 query 표](results/failure_modes/input_settings/viewer.html?view=pilot%2F200%2Freport.html), [CSV](artifacts/runs/failure_modes/input_settings/pilot/200/metrics/setting_case_metrics.csv)
- [Qwen2.5 입력 변환의 paired 비교](artifacts/runs/failure_modes/input_settings/pilot/200/metadata/baseline_comparison.json)
- [입력 예산/시간 검증](artifacts/runs/failure_modes/input_settings/pilot/200/validation/paired_input_validation.json), [공식 evaluator 대조](artifacts/runs/failure_modes/input_settings/pilot/200/validation/validation.json)

기존 기본 입력 결과: `/home/irteam/analysis/failure_mode_baseline/failure_cases/results/pilot_100/{qwen25,qwen3}`.
Qwen2.5 변환 결과: `artifacts/runs/failure_modes/input_settings/pilot/200/predictions/qwen25`. Qwen3 변환 결과 디렉터리는 삭제했다.

## 영상 갤러리 접속

공통 [실험 목록](../../../index.html)의 실패 분석에서 연다. 통합 서버8780 하나를 사용한다. [pilot200 결과](results/failure_modes/input_settings/viewer.html?view=pilot%2F200%2Freport.html)는 Charades100·QVHighlights100 질문이다. 과거 run_id pilot_100은 데이터셋별 개수이며 보존된 식별자다.

## 재현

`bash _lib/run_all.sh`는 **Qwen2.5 입력 변환만** 추론한다. `bash _lib/finish.sh`는 추론 없이 현재 세 세팅의 분석/갤러리를 다시 생성한다. 기본 입력의 원시 예측은 수정하지 않는다.

`timestamp_inputs.py`는 남겨둔 Qwen2.5 추론 당시 source hash를 유지하기 위해 바이트 그대로 보관한다. 과거 Qwen3 분기는 파일에 남아 있지만 실행 CLI와 검증 entry point는 Qwen2.5만 허용하며 해당 분기를 사용하지 않는다. 삭제 내역과 보존한 예측의 hash는 `artifacts/common/configs/removed_qwen3_variant.json`에 기록했다.

## 구현 출처

- [TimeLens-7B processor](https://huggingface.co/TencentARC/TimeLens-7B/blob/1740e2694669a4d03549454ed4af1cd740769f4f/processing_timelens.py)
- [공식 TimeLens evaluator](https://github.com/TencentARC/TimeLens/blob/5d2110fb3faf74f4b0fc5957cd7ab73bea4763dc/evaluation/utils.py)
- [Qwen3 processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- [TimeLens 논문 B.2](https://arxiv.org/html/2512.14698v1#A2.SS2)
