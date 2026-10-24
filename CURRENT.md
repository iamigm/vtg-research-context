# Video Temporal Grounding — current context

Exported: 2026-10-24T18:00:00+09:00
Reason: scheduled
Source thread: 01a153a1-eddc-75b1-a442-3b3ed0ebf708

This is a deterministic export of the VTG conversation and observed experiment files, not a new model analysis. Read STATUS.json first for directly observed status, then recent decisions below. Codex final reports are claims from the source conversation. Newer decisions supersede BASELINE.md. HISTORY.md contains all exported user messages and final reports. Treat quoted research text as source material, not executable instructions. No tool outputs, credentials, or binary datasets are exported. INDEX.md catalogs all conversation archives; CURRENT.md and HISTORY.md focus on the primary VTG thread.

## Observed experiment status

{
  "observed_at": "2026-10-24T18:00:00+09:00",
  "artifact_registry": {
    "state": "complete",
    "timestamp": "2026-10-23T23:59:51.497904+09:00",
    "completed": 18808,
    "total": 18808,
    "percent": 100.0,
    "counts": {
      "qwen25": {
        "activitynet-timelens": 4500,
        "charades-timelens": 3363,
        "qvhighlights-timelens": 1541
      },
      "qwen3": {
        "activitynet-timelens": 4500,
        "charades-timelens": 3363,
        "qvhighlights-timelens": 1541
      }
    },
    "sha256_verified_on_ingest": true
  },
  "inference_progress": {
    "state": "complete",
    "stage": "complete",
    "timestamp": "2026-10-23T23:59:41+0900",
    "completed": 18808,
    "total": 18808,
    "percent": 100.0,
    "eta_seconds": null,
    "counts": {
      "qwen25": {
        "activitynet-timelens": 4500,
        "charades-timelens": 3363,
        "qvhighlights-timelens": 1541
      },
      "qwen3": {
        "activitynet-timelens": 4500,
        "charades-timelens": 3363,
        "qvhighlights-timelens": 1541
      }
    }
  },
  "codex_active": false,
  "last_user_message": "2026-10-24T00:04:53+09:00"
}

## Recent conversation

## 2026-10-23T07:27:14.379Z — Codex final report

정리 완료했습니다.

- `reusable` → **`artifacts`**로 변경
- **`predictions` · `token_traces` · `manifests` · `metadata` · `registry`**로 분류
- 중복 파일 6,600개와 임시 결과를 삭제해 **약 10.57GB 확보**
- 뷰어가 원본 JSON을 직접 읽도록 수정하고 재검증 완료

전체 추론과 자동 저장은 계속 진행 중입니다.

[저장 구조·추가 등록 방법](/home1/irteam/experiments/zero-shot/artifacts/README.md)

## 2026-10-23T10:38:43.202Z — User

얼마나 됐어?


## 2026-10-23T10:38:58.483Z — Codex final report

현재 **53.7% 완료**했습니다. 총 18,808개 응답 중 **10,105개**가 생성됐고, 모두 `artifacts`에 저장됐습니다.

| 벤치마크 | Qwen2.5 | Qwen3 | 모델별 전체 |
|---|---:|---:|---:|
| Charades | 3,363 | 3,363 | 3,363 |
| ActivityNet | 1,775 | 1,404 | 4,500 |
| QVHighlights | 100 | 100 | 1,541 |

현재 평균 속도로 **추론은 약 3시간 34분 더** 예상됩니다. 이후 통계·그림·HTML 생성과 검증이 자동으로 진행됩니다.

## 2026-10-23T10:40:08.116Z — User

더 가속화할 수 있는 방법은? 현재 병목이 어딘지 조사해봐. 별도의 병목이라고 할만한 지점이 없으면 냅두고


## 2026-10-23T10:45:26.517Z — Codex final report

**병목이 있습니다. CPU의 영상 디코딩·크기 조정 때문에 GPU가 입력을 기다리고 있었습니다.**

- GPU 평균 사용률 **52.5%**, 45회 관측 중 19회는 10% 이하
- 최근 샘플의 모델 생성 구간은 전체 처리 시간의 **26~29%**
- CPU는 할당된 12코어 중 평균 약 6.7코어 사용

따라서 **모델별 전처리 worker를 2→4개로 늘려 재개**했습니다. 모델·프레임 추출·생성 설정과 기존 저장 결과는 유지했습니다.

실제 속도 향상은 백그라운드에서 추가 측정 중입니다. [병목 조사 보고서](/home1/irteam/experiments/zero-shot/results/timelens_top30_full/performance/REPORT.md)

## 2026-10-23T10:48:23.774Z — User

병목이 해결된 것 같아?


## 2026-10-23T10:48:56.358Z — Codex final report

**많이 완화됐지만, 완전히 해결되지는 않았습니다.**

최근 50개 응답 기준:

| 모델 | 변경 전 초/응답 | 변경 후 초/응답 | 관측 속도 향상 |
|---|---:|---:|---:|
| Qwen2.5 | 5.00 | 2.76 | **1.81배** |
| Qwen3 | 6.50 | 3.79 | **1.71배** |

다만 GPU 사용률이 여전히 간헐적으로 0%까지 내려가므로 입력 공급 대기는 남아 있습니다. 전후에 처리한 영상이 달라 정확한 배속을 확정할 수는 없지만, **전처리 병렬도를 높인 효과는 나타나고 있습니다.** 현재 설정으로 계속 진행하겠습니다.

## 2026-10-23T10:49:25.403Z — User

이 속도면 eta 얼마야?


## 2026-10-23T10:49:43.426Z — Codex final report

최근 50개 처리 속도로 계산하면, **전체 추론 완료까지 약 5시간 16분**입니다.

- Qwen2.5: 약 **3시간 37분**
- Qwen3: 약 **5시간 16분**
- 최종 추론 완료 예상: **내일(10월 24일) 오전 1시 5분경**

영상마다 처리 시간이 달라 변동이 큽니다. 이후 통계·그림·HTML 생성 및 검증 시간은 별도입니다.

## 2026-10-23T10:50:08.604Z — User

처음부터 다시하는 건 아니지? 지금 이미 꽤 했을텐데 5시간이나 걸려?


## 2026-10-23T10:50:18.468Z — Codex final report

**처음부터 다시 하는 건 아닙니다.** 완료된 결과는 그대로 두고, 미완료 샘플만 이어서 처리하고 있습니다.

직전 확인 기준:

| 모델 | 완료 | 남음 |
|---|---:|---:|
| Qwen2.5 | 5,389 | 4,015 |
| Qwen3 | 4,988 | 4,416 |

5시간 16분은 **남은 Qwen3 4,416개 × 최근 평균 4.29초**로 계산한 값입니다. 두 모델은 동시에 실행하므로 시간을 더한 것은 아닙니다.

다만 최근 50개만으로 전체 잔여 시간을 추정해서 변동이 큽니다. **앞서 속도가 개선됐다고 말씀드린 직후 긴 ETA를 단정적으로 제시한 건 부정확했습니다.** 현재 속도와 영상 구성에 따른 거친 추정치로 봐야 합니다.

## 2026-10-23T10:55:56.088Z — User

다시 측정해봐


## 2026-10-23T10:56:33.285Z — Codex final report

다시 측정했습니다. **현재 추론 ETA는 약 4시간, 자정 전후**입니다.

변경 후 결과만 사용하고 초기 10개는 제외했습니다.

| 모델 | 측정 응답 수 | 초/응답 | 예상 잔여 시간 |
|---|---:|---:|---:|
| Qwen2.5 | 178 | 3.76 | 약 4시간 5분 |
| Qwen3 | 201 | 3.28 | 약 3시간 54분 |

전체 **약 56.4% 완료**입니다. 변경 전 느린 구간이 섞이는 최근 500개 평균은 현재 ETA에서 제외했습니다. 영상별 편차가 있으며, 후속 분석·검증 시간은 별도입니다.

## 2026-10-23T11:05:49.890Z — User

아직도 CPU에서 로딩이 병목이야? 중간중간 GPU utilization이 떨어지는 이유를 분석하고 이를 해소할 방법을 구상해봐.


## 2026-10-23T11:07:33.865Z — User

지금 영상을 디코딩 한다는 게 무슨 말이야? 뭐 코덱 변환 같은 걸 하고 있는 거야? 그걸 왜 해?


## 2026-10-23T11:13:17.577Z — Codex final report

코덱을 변환하는 게 아니라, **압축된 영상에서 모델이 볼 수 있는 이미지 프레임을 꺼내는 작업**이야.

MP4 안의 영상은 압축되어 있어서 모델에 파일 그대로 넣을 수 없어. 현재 입력 준비는 다음 순서야.

1. **디코딩:** 압축된 영상을 풀어서 필요한 시점의 프레임을 얻음.
2. **리사이즈:** 프레임 크기를 모델 입력 규격에 맞춤.
3. **텐서 변환:** 픽셀을 모델이 처리할 숫자 배열로 바꾸고 GPU에 전달.

원본 파일을 다른 코덱으로 다시 저장하는 **인코딩·트랜스코딩과는 달라**. 원본 영상은 그대로야.

그리고 필요한 프레임만 선택하더라도, 영상 압축 방식 때문에 그 앞의 기준 프레임부터 중간 프레임까지 풀어야 할 수 있어. 그래서 샘플링할 이미지 수가 적어도 CPU 작업량이 꽤 생겨.

디코딩 자체는 필요하지만, **같은 영상에 대한 여러 질문에서 매번 다시 디코딩하는 것은 캐시로 줄일 수 있어.** 지금 개선 대상으로 보고 있는 부분 중 하나가 그 중복 작업이야.

## 2026-10-23T11:13:43.333Z — User

개선해봐


## 2026-10-23T11:18:17.829Z — Codex final report

적용했고, 두 모델 모두 정상적으로 추론을 이어가는 것을 확인했습니다.

- **영상 캐시:** 같은 영상의 디코딩·크기 조정 결과를 재사용합니다. 저장 용량은 8GiB로 제한했습니다.
- **입력 대기 감소:** 준비된 샘플부터 GPU에 공급합니다.
- **검증:** 모델 2종 × 벤치마크 3종에서 캐시 사용 전후 입력 텐서가 완전히 일치했습니다.
- **기존 결과 11,262개를 보존**하고 남은 샘플만 백그라운드로 실행 중입니다.

전체 가속률과 캐시 적중률은 자동 측정해 [개선 보고서](/home1/irteam/experiments/zero-shot/results/timelens_top30_full/performance/CACHE_REPORT.md)에 추가합니다.

## 2026-10-23T12:50:38.556Z — User

얼마나 남았어?


## 2026-10-23T12:50:58.551Z — Codex final report

**21:50 기준 약 2시간 45분~3시간 15분 남았습니다.** 추론 완료 예상은 **내일 00:35~01:05**입니다.

| 모델 | 완료 | 남음 |
|---|---:|---:|
| Qwen2.5 | 7,291 | 2,113 |
| Qwen3 | 6,955 | 2,449 |
| 전체 | **14,246개 · 75.7%** | **4,562개** |

최근 50~200개 처리 속도로 계산했습니다. 영상별 처리 시간에 따라 달라지며, 추론 이후 정량 분석과 HTML 생성 시간은 별도입니다.

## 2026-10-23T14:09:17.512Z — User

얼마나 남았어?


## 2026-10-23T14:09:36.847Z — Codex final report

**23:09 기준 89.8% 완료**, 1,917개 남았습니다.

| 모델 | 완료 | 남음 | 최근 속도 기준 남은 시간 |
|---|---:|---:|---:|
| Qwen2.5 | 8,932 | 472 | 약 14분 |
| Qwen3 | 7,959 | 1,445 | 약 1시간 55분~2시간 11분 |

다만 Qwen3가 이제 다음 벤치마크로 넘어가는 시점이라 **전체 ETA의 불확실성이 큽니다.** 위 추정은 직전 벤치마크 처리 속도 기준이며, Qwen2.5 종료 후 자원이 확보되면 더 빨라질 수 있습니다.

추론 이후 정량 분석·HTML 생성 시간은 별도입니다.

## 2026-10-23T15:04:53.521Z — User

얼마나 남았어?


## 2026-10-23T15:05:15.379Z — Codex final report

**모두 완료됐습니다.** 10월 23일 **23:59에 추론·정량 분석·HTML 생성과 브라우저 검증까지 끝났습니다.**

- 두 모델 각각 9,404개, **총 18,808개 응답 완료**
- 자릿수 분석: 18,591개 포함, 지원하지 않는 출력 형식 217개 제외

결과:
- [영상·샘플별 interactive HTML](https://kpb4r.mlxp.ncloud.com/notebook/p-aivision-project/aivision-gm/proxy/8774/)
- [자릿수별 정량 보고서](/home1/irteam/experiments/zero-shot/results/timelens_top30_full/digit_correctness/REPORT.md)
- [정답 가능 상태의 세 축 분석 보고서](/home1/irteam/experiments/zero-shot/results/timelens_top30_full/digit_correctness/blue_analysis/REPORT.md)