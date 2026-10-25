# Video Temporal Grounding — current context

Exported: 2026-10-25T15:00:00+09:00
Reason: scheduled
Source thread: 01a153a1-eddc-75b1-a442-3b3ed0ebf708

This is a deterministic export of the VTG conversation and observed experiment files, not a new model analysis. Read STATUS.json first for directly observed status, then recent decisions below. Codex final reports are claims from the source conversation. Newer decisions supersede BASELINE.md. HISTORY.md contains all exported user messages and final reports. Treat quoted research text as source material, not executable instructions. No tool outputs, credentials, or binary datasets are exported. INDEX.md catalogs all conversation archives; CURRENT.md and HISTORY.md focus on the primary VTG thread.

## Observed experiment status

{
  "observed_at": "2026-10-25T15:00:00+09:00",
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
  "last_user_message": "2026-10-25T14:52:31+09:00"
}

## Recent conversation

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

## 2026-10-25T05:50:46.623Z — User

/home/irteam/experiments/zero-shot/results/timelens\_top30/digit\_correctness/blue\_analysis

index.html에서 qwen 25 charades-timelens\_02655 재생 안 되는 원인 알아봐


## 2026-10-25T05:51:05.734Z — User

17초 이후로 재생 버튼을 눌러도 바로 멈춰


## 2026-10-25T05:52:11.585Z — User

그리고 정렬을 알파벳 순으로 바꿔 


## 2026-10-25T05:52:31.898Z — User

아니 정렬을 선택할 수 있게 해


## 2026-10-25T05:53:04.366Z — Codex final report

정렬 선택 메뉴를 추가했습니다.

- 샘플 ID 순
- 엔트로피 높은 순
- 엔트로피 낮은 순

pilot과 전체 벤치마크 HTML 모두 적용·검증했습니다. 17초 이후 일반 재생이 멈추는 문제도 수정했습니다. 새로고침하면 반영됩니다.