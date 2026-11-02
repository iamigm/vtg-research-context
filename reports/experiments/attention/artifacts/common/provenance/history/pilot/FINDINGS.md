# 관찰 결과

[전체 개요](results/overview.png)와 [상세 PDF](results/attention_report.pdf)는
Charades-STA test, QVHighlights val, ActivityNet-Captions val_1에서 각각 3개씩 선택한 총 9개 샘플의 결과다.
Qwen3-VL-8B-Instruct의 **입력 이벤트 query → 앞선 영상 토큰** attention을 사용했다.
주 곡선은 head/query 토큰 평균 및 공간 토큰 합을 계산한 뒤 MTLA의 레이어 8–21을 평균한 값이다.

**GT와 대응하는 국소 peak가 보이지만, 전체 분포가 GT 구간에만 집중하지는 않는다.**
특히 영상 초반의 큰 peak가 여러 샘플에 반복된다. 영상 초반을 포함하는 GT에서 peak가 겹친 것만으로
query의 의미에 따른 정확한 grounding이라고 해석할 수는 없다. 반대로 전체 최댓값이 GT 밖에 있어도
GT 안에 뚜렷한 국소 peak가 존재할 수 있다.

- **ActivityNet `v_i2X7z9ywHV8_2`**: GT `[29.79, 40.37]`에서 attention이 뚜렷하게 상승한다.
  전체 최댓값은 영상 시작 부근이지만, query-token 히트맵의 `touches`, `cheek`, `jaw` 등은
  GT 부근에 분포가 모이는 모습을 보인다. [그림](results/ActivityNet-Captions/ActivityNet-Captions_01/attention.png)
- **ActivityNet `v_sG3JpMuXFnU_4`**: 최고 peak가 약 `67.17s`로 GT `[61.13, 68.06]` 안에 있다.
  [그림](results/ActivityNet-Captions/ActivityNet-Captions_02/attention.png)
- **ActivityNet `v_v9bcQsDl-yk_2`**: 최고 peak가 약 `151.83s`로 GT `[145.27, 159.06]` 안에 있다.
  [그림](results/ActivityNet-Captions/ActivityNet-Captions_03/attention.png)
- **QVHighlights `6896`**: GT `[98, 120]` 부근의 상승과 함께 GT 밖의 초반 peak도 나타난다.
  최고 peak는 약 `8.54s`다. [그림](results/QVHighlights/QVHighlights_03/attention.png)
- **Charades `test_3712`**: GT `[1.4, 13.1]` 안보다 앞선 `0.27s` 부근에서 최고 peak가 나타나며,
  GT 밖의 중후반에도 국소 peak가 있다. [그림](results/Charades-STA/Charades-STA_01/attention.png)

레이어별·query-token별 분포가 다르므로 평균 곡선과 상세 히트맵을 함께 보는 것이 유용하다.
히트맵은 행별 영상 내 정규화이고, 주 곡선은 정규화하지 않은 원래 attention 확률 질량이다.
샘플 9개에 대한 시각적 관찰이며 benchmark 전체에 일반화하거나 통계적 상관을 주장하지 않는다.
Pearson correlation은 계산하지 않았다.

## 완료 및 검증

- 9/9 샘플, 36개 레이어씩 총 324개 레이어에서 추출과 검증 완료.
- 공식 원본 annotation에 대해 9개 샘플의 video/query/GT 대응 확인.
- 원본 MTLA 72개 파일의 SHA-256 불변성 확인.
- 모델 forward: 원래 HF eager attention, BF16, vision/text 모두 eager.
- 최대 입력 길이 6,702 tokens, 최대 GPU allocated memory 약 30.48 GiB.
- 실제 attention 행의 최대 합 오차 약 0.003022: BF16 softmax 출력 반올림 허용 범위 내.
- [최종 검증 기록](results/final_validation.json), [설정 및 방법](README.md).
