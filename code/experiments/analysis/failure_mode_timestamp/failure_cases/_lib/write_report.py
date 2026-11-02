"""Report only Qwen2.5 native/converted and Qwen3 native results."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
from urllib.parse import quote
import json
ROOT=relocate_path(Path(__file__)).resolve().parents[1];DEST=relocate_path(ROOT/'reports/pilot_100')
def main():
 settings=json.loads((relocate_path(DEST/'setting_summary.json')).read_text());comparison=json.loads((relocate_path(DEST/'baseline_comparison.json')).read_text())
 assert settings['complete'] and comparison['complete']
 lines=['# 세 세팅의 zero-shot failure 분석','', '**Qwen2.5 기본 / Qwen2.5 + TimeLens-7B식 입력 / Qwen3 기본**을 비교한다. 데이터셋별 같은 100 query이며 전수평가는 아니다. Qwen3의 별도 입력 변환 실험은 삭제했다.','', '## 정확한 입력 세팅','', '| 항목 | Qwen2.5 기본 | Qwen2.5 + TimeLens-7B식 입력 | Qwen3 기본 |','|---|---|---|---|','| 시각 블록 | 서로 다른 두 프레임 | 선택한 한 프레임 복제 | 서로 다른 두 프레임 |','| 서로 다른 프레임 빈도 | 2 FPS | 약 1 FPS | 2 FPS |','| 절대 시간 표현 | native temporal MRoPE | `12.3s:` 텍스트 | `<12.3 seconds>` 텍스트 |','| 텍스트 timestamp 기준 | 해당 없음 | 선택한 프레임 시간 | 두 프레임 시간의 평균 |','| modality | video | 독립 image | video |','| 추가 timestamp 설명 | 없음 | 있음 | 없음 |','', '모두 원래 Instruct 가중치이며 추가 학습은 없다. TimeLens 학습 모델을 평가한 결과가 아니다. Qwen3 native에는 이미 textual timestamp가 있으며 공식 TimeLens-8B 입력 경로와 같은 방식이다. **Qwen2.5 변환은 프레임 복제·modality·시간 표현·지시문을 함께 바꾸므로 단일 요인의 효과가 아니다.**','', '[프롬프트 전문·전처리·공통 조건](../../README.md).','', '## 성능','', '| 세팅 | 데이터셋 | n | R@1@0.3 | R@1@0.5 | R@1@0.7 | mIoU |','|---|---|---:|---:|---:|---:|---:|']
 for item in settings['settings']:
  for ds in ['charades-timelens','qvhighlights-timelens']:
   g=settings['groups'][item['key']+'/'+ds];lines.append(f'| {item["label"]} | {ds} | {g["n"]} | {g["R1@0.3"]:.1f} | {g["R1@0.5"]:.1f} | {g["R1@0.7"]:.1f} | {g["mIoU"]:.1f} |')
 lines+=['', '단위 %. 공식 parser의 첫 구간과 정수 초 반올림 규칙을 적용했다.','', '## Qwen2.5 입력 변환의 영향','', '| 데이터셋 | 실패→성공 | 성공→실패 | R@1@0.5 변화 | 95% paired bootstrap 구간 | 무겹침 실패 변화 |','|---|---:|---:|---:|---|---|']
 for key,g in comparison['groups'].items():
  c=g['outcomes'];ci=g['paired_cluster_bootstrap_95pct_R1_delta_pp'];lines.append(f'| {key.split("/")[1]} | {c.get("recovered",0)} | {c.get("regressed",0)} | {g["delta_R1_05_pp"]:+.1f}%p | [{ci[0]:+.1f}, {ci[1]:+.1f}]%p | {g["baseline"]["failure_counts"].get("no_overlap",0)} → {g["timestamp"]["failure_counts"].get("no_overlap",0)} |')
 lines+=['', '변환 뒤 Qwen2.5의 잘못된 시점 선택과 영상 범위 초과가 늘었다. 입력만 바꾸면 성능이 개선된다는 근거가 아니다. Bootstrap은 영상 단위 2,000회이며 표본 내 탐색적 추정이다. Qwen3 native는 별도 기준 모델로 함께 표시하며 Qwen2.5 변환 효과의 paired 계산에는 포함하지 않는다.','', '[Qwen2.5 paired 결과](baseline_paired_cases.csv) · [실패 유형 전환](failure_transitions.png) · [GT 길이별 분석](stratified_analysis.json) · [프레임 보존 점검](sampling_analysis.json).','', '## 대표 영상 검토','', '[동영상 갤러리](cases/report.html)는 GT, Qwen2.5 기본, Qwen2.5 TimeLens-7B식 입력, Qwen3 기본의 네 타임라인을 보여준다. 모델 예측은 세 개다.']
 for r in json.loads((relocate_path(DEST/'manual_review.json')).read_text()):
  key=r['sample_id'].replace('/','__');lines+=['',f'### {r["sample_id"]}','',r['observation'],'','해석 후보: '+r['hypothesis'],'','검토 한계: '+r['limitation'],'',f'[동영상 재생](cases/report.html#{quote(r["sample_id"],safe="")}) · [프레임 요약](cases/{key}.jpg) · [metadata](cases/{key}.json)']
  if r.get('detail_timestamps'):lines+=['',f'[GT·예측 주변 확대](cases/{key}_detail.jpg)']
 lines+=['', '## 검증과 산출물','', '- Qwen2.5 변환 200개에서 baseline과 시각 토큰 수·공간 grid·temporal group 수 및 실제 frame index/FPS 정합성을 검증했다. [결과](paired_input_validation.json).','- 공식 evaluator와 Qwen2.5 변환 결과의 IoU 차이는 0이다. [결과](validation.json). 기존 두 native 모델의 검증은 baseline 디렉터리에 보존되어 있다.','- 기본 입력 두 모델 및 Qwen2.5 변환의 원시 예측은 수정하지 않았다. [보존 hash와 삭제 내역](../../configs/removed_qwen3_variant.json).','', '[세 세팅 전체 query 표](report.html) · [세 세팅 CSV](setting_case_metrics.csv) · [성능 비교 그림](summary.png) · [입력 세팅 전문](../../README.md).']
 write_result_text((relocate_path(DEST/'REPORT.md')), '\n'.join(lines)+'\n')
if __name__=='__main__':main()
