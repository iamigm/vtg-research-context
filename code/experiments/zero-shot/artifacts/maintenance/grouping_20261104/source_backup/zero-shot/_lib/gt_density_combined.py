"""Combine both allocation experiments without treating reused outputs as independent."""
import sys
from pathlib import Path
import numpy as np
from adaptive_temporal_sampling import read,write,need,fileinfo,ROOT
from adaptive_temporal_report import csvwrite
from boundary_oracle_analysis import bootstrap_weights,estimate
RUN='gt_density_allocation_comparison_v1'
A='gt_extent_boundary_density_pilot300_v1';B='gt_actual_density_pilot300_v1'
def main():
    out=ROOT/'artifacts/runs'/RUN;report=ROOT/'results'/RUN;report.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'artifacts/runs'/r for r in [A,B]]
    for p in paths:need(read(p/'metadata/status.json')['state']=='complete','Both runs must complete')
    need(read(paths[0]/'metadata/cohort.json')==read(paths[1]/'metadata/cohort.json'),'Cohort mismatch')
    records=[read(p/'metrics/evaluation_records.json') for p in paths];summaries=[read(p/'metrics/summary.json') for p in paths]
    tables=[{(r['sample_key'],r['condition']):r for r in data} for data in records]
    combined=[]
    for dataset in sorted({r['dataset'] for r in records[0]})+['all']:
        samples=sorted([r for r in records[0] if r['condition']=='uniform' and (dataset=='all' or r['dataset']==dataset)],key=lambda r:r['sample_key']);keys=[r['sample_key'] for r in samples]
        weights=bootstrap_weights([r['dataset']+'/'+r['video_id'] for r in samples])
        for ratio in [2,4,8]:
            for who in ['gt','random']:
                conditions=[f'gt_boundary_r{ratio}'] if who=='gt' else [f'random_boundary_r{ratio}_s{s}' for s in [42,43]]
                values=[np.array([np.mean([t[k,c]['iou'] for c in conditions]) for k in keys]) for t in tables];diff=values[1]-values[0];e=estimate(diff,weights)
                row=dict(dataset=dataset,ratio=ratio,who=who,N=len(keys),delta_pp=e['mean']*100,lower_pp=e['ci95'][0]*100,upper_pp=e['ci95'][1]*100)
                for endpoint in ['start','end']:
                    def errors(t):
                        return np.array([np.mean([t[k,c][endpoint+'_error'] for c in conditions]) if all(t[k,c][endpoint+'_error'] is not None for c in conditions) else np.nan for k in keys])
                    d=errors(tables[0])-errors(tables[1]);row[endpoint+'_error_reduction']=estimate(d,weights) if np.isfinite(d).any() else None
                combined.append(row)
    write(out/'metrics/cross_design_comparisons.json',combined)
    metrics=[dict(design=label,**r) for label,summary in zip(['same_focus_count','same_density_ratio'],summaries) for r in summary['metrics']];csvwrite(out/'metrics/combined_metrics.csv',metrics)
    write(out/'metadata/sources.json',dict(sources=[fileinfo(p/'metrics/summary.json') for p in paths],source_runs=[A,B]))
    lines=['# 전체 동작과 시작·끝: 두 프레임 배정 방식의 통합 비교','',
    'pilot300 전체: Charades·ActivityNet·QVHighlights 각100개. 비율2·4·8, Random seed42·43. 같은 질문·모델·프레임 수·공간 해상도·영상 토큰 예산을 유지했습니다. 두 실험은 동일 표본의 대응 비교이며 600개 독립 질문으로 합치지 않습니다.','',
    '첫 실험은 GT 전체 길이 L로 K=round(F×rL/(rL+T−L))를 정하고 전체·경계에 같은 목표 K장을 배정합니다. 두 번째 실험은 각 조건의 실제 집중 길이 D로 K=round(F×rD/(rD+T−D))를 정해 안/밖 밀도 비율을 맞춥니다. 경계는 시작·끝 ±1초의 합집합을 영상 범위와 교차한 영역입니다.','',
    '전체 조건과 Uniform은 두 방식에서 같으며 검증된 동일 입력 응답을 재사용했습니다. Random 위치도 두 실험·비율·전체/경계 사이에 동일하게 고정했습니다. 원본 프레임 부족과 최소 배정 때문에 목표 배정과 실제 배정이 다를 수 있고 각 보고서에 기록했습니다.','',
    '시간 구간 겹침률(IoU)=예측과 정답의 교집합 길이/합집합 길이. mIoU는 전체 질문의 평균이며 높을수록 좋습니다. 실패는0점이며 raw 초 단위, 반올림·clipping 없이 평가합니다. 평균 절대 오차(MAE)=평균 |예측 시각−정답 시각|로 낮을수록 좋고 유효 응답만 사용합니다. pp는 백분율 차이입니다. 영상 단위 bootstrap 20,000회 95% 구간은 탐색적이며 여러 비교를 보정한 확증 검정이 아닙니다.','',
    '| 벤치마크 | 비율 | 경계 조건 | 밀도 일치 − 프레임 수 공유: mIoU 차이 pp [95% 구간] |','|---|---:|---|---|']
    for x in combined:lines.append(f"| {x['dataset']} | {x['ratio']} | {x['who']} | {x['delta_pp']:+.2f} [{x['lower_pp']:+.2f}, {x['upper_pp']:+.2f}] |")
    lines+=['','위 차이가 양수면 실제 집중 길이로 배분하는 두 번째 방식이 유리한 방향입니다. 0을 포함하면 우열이 불확실합니다. GT 경계가 좋아져도 대응 Random과 Uniform 비교를 함께 봐야 정확한 위치 정보의 이득을 판단할 수 있습니다.','', '| 배정 방식 | 벤치마크 | 비율 | GT 경계 − GT 전체 pp [95% 구간] |','|---|---|---:|---|']
    for label,summary in zip(['집중 프레임 수 공유','안팎 밀도 일치'],summaries):
        for x in summary['comparisons']:
            if x['a'].startswith('gt_boundary') and x['b'].startswith('gt_whole'):lines.append(f"| {label} | {x['dataset']} | {x['a'].split('_')[-1][1:]} | {x['delta_pp']:+.2f} [{x['lower_pp']:+.2f}, {x['upper_pp']:+.2f}] |")
    lines+=['','가장 좋은 비율만 골라 일반화하지 않고 모든 비율과 벤치마크 결과를 함께 제시했습니다. 같은 입력의 재사용은 추가 반복 증거가 아닙니다.','',f'[집중 프레임 수 공유 실험](../{A}/REPORT.md) · [실제 집중 길이로 밀도 일치 실험](../{B}/REPORT.md)',f'[통합 수치](../../artifacts/runs/{RUN}/metrics/combined_metrics.csv) · [시작·종료 오차 변화와 신뢰구간](../../artifacts/runs/{RUN}/metrics/cross_design_comparisons.json)']
    lines+=['','실행 중 속도 개선: CPU 입력 준비를 병렬화하고 최대4개 GPU 배치를 적용했습니다. 세 벤치마크 8개 점검 입력에서 단일/배치 생성 토큰이 정확히 일치했습니다. 모든 입력에서 동일 출력을 보장하는 검증은 아니며, 개별 응답에 실제 batch_size와 batch_group을 기록했습니다. 배치 wall time은 응답 수로 나누어 비용을 집계했습니다. CPU 준비 시간과 GPU 시간은 겹치므로 단순 합계가 실제 wall time은 아닙니다. 각 실행의 environment/prefetch_operation.json 및 logs/prefetch.jsonl에서 적용 시점과 비용을 확인할 수 있습니다.']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from gt_dataset_diagnostics import main as rebuild_readable_report
    rebuild_readable_report()
    from artifact_store import register
    register(RUN+'.analyses','analyses',report,'Combined focus-count and actual-duration density comparison','_lib/gt_density_combined.py','pilot_300','complete',[A+'.analyses',B+'.analyses'])
    write(out/'metadata/status.json',dict(state='complete',report=str(report/'REPORT.md')))
if __name__=='__main__':main()
