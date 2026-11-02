"""Complete delivery audit and cost/diagnostic appendix; no inference.

--watch waits for the detached supervisor, then runs this audit automatically.
"""
from adaptive_temporal_sampling import *
import csv,re
import numpy as np

def deliver():
    from adaptive_temporal_report import score,csvwrite,coverage
    status=read(OUT/'metadata/status.json');need(status['state']=='complete','Inference/report supervisor not complete')
    final=read(OUT/'validation/final.json');need(final['passed'] and final['n_final']==600 and final['new_scouts']==100,'Full validation missing')
    need(read(OUT/'validation/original_pts_timebase.json')['passed'],'Exact source time-base/PTS validation failed')
    need(read(OUT/'validation/integrity_fixtures.json')['passed'],'Integrity fixtures failed')
    need(read(OUT/'validation/cpu_prepare_status.json')['state']=='complete','Parallel CPU preparation incomplete')
    for kind in ['predictions','manifests','inputs','analyses']:need(read(STORE/'registry'/(RUN+'.'+kind+'.json'))['state']=='complete','Registry not complete')
    cohort=read(OUT/'metadata/cohort.json');summary=read(OUT/'metrics/summary.json');records=read(OUT/'metadata/baseline_records.json')
    basemap={r['sample']['sample_key']:r for r in records};evaluations=[];breakdowns=[];scout_diagnostics=[];random_means=[];new_coverage=[]
    with (OUT/'metrics/compute_cost.csv').open() as f:costs=list(csv.DictReader(f))
    with (OUT/'metrics/sampling_diagnostics.csv').open() as f:diags={(r['sample_key'],r['condition']):r for r in csv.DictReader(f)}
    quartiles=np.quantile([r['B'] for r in records],[.25,.5,.75]);vocab=[]
    for sample in cohort:
        key=sample['sample_key'];gt=sample['span'][0];T=sample['duration'];scout=read(OUT/'scouts'/(key+'.json'))
        u=read(OUT/'responses/uniform'/(key+'.json'))
        need(scout['identity']['input_hash']==u['identity']['input_hash'] and scout['generated_ids']==u['generated_ids'],'Scout/Uniform input or generated-ID mismatch')
        scout_eval=score(scout['answer'],gt,T);sv=scout_eval['iou'];bi=basemap[key]['B']
        evaluations.append(dict(sample_key=key,condition='scout',gt=gt,annotation_duration=T,**scout_eval))
        scout_diagnostics.append(dict(sample_key=key,eos_reached=scout['eos_reached'],**scout_eval))
        seed_values=[]
        for condition in CONDITIONS:
            response=read(OUT/'responses'/condition/(key+'.json'));val=score(response['answer'],gt,T)
            evaluations.append(dict(sample_key=key,condition=condition,gt=gt,annotation_duration=T,**val))
            if condition.startswith('random'):seed_values.append((condition,val['iou']))
            d=diags[(key,condition)]
            plan=read(OUT/'plans'/condition/(key+'.json'));baseline_ids=set(basemap[key]['baseline_metadata']['decoder_metadata']['frames_indices'])
            truly_new=[t for i,t in zip(plan['frame_ids'],plan['pts']) if i not in baseline_ids]
            new_coverage.append(dict(sample_key=key,condition=condition,new_vs_uniform_frames=len(truly_new),focus_frames_already_in_uniform=sum(i in baseline_ids for i in plan['focus']+plan['fill']),**coverage(truly_new,gt)))
            row=dict(sample_key=key,video_id=basemap[key]['video_id'],condition=condition,iou=val['iou'],scout_iou=sv,
                gt_length_ratio=(gt[1]-gt[0])/T,B=bi,fill=float(d['fill_fraction'])>0,fallback=d['pred_fallback']=='True',
                temporal_coverage=float(d['temporal_coverage']),gt_frame_fraction=float(d['gt_frame_fraction']))
            vocab.append(row)
        row=dict(sample_key=key,video_id=basemap[key]['video_id'],n_query=1,n_seeds=3)
        row.update({condition+'_iou':v for condition,v in sorted(seed_values)})
        row.update(mean_iou=float(np.mean([v for _,v in seed_values])),mean_r03=float(np.mean([v>=.3 for _,v in seed_values])),mean_r05=float(np.mean([v>=.5 for _,v in seed_values])),mean_r07=float(np.mean([v>=.7 for _,v in seed_values])))
        random_means.append(row)
    for kind in ['scout_iou','gt_length_ratio','B_quartile','fill','fallback']:
        groups={}
        for row in vocab:
            if kind=='scout_iou':group=['<.3','.3-.5','.5-.7','>=.7'][int(np.searchsorted([.3,.5,.7],row['scout_iou'],side='right'))]
            elif kind=='gt_length_ratio':group=['<.2','.2-.4','>=.4'][int(np.searchsorted([.2,.4],row['gt_length_ratio'],side='right'))]
            elif kind=='B_quartile':group='Q'+str(int(np.searchsorted(quartiles,row['B'],side='right'))+1)
            else:group=str(row[kind])
            groups.setdefault((row['condition'],group),[]).append(row)
        for (condition,group),rr in sorted(groups.items()):
            breakdowns.append(dict(axis=kind,condition=condition,group=group,N=len(rr),miou=float(np.mean([r['iou'] for r in rr])),r05=float(np.mean([r['iou']>=.5 for r in rr])),temporal_coverage=float(np.mean([r['temporal_coverage'] for r in rr])),gt_frame_fraction=float(np.mean([r['gt_frame_fraction'] for r in rr])),interpretation='post-hoc diagnostic only; no tuning or selection'))
    write(OUT/'metrics/evaluation_records.json',evaluations);csvwrite(OUT/'metrics/diagnostic_breakdowns.csv',breakdowns)
    write(OUT/'metrics/scout_diagnostics.json',scout_diagnostics);csvwrite(OUT/'metrics/random_seed_means.csv',random_means)
    csvwrite(OUT/'metrics/new_observation_coverage.csv',new_coverage)
    need(len(list((OUT/'responses').rglob('*.json')))==600 and len(list((OUT/'scouts').glob('*.json')))==100,'Response count or key collision')
    need(read(OUT/'metadata/plan_dependencies.json')['reconstructed_plans']==600,'Plan provenance incomplete')
    for r in records:
        sample=r['sample'];old=LEGACY/'qwen3'/sample['dataset']/sample['sample_key']/'metadata.json'
        need(read(old)==r['baseline_metadata'],'Historical legacy metadata changed')
    for source in read(OUT/'validation/baseline_generation_provenance.json')['sources']:need(sha(source['path'])==source['sha256'],'Historical generation config changed')
    def summed(rr,k):return sum(float(r[k]) for r in rr if r.get(k))
    scouts=[r for r in costs if r['stage']=='scout'];scout_gpu=summed(scouts,'generation_seconds');scout_tokens=int(summed(scouts,'generated_tokens'))
    table=[]
    for condition in CONDITIONS:
        rr=[r for r in costs if r['stage']=='final' and r['condition']==condition]
        table.append(dict(condition=condition,n=100,final_gpu_wall_seconds=summed(rr,'generation_seconds'),prefill_event_seconds=summed(rr,'prefill_seconds'),decode_event_seconds=summed(rr,'decode_seconds'),input_cpu_wall_sum_seconds=summed(rr,'input_prepare_seconds'),policy_scout_plus_final_gpu_seconds=scout_gpu+summed(rr,'generation_seconds'),peak_allocated_GiB=max(float(r['peak_allocated_bytes']) for r in rr)/2**30,output_tokens=int(summed(rr,'generated_tokens')),scout_tokens=scout_tokens))
    csvwrite(OUT/'metrics/compute_cost_summary.csv',table)
    budget=read(OUT/'metrics/budget_reference.json');config=read(OUT/'metadata/config.json')
    summary.update(state='complete',delivery_validation_passed=True,physical_new_responses=700,reused_responses=0,
        inference_infrastructure_failures=0,pre_inference_implementation_failures=2,
        input_budget=budget,compute_cost_by_condition=table)
    summary['scout_diagnostics']=dict(parse_failures=sum(r['parse_failed'] for r in scout_diagnostics),invalid_intervals=sum(r['invalid_interval'] for r in scout_diagnostics),out_of_range=sum(r['out_of_range'] for r in scout_diagnostics),eos_not_reached=sum(not r['eos_reached'] for r in scout_diagnostics))
    summary['cost'].update(scout_generated_tokens=scout_tokens,total_generated_tokens=scout_tokens+summary['cost']['final_generated_tokens'],successful_supervisor_wall_seconds=status['elapsed_seconds'],candidate_timing_definition='sum of per-query elapsed preparation/bridge tasks, including preparation reused across operational restarts; parallel tasks overlap, so not critical-path wall time or CPU core seconds')
    write(OUT/'metrics/summary.json',summary)
    appendix=['','<!-- delivery-details -->','',
        '**전량 완료 및 전달 검증:** 신규 scout 100 + final 600 = **700/700 응답**, 과거 응답 재사용 0, 추론 인프라 실패 0. 최초 구현 검증 중 발생한 2건의 오류(공식 query 정규화·tuple/JSON 동등성)는 GPU 추론 전에 수정했고 provenance/attempt_*/에 보존했다. 운영상 준비 재시작과 병렬 준비도 해당 로그에 남겼다.','',
        f'실제 F 범위 {budget["F_range"]}, B 범위 {budget["B_range"]}, 서로 다른 grid {budget["distinct_grids"]}개. 각 query 내 조건 차이는 F/B/grid 모두 0. 전체 100개의 single-pass 합계는 F={budget["F_sum_per_single_pass"]:,}, B={budget["B_sum_per_single_pass"]:,}; 두 단계 정책은 각각 그 2배다. 물리적으로 scout를 공유한 전체 700응답의 합계는 F={7*budget["F_sum_per_single_pass"]:,}, B={7*budget["B_sum_per_single_pass"]:,}다.','',
        '| 조건 | Final GPU wall s | Prefill/decode CUDA-event s | CPU 입력 준비 합 s | Scout+final 정책 GPU wall s | Peak allocated GiB | Final 출력 토큰 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in table:appendix.append(f"| {r['condition']} | {r['final_gpu_wall_seconds']:.1f} | {r['prefill_event_seconds']:.1f}/{r['decode_event_seconds']:.1f} | {r['input_cpu_wall_sum_seconds']:.1f} | {r['policy_scout_plus_final_gpu_seconds']:.1f} | {r['peak_allocated_GiB']:.2f} | {r['output_tokens']} |")
    uniform=table[0]
    appendix += ['',f'Single-pass Uniform은 위 Uniform final 자체({uniform["final_gpu_wall_seconds"]:.1f}초)다. 공통 scout는 물리적으로 {scout_gpu:.1f}초·{scout_tokens}출력 토큰을 한 번 계산했고, 각 두 단계 정책 비용에는 이를 한 번씩 귀속했다. 조건별 정책 비용을 합치면 공통 scout를 중복 계상하므로 물리적 전체 실행 비용과 구분한다. Pred의 전체 비용이 single-pass Uniform과 같다는 뜻이 아니다. GT annotation 취득 비용은 미계측이다.','',
        'CPU 후보 준비+bridge 값은 query별 작업 경과시간 합계다. 선행 병렬 작업의 시간이 겹치므로 전체 wall time이나 CPU core seconds로 해석하지 않는다. 실제 성공 supervisor wall time은 summary.json에 별도로 기록했다. cold/warm cache, scout/final, CPU 준비, GPU forward CUDA-event 시간과 생성 전체 wall time은 비용 CSV에서 구분한다. CUDA-event 계측은 forward 경과시간이며 순수 연산 FLOPs 측정은 아니다.','',
        '원본 time base와 정수 presentation timestamps를 demux한 96개 영상 모두 exact zero-origin CFR였다. 원본 RGB decoder는 Decord로 유지했다. GT 위치를 사용한 timestamp 목록 자체도 GT-guided oracle 정보의 일부다.','',
        'scout와 final 700개 응답의 파싱한 원문 구간·유효성·GT·경계오차는 evaluation_records.json에, scout IoU/GT 길이 비율/B 사분위/fill/fallback별 N과 결과는 diagnostic_breakdowns.csv에 보존했다. breakdown은 사후 진단이며 주결론이나 parameter 선택에 사용하지 않았다. Random의 query별 세 seed와 평균은 random_seed_means.csv에 있다. Random-mean의 recall은 세 seed 성공 지표의 query별 평균이며 합성 평균 IoU를 threshold한 실제 응답으로 해석하지 않는다.','',
        f'Config canonical JSON hash: `{objhash(config)}`; config 파일 bytes SHA-256: `{sha(OUT/"metadata/config.json")}`. canonical hash는 key 정렬·compact JSON 직렬화로 계산하므로 파일 bytes hash와 다를 수 있다.','',
        f'[비용 집계](../../artifacts/runs/{RUN}/metrics/compute_cost_summary.csv) · [응답별 평가](../../artifacts/runs/{RUN}/metrics/evaluation_records.json) · [사후 breakdown](../../artifacts/runs/{RUN}/metrics/diagnostic_breakdowns.csv) · [exact PTS 검증](../../artifacts/runs/{RUN}/validation/original_pts_timebase.json) · [전달 검증](../../artifacts/runs/{RUN}/validation/delivery.json)','',
        '보고서 전체 재생성 명령(원본 응답 보존, GPU 추론 없음):',
        '```bash','cd /home1/irteam/experiments/zero-shot','.venv/bin/python _lib/adaptive_temporal_report.py','.venv/bin/python _lib/adaptive_temporal_finalize.py','```']
    appendix += ['', 'anchors/focus는 guided 입력의 슬롯 분해다. Uniform 및 Uniform fallback에는 별도 focus 슬롯이 없으므로 해당 빈집합 coverage를 전체 영상 coverage로 해석하지 않는다. focus가 기존 baseline 프레임도 선택할 수 있어, 실제 Uniform에 없던 source ID의 관측 coverage를 new_observation_coverage.csv에 별도로 기록했다.','',
        '| 비교 | 변화 | ID 순 첫 사례 | ΔIoU | 원본 응답 |','|---|---|---|---:|---|']
    with (OUT/'metrics/paired_samples.csv').open() as f:paired=list(csv.DictReader(f))
    for comparison in ['gt-uniform','pred-uniform']:
        condition=comparison.split('-')[0]
        for label,test in [('개선',lambda x:x>0),('악화',lambda x:x<0),('동일',lambda x:x==0)]:
            selected=next((r for r in paired if r['comparison']==comparison and test(float(r['delta_iou']))),None)
            if selected:
                key=selected['sample_key'];prefix=f'../../artifacts/runs/{RUN}/responses'
                appendix.append(f"| {comparison} | {label} | {key} | {float(selected['delta_iou']):+.4f} | [Uniform]({prefix}/uniform/{key}.json), [{condition}]({prefix}/{condition}/{key}.json) |")
            else:appendix.append(f'| {comparison} | {label} | 해당 사례 없음 | — | — |')
    appendix += ['', '사례는 각 변화 유형에서 sample ID 순 첫 query를 제시한 사후 설명이며 대표성이나 일반화를 보장하지 않는다. 전체 query와 모든 조건은 paired_samples.csv 및 원본 responses/에 보존했다.']
    report=REPORT/'REPORT.md';body=report.read_text().split('<!-- delivery-details -->')[0].rstrip();report.write_text(body+'\n'+'\n'.join(appendix)+'\n')
    # Validate local Markdown targets after final links have been added.
    checked=[]
    for md in REPORT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if target.startswith(('https:','http:','#')):continue
            path=(md.parent/target.split('#')[0]).resolve()
            # delivery.json is published atomically immediately below.
            need(path.exists() or path==OUT/'validation/delivery.json','Broken report link '+str(path));checked.append(str(path))
    producer=fileinfo(Path(__file__));snapshot=OUT/'code'/('delivery_'+producer['sha256'][:12]+'.py')
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    write(OUT/'validation/delivery.json',dict(passed=True,timestamp=stamp(),new_scouts=100,new_finals=600,reused_responses=0,exact_pts_videos=96,uniform_input_and_generated_ids_equal=100,links_checked=len(checked),producer=producer))
    from artifact_store import register
    graph={
        'manifests':('manifests',OUT/'metadata',['pilot_300.cohort','full_benchmark_v1.predictions','pilot8fps.v1.predictions']),
        'inputs':('inputs',OUT/'inputs',[RUN+'.manifests']),
        'scouts':('predictions',OUT/'scouts',[RUN+'.inputs',RUN+'.manifests']),
        'predictions':('predictions',OUT/'responses',[RUN+'.inputs',RUN+'.manifests',RUN+'.scouts']),
        'analyses':('analyses',REPORT,[RUN+'.predictions',RUN+'.scouts',RUN+'.inputs',RUN+'.manifests'])}
    for suffix,(kind,path,deps) in graph.items():register(RUN+'.'+suffix,kind,path,'Validated adaptive temporal pilot: '+suffix,str(Path(__file__).relative_to(ROOT)),'pilot_300','complete',deps)
    # No abandoned staging files should remain after all writers finish.
    for tmp in OUT.rglob('*.tmp'):
        if tmp.is_file():tmp.unlink()
    sources=read(OUT/'metadata/source_manifest.json');sources['delivery_producer']=producer;write(OUT/'metadata/source_manifest.json',sources)
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and 'logs' not in p.relative_to(OUT).parts and p.name not in ['status.json','run.lock','output_manifest.json']:files.append(fileinfo(p))
    write(OUT/'validation/output_manifest.json',dict(files=files,reports=[fileinfo(p) for p in sorted(REPORT.glob('*.md'))],mutable_excluded=['logs','status.json','run.lock','output_manifest.json']))
    print(json.dumps(dict(state='complete',delivery_passed=True,report=str(report),comparisons=summary['comparisons'],cost=summary['cost']),ensure_ascii=False),flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    if args.watch:
        while True:
            s=read(OUT/'metadata/status.json')
            if s['state']=='complete':break
            if s['state']=='aborted':
                write(OUT/'validation/delivery.json',dict(state='skipped',reason='supervisor aborted',passed=False));return
            time.sleep(20)
    try:deliver()
    except BaseException:
        write(OUT/'validation/delivery.json',dict(state='failed',passed=False,error=traceback.format_exc(),timestamp=stamp()));raise

if __name__=='__main__':main()
