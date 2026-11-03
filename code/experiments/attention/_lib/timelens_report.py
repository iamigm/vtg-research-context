"""Final, reproducible aggregation for `run.py timelens report`."""
from result_layout import relocate_path, write_result_text
import csv
import json
from timelens_paths import QWEN3_ZERO
from timelens_measurements import load_answer


def build_final_report(out, samples):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def read(p):
        return json.loads(p.read_text())

    def dump(name, value):
        write_result_text((relocate_path(out/name)), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def csv_file(name, rows):
        with (relocate_path(out/name)).open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)

    completion=read(relocate_path(out/'completion.json'))
    assert completion['passed'] and completion['completed_pairs']==len(samples)==9404
    orders=['video_query','query_video']
    datasets=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
    records=[]; pairs=[]
    for sample in samples:
        folder=relocate_path(out/sample['dataset']/sample['sample_key'])
        meta=read(relocate_path(QWEN3_ZERO/sample['dataset']/sample['sample_key']/'metadata.json'))
        assert meta['exact_token_block_swap'] and meta['non_text_inputs_bitwise_identical']
        for order in orders:
            a=load_answer(relocate_path(folder/order))
            v=read(relocate_path(folder/order/'validation.json'))
            assert v['passed'] and v['key_mass_sum_verified'] and v['predictive_consumed_shift_verified']
            assert a['sample_key']==sample['sample_key'] and a['order']==order
            assert v['answer_tokens']==len(a['token_ids'])
            assert a['gt']==sample['span'][0]
            m=a['attention']['consumed']
            assert m is not None and all(np.isfinite(float(x)) for x in m.values())
            row=dict(dataset=sample['dataset'],sample_key=sample['sample_key'],order=order,
                     answer_tokens=len(a['token_ids']),**m)
            records.append(row)
        p=read(relocate_path(folder/'comparison.json'))
        assert p['sample_key']==sample['sample_key'] and p['dataset']==sample['dataset']
        pairs.append(p)

    def aggregate(group):
        n=len(group)
        result=dict(samples=n)
        for key in ['video_mass','gt_mass','raw_gt_mass','early_peak','peak_in_gt','entropy','answer_tokens']:
            result[key]=float(np.mean([r[key] for r in group]))
        return result

    tables=[]; paired={}
    for ds in datasets:
        for order in orders:
            r=dict(dataset=ds,order=order,**aggregate([r for r in records if r['dataset']==ds and r['order']==order]))
            tables.append(r)
        group=[p for p in pairs if p['dataset']==ds]
        paired[ds]=dict(samples=len(group),
            gt_mass_gain_query_first=float(np.mean([p['gt_mass_gain_query_first'] for p in group])),
            attention_tv=float(np.mean([p['attention_tv'] for p in group])),
            attention_js_bits=float(np.mean([p['attention_js_bits'] for p in group])))
    metric_keys=['video_mass','gt_mass','raw_gt_mass','early_peak','peak_in_gt','entropy']
    overall={order:dict(query_weighted=aggregate([r for r in records if r['order']==order]),
                       dataset_macro={k:float(np.mean([r[k] for r in tables if r['order']==order])) for k in metric_keys}) for order in orders}
    validation=dict(passed=True,validated_runs=len(records),validated_pairs=len(pairs),
        checks=['all saved validation markers','exact input block swap and non-text input equality markers',
                'canonical prediction SHA256 and sample identity','finite attention measurements'],
        npz_arrays_reloaded_in_this_aggregation=False)
    result=dict(completed_at=completion['finished_at'],model=read(relocate_path(out/'run_config.json'))['model'],
        samples=len(samples),runs=len(records),attention_definition='Consumed answer rows excluding EOS; L8–21; head/token mean; live Q/K FP32 reconstruction.',
        video_mass_definition='Attention on video keys divided by all causal-key attention.',
        gt_mass_definition='GT attention divided by video attention; timestamp-bin overlap weighting.',
        raw_gt_mass_definition='GT attention divided by all causal-key attention.',
        early_peak_definition='Largest temporal attention bin lies in the first 10% of video duration.',
        tables=tables,paired=paired,overall=overall,validation=validation)
    dump('final_results.json',result);dump('final_aggregation_validation.json',validation)
    csv_file('dataset_metrics.csv',tables)

    names=['Charades','ActivityNet','QVHighlights']
    panels=[('video_mass','Answer attention on video (%)'),('gt_mass','GT share within video attention (%)'),
            ('raw_gt_mass','Answer attention on GT (%)'),('early_peak','Peak in first 10% of video (%)')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(key,title) in zip(axes.flat,panels):
        for j,order in enumerate(orders):
            values=[100*next(r[key] for r in tables if r['dataset']==ds and r['order']==order) for ds in datasets]
            bars=ax.bar(np.arange(3)+(j-.5)*.36,values,.36,color=['#2463a7','#db7b28'][j],label=['video → query','query → video'][j])
            ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
        ax.set_xticks(range(3),names);ax.set_title(title,fontsize=11)
        ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('TimeLens answer-to-video attention · Qwen3-VL-8B-Instruct · 9,404 paired queries',fontsize=14)
    fig.savefig(relocate_path(out/'final_comparison.png'),dpi=170)
    fig.savefig(relocate_path(out/'final_comparison.svg'));plt.close(fig)

    lines=['# TimeLens answer → video attention','',
        '전체 9,404쌍의 attention 측정 결과다. 원본 예측은 공용 pure zero-shot 저장소에서 읽고, 이 폴더에는 attention 배열·측정값·검증과 비교 결과만 둔다.','',
        '수치쌍은 **video → query / query → video**, 단위는 %다.','',
        '| 벤치마크 | 영상 attention 총량 | 영상 내 GT 비중 | 전체 key 중 GT 비중 | 초반 10% peak 빈도 |','|---|---:|---:|---:|---:|']
    for ds in datasets:
        a,b=[next(r for r in tables if r['dataset']==ds and r['order']==o) for o in orders]
        lines.append('| '+ds+' | '+' | '.join(f"{100*a[k]:.2f} / {100*b[k]:.2f}" for k in ['video_mass','gt_mass','raw_gt_mass','early_peak'])+' |')
    lines+=['','역순에서는 영상 attention 총량은 증가하지만 영상 내부 GT 비중은 감소한다. 초반 peak 빈도도 낮아지지만 여전히 높다. 서로 다른 생성 답변의 관측값이므로 이 변화가 성능 차이의 원인이라고 단정하지 않는다.','',
        'Consumed answer 행(EOS 제외), L8–21, head·token 평균이다. 실제 post-RoPE·KV-cache Q/K로 FP32 softmax를 재구성하며 FlashAttention 출력은 변경하지 않았다.','',
        '[지표 CSV](dataset_metrics.csv) · [전체 집계](final_results.json) · [그래프](final_comparison.png) · [검증](final_aggregation_validation.json)','',
        '[순수 예측 저장소](../../../zero-shot/predictions/README.md) · [Zero-shot 성능 비교](../../../zero-shot/results/baseline/input_order/overview/REPORT.md)','']
    write_result_text((relocate_path(out/'final_report.md')), '\n'.join(lines))
    index=relocate_path(out/'report.html')
    text=index.read_text().replace('final_report.html','final_report.md')
    if 'href="final_report.md"' not in text:
        text=text.replace('</h1>','</h1><p><a href="final_report.md">Attention 최종 집계</a></p>',1)
    write_result_text(index, text)
    print(f'Attention report: {out}/final_report.md; {len(pairs)} pairs verified')
