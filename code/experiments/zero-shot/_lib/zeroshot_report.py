"""Verify reusable predictions and report zero-shot performance, without attention."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
import csv,hashlib,json,re,statistics,sys
from timelens_paths import ZEROSHOT,QWEN3_ZERO,PILOT_SAMPLES,baseline_path
from timelens_store import iter_predictions

ROOT=relocate_path(Path(__file__)).resolve().parents[1]
sys.path.insert(0,str(relocate_path(ROOT/'runtime/timelens_deps')))
sys.path.insert(0,str(relocate_path(ROOT/'vendor/TimeLens')))
from timelens.utils import extract_time,iou


def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    write_result_text(p, json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    rows=[];full={};indexed={};checked=0;invalid=[]
    for line in (relocate_path(ZEROSHOT/'index.jsonl')).read_text().splitlines():
        item=json.loads(line);key=(item['model'],item['dataset'],item['sample_key'],item['order'])
        assert key not in indexed
        indexed[key]=item
    assert len(indexed)==19108
    datasets=['charades-timelens','activitynet-timelens','qvhighlights-timelens']
    def stats(records,model,cohort,order,ds):
        return dict(model=model,cohort=cohort,order=order,dataset=ds,n=len(records),
            miou=statistics.mean(r['iou'] for r in records),
            **{f'r{int(t*10):02d}':statistics.mean(r['iou']>=t for r in records) for t in [.3,.5,.7]})
    for model,order in [('qwen3','video_query'),('qwen3','query_video'),('qwen25','video_query')]:
        records=list(iter_predictions(model,order))
        assert len(records)==(9404 if model=='qwen3' else 300)
        full[model,order]=records
        for r in records:
            assert not {'attention','elapsed_seconds','baseline_tokens_identical','steps'}.intersection(r)
            item=indexed[model,r['dataset'],r['sample_key'],order]
            path=relocate_path(ZEROSHOT/item['path'])
            assert path==baseline_path(model,r,order)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            assert r['order']==order
            parsed=extract_time(r['answer']) or [[r['duration']+10,r['duration']+20]]
            assert r['timestamps']==[[round(s),round(e)] for s,e in parsed]
            assert abs(iou(r['gt'],r['timestamps'][0])-r['iou'])<1e-12
            raw=extract_time(r['answer'])
            if r['timestamps'][0][0]>=r['timestamps'][0][1] or not raw:
                invalid.append(dict(model=model,dataset=r['dataset'],sample_key=r['sample_key'],order=order,
                    prediction_path='../../predictions/'+item['path'],
                    invalid_before_rounding=bool(raw and raw[0][0]>=raw[0][1]),
                    rounding_collapsed=bool(raw and raw[0][0]<raw[0][1] and r['timestamps'][0][0]>=r['timestamps'][0][1]),
                    parse_failed=not raw))
            checked+=1
        for ds in datasets:
            group=[r for r in records if r['dataset']==ds]
            row=stats(group,model,'full_9404' if model=='qwen3' else 'pilot_300',order,ds)
            rows.append(row)
            if model=='qwen3':
                log=(relocate_path(QWEN3_ZERO/f'{ds}_{order}.log')).read_text()
                for label,key in [('mIOU','miou'),('IOU 0.3','r03'),('IOU 0.5','r05'),('IOU 0.7','r07')]:
                    official=float(re.search('^'+re.escape(label)+r': ([\d.eE+-]+)',log,re.M)[1])/100
                    assert abs(official-row[key])<1e-10
    pilot=read(PILOT_SAMPLES);keys={r['sample_key'] for r in pilot}
    for ds in datasets:
        group=[r for r in full['qwen3','video_query'] if r['dataset']==ds and r['sample_key'] in keys]
        assert len(group)==100
        rows.append(stats(group,'qwen3','pilot_300','video_query',ds))
    target=relocate_path(ROOT/'results/input_order');target.mkdir(parents=True,exist_ok=True)
    if invalid:
        with (relocate_path(target/'invalid_prediction_refs.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(invalid[0]));writer.writeheader();writer.writerows(invalid)
    summary=dict(prediction_cache=str(ZEROSHOT),validated_records=checked,metrics=rows,
        default_baseline_order='video_query',reverse_order='query_video (explicit comparison variant)')
    write(relocate_path(target/'summary.json'),summary)
    with (relocate_path(target/'benchmark_metrics.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Zero-shot 성능 비교','',
        '공용 저장소의 원본 예측만으로 재집계했다. Attention 측정값이나 NPZ를 읽지 않는다. Qwen3 전체 9,404개 입력 순서 비교와 두 모델의 동일 pilot 300개를 구분한다.','',
        '| 모델 | 표본 | 순서 | 벤치마크 | N | mIoU (%) | R@1 .3 (%) | R@1 .5 (%) | R@1 .7 (%) |','|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['cohort']} | {r['order']} | {r['dataset']} | {r['n']} | "+' | '.join(f'{100*r[k]:.2f}' for k in ['miou','r03','r05','r07'])+' |')
    lines+=['','Qwen3는 역순에서 mIoU와 R@1 .3/.5가 세 벤치마크 모두 낮아졌다. Charades의 R@1 .7은 역순이 1.13%p 높다. 모델 간 비교는 동일 pilot_300 행을 사용해야 한다.','',
        '[실패 예측 참조](invalid_prediction_refs.csv) · [집계 CSV](benchmark_metrics.csv) · [전체 지표](summary.json) · [순수 예측과 재사용 방법](../../predictions/README.md)','',
        '[Span-head 실험](../timelens_span_head/REPORT.md) · [Top-30 실험](../timelens_top30/REPORT.md) · [별도 attention 분석](../../../attention/results/timelens_order/final_report.md)','',
        '순수 예측 캐시는 기존 full-head 생성 결과에서 측정 필드만 분리했다. Qwen3 원본 생성에는 출력을 변경하지 않는 attention 계측이 있었고, Qwen2.5는 full-head 기준선 실행에서 얻었다. 새 추론을 실행한 결과는 아니다. 디코더에 관한 후속 진단은 [기록](../timelens_top30/mmco_diagnosis/REPORT.md)을 참고한다.','']
    write_result_text((relocate_path(target/'REPORT.md')), '\n'.join(lines))
    validation=dict(passed=True,records_checked=checked,no_attention_or_measurement_fields=True,
        per_record_sha256_verified=True,parsing_and_iou_verified=True,official_full_benchmark_groups=6,
        qwen3_default_records=9404,qwen3_reverse_variant_records=9404,qwen25_default_records=300)
    write(relocate_path(ZEROSHOT/'validation.json'),validation)
    print(json.dumps(validation,indent=2));print(f'Report: {target}/REPORT.md')


if __name__=='__main__':main()
