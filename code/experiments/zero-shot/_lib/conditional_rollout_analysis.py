"""CPU-only audit and interpretation of completed conditional budget trees."""
import collections,csv,json,re,statistics
from pathlib import Path
from conditional_rollout_budget import OUT,REPORT,iou,write

def parse(answer):
    tokens=re.findall(r'\d+(?::\d+){1,2}(?:\.\d+)?|\d+(?:\.\d+)?',answer)
    def seconds(token):
        value=0.
        for part in token.split(':'):value=value*60+float(part)
        return value
    assert len(tokens)>=2,answer
    return list(map(seconds,tokens[:2]))

def main():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'trees').glob('*.json'))]
    assert len(records)==30
    data=[];special=[]
    for r in records:
        row=r['sample'];gt=row['span'][0];s,e=parse(r['greedy_answer'])
        baseline=iou(s,e,gt) if 0<=s<e else 0.
        for tree in r['trees']:
            final=tree['snapshots'][-1]
            strict=[x for x in tree['leaves'] if x['valid']]
            positive=[x for x in tree['leaves'] if 0<=x['start']<x['end']]
            def best(leaves):
                return max([iou(x['start'],x['end'],gt) for x in leaves] or [0.])
            bestleaf=max(strict,key=lambda x:(iou(x['start'],x['end'],gt),x['mass'])) if strict else None
            entry=dict(sample_key=row['sample_key'],dataset=row['dataset'],video=row['video_path'],
                duration=row['duration'],gt_ratio=(gt[1]-gt[0])/row['duration'],
                policy=tree['policy'],baseline_iou=baseline,
                baseline_strict_iou=baseline if 0<=s<e<=row['duration'] else 0.,
                oracle_strict=best(strict),oracle_positive=best(positive),
                valid_mass=sum(x['mass'] for x in strict),best_leaf_mass=bestleaf['mass'] if bestleaf else None,
                baseline_out_of_range=e>row['duration'],**final)
            for k in [1,4,8,16,32]:
                entry[f'top{k}_oracle']=best(sorted(strict,key=lambda x:-x['mass'])[:k])
            data.append(entry)
            if tree['policy']=='joint' and final['grammar_rejected_mass']>.05:
                special.append(dict(sample_key=row['sample_key'],greedy=r['greedy_answer'],
                    rejected_mass=final['grammar_rejected_mass']))
    write(OUT/'metrics/interpretation.json',dict(records=data,format_exclusions=special))
    avg=lambda rows,k:statistics.mean(x[k] for x in rows)
    scores=lambda rows,k:[avg(rows,k)]+[sum(x[k]>=t for x in rows)/len(rows) for t in [.3,.5,.7]]
    base=[x for x in data if x['policy']=='joint']
    lines=['# 완료 결과 해석과 측정 제한','',
        '30/30 완료, 2026-11-03 17:34 KST. 마지막 성공 실행의 경과시간 3,363초(약56분). 데이터셋별10개인 진단 표본이며 일반 벤치마크 평균이 아니다. 기존 REPORT.md는 동결된 v1의 엄격한 범위 판정을 유지한다.', '',
        '지표: S=발견한 시작-prefix 질량, J=끝까지 완성한 구간 질량, V=0≤s<e≤영상 길이를 만족하는 완성 구간 질량. 모두 anchor 조건부 raw 확률이다. mIoU=질문별 시간 IoU 평균, R@τ=IoU≥τ 비율이다. 질량·성공률·IoU는 높을수록, 계산량은 낮을수록 좋다.', '',
        '| 정책, 512 확장 | 시작 S | 완성 J | 유효 구간 V | 시작 미확보 | 시작 아래 미완성 | decoder 토큰 | 탐색 초/질문 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pol in ['joint','balanced']:
        rs=[x for x in data if x['policy']==pol];S=avg(rs,'start_mass');J=avg(rs,'joint_mass')
        lines.append(f'| {pol} | {S:.4f} | {J:.4f} | {avg(rs,"valid_mass"):.4f} | {1-S:.4f} | {S-J:.4f} | {avg(rs,"incremental_decoder_tokens"):.1f} | {avg(rs,"elapsed_seconds"):.1f} |')
    lines+=['', '같은 prefix 확장 수라도 계산량이 다르다. joint는 balanced보다 decoder 토큰이 약25.7% 적고 J는 약10.0pp 높았다. 이는 두 정책의 최종 checkpoint 비교이며 모든 동일 계산 예산에서의 우위를 증명하지는 않는다.', '',
        'joint에서 미완성 질량의 약81.5%는 이미 발견한 시작 아래 종료가 완성되지 않은 부분이다: (S−J)/(1−J). 따라서 시작 폭만 늘리는 방식보다 종료 조건부 탐색에 예산을 남기는 설계가 필요하다.', '',
        '| 데이터셋, joint 512 | N | S | J | 문법 제외 질량 |',
        '|---|---:|---:|---:|---:|']
    for ds in sorted({x['dataset'] for x in data}):
        rs=[x for x in base if x['dataset']==ds]
        lines.append(f'| {ds} | {len(rs)} | {avg(rs,"start_mass"):.4f} | {avg(rs,"joint_mass"):.4f} | {avg(rs,"grammar_rejected_mass"):.4f} |')
    lines+=['', '형식 차이의 영향:', '']
    for x in special:
        lines.append(f'- `{x["sample_key"]}`: greedy `{x["greedy"]}`, 문법 제외 {x["rejected_mass"]:.1%}.')
    lines+=['', '특히 MM:SS를 지원하지 않아 유효한 시간 표현을 제외했다. 이 사례의 낮은 J를 순수한 탐색 예산 부족으로 해석할 수 없다. 다른 사례의 제외 질량 원인도 생성 토큰 분포를 추가 기록해야 확정할 수 있다.', '',
        '조건부 종료 분포는 joint에서30개 중10개가 TV 하한>.01,7개가>.1이었다. 그러나 평균 TV 범위가[.0992,.6523]으로 넓다. 시작이 다른 경우 종료 분포가 달라지는 사례는 확인되지만 그 크기를 전체 표본에서 정밀 추정하기에는 종료 탐색량이 부족하다. 실제로 서로 다른 사건인지는 영상을 검토하지 않아 미확인이다.', '',
        '같은 정책 내 일부 저예산 checkpoint에서는 높은 확률의 완성이 아직 없는데 극소확률 후보가 먼저 완성될 수 있다. oracle은 이런 후보도 활용하므로 높은 oracle만으로 실용적 후보 생성이 입증되지는 않는다.', '',
        '| 후보 처리, joint 512 | mIoU | R@.3 | R@.5 | R@.7 |',
        '|---|---:|---:|---:|---:|']
    for name,key in [('관측 유효 후보 확률1위','top1_oracle'),('상위4개 oracle','top4_oracle'),('상위16개 oracle','top16_oracle'),('전체 관측 유효 후보 oracle','oracle_strict')]:
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in scores(base,key))+' |')
    lines+=['', f'joint 최선 유효 후보 경로 확률의 중앙값은 {statistics.median(x["best_leaf_mass"] for x in base if x["best_leaf_mass"] is not None):.6g}이다. 경로확률은 후보 집합 내 정규화 확률이 아니다. 상위16개도 GT가 선택하는 oracle이며 실제 선택기 성능이 아니다.', '',
        'v1은 영상 끝을 조금이라도 넘으면 후보를 제외한다. greedy30개 중 '+str(sum(x['baseline_out_of_range'] for x in base))+'개가 이 조건에 걸린다. 이를 통상적인 시간 IoU 계산과 혼동하지 않도록 두 기준을 별도 비교한다. 아래 값은 반올림·clipping 없이 계산한다.', '',
        '| 동일 평가 기준 | 대상 | mIoU | R@.3 | R@.5 | R@.7 |',
        '|---|---|---:|---:|---:|---:|']
    for criterion,items in [('엄격한 영상 범위', [('greedy','baseline_strict_iou'),('joint 전체 oracle','oracle_strict')]),('범위 초과 허용, 양의 구간 IoU',[('greedy','baseline_iou'),('joint 전체 oracle','oracle_positive')])]:
        for label,key in items:lines.append('| '+criterion+' | '+label+' | '+' | '.join(f'{v:.4f}' for v in scores(base,key))+' |')
    lines+=['', '이 비교의 oracle−greedy 차이는 후보 집합의 상한 차이다. 영상 범위 필터의 효과, 탐색 효과, GT 선택 효과가 섞이지 않도록 실제 선택기를 별도로 평가해야 한다.', '',
        '후속 설계는 [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md)에 기록했다. 후속 GPU 추론은 시작하지 않았다. 상세 분포별 진단은 [STRATA.md](STRATA.md)에 있다.']
    (REPORT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    # Required duration/GT-ratio axes and crossed tables. All cells retain counts.
    duration=[0,30,60,120,180,float('inf')];ratio=[0,.05,.1,.2,.4,1.000001]
    def binof(value,bins):return next(i for i in range(5) if bins[i]<=value<bins[i+1])
    for x in data:x['duration_bin']=binof(x['duration'],duration);x['ratio_bin']=binof(x['gt_ratio'],ratio)
    text=['# 표본 분포별 진단','', '각 셀은 n/점수이다. n<5는 small로 표시한다. 빈 셀은0/—이다. 모든 수치는 엄격한 영상 범위 후보 oracle이며 최종 선택 성능이 아니다. 개선 pp는 같은 질문의 greedy strict 기준 성공률 차이다. 작은 셀은 일반화하지 않는다.', '']
    for pol in ['joint','balanced']:
        rs=[x for x in data if x['policy']==pol]
        for axis,bins in [('duration_bin',duration),('ratio_bin',ratio)]:
            text += [f'## {pol}: {axis}', '', '| 구간 | n | mIoU | R@.3 (Δpp) | R@.5 (Δpp) | R@.7 (Δpp) |','|---|---:|---:|---:|---:|---:|']
            for b in range(5):
                cell=[x for x in rs if x[axis]==b]
                if not cell:text.append(f'| [{bins[b]},{bins[b+1]}) | 0 | — | — | — | — |');continue
                ss=scores(cell,'oracle_strict');bb=scores(cell,'baseline_strict_iou')
                text.append(f'| [{bins[b]},{bins[b+1]}) | {len(cell)}'+(' small' if len(cell)<5 else '')+f' | {ss[0]:.4f} | '+' | '.join(f'{ss[j]:.4f} ({(ss[j]-bb[j])*100:+.1f})' for j in range(1,4))+' |')
        for metric in range(4):
            text += [f'## {pol}: 길이 × GT비율, '+['mIoU','R@.3','R@.5','R@.7'][metric],'', '| 영상 길이 | [0,.05) | [.05,.1) | [.1,.2) | [.2,.4) | [.4,1] |','|---|---|---|---|---|---|']
            for a in range(5):
                cells=[]
                for b in range(5):
                    cell=[x for x in rs if x['duration_bin']==a and x['ratio_bin']==b]
                    cells.append(f'{len(cell)}/'+(f'{scores(cell,"oracle_strict")[metric]:.4f}'+(' small' if len(cell)<5 else '') if cell else '—'))
                text.append(f'| [{duration[a]},{duration[a+1]}) | '+' | '.join(cells)+' |')
    (REPORT/'STRATA.md').write_text('\n'.join(text)+'\n')
    # Check scalar summaries against the frozen primary CSV rather than recomputing inference.
    frozen=list(csv.DictReader((OUT/'metrics/budget.csv').open()))
    for x in data:
        row=next(r for r in frozen if r['sample_key']==x['sample_key'] and r['policy']==x['policy'] and r['expansions']=='512')
        assert abs(float(row['oracle_iou'])-x['oracle_strict'])<1e-12
    write(OUT/'validation/interpretation.json',dict(passed=True,records=30,primary_scores_match=True,
          baseline_scope='same saved input and greedy answer; continuous timestamps; strict and unrestricted separately'))
    print((REPORT/'ANALYSIS.md').read_text())

if __name__=='__main__':main()
