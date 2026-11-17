"""Publish one within-path-first viewer per scope from existing immutable traces."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path('/home/irteam/experiments/zero-shot')
sys.path.insert(0,str(ROOT/'_lib'))
from result_layout import relocate_path
from token_uncertainty_output import write_result_text

RESULT=ROOT/'results/uncertainty/token_analysis'
ART=ROOT/'artifacts/runs/uncertainty/token_analysis'
ASSETS=ROOT/'_lib/token_viewer'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'
    if not path.exists() or path.read_text()!=text:
        temp=path.with_name(path.name+'.tmp');temp.write_text(text);temp.replace(path)


def byte_index(path):
    """Offsets refer to existing JSON objects; no candidate records are copied."""
    text=path.read_text();decoder=json.JSONDecoder();pos=text.index('[')+1;offset=len(text[:pos].encode());records=defaultdict(list)
    while True:
        start=pos
        while text[pos] in ' \r\n\t,':pos+=1
        offset+=len(text[start:pos].encode())
        if text[pos]==']':break
        record,end=decoder.raw_decode(text,pos);size=len(text[pos:end].encode())
        records[record['model']+'/'+record['sample_key']].append([record['endpoint'],record['ar_step'],offset,offset+size-1])
        pos=end;offset+=size
    return records


def catalog(scope):
    base=RESULT/scope;out=ART/scope
    sources={
        'paths':out/'digit_correctness/within_path/metrics/per_path.json',
        'blue_csv':out/'digit_correctness/blue_analysis/metrics/per_blue_step.csv',
        'blue_data':relocate_path(base/'digit_correctness/blue_analysis/data.js'),
        'all_steps':out/'top30_renormalized/metrics/per_step.csv.gz',
    }
    target=out/'metrics/viewer_catalog.json';meta=out/'metadata/viewer_sources.json'
    stamp={key:[str(p.relative_to(ROOT)),p.stat().st_size,p.stat().st_mtime_ns] for key,p in sources.items()}
    definition_key='first_error_including_format'
    if target.exists() and meta.exists():
        previous=json.loads(meta.read_text())
        if previous.get('source_stats')==stamp and previous.get('definition_key')==definition_key:return target
    result={}
    for r in json.loads(sources['paths'].read_text()):
        key=r['model']+'/'+r['sample_key'];entry=result.setdefault(key,{'endpoints':{},'blue_ranges':[]})
        entry['endpoints'][r['endpoint']]={k:r.get(k) for k in ('paired','error_kind','error_role','error_step','error_entropy','previous_step','previous_entropy','delta_previous','delta_prior_mean','risk_mean_entropy','response_risk_mean_entropy','prior_safe_steps')}
        e=entry['endpoints'][r['endpoint']]
        e['error_region']='none' if r['error_kind']=='none' else 'format' if r['error_kind']=='format' else 'fractional' if r['error_role']=='tenths' else 'integer'
        entry['endpoints'][r['endpoint']].update(safe_high=False,next_failure=False,min_safe=None,max_unsafe=None)
    needed=set()
    for key,record in result.items():
        for e in record['endpoints'].values():
            if e['error_kind'] in ('numeric','format'):needed.update(((key,e['error_step']),(key,e['error_step']-1)))
    entropies={}
    with gzip.open(sources['all_steps'],'rt') as stream:
        for r in csv.DictReader(stream):
            key=(r['model']+'/'+r['sample_key'],int(r['step']))
            if r['mode']=='raw' and key in needed:entropies[key]=float(r['top30_entropy_nats'])
    for key,record in result.items():
        for e in record['endpoints'].values():
            e.update(previous_token_step=None,previous_token_entropy=None,delta_previous_token=None,token_error_entropy=None)
            if e['error_kind'] not in ('numeric','format'):continue
            k=e['error_step'];h=entropies[key,k]
            if e['error_entropy'] is not None:assert abs(e['error_entropy']-h)<1e-10
            e['token_error_entropy']=h
            if (key,k-1) in entropies:
                previous=entropies[key,k-1]
                e.update(previous_token_step=k-1,previous_token_entropy=previous,delta_previous_token=h-previous)
    with sources['blue_csv'].open() as stream:
        for r in csv.DictReader(stream):
            e=result[r['model']+'/'+r['sample_key']]['endpoints'][r['endpoint']]
            e['safe_high'] |= float(r['entropy'])>=1.5
            e['next_failure'] |= r['outcome']=='next_numeric_failure'
            safe=float(r['safe_lower']);unsafe=float(r['unsafe_mass'])
            e['min_safe']=safe if e['min_safe'] is None else min(e['min_safe'],safe)
            e['max_unsafe']=unsafe if e['max_unsafe'] is None else max(e['max_unsafe'],unsafe)
    for key,ranges in byte_index(sources['blue_data']).items():result[key]['blue_ranges']=ranges
    save(target,result)
    save(meta,dict(definition_key=definition_key,source_stats=stamp,source_sha256={key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in sources.items()},
                   definition='Delta previous numeric skips format tokens; delta previous token uses raw H(k)-H(k-1) at the first numeric or format error, including outside-endpoint tokens. Fractional numeric error is role tenths; decimal-point format errors are separate. No copied probabilities.',responses=len(result)))
    return target


def publish():
    groups={};aliases={}
    for scope in ('pilot/300','full/9404'):
        base=RESULT/scope;out=ART/scope
        if not (out/'digit_correctness/within_path/metrics/summary.json').exists():continue
        index=catalog(scope);web=out/'web';web.mkdir(parents=True,exist_ok=True)
        rel=lambda path:os.path.relpath(path,base)
        config=dict(scope=scope,queries=int(scope.split('/')[1]),catalog=rel(index),
                    trace=rel(relocate_path(base/'digit_correctness/traces'))+'/',
                    raw=rel(relocate_path(base/'digit_correctness/raw'))+'/',
                    blue=rel(relocate_path(base/'digit_correctness/blue_analysis/data.js')),
                    media_base=rel(base/'digit_correctness')+'/',
                    integrated=rel(relocate_path(base/'integrated/traces'))+'/' if scope.startswith('pilot') else None)
        for name in ('viewer.js','style.css'):
            p=web/('token_viewer.'+name.split('.')[-1]);text=(ASSETS/name).read_text()
            if name=='style.css':text=text.replace('__FONT__',os.path.relpath(relocate_path(base/'digit_correctness/NotoSansKR.ttf'),web))
            if not p.exists() or p.read_text()!=text:p.write_text(text)
        html=(ASSETS/'report.html').read_text().replace('__CONFIG__',json.dumps(config,ensure_ascii=False))
        html=html.replace('__CSS__',rel(web/'token_viewer.css')).replace('__JS__',rel(web/'token_viewer.js'))
        html=html.replace('__MANIFEST__',rel(relocate_path(base/'digit_correctness/manifest.js')))
        write_result_text(base/'report.html',html)
        save(out/'metadata/viewer_publication.json',dict(
            entry=str((base/'report.html').relative_to(ROOT)),scope=scope,queries=config['queries'],
            source_catalog=str(index.relative_to(ROOT)),primary='within_path_entropy',
            supporting=['token_distribution','rounded_endpoint_feasibility','fixed_gt_iou','role_comparison'],
            producer=Path(__file__).name,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            assets={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}))
        # Keep the old root scientific report, including its relative source links.
        main=base/'REPORT.md'
        if main.exists() and '<!-- token-unified-report -->' not in main.read_text():
            old=base/'TOKEN_DISTRIBUTION.md'
            if not old.exists() or old.read_text()!=main.read_text():old.write_text(main.read_text())
        summary=json.loads((out/'digit_correctness/within_path/metrics/summary.json').read_text())
        pairs=[r for r in summary['paired'] if r['endpoint']!='all']
        lines=[f'# Token uncertainty · within-path 비교 · {scope.split("/")[0]} · {summary["queries"]:,} queries','<!-- token-unified-report -->','',
            '## 무엇을 먼저 보는가','',
            '한 sample의 같은 decoding 경로에서 step별 entropy가 어떻게 변하는지가 주 분석이다. 결과 탭에서 왼쪽 sample을 고르면 '
            '응답 전체의 entropy 곡선과 최초 오류를 본다. step을 누르면 해당 token distribution과 정답 가능성을 보조 패널에서 확인한다. '
            'start/end는 별도로 paired 비교하며, 회색의 오류 이후 step은 그 비교에서 제외한다.','',
            f'{scope}의 실제 query 수는 {summary["queries"]:,}개다. 두 모델 중 지원 표기의 {summary["source_responses"]:,}개 응답을 within-path 분석에 사용했다. '
            '지원 표기 밖 응답도 영상·응답·전체 token distribution은 확인할 수 있다.','',
            '## 지표와 필터 기준','',
            '모든 entropy와 후보 확률은 top-30 재정규화 기준이다. 각 step의 top-30 원래 확률 합을 M이라 두고 qᵢ=pᵢ/M, H₃₀=−Σqᵢlnqᵢ로 계산한다. 단위는 nat이며 0–ln(30)≈3.401이다. '
            'entropy가 높으면 top-30 내부 후보 확률이 더 고르다. top-30 밖의 probability mass는 포함하지 않는다. '
            'Δprevious=H₃₀(최초 numeric 오류)−H₃₀(직전 정답 가능 numeric step)이다. 양수면 오류 때 더 높았다. '
            '정답 가능성은 허용 문법 아래 반올림 endpoint 정답으로 완성할 수 있는지의 판정이며 최종 정답 확률이 아니다.','',
            '| 특수 필터 기준 | 계산·분류 | 포함 범위 |',
            '|---|---|---|',
            '| 직전 숫자 대비 상승·감소 (기존) | Δprevious = H(최초 numeric 오류) − H(직전 정답 가능 numeric step) | 같은 endpoint에서 이전 숫자가 있는 paired 경로. 중간 소수점·서식은 건너뜀 |',
            '| 직전 token 대비 상승·감소 | Δtoken = H(k) − H(k−1), k는 최초 오류(numeric 또는 format)의 decoding step | 바로 앞 decoding token. 소수점·공백·endpoint 밖 token도 포함. 첫 숫자·서식 오류도 직전 token이 있으면 포함 |',
            '| 소수부 최초 오류 | 최초 numeric 오류의 role이 tenths | 허용된 소수 한 자리 표기의 소수 첫째 자리. 소수점 자체의 오류는 format으로 분리 |',
            '| 정수부 / 서식 / 오류 없음 | 최초 오류의 숫자 role / format 판정 / 최초 오류 없음 | 같은 버튼 그룹 안에서는 OR, 서로 다른 그룹 및 정량 범위는 AND로 결합 |','',
            '**표 해설.** 두 상승·감소 기준은 비교 상대가 다르다. 예를 들어 정수 숫자 H=1.2, 소수점 H=0.01, '
            '최초 소수부 오류 H=0.8이면 기존 숫자 기준은 −0.4 nat으로 감소지만 바로 직전 token 기준은 +0.79 nat으로 상승이다. '
            '이는 계산 설명용 가상 예시다. 양수만 상승, 음수만 감소이며 0과 비교 상대가 없는 경로는 둘 다 아니다. '
            '모든 H는 raw top-30 재정규화 entropy다. 아래 기존 정량 표는 직전 숫자 기준을 유지한다.','',
            '같은 필터 그룹 안에서 여러 버튼을 선택하면 하나라도 만족하는 경로를 포함한다(OR). 서로 다른 그룹과 정량 범위는 모두 만족해야 한다(AND). 선택한 버튼을 다시 누르면 해제하며 전체는 해당 그룹의 제한을 없앤다. 낮은 entropy 필터와 최초 오류 entropy 범위·정렬은 numeric·format 최초 오류 모두에 적용한다. 좌측 빠른 필터의 낮은 entropy는 ≤0.5 nat, 높은 entropy는 ≥1.5 nat인 탐색용 고정 기준이다. '
            '이는 검증된 오류 탐지 임계값이 아니다. 수치 범위는 선택한 모델·endpoint의 sample별 지표에 AND로 적용한다. '
            '오류 전후 차이가 없는 경로는 해당 범위를 지정했을 때 제외하며, 필터 후 숫자는 고유 query 수다.','',
            '## 같은 경로의 오류 전후 차이','',
            '| 모델 | dataset | endpoint | paired query 수 | 오류 − 직전 숫자 (nat) [95% CI] |',
            '|---|---|---|---:|---:|']
        for r in pairs:
            v=r['delta_previous'];lines.append(f'| {r["model"]} | {r["dataset"]} | {r["endpoint"]} | {v["queries"]} | {v["mean"]:.4f} [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}] |')
        positive=sum(r['delta_previous']['ci95'][0]>0 for r in pairs)
        lines+=['','**표 해설.** 각 sample에서 오류 entropy와 직전 숫자의 entropy를 뺀 후 query 사이에서 평균한 값이다. '
            f'{positive}/{len(pairs)}개 그룹의 CI가 0 위에 있다. 이는 그룹 평균의 상승이며 모든 sample에서 상승했다는 뜻은 아니다. '
            'CI는 query 단위 bootstrap 2,000회로 구했다. 자릿수·순서를 통제한 결과와 제외 경로는 within-path 상세 appendix에서 함께 읽는다.','',
            '## 결과와 appendix','',
            '[통합 sample viewer](report.html) · [전체 appendix 목록](APPENDIX.md)','',
            'sample 기준 수준만 제거해도 prefix·자릿수·후보 집합의 차이는 남는다. 저장된 greedy 경로의 사후 비교이므로 '
            '인과 효과나 온라인 오류 탐지 성능을 뜻하지 않는다. pilot과 full은 별도 범위이며 독립 표본처럼 합산하지 않는다.']
        if (base/'entropy_error/REPORT.md').exists():
            lines+=['', '[Entropy 상승·최초 오류의 정량 비교와 영상 증거 실험안](entropy_error/REPORT.md)은 appendix에서 확인한다. '
                    '특수 필터별 기술통계와 같은 경로의 유지 step을 대조한 결과, 아직 실행하지 않은 영상 개입 설계를 구분했다.']
        from saved_decoding_entropy_links import links_from
        saved_links=links_from(base)
        if saved_links:
            lines+=['', '다른 저장 자료의 재집계: '+' · '.join(saved_links)]
        write_result_text(main,'\n'.join(lines)+'\n')
        # Retired views are replaced, not embedded as nested viewers.
        for olddir in ('digit_correctness','digit_correctness/blue_analysis','digit_correctness/within_path','integrated'):
            old=base/olddir/'report.html';aliases[scope+'/'+olddir+'/report.html']=scope+'/report.html'
            if old.exists():old.unlink()
            for name in ('preview.png','preview_mobile.png'):(base/olddir/name).unlink(missing_ok=True)
        # All scientific MDs remain in place and become documents of this scope.
        documents=[p for p in sorted(base.rglob('*.md')) if p not in (main,base/'APPENDIX.md')]
        appendix=[f'# Token uncertainty appendix · {scope.split("/")[0]} · {summary["queries"]:,} queries','',
                  '같은 자료의 서로 다른 정량 분석이다. sample 시각화는 [통합 결과](report.html) 하나에서 제공한다.','']
        for p in documents:
            title=next((x[2:] for x in p.read_text().splitlines() if x.startswith('# ')),p.stem)
            appendix.append('- ['+title+']('+os.path.relpath(p,base)+')')
        if saved_links:
            appendix+=['', '## 다른 실험의 저장된 decoding 자료', '',
                       '원래 계열·실제 query 범위의 appendix로 연결하며 현재 scope와 합산하지 않는다.', '',
                       *['- '+link for link in saved_links]]
        write_result_text(base/'APPENDIX.md','\n'.join(appendix)+'\n')
        for p in base.rglob('*.md'):
            text=p.read_text()
            def replace(m):
                value=m[1]
                if '://' in value or value.startswith('#'):return m[0]
                target=Path(os.path.normpath(str(p.parent/value.split('#')[0])))
                retired=[base/x/'report.html' for x in ('digit_correctness','digit_correctness/blue_analysis','digit_correctness/within_path','integrated')]
                return ']('+os.path.relpath(base/'report.html',p.parent)+')' if target in retired else m[0]
            updated=re.sub(r'\]\(([^)]+)\)',replace,text)
            if updated!=text:p.write_text(updated)
        groups[scope]={'analysis_path':[],'report':scope+'/REPORT.md'}
    save(ART/'overview/web/view_groups.json',dict(scopes=groups,aliases=aliases))


if __name__=='__main__':publish()
