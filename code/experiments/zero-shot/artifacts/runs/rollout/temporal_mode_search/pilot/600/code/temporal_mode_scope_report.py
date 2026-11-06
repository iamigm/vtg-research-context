"""Publish only active seeds while preserving every original raw response."""
import json
import re
import sys
from pathlib import Path
from temporal_mode_scope import *
import temporal_mode_report as original

ORIGINAL_ROWS=original.per_query


def selected_rows(cohort):
    rows,complete=ORIGINAL_ROWS(cohort)
    return active_rows(rows),complete


def active_text(text):
    return (text.replace('seed42/43/44','seed42/43')
        .replace('seed42/43/44','seed42/43').replace('3조건×3seed','3조건×2seed')
        .replace('3seed','2seed').replace('최대5,400','최대3,600'))


def report(initialize=False):
    active=scope()
    immutable_config_hash=digest(CONFIG)
    original.CONFIG=dict(CONFIG,seeds=active['seeds'])
    original.per_query=selected_rows
    if initialize:
        original.initialize()
    else:
        original.report()
    assert digest(CONFIG)==immutable_config_hash,'Report changed immutable generation config'
    specification=active_text(original.specification())
    specification+='\n## 현재 실행 범위\n\n'
    specification+='사용자 요청으로 active seed는42·43 두 개다. 기존 seed44 원자료는 보존하지만 신규 생성·집계·뷰어 선택에서 제외한다. '
    specification+='고정600 query×3조건×2seed=3,600개 최종 선택이 목표다. 영상 SHA-256 modulo3으로 세 GPU worker에 겹치지 않게 배정한다. '
    specification+='worker마다 Torch threads2와 별도 processor의 CPU 입력 prefetch1개를 사용한다. 입력·모델·raw 확률·budget·RNG·single-token rollout 연산은 기존 동결 코드와 동일하다. '
    specification+='`metadata/config.json`은 최초 생성 설정 원본이며 실제 실행 범위는 `metadata/execution_scope.json`과 `metadata/effective_config.json`이 원본이다.\n'
    (RESULT/'SPEC.md').write_text(specification)
    path=RESULT/'REPORT.md'
    text=active_text(path.read_text())
    text+='\n현재 active seed:42·43. 기존 seed44는 원자료에 남아 있으나 모든 평균·성능·비용·실패율에서 제외한다. '
    text+='실제 실행 분모는600 query 및 최대3,600 반복 선택이다.\n'
    path.write_text(text)
    page=RESULT/'detail.html'
    text=page.read_text().replace('<option>44</option>','')
    text=re.sub(r'<style data-active-scope>.*?</style>','',text,flags=re.S)
    responsive='''<style data-active-scope>
    .cards{grid-template-columns:repeat(4,minmax(0,1fr))}
    .card,.mode,.status,.muted,.evidence,#heading{min-width:0;overflow-wrap:anywhere}
    .mode-grid{grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr))}
    .controls>*{max-width:100%}pre{max-width:100%}
    @media(max-width:850px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}
    </style>'''
    text=text.replace('</head>',responsive+'</head>')
    page.write_text(text)
    script=OUT/'web/viewer.js'
    text=script.read_text().replace('${s.completed_conditions??0}/5400','${s.completed_conditions??0}/${s.total_conditions??3600}')
    text=text.replace("if(params.has('seed'))$('seed').value=params.get('seed');", "if(['42','43'].includes(params.get('seed')))$('seed').value=params.get('seed');")
    script.write_text(text)
    write(OUT/'metrics/execution_scope.json',dict(seeds=active['seeds'],excluded_seeds=active['excluded_seeds'],
        queries=600,total_conditions=3600,scope_hash=digest(active)))


if __name__=='__main__':
    report('--initialize' in sys.argv)
