#!/usr/bin/env python3
"""One entry point for attention extraction, analysis, reporting and validation."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
PYTHON = ROOT / '.venv/bin/python'
DATASETS = ['Charades-STA', 'QVHighlights', 'ActivityNet-Captions']
ANALYSES = {
    'query': ['query_analysis', 'query_plot', 'query_report'],
    'generation': ['generation_analysis', 'generation_plot'],
    'distribution': ['distribution'],
    'heads': ['heads'],
    'dynamics': ['dynamics', 'dynamics_plot', 'dynamics_report'],
    'hidden': ['hidden_analysis'],
    'spatial': ['spatial_analysis', 'spatial_viewer'],
}
CAPTURES = {'query': 'input_pipeline', 'generation': 'generation',
            'hidden': 'hidden_capture', 'spatial': 'spatial_counterfactual'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def invoke(module, extra=()):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / '_lib'), str(ROOT / 'python_deps'),
                                       env.get('PYTHONPATH', '')])
    env.setdefault('MPLBACKEND', 'Agg')
    print(f'Running {module}', flush=True)
    subprocess.run([str(PYTHON), str(ROOT / '_lib' / f'{module}.py'), *extra],
                   cwd=ROOT, env=env, check=True)


def report():
    """Join by sample identity; never pool distinct attention conventions."""
    samples = read(RESULTS / 'samples.json')
    query = {x['sample_key']: x for x in read(RESULTS / 'query/sample_metrics.json')}
    generation = {x['sample_key']: x for x in read(RESULTS / 'generation/sample_metrics.json')}
    with (RESULTS / 'hidden/metrics.csv').open() as stream:
        hidden = {x['sample_key']: x for x in csv.DictReader(stream)
                  if x['scope'] == 'L8_21' and x['temperature'] == '0.1' and x['region'] == 'full'}
    rows = []
    for sample in samples:
        key = sample['sample_key']
        q, g, h = query[key], generation[key], hidden[key]
        for item in (q, g, h):
            assert item['dataset'] == sample['dataset']
        d = read(RESULTS / 'dynamics' / sample['dataset'] / key / 'dynamics.json')
        m = d['scopes']['predictive']['metrics']
        tv = [x for x in m['tv_previous'] if x is not None]
        row = dict(dataset=sample['dataset'], sample_key=key, qid=sample['qid'],
                   query=sample['query'], answer=g['response'],
                   query_gt_mass=q['attention_mass_in_gt'], query_peak_in_gt=q['peak_in_gt'],
                   query_early_peak=q['peak_in_first_10_percent'],
                   answer_gt_mass=g['scopes']['answer']['gt_mass'],
                   answer_peak_in_gt=g['scopes']['answer']['peak_in_gt'],
                   answer_early_peak=g['scopes']['answer']['peak_in_first_10_percent'],
                   predictive_gt_mass=g['scopes']['predictive']['gt_mass'],
                   answer_tokens=g['answer_tokens'], answer_truncated=g['truncated'],
                   best_pair_tiou=g['best_pair_tiou'],
                   dynamics_mean_adjacent_tv=sum(tv)/len(tv) if tv else None,
                   hidden_attention_tv=float(h['tv']), hidden_attention_js_bits=float(h['js_bits']),
                   hidden_attention_spearman=float(h['spearman']))
        rows.append(row)
    assert len(rows) == len(query) == len(generation) == len(hidden) == 300
    write(RESULTS / 'sample_metrics.json', rows)
    with (RESULTS / 'sample_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    gs = read(RESULTS / 'generation/summary.json')
    ds = read(RESULTS / 'dynamics/summary.json')
    hs = read(RESULTS / 'hidden/summary.json')
    summaries = {}
    table = ['| 데이터셋 | 샘플 | Query peak∈GT | Answer peak∈GT | Answer 초반 peak | 토큰 간 TV 중앙값 | Hidden–attention TV 평균 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        g = gs[name]; d = ds[name]['scopes']['predictive']; h = hs[name]['L8_21__tau0.1__full']
        s = dict(samples=g['samples'], query_peak_in_gt_rate=g['scopes']['query']['peak_in_gt_rate'],
                 answer_peak_in_gt_rate=g['scopes']['answer']['peak_in_gt_rate'],
                 answer_early_peak_rate=g['scopes']['answer']['early_peak_rate'],
                 dynamics_median_sample_mean_adjacent_tv=d['median_sample_mean_adjacent_tv'],
                 hidden_attention_mean_tv=h['tv']['mean'])
        summaries[name] = s
        table.append(f"| {name} | {s['samples']} | {s['query_peak_in_gt_rate']:.0%} | "
                     f"{s['answer_peak_in_gt_rate']:.0%} | {s['answer_early_peak_rate']:.0%} | "
                     f"{s['dynamics_median_sample_mean_adjacent_tv']:.3f} | {s['hidden_attention_mean_tv']:.3f} |")
    configs = {stage: read(RESULTS / stage / 'run_config.json') for stage in ['query', 'generation', 'hidden']}
    write(RESULTS / 'summary.json', dict(samples=len(samples), datasets=summaries,
          generated_tokens=sum(x['answer_tokens'] for x in rows),
          conventions={'query': 'input event sentence; L8–21, all heads',
                       'answer': 'consumed generated-token rows',
                       'predictive': 'rows that selected generated tokens',
                       'hidden': 'decoder output cosine, L8–21, temperature 0.1'},
          recorded_run_configs=configs))
    text = '\n'.join(['# MTLA attention 통합 실험 결과', '',
        '같은 Qwen3-VL-8B-Instruct와 같은 300개 입력(데이터셋별 100개)을 사용한 분석을 통합했다. '
        '기존 측정값을 재사용했으며, 통합 과정에서 모델 추론을 새로 수행하지 않았다. '
        '9개 탐색 샘플은 300개 실험의 부분집합이며 모든 저장 배열이 bitwise 동일함을 확인했다.', '',
        '[통합 뷰어](index.html) · [통합 CSV](sample_metrics.csv) · [요약 JSON](summary.json)', '',
        *table, '',
        'Answer는 생성 토큰을 처리하는 consumed 행이다. 토큰 간 TV는 predictive 행에서 '
        '영상 내부 분포를 정규화한 뒤 샘플 내 평균 → 샘플 간 중앙값으로 계산한다. '
        '초반 peak는 영상 앞 10%에 해당한다. Hidden 비교는 cosine을 온도 0.1로 softmax한 분포와 '
        '영상 내부 attention 분포 사이의 TV다. 서로 다른 정의의 수치를 하나의 평균으로 섞지 않는다.', '',
        'Query보다 답변 attention의 최고점이 영상 초반에 더 자주 나타나고, GT 내부 최고점 비율은 낮다. '
        '이 지표만으로 영상 정보의 인과적 사용 여부를 판단할 수는 없다.', '',
        '## 상세 결과', '',
        '- [Query attention](query/index.html)',
        '- [Query–answer 비교](generation/index.html)',
        '- [분포 비교](generation/distribution_analysis/REPORT.md)',
        '- [레이어·head 분석](generation/head_analysis/index.html)',
        '- [생성 토큰별 변화](dynamics/index.html)',
        '- [Hidden similarity 비교](hidden/index.html)',
        '- [공간 attention·query 교체 대조](spatial/index.html)', '',
        '샘플 선택, 전처리, capture 설정은 각 분석의 run_config.json과 원본 메타데이터에 보존했다. '
        '공통 실행 명령은 상위 README.md를 참고한다.', ''])
    (RESULTS / 'REPORT.md').write_text(text)
    links = [('Query', 'query/index.html'), ('Query / answer', 'generation/index.html'),
             ('분포 비교', 'generation/distribution_analysis/REPORT.md'),
             ('Layer / head', 'generation/head_analysis/index.html'), ('Token dynamics', 'dynamics/index.html'),
             ('Hidden similarity', 'hidden/index.html'), ('Spatial', 'spatial/index.html')]
    links = [(label, path) for label, path in links if (RESULTS / path).is_file()]
    cards = ''.join(f'<a href="{path}">{html.escape(label)}</a> ' for label, path in links)
    body = []
    for row in rows:
        key = row['sample_key']; rel = f"{row['dataset']}/{key}"
        body.append('<tr><td>' + html.escape(key) + '</td><td>' + html.escape(row['query']) + '</td><td>' +
                    html.escape(row['answer']) + '</td><td>' +
                    f"{row['query_gt_mass']:.3f} / {row['answer_gt_mass']:.3f}" + '</td><td>' +
                    f'<a href="query/{rel}/attention.png">Query</a> · '
                    f'<a href="generation/{rel}/comparison.png">Answer</a> · '
                    f'<a href="dynamics/{rel}/token_dynamics.png">Tokens</a> · '
                    f'<a href="hidden/{rel}/comparison.png">Hidden</a></td></tr>')
    summary_rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(c.strip())}</td>' for c in line.strip('|').split('|')) + '</tr>' for line in [table[0], *table[2:]])
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>MTLA attention 통합 실험</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:35px auto;padding:0 24px;color:#172b44}a{color:#175ab8}nav a{display:inline-block;padding:10px 18px;background:#eef4fa;margin:4px;border-radius:6px}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}input{font:inherit;padding:10px;width:95%}small{color:#596579}</style>
<h1>MTLA attention 통합 실험</h1><p>Qwen3-VL-8B-Instruct · 동일한 300개 입력 · L8–21 · 기존 측정 결과 통합</p>
<nav>''' + cards + '''</nav><p><a href="REPORT.md">통합 보고서</a> · <a href="sample_metrics.csv">통합 CSV</a> · <a href="summary.json">요약 JSON</a></p>
<table>''' + summary_rows + '''</table><small>Answer: consumed 행. 토큰 간 TV: predictive 행의 샘플 내 평균 후 중앙값. 초반: 영상 앞 10%. Hidden: cosine temperature 0.1.</small>
<h2>샘플별 결과</h2><input id="search" placeholder="데이터셋, query, 답변 또는 샘플 ID 검색"><p id="count">300 samples</p>
<table><thead><tr><th>샘플</th><th>Query</th><th>Answer</th><th>GT mass<br>Query / Answer</th><th>상세 그림</th></tr></thead><tbody id="samples">''' + ''.join(body) + '''</tbody></table>
<script>document.querySelector('#search').addEventListener('input',e=>{let n=0;for(const r of document.querySelectorAll('#samples tr')){r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase());if(!r.hidden)n++}document.querySelector('#count').textContent=n+' samples'});</script></html>'''
    if (RESULTS / 'zeroshot/REPORT.md').exists():
        page = page.replace('<nav>', '<p><strong>공통 zero-shot 결과:</strong> <a href="zeroshot/REPORT.md">Qwen3 전체 9,404쌍 및 Qwen2.5 pilot 300개 · 모델별 결과와 공유 기준선</a></p><nav>', 1)
    (RESULTS / 'index.html').write_text(page)
    print(f'Combined {len(rows)} samples: {RESULTS / "index.html"}', flush=True)


def verify_layout():
    """Read-only checks of migrated raw data, paths and the joined result."""
    migration = read(ROOT / 'provenance/migration.json')
    # Validation summaries may be regenerated. All scientific arrays, sample
    # metadata, answers, and numerical tables remain covered by the migration hash.
    mutable = {'validation.json', 'final_validation.json', 'artifact_validation.json',
               'unit_validation.json', 'browser_validation.json', 'validation_summary.json',
               'analysis_provenance.json'}
    checked = 0
    for rel, digest in migration['preserved_data_sha256'].items():
        p = ROOT / rel
        if len(p.relative_to(RESULTS).parts) <= 2 and p.name in mutable:
            continue
        assert p.is_file(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Changed migrated data: {rel}'
        checked += 1
    samples = read(RESULTS / 'samples.json')
    for stage in ['query', 'generation', 'hidden', 'dynamics']:
        assert read(RESULTS / stage / 'samples.json') == samples
    rows = read(RESULTS / 'sample_metrics.json')
    assert [s['sample_key'] for s in samples] == [r['sample_key'] for r in rows]
    broken = []
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in {'src', 'href'} or not value:
                    continue
                parts = urlsplit(value)
                if parts.scheme or parts.netloc or not parts.path:
                    continue
                target = page.parent / unquote(parts.path)
                if not target.is_file():
                    broken.append((str(page.relative_to(ROOT)), value))
    for page in RESULTS.rglob('*.html'):
        Links().feed(page.read_text())
    assert not broken, broken[:20]
    for p in ROOT.rglob('*'):
        if p.is_symlink():
            assert p.exists(), f'Broken symlink: {p}'
    return dict(preserved_data_files=checked, samples=len(samples), html_links_valid=True,
                symlinks_valid=True, pilot_arrays_identical=len(migration['pilot_absorbed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    timelens = sub.add_parser('timelens', help='TimeLens input-order experiment: answer-to-video attention')
    timelens.add_argument('action', choices=['prepare', 'smoke', 'start', 'status', 'report'])
    zeroshot = sub.add_parser('zeroshot', help='Shared zero-shot results across models and downstream experiments')
    zeroshot.add_argument('action', choices=['report'])
    span_head = sub.add_parser('span-head', help='TimeLens pilot: restricted LM head vs original zero-shot')
    span_head.add_argument('action', choices=['run', 'smoke', 'report'])
    span_head.add_argument('--model', choices=['qwen3', 'qwen25'], default='qwen3')
    span_head.add_argument('--shard', type=int, default=0)
    span_head.add_argument('--shards', type=int, default=2)
    sub.add_parser('report', help='Rebuild the combined report from saved metrics; no inference')
    capture = sub.add_parser('run', help='Resume extraction; completed samples are skipped')
    capture.add_argument('--stage', choices=['all', *CAPTURES], default='all')
    capture.add_argument('--only', nargs='+', help='Sample keys from the shared manifest')
    capture.add_argument('--limit', type=int, help='First N samples of the shared manifest')
    analysis = sub.add_parser('analyze', help='Recompute metrics and plots from saved arrays')
    analysis.add_argument('--stage', choices=['all', *ANALYSES], default='all')
    verify = sub.add_parser('verify', help='Validate migration, saved arrays, metrics, figures and browsers')
    verify.add_argument('--layout-only', action='store_true', help='Only migration hashes, paths and joined result')
    args = parser.parse_args()
    if args.command == 'zeroshot':
        invoke('zeroshot_report')
    elif args.command == 'timelens':
        invoke('timelens_order', [args.action])
    elif args.command == 'span-head':
        if args.action == 'report':
            invoke('timelens_span_report')
        else:
            options = ['--model', args.model, '--shard', str(args.shard), '--shards', str(args.shards)]
            if args.action == 'smoke':
                options.append('--smoke')
            invoke('timelens_span_head', options)
    elif args.command in {None, 'report'}:
        report()
    elif args.command == 'run':
        samples = read(RESULTS / 'samples.json')
        keys = [s['sample_key'] for s in samples]
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        if args.only and set(args.only) - set(keys):
            parser.error('Unknown sample keys: ' + ', '.join(sorted(set(args.only) - set(keys))))
        selected = [k for k in keys[:args.limit] if not args.only or k in args.only]
        if not selected:
            parser.error('No samples selected')
        stages = list(CAPTURES) if args.stage == 'all' else [args.stage]
        for stage in stages:
            if stage == 'spatial' and not (RESULTS / 'spatial/strong_samples.json').exists():
                invoke('spatial_analysis')
            invoke(CAPTURES[stage], ['--only', *selected])
        print('Extraction complete. Rebuild derived metrics with: .venv/bin/python run.py analyze')
    elif args.command == 'analyze':
        stages = list(ANALYSES) if args.stage == 'all' else [args.stage]
        for stage in stages:
            for module in ANALYSES[stage]:
                invoke(module)
        report()
    else:
        layout = verify_layout()
        if not args.layout_only:
            for module in ['query_verify', 'generation_verify', 'hidden_verify',
                           'dynamics_browser', 'dynamics_verify', 'hidden_browser', 'spatial_verify', 'report_browser']:
                invoke(module)
        result = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
                      layout=layout, full_validation=not args.layout_only)
        write(RESULTS / ('layout_validation.json' if args.layout_only else 'validation.json'), result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
