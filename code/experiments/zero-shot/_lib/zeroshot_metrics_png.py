"""Render legacy predictions metrics after checking against source records."""
from result_layout import relocate_path, write_result_text
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from timelens_store import iter_predictions

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
METRICS = ('miou', 'r03', 'r05', 'r07')
NAMES = {'charades-timelens': 'Charades', 'activitynet-timelens': 'ActivityNet',
         'qvhighlights-timelens': 'QVHighlights'}


def main():
    source = relocate_path(ROOT / 'results/input_order/benchmark_metrics.csv')
    with source.open() as f:
        rows = list(csv.DictReader(f))
    cache = {}
    for r in rows:
        key = (r['model'], r['order'], r['cohort'])
        if key not in cache:
            cache[key] = list(iter_predictions(key[0], key[1],
                cohort='full' if key[2] == 'full_9404' else 'pilot_300'))
        records = [x for x in cache[key] if x['dataset'] == r['dataset']]
        assert len(records) == int(r['n'])
        values = [statistics.mean(x['iou'] for x in records)]
        values += [statistics.mean(x['iou'] >= t for x in records) for t in (.3, .5, .7)]
        for metric, value in zip(METRICS, values):
            assert abs(value - float(r[metric])) < 1e-12, (key, metric)

    out = relocate_path(ROOT / 'predictions/quantitative_results_v1')
    out.mkdir(parents=True, exist_ok=True)
    for cohort, filename, title, subtitle in [
        ('full_9404', 'full_benchmark.png', 'Zero-shot | Full benchmark',
         'Qwen3 input-order comparison | 9,404 samples per order'),
        ('pilot_300', 'pilot300.png', 'Zero-shot | Pilot 300',
         'Qwen2.5 vs Qwen3 | Same 300 samples (100 per benchmark)')]:
        selected = [r for r in rows if r['cohort'] == cohort]
        data = [[NAMES[r['dataset']], 'Qwen2.5' if r['model'] == 'qwen25' else 'Qwen3',
                 r['order'], f"{int(r['n']):,}"] +
                [f'{100 * float(r[m]):.2f}' for m in METRICS] for r in selected]
        fig, ax = plt.subplots(figsize=(15, 5.2))
        fig.patch.set_facecolor('#f8fafc')
        ax.axis('off')
        fig.text(.045, .89, title, fontsize=23, weight='bold', color='#0f172a')
        fig.text(.045, .82, subtitle, fontsize=12, color='#475569')
        table = ax.table(cellText=data, colLabels=['Benchmark', 'Model', 'Input order', 'N',
                         'mIoU (%)', 'R@1 .3 (%)', 'R@1 .5 (%)', 'R@1 .7 (%)'],
                         colWidths=[.16, .10, .16, .08, .12, .126, .126, .126],
                         cellLoc='center', bbox=[0, .1, 1, .72])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('#e2e8f0')
            if i == 0:
                cell.set_facecolor('#1e293b')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#ffffff' if (i - 1) // 3 == 0 else '#eaf1f8')
                cell.set_text_props(color='#0f172a')
        fig.text(.045, .065, 'Source: zero-shot/predictions (legacy baselines) | All metrics in %; higher is better.',
                 fontsize=10, color='#475569')
        fig.subplots_adjust(left=.045, right=.955, top=.86, bottom=.12)
        path = relocate_path(out / filename)
        fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'Saved {path}', flush=True)
    print(f'Validated all {len(rows)} metric rows against original predictions.', flush=True)


if __name__ == '__main__':
    main()
