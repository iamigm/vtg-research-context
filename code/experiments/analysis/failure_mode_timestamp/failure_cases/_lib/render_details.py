"""Reproduce targeted visual reviews recorded in manual_review.json."""
from result_layout import relocate_path, write_result_text
import json
from pathlib import Path
import decord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
DEST = relocate_path(ROOT / 'reports/pilot_100/cases')
reviews = json.loads((relocate_path(DEST.parent / 'manual_review.json')).read_text())
samples = {r['sample_id']: r for r in json.loads((relocate_path(ROOT / 'data/samples_100.json')).read_text())}
for review in reviews:
    if not review.get('detail_timestamps'):
        continue
    sid = review['sample_id']
    video = decord.VideoReader(samples[sid]['video_path'])
    fps = video.get_avg_fps()
    indices = [min(len(video)-1, max(0, round(t*fps))) for t in review['detail_timestamps']]
    frames = video.get_batch(indices).asnumpy()
    nrows = (len(frames)+3)//4
    fig, axes = plt.subplots(nrows, 4, figsize=(14, 3*nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, frame, idx in zip(axes.flat, frames, indices):
        ax.imshow(frame)
        ax.set_title(f'{idx/fps:.2f}s')
    fig.suptitle(sid+' | Targeted GT / prediction review; not model input frames')
    fig.tight_layout()
    key = sid.replace('/', '__')+'_detail'
    fig.savefig(relocate_path(DEST / (key+'.jpg')), dpi=160)
    plt.close(fig)
    write_result_text((relocate_path(DEST / (key+'.json'))), json.dumps({
        'sample_id': sid, 'review_timestamps': [idx/fps for idx in indices],
        'selection': 'Targeted visual inspection around GT and prediction; not model input frames.'
    }, indent=2))
    print(sid, flush=True)
