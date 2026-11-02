"""Build an offline, portable video viewer without changing inference artifacts."""
from result_layout import relocate_path, write_result_text, copy_result_file
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = relocate_path(Path(__file__)).resolve().parents[1]
DEST = relocate_path(ROOT / 'reports/pilot_100/cases')
ASSETS = relocate_path(relocate_path(Path(__file__)).parent / 'video_gallery')

def build():
    cases = json.loads((relocate_path(DEST / 'selection.json')).read_text())['cases']
    reviews = {r['sample_id']: r for r in json.loads((relocate_path(DEST.parent / 'manual_review.json')).read_text())}
    predictions = {}
    baseline = relocate_path(ROOT.parents[1] / 'failure_mode_baseline/failure_cases')
    for root, variant in [(baseline, 'baseline'), (ROOT, 'timestamp')]:
        for model in (['qwen25', 'qwen3'] if variant=='baseline' else ['qwen25']):
            for row in map(json.loads, (relocate_path(root / 'results/pilot_100' / model / 'predictions.jsonl')).read_text().splitlines()):
                predictions[(row['sample_id'], variant, model)] = row
    media = relocate_path(DEST / 'media')
    media.mkdir(exist_ok=True)
    manifest_path = relocate_path(DEST / 'video_manifest.json')
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    ffmpeg = os.environ.get('FFMPEG', '/home/irteam/envs/dit/bin/ffmpeg')
    for case in cases:
        assert {(m['model_key'],m['variant']) for m in case['models']} == {('qwen25','baseline'),('qwen25','timestamp'),('qwen3','baseline')}
        key = case['sample_id'].replace('/', '__')
        source = relocate_path(Path(case['video_path']))
        target = media / (key + '.mp4')
        stat = source.stat()
        fingerprint = {'path': str(source.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        cached = old.get(case['sample_id'], {})
        if not target.exists() or cached.get('source') != fingerprint:
            tmp = target.with_suffix('.tmp.mp4')
            # Selected sources are H.264. Preserve video frames and source timing;
            # remux with faststart for seeking, exclude audio (models saw no audio).
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                            '-map', '0:v:0', '-c:v', 'copy', '-an', '-movflags', '+faststart', str(tmp)], check=True)
            tmp.replace(target)
        case['media'] = 'media/' + target.name
        case['key'] = key
        case['review'] = reviews.get(case['sample_id'])
        for model in case['models']:
            row = predictions[(case['sample_id'], model['variant'], model['model_key'])]
            model['raw_timestamps'] = row['parsed_timestamps']
        manifest[case['sample_id']] = {'source': fingerprint, 'media': case['media'],
            'size': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'conversion': 'H.264 video stream copy, faststart, no audio; no crop or frame resampling'}
    write_result_text((relocate_path(DEST / 'viewer_data.js')), 'window.VIDEO_CASES = ' + json.dumps(cases, ensure_ascii=False) + ';\n')
    write_result_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name in ['report.html', 'viewer.css', 'viewer.js', 'font.otf', 'FONT_LICENSE.txt']:
        copy_result_file(relocate_path(ASSETS / name), relocate_path(DEST / name))
    print(f'Video gallery: {len(cases)} cases, {sum(r["size"] for r in manifest.values())/1e6:.1f} MB')

if __name__ == '__main__':
    build()
