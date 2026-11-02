"""Record reproducibility hashes of code, manifests, metrics and predictions."""
from result_layout import relocate_path, write_result_text, result_glob
from pathlib import Path
import hashlib,json
ROOT=relocate_path(Path(__file__)).resolve().parents[1]
paths=[]
for pattern in ['scripts/*.py','scripts/*.sh','scripts/video_gallery/*','tests/*.py','configs/*','data/*.json','results/pilot_100/*/*.json','results/pilot_100/*/*.jsonl','results/pilot_100/*/*.txt','reports/pilot_100/*.json','reports/pilot_100/*.csv','reports/pilot_100/REPORT.md','reports/pilot_100/*.png','reports/pilot_100/*.pdf','reports/pilot_100/*.html','reports/pilot_100/cases/*','reports/pilot_100/cases/media/*.mp4','README.md']:
    paths.extend(result_glob(ROOT, pattern))
checks={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name!='artifact_sha256.json'}
write_result_text((relocate_path(ROOT/'reports/pilot_100/artifact_sha256.json')), json.dumps(checks,indent=2)+'\n')
print(f'Hashed {len(checks)} artifacts; weights are separately indexed by revision/config metadata in the backbone registry.')
