"""Download pinned joint-space backbones into the shared registry."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from huggingface_hub import snapshot_download

ROOT = Path('/home/irteam/backbones')
MODELS = [
    ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41'),
    ('Qwen/Qwen3-VL-Embedding-8B', '2c4565515e0f265c6511776e7193b22c0968ddc7'),
]

def main():
    entries = []
    for repo, revision in MODELS:
        name = repo.split('/')[-1]
        path = ROOT / 'models' / name
        print('Downloading', repo, revision, flush=True)
        snapshot_download(repo, revision=revision, local_dir=path,
                          allow_patterns=['*.json', '*.txt', '*.jinja', '*.md', '*.safetensors', 'scripts/*.py'],
                          max_workers=4)
        weights = sorted(path.glob('*.safetensors'))
        index = path / 'model.safetensors.index.json'
        expected = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        missing = [x for x in expected if not (path/x).is_file()]
        assert not missing, missing
        # Full hashes make persistent downloads auditable, beyond shard existence.
        hashes = {}
        for p in weights:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(8*1024**2), b''):
                    h.update(block)
            hashes[p.name] = h.hexdigest()
        entries.append(dict(name=name, repo_id=repo, revision=revision, path=str(path),
            resolved_path=str(path.resolve()), action='downloaded for similarity pilot',
            is_symlink=False, weight_files=len(weights), missing_weight_files=missing,
            weight_bytes_present=sum(p.stat().st_size for p in weights), complete_shard_set=True,
            metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.json')},
            weight_sha256=hashes))
        print('Verified', name, flush=True)
    manifest = dict(downloaded_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), models=entries)
    (ROOT/'manifests/similarity_download.json').write_text(json.dumps(manifest, indent=2)+'\n')
    registry_path = ROOT/'manifests/registry.json'
    registry = json.loads(registry_path.read_text())
    names = {e['name'] for e in entries}
    registry = [e for e in registry if e['name'] not in names] + entries
    tmp = registry_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(registry_path)

if __name__ == '__main__':
    main()
