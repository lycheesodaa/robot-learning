#!/usr/bin/env python3
"""Download immutable complete HF snapshots; verify sizes and LFS SHA256 offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(folder, manifest):
    for entry in manifest['files']:
        path = folder / entry['path']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise RuntimeError(f'Missing or wrong size: {path}')
        if entry.get('sha256') and digest(path) != entry['sha256']:
            raise RuntimeError(f'SHA256 mismatch: {path}')
    index = json.loads((folder / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (folder / shard).is_file():
            raise RuntimeError(f'Missing shard: {shard}')
    stats = json.loads((folder / 'norm_stats.json').read_text())
    if manifest['norm_tag'] not in stats.get('metadata_by_tag', {}):
        raise RuntimeError(f'Normalization tag missing: {manifest["norm_tag"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all', 'libero', 'so101'], default='all')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    configs = json.loads((ROOT / 'configs/models.json').read_text())
    for name, config in configs.items():
        if args.model not in ('all', name):
            continue
        folder = ROOT / 'artifacts/models' / name / config['revision']
        manifest_path = folder / 'snapshot-manifest.json'
        if args.verify_only:
            manifest = json.loads(manifest_path.read_text())
            if manifest['revision'] != config['revision'] or manifest['repo_id'] != config['repo_id']:
                raise RuntimeError('Manifest does not match configured model')
        else:
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().model_info(config['repo_id'], revision=config['revision'], files_metadata=True)
            if info.sha != config['revision']:
                raise RuntimeError('Revision mismatch')
            manifest = {k: config[k] for k in ('repo_id', 'revision', 'norm_tag')}
            manifest['files'] = [{'path': f.rfilename, 'size': f.size,
                                  'sha256': f.lfs.sha256 if f.lfs else None}
                                 for f in info.siblings]
            folder.mkdir(parents=True, exist_ok=True)
            # Save inventory before download so interrupted runs can be audited.
            (folder / 'download-inventory.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'Downloading {name}: {sum(f["size"] for f in manifest["files"])/1e9:.2f} GB', flush=True)
            snapshot_download(config['repo_id'], revision=config['revision'], local_dir=folder, max_workers=4)
        print(f'Verifying {name} (including weight hashes)...', flush=True)
        verify(folder, manifest)
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2)+'\n')
        tmp.replace(manifest_path)
        print(f'COMPLETE {name}: {folder}', flush=True)


if __name__ == '__main__':
    main()
