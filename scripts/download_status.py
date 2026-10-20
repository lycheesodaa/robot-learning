#!/usr/bin/env python3
"""Read-only status; completion manifests indicate successful integrity verification."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for name,c in json.loads((ROOT/'configs/models.json').read_text()).items():
    folder=ROOT/'artifacts/models'/name/c['revision']
    inventory=folder/'download-inventory.json'
    if not inventory.exists():
        print(f'{name}: not started')
        continue
    files=json.loads(inventory.read_text())['files']
    total=sum(f['size'] for f in files)
    received=0
    for f in files:
        path=folder/f['path']
        if path.is_file() and path.stat().st_size==f['size']:
            received+=f['size']
        else:
            parts=folder/'.http-parts'/f['path']
            if parts.exists():
                received+=sum(p.stat().st_size for p in parts.iterdir() if p.is_file() and p.name.isdigit())
    state='VERIFIED COMPLETE' if (folder/'snapshot-manifest.json').exists() else 'downloading/unverified'
    print(f'{name}: {received/1e9:.2f}/{total/1e9:.2f} GB in complete files/parts; {state}')
