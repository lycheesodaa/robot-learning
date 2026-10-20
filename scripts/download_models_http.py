#!/usr/bin/env python3
"""Resumable HTTP range fallback for networks where Xet transfers stall.

Downloads fixed 64 MiB parts independently, validates Content-Range, then checks
complete files against Hub hashes. Never interprets Xet partial files as prefixes.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from download_models import ROOT, verify, digest

PART_SIZE = 64 * 1024 * 1024


def transfer_part(url, target, start, end, total):
    length = end-start+1
    if target.exists() and target.stat().st_size == length:
        return
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                    raise RuntimeError('Server returned an unexpected byte range')
                tmp = target.with_suffix('.tmp')
                with tmp.open('wb') as output:
                    shutil.copyfileobj(response, output, length=1024*1024)
                if tmp.stat().st_size != length:
                    raise RuntimeError('Incomplete range response')
                tmp.replace(target)
                return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 16))


def main():
    import argparse
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    configs = json.loads((ROOT/'configs/models.json').read_text())
    jobs = []
    manifests = []
    for name,c in configs.items():
        folder = ROOT/'artifacts/models'/name/c['revision']
        folder.mkdir(parents=True,exist_ok=True)
        info = HfApi().model_info(c['repo_id'],revision=c['revision'],files_metadata=True)
        if info.sha != c['revision']:
            raise RuntimeError('Revision mismatch')
        manifest = {k:c[k] for k in ('repo_id','revision','norm_tag')}
        manifest['files'] = [{'path':f.rfilename,'size':f.size,'sha256':f.lfs.sha256 if f.lfs else None}
                             for f in info.siblings]
        (folder/'download-inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
        manifests.append((folder,manifest))
        for entry in manifest['files']:
            dest = folder/entry['path']
            if dest.exists() and dest.stat().st_size == entry['size']:
                if not entry['sha256'] or digest(dest) == entry['sha256']:
                    continue
            if entry['size'] < PART_SIZE:
                hf_hub_download(c['repo_id'],entry['path'],revision=c['revision'],local_dir=folder)
                continue
            parts = folder/'.http-parts'/entry['path']
            parts.mkdir(parents=True,exist_ok=True)
            url = f'https://huggingface.co/{c["repo_id"]}/resolve/{c["revision"]}/{entry["path"]}'
            jobs.append((url,dest,parts,entry))
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for url,dest,parts,entry in jobs:
            for start in range(0,entry['size'],PART_SIZE):
                end = min(start+PART_SIZE,entry['size'])-1
                tasks.append(pool.submit(transfer_part,url,parts/str(start),start,end,entry['size']))
        completed = 0
        for future in concurrent.futures.as_completed(tasks):
            future.result()
            completed += 1
            if completed % 16 == 0 or completed == len(tasks):
                print(f'Parts verified: {completed}/{len(tasks)}',flush=True)
    for url,dest,parts,entry in jobs:
        tmp = dest.with_suffix('.assembling')
        h = hashlib.sha256()
        with tmp.open('wb') as output:
            for start in range(0,entry['size'],PART_SIZE):
                with (parts/str(start)).open('rb') as part:
                    for block in iter(lambda:part.read(8*1024*1024),b''):
                        output.write(block)
                        h.update(block)
        if tmp.stat().st_size != entry['size'] or h.hexdigest() != entry['sha256']:
            raise RuntimeError(f'Hash mismatch assembling {dest}; preserve parts for diagnosis')
        tmp.replace(dest)
        shutil.rmtree(parts)
        print(f'Assembled and SHA256 verified {dest.name}',flush=True)
    for folder,manifest in manifests:
        verify(folder,manifest)
        temp = folder/'snapshot-manifest.tmp'
        temp.write_text(json.dumps(manifest,indent=2)+'\n')
        temp.replace(folder/'snapshot-manifest.json')
        print(f'COMPLETE {folder}',flush=True)


if __name__ == '__main__':
    main()
