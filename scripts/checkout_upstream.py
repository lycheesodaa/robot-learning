#!/usr/bin/env python3
"""Checkout pinned source, retrying incomplete fetches without overwriting changes."""
import argparse
import json
from pathlib import Path
import subprocess
import time
ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT/'configs/upstream-lock.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=list(lock))
    args = p.parse_args()
    item = lock[args.name]
    folder = ROOT/'artifacts/upstream'/args.name
    if not folder.exists():
        folder.mkdir(parents=True)
        subprocess.run(['git','init','--quiet',str(folder)],check=True)
        subprocess.run(['git','-C',str(folder),'remote','add','origin',item['repository']],check=True)
    origin = subprocess.check_output(['git','-C',str(folder),'remote','get-url','origin'],text=True).strip()
    if origin != item['repository']:
        raise SystemExit('Existing checkout has a different origin; preserve it and resolve manually')
    dirty = subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip()
    if dirty:
        raise SystemExit('Existing checkout has changes; preserve them and resolve manually')
    head = subprocess.run(['git','-C',str(folder),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
    if head.returncode == 0:
        if head.stdout.strip() != item['revision']:
            raise SystemExit('Existing checkout is a different revision; preserve it and resolve manually')
    else:
        for attempt in range(3):
            fetch = subprocess.run(['git','-c','http.version=HTTP/1.1','-C',str(folder),
                                    'fetch','--depth','1','origin',item['revision']])
            if fetch.returncode == 0:
                break
            if attempt == 2:
                raise SystemExit('Pinned fetch failed; rerun to retry this empty checkout')
            time.sleep(2**attempt)
        subprocess.run(['git','-C',str(folder),'checkout','--detach','FETCH_HEAD'],check=True)
    print(f'{args.name}: {item["revision"]} at {folder}')


if __name__ == '__main__':
    main()
