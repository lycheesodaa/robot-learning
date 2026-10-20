#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description='List or selectively download a revision-pinned dataset')
    p.add_argument('repo_id')
    p.add_argument('--list', action='store_true')
    p.add_argument('--revision')
    p.add_argument('--include', nargs='+')
    args = p.parse_args()
    if not args.list and (not args.revision or not args.include):
        p.error('Downloads require --revision FULL_SHA and --include patterns')
    info = HfApi().dataset_info(args.repo_id, revision=args.revision or 'main', files_metadata=True)
    if args.list:
        print(json.dumps({'repo_id':args.repo_id,'revision':info.sha,'files':[
            {'path':f.rfilename,'size':f.size} for f in info.siblings]},indent=2))
        return
    if args.revision != info.sha:
        p.error('--revision must be the full immutable commit SHA')
    selected = [f.rfilename for f in info.siblings if any(fnmatch.fnmatchcase(f.rfilename,pat) for pat in args.include)]
    if not selected:
        p.error('No files match the requested patterns')
    folder = ROOT/'artifacts/datasets'/args.repo_id.replace('/','--')/info.sha
    snapshot_download(args.repo_id, repo_type='dataset', revision=info.sha, allow_patterns=selected, local_dir=folder)
    (folder/'selection-manifest.json').write_text(json.dumps({'repo_id':args.repo_id,'revision':info.sha,'files':selected},indent=2)+'\n')


if __name__ == '__main__':
    main()
