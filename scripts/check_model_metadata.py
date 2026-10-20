#!/usr/bin/env python3
"""Offline CPU validation of processors and model architecture without weight allocation."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from accelerate import init_empty_weights


def main():
    results = {}
    for name,c in json.loads((ROOT/'configs/models.json').read_text()).items():
        folder=ROOT/'artifacts/models'/name/c['revision']
        config=AutoConfig.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
        processor=AutoProcessor.from_pretrained(folder,trust_remote_code=True,local_files_only=True,use_fast=False)
        cls=get_class_from_dynamic_module(config.auto_map['AutoModelForImageTextToText'],str(folder),local_files_only=True)
        with init_empty_weights():
            model=cls(config)
        stats=json.loads((folder/'norm_stats.json').read_text())['metadata_by_tag'][c['norm_tag']]
        if len(stats['action_stats']['mean']) != c['action_dim'] or len(stats['state_stats']['mean']) != c['state_dim']:
            raise ValueError('State/action dimensions differ from normalization metadata')
        results[name]={'processor':type(processor).__name__,'model':type(model).__name__,
                       'parameter_count':sum(p.numel() for p in model.parameters()),'norm_tag':c['norm_tag'],
                       'action_horizon':stats['action_horizon'],'revision':c['revision']}
        print(f'{name}: processor and architecture OK',flush=True)
        modules=ROOT/'artifacts'/f'{name}-modules.txt'
        modules.write_text('\n'.join(n for n,_ in model.named_modules())+'\n')
        del model,processor
    (ROOT/'artifacts/metadata-validation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
