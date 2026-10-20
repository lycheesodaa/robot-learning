#!/usr/bin/env python3
"""Offline official sample inference or dry-run prediction from a wire-v1 recording."""
import argparse
import json
import os
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
from PIL import Image
from vla_safety.types import Observation
from vla_safety.policies.molmoact2 import MolmoAct2Policy
from vla_safety.policies.wire import decode_observation, encode_action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['libero', 'so101'], required=True)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--device', default='cuda')
    p.add_argument('--recording', type=Path)
    p.add_argument('--list-modules', action='store_true')
    p.add_argument('--seed', type=int, default=1000)
    args = p.parse_args()
    import torch
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Run scripts/diagnose.py and repair GPU access before inference.')
    c = json.loads((ROOT/'configs/models.json').read_text())[args.model]
    folder = ROOT/'artifacts/models'/args.model/c['revision']
    policy = MolmoAct2Policy(c, folder, args.device, args.dtype)
    if args.list_modules:
        print('\n'.join(name for name, _ in policy.model.named_modules()))
        return
    obs = (decode_observation(json.loads(args.recording.read_text())) if args.recording else
           Observation(tuple(Image.open(folder/'assets'/name).convert('RGB') for name in c['sample_images']),
                       c['sample_instruction'], c['sample_state'], c['embodiment'], time.time()))
    print(json.dumps(encode_action(policy.predict(obs)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
