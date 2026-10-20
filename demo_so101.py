#!/usr/bin/env python3
"""One offline SO-101 observation -> action chunk; no robot connection or actuation.

Run with .envs/policy/bin/python demo_so101.py --device cuda:0
Open artifacts/so101_demo/report.html to inspect the input and output together.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))

import numpy as np
from PIL import Image
import torch
from vla_safety.policies.molmoact2 import MolmoAct2Policy
from vla_safety.policies.wire import encode_action
from vla_safety.types import Observation


def write_report(folder, inputs, result, names):
    """Portable HTML with the original sample images and all predicted steps."""
    esc = html.escape
    images = ''.join(
        f'<figure><img src="{esc(name, quote=True)}" alt="Camera {i}">'
        f'<figcaption>{esc(name)}</figcaption></figure>'
        for i, name in enumerate(inputs['images'])
    )
    headers = '<th>Step</th>' + ''.join(f'<th>{esc(n)}</th>' for n in names)
    rows = ''.join('<tr><td>'+str(i)+'</td>'+''.join(f'<td>{v:.5f}</td>' for v in row)+'</tr>'
                   for i, row in enumerate(result['values']))
    state = ', '.join(f'{name}={value:.5f}' for name, value in zip(names, inputs['state']))
    content = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101: single input-output run</title>
<style>
body {{font:16px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}
.cameras {{display:flex;flex-wrap:wrap;gap:20px}} figure {{margin:0;flex:1;min-width:250px}}
img {{max-width:100%;height:auto}} figcaption, .metadata {{color:#526075;font-size:14px}}
pre {{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:16px}}
table {{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td {{text-align:right;padding:8px;border-bottom:1px solid #ddd}} .table {{overflow-x:auto}}
</style>
<h1>SO-101: one observation → one action chunk</h1>
<p class="metadata">{esc(result['model_id'])} · revision {esc(result['revision'])}</p>
<h2>Input</h2><div class="cameras">{images}</div>
<p><strong>Instruction:</strong> {esc(inputs['instruction'])}</p>
<p><strong>State order and values:</strong></p><pre>{esc(state)}</pre>
<h2>Output</h2>
<p>{len(result['values'])} predicted future steps × {len(names)} action dimensions.
Control mode: <strong>{esc(result['control_mode'])}</strong>.</p>
<p>Values use the checkpoint's native joint/gripper convention. They are predictions,
not measured motion or a validated motor-command sequence. No robot was connected.</p>
<p class="metadata">Device: {esc(result['device'])} ({esc(result['device_name'])});
dtype: {esc(result['dtype'])}; seed: {result['seed']}; model load: {result['load_seconds']:.2f} s;
first inference: {result['inference_seconds']:.2f} s (no warm-up).</p>
<div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>
<p>Raw files: <a href="input.json">input.json</a> · <a href="prediction.json">prediction.json</a>
· <a href="actions.csv">actions.csv</a></p></html>'''
    (folder / 'report.html').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='cuda:0, cuda:1, or cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--instruction', help='Override the instruction for the same recorded scene')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so101_demo')
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type not in ('cpu', 'cuda'):
        parser.error('Use cpu or a cuda device')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is not visible in this process. Check nvidia-smi in a host terminal; '
                         'a sandbox may hide working GPUs. Use --device cpu for CPU inference.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error('Requested CUDA device does not exist')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.manual_seed(args.seed)

    # 1. Load the fixed checkpoint configuration and its official sample observation.
    config = json.loads((ROOT / 'configs/models.json').read_text())['so101']
    snapshot = ROOT / 'artifacts/models/so101' / config['revision']
    metadata = json.loads((snapshot / 'norm_stats.json').read_text())['metadata_by_tag'][config['norm_tag']]
    names = metadata['action_stats']['names']
    state_names = metadata['state_stats']['names']
    if names != state_names:
        raise ValueError('This demonstration expects matching state/action dimension names')
    instruction = args.instruction if args.instruction is not None else config['sample_instruction']
    image_paths = [snapshot / 'assets' / name for name in config['sample_images']]
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    observation = Observation(tuple(images), instruction, config['sample_state'], 'so101', time.time())
    print('INPUT', flush=True)
    print('Instruction:', instruction, flush=True)
    print('Images:', *(str(p) for p in image_paths), sep='\n  ', flush=True)
    print('State:', dict(zip(state_names, observation.state.tolist())), flush=True)

    # 2. Load the complete local weights; the wrapper disables CUDA graphs for hooks.
    started = time.perf_counter()
    policy = MolmoAct2Policy(config, snapshot, str(device), args.dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    # 3. Exactly one prediction call. This produces a chunk, not a closed-loop rollout.
    started = time.perf_counter()
    action = policy.predict(observation)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    expected = (metadata['action_horizon'], config['action_dim'])
    if action.values.shape != expected or not np.isfinite(action.values).all():
        raise RuntimeError(f'Invalid output shape or values: {action.values.shape}, expected {expected}')

    # 4. Save exact inputs, all actions, provenance, timing and a portable HTML report.
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        target = folder / path.name
        if target.resolve() != path.resolve():
            shutil.copyfile(path, target)
    inputs = {'images': [p.name for p in image_paths], 'instruction': instruction,
              'state': observation.state.tolist(), 'state_names': state_names,
              'embodiment': observation.embodiment, 'timestamp': observation.timestamp}
    result = encode_action(action)
    result.update(model_id=config['repo_id'], revision=config['revision'], action_names=names,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  dtype=args.dtype, seed=args.seed, load_seconds=load_seconds,
                  inference_seconds=inference_seconds,
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None)
    (folder / 'input.json').write_text(json.dumps(inputs, indent=2, allow_nan=False)+'\n')
    (folder / 'prediction.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (folder / 'actions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', *names])
        writer.writerows((i, *row) for i, row in enumerate(action.values.tolist()))
    write_report(folder, inputs, result, names)
    print(f'\nOUTPUT: {action.values.shape[0]} steps x {action.values.shape[1]} dimensions; all finite')
    print('Columns:', ', '.join(names))
    print('First five predicted steps:\n', np.array2string(action.values[:5], precision=5))
    print(f'Model load: {load_seconds:.2f}s; first inference: {inference_seconds:.2f}s')
    print('Report:', folder / 'report.html')
    print('Full output:', folder / 'prediction.json')


if __name__ == '__main__':
    main()
