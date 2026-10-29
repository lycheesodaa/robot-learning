#!/usr/bin/env python3
"""Fragile pick-and-place with the v6 LoRA MolmoAct2 in so101-nexus: scene survey, single rollouts and suites.

Runs in .envs/lerobot (Python 3.12, LeRobot 0.6.1, so101-nexus 0.4.8, MuJoCo 3.10.0).
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from vla_safety.archive import RunArchive
from vla_safety.pickplace.environment import PickPlaceEnvironment, load_config
from vla_safety.pickplace.runner import run_episode, run_suite
from vla_safety.simulation.runner import record_progress

DEFAULT = ROOT / 'configs/simulation/fragile_pickplace.json'


def narrow(config, args):
    """Run a subset of the configured factors; the archived config records what ran."""
    suite = config['suite']
    if args.layouts:
        known = {x['id']: x for x in suite['layouts']}
        if not set(args.layouts) <= set(known): raise SystemExit(f'Unknown layout ids; configured: {sorted(known)}')
        suite['layouts'] = [known[x] for x in args.layouts]
    if args.fragilities: suite['fragilities'] = [x == 'fragile' for x in args.fragilities]
    if args.rules: suite['rules'] = args.rules
    if args.seeds: suite['seeds'] = args.seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('survey', help='Cube/disc positions and first frames for candidate scene seeds (no model)')
    p.add_argument('scene_seeds', type=int, nargs='+')
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path, required=True)
    for name, help_text in [('rollout', 'Run one closed-loop episode'), ('suite', 'Run matched fragility x rule x seed episodes')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--config', type=Path, help='Default: configs/simulation/fragile_pickplace.json, or the saved config when resuming')
        p.add_argument('--output', type=Path, help='New directory; existing output is never overwritten')
        p.add_argument('--device', help='Policy device override, e.g. cuda:1')
        if name == 'rollout':
            p.add_argument('--fragility', choices=['fragile', 'durable'], default='fragile')
            p.add_argument('--rule', choices=['implicit', 'explicit'], default='implicit')
            p.add_argument('--seed', type=int, default=1000)
            p.add_argument('--layout', help='Layout id from the JSON configuration')
            p.add_argument('--base-prompt-only', action='store_true',
                           help="Send task.instruction without the property word (v6's training prompt); --fragility then only labels scoring")
        else:
            p.add_argument('--resume', action='store_true', help='Continue the suite in --output')
            p.add_argument('--layouts', nargs='+', help='Subset of layout ids')
            p.add_argument('--fragilities', nargs='+', choices=['fragile', 'durable'])
            p.add_argument('--rules', nargs='+', choices=['implicit', 'explicit'])
            p.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    if args.command == 'survey':
        from PIL import Image
        import numpy as np
        config = load_config(args.config); args.output.mkdir(parents=True, exist_ok=False); rows = []
        for seed in args.scene_seeds:
            env = PickPlaceEnvironment(config, {'id': f'scene{seed}', 'scene_seed': seed, 'target_xy_offset_m': [0, 0], 'split': 'discovery'})
            try:
                env.reset(); rows.append(env.initialization)
                Image.fromarray(np.concatenate(env.images(), axis=1)).save(args.output / f'scene{seed}.png')
            finally: env.close()
            print(json.dumps({k: rows[-1][k] for k in ('scene_seed', 'cube_xyz_m', 'target_xyz_m', 'cube_radius_m', 'in_reach')}))
        (args.output / 'survey.json').write_text(json.dumps(rows, indent=2))
        return
    resume = getattr(args, 'resume', False)
    if resume and not (args.output and (args.output / 'config.json').exists()):
        parser.error('--resume needs --output pointing at an existing suite')
    config = load_config(args.config or (args.output / 'config.json' if resume else DEFAULT))
    if args.device: config['policy']['device'] = args.device
    output = (args.output or RunArchive.new_path(ROOT / config['output_root'], args.command)).resolve()
    print(f'Artifacts: {output}', flush=True)
    if args.command == 'suite':
        if not resume: narrow(config, args)
        elif any((args.layouts, args.fragilities, args.rules, args.seeds)): parser.error('Factor subsets come from the saved config when resuming')
        result = run_suite(config, ROOT, output, resume=resume)
        print(json.dumps(result, indent=2))
        print(json.dumps(json.loads((output / 'summary.json').read_text()), indent=2))
        record_progress(config, ROOT, output, 'pickplace suite', result,
                        f"suite status={result['status']}; episodes={result['episodes']}; factors="
                        f"layouts={[x['id'] for x in config['suite']['layouts']]}, fragile={config['suite']['fragilities']}, "
                        f"rules={config['suite']['rules']}, seeds={config['suite']['seeds']}.")
        if result['status'] != 'completed': raise SystemExit(1)
    else:
        layout = next((x for x in config['suite']['layouts'] if x['id'] == args.layout), None) if args.layout else config['suite']['layouts'][0]
        if layout is None: parser.error('Unknown layout id')
        result = run_episode(config, ROOT, output, fragile=args.fragility == 'fragile', rule=args.rule, seed=args.seed, layout=layout,
                             base_prompt_only=args.base_prompt_only)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, 'pickplace rollout', result,
                        f"prompt={result['condition']['instruction']!r}; seed={args.seed}; "
                        f"termination={result['termination']}; grasped={result.get('grasped')}; "
                        f"first release height={result.get('first_release_height_m')} m ({result.get('first_release_kind')}); "
                        f"placement={result.get('task_success')}; broken={result.get('broken')}.")
        if result['termination'] in ('error', 'cancelled'): raise SystemExit(1)


if __name__ == '__main__':
    main()
