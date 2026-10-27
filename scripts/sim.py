#!/usr/bin/env python3
"""Closed-loop SO-101 MuJoCo fragile-release rollouts: apparatus checks, single runs, matched suites and suite reports."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))

from vla_safety.archive import RunArchive
from vla_safety.simulation.environment import load_config
from vla_safety.simulation.runner import run_checks, run_episode, run_suite, target_specs, record_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('check', 'Validate MuJoCo grasp, controls, cameras and scripted releases'),
                            ('rollout', 'Run one closed-loop simulated release'),
                            ('suite', 'Run matched simulated release experiments')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--config', type=Path, default=ROOT / 'configs/simulation/fragile_release.json')
        p.add_argument('--output', type=Path, help='New directory; existing output is never overwritten')
        p.add_argument('--device', help='Explicit policy device override, e.g. cuda:0 or cpu')
        p.add_argument('--controller', choices=['molmoact2', 'scripted', 'scripted-high'], default='molmoact2')
        p.add_argument('--no-render', action='store_true', help='Physics-only scripted diagnostics; no video or model observations')
        if name == 'rollout':
            p.add_argument('--height', choices=['low', 'high'], default='high')
            p.add_argument('--fragility', choices=['fragile', 'durable'], default='fragile')
            p.add_argument('--rule', choices=['implicit', 'explicit'], default='explicit')
            p.add_argument('--seed', type=int, default=1000)
            p.add_argument('--layout', help='Layout id from the JSON configuration')
            p.add_argument('--condition', choices=['baseline','zero','top_neuron','top_group','random_group','whole_component'], default='baseline')
    p = sub.add_parser('report', help='Per-cell distributions and paired contrasts for a finished suite (no model or simulator)')
    p.add_argument('suite', type=Path)
    args = parser.parse_args()
    if args.command == 'report':
        from vla_safety.simulation.report import write_report
        for path in write_report(args.suite).values():
            print(path)
        return
    config = load_config(args.config)
    if args.device:
        config['policy']['device'] = args.device
    if args.no_render and args.command != 'check' and args.controller == 'molmoact2':
        parser.error('MolmoAct2 needs rendered observations; --no-render is for scripted diagnostics')
    output = args.output or RunArchive.new_path(ROOT / config['output_root'], args.command)
    output = output.resolve()
    print(f'Artifacts: {output}', flush=True)
    if args.command == 'check':
        result = run_checks(config, ROOT, output, render=not args.no_render)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, args.command, result)
        if not result['ok']: raise SystemExit(1)
    elif args.command == 'suite':
        result = run_suite(config, ROOT, output, controller=args.controller, render=not args.no_render)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, args.command, result)
        if result['status'] != 'completed': raise SystemExit(1)
    else:
        layout = next((x for x in config['suite']['layouts'] if x['id'] == args.layout), None) if args.layout else config['suite']['layouts'][0]
        if layout is None: parser.error('Unknown layout id')
        policy = None; specs = []; descriptor = {'condition': 'baseline'}
        if args.condition != 'baseline':
            if args.controller != 'molmoact2': parser.error('Suppression requires MolmoAct2')
            from vla_safety.policy import load_policy
            policy, _ = load_policy(config['policy'], ROOT)
            choices = target_specs(config, ROOT, policy)
            if args.condition not in choices: parser.error('Set interventions.ranking_archive to a matching concept archive')
            specs, descriptor = choices[args.condition]
        result, _ = run_episode(config, ROOT, output, height=args.height, fragile=args.fragility == 'fragile',
                                rule=args.rule, seed=args.seed, layout=layout, controller=args.controller,
                                policy=policy, specs=specs, descriptor=descriptor, render=not args.no_render)
        print(json.dumps({k:v for k,v in result.items() if k != 'trace'}, indent=2))
        record_progress(config, ROOT, output, args.command, result)
        if result['termination'] in ('error','cancelled'): raise SystemExit(1)


if __name__ == '__main__':
    main()
