#!/usr/bin/env python3
"""Pen-to-cup transfer past a block tower in a red keep-out zone: apparatus checks, rollouts, suites, reports, prompt checks."""
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
from vla_safety.obstacle.environment import CONTROLLERS, SCENES, load_config
from vla_safety.obstacle.runner import run_checks, run_episode, run_suite
from vla_safety.simulation.runner import record_progress

DEFAULT_CONFIG = ROOT / 'configs/simulation/obstacle_avoidance.json'


def describe(command, result):
    """Factual one-line progress entry; never a research conclusion."""
    if command == 'check':
        failed = [o['episode'] for o in result['scripted'] if not o['passed']]
        return f"apparatus checks passed={result['ok']}" + (f"; failed: {', '.join(failed)}" if failed else '') + '; this does not test policy competence.'
    if command == 'rollout':
        return (f"controller={result['controller']}; arm={result['condition']['arm']}; scene={result['condition']['scene']}; "
                f"termination={result['termination']}; grasped={result.get('grasped')}; crossing={result.get('crossing_mode')}; "
                f"carry incursion={result.get('carry_incursion')}; tower contact={result.get('tower_contact')}; placed={result.get('task_success')}.")
    return f"suite status={result['status']}; episodes={result['episodes']}; inspect per-cell results before drawing conclusions."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('check', 'Validate start states, replay and scoring with scripted direct/over/around paths'),
                            ('rollout', 'Run one closed-loop episode'),
                            ('suite', 'Run layouts × scenes × seeds × arms from shared start states')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--config', type=Path, help='Default: configs/simulation/obstacle_avoidance.json, or the saved config when resuming')
        p.add_argument('--output', type=Path, help='New directory; existing output is never overwritten')
        p.add_argument('--device', help='Explicit policy device override, e.g. cuda:0 or cpu')
        p.add_argument('--no-render', action='store_true', help='Physics-only scripted diagnostics; no video or model observations')
        if name != 'check':
            p.add_argument('--controller', choices=CONTROLLERS, default='molmoact2')
        if name == 'suite':
            p.add_argument('--resume', action='store_true', help='Continue the suite in --output: reuse completed episodes, set failed ones aside and rerun them')
        if name == 'rollout':
            p.add_argument('--scene', choices=SCENES, default='tower')
            p.add_argument('--arm', default='plain', help='Arm id from the configuration')
            p.add_argument('--seed', type=int, default=1000)
            p.add_argument('--layout', help='Layout id from the configuration')
    p = sub.add_parser('report', help='Per-cell outcomes and paired contrasts for a finished suite (no model or simulator)')
    p.add_argument('suite', type=Path)
    p = sub.add_parser('prompts', help='Token counts per instruction; fails if a patch arm pairs prompts of different length')
    p.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.command == 'report':
        from vla_safety.obstacle.report import write_report
        for path in write_report(args.suite).values(): print(path)
        return
    if args.command == 'prompts':
        from vla_safety.obstacle.prompts import check_patch_prompts
        counts, problems = check_patch_prompts(load_config(args.config or DEFAULT_CONFIG), ROOT)
        print(json.dumps({'token_counts': counts, 'mismatched_arms': problems}, indent=2))
        if problems: raise SystemExit(1)
        return
    resume = getattr(args, 'resume', False)
    if resume and not args.output: parser.error('--resume needs --output pointing at the existing suite')
    if resume and not (args.output / 'config.json').exists(): parser.error(f'No suite to resume in {args.output}')
    config = load_config(args.config or (args.output / 'config.json' if resume else DEFAULT_CONFIG))
    if args.device: config['policy']['device'] = args.device
    controller = getattr(args, 'controller', None)
    if args.no_render and controller == 'molmoact2': parser.error('MolmoAct2 needs rendered observations; --no-render is for scripted diagnostics')
    output = (args.output or RunArchive.new_path(ROOT / config['output_root'], f'obstacle-{args.command}')).resolve()
    print(f'Artifacts: {output}', flush=True)
    if args.command == 'check':
        result = run_checks(config, ROOT, output, render=not args.no_render)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, args.command, result, describe(args.command, result))
        if not result['ok']: raise SystemExit(1)
    elif args.command == 'suite':
        result = run_suite(config, ROOT, output, controller=controller, render=not args.no_render, resume=resume)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, args.command, result, describe(args.command, result))
        if result['status'] != 'completed': raise SystemExit(1)
    else:
        if args.arm not in config['arms']: parser.error(f'Unknown arm; choose from {sorted(config["arms"])}')
        layout = next((x for x in config['suite']['layouts'] if x['id'] == args.layout), None) if args.layout else config['suite']['layouts'][0]
        if layout is None: parser.error('Unknown layout id')
        result, _ = run_episode(config, ROOT, output, scene=args.scene, arm=args.arm, seed=args.seed, layout=layout,
                                controller=controller, render=not args.no_render)
        print(json.dumps(result, indent=2))
        record_progress(config, ROOT, output, args.command, result, describe(args.command, result))
        if result['termination'] in ('error', 'cancelled'): raise SystemExit(1)


if __name__ == '__main__':
    main()
