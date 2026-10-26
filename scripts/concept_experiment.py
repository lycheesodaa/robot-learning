#!/usr/bin/env python3
"""Configure/run concept studies and export or inspect them without a model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))

from vla_safety.concepts.archive import RunArchive, atomic_json, read_events, verify_archive
from vla_safety.concepts.prompts import default_bank, expand_scenes, session_split, write_manifest
from vla_safety.concepts.replay import bundle_archive, export_replay
from vla_safety.concepts.fragile_release import release_settings, release_bank, expand_release_scenes, evaluate_release_trace


def config_template(manifest, bank, model):
    return {'schema_version': 1, 'concept': {'id': 'fragility', 'definition': 'Susceptibility to physical damage'},
            'also_analyse': ['gentle_handling'], 'model': model, 'device': 'cuda:0', 'dtype': 'float32',
            'manifest': str(manifest.resolve()), 'prompt_bank': str(bank.resolve()),
            'seeds': [1000, 1001, 1002], 'num_steps': 10, 'components': ['*'],
            'capture_raw': False, 'pooling': 'all_emitted_positions_mean', 'max_capture_mib': 256,
            'top_k': 32, 'bootstrap': 500, 'output_root': str(ROOT / 'artifacts/concepts'),
            'interventions': {'enabled': True, 'group_size': 8, 'max_examples': 4, 'patch': True}}


def prepare(args):
    from PIL import Image
    task = args.profile == 'fragile-release'
    settings = release_settings(args.max_release_height_m, args.low_height_m, args.high_height_m) if task else None
    if args.scene_manifest and not task:
        raise ValueError('--scene-manifest currently requires --profile fragile-release')
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    scenes = []
    if args.synthetic:
        image_paths = []
        for i, color in enumerate(('#54709b', '#80a188')):
            path = destination / f'camera{i}.png'
            Image.new('RGB', (96, 96), color).save(path)
            image_paths.append(str(path))
        for i in range(6):
            scenes.append({'id': f'fixture-{i}', 'group_id': f'fixture-{i}',
                           'images': image_paths, 'state': [0.]*6, 'embodiment': 'so101',
                           'instruction': 'Move the synthetic target.',
                           'split': ('discovery', 'selection', 'evaluation')[i//2]})
        bank = [{'id': f'family-{i}', 'split': ('discovery', 'selection', 'evaluation')[i//2],
                 'variants': [{'fragility': {'positive': 'fragile', 'negative': 'durable'},
                               'gentle_handling': {'positive': 'gently', 'negative': 'normally'}}]}
                for i in range(6)]
        # Synthetic wording repeats intentionally; this is a software fixture only.
        rows = [r for r in expand_scenes(scenes, bank) if r['split'] != 'diagnostic']
        model = 'synthetic'
    elif args.scene_manifest:
        model = args.model
        source = args.scene_manifest.resolve()
        scenes = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not scenes:
            raise ValueError('Scene manifest is empty')
        for i, scene in enumerate(scenes):
            copied = []
            for camera, relative in enumerate(scene['images']):
                path = source.parent / relative
                target = destination / 'images' / f'{i}-{camera}{path.suffix}'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied.append(str(target))
            scene['images'] = copied
        bank = release_bank(settings)
        rows = []
    elif args.recordings:
        model = 'so101'
        for session in args.recordings:
            session = session.resolve()
            with (session / 'inferences.jsonl').open() as f:
                for number, line in enumerate(f):
                    if number % args.stride:
                        continue
                    row = json.loads(line)
                    paths = [session / f"{row['index']:05d}_{camera}.jpg" for camera in ('cam0', 'cam1')]
                    if not all(path.is_file() for path in paths):
                        raise ValueError(f'Missing recorded cameras for {session.name}/{row["index"]}')
                    local_images = []
                    for path in paths:
                        target = destination / 'images' / session.name / path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                        local_images.append(str(target))
                    scenes.append({'id': f"{session.name}-{row['index']}", 'group_id': session.name,
                                   'images': local_images, 'instruction': row['task'], 'state': row['state'],
                                   'embodiment': 'so101', 'timestamp': row['client_time'],
                                   'split': session_split(session.name)})
        bank = default_bank()
        if args.small:
            bank = [{**b, 'variants': b['variants'][:1]} for b in bank[:2]]
        rows = expand_scenes(scenes, bank, exploratory=args.small)
    else:
        model = args.model
        config = json.loads((ROOT / 'configs/models.json').read_text())[model]
        snapshot = ROOT / 'artifacts/models' / model / config['revision']
        image_paths = []
        for name in config['sample_images']:
            path = destination / name
            shutil.copyfile(snapshot / 'assets' / name, path)
            image_paths.append(str(path))
        scenes = [{'id': 'official-sample', 'group_id': 'official-sample', 'images': image_paths,
                   'state': config['sample_state'], 'embodiment': config['embodiment'],
                   'instruction': config['sample_instruction'].replace('drop it', 'place it'),
                   'split': 'exploratory'}]
        bank = default_bank()
        if args.small:
            bank = [{**b, 'variants': b['variants'][:1]} for b in bank[:2]]
        rows = expand_scenes(scenes, bank, exploratory=True)
    if task:
        bank = release_bank(settings)
        if args.synthetic:
            # Repeated toy wording is intentional and cannot test language generalization.
            bank = [{**bank[0], 'id': f'fixture-family-{i}',
                     'split': ('discovery', 'selection', 'evaluation')[i//2],
                     'variants': bank[0]['variants'][:1]} for i in range(6)]
        elif args.small:
            bank = [{**b, 'variants': b['variants'][:1]} for b in bank[:2]]
        if not args.scene_manifest:
            for scene in scenes:
                scene['original_instruction'] = scene['instruction']
                # A neutral shared task, without an unconditional "drop" instruction.
                scene['instruction'] = 'Place the target object on the receiving surface.'
        rows = expand_release_scenes(scenes, bank, settings, grounded=bool(args.scene_manifest),
                                     exploratory=args.small and not args.synthetic)
        if args.synthetic:
            rows = [r for r in rows if r['split'] != 'diagnostic']
    # Manifest image paths are relative so the prepared dataset can also be moved.
    for row in rows:
        row['images'] = [os.path.relpath(p, destination) for p in row['images']]
    manifest_path, bank_path = destination / 'examples.jsonl', destination / 'prompt_bank.json'
    write_manifest(manifest_path, rows)
    atomic_json(bank_path, bank)
    config = config_template(manifest_path, bank_path, model)
    if task:
        config['task_profile'] = settings
        config['also_analyse'] = ['release_height', 'release_rule']
    if args.small or args.synthetic:
        config['seeds'] = [1000]
        config['bootstrap'] = 50
        config['interventions']['max_examples'] = 1
    if args.synthetic:
        config.update(device='cpu', num_steps=3, top_k=8)
        config['interventions']['group_size'] = 2
    atomic_json(destination / 'config.json', config)
    return destination / 'config.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Create editable prompt bank, manifest and config')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=['so101', 'libero'], default='so101')
    p.add_argument('--small', action='store_true', help='Small smoke dataset: 8 concept or 16 release prompts, one seed')
    p.add_argument('--profile', choices=['concept', 'fragile-release'], default='concept')
    p.add_argument('--max-release-height-m', type=float, default=.05, help='Illustrative threshold in metres')
    p.add_argument('--low-height-m', type=float, default=.02)
    p.add_argument('--high-height-m', type=float, default=.20)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--synthetic', action='store_true', help='Known toy components; no MolmoAct2 evidence')
    source.add_argument('--recordings', nargs='+', type=Path)
    source.add_argument('--scene-manifest', type=Path, help='Annotated matched fragile/durable × high/low scenes')
    p.add_argument('--stride', type=int, default=30, help='Import one out of N recorded frames')
    p = sub.add_parser('run', help='Run one experiment or a sequential suite')
    p.add_argument('--config', type=Path, nargs='+', required=True)
    p.add_argument('--device')
    p.add_argument('--concept', help='Existing label key in the manifest')
    p = sub.add_parser('components', help='Load checkpoint and enumerate supported capture sites')
    p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('replay', help='Generate/recover standalone offline HTML without model imports')
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, help='Default: replay-export.html beside the run')
    p = sub.add_parser('verify', help='Check archived file SHA256 and sizes')
    p.add_argument('run', type=Path)
    p = sub.add_parser('bundle', help='ZIP the entire verified run for transfer')
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('inspect', help='Inspect a committed event without torch')
    p.add_argument('run', type=Path)
    p.add_argument('--event', type=int, default=-1)
    p = sub.add_parser('cancel', help='Request cancellation between predictions')
    p.add_argument('run', type=Path)
    p = sub.add_parser('score-release', help='Score an external simulator/measured release trace without a model')
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--max-release-height-m', type=float, default=.05)
    p.add_argument('--output', type=Path, required=True)
    for name, help_text in [('sim-check', 'Validate MuJoCo grasp, controls, cameras and scripted releases'),
                            ('rollout', 'Run one closed-loop simulated release'),
                            ('rollout-suite', 'Run matched simulated release experiments')]:
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
    args = parser.parse_args()
    if args.command in ('sim-check', 'rollout', 'rollout-suite'):
        from vla_safety.simulation.environment import load_config
        from vla_safety.simulation.runner import run_checks, run_episode, run_suite, target_specs, record_progress
        from vla_safety.concepts.archive import RunArchive
        config = load_config(args.config)
        if args.device:
            config['policy']['device'] = args.device
        if args.no_render and args.command != 'sim-check' and args.controller == 'molmoact2':
            parser.error('MolmoAct2 needs rendered observations; --no-render is for scripted diagnostics')
        output = args.output or RunArchive.new_path(ROOT / config['output_root'], args.command)
        output = output.resolve()
        print(f'Artifacts: {output}', flush=True)
        if args.command == 'sim-check':
            result = run_checks(config, ROOT, output, render=not args.no_render)
            print(json.dumps(result, indent=2))
            record_progress(config, ROOT, output, args.command, result)
            if not result['ok']: raise SystemExit(1)
        elif args.command == 'rollout-suite':
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
                from vla_safety.concepts.runner import load_policy
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
        return
    if args.command == 'prepare':
        if args.stride < 1:
            parser.error('--stride must be positive')
        print(prepare(args))
    elif args.command == 'run':
        from vla_safety.concepts.runner import run_experiment
        for path in args.config:
            config = json.loads(path.read_text())
            if args.device:
                config['device'] = args.device
            if args.concept:
                config['concept'] = {'id': args.concept}
                config['also_analyse'] = []
            print(run_experiment(config, ROOT), flush=True)
    elif args.command == 'components':
        from vla_safety.concepts.runner import load_policy
        from vla_safety.concepts.capture import enumerate_components, inventory
        config = json.loads(args.config.read_text())
        policy, _ = load_policy(config, ROOT)
        components = policy.components if config['model'] == 'synthetic' else enumerate_components(policy.model)
        print(json.dumps(inventory(components), indent=2))
    elif args.command == 'replay':
        output = args.output or args.run.parent / (args.run.name + '-replay-export.html')
        print(export_replay(args.run, output))
    elif args.command == 'verify':
        result = verify_archive(args.run)
        print(json.dumps(result, indent=2))
        if not result['ok']:
            raise SystemExit(1)
    elif args.command == 'bundle':
        print(bundle_archive(args.run, args.output))
    elif args.command == 'inspect':
        print(json.dumps(read_events(args.run)[args.event], indent=2))
    elif args.command == 'cancel':
        status = json.loads((args.run / 'status.json').read_text())
        if status['status'] != 'running':
            parser.error('Run is already finished')
        (args.run / 'CANCEL').touch()
        print('Cancellation requested.')
    elif args.command == 'score-release':
        if args.output.exists():
            parser.error('Output exists; choose a new result path')
        result = evaluate_release_trace(json.loads(args.trace.read_text()), args.max_release_height_m)
        atomic_json(args.output, result)
        print(args.output)


if __name__ == '__main__':
    main()
