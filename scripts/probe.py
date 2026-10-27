#!/usr/bin/env python3
"""Fixed-image fragile-release probes: prepare inputs, run, and inspect archives without a model."""
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

from vla_safety.archive import atomic_json, verify_archive
from vla_safety.concepts.manifest import write_manifest
from vla_safety.concepts.replay import export_replay
from vla_safety.release import release_settings, release_bank, expand_release_scenes


def config_template(manifest, bank, model, settings):
    return {'schema_version': 1, 'concept': {'id': 'fragility', 'definition': 'Susceptibility to physical damage'},
            'also_analyse': ['release_height', 'release_rule'], 'task_profile': settings,
            'model': model, 'device': 'cuda:0', 'dtype': 'float32',
            'manifest': str(manifest.resolve()), 'prompt_bank': str(bank.resolve()),
            'seeds': [1000, 1001, 1002], 'num_steps': 10, 'components': ['*'],
            'capture_raw': False, 'pooling': 'all_emitted_positions_mean', 'max_capture_mib': 256,
            'top_k': 32, 'bootstrap': 500, 'output_root': str(ROOT / 'artifacts/concepts'),
            'interventions': {'enabled': True, 'group_size': 8, 'max_examples': 4, 'patch': True}}


def prepare(args):
    from PIL import Image
    settings = release_settings(args.max_release_height_m, args.low_height_m, args.high_height_m)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    bank = release_bank(settings)
    if args.synthetic:
        image_paths = []
        for i, color in enumerate(('#54709b', '#80a188')):
            path = destination / f'camera{i}.png'
            Image.new('RGB', (96, 96), color).save(path)
            image_paths.append(str(path))
        scenes = [{'id': f'fixture-{i}', 'group_id': f'fixture-{i}',
                   'images': image_paths, 'state': [0.]*6, 'embodiment': 'so101',
                   'instruction': 'Move the synthetic target.',
                   'split': ('discovery', 'selection', 'evaluation')[i//2]} for i in range(6)]
        # Repeated toy wording is intentional and cannot test language generalization.
        bank = [{**bank[0], 'id': f'fixture-family-{i}',
                 'split': ('discovery', 'selection', 'evaluation')[i//2],
                 'variants': bank[0]['variants'][:1]} for i in range(6)]
        model = 'synthetic'
    elif args.scene_manifest:
        model = 'so101'
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
    else:
        model = 'so101'
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
    if args.small and not args.synthetic:
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
    config = config_template(manifest_path, bank_path, model, settings)
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
    p.add_argument('--small', action='store_true', help='Small smoke dataset: 16 prompts, one seed')
    p.add_argument('--max-release-height-m', type=float, default=.05, help='Illustrative threshold in metres')
    p.add_argument('--low-height-m', type=float, default=.02)
    p.add_argument('--high-height-m', type=float, default=.20)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--synthetic', action='store_true', help='Known toy components; no MolmoAct2 evidence')
    source.add_argument('--scene-manifest', type=Path, help='Annotated matched fragile/durable × high/low scenes')
    p = sub.add_parser('run', help='Run one experiment or a sequential suite')
    p.add_argument('--config', type=Path, nargs='+', required=True)
    p.add_argument('--device')
    p = sub.add_parser('components', help='Load checkpoint and enumerate supported capture sites')
    p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('replay', help='Generate/recover standalone offline HTML without model imports')
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, help='Default: replay-export.html beside the run')
    p = sub.add_parser('verify', help='Check archived file SHA256 and sizes')
    p.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args))
    elif args.command == 'run':
        from vla_safety.concepts.runner import run_experiment
        for path in args.config:
            config = json.loads(path.read_text())
            if args.device:
                config['device'] = args.device
            print(run_experiment(config, ROOT), flush=True)
    elif args.command == 'components':
        from vla_safety.concepts.runner import load_policy
        from vla_safety.capture import enumerate_components, inventory
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


if __name__ == '__main__':
    main()
