"""Fixed-image fragile-release workflow. Policy imports and loading occur only when running."""
import csv
import json
from pathlib import Path
import time
import traceback

import numpy as np

from vla_safety.archive import RunArchive, digest, provenance
from vla_safety.capture import capture_components, enumerate_components, inventory
from vla_safety.release import validate_release_settings, validate_release_rows
from .analysis import analyse
from .manifest import load_manifest


def validate_config(config):
    if config.get('schema_version') != 1:
        raise ValueError('Expected configuration schema_version=1')
    if not isinstance(config.get('concept'), dict) or not config['concept'].get('id'):
        raise ValueError('A concept id is required')
    ids = [config['concept']['id'], *config.get('also_analyse', [])]
    for concept_id in ids:
        if not isinstance(concept_id, str) or not concept_id or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in concept_id):
            raise ValueError('Concept ids must contain only letters, digits, _ or -')
    if len(set(ids)) != len(ids):
        raise ValueError('Concept ids must be unique')
    if not config.get('seeds') or any(type(s) is not int or s < 0 for s in config['seeds']):
        raise ValueError('Provide nonnegative integer seeds')
    if len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Seeds must be unique')
    if config.get('model') not in ('so101', 'synthetic'):
        raise ValueError('Unknown model')
    if config.get('num_steps', 10) < 1 or config.get('max_capture_mib', 256) <= 0:
        raise ValueError('Flow steps and capture limit must be positive')
    if config.get('interventions', {}).get('max_examples', 4) < 0:
        raise ValueError('max_examples must be nonnegative')
    if config.get('interventions', {}).get('group_size', 8) < 1 or config.get('top_k', 32) < 1:
        raise ValueError('Group size and top_k must be positive')
    if config.get('pooling', 'all_emitted_positions_mean') != 'all_emitted_positions_mean':
        raise ValueError('Only all_emitted_positions_mean pooling is currently supported')
    if not config.get('task_profile'):
        raise ValueError('A fragile_release task_profile is required')
    validate_release_settings(config['task_profile'])


def load_policy(config, root):
    if config['model'] == 'synthetic':
        from .fixture import synthetic_release_policy
        return synthetic_release_policy(config['task_profile']), {'model_id': 'synthetic-release-fixture', 'synthetic': True}
    from vla_safety.policy import load_policy as load_model
    return load_model(config, root)


def _save_sources(archive, root):
    """Keep the exact local implementation, including uncommitted changes."""
    candidates = list((root / 'src/vla_safety').rglob('*.py'))
    candidates += [root / 'scripts/probe.py', root / 'scripts/sim.py', root / 'pyproject.toml']
    for source in candidates:
        if source.is_file():
            destination = archive.folder / 'source' / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())


def run_experiment(config, root, *, output=None):
    """All generated files live inside the unique run folder, even on failure."""
    from PIL import Image
    from vla_safety.types import Observation
    from vla_safety.policy import encode_action
    from vla_safety.hooks import HookSpec, intervention_hooks
    root = Path(root).resolve()
    config = dict(config)
    validate_config(config)
    concept = config['concept']['id']
    output_root = Path(config.get('output_root', root / 'artifacts/concepts'))
    if not output_root.is_absolute():
        output_root = root / output_root
    output = Path(output) if output else RunArchive.new_path(output_root, concept)
    archive = RunArchive(output, config, provenance(root))
    print(f'Run archive: {archive.folder}', flush=True)
    _save_sources(archive, root)

    try:
        manifest = Path(config['manifest'])
        if not manifest.is_absolute():
            manifest = root / manifest
        rows = load_manifest(manifest)
        validate_release_rows(rows, config['task_profile'])
        if any(row.get('labels', {}).get(concept) not in (0, 1) for row in rows):
            raise ValueError(f'Missing binary labels for {concept}')
        frozen = []
        for row in rows:
            saved = dict(row)
            saved['source_images'] = [{'path': path, 'sha256': digest(path)} for path in row['images']]
            saved['images'] = []
            for path in row['images']:
                with Image.open(path) as image:
                    saved['images'].append(archive.image(image))
            frozen.append(saved)
        input_ref = archive.json('inputs/manifest.json', frozen)
        if config.get('prompt_bank'):
            bank_path = root / config['prompt_bank']
            archive.json('inputs/prompt_bank.json', json.loads(bank_path.read_text()))
        archive.event('inputs_frozen', detail=input_ref, examples=len(frozen),
                      source_manifest_sha256=digest(manifest))
        archive.event('model_loading', model=config['model'])
        policy, model_metadata = load_policy(config, root)
        archive.metadata.update(model_metadata)
        archive.json('provenance.json', archive.metadata)
        components = (policy.components if config['model'] == 'synthetic' else
                      enumerate_components(policy.model, config.get('components', ['*'])))
        component_ref = archive.json('components.json', inventory(components))
        archive.event('model_ready', detail=component_ref)
        max_bytes = int(config.get('max_capture_mib', 256) * 1024**2)
        num_steps = config.get('num_steps', 10)
        raw = config.get('capture_raw', False)
        baseline = {}
        baseline_shapes = {}

        def predict(row, seed, intervention=None, specs=(), raw_override=None, selected_components=None):
            images = []
            for path in row['images']:
                with Image.open(archive.folder / path) as im:
                    images.append(im.convert('RGB'))
            obs = Observation(tuple(images), row['instruction'], row['state'], row['embodiment'],
                              row.get('timestamp', 0.))
            capture_set = components if selected_components is None else selected_components
            started = time.perf_counter()
            # Capture is registered first: intervention events expose pre-intervention values.
            with capture_components(policy.model, capture_set, raw=raw if raw_override is None else raw_override,
                                    max_bytes=max_bytes) as captures:
                with intervention_hooks(policy.model, list(specs)):
                    action = policy.predict(obs, seed=seed, num_steps=num_steps)
            elapsed = time.perf_counter() - started
            if not any(captures.values()):
                raise RuntimeError('Selected components did not execute')
            archive.prediction(row, seed, encode_action(action), captures, elapsed=elapsed, intervention=intervention)
            return action.values, captures

        for row in frozen:
            for seed in config['seeds']:
                values, captures = predict(row, seed)
                baseline[(row['id'], seed)] = values
                baseline_shapes[(row['id'], seed)] = {
                    name: [r['shape'] for r in records] for name, records in captures.items()}
        archive.event('capture_completed', predictions=len(baseline))
        jobs = [(name, None) for name in [concept, *config.get('also_analyse', [])]]
        jobs.append(('fragility_height_interaction', ('fragility', 'release_height')))
        for analysis_concept, interaction in jobs:
            archive.event('analysis_started', concept=analysis_concept)
            result, arrays = analyse(archive.folder, analysis_concept,
                                     top_k=config.get('top_k', 32), bootstrap=config.get('bootstrap', 500),
                                     interaction=interaction,
                                     progress=lambda **data: archive.event('analysis_progress', concept=analysis_concept, **data))
            if arrays:
                archive.arrays(f'analysis/{analysis_concept}/probes.npz', arrays)
            ref = archive.json(f'analysis/{analysis_concept}/results.json', result)
            csv_path = archive.folder / 'analysis' / analysis_concept / 'rankings.csv'
            with csv_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['component', 'channel', 'effect', 'consistency', 'pairs', 'ci95'])
                writer.writeheader()
                writer.writerows(result['rankings'])
            archive.event('analysis', concept=analysis_concept, detail=ref)
            if interaction:
                main_result = result

        settings = config.get('interventions', {})
        if settings.get('enabled', True) and main_result['rankings']:
            # Discovery-only targets. Confirmatory runs use frozen evaluation inputs.
            candidates = [r for r in main_result['rankings'] if any(
                c.name == r['component'] and c.kind == 'mlp_neurons' for c in components)]
            target = candidates[0] if candidates else main_result['rankings'][0]
            component = next(c for c in components if c.name == target['component'])
            neurons = [r['channel'] for r in main_result['rankings'] if r['component'] == component.name]
            group = tuple(neurons[:min(settings.get('group_size', 8), len(neurons))])
            width = next(c['channels'] for c in main_result['components'] if c['component'] == component.name)
            rng = np.random.default_rng(config['seeds'][0])
            random_group = tuple(int(n) for n in rng.choice(width, size=len(group), replace=False))
            split = 'evaluation' if main_result['evidence'].startswith('held_out_') else main_result['ranking_split']
            evaluation = [r for r in frozen if r['split'] == split and
                          r['labels']['fragility'] == r['labels']['release_height'] == 1]
            donor_factor = 'release_height'
            interventions = []
            for row in evaluation[:settings.get('max_examples', 4)]:
                donor = next((r for r in frozen if r.get('pair_id') == row.get('pair_id')
                              and r['labels'][donor_factor] == 0 and all(r['labels'].get(k) == v
                              for k, v in row['labels'].items() if k != donor_factor)), None)
                for seed in config['seeds']:
                    base = baseline[(row['id'], seed)]
                    variants = [('zero', group, 0.), ('top_neuron', (neurons[0],), 1.),
                                ('top_group', group, 1.), ('random_group', random_group, 1.),
                                ('whole_component', (), 1.)]
                    for condition, chosen, strength in variants:
                        spec = HookSpec(component.module, 'ablate', strength, chosen, site=component.site)
                        descriptor = {'condition': condition, 'component': component.name, 'neurons': list(chosen),
                                      'strength': strength, 'capture_phase': 'before_intervention'}
                        values, _ = predict(row, seed, descriptor, [spec])
                        delta = values - base
                        record = {'example_id': row['id'], 'seed': seed, **descriptor,
                                  'mean_absolute_delta_by_dimension': np.abs(delta).mean(axis=0).tolist()}
                        if donor:
                            opposite = baseline[(donor['id'], seed)]
                            record['counterpart_distance_reduction_by_dimension'] = (
                                np.abs(base-opposite).mean(axis=0) - np.abs(values-opposite).mean(axis=0)).tolist()
                        interventions.append(record)
                    if donor and settings.get('patch', True):
                        expected = baseline_shapes[(row['id'], seed)][component.name]
                        donor_shapes = baseline_shapes[(donor['id'], seed)][component.name]
                        if expected != donor_shapes:
                            archive.event('intervention_skipped', condition='patch', component=component.name,
                                          example_id=row['id'], donor_id=donor['id'], seed=seed,
                                          reason='Patch donor/recipient shapes or invocation counts differ')
                            continue
                        # Save donor raw values even when regular capture is pooled.
                        _, donor_captures = predict(donor, seed, {'condition': 'patch_donor', 'component': component.name},
                                                       raw_override=True, selected_components=[component])
                        records = iter(donor_captures[component.name])
                        consumed = []
                        def patch_value(selected):
                            try:
                                value = next(records)['raw'][..., list(group)]
                            except StopIteration:
                                raise ValueError('Patch donor has fewer invocations') from None
                            if tuple(value.shape) != tuple(selected.shape):
                                raise ValueError('Patch donor/recipient shapes differ')
                            consumed.append(1)
                            return value
                        spec = HookSpec(component.module, 'patch', 1., group, patch_value, site=component.site)
                        descriptor = {'condition': 'patch', 'component': component.name, 'donor_id': donor['id'],
                                      'neurons': list(group), 'capture_phase': 'before_intervention'}
                        try:
                            values, _ = predict(row, seed, descriptor, [spec])
                            if len(consumed) != len(donor_captures[component.name]):
                                raise ValueError('Patch donor has more invocations')
                            opposite = baseline[(donor['id'], seed)]
                            interventions.append({'example_id': row['id'], 'seed': seed, **descriptor,
                                                  'mean_absolute_delta_by_dimension': np.abs(values-base).mean(axis=0).tolist(),
                                                  'counterpart_distance_reduction_by_dimension': (
                                                      np.abs(base-opposite).mean(axis=0)-np.abs(values-opposite).mean(axis=0)).tolist()})
                        except ValueError as exc:
                            archive.event('intervention_skipped', **descriptor, reason=str(exc))
            ref = archive.json('interventions/results.json', {'target_selection': 'discovery_only' if split == 'evaluation' else 'exploratory',
                                                            'results': interventions})
            archive.event('interventions_completed', detail=ref)
        archive.finish()
    except (Exception, KeyboardInterrupt) as exc:
        archive.json('error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        archive.finish('cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed', str(exc))
        raise
    return archive.folder

