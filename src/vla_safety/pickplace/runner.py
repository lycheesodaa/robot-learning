"""Closed-loop v6 pick-and-place episodes with fragility in the instruction."""
import html
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from vla_safety.archive import atomic_json, digest, provenance
from vla_safety.simulation.runner import _jsonline, _seal, _set_aside
from .environment import task_prompt, PickPlaceEnvironment, ReleaseMonitor, instruction, validate_config
from . import policy as v6


def _report(folder, result):
    (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Pick-and-place rollout</title>'
        '<h1>Pick-and-place rollout</h1><video controls width="960" src="rollout.mp4"></video>'
        '<p>front_low (cam0) then wrist (cam1). Simulated behaviour; breakage is the impact-speed rule.</p><pre>' +
        html.escape(json.dumps(result, indent=2)) + '</pre>')


def run_episode(config, root, folder, *, fragile=True, rule='implicit', seed=1000, layout=None, bundle=None, render=True,
                base_prompt_only=False):
    """base_prompt_only sends v6's training prompt alone; `fragile` then only labels the breakage scoring."""
    validate_config(config)
    if type(seed) is not int or seed < 0 or type(fragile) is not bool or rule not in ('implicit', 'explicit'):
        raise ValueError('Invalid episode factors')
    root = Path(root); folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    layout = layout or config['suite']['layouts'][0]
    atomic_json(folder/'config.json', config)
    text = task_prompt(config['task']) if base_prompt_only else instruction(config, fragile, rule)
    condition = {'fragile': fragile, 'rule': None if base_prompt_only else rule, 'seed': seed, 'layout': layout, 'controller': 'molmoact2',
                 'instruction': text, 'prompt': 'base_only' if base_prompt_only else 'fragility+rule',
                 'evidence': 'scoring_label_only' if base_prompt_only else 'simulated_instruction_conditioning'}
    atomic_json(folder/'condition.json', condition)
    atomic_json(folder/'status.json', {'status': 'running'})
    env = writer = monitor = None; result = None; termination = 'error'; step = 0
    ctrl = config['control']
    try:
        env = PickPlaceEnvironment(config, layout)
        env.reset(text)
        atomic_json(folder/'initialization.json', env.initialization)
        metadata = provenance(root)
        metadata.update(simulator=config['simulator'], mujoco=env.mj.__version__, initialization=env.initialization)
        if bundle is None:
            bundle, model_metadata = v6.load_policy(config['policy'], root)
            metadata.update(model_metadata)
        else:
            metadata.update(model_id=config['policy']['repo_id'], revision=config['policy']['revision'])
        atomic_json(folder/'provenance.json', metadata)
        for source in [*(root/'src/vla_safety/pickplace').glob('*.py'), root/'scripts/pickplace.py']:
            target = folder/'source'/source.relative_to(root); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        monitor = ReleaseMonitor(config['task'], fragile, env.U.config.goal_thresh)
        cam0, cam1 = env.images()
        from PIL import Image
        Image.fromarray(cam0).save(folder/'initial_cam0.png'); Image.fromarray(cam1).save(folder/'initial_cam1.png')
        if render:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(folder/'rollout.mp4'), fps=ctrl['video_fps'], codec='libx264', macro_block_size=2)
            writer.append_data(np.concatenate([cam0, cam1], axis=1))
        v6.begin_episode(bundle, seed)
        opening_rad = np.deg2rad(config['task']['opening_delta_deg']); env_step = 0
        with (folder/'telemetry.jsonl').open('w') as telemetry, (folder/'actions.jsonl').open('w') as actions:
            for step in range(ctrl['max_policy_steps']):
                if (folder/'CANCEL').exists() or (folder.parent/'CANCEL').exists():
                    termination = 'cancelled'; break
                cam0, cam1 = env.images(); state = env.policy_state(); q_before = env.obs['state'].astype(float)
                started = time.perf_counter()
                action = v6.select_action(bundle, cam0, cam1, state, text)
                elapsed = time.perf_counter() - started
                proposed, target = env.to_sim(action)
                opening = bool(target[5] - q_before[5] > opening_rad)
                if opening: monitor.opening(step)
                _jsonline(actions, {'policy_step': step, 'time_s': float(env.U.data.time), 'state': state.tolist(),
                                    'action': action.tolist(), 'proposed_radians': proposed.tolist(), 'executed_radians': target.tolist(),
                                    'clipped': bool(np.any(np.abs(proposed - target) > 1e-6)), 'opening_command': opening,
                                    'inference_wall_s': elapsed})
                for _ in range(ctrl['env_steps_per_action']):
                    info = env.step(target); env_step += 1
                    row = env.telemetry(info); row.update(policy_step=step)
                    monitor.update(row, step)
                    _jsonline(telemetry, row)
                    if writer and env_step % ctrl['video_every_env_steps'] == 0:
                        writer.append_data(np.concatenate(env.images(), axis=1))
                    if monitor.success: break
                if monitor.success:
                    termination = 'success'; break
            else: termination = 'timeout'
        result = monitor.result()
        result.update(termination=termination, condition=condition, policy_steps=step + 1,
                      simulation_duration_s=float(env.U.data.time), initialization=env.initialization)
    except BaseException as exc:
        result = monitor.result() if monitor is not None else {}
        termination = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'error'
        result.update(termination=termination, error=str(exc), condition=condition)
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        for resource in (writer, env):
            if resource is not None:
                try: resource.close()
                except Exception as exc: result.setdefault('cleanup_errors', []).append(str(exc)); termination = result['termination'] = 'error'
        atomic_json(folder/'result.json', result)
        atomic_json(folder/'status.json', {'status': termination})
        _report(folder, result); _seal(folder)
    return result


def _completed(path, initialization):
    """A sealed success/timeout episode from the same start state, else None."""
    if not (path/'files.json').exists() or json.loads((path/'status.json').read_text())['status'] not in ('success', 'timeout'):
        return None
    saved = json.loads((path/'initialization.json').read_text())
    if saved['state_sha256'] != initialization['state_sha256']:
        raise ValueError(f'{path.name}: saved start state differs from the rebuilt one; resume would break pairing')
    return json.loads((path/'result.json').read_text())


def summarize(results):
    groups = {}
    for r in results:
        c = r['condition']; groups.setdefault(f"f{int(c['fragile'])}-{c['rule']}", []).append(r)
    summary = {}
    for key, group in groups.items():
        valid = [r for r in group if r['termination'] in ('success', 'timeout')]
        heights = [r['first_release_height_m'] for r in valid if r.get('first_release_height_m') is not None]
        n = len(valid)
        summary[key] = {'episodes': len(group), 'valid_episodes': n, 'failed_or_cancelled': len(group) - n,
                        'grasp_rate': sum(r['grasped'] for r in valid)/n if n else None,
                        'release_rate': sum(r['release_occurred'] for r in valid)/n if n else None,
                        'placement_success_rate': sum(r['task_success'] for r in valid)/n if n else None,
                        'first_release_heights_m': heights,
                        'mean_first_release_height_m': float(np.mean(heights)) if heights else None,
                        'commanded_first_release_count': sum(r.get('first_release_kind') == 'commanded' for r in valid),
                        'slip_first_release_count': sum(r.get('first_release_kind') == 'slip' for r in valid),
                        'rule_violation_count': sum(r['rule_violation'] for r in valid),
                        'broken_count': sum(r['broken'] for r in valid)}
    return summary


def run_suite(config, root, folder, resume=False):
    """With resume, completed episodes are reused and failed/partial ones are rerun."""
    validate_config(config)
    root = Path(root); folder = Path(folder)
    if resume:
        if json.loads((folder/'config.json').read_text()) != config:
            raise ValueError('Configuration differs from the suite being resumed')
        (folder/'CANCEL').unlink(missing_ok=True)
        if (folder/'error.txt').exists(): _set_aside(folder, folder/'error.txt')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder/'config.json', config)
    atomic_json(folder/'status.json', {'status': 'running'})
    results = []; failure = None; initializations = []
    try:
        bundle, _ = v6.load_policy(config['policy'], root)
        for layout in config['suite']['layouts']:
            env = PickPlaceEnvironment(config, layout)
            try: env.reset(); initializations.append({'layout': layout['id'], **env.initialization})
            finally: env.close()
            atomic_json(folder/'initializations.json', initializations)
            for seed, fragile, rule in itertools.product(config['suite']['seeds'], config['suite']['fragilities'], config['suite']['rules']):
                if (folder/'CANCEL').exists(): failure = 'cancelled'; break
                stem = f"{layout['id']}-f{int(fragile)}-{rule}-s{seed}"
                result = None
                if resume and (folder/stem).exists():
                    result = _completed(folder/stem, initializations[-1])
                    if result is None: _set_aside(folder, folder/stem)
                print(stem, '(completed earlier)' if result else '', flush=True)
                if result is None:
                    result = run_episode(config, root, folder/stem, fragile=fragile, rule=rule, seed=seed, layout=layout, bundle=bundle)
                    if result.get('initialization', {}).get('state_sha256') not in (None, initializations[-1]['state_sha256']):
                        raise RuntimeError(f'{stem}: start state differs from the layout snapshot')
                results.append({'episode': stem, **result})
                atomic_json(folder/'results.json', results)
                if result['termination'] in ('error', 'cancelled'): failure = result['termination']; break
            if failure: break
    except BaseException:
        failure = 'error'; (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        atomic_json(folder/'summary.json', summarize(results))
        atomic_json(folder/'status.json', {'status': failure or 'completed', 'episodes': len(results)})
        _seal(folder)
    return {'status': failure or 'completed', 'episodes': len(results)}
