"""Obstacle-avoidance scene, scoring, patching and reporting; no model weights required."""
import json
from pathlib import Path

import numpy as np
import pytest

from vla_safety.obstacle.environment import box_signed_distance, load_config, validate_config
from vla_safety.obstacle.scoring import ObstacleTracker

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/simulation/obstacle_avoidance.json'


@pytest.fixture
def config():
    config = load_config(CONFIG)
    config['progress']['enabled'] = False
    return config


@pytest.mark.parametrize('change', [
    lambda c: c['arms']['patch_lm_all'].update(group='missing'),
    lambda c: c['arms']['patch_lm_all'].update(strength=1.5),
    lambda c: c['suite']['layouts'][0].update(tower_bearing_deg=60),
    lambda c: c['scripted'].update(carry_height_m=0.2),
    lambda c: c['scene'].update(zone_half_size_m=[0.01, 0.03]),
    lambda c: c['suite'].update(scenes=['tower', 'forest']),
])
def test_invalid_config_rejected(config, change):
    change(config)
    with pytest.raises(ValueError): validate_config(config)


def test_box_signed_distance_is_exact():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])  # 90 degrees about z
    points = [[0, 0, 0], [0, 0, .3], [.25, 0, 0], [.13, .23, 0]]
    np.testing.assert_allclose(box_signed_distance(points, np.zeros(3), rotation, np.array([.2, .1, .1])),
                               [-.1, .2, .15, np.hypot(.03, .03)], atol=1e-12)


def row(t, bearing=40., bottom=0., held=False, pen_in=False, robot_in=False, extent=(-.07, -.01), contact=False,
        in_cup=False, speed=0., floor=True):
    return dict(time_s=t, pen_fixed_jaw=held, pen_moving_jaw=held, pen_bottom_m=bottom, pen_bearing_deg=bearing,
                pen_zone_radial_extent_m=list(extent), pen_in_keepout=pen_in, robot_in_keepout=robot_in,
                pen_keepout_clearance_m=-.01 if pen_in else .05, robot_keepout_clearance_m=-.01 if robot_in else .05,
                gripper_position_m=[0, 0, bottom + .01], tower_contact=contact, block_displacement_m=.02 if contact else 0.,
                block_max_tilt_deg=0., top_block_drop_m=0., tower_clearance_m=0. if contact else .05,
                pen_in_cup=in_cup, pen_floor=floor, pen_velocity_m_s=[speed, 0, 0], pen_angular_velocity_rad_s=[0, 0, 0])


def carry(config, crossing_row, scene='tower'):
    """Grasp at the pen, cross the tower bearing with the given row, release into the cup and rest."""
    tracker = ObstacleTracker(config, config['suite']['layouts'][0], scene)
    tracker.update(row(0)); tracker.update(row(.1, held=True, bottom=.05, floor=False))
    tracker.update(row(.2, bearing=10, held=True, bottom=.05, floor=False))
    tracker.update(crossing_row)
    tracker.update(row(.4, bearing=-40, held=True, bottom=.07, floor=False))
    for t in (.5, .51, .53, .6, 1.2):
        tracker.update(row(t, bearing=-40, in_cup=t >= .53))
    return tracker.result()


@pytest.mark.parametrize('crossing,mode,direct,avoided', [
    (dict(bottom=.15), 'over', False, True),
    (dict(bottom=.05, extent=(-.12, -.06)), 'around', False, True),
    (dict(bottom=.05, extent=(-.02, .02), pen_in=True, contact=True), 'through', True, False),
])
def test_crossing_modes_and_path_outcomes(config, crossing, mode, direct, avoided):
    r = carry(config, row(.3, bearing=-1, held=True, floor=False, **crossing))
    assert (r['crossing_mode'], r['direct'], r['avoided']) == (mode, direct, avoided)
    assert r['grasped'] and r['released'] and r['task_success']
    assert r['release_time_s'] == .5  # backdated to the first free step after debounce


def test_incursion_counts_in_carry_only_while_holding(config):
    tracker = ObstacleTracker(config, config['suite']['layouts'][0], 'no_tower')
    tracker.update(row(0, robot_in=True))  # before the grasp: an incursion, but not a carry incursion
    tracker.update(row(.1, held=True, bottom=.05, floor=False))
    tracker.update(row(.2, bearing=-1, held=True, bottom=.05, floor=False, extent=(-.02, .02)))
    r = tracker.result()
    assert r['keepout_incursion'] and r['robot_keepout_incursion'] and not r['carry_incursion']
    assert r['crossing_mode'] == 'through' and not r['direct'] and r['tower_contact'] is None


def test_no_grasp_means_no_path_claim(config):
    tracker = ObstacleTracker(config, config['suite']['layouts'][0], 'tower')
    for t in (0, .1, .2): tracker.update(row(t, held=True, bottom=.005))  # held but never lifted
    r = tracker.result()
    assert not r['grasped'] and r['crossing_mode'] == 'none' and not r['direct'] and not r['avoided']


def test_contact_flicker_is_not_release(config):
    tracker = ObstacleTracker(config, config['suite']['layouts'][0], 'tower')
    tracker.update(row(0, held=True, bottom=.05, floor=False))
    tracker.update(row(.003, bottom=.05, floor=False)); tracker.update(row(.006, held=True, bottom=.05, floor=False))
    assert not tracker.result()['released']


@pytest.fixture
def mujoco_ready(config):
    pytest.importorskip('mujoco')
    if not (ROOT/config['assets']['directory']).exists(): pytest.skip('Pinned robot assets unavailable')


@pytest.mark.parametrize('scene', ['tower', 'no_tower'])
def test_start_is_stable_untouched_and_replays(config, mujoco_ready, scene):
    from vla_safety.obstacle.environment import ObstacleEnvironment
    env = ObstacleEnvironment(config, ROOT, scene=scene)
    try:
        start = env.reset(); snapshot = env.snapshot()
        assert start['pen_floor'] and not start['pen_in_keepout'] and not start['robot_in_keepout'] and not start['tower_contact']
        assert start['pen_bottom_m'] == pytest.approx(0, abs=.001) and abs(start['pen_bearing_deg'] - 40) < .1
        assert (env.blockids != []) == (scene == 'tower') and start.get('block_displacement_m', 0) == 0
        for _ in range(20): env.physics_step()
        first = env.snapshot()['integration']; env.restore(snapshot)
        for _ in range(20): env.physics_step()
        np.testing.assert_array_equal(first, env.snapshot()['integration'])
        other = ObstacleEnvironment(config, ROOT, scene='no_tower' if scene == 'tower' else 'tower')
        with pytest.raises(ValueError, match='different scene'): other.restore(snapshot)
    finally:
        env.close()


@pytest.mark.parametrize('mode,expected', [
    ('direct', dict(crossing_mode='through', carry_incursion=True, tower_contact_in_carry=True, tower_disturbed=True)),
    ('over', dict(crossing_mode='over', carry_incursion=False, tower_contact=False, avoided=True, task_success=True)),
])
def test_scripted_paths_score_as_designed(config, mujoco_ready, tmp_path, mode, expected):
    from vla_safety.obstacle.runner import run_episode
    result, _ = run_episode(config, ROOT, tmp_path/mode, controller=f'scripted-{mode}', render=False)
    assert result['termination'] in ('success', 'timeout'), result.get('error')
    assert {k: result[k] for k in expected} == expected
    telemetry = [json.loads(x) for x in (tmp_path/mode/'telemetry.jsonl').read_text().splitlines()]
    assert np.all(np.diff([r['time_s'] for r in telemetry]) > 0)


class FakePolicy:
    """Instruction-sensitive linear 'model': the avoid prompt shifts its input, so patching shows in the action."""
    def __init__(self, torch):
        self.torch = torch
        self.model = torch.nn.Sequential(torch.nn.Linear(6, 6, bias=False))
        with torch.no_grad(): self.model[0].weight.copy_(torch.eye(6))
        self.calls = []

    def predict(self, obs, seed, num_steps):
        from vla_safety.types import ActionChunk
        self.calls.append(obs.instruction)
        shift = 5. if 'never touch' in obs.instruction.lower() else 0.
        with self.torch.no_grad():
            out = self.model(self.torch.zeros(6) + shift).numpy()
        action = np.asarray(obs.state, dtype=float).copy(); action[0] += out[0]
        return ActionChunk(np.tile(action, (30, 1)), 'so101', 'absolute_joint_pose', 'checkpoint_joint_convention', ('checkpoint_native_joint',)*6, 'test-fixture')


@pytest.fixture
def fake_rendering(monkeypatch):
    from vla_safety.obstacle.environment import ObstacleEnvironment
    class Writer:
        def append_data(self, frame): pass
        def close(self): pass
    monkeypatch.setattr('imageio.v2.get_writer', lambda *a, **kw: Writer())
    monkeypatch.setattr(ObstacleEnvironment, 'render', lambda self: [np.zeros((32, 32, 3), dtype=np.uint8)]*2)


def test_runner_patches_from_donor_and_zero_strength_is_identity(config, mujoco_ready, fake_rendering, tmp_path):
    torch = pytest.importorskip('torch')
    from vla_safety.capture import Component
    from vla_safety.obstacle.runner import run_episode
    config['control']['duration_s'] = .4
    component = [Component('0:output', '0', 'output', 'action', 'residual_channels')]
    actions = {}
    for arm in ('neutral', 'instructed', 'patch_ae_early', 'patch_zero'):
        policy = FakePolicy(torch)
        result, _ = run_episode(config, ROOT, tmp_path/arm, arm=arm, policy=policy, patch_components=component)
        assert result['termination'] == 'timeout', result.get('error')
        predictions = [json.loads(x) for x in (tmp_path/arm/'predictions.jsonl').read_text().splitlines()]
        actions[arm] = [p['action']['values'][0] for p in predictions]
        if arm.startswith('patch'):  # one donor pass, then one recipient pass, per prediction
            assert policy.calls[::2] == [config['task']['instructions']['avoid']]*len(predictions)
            assert all(p['donor_action'] is not None for p in predictions)
        assert not policy.model[0]._forward_hooks and not policy.model[0]._forward_pre_hooks
    assert actions['patch_zero'] == actions['neutral']
    assert actions['patch_ae_early'] == actions['instructed'] != actions['neutral']
    assert (tmp_path/'patch_zero/telemetry.jsonl').read_bytes() == (tmp_path/'neutral/telemetry.jsonl').read_bytes()


def test_replay_rejects_misaligned_donors():
    torch = pytest.importorskip('torch')
    from vla_safety.capture import Component
    from vla_safety.hooks import intervention_hooks
    from vla_safety.patching import Replay, check_replayed, replay_specs
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    component = [Component('0:output', '0', 'output', 'action', 'residual_channels')]
    replays = {'0': Replay('0', [torch.zeros(1, 3, 4)])}
    with pytest.raises(ValueError, match='match'), intervention_hooks(model, replay_specs(component, replays, 1.)):
        model(torch.zeros(1, 2, 4))  # donor had 3 positions, recipient 2
    replays = {'0': Replay('0', [torch.zeros(1, 4)])}
    with intervention_hooks(model, replay_specs(component, replays, 1.)):
        model(torch.zeros(1, 4))
        with pytest.raises(RuntimeError, match='more often'): model(torch.zeros(1, 4))
    replays = {'0': Replay('0', [torch.zeros(1, 4)]*2)}
    with intervention_hooks(model, replay_specs(component, replays, 1.)): model(torch.zeros(1, 4))
    with pytest.raises(RuntimeError, match='1 of 2'): check_replayed(replays, 1.)
    check_replayed(replays, 0.)


def test_patch_prompts_are_token_matched(config):
    pytest.importorskip('tokenizers')
    from vla_safety.obstacle.prompts import check_patch_prompts
    revision = json.loads((ROOT/'configs/models.json').read_text())['so101']['revision']
    if not (ROOT/'artifacts/models/so101'/revision/'tokenizer.json').exists(): pytest.skip('Tokenizer unavailable')
    counts, problems = check_patch_prompts(config, ROOT)
    assert not problems and counts['neutral'] == counts['avoid'] == counts['sham'] > counts['plain']


def test_report_pairs_arms_and_scenes(tmp_path):
    from vla_safety.obstacle.report import write_report
    results = []
    for layout, split in [('r17', 'discovery'), ('r18p', 'evaluation')]:
        for scene in ('tower', 'no_tower'):
            for seed in (1000, 1001):
                for arm in ('neutral', 'patch_lm_all'):
                    avoided = arm == 'patch_lm_all' and scene == 'tower'
                    error = layout == 'r18p' and seed == 1001 and arm == 'neutral' and scene == 'tower'
                    results.append({'episode': f'{layout}-{scene}-{arm}-s{seed}', 'termination': 'error' if error else 'success',
                                    'condition': {'layout': {'id': layout, 'split': split}, 'scene': scene, 'arm': arm, 'seed': seed},
                                    'grasped': True, 'crossed': True, 'crossing_mode': 'over' if avoided else 'through',
                                    'crossing': {'pen_bottom_m': .13 if avoided else .05}, 'direct': not avoided, 'avoided': avoided,
                                    'tower_contact': (not avoided) if scene == 'tower' else None, 'task_success': True,
                                    'predictions': 10, 'summary': {'arm_inactive': False}})
    suite = tmp_path/'suite'; suite.mkdir()
    (suite/'results.json').write_text(json.dumps(results)); (suite/'status.json').write_text('{"status": "error"}')
    before = sorted(p.name for p in suite.iterdir())
    report = json.loads(write_report(suite)['json'].read_text())
    assert sorted(p.name for p in suite.iterdir()) == before
    cell = next(c for c in report['cells'] if (c['scope'], c['scene'], c['arm']) == ('all', 'tower', 'patch_lm_all'))
    assert cell['avoided_given_crossed'] == {'k': 4, 'n': 4} and cell['modes']['over'] == 4
    contrast = next(c for c in report['contrasts'] if (c['contrast'], c['scene']) == ('patch_lm_all_vs_neutral', 'tower'))
    assert (contrast['pairs'], contrast['reference_direct_compared_avoided']) == (3, 3)  # the failed episode drops its pair
    assert contrast['crossing_height_difference_cm']['median'] == pytest.approx(8)
    scene = next(c for c in report['contrasts'] if c['contrast'] == 'neutral_tower_vs_no_tower')
    assert scene['pairs'] == 3 and scene['reference_direct_compared_avoided'] == 0


def test_progress_entry_uses_the_obstacle_description(config, tmp_path):
    from vla_safety.simulation.runner import record_progress
    plan = tmp_path/'plan.md'; plan.write_text('# Plan\n')
    config['progress'].update(enabled=True, markdown_path='plan.md')
    record_progress(config, tmp_path, tmp_path/'run', 'check', {'ok': True}, 'scripted direct/over/around scored as expected.')
    text = plan.read_text()
    assert 'scripted direct/over/around' in text and 'does not test policy competence' not in text
