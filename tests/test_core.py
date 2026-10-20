from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from vla_safety.types import ActionChunk, Observation
from vla_safety.constraints.base import checked_filter, IdentityFilter, GeometricFilter, ConstraintFailure, FilterResult
from vla_safety.evaluation.records import StepRecord, EpisodeRecord, summarize, validate_splits
from vla_safety.policies.wire import encode_observation, decode_observation


def action():
    return ActionChunk(np.zeros((2,7)), 'libero_franka', 'delta_end_effector_pose', 'controller',
                       ('scaled',)*7, 'repo@revision:libero')


def test_action_conventions_and_nonfinite():
    a = action()
    with pytest.raises(ValueError, match='control_mode'):
        a.require_compatible(replace(a, control_mode='absolute_joint_pose'))
    with pytest.raises(ValueError, match='finite'):
        replace(a, values=np.full((2,7), np.nan))
    assert not a.values.flags.writeable


def test_filters_fail_closed():
    a = action()
    assert checked_filter(IdentityFilter(), a, {}).action is a
    with pytest.raises(ConstraintFailure):
        checked_filter(GeometricFilter(), a, {})
    class Bad:
        def filter(self, proposed, state):
            return FilterResult(replace(proposed, values=np.ones((2,7))), False, '', 0)
    with pytest.raises(ConstraintFailure):
        checked_filter(Bad(), a, {})


def test_metrics_and_leakage():
    safe = StepRecord([[0]], [[0]], None, [], 0, False, 0, {})
    unsafe = StepRecord([[1]], [[0]], 2, ['collision'], 3, True, 1, {})
    a = EpisodeRecord('task', 'layout', 'pair', 'discovery', 1, 'rev', 'vanilla', True, [safe,unsafe])
    b = EpisodeRecord('task2','layout2','pair2','evaluation', 1, 'rev', 'combined', False, [safe])
    m = summarize([a,b])
    assert m['task_success_rate'] == .5
    assert m['episode_violation_rate'] == .5
    assert m['mean_cumulative_safety_cost'] == 1.5
    assert m['intervention_frequency'] == pytest.approx(1/3)
    assert m['mean_proposed_risk'] == 2
    validate_splits([a,b])
    with pytest.raises(ValueError, match='leakage'):
        validate_splits([a, replace(a, split='evaluation')])


def test_wire_roundtrip():
    obs = Observation((Image.new('RGB',(8,8),'red'),), 'test', [1,2], 'so101', 0)
    result = decode_observation(encode_observation(obs))
    np.testing.assert_array_equal(obs.state, result.state)
    np.testing.assert_array_equal(np.asarray(obs.images[0]), np.asarray(result.images[0]))
