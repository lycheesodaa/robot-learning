from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
from vla_safety.types import ActionChunk, Observation


def action():
    return ActionChunk(np.zeros((2,6)), 'so101', 'absolute_joint_pose', 'checkpoint_joint_convention',
                       ('checkpoint_native_joint',)*6, 'repo@revision:so100_so101_molmoact2')


def test_action_conventions_and_nonfinite():
    a = action()
    with pytest.raises(ValueError, match='control_mode'):
        a.require_compatible(replace(a, control_mode='delta_end_effector_pose'))
    with pytest.raises(ValueError, match='finite'):
        replace(a, values=np.full((2,6), np.nan))
    assert not a.values.flags.writeable


def test_observation_validation():
    image = Image.new('RGB', (8, 8))
    obs = Observation((image, image), 'place it', [1, 2], 'so101', 0)
    assert not obs.state.flags.writeable
    with pytest.raises(ValueError, match='finite'):
        Observation((image,), 'place it', [np.nan], 'so101', 0)
    with pytest.raises(ValueError, match='required'):
        Observation((), 'place it', [0], 'so101', 0)
