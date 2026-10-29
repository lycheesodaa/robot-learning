#!/usr/bin/env python3
"""Run robot_learning_rulebased's online fragile-release evaluation with zero-shot MolmoAct2.

Everything is the rulebased pipeline, unmodified (scripts/collect_vla_rollouts.py ->
safety/vla_runner.rollout_logged -> eval/eval_policy.py: env, prompt, obs_to_batch/act_to_sim,
6 env steps per action, re-query every 10 actions, outcome monitor, shield). Only the policy
differs: allenai/MolmoAct2-SO100_101 zero-shot instead of the v6 LoRA. It is normalized with
its own norm_tag statistics; the rulebased `--policy zeroshot` path normalized with the
collected sim dataset's statistics instead, and needs that dataset on disk.

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .envs/lerobot/bin/python scripts/rulebased_zeroshot.py \
      --tag test9000 --seeds 9000-9999 --continue-after-success \
      --assignment <rulebased>/artifacts/eval/online_assignment_9000_9999.json --only-context fragile \
      --out artifacts/rulebased_zeroshot/artifacts/rollouts

Any collect_vla_rollouts.py argument works; --policy is ignored.
"""
import os
import runpy
import sys
from pathlib import Path

RULEBASED = Path(os.environ.get('RULEBASED', Path(__file__).resolve().parents[2] / 'robot_learning_rulebased/molmoact2-so101-sim'))
sys.path[:0] = [str(RULEBASED), str(RULEBASED / 'src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# eval_policy imports LeRobotDatasetMetadata (needs the `datasets` package) only for its own main();
# a stand-in keeps the .envs/lerobot environment unchanged.
try:
    import lerobot.datasets.lerobot_dataset  # noqa: F401
except ImportError:
    import types
    stub = types.ModuleType('lerobot.datasets.lerobot_dataset')
    stub.LeRobotDatasetMetadata = None
    sys.modules['lerobot.datasets.lerobot_dataset'] = stub

from safety import vla_runner as R  # noqa: E402  (builds the sim env, as the rulebased scripts do)


def load_zeroshot(spec=None, n_action_steps=10):
    """eval_policy.build_policy('zeroshot', ...) with the checkpoint's own normalization."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_policy_config, make_pre_post_processors
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    cfg = make_policy_config(
        'molmoact2', checkpoint_path='allenai/MolmoAct2-SO100_101', norm_tag='so100_so101_molmoact2',
        model_dtype='bfloat16', chunk_size=30, n_action_steps=30,
        setup_type='single so100/so101 robotic arm in molmoact2', control_mode='absolute joint pose',
        image_keys=['observation.images.cam0', 'observation.images.cam1'],
        normalize_gripper=True, inference_action_mode='continuous', device='cuda')
    cfg.input_features = {k: PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
                          for k in ('observation.images.cam0', 'observation.images.cam1')}
    cfg.input_features['observation.state'] = PolicyFeature(FeatureType.STATE, (6,))
    cfg.output_features = {'action': PolicyFeature(FeatureType.ACTION, (6,))}
    pol = MolmoAct2Policy(cfg).to('cuda')
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=None)  # None -> norm_tag stats
    pol.config.per_episode_seed = True
    pol.config.n_action_steps = n_action_steps
    pol.config.eval_seed = 1000
    pol.eval()
    return pol, pre, post


R.load_policy = load_zeroshot

# Diagnostics (default off): ZS_PAN_SIGN=-1 flips shoulder_pan in both obs_to_batch and act_to_sim;
# ZS_CAM0=<realism preset> moves cam0 (e.g. 'shoulder', behind the robot, instead of 'front_low').
if os.environ.get('ZS_PAN_SIGN'):
    R.EP.JS[0] = float(os.environ['ZS_PAN_SIGN'])
if os.environ.get('ZS_CAM0'):
    R.EP.realism.set_camera(R.env, os.environ['ZS_CAM0'])
# ZS_ACTIONS=<n> executes n of the 30 predicted actions per prediction (rulebased: 10, v6's chunk);
# ZS_SUB=<k> holds each action for k env steps of 0.02 s (rulebased: 6, i.e. 0.12 s). Scale --max-pol
# by 6/k to keep the episode's simulated duration.
if os.environ.get('ZS_ACTIONS'):
    load_zeroshot.__defaults__ = (None, int(os.environ['ZS_ACTIONS']))
if os.environ.get('ZS_SUB'):
    R.SUB = R.EP.SUB = int(os.environ['ZS_SUB'])


class _Cycle:
    """rollout_logged runs `for _ in range(SUB)` once per action; cycling the count gives rates
    the 0.02 s env step does not divide, e.g. 2,2,1 = 3 actions per 0.1 s = 30 Hz."""
    def __init__(self, counts):
        self.counts, self.i = counts, -1

    def __index__(self):
        self.i += 1
        return self.counts[self.i % len(self.counts)]


# ZS_SUB_PATTERN=2,2,1 cycles the env steps per action (overrides ZS_SUB). The cycle continues
# across episodes, so it starts at a different phase each episode; harmless for one-action offsets.
if os.environ.get('ZS_SUB_PATTERN'):
    R.SUB = _Cycle([int(x) for x in os.environ['ZS_SUB_PATTERN'].split(',')])

# ZS_SHIELD_HOLDING=1 adds v2's encoder `holding` to the shield's v4_raw_full features, so a rule
# can combine them (v4 alone has no holding signal). v2 calls a settled gripper above -8.5 deg
# blocked, calibrated on v6, which closes to -10 deg; zero-shot's "closed" stops near -7 deg, so its
# empty gripper reads as holding. ZS_SHIELD_HOLDING=gap:<deg> instead calls it holding when the
# previous command was "close", the gripper has settled, and it stopped more than <deg> short of
# that command, which does not depend on where the policy's "closed" is. Either way it also adds
# `dcmd_gripper`, the proposed gripper target minus the previous sent command (0 at t=0): an
# opening of the command, which v4's dtarget_gripper (target minus angle) misses while a held
# object keeps the angle above the command.
if os.environ.get('ZS_SHIELD_HOLDING'):
    import numpy as np
    from safety import vocab as Vocab
    _compute = Vocab.compute
    _mode = os.environ['ZS_SHIELD_HOLDING']
    _gap = np.deg2rad(float(_mode.split(':')[1])) if _mode.startswith('gap:') else None

    def _gap_holding(q_before, q_before_prev, q_sent_prev):
        if q_before_prev is None or q_sent_prev is None:
            return False
        q, cmd = float(q_before[5]), float(q_sent_prev[5])
        return bool(cmd < Vocab.ENC_CLOSED_CMD_RAD and q - cmd > _gap
                    and abs(q - float(q_before_prev[5])) < Vocab.ENC_SETTLED_RAD)

    def _compute_with_holding(U, q_before, q_target, fragile, q_before_prev=None, q_target_prev=None):
        out = _compute(U, q_before, q_target, fragile, q_before_prev, q_target_prev)
        holding = (out['v2_engineered_encoder']['holding'] if _gap is None
                   else _gap_holding(q_before, q_before_prev, q_target_prev))
        dcmd = 0.0 if q_target_prev is None else float(q_target[5] - q_target_prev[5])
        out['v4_raw_full'] = dict(out['v4_raw_full'], holding=holding, dcmd_gripper=dcmd)
        return out
    Vocab.compute = _compute_with_holding
sys.argv = [str(RULEBASED / 'scripts/collect_vla_rollouts.py'), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
