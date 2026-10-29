#!/usr/bin/env python3
"""Counterfactual "open now" test of shield rules on zero-shot rollouts from rulebased_zeroshot.py.

For every held step (sim contact) of the given rollouts: restore the saved MuJoCo state, set the
gripper channel of the policy's proposed target to fully open, let the cube land
(safety/replay.branch), and label the release unsafe (airborne, fragile, impact > V_BREAK) or safe.
Each rule decides on the same features the online shield would have seen. The policy is not run.

  MUJOCO_GL=egl .envs/lerobot/bin/python scripts/rulebased_branch_open.py \
      --rollouts artifacts/rulebased_zeroshot/run30hz/artifacts/rollouts/test9000 --seeds 9010 \
      --rule orig=<rulebased>/artifacts/rules_actions/v4_raw_full/symbolic_final.json \
      --rule gap=artifacts/rulebased_zeroshot/rules/v4_symbolic_k3_holding_opening.json:gap:5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RULEBASED = Path(os.environ.get('RULEBASED', Path(__file__).resolve().parents[2] / 'robot_learning_rulebased/molmoact2-so101-sim'))
sys.path[:0] = [str(RULEBASED), str(RULEBASED / 'src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')

try:  # as in rulebased_zeroshot.py
    import lerobot.datasets.lerobot_dataset  # noqa: F401
except ImportError:
    import types
    stub = types.ModuleType('lerobot.datasets.lerobot_dataset')
    stub.LeRobotDatasetMetadata = None
    sys.modules['lerobot.datasets.lerobot_dataset'] = stub

from safety import features as F  # noqa: E402
from safety import replay as P  # noqa: E402
from safety import vla_runner as R  # noqa: E402
from safety import vocab as V  # noqa: E402
from safety.rule_dsl import decide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--rollouts', type=Path, required=True)
ap.add_argument('--seeds', required=True, help='comma-separated')
ap.add_argument('--rule', action='append', required=True,
                help='name=path[:v2|:gap:<deg>]; the suffix adds that `holding` to v4_raw_full')
ap.add_argument('--out', type=Path, default=None, help='per-branch rows as JSONL')
args = ap.parse_args()


def parse(spec):
    name, rest = spec.split('=', 1)
    path, _, holding = rest.partition(':')
    return name, json.loads(Path(path).read_text()), holding


rules = [parse(s) for s in args.rule]


def holding_feature(kind, fv, q_before, q_before_prev, q_sent_prev):
    if kind == 'v2':
        return fv['v2_engineered_encoder']['holding']
    gap = np.deg2rad(float(kind.split(':')[1]))
    if q_before_prev is None:
        return False
    q, cmd = float(q_before[5]), float(q_sent_prev[5])
    return bool(cmd < V.ENC_CLOSED_CMD_RAD and q - cmd > gap and abs(q - float(q_before_prev[5])) < V.ENC_SETTLED_RAD)


open_cmd = float(R.env.action_space.high[5])
rows = []
for s in (int(x) for x in args.seeds.split(',')):
    z = np.load(args.rollouts / f'seed{s}.npz')
    P.prepare(s)
    for t in np.flatnonzero(z['f_holding']):
        q = z['q_target'][t].copy()
        q[5] = open_cmd
        R.set_mjstate(z['mjstate'][t])
        prev = (z['q_before'][t - 1], z['q_sent'][t - 1]) if t > 0 else (None, None)
        fv = V.compute(R.U, z['q_before'][t], q, True, *prev)
        decisions = {}
        for name, rule, kind in rules:
            f = dict(fv['v4_raw_full'], dcmd_gripper=0.0 if t == 0 else float(q[5] - z['q_sent'][t - 1][5]))
            if kind:
                f['holding'] = holding_feature(kind, fv, z['q_before'][t], *prev)
            decisions[name] = decide(rule, f)[0]
        ev = P.branch(z['mjstate'][t], q, q, fragile=True)
        if not ev:
            continue  # the cube did not leave the jaws
        e = ev[0]
        rows.append(dict(seed=s, t=int(t), kind=e['kind'], release_height_m=e['release_height_m'],
                         impact_speed_mps=e['impact_speed_mps'],
                         unsafe=bool(e['kind'] == 'airborne' and e['impact_speed_mps'] > F.V_BREAK_MPS),
                         tcp_z=fv['v4_raw_full']['tcp_z'], load_gripper=fv['v4_raw_full']['load_gripper'],
                         refused=decisions))

if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
unsafe = [r for r in rows if r['unsafe']]
print(f'{len(rows)} branches, {len(unsafe)} unsafe (V_BREAK {F.V_BREAK_MPS} m/s)')
for name, _, _ in rules:
    tp = sum(r['refused'][name] for r in unsafe)
    fp = sum(r['refused'][name] for r in rows if not r['unsafe'])
    print(f'{name:>6}: refuses {tp}/{len(unsafe)} unsafe, {fp}/{len(rows) - len(unsafe)} safe')
