#!/usr/bin/env python3
"""serve_molmoact2.py for the so101-nexus MolmoAct2 fine-tune on a real SO-101, with the
fragile-release shield learned in simulation (robot_learning_rulebased).

  .envs/lerobot/bin/python server/serve_molmoact2_nexus.py --host=127.0.0.1 --port=8080 --fps=8 \
      --no_similarity_filter --record_dir=artifacts/real_runs \
      --fragile --shield=shadow --shield_calibration=server/shield_artifacts/calibration.json

Drive it with the client branch's nexus client (client/run_client.sh with NEXUS=1, or
run_client.ps1 -Nexus). That client checks every queued action again against live joint and
load readings right before sending it, and it empties its queue when the shield latches. The
stock client works too, but it executes whatever it has already queued.

What differs from the stock checkpoint path (these choices follow eval/eval_policy.py in the sim repo):
  * The checkpoint is chosen here (--checkpoint), not by the client. The conversions below are
    only right for this checkpoint.
  * The nexus checkpoint's processor applies no joint correction. This server converts the
    state from the real frame to the checkpoint frame and converts actions back, using
    fragile_shield.real_to_ckpt / ckpt_to_real. The gripper is rescaled from the real 0-100 to
    the checkpoint's 0 (closed) .. 45 (open). The checkpoint was trained on a binary gripper,
    so it asks for 0 or 45. Real gripper targets are clipped to 0-100.
  * inference_action_mode is set to 'continuous', because training does not save it.
  * The checkpoint predicts 10 steps, trained at one action per 0.12 s of sim time (6 control
    steps of 0.02 s). Run the server and the client at fps 8 with actions_per_chunk 10. The
    client's request is capped at the checkpoint's chunk size.
  * The prompt it was trained on is "Pick up the red cube and place it on the blue circle."

All serve_molmoact2.py options work (see that file). --rtc_horizon defaults to 10 here, the
chunk size. Added options:

  --checkpoint=PATH        default: artifacts/models/so101_nexus_v6/<rev> if present, else
                           ataghof/molmoact2-so101nexus-lora-champion
  --shield=MODE            off | shadow (default: decide and log, change nothing) |
                           hold (a refused action keeps the previous gripper command; the arm
                           joints move as proposed; the sim's gripper-hold) |
                           abort (the first refusal latches: from then on every request gets an
                           empty chunk, without inference, until the client reconnects)
  --shield_rule=PATH       refuse_if rule in sim units (default shield_rules/v4_symbolic_k3.json)
  --shield_calibration=PATH  JSON written by client/calibrate_shield.py; hold and abort need one
  --fragile                the object is fragile (the rule's `fragile` input). Without it every
                           rule here allows everything.

The shield checks every action of a chunk. Action k is checked against the state the arm should
be in when it runs: the observed state for k = 0, and action k-1's arm targets for k > 0.
The gripper position is taken from the observation for every k, because a held object stops
the jaws short of their target. Servo loads come only from the observation. The server attaches
the shield config (rule, calibration, mode) and its decision to every action it sends, so the
nexus client applies the identical rule.
"""
import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

import serve_molmoact2 as base  # installs the artifacts/logs logger before policy_server loads
import fragile_shield as S
from serve_molmoact2 import OBS_STATE, policy_server

REPO = Path(__file__).resolve().parents[1]
HUB_CHECKPOINT = 'ataghof/molmoact2-so101nexus-lora-champion'
LOCAL_CHECKPOINT = REPO / 'artifacts/models/so101_nexus_v6/4fae052f77777d7fb4e423d50a7ade88db542e51'
DEFAULT_RULE = Path(__file__).resolve().parent / 'shield_rules/v4_symbolic_k3.json'
JOINT_KEYS = [f'{j}.pos' for j in S.JOINTS]
LOAD_KEYS = [f'{j}.load' for j in S.JOINTS]  # sent by the nexus client; the policy never sees them
MAX_VELOCITY_DT = 1.0  # s; older previous observations give no velocity


def _state(raw, keys):
    return np.array([raw[k] for k in keys], dtype=np.float64)


def install_nexus(checkpoint, cal):
    """Load the nexus checkpoint whatever the client asks for, and convert state and actions
    between the real frame and the checkpoint frame around inference."""
    server_cls = policy_server.PolicyServer
    send_policy_instructions = server_cls.SendPolicyInstructions
    predict_action_chunk = server_cls._predict_action_chunk

    def wrapped_send_policy_instructions(self, request, context):
        specs = policy_server.pickle.loads(request.data)  # nosec: same trust as the stock server
        if specs.pretrained_name_or_path != checkpoint:
            self.logger.info(f'Client asked for {specs.pretrained_name_or_path}; serving {checkpoint}')
        specs.pretrained_name_or_path = checkpoint
        request = policy_server.services_pb2.PolicySetup(data=policy_server.pickle.dumps(specs))
        result = send_policy_instructions(self, request, context)
        if self.policy is None:
            return result
        cfg = self.policy.config
        if cfg.joint_signs is not None:
            raise RuntimeError(f'{checkpoint} applies its own joint correction; this server would apply it twice')
        names = self.lerobot_features[OBS_STATE]['names']
        if list(names) != JOINT_KEYS:
            raise RuntimeError(f'Expected state {JOINT_KEYS}, client sends {names}')
        cfg.inference_action_mode = 'continuous'
        if self.actions_per_chunk > cfg.chunk_size:
            self.logger.warning(f'Client asked for {self.actions_per_chunk} actions per chunk; '
                                f'the checkpoint predicts {cfg.chunk_size}')
            self.actions_per_chunk = cfg.chunk_size
        self.logger.info(
            f'Nexus frames: gripper real {cal.gripper_closed:g}..{cal.gripper_open:g} = checkpoint 0..'
            f'{S.CKPT_GRIPPER_OPEN:g} | chunk {self.actions_per_chunk} at {self.config.fps} fps')
        return result

    def wrapped_predict_action_chunk(self, observation_t):
        raw = observation_t.get_observation()
        ckpt = S.real_to_ckpt(_state(raw, JOINT_KEYS), cal)
        converted = copy.copy(observation_t)  # keeps must_go and the RTC receive time
        converted.observation = {**raw, **dict(zip(JOINT_KEYS, ckpt.tolist()))}
        chunk = predict_action_chunk(self, converted)
        self.last_processed_obs = observation_t  # the similarity filter compares real-frame states
        for a in chunk:
            real = S.ckpt_to_real(a.get_action().double().numpy(), cal)
            real[5] = np.clip(real[5], 0.0, 100.0)
            a.action = torch.from_numpy(real).to(a.get_action().dtype)
        return chunk

    server_cls.SendPolicyInstructions = wrapped_send_policy_instructions
    server_cls._predict_action_chunk = wrapped_predict_action_chunk


def install_shield(rule, mode, fragile, cal):
    server_cls = policy_server.PolicyServer
    send_policy_instructions = server_cls.SendPolicyInstructions
    predict_action_chunk = server_cls._predict_action_chunk
    config = S.shield_config(rule, mode, fragile, cal)
    rule_feats = sorted(S.rule_features(rule))

    def wrapped_send_policy_instructions(self, request, context):
        # A new connection is the operator reset: an earlier latch is cleared here only.
        self._shield_latched = None  # (timestep, reason) once abort fired
        self._shield_prev_obs = None  # (client time, real state) of the last inference
        self._shield_last_sent = None  # (first timestep, [T, 6] real actions) last sent
        self._shield_info = {}
        self.logger.info(f'Shield {mode} | fragile: {fragile} | {S.to_text(rule)}')
        return send_policy_instructions(self, request, context)

    def wrapped_predict_action_chunk(self, observation_t):
        t0 = observation_t.get_timestep()
        if self._shield_latched is not None:
            self.logger.warning(f'Shield latched at #{self._shield_latched[0]} ({self._shield_latched[1]}); '
                                f'no actions for #{t0}. Reconnect the client to resume.')
            return []

        chunk = predict_action_chunk(self, observation_t)
        raw = observation_t.get_observation()
        state = _state(raw, JOINT_KEYS)
        load = _state(raw, LOAD_KEYS) if all(k in raw for k in LOAD_KEYS) else None
        client_time = observation_t.get_timestamp()

        q_prev, dt = None, None
        if self._shield_prev_obs is not None:
            prev_time, prev_state = self._shield_prev_obs
            if 0 < client_time - prev_time <= MAX_VELOCITY_DT:
                q_prev, dt = prev_state, client_time - prev_time
        self._shield_prev_obs = (client_time, state)

        cmd = None  # the command the client was executing at capture time
        if self._shield_last_sent is not None:
            first, sent = self._shield_last_sent
            if 0 <= t0 - first < len(sent):
                cmd = sent[t0 - first]

        actions = np.stack([a.get_action().double().numpy() for a in chunk])
        decisions, shown, q_before = [], None, state
        for k in range(len(actions)):
            feats = S.features(q_before, actions[k], fragile, cal, load_raw=load,
                               q_prev_real=q_prev, dt=dt, prevcmd_real=cmd)
            refuse, reason = S.decide(rule, feats)
            if mode == 'off':
                refuse, reason = False, 'off'
            decisions.append((refuse, reason))
            if k == 0 or (refuse and not any(r for r, _ in decisions[:-1])):
                shown = (k, feats)  # log the first action, or the first refused one
            if refuse and mode == 'abort':
                self._shield_latched = (t0 + k, reason)
                break
            if refuse and mode == 'hold':
                actions[k, 5] = cmd[5] if cmd is not None else state[5]
            q_prev, dt = q_before, self.config.environment_dt
            q_before = np.concatenate([actions[k, :5], state[5:]])  # jaws stay where the object stops them
            cmd = actions[k]

        refused = [k for k, (r, _) in enumerate(decisions) if r]
        k, feats = shown
        self._shield_info = {
            'shield_mode': mode,
            'shield_refused': refused,
            'shield_reasons': sorted({why for r, why in decisions if r}),
            'shield_features_step': k,
            'shield_features': {f: feats.get(f) for f in rule_feats},
            'shield_latched': self._shield_latched is not None,
        }
        values = ', '.join(f'{f} {feats[f]:.3f}' if isinstance(feats.get(f), float) else f'{f} {feats.get(f)}'
                           for f in rule_feats)
        verdict = (f'REFUSE {len(refused)}/{len(decisions)} from step {refused[0]} '
                   f'({", ".join(self._shield_info["shield_reasons"])})' if refused else 'allow')
        log = self.logger.warning if refused and mode != 'shadow' else self.logger.info
        log(f'Shield #{t0} [{mode}] {verdict} | step {k}: {values}')

        if self._shield_latched is not None:
            self.logger.warning(f'Shield LATCHED at #{self._shield_latched[0]}: no further actions are sent '
                                'until the client reconnects.')
            return []

        for a, row, (r, why) in zip(chunk, actions, decisions):
            a.action = torch.from_numpy(row).to(a.get_action().dtype)
            a.shield = {'config': config, 'refused': r, 'reason': why}
        self._shield_last_sent = (t0, actions)
        return chunk

    server_cls.SendPolicyInstructions = wrapped_send_policy_instructions
    server_cls._predict_action_chunk = wrapped_predict_action_chunk


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--no_similarity_filter', action='store_true')
    parser.add_argument('--record_dir', type=Path)
    parser.add_argument('--rtc', action='store_true')
    parser.add_argument('--rtc_horizon', type=int, default=10)
    parser.add_argument('--rtc_delay', type=int)
    parser.add_argument('--rtc_net_latency', type=float, default=0.25)
    parser.add_argument('--checkpoint', default=str(LOCAL_CHECKPOINT) if LOCAL_CHECKPOINT.exists() else HUB_CHECKPOINT)
    parser.add_argument('--shield', choices=S.MODES, default='shadow')
    parser.add_argument('--shield_rule', type=Path, default=DEFAULT_RULE)
    parser.add_argument('--shield_calibration', type=Path)
    parser.add_argument('--fragile', action='store_true')
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    cal = S.Calibration.load(args.shield_calibration) if args.shield_calibration else S.Calibration()
    if args.shield in ('hold', 'abort') and not cal.measured:
        raise SystemExit(f'--shield={args.shield} needs a measured --shield_calibration '
                         '(client/calibrate_shield.py). Sim defaults are only good for --shield=shadow.')
    rule = S.load_rule(args.shield_rule)
    if cal.thresholds:
        rule = S.with_thresholds(rule, cal.thresholds)
    logging.getLogger('policy_server').info(f'Shield rule after calibration: {S.to_text(rule)}')

    if args.no_similarity_filter:
        base.disable_similarity_filter()
    # Innermost first: RTC works in the checkpoint frame, the shield and motion log in the real frame.
    if args.rtc:
        base.install_rtc(args.rtc_horizon, args.rtc_delay, args.rtc_net_latency)
    install_nexus(args.checkpoint, cal)
    install_shield(rule, args.shield, args.fragile, cal)
    base.install_motion_log(args.record_dir)
    policy_server.serve()


if __name__ == '__main__':
    main()
