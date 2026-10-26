#!/usr/bin/env python3
"""LeRobot 0.6.1 async policy server with MolmoAct2 enabled.

lerobot.async_inference only accepts policy types listed in SUPPORTED_POLICIES,
which omits molmoact2. This adds it, then runs the stock server, so every
policy_server option still works:

  .envs/lerobot/bin/python server/serve_molmoact2.py --host=127.0.0.1 --port=8080 --fps=30

The server unpickles whatever clients send. Bind to 127.0.0.1 and reach it over
an SSH tunnel; never expose the port.

Every inference also logs one "Chunk #..." line comparing the motion the policy
asked for with the motion the arm made since the previous inference (joint units:
degrees, gripper 0-100). These options are handled here and not passed on:

  --no_similarity_filter  Run inference on every observation, instead of skipping
                          those whose joints moved < 1 (L2) since the last inference.
                          While the arm is still, the stock filter lets only must_go
                          observations through, so the policy re-plans every ~4 s.
  --record_dir=DIR        Save each inference's camera frames (JPEG) and a line in
                          inferences.jsonl with the task, joint state and returned
                          chunk. One subfolder per client connection.
  --rtc                   Real-time chunking: guide each new chunk toward the part of
                          the previous chunk the arm will still be executing when it
                          arrives, so consecutive chunks join without a jump. Run the
                          client with --aggregate_fn_name=latest_only, so the arm
                          executes exactly the chunks the server remembers.
  --rtc_horizon=N         Steps of the previous chunk to follow (default 20): fully
                          for the delay steps, then fading linearly to zero at N.
  --rtc_delay=N           Fix the number of steps the client executes before a chunk
                          arrives, instead of estimating it from the client's action
                          rate and the measured latency.
  --rtc_net_latency=S     Upload plus download time the server cannot measure
                          (default 0.25 s, about one observation upload over the tunnel).
"""
import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.async_inference import helpers
from lerobot.utils.utils import init_logging

LOG_DIR = Path(__file__).resolve().parents[1] / 'artifacts/logs'


def get_logger(name, log_to_file=True):
    # helpers.get_logger, but writing to artifacts/logs/ instead of ./logs/.
    log_file = None
    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f'{name}_{int(time.time())}.log'
    init_logging(log_file=log_file, display_pid=False)
    return logging.getLogger(name)


# PolicyServer creates its logger when policy_server is imported, through a
# by-name import of get_logger, so the replacement must be in place first.
helpers.get_logger = get_logger

from lerobot.async_inference import constants, policy_server  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

# policy_server imported this same list object, so appending updates its check.
if 'molmoact2' not in constants.SUPPORTED_POLICIES:
    constants.SUPPORTED_POLICIES.append('molmoact2')

OBS_STATE = 'observation.state'
IMAGE_PREFIX = 'observation.images.'


def disable_similarity_filter():
    # policy_server imported observations_similar by name, so patch that reference.
    policy_server.observations_similar = lambda *args, **kwargs: False


def install_rtc(horizon, fixed_delay, net_latency):
    server_cls = policy_server.PolicyServer
    send_policy_instructions = server_cls.SendPolicyInstructions
    enqueue_observation = server_cls._enqueue_observation
    predict_action_chunk = server_cls._predict_action_chunk

    def wrapped_send_policy_instructions(self, request, context):
        result = send_policy_instructions(self, request, context)
        self.policy.config.rtc_config = RTCConfig(execution_horizon=horizon)
        self.policy.init_rtc_processor()
        # RTC's guidance only needs gradients w.r.t. the trajectory; without this,
        # every denoising step also builds a graph through the action expert.
        self.policy.requires_grad_(False)
        self._rtc_prev = None  # (first timestep, normalized [T, A] chunk) last sent
        self._rtc_prev_obs = None  # (timestep, client time) of the last inference
        self._rtc_rate = None  # actions the client executes per second
        self._rtc_inference_s = 0.45
        self._rtc_info = {}
        return result

    def wrapped_enqueue_observation(self, obs):
        obs.server_received = time.time()
        return enqueue_observation(self, obs)

    def wrapped_get_action_chunk(self, observation):
        chunk = self.policy.predict_action_chunk(observation, **self._rtc_kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]
        self._rtc_chunk = chunk.squeeze(0).clone()  # before unnormalization, as RTC needs
        return chunk

    def wrapped_predict_action_chunk(self, observation_t):
        timestep, client_time = observation_t.get_timestep(), observation_t.get_timestamp()
        if self._rtc_prev_obs is not None:
            steps, seconds = timestep - self._rtc_prev_obs[0], client_time - self._rtc_prev_obs[1]
            if steps > 0 and seconds > 0:  # both on the client clock, so no clock skew
                rate = steps / seconds
                self._rtc_rate = rate if self._rtc_rate is None else 0.7 * self._rtc_rate + 0.3 * rate
        self._rtc_prev_obs = (timestep, client_time)

        started = time.time()
        latency = (started - getattr(observation_t, 'server_received', started)
                   + self._rtc_inference_s + net_latency)
        rate = self._rtc_rate or self.config.fps
        # Index 0 is the action the client had just executed at capture time; the client
        # drops every index it passes before the chunk arrives, so those must match.
        delay = fixed_delay if fixed_delay is not None else math.ceil(rate * latency) + 1

        leftover = None
        if self._rtc_prev is not None and 0 <= timestep - self._rtc_prev[0] < len(self._rtc_prev[1]):
            leftover = self._rtc_prev[1][timestep - self._rtc_prev[0]:].unsqueeze(0)
        self._rtc_kwargs = {'prev_chunk_left_over': leftover,
                            'inference_delay': delay if leftover is not None else 0}
        chunk = predict_action_chunk(self, observation_t)
        self._rtc_inference_s = 0.7 * self._rtc_inference_s + 0.3 * (time.time() - started)
        self._rtc_prev = (timestep, self._rtc_chunk)

        self._rtc_info = {
            'rtc_prefix_steps': 0 if leftover is None else leftover.shape[1],
            'rtc_delay': self._rtc_kwargs['inference_delay'],
            'rtc_client_rate': self._rtc_rate,
            'rtc_latency_s': latency,
        }
        self.logger.info(
            f'RTC #{timestep} | prefix {self._rtc_info["rtc_prefix_steps"]} steps, '
            f'delay {self._rtc_info["rtc_delay"]} | client {rate:.1f} actions/s, '
            f'estimated latency {latency:.2f}s'
        )
        return chunk

    server_cls.SendPolicyInstructions = wrapped_send_policy_instructions
    server_cls._enqueue_observation = wrapped_enqueue_observation
    server_cls._get_action_chunk = wrapped_get_action_chunk
    server_cls._predict_action_chunk = wrapped_predict_action_chunk


def _write_record(folder, record, images, logger):
    try:
        for cam, frame in images.items():
            Image.fromarray(frame).save(folder / f"{record['index']:05d}_{cam}.jpg", quality=90)
        with open(folder / 'inferences.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.warning(f'Could not record inference {record["index"]}: {e}')


def install_motion_log(record_dir):
    server_cls = policy_server.PolicyServer
    send_policy_instructions = server_cls.SendPolicyInstructions
    predict_action_chunk = server_cls._predict_action_chunk
    writer = ThreadPoolExecutor(max_workers=1)  # keeps JPEG encoding off the inference path

    def wrapped_send_policy_instructions(self, request, context):
        self._prev_inference_state = None
        self._record_index = 0
        self._record_folder = None
        if record_dir is not None:
            self._record_folder = record_dir / datetime.now().strftime('%Y%m%d-%H%M%S')
            self._record_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Recording inferences to {self._record_folder}')
        return send_policy_instructions(self, request, context)

    def wrapped_predict_action_chunk(self, observation_t):
        chunk = predict_action_chunk(self, observation_t)
        try:
            raw = observation_t.get_observation()
            names = self.lerobot_features[OBS_STATE]['names']
            state = np.array([raw[n] for n in names], dtype=np.float32)
            actions = np.stack([a.get_action().numpy() for a in chunk])
            requested = actions - state

            prev = getattr(self, '_prev_inference_state', None)
            moved = None if prev is None else float(np.linalg.norm(state - prev))
            self._prev_inference_state = state
            worst = int(np.abs(requested).max(axis=0).argmax())
            summary = {
                'chunk_end_delta_norm': float(np.linalg.norm(requested[-1])),
                'max_abs_delta': float(np.abs(requested).max()),
                'max_abs_delta_joint': names[worst],
                'moved_since_last_inference_norm': moved,
            }
            self.logger.info(
                f'Chunk #{observation_t.get_timestep()} (must_go: {observation_t.must_go}) | '
                f'requested: end-of-chunk |d| {summary["chunk_end_delta_norm"]:.2f}, '
                f'max |d| {summary["max_abs_delta"]:.2f} ({names[worst]}) | '
                f'moved since last inference: {"n/a" if moved is None else f"{moved:.2f}"}'
            )

            folder = getattr(self, '_record_folder', None)
            if folder is not None:
                record = {
                    'index': self._record_index,
                    'timestep': observation_t.get_timestep(),
                    'must_go': observation_t.must_go,
                    'client_time': observation_t.get_timestamp(),
                    'server_time': time.time(),
                    'task': raw.get('task'),
                    'joint_names': names,
                    'state': state.tolist(),
                    'chunk': actions.tolist(),
                    **summary,
                    **getattr(self, '_rtc_info', {}),
                }
                self._record_index += 1
                cams = [k.removeprefix(IMAGE_PREFIX) for k in self.lerobot_features if k.startswith(IMAGE_PREFIX)]
                writer.submit(_write_record, folder, record, {c: raw[c] for c in cams}, self.logger)
        except Exception as e:
            self.logger.warning(f'Motion log failed for observation #{observation_t.get_timestep()}: {e}')
        return chunk

    server_cls.SendPolicyInstructions = wrapped_send_policy_instructions
    server_cls._predict_action_chunk = wrapped_predict_action_chunk


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--no_similarity_filter', action='store_true')
    parser.add_argument('--record_dir', type=Path)
    parser.add_argument('--rtc', action='store_true')
    parser.add_argument('--rtc_horizon', type=int, default=20)
    parser.add_argument('--rtc_delay', type=int)
    parser.add_argument('--rtc_net_latency', type=float, default=0.25)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # the rest goes to the stock server's parser

    if args.no_similarity_filter:
        disable_similarity_filter()
    if args.rtc:  # before the motion log, so its records include the RTC fields
        install_rtc(args.rtc_horizon, args.rtc_delay, args.rtc_net_latency)
    install_motion_log(args.record_dir)
    policy_server.serve()


if __name__ == '__main__':
    main()
