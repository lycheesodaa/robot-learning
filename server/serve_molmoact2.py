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
degrees, gripper 0-100). Two diagnostic options are handled here and not passed on:

  --no_similarity_filter  Run inference on every observation, instead of skipping
                          those whose joints moved < 1 (L2) since the last inference.
                          While the arm is still, the stock filter lets only must_go
                          observations through, so the policy re-plans every ~4 s.
  --record_dir=DIR        Save each inference's camera frames (JPEG) and a line in
                          inferences.jsonl with the task, joint state and returned
                          chunk. One subfolder per client connection.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.async_inference import constants, policy_server

# policy_server imported this same list object, so appending updates its check.
if 'molmoact2' not in constants.SUPPORTED_POLICIES:
    constants.SUPPORTED_POLICIES.append('molmoact2')

OBS_STATE = 'observation.state'
IMAGE_PREFIX = 'observation.images.'


def disable_similarity_filter():
    # policy_server imported observations_similar by name, so patch that reference.
    policy_server.observations_similar = lambda *args, **kwargs: False


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
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # the rest goes to the stock server's parser

    if args.no_similarity_filter:
        disable_similarity_filter()
    install_motion_log(args.record_dir)
    policy_server.serve()


if __name__ == '__main__':
    main()
