#!/usr/bin/env python3
"""Loopback check of a running MolmoAct2 policy server; no robot needed.

Sends what lerobot's robot_client sends for an so101_follower with cameras
cam0/cam1: the official MolmoAct2 sample images and sample state (converted to
the SO101 arm frame), then prints the returned action chunk.

  .envs/lerobot/bin/python server/smoke_client.py --server_address=127.0.0.1:8080
"""
import argparse
import json
import pickle  # nosec
from pathlib import Path
import time

import grpc
import numpy as np
from PIL import Image

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

ROOT = Path(__file__).resolve().parents[1]
MOTORS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
# From the checkpoint config: model_state = signs * arm_state + offsets.
JOINT_SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
JOINT_OFFSETS = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server_address', default='127.0.0.1:8080')
    parser.add_argument('--pretrained_name_or_path', default='lerobot/MolmoAct2-SO100_101-LeRobot')
    parser.add_argument('--policy_device', default='cuda')
    parser.add_argument('--actions_per_chunk', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3, help='Inference calls after warm-up')
    args = parser.parse_args()

    config = json.loads((ROOT / 'configs/models.json').read_text())['so101']
    snapshot = ROOT / 'artifacts/models/so101' / config['revision']
    images = [np.asarray(Image.open(snapshot / 'assets' / name).convert('RGB')) for name in config['sample_images']]
    arm_state = JOINT_SIGNS * (np.array(config['sample_state']) - JOINT_OFFSETS)

    # Same features and handshake as lerobot.async_inference.robot_client.
    hw_features = {f'{m}.pos': float for m in MOTORS} | {f'cam{i}': image.shape for i, image in enumerate(images)}
    features = hw_to_dataset_features(hw_features, 'observation', use_video=False)
    policy_config = RemotePolicyConfig('molmoact2', args.pretrained_name_or_path, features,
                                       args.actions_per_chunk, args.policy_device)
    stub = services_pb2_grpc.AsyncInferenceStub(grpc.insecure_channel(args.server_address, grpc_channel_options()))
    stub.Ready(services_pb2.Empty())
    started = time.perf_counter()
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
    print(f'Policy loaded on server in {time.perf_counter() - started:.1f}s')

    raw = {f'{m}.pos': float(v) for m, v in zip(MOTORS, arm_state)}
    raw |= {f'cam{i}': image for i, image in enumerate(images)}
    raw['task'] = config['sample_instruction']
    for step in range(args.repeats + 1):
        obs = TimedObservation(timestamp=time.time(), timestep=step * 100, observation=raw, must_go=True)
        started = time.perf_counter()
        stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation))
        reply = stub.GetActions(services_pb2.Empty())
        elapsed = time.perf_counter() - started
        if not reply.data:
            raise SystemExit('Server returned no actions; check the server log for the error.')
        actions = np.stack([a.get_action().numpy() for a in pickle.loads(reply.data)])  # nosec
        print(f'{"Warm-up" if step == 0 else "Call " + str(step)}: {actions.shape} in {elapsed:.2f}s')

    assert actions.shape == (args.actions_per_chunk, len(MOTORS)), actions.shape
    assert np.isfinite(actions).all()
    np.set_printoptions(precision=2, suppress=True)
    print('Instruction:', raw['task'])
    print('State (arm frame):', arm_state)
    print('First 5 actions (arm frame):\n', actions[:5])
    print('Last action:', actions[-1])


if __name__ == '__main__':
    main()
