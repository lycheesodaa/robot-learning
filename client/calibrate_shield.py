#!/usr/bin/env python3
"""Measure what the fragile-release shield needs from this arm and object; write the JSON that
server/serve_molmoact2_nexus.py takes as --shield_calibration (copy it to the server).

  python client/calibrate_shield.py --out shield_calibration.json --object "foam egg"
  python client/calibrate_shield.py --config client/so101_molmoact2_nexus.yaml --robot.port=COM7 ...

Guided, about two minutes. Keep a hand on the arm: the first steps run with torque off.
  1. gripper ends    jaws closed and fully open by hand; these readings are the sim's
                     -10 / +100 deg, i.e. the checkpoint's 0 / 45 (gripper scaling)
  2. set-down height pose the jaws around the object as it rests on the table; the FK
                     TCP height here becomes the sim's 0.0295 m (recalibrates tcp_z)
  3. squeeze load    torque on, the gripper closes on the object, then on nothing. Gives
                     the sign of Present_Load and the load_gripper threshold, halfway
                     between the two (the sim's -0.09 N m is too fine for a real servo)
Use the object you will run with: a soft object squeezes with less load than the sim cube.
At the end torque stays on and the arm holds its pose; cut power to release it.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np

from lerobot.async_inference.robot_client import RobotClientConfig
from lerobot.robots import make_robot_from_config

import fragile_shield as S

SETTLE_S = 1.5
SAMPLES = 20
MIN_SEPARATION_NM = 0.1  # squeeze must exceed the empty-close load by this much


def sample(bus, register, motor, n=SAMPLES, period=0.05):
    values = []
    for _ in range(n):
        values.append(float(bus.read(register, motor, normalize=register != 'Present_Load')))
        time.sleep(period)
    return statistics.median(values)


def joints(bus):
    pos = bus.sync_read('Present_Position')
    return np.array([pos[m] for m in S.JOINTS])


def ask(text):
    input(f'\n{text}\n  press ENTER when ready ')


def squeeze(bus, closed, open_):
    """Close on whatever is between the jaws; return the settled Present_Load (raw, signed)."""
    bus.write('Goal_Position', 'gripper', closed)
    time.sleep(SETTLE_S)
    load = sample(bus, 'Present_Load', 'gripper')
    position = sample(bus, 'Present_Position', 'gripper', n=5)
    bus.write('Goal_Position', 'gripper', open_)
    time.sleep(SETTLE_S)
    return load, position


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(Path(__file__).with_name('so101_molmoact2_nexus.yaml')))
    parser.add_argument('--out', type=Path, default=Path('shield_calibration.json'))
    parser.add_argument('--object', default='', help='Name of the test object, kept in the notes')
    parser.add_argument('--force', action='store_true',
                        help='Write measured=true even if holding and empty loads are not separable')
    args, overrides = parser.parse_known_args()
    cfg = draccus.parse(RobotClientConfig, config_path=args.config, args=overrides)

    robot = make_robot_from_config(cfg.robot)
    if not robot.calibration:
        raise SystemExit(f'No calibration for id {cfg.robot.id!r}; run lerobot-calibrate first.')
    bus = robot.bus
    bus.connect()
    try:
        if not bus.is_calibrated:
            raise SystemExit('Motor calibration differs from the file; connect once with the client or re-run lerobot-calibrate.')
        ask('Support the arm: torque is about to be turned OFF on every joint.')
        bus.disable_torque()

        print('\n1. Gripper ends')
        ask('Close the jaws fully by hand and hold them closed.')
        closed = sample(bus, 'Present_Position', 'gripper')
        ask('Open the jaws fully by hand.')
        open_ = sample(bus, 'Present_Position', 'gripper')
        print(f'  closed {closed:.1f}, open {open_:.1f} (LeRobot 0-100)')
        if open_ - closed < 30:
            raise SystemExit('Open and closed readings are too close; check the gripper calibration.')

        print('\n2. Set-down height')
        ask('Put the object on the table and pose the arm with the jaws around it, at the height a grasp '
            'would hold it. Keep holding the arm.')
        cal = S.Calibration(gripper_closed=closed, gripper_open=open_)
        q = joints(bus)
        rest_z = float(S.tcp_position(S.real_to_sim_rad(q, cal))[2])
        print(f'  TCP height {rest_z * 100:.1f} cm above the base frame (sim: {S.SIM_TCP_REST_Z_M * 100:.2f} cm)')

        print('\n3. Squeeze load')
        ask('Keep the jaws around the object, open enough that they do not touch it. Torque goes ON and the '
            'arm will hold this pose; the gripper closes on the object for 1.5 s and reopens.')
        present = bus.sync_read('Present_Position')
        bus.sync_write('Goal_Position', present)  # torque-on must not jump to a stale goal
        bus.enable_torque()
        opening = present['gripper']
        hold_raw, hold_pos = squeeze(bus, closed, opening)
        ask('Take the object out of the jaws. The gripper closes on nothing and reopens.')
        empty_raw, empty_pos = squeeze(bus, closed, opening)
        print(f'  Present_Load holding {hold_raw:+.0f} at {hold_pos:.1f}, empty {empty_raw:+.0f} at {empty_pos:.1f}')
    finally:
        bus.disconnect(disable_torque=False)

    sign = -1.0 if hold_raw > empty_raw else 1.0  # squeezing must read negative, like the sim force
    nm = lambda raw: sign * raw / S.LOAD_FULL_SCALE * S.STALL_TORQUE_NM  # noqa: E731
    hold_nm, empty_nm = nm(hold_raw), nm(empty_raw)
    threshold = (hold_nm + empty_nm) / 2
    separable = empty_nm - hold_nm >= MIN_SEPARATION_NM
    print(f'\n  sim-equivalent load: holding {hold_nm:+.3f} N m, empty {empty_nm:+.3f} N m '
          f'-> load_gripper threshold {threshold:+.3f} (sim rule: -0.091)')
    if not separable:
        print(f'  WARNING: holding and empty differ by less than {MIN_SEPARATION_NM} N m; a load-based rule cannot '
              'tell whether this object is held. Grip it harder (a firmer object) or use a rule without load_gripper.')

    loads_signs = [1.0] * 6
    loads_signs[5] = sign
    cal = S.Calibration(
        gripper_closed=closed, gripper_open=open_, load_signs=loads_signs, tcp_rest_z_m=rest_z,
        thresholds={'load_gripper': round(threshold, 4)}, measured=separable or args.force,
        notes={
            'date': datetime.now().isoformat(timespec='seconds'), 'robot_id': cfg.robot.id, 'object': args.object,
            'gripper_load_raw': {'holding': hold_raw, 'empty': empty_raw},
            'gripper_settled': {'holding': hold_pos, 'empty': empty_pos},
            'load_nm': {'holding': hold_nm, 'empty': empty_nm},
            'arm_load_signs': 'not measured (1.0); rules on arm loads are uncalibrated',
            'separable': separable,
        })
    args.out.write_text(json.dumps(cal.to_dict(), indent=2))
    print(f'\nWrote {args.out}' + ('' if cal.measured else ' with measured=false: the server allows only --shield=shadow'))
    print('Copy it to the server and pass --shield_calibration=<path>. Torque is still on; cut power to release the arm.')


if __name__ == '__main__':
    main()
