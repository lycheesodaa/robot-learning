#!/usr/bin/env python3
"""Check whether the SO101 follower will jump when LeRobot enables torque.

LeRobot enables torque by writing only Torque_Enable=1; it never sets Goal_Position
to the present position first. Each servo then drives to whatever its Goal_Position
register holds, e.g. the last target from a previous run if the arm was moved by
hand since. This script reads that register without moving anything.

  python client/check_torque_hold.py                  # read only: goal vs present per joint
  python client/check_torque_hold.py --sync           # set goal = present (no motion), so torque-on holds pose
  python client/check_torque_hold.py --probe-firmware # 5 deg wrist_roll test: does torque-on use a stale goal?

Robot settings come from the client YAML; override as usual, e.g. --robot.port=COM7.
Never changes the torque state of the other joints; does not run calibration.
"""
import argparse
import time

import draccus

from lerobot.async_inference.robot_client import RobotClientConfig
from lerobot.robots import make_robot_from_config
from pathlib import Path

PROBE_MOTOR = 'wrist_roll'  # no gravity load, so a small test move is harmless
PROBE_DEG = 5.0


def table(bus):
    present = bus.sync_read('Present_Position')
    goal = bus.sync_read('Goal_Position')
    torque = bus.sync_read('Torque_Enable', normalize=False)
    print(f'{"joint":<14}{"torque":>7}{"present":>10}{"goal":>10}{"goal-present":>14}')
    for m in present:
        print(f'{m:<14}{"on" if torque[m] else "off":>7}{present[m]:>10.2f}{goal[m]:>10.2f}{goal[m] - present[m]:>14.2f}')
    return present, goal, torque


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(Path(__file__).with_name('so101_molmoact2.yaml')))
    parser.add_argument('--sync', action='store_true', help='Write Goal_Position = Present_Position')
    parser.add_argument('--probe-firmware', action='store_true',
                        help=f'Move {PROBE_MOTOR} by {PROBE_DEG} deg to see if torque-on uses the stored goal')
    parser.add_argument('--tolerance', type=float, default=3.0, help='Degrees counted as "no jump"')
    args, overrides = parser.parse_known_args()
    cfg = draccus.parse(RobotClientConfig, config_path=args.config, args=overrides)

    robot = make_robot_from_config(cfg.robot)
    if not robot.calibration:
        raise SystemExit(f'No calibration for id {cfg.robot.id!r}; run lerobot-calibrate first.')
    bus = robot.bus
    bus.connect()
    try:
        if not bus.is_calibrated:
            raise SystemExit('Motor calibration differs from the file; connect once with the client '
                             '(press ENTER to write it) or re-run lerobot-calibrate.')
        present, goal, torque = table(bus)
        worst = max(abs(goal[m] - present[m]) for m in present)
        if all(torque.values()):
            print('\nTorque is already on, so the arm is holding its goal; no jump on reconnect.')
        elif worst > args.tolerance:
            print(f'\nRISK: a stored goal is {worst:.1f} deg from the present pose. If the firmware uses it '
                  'on torque-on, the arm jumps there when the client connects. Run with --sync first.')
        else:
            print(f'\nOK: every stored goal is within {args.tolerance} deg of the present pose.')

        if args.probe_firmware:
            if any(torque.values()):
                raise SystemExit('Probe needs torque off on all joints (support the arm, then power-cycle or '
                                 'stop the client with disable_torque_on_disconnect=true).')
            start = present[PROBE_MOTOR]
            target = start + (PROBE_DEG if start < 0 else -PROBE_DEG)  # step toward the middle of the range
            bus.write('Goal_Position', PROBE_MOTOR, target)
            bus.enable_torque(PROBE_MOTOR)
            time.sleep(1.0)
            moved = bus.read('Present_Position', PROBE_MOTOR) - start
            bus.disable_torque(PROBE_MOTOR)
            bus.write('Goal_Position', PROBE_MOTOR, bus.read('Present_Position', PROBE_MOTOR))
            print(f'\nProbe: stored goal {target - start:+.1f} deg away, {PROBE_MOTOR} moved {moved:+.1f} deg on torque-on.')
            if abs(moved) > PROBE_DEG / 2:
                print('=> Firmware drives to the stored goal: a stale goal WILL make the arm jump. Use --sync before runs.')
            else:
                print('=> Firmware holds the present pose on torque-on: stale goals are harmless.')

        if args.sync:
            bus.sync_write('Goal_Position', bus.sync_read('Present_Position'))
            print('\nSynced Goal_Position to Present_Position:')
            table(bus)
            print('Start the client without moving the arm in between.')
    finally:
        bus.disconnect(disable_torque=False)  # leave torque exactly as it was


if __name__ == '__main__':
    main()
