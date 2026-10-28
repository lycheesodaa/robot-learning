"""Return the arm to the pose it started in whenever the robot client stops.

LeRobot's robot_client leaves the arm wherever the last action put it. install() patches
robot_client so every robot it builds records its joint positions on connect() and drives
back to them, slowly, before disconnect(). RobotClient.stop() calls disconnect() on a normal
exit, on Ctrl-C and on any exception in the control loop. Also routed there: SIGTERM/SIGHUP
(Linux) and Ctrl-Break (Windows). Closing the console window on Windows homes from the console
handler instead; Windows kills the process ~5 s after that event, hence MAX_S.

Not covered: kill -9, taskkill /F, a crashed interpreter or lost power. The process never
gets to run then, and the arm holds its last pose (disable_torque_on_disconnect: false).

A second Ctrl-C while homing stops the arm where it is.

Keys in `robot.homing_guard.hold_keys` are not homed: they keep their last command, e.g. a
gripper that must not drop what it holds (nexus_client.py adds gripper.pos during fragile runs).
"""

import signal
import sys
import threading
import time

from lerobot.async_inference import robot_client

SPEED = 60.0           # deg/s (gripper: %/s) for the joint with the furthest to go
MIN_S, MAX_S = 1.0, 3.0  # bounds on the homing duration
STEP_S = 1 / 30
SETTLE_S = 0.5         # keep the final goal for this long so the servos arrive

_robots: list["_Homing"] = []
_console_handler = None  # keep the ctypes callback alive


class _Homing:
    def __init__(self, robot):
        self.robot = robot
        self.home: dict[str, float] | None = None
        self.homing = False
        self.last_sent: dict[str, float] | None = None
        self.hold_keys: set[str] = set()  # left at their last command instead of homed
        # Serialises bus access between the control loop and homing from another thread.
        self.lock = threading.RLock()
        self._connect, self._send = robot.connect, robot.send_action
        self._observe, self._disconnect = robot.get_observation, robot.disconnect
        robot.connect, robot.send_action = self.connect, self.send_action
        robot.get_observation, robot.disconnect = self.get_observation, self.disconnect
        robot.homing_guard = self

    def connect(self, *args, **kwargs):
        self._connect(*args, **kwargs)
        obs = self._observe()
        self.home = {k: float(obs[k]) for k in self.robot.action_features}
        _robots.append(self)
        print(f"[home] start pose: { {k: round(v, 1) for k, v in self.home.items()} }", flush=True)

    def send_action(self, action):
        with self.lock:
            if self.homing:  # the policy no longer drives the arm
                return dict(action)
            performed = self._send(action)
            self.last_sent = {k: float(v) for k, v in performed.items()}
            return performed

    def get_observation(self):
        with self.lock:
            return self._observe()

    def disconnect(self, *args, **kwargs):
        self.go_home()
        self._disconnect(*args, **kwargs)

    def go_home(self):
        with self.lock:
            if self.home is None or self.homing:
                return
            self.homing = True
            try:
                obs = self._observe()
                start = {k: float(obs[k]) for k in self.home}
                home = dict(self.home)
                for k in self.hold_keys & set(self.last_sent or {}):
                    start[k] = home[k] = self.last_sent[k]
                    print(f"[home] keeping {k} at its last command {home[k]:.1f}", flush=True)
                dist = max(abs(home[k] - start[k]) for k in home)
                duration = min(max(dist / SPEED, MIN_S), MAX_S)
                print(f"[home] returning to start pose over {duration:.1f} s "
                      "(Ctrl-C again to stop the arm where it is)", flush=True)
                t0 = time.perf_counter()
                while True:
                    f = min((time.perf_counter() - t0) / duration, 1.0)
                    a = f * f * (3 - 2 * f)  # smoothstep: no jerk at either end
                    self._send({k: start[k] + (home[k] - start[k]) * a for k in home})
                    if f >= 1.0:
                        break
                    time.sleep(STEP_S)
                time.sleep(SETTLE_S)
                print("[home] back at start pose", flush=True)
            except KeyboardInterrupt:
                print("[home] interrupted; the arm holds its current pose", flush=True)
            except Exception as e:
                print(f"[home] failed: {e!r}", flush=True)


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _install_console_close_handler():
    import ctypes
    from ctypes import wintypes

    close_events = {2, 5, 6}  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT

    def handler(event):
        if event not in close_events:
            return False  # Ctrl-C / Ctrl-Break: let Python raise KeyboardInterrupt
        for robot in list(_robots):
            robot.go_home()  # waits for a homing already under way in the main thread
        return True

    global _console_handler
    _console_handler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)


def install() -> None:
    """Patch robot_client so every RobotClient (or subclass) built afterwards homes on exit."""
    make_robot = robot_client.make_robot_from_config

    def make_homing_robot(config):
        robot = make_robot(config)
        _Homing(robot)
        return robot

    robot_client.make_robot_from_config = make_homing_robot
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)
    if sys.platform == "win32":
        _install_console_close_handler()


def pop_no_home() -> bool:
    """Take --no_home out of sys.argv before draccus parses the robot_client config."""
    no_home = "--no_home" in sys.argv[1:]
    sys.argv = [a for a in sys.argv if a != "--no_home"]
    return no_home
