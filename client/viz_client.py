"""LeRobot 0.6.1 robot_client with full-process recording and a live browser viewer.

Drop-in replacement for `python -m lerobot.async_inference.robot_client`: takes the same
arguments, plus
    --viz_port=8765        port of the viewer (http://127.0.0.1:<port>/)
    --runs_dir=<path>      where runs are saved (default: <repo>/runs)
    --no_browser           don't open the viewer automatically
    --image_size=HxW       shrink camera frames to HxW before sending (see resize_client.py);
                           recorded frames are then what the policy sees. Omit to send full frames.
    --no_home              leave the arm where it is on exit instead of returning it to its
                           start pose (see home_on_exit.py)

Recorded per run (see viz_server.py for the on-disk layout):
  obs    every observation sent: joint state, both camera frames, must-go flag, queue size, upload time
  chunk  every action chunk received, matched to the observation it was predicted from
         (chunk timestamp == observation timestamp), so latency and server-skipped observations are exact
  act    every action executed: commanded vs actually sent (after max_relative_target clamping)
  loop   every control-loop iteration: duration, whether it acted and/or sent an observation
"""

import logging
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

import home_on_exit
import resize_client
from viz_server import DEFAULT_RUNS_DIR, RunRecorder, serve


class VizRobotClient(RobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        features = self.robot.observation_features
        self.state_keys = [k for k, v in features.items() if not isinstance(v, tuple)]
        self.cam_keys = [k for k, v in features.items() if isinstance(v, tuple)]
        self.action_keys = list(self.robot.action_features)
        self.rec: RunRecorder | None = None
        self._obs_count = 0
        self._obs_index_by_ts: dict[float, int] = {}
        self._rec_errors = 0

    def _record(self, fn, *args):
        """Recording must never take the robot down: log the first few failures and carry on."""
        if self.rec is None:
            return
        try:
            fn(*args)
        except Exception as e:
            self._rec_errors += 1
            if self._rec_errors <= 5:
                self.logger.warning(f"[viz] recording failed: {e!r}")

    # ---- observations -------------------------------------------------------------------
    def send_observation(self, obs):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        start = time.time()
        ok = super().send_observation(obs)
        self._record(self._record_obs, obs, start, time.time() - start, queue_size, ok)
        return ok

    def _record_obs(self, obs, start, send_s, queue_size, ok):
        raw = obs.get_observation()
        i = self._obs_count
        self._obs_count += 1
        self._obs_index_by_ts[round(obs.get_timestamp(), 6)] = i
        frames = []
        for cam in self.cam_keys:
            if cam in raw:
                name = f"obs{i:05d}_{cam}.jpg"
                self.rec.add_frame(name, raw[cam])
                frames.append(name)
        self.rec.add({
            "type": "obs", "i": i, "t": start, "ts": obs.get_timestamp(), "step": obs.get_timestep(),
            "must_go": bool(obs.must_go), "queue": queue_size, "send_ms": send_s * 1000, "ok": ok,
            "state": [float(raw[k]) for k in self.state_keys], "frames": frames,
        })

    # ---- action chunks ------------------------------------------------------------------
    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        received = time.time()
        with self.latest_action_lock:
            latest = self.latest_action
        with self.action_queue_lock:
            before = self.action_queue.qsize()
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)
        with self.action_queue_lock:
            after = self.action_queue.qsize()
        self._record(self._record_chunk, incoming_actions, received, latest, before, after)

    def _record_chunk(self, actions, received, latest, before, after):
        if not actions:
            return
        ts = actions[0].get_timestamp()
        self.rec.add({
            "type": "chunk", "t": received, "ts": ts, "obs_i": self._obs_index_by_ts.get(round(ts, 6)),
            "step0": actions[0].get_timestep(), "latest": latest,
            "stale": sum(a.get_timestep() <= latest for a in actions),
            "queue_before": before, "queue_after": after,
            "actions": [a.get_action().float().tolist() for a in actions],
        })

    # ---- executed actions (same as the parent, plus recording) --------------------------
    def control_loop_action(self, verbose: bool = False):
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
            queue_size = self.action_queue.qsize()

        command = self._action_tensor_to_action_dict(timed_action.get_action())
        start = time.time()
        performed = self.robot.send_action(dict(command))
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        self._record(lambda: self.rec.add({
            "type": "act", "t": start, "step": timed_action.get_timestep(), "queue": queue_size,
            "cmd": [command[k] for k in self.action_keys],
            "sent": [float(performed.get(k, command[k])) for k in self.action_keys],
        }))
        return performed

    # ---- control loop (same as the parent, plus per-iteration timing) -------------------
    def control_loop(self, task: str, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            wall_start = time.time()
            acted = sent = False
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                acted = True

            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
                sent = True

            elapsed = time.perf_counter() - control_loop_start
            self.logger.debug(f"Control loop (ms): {elapsed * 1000:.2f}")
            self._record(lambda: self.rec.add(
                {"type": "loop", "t": wall_start, "ms": elapsed * 1000, "acted": acted, "sent": sent}))
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def _pop_viz_args() -> dict:
    """Take our own flags out of sys.argv before draccus parses the robot_client config."""
    opts = {"viz_port": 8765, "runs_dir": DEFAULT_RUNS_DIR, "no_browser": False}
    rest = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--viz_port="):
            opts["viz_port"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--runs_dir="):
            opts["runs_dir"] = Path(arg.split("=", 1)[1])
        elif arg == "--no_browser":
            opts["no_browser"] = True
        else:
            rest.append(arg)
    sys.argv = rest
    return opts


VIZ = _pop_viz_args() if __name__ == "__main__" else {}


@draccus.wrap()
def main(cfg: RobotClientConfig):
    run(cfg, VizRobotClient)


def run(cfg: RobotClientConfig, client_cls: type[VizRobotClient]):
    logging.info(pformat(asdict(cfg)))
    client = client_cls(cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(VIZ["runs_dir"]) / run_id
    rec = RunRecorder(run_dir, {
        "run_id": run_id, "t0": time.time(), "task": cfg.task, "fps": cfg.fps,
        "actions_per_chunk": cfg.actions_per_chunk, "chunk_size_threshold": cfg.chunk_size_threshold,
        "aggregate_fn_name": cfg.aggregate_fn_name, "policy_type": cfg.policy_type,
        "pretrained_name_or_path": cfg.pretrained_name_or_path,
        "max_relative_target": cfg.robot.max_relative_target,
        "state_keys": client.state_keys, "action_keys": client.action_keys, "cam_keys": client.cam_keys,
    })
    client.rec = rec
    server = serve(rec, VIZ["viz_port"])
    url = f"http://127.0.0.1:{VIZ['viz_port']}/"
    client.logger.info(f"[viz] recording to {run_dir} | live viewer at {url}")
    if not VIZ["no_browser"]:
        webbrowser.open(url)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            rec.add({"type": "end", "t": time.time()})
            rec.close()
            server.shutdown()
            client.logger.info("Client stopped")
            print(f"\n[viz] run saved to {run_dir}\n[viz] replay: python client/viz_server.py \"{run_dir}\"")


if __name__ == "__main__":
    if any(a.startswith("--image_size=") for a in sys.argv[1:]):
        resize_client.install(*resize_client.pop_image_size())
    if not home_on_exit.pop_no_home():
        home_on_exit.install()
    register_third_party_plugins()
    main()
