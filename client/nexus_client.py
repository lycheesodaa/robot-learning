"""Robot client for server/serve_molmoact2_nexus.py: the so101-nexus MolmoAct2
fine-tune with the fragile-release shield learned in simulation.

Drop-in replacement for viz_client.py, with the same flags, recording and viewer. Run it
through the launchers with NEXUS=1 (bash) or -Nexus (PowerShell), with config
so101_molmoact2_nexus.yaml. Compared with viz_client.py:

  * Every observation carries the servo loads as '<motor>.load' (signed Present_Load,
    -1000..1000). The server's shield reads them; the policy never sees them.
  * The shield runs again on every queued action, right before it is sent to the arm. It uses
    the rule, mode and calibration the server attached to the chunk, with joint positions and
    loads read from the bus at that moment. The server checks the state it predicted for the
    chunk; this check uses the arm's actual state, so it also catches queued actions that went
    stale while the chunk travelled.
      shadow  record the decision only
      hold    a refused action keeps the previous gripper command; the arm joints still move
      abort   the first refusal latches
  * Latch: after a refusal here in abort mode, or when the server sends an empty chunk (its
    shield latched, so no further actions will arrive), the queue is emptied. Nothing more is
    sent to the arm, chunks still in flight are dropped, and no more observations are sent.
    The arm holds its last command, so the gripper keeps squeezing. Ctrl-C exits. While a
    fragile shield is active, homing leaves the gripper at its last command, so the arm does
    not drop the object on exit.
  * While the shield is in hold or abort mode, a new chunk replaces the queued actions
    (latest_only), so a held chunk is never averaged with an earlier opening.
"""

import sys
import threading
import time
from contextlib import nullcontext
from queue import Queue

import draccus
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.utils.import_utils import register_third_party_plugins

import fragile_shield as S
import home_on_exit
import resize_client
import viz_client

MAX_VELOCITY_DT = 1.0  # s; an older previous reading gives no velocity


class NexusRobotClient(viz_client.VizRobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        self.motors = list(self.robot.bus.motors)
        if self.motors != list(S.JOINTS) or self.action_keys != [f"{m}.pos" for m in S.JOINTS]:
            raise RuntimeError(f"nexus_client expects an SO-101 with joints {S.JOINTS}, got {self.action_keys}")
        self.shield: dict | None = None  # config the server attaches to every action
        self.rule = self.cal = None
        self.rule_feats: list[str] = []
        self.latched: str | None = None  # why the shield stopped the run
        self.last_cmd: np.ndarray | None = None  # last command actually written (real frame)
        self.prev_pos: tuple[float, np.ndarray] | None = None
        self._refusing = False
        self._latch_lock = threading.Lock()

        get_observation = self.robot.get_observation

        def get_observation_with_loads():
            obs = get_observation()
            obs.update({f"{m}.load": v for m, v in self._read("Present_Load").items()})
            return obs

        self.robot.get_observation = get_observation_with_loads

    def _read(self, register: str) -> dict[str, float]:
        guard = getattr(self.robot, "homing_guard", None)  # home_on_exit's bus lock
        with guard.lock if guard is not None else nullcontext():
            values = self.robot.bus.sync_read(register, num_retry=getattr(self.robot.config, "num_read_retries", 0))
        return {m: float(values[m]) for m in self.motors}

    # ---- shield config, sent by the server with every chunk -----------------------------
    def _set_shield(self, config: dict):
        if config == self.shield:
            return
        S.validate_rule(config["rule"])
        self.shield, self.rule = config, config["rule"]
        self.cal = S.Calibration(**config["calibration"])
        self.rule_feats = sorted(S.rule_features(self.rule))
        guard = getattr(self.robot, "homing_guard", None)
        if guard is not None and config["fragile"] and config["mode"] in ("hold", "abort"):
            guard.hold_keys.add("gripper.pos")
        self.logger.info(f"[shield] {config['mode']} | fragile: {config['fragile']} | {S.to_text(self.rule)}")
        self._record(lambda: self.rec.add({"type": "shield", "t": time.time(), "config": config}))

    def _latch(self, reason: str):
        with self._latch_lock:
            with self.action_queue_lock:
                self.action_queue = Queue()
            if self.latched is not None:
                return
            self.latched = reason
        guard = getattr(self.robot, "homing_guard", None)
        if guard is not None:
            guard.hold_keys.add("gripper.pos")
        self.logger.warning(
            f"[shield] LATCHED: {reason}. Queue emptied; nothing more is sent. The arm holds its last "
            "command and the gripper keeps its grip. Ctrl-C to exit (homing keeps the gripper as it is).")
        self._record(lambda: self.rec.add({"type": "latch", "t": time.time(), "reason": reason}))

    # ---- action chunks ------------------------------------------------------------------
    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        if self.latched is not None:
            return  # a chunk that was in flight when the shield latched
        if not incoming_actions:
            self._latch("the server sent an empty chunk: its shield latched and no further actions will arrive")
            return
        shield = getattr(incoming_actions[0], "shield", None)
        if shield is not None:
            self._set_shield(shield["config"])
        if self.shield is not None and self.shield["mode"] in ("hold", "abort"):
            aggregate_fn = lambda old, new: new  # noqa: E731  latest_only
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def actions_available(self):
        return self.latched is None and super().actions_available()

    def _ready_to_send_observation(self):
        return self.latched is None and super()._ready_to_send_observation()

    # ---- the per-action check -----------------------------------------------------------
    def _check(self, step: int, target: np.ndarray) -> dict | None:
        """Decide on `target` with live readings. Hold mode edits target's gripper in place;
        abort mode latches."""
        if self.shield is None or self.shield["mode"] == "off":
            return None
        pos, load = self._read("Present_Position"), self._read("Present_Load")
        q = np.array([pos[m] for m in self.motors])
        now = time.perf_counter()
        q_prev = dt = None
        if self.prev_pos is not None and now - self.prev_pos[0] <= MAX_VELOCITY_DT:
            q_prev, dt = self.prev_pos[1], now - self.prev_pos[0]
        self.prev_pos = (now, q)

        feats = S.features(q, target, self.shield["fragile"], self.cal,
                           load_raw=np.array([load[m] for m in self.motors]),
                           q_prev_real=q_prev, dt=dt, prevcmd_real=self.last_cmd)
        refuse, reason = S.decide(self.rule, feats)
        mode = self.shield["mode"]
        out = {"mode": mode, "refused": refuse, "reason": reason,
               "features": {f: feats.get(f) for f in self.rule_feats}}
        if refuse != self._refusing:
            self._refusing = refuse
            values = ", ".join(f"{f} {v:.3f}" if isinstance(v, float) else f"{f} {v}" for f, v in out["features"].items())
            self.logger.warning(f"[shield] action #{step}: {'REFUSE' if refuse else 'allow again'} "
                                f"({reason}, {mode}) | {values}")
        if refuse and mode == "hold":
            out["gripper_proposed"] = float(target[5])
            target[5] = self.last_cmd[5] if self.last_cmd is not None else q[5]
        elif refuse and mode == "abort":
            self._latch(f"action #{step} refused ({reason})")
        return out

    def control_loop_action(self, verbose: bool = False):
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
            queue_size = self.action_queue.qsize()

        step = timed_action.get_timestep()
        target = timed_action.get_action().double().numpy().copy()
        decision = self._check(step, target)
        if self.latched is not None:  # refused here, or the latch fired while this action was popped
            self._record(lambda: self.rec.add({"type": "blocked", "t": time.time(), "step": step,
                                               "cmd": target.tolist(), "shield": decision}))
            return None

        command = {k: float(v) for k, v in zip(self.action_keys, target)}
        start = time.time()
        performed = self.robot.send_action(dict(command))
        self.last_cmd = np.array([float(performed.get(k, command[k])) for k in self.action_keys])
        with self.latest_action_lock:
            self.latest_action = step

        self._record(lambda: self.rec.add({
            "type": "act", "t": start, "step": step, "queue": queue_size,
            "cmd": [command[k] for k in self.action_keys], "sent": self.last_cmd.tolist(), "shield": decision,
        }))
        return performed


@draccus.wrap()
def main(cfg: RobotClientConfig):
    viz_client.run(cfg, NexusRobotClient)


if __name__ == "__main__":
    viz_client.VIZ.update(viz_client._pop_viz_args())
    viz_client.VIZ["image_size"] = resize_client.pop_image_size(default=None)
    if viz_client.VIZ["image_size"]:
        resize_client.install(*viz_client.VIZ["image_size"])
    if not home_on_exit.pop_no_home():
        home_on_exit.install()
    register_third_party_plugins()
    main()
