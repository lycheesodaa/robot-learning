"""Fragile-release shield for the so101-nexus MolmoAct2 checkpoint on a real SO-101.

Ported from the simulation study in robot_learning_rulebased (molmoact2-so101-sim/safety/):
the refuse_if rule language and executor (rule_dsl.py), the real-arm feature vocabularies
(vocab.py: v2_engineered_encoder, v4_raw_full) and the sim conversions (eval/eval_policy.py).
Pure numpy, so the server and the robot client run the same code. This is a copy of
server/fragile_shield.py; keep the two identical.

Three frames:
  real  LeRobot so101_follower, use_degrees=True: arm joints in degrees, gripper 0-100
        (0 = the calibration's closed end). What the client reads and sends.
  ckpt  what the nexus checkpoint was trained on: arm = JOINT_SIGNS * real + JOINT_OFFSETS
        (the same correction the stock -LeRobot checkpoint bakes into its processor; the
        nexus checkpoint has none, so the server applies it), gripper 0 (closed) .. 45 (open).
  sim   so101-nexus joint radians, gripper -10 deg (closed) .. +100 deg (open). The learned
        rules use sim units, so features are computed in this frame. The nexus adapter
        (so101_nexus/lerobot_adapter/normalization.py) maps LeRobot degrees to sim radians
        1:1 and the gripper's 0-100 linearly onto -10..+100 deg.

Gripper scaling is the one calibration that is not a pure unit change: the real gripper's
0-100 depends on the range recorded by lerobot-calibrate. `Calibration.gripper_closed/open`
are the real readings at the sim's closed and open limits (default 0 and 100, the nexus
adapter's assumption); measure them with client/calibrate_shield.py.

Threshold recalibration: features are put into sim-equivalent units so that sim-learned
thresholds keep their meaning:
  tcp_z      FK height, shifted so that the measured real "object held, resting on the
             table" TCP height maps to the sim's (cube 1.5 cm + grasp offset 1.45 cm).
             A sim threshold of 0.049 m therefore stays "about 2 cm above set-down height".
  load_*     servo Present_Load (signed, 0.1 % of stall torque) * stall torque 2.94 N m
             (the menagerie STS3215 forcerange), signed like the sim actuator force:
             negative = pushing towards smaller angles (gripper: squeezing closed).
Thresholds that still do not transfer (e.g. the real squeeze load of a soft object) are
overridden per feature with `Calibration.thresholds`.
"""
import json
import math
import operator
from dataclasses import asdict, dataclass, field

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
VOCAB_JOINTS = ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper")  # names the rules use

JOINT_SIGNS = np.array([1., -1., 1., 1., 1., 1.])
JOINT_OFFSETS = np.array([0., 90., 90., 0., 0., 0.])
CKPT_GRIPPER_OPEN = 45.0
SIM_GRIPPER_DEG = (-10.0, 100.0)  # closed, open
STALL_TORQUE_NM = 2.94            # menagerie sts3215 forcerange; Present_Load 1000 = 100 %
LOAD_FULL_SCALE = 1000.0
SIM_TCP_REST_Z_M = 0.015 + 0.0145  # sim cube half size + TCP-above-cube-centre while grasped

# Sim feature constants (safety/features.py, safety/vocab.py)
OPEN_EPS_RAD = math.radians(5.0)
ENC_CLOSED_CMD_RAD = 0.0
ENC_BLOCKED_RAD = (math.radians(-8.5), math.radians(20.0))
ENC_SETTLED_RAD = math.radians(3.0)


# --------------------------------------------------------------------------- calibration


@dataclass
class Calibration:
    """Real-arm measurements the shield needs. Defaults are the sim's assumptions."""

    gripper_closed: float = 0.0      # real gripper reading at the sim's closed limit (-10 deg)
    gripper_open: float = 100.0      # real gripper reading at the sim's open limit (+100 deg)
    load_signs: list = field(default_factory=lambda: [1.0] * 6)  # makes Present_Load signed like sim force
    tcp_rest_z_m: float = SIM_TCP_REST_Z_M  # real FK TCP z with the object held, resting on the table
    thresholds: dict = field(default_factory=dict)  # feature -> real-arm threshold replacing the rule's
    measured: bool = False           # set by calibrate_shield.py; hold/abort refuse to run without it
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- frame conversion


def real_gripper_to_ckpt(g, cal):
    return (g - cal.gripper_closed) / (cal.gripper_open - cal.gripper_closed) * CKPT_GRIPPER_OPEN


def ckpt_gripper_to_real(g, cal):
    return cal.gripper_closed + g / CKPT_GRIPPER_OPEN * (cal.gripper_open - cal.gripper_closed)


def real_to_ckpt(q, cal):
    """(..., 6) real-frame joints -> checkpoint frame."""
    q = np.asarray(q, dtype=np.float64)
    out = JOINT_SIGNS * q + JOINT_OFFSETS
    out[..., 5] = real_gripper_to_ckpt(q[..., 5], cal)
    return out


def ckpt_to_real(a, cal):
    """(..., 6) checkpoint-frame actions -> real frame (exact inverse of real_to_ckpt)."""
    a = np.asarray(a, dtype=np.float64)
    out = (a - JOINT_OFFSETS) / JOINT_SIGNS
    out[..., 5] = ckpt_gripper_to_real(a[..., 5], cal)
    return out


def real_to_sim_rad(q, cal):
    """(..., 6) real-frame joints -> sim joint radians."""
    q = np.asarray(q, dtype=np.float64)
    deg = q.copy()
    lo, hi = SIM_GRIPPER_DEG
    deg[..., 5] = lo + real_gripper_to_ckpt(q[..., 5], cal) / CKPT_GRIPPER_OPEN * (hi - lo)
    return np.deg2rad(deg)


# --------------------------------------------------------------------------- forward kinematics

# so101_nexus/assets/SO101_menagerie/so101.xml (the MJCF the sim runs): body pos, quat (w x y z)
# of shoulder, upper_arm, lower_arm, wrist, gripper; each body's hinge turns about its local z.
_CHAIN = (
    ((0.0388353, 0.0, 0.0624), (0.0, 0.0, -1.0, 0.0)),
    ((-0.0303992, -0.0182778, -0.0542), (1.0, -1.0, -1.0, -1.0)),
    ((-0.11257, -0.028, 0.0), (1.0, 0.0, 0.0, 1.0)),
    ((-0.1349, 0.0052, 0.0), (1.0, 0.0, 0.0, -1.0)),
    ((5.55112e-17, -0.0611, 0.0181), (0.0172091, -0.0172091, 0.706897, 0.706897)),
)
_TCP_SITE = np.array([0.012, -0.000218, -0.098127])  # "gripperframe", the sim's TCP


def _quat_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


_CHAIN_MATS = tuple((np.array(p), _quat_mat(*q)) for p, q in _CHAIN)


def tcp_position(q_sim_rad):
    """Sim-frame TCP position (m, robot base frame, z up) for the first five joint angles."""
    R, p = np.eye(3), np.zeros(3)
    for (pos, rot), angle in zip(_CHAIN_MATS, q_sim_rad[:5]):
        p = p + R @ pos
        c, s = math.cos(angle), math.sin(angle)
        R = R @ rot @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return p + R @ _TCP_SITE


# --------------------------------------------------------------------------- features


def features(q_real, target_real, fragile, cal, load_raw=None, q_prev_real=None, dt=None,
             prevcmd_real=None, prev_target_real=None):
    """Rule inputs for one proposed action, in sim units (the v2_engineered_encoder and
    v4_raw_full vocabularies merged).

    q_real        joint state before the action (real frame)
    target_real   the proposed action (real frame)
    load_raw      signed Present_Load per motor (-1000..1000), or None if not read
    q_prev_real   joint state one control step earlier, with `dt` the seconds between them;
                  gives joint and TCP velocities
    prevcmd_real  the command sent at the previous step (real frame)
    Features that cannot be computed are left out, so a rule that needs them refuses
    (fail-closed, reason unknown_feature:<name>).
    """
    q = real_to_sim_rad(q_real, cal)
    tgt = real_to_sim_rad(target_real, cal)
    tcp = tcp_position(q)
    tcp_z = float(tcp[2] - cal.tcp_rest_z_m + SIM_TCP_REST_Z_M)
    f = {"fragile": bool(fragile)}
    for i, j in enumerate(VOCAB_JOINTS):
        f[f"q_{j}"] = float(q[i])
        f[f"dtarget_{j}"] = float(tgt[i] - q[i])
    f.update(tcp_x=float(tcp[0]), tcp_y=float(tcp[1]), tcp_z=tcp_z)

    delta = float(tgt[5] - q[5])
    f.update(opening=delta > OPEN_EPS_RAD, gripper_delta_rad=delta,
             height_above_support_m=tcp_z - SIM_TCP_REST_Z_M)

    if q_prev_real is not None and dt:
        qp = real_to_sim_rad(q_prev_real, cal)
        v = (tcp - tcp_position(qp)) / dt
        for i, j in enumerate(VOCAB_JOINTS):
            f[f"dq_{j}"] = float((q[i] - qp[i]) / dt)
        f.update(tcp_vx=float(v[0]), tcp_vy=float(v[1]), tcp_vz=float(v[2]),
                 tcp_speed_mps=float(np.linalg.norm(v)), tcp_vz_mps=float(v[2]))

    if load_raw is not None:
        for i, j in enumerate(VOCAB_JOINTS):
            f[f"load_{j}"] = float(cal.load_signs[i] * load_raw[i] / LOAD_FULL_SCALE * STALL_TORQUE_NM)

    if prevcmd_real is not None:
        cmd = real_to_sim_rad(prevcmd_real, cal)
        for i, j in enumerate(VOCAB_JOINTS):
            f[f"prevcmd_{j}"] = float(cmd[i])
        if q_prev_real is not None:  # encoder-only grasp estimate (vocab.holding_from_encoders)
            f["holding"] = bool(cmd[5] < ENC_CLOSED_CMD_RAD
                                and ENC_BLOCKED_RAD[0] < q[5] < ENC_BLOCKED_RAD[1]
                                and abs(q[5] - real_to_sim_rad(q_prev_real, cal)[5]) < ENC_SETTLED_RAD)
    return f


# --------------------------------------------------------------------------- rule language

FEATURE_TYPES = {"fragile": bool, "holding": bool, "opening": bool,
                 "height_above_support_m": float, "gripper_delta_rad": float,
                 "tcp_speed_mps": float, "tcp_vz_mps": float}
for _j in VOCAB_JOINTS:
    for _p in ("q", "dq", "dtarget", "load", "prevcmd"):
        FEATURE_TYPES[f"{_p}_{_j}"] = float
for _a in "xyz":
    FEATURE_TYPES[f"tcp_{_a}"] = FEATURE_TYPES[f"tcp_v{_a}"] = float

OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le,
       ">": operator.gt, ">=": operator.ge}


class RuleError(ValueError):
    pass


class _MissingFeature(Exception):
    pass


def validate_rule(rule):
    if not isinstance(rule, dict) or set(rule) != {"version", "effect", "condition"}:
        raise RuleError("rule must have exactly the keys version, effect, condition")
    if rule["version"] != 1 or rule["effect"] != "refuse_if":
        raise RuleError("only version 1 / effect 'refuse_if' is supported")
    _validate_node(rule["condition"])


def _validate_node(node):
    keys = set(node) if isinstance(node, dict) else None
    if keys in ({"all"}, {"any"}):
        kids = node["all" if "all" in keys else "any"]
        if not isinstance(kids, list) or not kids:
            raise RuleError("all/any needs a non-empty list")
        for k in kids:
            _validate_node(k)
    elif keys == {"not"}:
        _validate_node(node["not"])
    elif keys == {"const"}:
        if not isinstance(node["const"], bool):
            raise RuleError("const must be true or false")
    elif keys == {"feature", "op", "value"}:
        f, op, v = node["feature"], node["op"], node["value"]
        if f not in FEATURE_TYPES:
            raise RuleError(f"unknown feature {f!r}")
        if op not in OPS:
            raise RuleError(f"unknown operator {op!r}")
        if (FEATURE_TYPES[f] is bool) != isinstance(v, bool):
            raise RuleError(f"{f} compared with a value of the wrong type")
    else:
        raise RuleError(f"unrecognised node {node!r}")


def load_rule(path):
    with open(path) as f:
        rule = json.load(f)
    validate_rule(rule)
    return rule


def with_thresholds(rule, thresholds):
    """Copy of `rule` with each numeric atom on a feature in `thresholds` given that value."""
    def walk(n):
        if "all" in n or "any" in n:
            key = "all" if "all" in n else "any"
            return {key: [walk(k) for k in n[key]]}
        if "not" in n:
            return {"not": walk(n["not"])}
        if "feature" in n and n["feature"] in thresholds and not isinstance(n["value"], bool):
            return dict(n, value=float(thresholds[n["feature"]]))
        return dict(n)
    out = dict(rule, condition=walk(rule["condition"]))
    validate_rule(out)
    return out


def _eval(node, feats):
    if "all" in node:
        return all(_eval(k, feats) for k in node["all"])
    if "any" in node:
        return any(_eval(k, feats) for k in node["any"])
    if "not" in node:
        return not _eval(node["not"], feats)
    if "const" in node:
        return node["const"]
    x = feats.get(node["feature"])
    if x is None or (isinstance(node["value"], bool) != isinstance(x, bool)) or (
            not isinstance(x, bool) and not math.isfinite(x)):
        raise _MissingFeature(node["feature"])
    return OPS[node["op"]](x, node["value"])


def decide(rule, feats):
    """(refuse, reason): reason is 'rule', 'allowed' or 'unknown_feature:<name>' (fail-closed)."""
    try:
        return (True, "rule") if _eval(rule["condition"], feats) else (False, "allowed")
    except _MissingFeature as e:
        return True, f"unknown_feature:{e.args[0]}"


def rule_features(rule):
    out = set()

    def walk(n):
        for k in n.get("all", n.get("any", [])):
            walk(k)
        if "not" in n:
            walk(n["not"])
        if "feature" in n:
            out.add(n["feature"])
    walk(rule["condition"])
    return out


def to_text(rule):
    def fmt(n):
        if "all" in n or "any" in n:
            parts = [fmt(k) for k in n.get("all", n.get("any"))]
            s = (" AND " if "all" in n else " OR ").join(parts)
            return s if len(parts) == 1 else f"({s})"
        if "not" in n:
            return f"NOT {fmt(n['not'])}"
        if "const" in n:
            return "ALWAYS" if n["const"] else "NEVER"
        f, op, v = n["feature"], n["op"], n["value"]
        if isinstance(v, bool):
            return f if (op == "==") == v else f"NOT {f}"
        return f"{f} {op} {v:g}"
    return "REFUSE if " + fmt(rule["condition"]).strip("()")


# --------------------------------------------------------------------------- shield config

MODES = ("off", "shadow", "hold", "abort")


def shield_config(rule, mode, fragile, cal):
    """What the server attaches to every action it sends, so the client checks the very
    same rule (thresholds already recalibrated) with the same calibration."""
    return {"rule": rule, "mode": mode, "fragile": bool(fragile), "calibration": cal.to_dict()}
