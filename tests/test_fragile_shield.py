"""server/fragile_shield.py: frames match the rulebased bridge, FK matches the sim's MJCF, the ported
rule executor matches the sim's. No robot or weights."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'server'))
import fragile_shield as S  # noqa: E402

RULEBASED = ROOT.parent/'robot_learning_rulebased/molmoact2-so101-sim'
JS = np.array([1., -1., 1., 1., 1., 1.]); JO = np.array([0., 90., 90., 0., 0., 0.])


def test_default_frames_match_eval_policy():
    # Real degrees equal sim degrees; the real gripper's 0-100 spans the sim's -10..+100 deg.
    sim_deg = np.array([12., -30., 45., 20., -60., 40.])
    real = sim_deg.copy(); real[5] = (sim_deg[5] + 10.)/110.*100.
    expected = JS*sim_deg + JO; expected[5] = (sim_deg[5] + 10.)/110.*45.   # obs_to_batch / g2model
    cal = S.Calibration()
    np.testing.assert_allclose(S.real_to_ckpt(real, cal), expected, atol=1e-12)
    np.testing.assert_allclose(S.real_to_sim_rad(real, cal), np.deg2rad(sim_deg), atol=1e-12)


def test_ckpt_round_trip_with_measured_gripper():
    cal = S.Calibration(gripper_closed=4., gripper_open=91.)
    x = np.random.default_rng(0).uniform(-90, 90, (50, 6)); x[:, 5] = np.linspace(4, 91, 50)
    np.testing.assert_allclose(S.ckpt_to_real(S.real_to_ckpt(x, cal), cal), x, atol=1e-9)
    assert S.real_to_ckpt([0, 0, 0, 0, 0, 4.], cal)[5] == pytest.approx(0.)
    assert S.real_to_ckpt([0, 0, 0, 0, 0, 91.], cal)[5] == pytest.approx(S.CKPT_GRIPPER_OPEN)


def test_fk_matches_sim_mjcf():
    mujoco = pytest.importorskip('mujoco')
    so101_nexus = pytest.importorskip('so101_nexus')
    m = mujoco.MjModel.from_xml_path(str(so101_nexus.get_so101_mujoco_model_path()))
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in S.JOINTS]
    for q in np.random.default_rng(1).uniform(-1.6, 1.6, (50, 6)):
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        np.testing.assert_allclose(S.tcp_position(q), d.site_xpos[site], atol=1e-9)


def test_tcp_z_shift_and_load_units():
    q = np.array([0., -40., 80., 30., 0., 20.])
    base = S.features(q, q, True, S.Calibration(), load_raw=[0, 0, 0, 0, 0, -100])
    shifted = S.features(q, q, True, S.Calibration(tcp_rest_z_m=S.SIM_TCP_REST_Z_M + 0.01, load_signs=[1]*5 + [-1]),
                         load_raw=[0, 0, 0, 0, 0, -100])
    assert shifted['tcp_z'] == pytest.approx(base['tcp_z'] - 0.01)
    assert base['load_gripper'] == pytest.approx(-0.1*S.STALL_TORQUE_NM)
    assert shifted['load_gripper'] == pytest.approx(0.1*S.STALL_TORQUE_NM)


def test_bundled_rules_and_thresholds():
    rules = {Path(p).stem: S.load_rule(p) for p in glob.glob(str(ROOT/'server/shield_rules/*.json'))}
    assert 'v4_symbolic_k3' in rules
    rule = S.with_thresholds(rules['v4_symbolic_k3'], {'load_gripper': -0.5})
    assert S.decide(rule, {'fragile': True, 'tcp_z': 0.1, 'load_gripper': -0.3}) == (False, 'allowed')
    assert S.decide(rule, {'fragile': True, 'tcp_z': 0.1, 'load_gripper': -0.6}) == (True, 'rule')
    assert S.decide(rule, {'fragile': True, 'tcp_z': 0.1}) == (True, 'unknown_feature:load_gripper')


@pytest.mark.skipif(not RULEBASED.exists(), reason='robot_learning_rulebased not checked out next to this repo')
def test_decisions_match_rulebased_executor():
    sys.path.insert(0, str(RULEBASED))
    from safety import rule_dsl
    rows = [json.loads(line) for f in ('train', 'feedback', 'test')
            for line in open(RULEBASED/f'artifacts/events/v4_raw_full/{f}.jsonl')]
    for p in glob.glob(str(ROOT/'server/shield_rules/*.json')):
        rule = S.load_rule(p)
        assert all(S.decide(rule, r['features']) == rule_dsl.decide(rule, r['features']) for r in rows), p
