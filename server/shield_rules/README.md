# Shield rules

`refuse_if` rules learned in simulation by the fragile-release study (`robot_learning_rulebased`,
commit f8736b3, `molmoact2-so101-sim/artifacts/`). They use sim units: radians, metres, and
actuator force in N m. `fragile_shield.features` converts real-arm readings into these units. Real-arm
thresholds that differ go in the calibration file (`thresholds`), not here.

| file | rule | source | sim result |
|---|---|---|---|
| `v4_symbolic_k3.json` (default) | `fragile AND tcp_z > 0.04924 AND load_gripper <= -0.09125` | `rules_actions/v4_raw_full/symbolic_final.json` | best on feedback (score 0.971). Online, seeds 9000–9999: commanded breaks 20.6% → 0%, but slip-only breaks 7.1% → 15.6%; task success 5.7% → 13.5% |
| `v4_llm_propsample.json` | `fragile AND load_gripper < -0.5 AND q_gripper < 0.5` | `rules_actions/v4_raw_full/llm/x-test-2_propsample/final.json` | commanded breaks 2.8%, slip-only 9.2% |
| `v4_llm_balanced.json` | `fragile AND dtarget_gripper > 0.5 AND q_gripper < 0.5` | `rules_actions/v4_raw_full/llm/x-test-2/final.json` | 0 breaks, but also 0% task success: it blocks every release, including set-down on the table |
| `v2_symbolic.json` | `fragile AND gripper_delta_rad > 1.548` | `rules/v2_engineered_encoder/symbolic_final.json` | event-level rule from the earlier v2 vocabulary |
| `baseline_any_opening.json` | `dtarget_gripper > 0.0873` | report §3.5 baseline | refuses every opening of more than 5° |

Only the v4 symbolic rule has a height condition. The others block safe set-downs too.

Notes on the real-arm inputs (see `fragile_shield.py`):

- **`tcp_z`** is the forward-kinematics height, shifted so that the measured real "object held,
  resting on the table" height equals the sim's 0.0295 m. The threshold 0.04924 therefore means
  about 2 cm above set-down height.
- **`load_*`** is Present_Load × 2.94 N m / 1000, with its sign flipped where needed so that
  negative means squeezing towards closed. The sim threshold of −0.09 N m is only about 3% of
  stall torque. On a real gripper, and especially with a soft object, the threshold must come
  from `client/calibrate_shield.py`.
- **`holding`** is the encoder estimate from v2. The contact-based v1 signal does not exist on
  hardware.
- Velocities (`dq_*`, `tcp_v*`) are finite differences. On the server they come from two
  observations, so they are coarse. A feature that cannot be computed makes the rule refuse
  (fail-closed).
