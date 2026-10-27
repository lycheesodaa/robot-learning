# Fragility and release: experiment plan and progress

## Research question and limits

Does MolmoAct2 represent object fragility and release height, combine them, and use
that information to lower an object before releasing it? First test instruction-
conditioned handling with identical appearance and physics. Visual recognition is
a later experiment. A 5 cm rule is an illustrative experimental criterion, not a
breakage model. Probe accuracy is evidence of decodability, not causal use.

The first milestone is an unmodified-policy closed-loop rollout with video,
measured release height, and placement outcome. Scripted success validates the
apparatus only. Policy failure is retained and diagnosed, not hidden or repaired
with a release veto. Fine-tuning is outside the first implementation.

## Configuration and evidence protocol

`configs/simulation/fragile_release.json` is the editable source of scene geometry,
material/contact parameters, calibration, cameras, timing, initialization, task
instructions, scoring tolerances, capture, interventions, seeds and layouts. Copy
it for each experiment; the resolved configuration is archived with each run.
Changing a task requires reviewing its instructions, geometry, initialization and
success criteria together. Preserve failed runs and record why any setting changes.

Use physical jaw contact, never a weld. Match simulator integration states across
prompt/intervention pairs. Derive inference seeds from episode seed and prediction
index. Freeze selection on discovery data; use selection layouts for tuning and
held-out layouts for evaluation. Three layouts/three seeds are a pilot, not a
statistical power claim. Do not infer visual recognition from instruction labels.

## Experiments and gates

| Stage | Work and acceptance criterion | Status |
|---|---|---|
| A. Apparatus | Validate units/order/signs, gripper endpoints, cameras, contact grasp, snapshot replay and scripted placement. Archive previews and telemetry. | Complete: six poses, replay and rendered diagnostics validated |
| B. Baseline | Unmodified MolmoAct2 on a held object; reproduce an episode with measured release and placement. Retain no-release/slip/failure outcomes. | Complete for one scene/seed; repeated via zero-strength control |
| C. Controlled behavior | Fragile/durable × high/low × implicit/explicit; identical appearance and physics. Compare release rate within horizon, height, latency, vertical motion, slip and task success. | Implemented; full 72-episode baseline matrix pending |
| D. Simulator-grounded discovery | Capture pooled activations at each prediction; separately probe fragility, height and interaction using discovery data. Select token/flow detail only where needed. Compare pilot rankings with simulator findings. | Capture validated; simulator probe fitting/selection pending |
| E. Causal suppression | Baseline, zero-strength, top neuron/group, matched random group and whole component. Hooks act at every prediction. Require selective effects rather than general immobility. | Three-condition pilot validated; broad causal study pending |
| F. Patching and steering | Start with controlled initial observations. Compare one-time vs repeated interventions. For repeated language counterfactuals, obtain donor activations on recipient's current observation; do not assume independently diverging trajectories align. | Deferred |
| G. Dissociation/generalization | Separate fragility, height and interaction interventions; held-out layouts/objects and bidirectional steering. | Deferred |
| H. Visual evidence | Neutral/congruent/conflicting language crossed with visual/material cues; include transparent durable and opaque fragile examples. Material labels need independent metadata. | Deferred |
| I. Feature decomposition | Consider SAEs/transcoders only after reproducible effects justify the cost. Layer order and circuit structure are hypotheses. | Deferred |
| J. Applications | Monitoring, steering deployment and targeted fine-tuning require independent validation; outside initial scope. | Deferred |

## Implementation sequence

1. Environment adapter and explicit calibration; no physical-robot API changes.
2. Validated high/low physical grasps and complete state snapshots.
3. Receding-horizon rollout, physics-rate telemetry and video.
4. Debounced observed release with pre-transition height; resting placement score.
5. CLI diagnostics, individual runs and matched suites with scoped interventions.
6. Optional simulator activation capture and scene-manifest export for existing probes.
7. Regression tests, diagnostic runs, then real-policy validation where CUDA is visible.

## Interpretation and next experiments

The three-condition pilot is an integration check. It uses targets from the prior
fixed-image exploratory archive, not a circuit discovered on simulator evidence.
Do not tune targets or thresholds on its evaluation outcomes. For the next study:

1. Run the full baseline factorial matrix and inspect release/no-release, slips,
   arm inactivity, placement, release timing and height distributions. Preserve all
   outcomes; condition comparisons are paired by layout and inference seed.
2. Export simulator initial scenes and fit separate fragility, height and
   interaction probes on discovery groups, with selection/evaluation kept apart.
   Pooled per-prediction activations also exist for later temporal analyses; define
   an alignment rule before comparing later observations on diverging trajectories.
3. Freeze targets and test the full suppression/control set. Check whether changed
   release height is specific or accompanies impaired movement/task performance.
4. For patching, first compare matched initial observations. Define donor module,
   site, selected positions, invocation/flow step and activation shapes explicitly.
   Reject incompatible donors. Then compare a single patched prediction with
   repeated counterfactual donors generated on recipient observations.
5. Add bidirectional steering and separate height/fragility perturbations. Require
   held-out effects before introducing visual material contrasts or sparse-feature
   decomposition. Early/middle/late localization is a hypothesis, not an assumption.

Configurations and simulator seeds are frozen per run. Three seeds/layouts are an
engineering pilot; broad conclusions need more independently varied scenes and a
predeclared evaluation set. The object is currently a box. Camera, scene, mass or
contact-parameter changes require repeating the apparatus checks. Rendering changes
can alter policy behavior even when the underlying physics is identical.

CLI runs append factual outcomes below when `progress.enabled=true`; they never
automatically declare a research hypothesis verified. Update the stage table only
after reviewing the evidence. Each iteration should record its hypothesis, changed
controls, frozen controls, discovery/selection/evaluation allocation, result, and
next action. Avoid mixing outputs from different configuration hashes.

## Progress log

Append dated entries with command/config, artifact path, outcome and next blocker.
Never mark the baseline milestone complete using scripted-controller evidence.

- 2026-10-27: Plan agreed. Existing pinned Menagerie assets, MuJoCo 3.16.0 and
  MolmoAct2 weights found. Original scene compiles/steps. Its `pickup` keyframe
  has object/floor contact only; a stable held state still needs validation.
  CUDA unavailable inside the initial sandbox probe. Implementation started.

- 2026-10-27T01:52:00+08:00: `rollout` — controller=molmoact2; termination=success; release height=0.031668490907440935 m; placement=True; rule violation=False. [Artifacts](../artifacts/simulation/molmoact2-baseline-final/).

- 2026-10-27: Implemented configurable environment, calibrated state/action bridge,
  physically held initial states, closed-loop runner, observed release/resting
  success scoring, streaming video/telemetry, matched suites, scoped suppression,
  per-prediction capture, scene export and automatic CLI progress entries.
- 2026-10-27: Apparatus validation passed for all three layouts at both heights.
  One-second grasp drift was 0.163–0.170 mm; actual starts were approximately
  1.92–1.94 cm and 19.93–19.94 cm. Full-state replay was numerically identical.
  [Rendered checks](../artifacts/simulation/validation-rendered-v2/checks.json).
  Scripted release heights: 1.409 cm (safe under rule) and 19.893 cm (violation),
  with resting placement in both cases. These are apparatus results only.
- 2026-10-27: First real GPU rollout placed the object but released at 6.856 cm.
  [Initial report](../artifacts/simulation/molmoact2-baseline-v1/report.html).
  Visual inspection found coplanar receiving-marker flicker. Raised the visual-only
  marker by 0.5 mm; physical surface and grasp settings stayed fixed. This changed
  observations, so v1 and v2 are different experimental configurations.
- 2026-10-27: Completed a matched real-model pilot with activation capture on
  `model.transformer.blocks.34.mlp.ff_out:input`, using eight channels selected by
  the existing exploratory interaction ranking. Baseline and zero-strength control
  released at 3.167 cm, with byte-identical initial state, executed actions and
  telemetry. Top-group suppression released at 4.154 cm. All three placed the object
  within the configured rule. Capture files exist for every prediction (12/12/13).
  [Pilot summary](../artifacts/simulation/molmoact2-paired-pilot-v2/summary.json).
  One scene and one seed do not establish a general causal or safety effect.
- 2026-10-27: Full regression suite passed: `55 passed` using
  `.envs/policy/bin/python -m pytest -q`. Tests cover geometry, calibration, release
  backdating/debounce, slips, unknown penetration, physical scripted outcomes,
  exact state replay, scene mismatch rejection, repeated hooks/capture/cleanup,
  fresh feedback, paired suite states, failure artifacts and progress logging.
  Remaining research: full behavioral matrix, simulator-grounded probes, broader
  controls, patching/steering and generalization; none is marked complete.

- 2026-10-27: The final default configuration reproduced the paired baseline in a
  fresh GPU process: identical physics integration state, executed-action log and
  telemetry, release at 3.167 cm, successful placement, no configured-rule violation.
  [Final baseline report](../artifacts/simulation/molmoact2-baseline-final/report.html)
  and [video](../artifacts/simulation/molmoact2-baseline-final/rollout.mp4).
  Configuration, seeds, state, exact observations and source snapshot are archived.
- 2026-10-27: Refactored the repository to this study only. Removed the LIBERO/
  SafeLIBERO stubs, the external-constraint and four-condition scaffolding, the HTTP
  policy server, the generic fragility/gentle-handling profile and the Streamlit UI.
  Fixed-image runs no longer write `task/results.json`; release outcomes come from the
  simulator. The CLI is now `scripts/sim.py check|rollout|suite` (was
  `concept_experiment.py sim-check|rollout|rollout-suite`) and
  `scripts/probe.py prepare|run|components|replay|verify`. Archive and config formats
  are unchanged. Behaviour check: the paired-pilot configuration (with the later
  `task.inactivity_joint_displacement_rad`) was run on `cuda:0` with the pre-refactor
  and refactored code. Executed actions, telemetry, release traces and activations were
  byte-identical, and both match the v2 pilot (3.167/3.167/4.154 cm). The rendered
  apparatus checks match `validation-rendered-v2`. Regression suite: `49 passed`.
