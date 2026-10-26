# MolmoAct2 safety mechanisms

Research scaffold for the accompanying proposal, covering MolmoAct2-LIBERO and MolmoAct2-SO100_101. Compare vanilla, internal steering, external constraints, and combined safety. Dataset recommendations and training expectations are in [docs/datasets.md](docs/datasets.md).

## Quick start

From this directory:

```bash
conda activate "$PWD/.envs/policy"
# Recreate if needed:
# CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/policy" -f configs/environment-policy.yml
python scripts/download_models.py
python scripts/download_models.py --verify-only
python scripts/diagnose.py
python scripts/check_model_metadata.py
pytest -q
# GPU sample inference (run from a host terminal; a restricted sandbox hides the GPUs):
python scripts/smoke_inference.py --model libero
python scripts/smoke_inference.py --model so101
# CPU sample validation (slower; this machine has enough host RAM):
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python scripts/smoke_inference.py --model libero --device cpu
# Lower-memory mode; record this precision choice in results:
python scripts/smoke_inference.py --model libero --dtype bfloat16
```

Complete checkpoint revisions are pinned in `configs/models.json`. Snapshots include weights, tokenizer, processors, custom Python model code, normalization metadata, and official sample images. Loading executes that pinned custom code with `trust_remote_code=True`. Downloads resume through Hugging Face's local cache; `--verify-only` checks every file size, weight SHA256, shard references and normalization tags without network access. A completion manifest is written only after verification.

The two snapshots require approximately 43.6 GB. Keep additional room for environments, recordings and activations. All generated large files and environments are ignored by Git. Snapshot manifests live beside the weights. The proposal PDF is preserved.

## Concept experiments and offline replay

For **recognizing fragile objects and lowering them before release**, use the
[fragile-release study](docs/fragile-release.md). It crosses fragility, release height
and explicit handling instructions, ranks their interaction, and records release
outcomes when aligned trajectory data is available. Its default 5 cm limit is an
illustrative experiment parameter.

See [the concept workflow guide](docs/concept-experiments.md) for configurable
fragility/gentle-handling contrasts, activation capture, probes, controlled
interventions, and the optional local experiment UI. Every run saves its inputs,
activation arrays, outputs, provenance and ordered events under `artifacts/concepts/`.
The [experiment plan and progress log](docs/fragile-release-experiment-plan.md) tracks the closed-loop study; its controls live in [simulation JSON](configs/simulation/fragile_release.json).
Open a run's `replay.html` directly in a browser to inspect it without running the
model or a web server. Raw token-level capture is optional; all captured channel
means and flow invocations are saved by default. Synthetic verification is distinct
from real-checkpoint concept evidence.

## Implemented and scaffolded components

| Component | Status |
| --- | --- |
| Immutable snapshot download and integrity verification | Implemented |
| Local-only official sample inference / recorded-observation predictions | Implemented; both CPU samples verified, SO-101 GPU sample verified on 2026-10-19 |
| Action/observation contracts and local HTTP inference server | Implemented |
| Scoped capture, ablation, scaling, steering, activation patching | Implemented for explicitly selected tensor-output modules |
| Rollout records, safety/task metrics, split-leakage checks | Implemented |
| Four experiment configurations | Provided; not a completed experimental runner |
| LIBERO/SafeLIBERO rollout adapters | Explicit research stubs |
| Workspace/obstacle/forbidden-region geometric filter | Explicit research stub; requires calibrated dynamics and kinematics |
| SO-101 physical actuation | Explicit stub; recorded-input inference only |
| Concept contrasts, pooled activation ranking, probes and offline workflow replay | Implemented; real-checkpoint concept findings still require experiments |
| SO-101 MuJoCo held-object rollouts | Implemented: physical grasps, video, measured release, paired suppression and optional activations; see [simulation guide](docs/fragile-release.md#closed-loop-so-101-mujoco-experiments) |
| Broad simulator/generalization study and physical safety evaluation | Research work still to complete |

Geometric filtering never silently passes actions when unavailable. `checked_filter` raises `ConstraintFailure`; a future runner must stop execution on that exception. An initial geometric baseline is not a certified CBF controller or a reproduction of AEGIS.

## Research workflow

1. Check the LIBERO sample on GPU (`python scripts/smoke_inference.py --model libero`); only its CPU sample has been validated.
2. Integrate one seeded ordinary LIBERO rollout with 50 settling steps; verify image orientation, state order, gripper convention and action/controller scaling against the pinned upstream recipe.
3. Integrate SafeLIBERO in its own environment, then collect controlled safe/unsafe contrasts. See [the data protocol](docs/datasets.md#data-construction-protocol).
4. Stream selected activations to disk; fit probes/directions using only the discovery split. Modules should be enumerated from the loaded checkpoint, not guessed. Run `python scripts/smoke_inference.py --model libero --list-modules`.
5. Tune interventions on selection tasks, then evaluate held-out tasks/layouts under the four conditions. Log policy proposals separately from filtered execution.
6. Integrate SO-101 calibration and recording before any physical execution. The checkpoint's native joint values are not assumed to be the units accepted by your motor controller.

Hooks run eagerly: compilation and CUDA graphs are disabled. Action-expert modules can fire multiple times during flow integration; the capture sink must retain invocation order. Capturing everything can exhaust memory. Safety-layer localization is a hypothesis, not a property assumed by this code.

## Separate simulator processes

See [docs/setup.md](docs/setup.md) for pinned upstream environments, GPU diagnostics, and the server transport. Direct inference does not require installing LeRobot or a simulator. Use the policy server to avoid forcing old simulator dependencies into the model environment.

The SO-101 MuJoCo sim is the exception: it runs in `.envs/policy` (`pip install -e .[sim]`), so intervention hooks see every forward call. The robot model is MuJoCo Menagerie's `robotstudio_so101`, pinned to commit `f3b6013`:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git artifacts/sim/mujoco_menagerie
git -C artifacts/sim/mujoco_menagerie sparse-checkout set robotstudio_so101
git -C artifacts/sim/mujoco_menagerie checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
```

The server has no display. Render offscreen with `MUJOCO_GL=egl` (`MUJOCO_EGL_DEVICE_ID` picks the GPU). Call `Renderer.close()` before exit to avoid a harmless EGL error at teardown.

## Validation history

**2026-10-05**

- Both full snapshots (about 43.6 GB total) assembled and SHA256-verified. Inventories and completion manifests are stored beside each snapshot in `artifacts/models/`.
- Official sample inference succeeded offline on CPU in FP32: LIBERO returned finite `[10, 7]` actions; SO-101 returned finite `[30, 6]` actions. Outputs and logs are in `artifacts/*-cpu-smoke.*`.
- The 11 unit tests that existed then all passed. `pip check` found no broken requirements; source compilation passed. `artifacts/validation.json` summarizes these results.
- The policy environment is installed. LIBERO/SafeLIBERO have separate Python foundations and pinned source checkouts; their simulator dependencies and rollout adapters are still integration work. No closed-loop safety results or physical-robot validation are claimed.

**2026-10-19**

- Host GPUs confirmed working. Diagnostics are in `artifacts/diagnostics-host-2026-10-19.json`.
- SO-101 sample inference in FP32 on `cuda:0` returned finite `[30, 6]` actions: 11.13 s model load, 1.06 s first inference without warm-up, 21.81 GiB peak allocated GPU memory. These are one-run measurements, not a latency benchmark.
