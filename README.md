# MolmoAct2 fragile-release study

Does MolmoAct2 (SO-101 checkpoint) represent object fragility and release height,
combine them, and use that information to lower an object before releasing it? This
repository holds the closed-loop SO-101 MuJoCo apparatus, the fixed-image probe
pipeline and the intervention hooks used to answer that question.

- [Experiment plan and progress log](docs/fragile-release-experiment-plan.md): stages, gates and dated results.
- [Fragile-release guide](docs/fragile-release.md): simulator and probe workflows, configuration, artifacts and scoring.
- [Setup](docs/setup.md): environment, GPU diagnosis and checkpoint download.
- [Remote SO-101 inference](docs/so101_remote_inference.md): LeRobot server for the physical arm (separate from this study).

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
python scripts/smoke_inference.py
```

Then validate the simulator and run a baseline episode:

```bash
python scripts/sim.py check --output artifacts/simulation/my-check
python scripts/sim.py rollout --device cuda:0 --output artifacts/simulation/my-baseline
```

The checkpoint revision is pinned in `configs/models.json`. The snapshot includes
weights, tokenizer, processors, custom Python model code, normalization metadata and
official sample images. Loading it executes that pinned custom code with
`trust_remote_code=True`. `--verify-only` checks every file size, weight SHA256,
shard reference and normalization tag without network access. A completion manifest
is written only after verification. All generated large files and environments are
ignored by Git.

## Layout

| Path | Contents |
| --- | --- |
| `src/vla_safety/policy.py` | `MolmoAct2Policy`, cached `load_policy`, action encoding |
| `src/vla_safety/hooks.py` | Scoped capture/ablate/scale/steer/patch hooks on explicit modules |
| `src/vla_safety/capture.py` | Supported capture sites and bounded per-invocation activation capture |
| `src/vla_safety/archive.py` | Append-only run archives, checksums and provenance |
| `src/vla_safety/release.py` | Release settings, factorial prompt bank, manifest validation and release-trace scoring |
| `src/vla_safety/simulation/` | MuJoCo environment, closed-loop runner and release tracker |
| `src/vla_safety/concepts/` | Fixed-image probes: manifests, runner, analysis, offline replay, synthetic fixture |
| `scripts/sim.py` | `check`, `rollout`, `suite`, `report` |
| `scripts/probe.py` | `prepare`, `run`, `components`, `replay`, `verify` |
| `configs/simulation/fragile_release.json` | Every simulator control: scene, calibration, task, capture, interventions, suite |
| `configs/concepts/fragile_release.json` | Fixed-image probe run: threshold, manifest, seeds, capture and intervention settings |
| `client/` | Robot PC side: LeRobot robot client, launchers, live viewer, nexus client and shield calibration. Setup in [client/README.md](client/README.md) |
| `server/` | LeRobot gRPC policy server for the physical arm. `serve_molmoact2_nexus.py` serves the so101-nexus fine-tune with the fragile-release shield learned in `robot_learning_rulebased` (`fragile_shield.py`, `shield_rules/`) |

Hooks run eagerly; compilation and CUDA graphs are disabled. Action-expert modules
can fire several times during flow integration, and the capture sink keeps invocation
order. Capturing everything can exhaust memory. Where in the network a behaviour is
localized is a hypothesis; this code does not assume it.

## Simulator assets

The simulator runs in `.envs/policy` (`pip install -e .[sim]`), so intervention hooks
see every forward call. The robot model is MuJoCo Menagerie's `robotstudio_so101`,
pinned to commit `f3b6013`:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git artifacts/sim/mujoco_menagerie
git -C artifacts/sim/mujoco_menagerie sparse-checkout set robotstudio_so101
git -C artifacts/sim/mujoco_menagerie checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
```

The server has no display. Render offscreen with `MUJOCO_GL=egl`
(`MUJOCO_EGL_DEVICE_ID` picks the GPU). Call `Renderer.close()` before exit to avoid a
harmless EGL error at teardown.

## Validation history

Simulator and policy results are logged in the
[experiment plan](docs/fragile-release-experiment-plan.md#progress-log).

**2026-10-05**

- The SO-101 snapshot was assembled and SHA256-verified. Its inventory and completion manifest are stored beside it in `artifacts/models/`.
- Official sample inference succeeded offline on CPU in FP32 and returned finite `[30, 6]` actions. Outputs and logs are in `artifacts/so101-cpu-smoke.*`.

**2026-10-19**

- Host GPUs were confirmed working. Diagnostics are in `artifacts/diagnostics-host-2026-10-19.json`.
- SO-101 sample inference in FP32 on `cuda:0` returned finite `[30, 6]` actions:
  - model load: 11.13 s;
  - first inference without warm-up: 1.06 s;
  - peak allocated GPU memory: 21.81 GiB.

  These are one-run measurements, not a latency benchmark.
