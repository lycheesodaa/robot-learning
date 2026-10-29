# Environment and setup

## Policy environment

The policy environment is project-local at `.envs/policy`, with Python 3.11.13. Direct dependencies are pinned in `pyproject.toml` (`policy`, `sim` and `dev` extras); the checkpoint declares Transformers 5.3.0. The installed versions are recorded in `configs/policy-requirements.lock.txt`. The simulator and the model share this environment, so intervention hooks see every forward call.

Use `conda activate "$PWD/.envs/policy"` or prefix commands with `.envs/policy/bin/python`. Hugging Face cache files go in the project-local `.cache/huggingface`.

The LeRobot server for the physical arm uses its own environment, `.envs/lerobot`; see [so101_remote_inference.md](so101_remote_inference.md).

## Checkpoint download

```bash
python scripts/download_models.py
python scripts/download_models.py --verify-only
python scripts/check_model_metadata.py
```

The SO-101 snapshot is about 21 GB. Downloads resume through Hugging Face's local cache. Verification checks every file size, weight SHA256, shard reference and the normalization tag. `check_model_metadata.py` builds the architecture without allocating weights and writes the module list to `artifacts/so101-modules.txt`.

## GPU diagnosis

```bash
python scripts/diagnose.py > artifacts/diagnostics.json
nvidia-smi
```

**Checked on 2026-10-19:**

- The host driver is working: NVIDIA 580.178.04, with two RTX 6000 Ada GPUs exposing 49140 MiB each.
- The policy environment sees both GPUs and runs the SO-101 sample on `cuda:0` using PyTorch 2.7.1+cu126.
- `nvidia-smi` reports CUDA 13.0. That is the driver's capability; the installed PyTorch runtime uses CUDA 12.6.

The restricted sandbox hides `/dev/nvidia*`, so `nvidia-smi` and PyTorch CUDA discovery fail there even though the host is healthy. Run diagnosis, GPU inference and MolmoAct2 rollouts from a host terminal or another context with access to the GPU devices. Host diagnostics are recorded in `artifacts/diagnostics-host-2026-10-19.json`.

```bash
.envs/policy/bin/python scripts/smoke_inference.py --device cuda:0
```

The [checkpoint card](https://huggingface.co/allenai/MolmoAct2-SO100_101) gives these memory estimates from its own tests:

- FP32 without CUDA graphs: around 24 GB;
- FP32 with CUDA graphs: around 26 GB;
- BF16: under 16 GB.

Activation capture adds memory. `--dtype bfloat16` lowers memory use; record that precision choice with any results. GPU smoke tests validate shapes and finite actions, not closed-loop competence or safety.

## Conventions

SO-101 actions are absolute joint targets in the checkpoint's native joint/gripper convention. The simulator maps them through the explicit calibration in `configs/simulation/fragile_release.json`. No physical motor conversion is assumed here.

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
[experiment plan](fragile-release-experiment-plan.md#progress-log).

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
