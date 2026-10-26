# Environment and integration guide

## Policy environment

The policy environment is project-local at `.envs/policy`, with Python 3.11.13. Direct dependencies are pinned in `pyproject.toml`; the checkpoint declares Transformers 5.3.0. Runtime installation results are recorded in `configs/policy-requirements.lock.txt` once installation succeeds.

Use `conda activate "$PWD/.envs/policy"` or prefix commands with `.envs/policy/bin/python`. Hugging Face cache files go in project-local `.cache/huggingface` for download/smoke scripts.

## GPU diagnosis

```bash
python scripts/diagnose.py > artifacts/diagnostics.json
nvidia-smi
```

**Checked on 2026-10-19:** the host driver is working: NVIDIA 580.178.04, with two RTX 6000 Ada GPUs exposing 49140 MiB each. The policy environment sees both GPUs and successfully runs the SO-101 sample on `cuda:0` using PyTorch 2.7.1+cu126. The CUDA 13.0 version displayed by `nvidia-smi` is the driver capability; the installed PyTorch runtime is CUDA 12.6.

The restricted sandbox still hides `/dev/nvidia*`, causing `nvidia-smi` and PyTorch CUDA discovery to fail there even though the host is healthy. Run the diagnosis and GPU inference from a host terminal or an execution context permitted to access the GPU devices. The earlier missing-device observations alone cannot establish that the host driver was broken. No driver installation or changes were needed for this check. Host diagnostics are recorded in `artifacts/diagnostics-host-2026-10-19.json`.

```bash
.envs/policy/bin/python scripts/smoke_inference.py --model so101 --device cuda:0
```

The [checkpoint card](https://huggingface.co/allenai/MolmoAct2-SO100_101) estimates around 24 GB for FP32 without CUDA graphs, 26 GB with them, and under 16 GB for BF16 in its tests. Extra activation capture adds memory. GPU smoke tests validate shapes and finite actions, not closed-loop competence or safety.

## Upstream simulator isolation

`configs/upstream-lock.json` pins repositories for LIBERO, SafeLIBERO/AEGIS, LIBERO-Safety and the authors' LeRobot inference branch. The initial setup checks out LIBERO and SafeLIBERO under `artifacts/upstream/`; the other two remain optional. `scripts/checkout_upstream.py NAME` checks out a detached revision into `artifacts/upstream/NAME` and refuses a different existing checkout. The three LIBERO variants all expose overlapping package names; never install them together.

```bash
python scripts/checkout_upstream.py libero
python scripts/checkout_upstream.py safelibero
python scripts/checkout_upstream.py libero_safety
python scripts/checkout_upstream.py lerobot
```

The initial setup creates `.envs/libero` and `.envs/safelibero`. Recreate them if needed using the supplied simulator environment files; the optional LIBERO-Safety environment is not created automatically. They establish a pinned Python/pip foundation; simulator installation is an integration step, not claimed as validated here. Follow each pinned checkout's own requirements and install its package editable in its corresponding environment. For SafeLIBERO use `artifacts/upstream/safelibero/safelibero/requirements.txt` and its adjacent `setup.py`; the repository-root requirements belong to the broader AEGIS stack. Its simulator requirement leaves robosuite unpinned, so resolve and lock that dependency during simulator integration rather than assuming a current release is compatible. SafeLIBERO's published setup uses Python 3.8.13. The model server stays on Python 3.11. The LIBERO-Safety fork also needs its supplied robosuite and asset archive. Do not run upstream installation with sudo or install the forks into `.envs/policy`.

```bash
CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/libero" -f configs/environment-libero.yml
CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/safelibero" -f configs/environment-safelibero.yml
CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/libero-safety" -f configs/environment-libero-safety.yml
```

After integration, freeze each environment independently. Pinning upstream source does not automatically freeze transitive dependencies in those uninstalled simulator environments. Do not claim paper-reproduction fidelity until controller preprocessing, seed handling, versions and initial-state loading have been tested.

## Local policy-server interface

```bash
python scripts/policy_server.py --model libero --port 8000
```

The server binds only to `127.0.0.1`, handles one prediction at a time, and exposes `POST /predict`. Request JSON contains `version: 1`, `images_png_base64` (ordered PNG images), `instruction`, `state`, `embodiment`, and `timestamp`. Response JSON contains `version: 1`, a `[horizon, dimension]` `values` array, `embodiment`, `control_mode`, `frame`, `units`, and `normalization` provenance. It has no actuation endpoint. Client-side helpers are in `vla_safety.policies.wire`; old simulator environments can produce the same JSON using standard libraries and Pillow without importing this Python-3.11 package. Treat non-200 responses as failed inference, never as permission to reuse an unvalidated action.

SO-101 recorded inputs use this same request JSON format:

```bash
python scripts/smoke_inference.py --model so101 --recording artifacts/recording.json
```

For LIBERO, preserve agent view followed by wrist view and validate upstream image rotation before rollout. Delta pose values remain controller-scaled; do not interpret their first three coordinates as metres or directly integrate them into geometry. For SO-101, the native joint convention must be explicitly mapped to your calibration; no live conversion has been invented.

## HTTP fallback for stalled Xet transfers

Stop an active standard downloader before using the alternate path:

```bash
python scripts/download_models_http.py --workers 12
python scripts/download_models.py --verify-only
```

It stores independent 64 MiB parts under the ignored snapshot directory, checks HTTP byte ranges and final weight SHA256, and resumes complete parts after interruption. It does not assume Xet partial files are contiguous or reusable. Keep only one download process active for a snapshot. Original Xet partials are preserved and may be removed after the completion manifests and integrity checks pass.
