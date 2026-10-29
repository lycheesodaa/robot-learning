# MolmoAct2 on SO-101: remote inference and safety study

This repository has two parts:

1. **Over-the-air inference for a physical SO-101 arm.** A GPU server runs the
   [MolmoAct2](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) vision-language-action policy, and a
   laptop or robot PC drives the arm and two cameras. They talk over LeRobot's async gRPC protocol through an SSH
   tunnel. The setup below is all you need for this part.
2. **The fragile-release study.** Does MolmoAct2 represent object fragility and release height, combine them, and
   lower an object before releasing it? This part holds a closed-loop SO-101 MuJoCo simulator, fixed-image probes,
   activation intervention hooks, and a learned safety shield that also runs on the real arm.

```
laptop / robot PC                                       GPU server
run_client.ps1 / .sh ── 127.0.0.1:8080 ══ ssh -L ══► 127.0.0.1:8080  run_server.sh
  SO-101 arm + cam0/cam1                                              MolmoAct2 on CUDA
```

## Setup for online robot inference

Details can be found in [`docs/so101_remote_inference.md`](docs/so101_remote_inference.md) in the repository's
`main` branch.

### 0. Install the environments (once per machine)

**GPU server** (Linux, NVIDIA GPU with about 24 GB free, Python 3.12+):

```bash
git clone -b main https://github.com/lycheesodaa/robot-learning.git robot_learning && cd robot_learning
./setup_server.sh          # creates .envs/lerobot and installs server/requirements.txt
```

**Client** (your laptop or robot PC, Python 3.12+ and an OpenSSH client):

```powershell
# Windows
git clone -b client https://github.com/lycheesodaa/robot-learning.git robot_learning; cd robot_learning
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, if Windows refuses to run .ps1 scripts
.\setup_client.ps1         # creates .\venv and installs client\requirements.txt (CPU-only torch)
```
```bash
# Linux / macOS
git clone -b client https://github.com/lycheesodaa/robot-learning.git robot_learning && cd robot_learning
./setup_client.sh
```

Then, on the client, set up the arm and cameras. Activate the venv first (`.\venv\Scripts\Activate.ps1` or
`source venv/bin/activate`) so the `lerobot-*` tools are on PATH:

1. Find the arm's serial port with `lerobot-find-port`, and the camera indices with `lerobot-find-cameras opencv`.
2. Put them in `client/so101_molmoact2.yaml`: `robot.port`, and `index_or_path` for `cam0` (external view)
   and `cam1` (second view).
3. Calibrate the arm once: `lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower`.
   The `id` must match `robot.id` in the YAML.
4. Check torque-on behaviour once per arm: `python client/check_torque_hold.py --probe-firmware`. If it says the
   arm WILL jump, run `python client/check_torque_hold.py --sync` before every session.

### 1. Configure `ssh` credentials to connect to the server (or your own server anywhere)

1. Download the `ihealthtech` IdentityFile.
2. Place it in a folder of your choice, and add this to `~/.ssh/config` (Windows: `%USERPROFILE%\.ssh\config`),
   with the path changed to where you put it:
   ```
   Host ihealthtech4
       HostName 172.16.163.102
       User weisoon
       IdentityFile <path_of_choice>\ihealthtech
   ```
3. Check that `ssh ihealthtech4` logs in without a password prompt. OpenSSH refuses a key file that other users
   can read. On Linux/macOS, run `chmod 600 <path_of_choice>/ihealthtech`. On Windows, run
   `icacls <path_of_choice>\ihealthtech /inheritance:r /grant:r "$($env:USERNAME):R"`.

Use the alias `ihealthtech4` as the SSH target from here on. `weisoon@172.16.163.102` would skip the
`IdentityFile` line.

### 2. Run the server

1. SSH in and `cd` into the repository you cloned in step 0.
2. Start a terminal multiplexer with `screen -S molmoact2` or `tmux new -s molmoact2`. This keeps the server
   running after you disconnect. Detach with `Ctrl-a d` (screen) or `Ctrl-b d` (tmux).
3. Run:
   ```bash
   ./run_server.sh --gpu 1
   ```
   This is equivalent to
   `CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_XET=1 .envs/lerobot/bin/python server/serve_molmoact2.py --host=127.0.0.1 --port=8080 --fps=30 --record_dir=artifacts/inference_runs --no_similarity_filter`.
   `./run_server.sh --help` lists the options. Optionally, run `./check_server.sh` from a second shell to test the
   server without a robot.

### 3. On the client machine (your personal laptop)

Start sending requests :)

```powershell
# Windows
.\run_client.ps1
```
```bash
# Linux / macOS
./run_client.sh
```

The SSH target and task are set at the top of the script; edit them there. Everything else (arm port, cameras,
checkpoint, chunking) is in `client/so101_molmoact2.yaml`. Edit that file directly. The script opens the SSH tunnel and runs the client. A live viewer opens at http://127.0.0.1:8765/, and the
run is recorded under `runs/`. Stop with `Ctrl-C`: the arm returns to its start pose and the tunnel closes.
The first connection takes about 30 s while the server loads the policy. **Keep a hand near the arm's power
switch**, especially on the first run.

## Script reference

All scripts are in the repository root. Run them from anywhere.

| Script | Where | What it does |
| --- | --- | --- |
| `setup_server.sh` | GPU server | Creates `.envs/lerobot` and installs `server/requirements.txt`. Safe to re-run |
| `run_server.sh [--nexus] [--gpu N] [--port N] [--fps N] [args...]` | GPU server | Starts the policy server on `127.0.0.1`. `--nexus` serves the so101-nexus fine-tune with the fragile-release shield (in shadow mode; add `--shield=hold` to enforce it). Other args go to the server, such as `--rtc` |
| `check_server.sh [port]` | GPU server | Robot-free test of a running server |
| `setup_client.ps1` / `setup_client.sh` | Client | Creates `venv` and installs `client/requirements.txt`. Safe to re-run |
| `run_client.ps1` / `run_client.sh` | Client | Runs the client with the SSH target and task set in the script, using the `venv` (no need to activate it). Other settings are in `client/so101_molmoact2.yaml` (`client/so101_molmoact2_nexus.yaml` for the commented-out nexus line, which pairs with `run_server.sh --nexus`) |

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/so101_remote_inference.md](docs/so101_remote_inference.md) | Remote inference in full: server options, client parameters, real-time chunking, the so101-nexus checkpoint and its shield, troubleshooting |
| [client/README.md](client/README.md) | Robot PC hardware, setup, launcher options, run replay and safety behaviour |
| [docs/setup.md](docs/setup.md) | Research environment (`.envs/policy`), checkpoint download, GPU diagnosis, simulator assets, validation history |
| [docs/fragile-release.md](docs/fragile-release.md) | Fragile-release study: simulator and probe workflows, configuration, artifacts and scoring |
| [docs/fragile-release-experiment-plan.md](docs/fragile-release-experiment-plan.md) | Fragile-release stages, gates and dated progress log |
| [docs/fragile-pickplace-experiment-plan.md](docs/fragile-pickplace-experiment-plan.md) | Fragility in pick-and-place with the so101-nexus fine-tune |
| [docs/obstacle-avoidance-experiment-plan.md](docs/obstacle-avoidance-experiment-plan.md) | Obstacle avoidance under activation patching |
| `docs/results-*.md` | Results write-ups for individual experiment matrices |
| [server/shield_rules/README.md](server/shield_rules/README.md) | The learned shield rules and their results in simulation |

## Research quick start

The study uses its own environment, `.envs/policy` (Python 3.11), separate from the inference server's
`.envs/lerobot`. See [docs/setup.md](docs/setup.md) for details.

```bash
CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs" conda env create -p "$PWD/.envs/policy" -f configs/environment-policy.yml
conda activate "$PWD/.envs/policy"
python scripts/download_models.py && python scripts/download_models.py --verify-only
python scripts/diagnose.py
pytest -q
python scripts/sim.py check --output artifacts/simulation/my-check
python scripts/sim.py rollout --device cuda:0 --output artifacts/simulation/my-baseline
```

The checkpoint revision is pinned in `configs/models.json`. Loading it executes the pinned custom model code with
`trust_remote_code=True`. Run GPU commands from a host terminal; a restricted sandbox hides the GPUs.

## Layout

| Path | Contents |
| --- | --- |
| `*.sh`, `*.ps1` (root) | Setup and run scripts for the server and client (see the script reference above) |
| `server/` | LeRobot gRPC policy server for the physical arm. `serve_molmoact2_nexus.py` adds the so101-nexus fine-tune and the fragile-release shield (`fragile_shield.py`, `shield_rules/`) |
| `client/` | Robot PC side: launchers, robot clients with recording and live viewer, nexus client, torque check and shield calibration |
| `src/vla_safety/` | Study code: policy loading, intervention hooks, activation capture, run archives, the MuJoCo simulators (`simulation/`, `pickplace/`, `obstacle/`) and fixed-image probes (`concepts/`) |
| `scripts/` | Study entry points: `sim.py`, `probe.py`, `pickplace.py`, `obstacle.py`, model download and diagnostics |
| `configs/` | Simulator, probe and environment configs; the pinned model revision |
| `docs/` | Guides, experiment plans and results (see above) |
| `tests/` | `pytest` suite |
