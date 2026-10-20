# MolmoAct2 on SO-101

Runs the [MolmoAct2](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) vision-language-action
policy on an SO-101 arm using LeRobot 0.6.1's async inference. The policy runs on a remote GPU server, and
a local client drives the arm. The two talk over gRPC through an SSH tunnel.

```
 laptop / robot PC                                  GPU server
 ┌──────────────────────────────┐   SSH tunnel   ┌───────────────────────────┐
 │ client/viz_client.py         │  127.0.0.1:8080│ server/serve_molmoact2.py │
 │  SO-101 + 2 cameras ─ obs ──────────────────────▶ MolmoAct2 (CUDA)        │
 │  live viewer :8765 ◀ actions ◀───────────────────                         │
 └──────────────────────────────┘                └───────────────────────────┘
```

## Hardware

**Robot side**
- **SO-101 follower arm**: 6 Feetech STS3215 servos plus the Feetech USB serial bus board, and its power
  supply. This is the only arm needed: the policy replaces the leader arm.
- **Two USB cameras** that OpenCV can open:
  - `cam0`, top view at 640×480 @ 30 fps
  - `cam1`, side view at 320×240 @ 30 fps

  The checkpoint was trained on top and side views, so mount the cameras to match.
- **A PC** running Windows (PowerShell) or Linux/macOS (bash), with free USB ports for the arm and both cameras.
  The client runs on the CPU.

**Server side**
- **A Linux machine with an NVIDIA CUDA GPU** that has enough memory to run MolmoAct2 inference. See the
  model card for sizing.
- **SSH access** to the server from the robot PC. Key-based login is recommended so the tunnel opens without a prompt.

## Software

| | Robot PC | GPU server |
|---|---|---|
| Python | 3.12+ (developed on 3.13) | 3.12+ |
| Packages | [requirements.txt](requirements.txt) | [server/requirements.txt](server/requirements.txt) |
| Also needs | OpenSSH client (`ssh` on PATH) | NVIDIA driver plus a CUDA build of PyTorch, and access to the Hugging Face Hub (or a local copy of the checkpoint) |

### Robot PC setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Then set up the arm and cameras:

1. **Find the arm's serial port:** run `lerobot-find-port` (e.g. `COM3` or `/dev/ttyACM0`).
2. **Find the camera indices:** run `lerobot-find-cameras opencv`.
3. **Edit [client/so101_molmoact2.yaml](client/so101_molmoact2.yaml):** set `robot.port`, the `index_or_path`
   of both cameras, and `robot.id`.
4. **Calibrate the arm** if you haven't already:
   ```
   lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower
   ```
   The `id` here must match `robot.id` in the YAML, because it picks which saved calibration is loaded.

### GPU server setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r server/requirements.txt   # install a CUDA build of torch first if pip's default doesn't match your driver
python server/serve_molmoact2.py --host=127.0.0.1 --port=8080 --fps=30
```

Binding to `127.0.0.1` keeps the server reachable only through the SSH tunnel. Use `CUDA_VISIBLE_DEVICES`
to choose the GPU. The checkpoint named in the client config (`pretrained_name_or_path`) is downloaded on the
server when the first client connects.

## Running

With the server running, start the client from the repo root:

```powershell
.\client\run_client.ps1 -SshTarget user@gpu-host -Task "pick up the red pen"
```
```bash
client/run_client.sh user@gpu-host "pick up the red pen"
```

The launcher does three things:
1. Opens the SSH tunnel.
2. Runs the client, which shrinks frames to 224×224 before sending, records the run under `runs/<timestamp>/`,
   and opens a live viewer at http://127.0.0.1:8765/.
3. Closes the tunnel when the client exits.

Any extra arguments are passed through to LeRobot's `robot_client`, for example `--robot.port=COM7`.
Useful launcher options:

| Option | Effect |
|---|---|
| `-NoViz` / `NO_VIZ=1` | Use the plain client: no recording and no viewer |
| `-LocalPort` / `-RemotePort` | Change the tunnel's local and remote ports |
| `-ImageSize ''` | Send full-resolution frames instead of 224×224 |

To replay a finished run in the viewer:

```
python client/viz_server.py                       # latest run
python client/viz_server.py runs/20261020_182433
```

## Safety

- `max_relative_target: 15` limits each joint to 15° per step. Keep this limit until the behaviour looks sane.
- `disable_torque_on_disconnect: false` makes the arm **hold its last pose** after Ctrl-C instead of going limp.
  To release the arm, cut its power.
- Keep a hand on the power switch for the first runs.

## Layout

| Path | Contents |
|---|---|
| `client/run_client.ps1`, `client/run_client.sh` | Launchers: open the SSH tunnel, run the client, close the tunnel |
| `client/so101_molmoact2.yaml` | Robot, camera and policy config |
| `client/viz_client.py` | Robot client with run recording and the live viewer |
| `client/viz_server.py`, `client/viz.html` | Viewer and run replay |
| `client/resize_client.py` | Robot client that resizes frames, without recording |
| `server/serve_molmoact2.py` | LeRobot policy server with `molmoact2` added to its supported policies |
