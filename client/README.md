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
  - `cam0`, external view of the workspace at 320×240 @ 30 fps
  - `cam1`, wrist camera on the gripper at 320×240 @ 30 fps

  A built-in laptop webcam also shows up as an OpenCV index; don't use it for either key.
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
| Packages | [client/requirements.txt](requirements.txt) (CPU-only torch) | [server/requirements.txt](../server/requirements.txt) |
| Also needs | OpenSSH client (`ssh` on PATH) | NVIDIA driver plus a CUDA build of PyTorch, and access to the Hugging Face Hub (or a local copy of the checkpoint) |

### Robot PC setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install -r client/requirements.txt
```

Then set up the arm and cameras:

1. **Find the arm's serial port:** run `lerobot-find-port` (e.g. `COM3` or `/dev/ttyACM0`).
2. **Find the camera indices:** run `lerobot-find-cameras opencv`.
3. **Edit [client/so101_molmoact2.yaml](so101_molmoact2.yaml):** set `robot.port`, the `index_or_path`
   of both cameras, and `robot.id`.
4. **Calibrate the arm** if you haven't already:
   ```
   lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower
   ```
   The `id` here must match `robot.id` in the YAML, because it picks which saved calibration is loaded.
5. **Check torque-on behaviour** (once per arm). With the arm resting and torque off, run
   `python client/check_torque_hold.py --probe-firmware`. If it reports the arm WILL jump to a stale target,
   run `python client/check_torque_hold.py --sync` before each session.

### GPU server setup

The server code is in [server/](../server/); run these from the repo root on the GPU server.

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
| `-NoHome` / `NO_HOME=1` | Leave the arm where it is on exit instead of returning it to its start pose |
| `-Nexus` / `NEXUS=1` | Client for the so101-nexus server with the fragile-release shield (below) |

To replay a finished run in the viewer:

```
python client/viz_server.py                       # latest run
python client/viz_server.py runs/20261020_182433
```

## so101-nexus checkpoint with the fragile-release shield

This mode runs the sim-trained [so101-nexus fine-tune](https://huggingface.co/ataghof/molmoact2-so101nexus-lora-champion)
on the real arm. It adds the safety rule learned in simulation (`robot_learning_rulebased`), which refuses to open the
gripper while it is squeezing a fragile object above the table. The server is `server/serve_molmoact2_nexus.py`. Its docstring and `server/shield_rules/README.md` cover the conversions and the rules.

1. **Calibrate the shield once per arm and object** (about two minutes, guided):
   ```
   python client/calibrate_shield.py --object "foam egg"
   ```
   The script measures three things. First, the gripper's closed and open readings, which set the gripper scaling to
   the checkpoint's 0–45. Second, the forward-kinematics TCP height with the object resting in the jaws on the table.
   Third, the gripper load when squeezing the object versus closing on nothing, which sets the sign and the
   `load_gripper` threshold. Use the object you will run with. It writes `server/shield_artifacts/calibration.json`; commit it and pull on
   the server, or copy it there.
2. **Start the server** in shadow mode first. In shadow mode it logs every decision and changes nothing:
   ```bash
   python server/serve_molmoact2_nexus.py --host=127.0.0.1 --port=8080 --fps=8 --no_similarity_filter \
       --record_dir=artifacts/real_runs --fragile --shield=shadow --shield_calibration=server/shield_artifacts/calibration.json
   ```
   Then use `--shield=hold` (refused openings keep the gripper closed; the arm keeps moving) or `--shield=abort`
   (the first refusal stops the run). Both require a calibration file whose `measured` field is true.
3. **Run the client** with the prompt the checkpoint was trained on:
   ```powershell
   .\client\run_client.ps1 -Nexus -SshTarget user@gpu-host -Task "Pick up the red cube and place it on the blue circle."
   ```
   ```bash
   NEXUS=1 client/run_client.sh user@gpu-host "Pick up the red cube and place it on the blue circle."
   ```

What `-Nexus` changes:

- **Config `so101_molmoact2_nexus.yaml`:** fps 8 with chunks of 10, because the checkpoint was trained at one
  action per 0.12 s. Aggregation is `latest_only`, and frames are sent as captured (`image_size: null`; see the config).
- **Loads in observations:** every observation includes the servo loads for the server's shield.
- **Live re-check before sending:** each queued action is checked again, right before it is sent, against the joint
  positions and loads read at that moment. Actions that went stale in the queue are caught too.
- **Latch:** on a refusal in abort mode, or when the server sends an empty chunk (its shield latched and nothing
  more will come), the client empties its queue, drops chunks still in flight, and stops sending. The arm holds its
  last command, so the gripper keeps its grip. Press Ctrl-C to exit. To resume, start the client again; a new
  connection clears the server's latch.
- **Homing keeps the grip:** during a fragile run in hold or abort mode, homing on exit leaves the gripper at its
  last command, so the arm does not drop the object.

Runs record `shield`, `latch` and `blocked` events in `events.jsonl`. The server records each chunk's decision in
`inferences.jsonl`.

## Safety

- `max_relative_target: 15` limits each joint to 15° per step. Keep this limit until the behaviour looks sane.
- When the client stops (normal exit, Ctrl-C, an error, SIGTERM, Ctrl-Break or closing the console window),
  the arm returns to the pose it started in over 1–3 s. Press Ctrl-C again to stop it where it is.
  A hard kill (`kill -9`, `taskkill /F`) or power loss skips this. The gripper also returns to its start
  opening, so it drops whatever it holds. The exception is a fragile `-Nexus` run with the shield in hold or
  abort mode, where the gripper keeps its last command.
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
| `client/check_torque_hold.py` | Checks whether turning on torque would make the arm jump; `--sync` removes the risk |
| `client/home_on_exit.py` | Returns the arm to its start pose whenever the client stops |
| `client/nexus_client.py`, `client/so101_molmoact2_nexus.yaml` | Client and config for the so101-nexus server with the fragile-release shield |
| `client/calibrate_shield.py` | Measures the gripper scaling, set-down height and squeeze load the shield needs |
