# SO101 with remote MolmoAct2 inference

The robot PC drives the SO101 arm and two cameras with LeRobot's `robot_client`. The GPU server runs MolmoAct2 (`server/serve_molmoact2.py`) and returns chunks of 30 joint targets. They talk over gRPC through an SSH tunnel, so the server port is never exposed on the network.

```
robot PC                                    GPU server (ihealthtech4)
robot_client ──► 127.0.0.1:8080 ══ ssh -L ══► 127.0.0.1:8080  serve_molmoact2.py
  arm + cam0/cam1                                              MolmoAct2 on CUDA
```

Both sides use **LeRobot 0.6.1 on Python 3.12**.

## Files

The server files are in `server/` and the robot PC files in `client/` ([client/README.md](../client/README.md) has the robot PC setup). Run the robot PC commands below from the repo root.

| File | Purpose |
| --- | --- |
| [server/serve_molmoact2.py](../server/serve_molmoact2.py) | Stock LeRobot policy server with `molmoact2` added to its allow-list; takes the same options |
| [server/smoke_client.py](../server/smoke_client.py) | Robot-free loopback test: sends the official sample images and state, prints the action chunk |
| [server/requirements.txt](../server/requirements.txt) | Server packages, for a fresh env outside `.envs/lerobot` |
| [client/requirements.txt](../client/requirements.txt) | Robot PC packages: LeRobot with the Feetech and async extras, and CPU-only torch |
| [server/serve_molmoact2_nexus.py](../server/serve_molmoact2_nexus.py) | Serves the so101-nexus fine-tune with the fragile-release shield (section 5) |
| [server/fragile_shield.py](../server/fragile_shield.py), [server/shield_rules/](../server/shield_rules/) | Shield rule language, forward kinematics and real-to-sim features; the learned rules |
| [client/so101_molmoact2.yaml](../client/so101_molmoact2.yaml) | All client settings (robot, cameras, checkpoint, chunking); copy it per setup |
| [client/run_client.ps1](../client/run_client.ps1) | Windows: opens the tunnel, runs the client, closes the tunnel |
| [client/run_client.sh](../client/run_client.sh) | The same for Linux and macOS |
| [client/check_torque_hold.py](../client/check_torque_hold.py) | Checks whether turning on torque would make the arm jump (read-only by default); `--probe-firmware` makes a 5° test move; `--sync` removes the risk |
| [client/nexus_client.py](../client/nexus_client.py), [client/so101_molmoact2_nexus.yaml](../client/so101_molmoact2_nexus.yaml) | Client and config for the nexus server (section 5) |
| [client/calibrate_shield.py](../client/calibrate_shield.py) | Measures the shield calibration for an arm and object |

## 1. Server (GPU machine)

One-time: the env is `.envs/lerobot` (`lerobot[molmoact2,async]==0.6.1`). Elsewhere, run `pip install "lerobot[molmoact2,async]==0.6.1"` in a Python 3.12 env.

Start the server in `screen` so it survives your SSH session:

```bash
screen -S molmoact2
cd ~/phd/robot_learning
CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_XET=1 \
  .envs/lerobot/bin/python server/serve_molmoact2.py --host=127.0.0.1 --port=8080 --fps=30 --record_dir=artifacts/inference_runs \
  --no_similarity_filter 
# detach: Ctrl-a d    reattach: screen -r molmoact2    list: screen -ls
```

- **`--host=127.0.0.1` is required for safety.** The server unpickles whatever it receives, so anyone who can reach the port can run code as you. Only connect through the tunnel.
- **`CUDA_VISIBLE_DEVICES`** chooses the GPU; the client's `policy_device: cuda` then means that card. FP32 needs about 22 GB.
- **`HF_HUB_DISABLE_XET=1`**: Hugging Face's Xet transfers stall on this network, and plain HTTP works.
- The policy loads when a client connects, not at startup. **Every new client connection reloads it**, and the first inference also compiles CUDA graphs. Expect a pause before the arm moves.
- Measured on 2026-10-20 (RTX 6000 Ada, FP32, loopback): about 28 s to load the policy per connection, about 1.5 s for the first chunk, and then about 0.4 s per 30-action chunk on the server. GPU memory stayed at about 24 GB across reconnects.
- **One robot per server process.** A new client resets the previous one. For a second robot, start another server on another port, such as `--port=8081`.

Check the server without a robot, from a second shell on the server:

```bash
.envs/lerobot/bin/python server/smoke_client.py --server_address=127.0.0.1:8080
```

## 2. Robot PC: one-time setup

1. **Python env.** Activate it (Windows: `.\venv\Scripts\Activate.ps1`), then run `pip install -r client/requirements.txt`. It installs CPU-only torch, because the robot PC never runs the policy.
2. **SSH key login** to the server, so the tunnel doesn't stop to ask for a password:
   ```powershell
   ssh-keygen -t ed25519                     # accept defaults
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh weisoon@172.16.163.102 "cat >> ~/.ssh/authorized_keys"
   ```
   On Linux/macOS, use `ssh-copy-id weisoon@172.16.163.102`. An optional alias goes in `~/.ssh/config` (Windows: `%USERPROFILE%\.ssh\config`); after that, `molmo-gpu` works anywhere a target is expected:
   ```
   Host molmo-gpu
       HostName 172.16.163.102
       User weisoon
   ```
3. **Arm.** Run `lerobot-find-port`, then calibrate with `lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower`. Run `lerobot-setup-motors` first only on a brand-new kit.
4. **Cameras.** `lerobot-find-cameras opencv` lists indices. Use a top view as `cam0` and a side view as `cam1`, matching the training data. Top plus wrist also works, with lower expected quality.
5. **Config.** Copy `client/so101_molmoact2.yaml` and set `robot.port`, `robot.id`, and the camera indices. Keep one copy per robot or camera setup.
6. **Torque-on behaviour (once per arm).** When the client connects, LeRobot turns torque on without first setting each servo's target to its current position. So the arm drives to whatever target is stored in the servo, which may be a stale one from the last run if the arm was moved by hand since. With the arm resting and torque off, run:
   ```powershell
   python client\check_torque_hold.py --probe-firmware   # reads targets vs present pose, then makes a 5° wrist_roll test move
   ```
   It prints each joint's stored target against its present position. It then says whether this firmware drives to a stale stored target when torque comes on ("WILL make the arm jump") or holds the present pose. If it drives to the stored target, run `python client\check_torque_hold.py --sync` before each session. That sets every stored target to the present pose without moving anything. Then start the client without touching the arm.

## 3. Each run

Start the server (section 1). Then, on the robot PC:

```powershell
# Windows
.\client\run_client.ps1 -SshTarget weisoon@172.16.163.102 -Task "pick up the red cube"
```
```bash
# Linux / macOS
client/run_client.sh weisoon@172.16.163.102 "pick up the red cube"
```

Stop with `Ctrl-C`. The tunnel closes with the client. With `disable_torque_on_disconnect: false`, the arm holds its last pose after Ctrl-C. To release it, support the arm and switch off its power, or set the option to `true` to have it go limp on disconnect. Keep a hand near the arm's power switch, especially on the first run with a new checkpoint or scene.

Any extra arguments go straight to `robot_client` and override the YAML:

```powershell
.\client\run_client.ps1 -SshTarget molmo-gpu -Task "put the lemon in the bowl" `
    -Config .\my_setup.yaml -LocalPort 8081 -RemotePort 8081 `
    --robot.port=COM7 --pretrained_name_or_path=me/my-finetuned-molmoact2 --chunk_size_threshold=0.6
```

For `run_client.sh`, set `CONFIG`, `LOCAL_PORT`, `REMOTE_PORT` and `PYTHON` as environment variables.

**Doing it by hand** shows what the scripts do. Run the tunnel in one terminal and leave it open:

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 8080:127.0.0.1:8080 weisoon@172.16.163.102
```

Then run the client in a second terminal:

```powershell
python -m lerobot.async_inference.robot_client --config_path=client/so101_molmoact2.yaml `
    --server_address=127.0.0.1:8080 --task="pick up the red cube"
```

## 4. Parameters

All of these live in the YAML and can be overridden as `--name=value`.

| Parameter | Default | What it does |
| --- | --- | --- |
| `task` | — | Instruction sent to the model with every observation. Short, imperative and concrete works best: "pick up the red cube and place it in the bowl". |
| `robot.port` | `COM5` | Serial port of the follower arm. |
| `robot.id` | `my_follower` | Calibration name. It must match what you calibrated with. |
| `robot.cameras.cam0/cam1` | indices 0, 1 at 640×480@30 | **Names must stay `cam0`/`cam1`**, because the checkpoint expects them and the client cannot rename. The server resizes frames to the checkpoint's declared 224×224, and the MolmoAct2 image processor then upscales each one to 378×378 for the vision encoder. |
| `image_size` | `224x224` (`null` in the nexus YAML) | Client-side only; the client removes it before LeRobot parses the YAML. Each frame is shrunk to this HxW before sending, or sent at capture resolution with `null`. The server resizes to the checkpoint's input either way, so this only decides where the resize happens and how much goes over the tunnel. `--image_size=`, `-ImageSize` and `IMAGE_SIZE` override it. |
| `robot.max_relative_target` | `10` | Largest per-step change for each joint, in degrees. A safety cap; use `null` to remove it once you trust the policy. |
| `robot.disable_torque_on_disconnect` | `false` | `false`: the arm holds its last pose when the client stops. `true` (LeRobot's default): torque off on exit, and the arm drops under gravity. |
| `robot.use_degrees` | `true` | Keep `true`. The checkpoint's joint convention is in degrees. |
| `policy_type` | `molmoact2` | Must be `molmoact2` for this server. |
| `pretrained_name_or_path` | `lerobot/MolmoAct2-SO100_101-LeRobot` | Hub repo or **server-side** path. Swap in a fine-tuned checkpoint here. |
| `policy_device` | `cuda` | Device **on the server**. Choose the card with `CUDA_VISIBLE_DEVICES` there, or use `cuda:1` here. |
| `server_address` | `127.0.0.1:8080` | The local end of the tunnel. The launchers set it from `-LocalPort`/`LOCAL_PORT`. |
| `fps` | `30` | Control rate. Keep it equal to the server's `--fps` and the camera fps. |
| `actions_per_chunk` | `30` | Actions used from each prediction; the model outputs 30 (1 s at 30 fps). |
| `chunk_size_threshold` | `0.5` | Asks for a new chunk when the queue falls below this fraction (0.5 means 15 actions, 0.5 s). Raise it if the arm pauses between chunks, because inference plus network time is longer than the remaining queue. At 0.5 there is about 0.1 s of margin over the server's 0.4 s, so try 0.6–0.7 if your network round trip is slow. |
| `aggregate_fn_name` | `weighted_average` | How overlapping actions from consecutive chunks blend: `weighted_average` (0.3 old + 0.7 new), `latest_only`, `average`, or `conservative` (0.7 old + 0.3 new). |

Two settings live only on the server: `--fps`, and `--obs_queue_timeout` (seconds to wait for an observation).

### Diagnosing a hesitant arm

Each inference logs a line like `Chunk #534 (must_go: True) | requested: end-of-chunk |d| 0.62, max |d| 0.41 (wrist_flex.pos) | moved since last inference: 0.18`. The units are degrees, and 0–100 for the gripper. The line shows whether a stall comes from the policy asking for almost no motion, or from the arm not following what it asked for.

`serve_molmoact2.py` also takes two diagnostic options:

- `--no_similarity_filter`: by default the server skips observations whose joints moved less than 1 (L2) since the last inference. While the arm is still, only must_go observations get through, so the policy re-plans about every 4 s. This flag makes it re-plan on every observation. Continuous MolmoAct2 samples fresh noise on every call, so more inferences give the policy more chances to produce a chunk that commits.
- `--record_dir=DIR`: saves each inference's camera frames as JPEG, plus a line in `inferences.jsonl` with the task, joint state and returned chunk. Each client connection gets its own subfolder. Expect about 100 KB per inference.

### Smoother motion: real-time chunking

Each chunk is smooth on its own; the jerks come where one chunk hands over to the next. MolmoAct2 samples fresh noise on every call, so consecutive chunks disagree on the timesteps they share, by 2–8° on average in the recorded runs. `--rtc` turns on real-time chunking (Black et al., 2025). The server remembers the last chunk it sent. When it denoises the next one, it pulls the steps the arm will execute before the new chunk arrives onto the old chunk, then fades the pull out over the following steps.

```bash
# server: add to the usual command
.envs/lerobot/bin/python server/serve_molmoact2.py ... --rtc
# client: execute exactly what the server sent, with no blending
client/run_client.sh weisoon@172.16.163.102 "pick up the pen" --aggregate_fn_name=latest_only
```

Use `latest_only` on the client with `--rtc`. The server guides each chunk toward the chunk it sent last; the client's blending would make the arm execute something else.

A replay of 80 recorded inferences on 2026-10-25 lowered the jump at each handover from 2.07° to 0.18° on average (90th percentile 8.5° to 0.7°). Each round trip took 0.49 s instead of 0.41 s, because the guided denoising loop runs without CUDA graphs. These are open-loop replays; they have not been run on the arm.

Each inference logs `RTC #<timestep> | prefix <n> steps, delay <d> | client <r> actions/s, estimated latency <s>`. The delay is the number of steps the client executes before the chunk arrives, and those steps follow the old chunk exactly. The server estimates it from the client's action rate (from the client's own timestamps, so clock skew does not matter), the time the observation waited on the server, the inference time, and `--rtc_net_latency` (default 0.25 s, for upload and download). If the arm still jumps when a chunk arrives, raise `--rtc_net_latency` or fix the delay with `--rtc_delay=N`. A delay that is too high costs a little reactivity; one that is too low brings the jump back. `--rtc_horizon` (default 20) sets the step where the guidance has faded to zero.

## 5. so101-nexus checkpoint with the fragile-release shield

`server/serve_molmoact2_nexus.py` runs the [so101-nexus fine-tune](https://huggingface.co/ataghof/molmoact2-so101nexus-lora-champion), which was trained in simulation, on the real arm. It also adds the `refuse_if` shield learned in the fragile-release study (`robot_learning_rulebased`). The shield refuses to open the gripper while it is squeezing a fragile object above the table. Every `serve_molmoact2.py` option still works. On the robot PC, the launchers' `-Nexus` / `NEXUS=1` mode runs `client/nexus_client.py` with `client/so101_molmoact2_nexus.yaml`.

How this differs from the base checkpoint:

- **The server chooses the checkpoint** with `--checkpoint`, and ignores `pretrained_name_or_path` in the client YAML. The default is `artifacts/models/so101_nexus_v6/<rev>` if that folder exists, otherwise the Hub repo. The conversions below only fit this checkpoint.
- **Joint frames.** The checkpoint's processor applies no SO101 joint correction. So the server converts the state from the real frame to the checkpoint frame and converts actions back (`fragile_shield.real_to_ckpt` / `ckpt_to_real`). The gripper is rescaled from the real 0–100 to the checkpoint's 0 (closed) to 45 (open). The checkpoint was trained on a binary gripper, so it asks for either 0 or 45.
- **Rate.** Each chunk has 10 actions, one per 0.12 s, so run the server and client at fps 8. `--rtc_horizon` defaults to 10 here.
- **One prompt.** The checkpoint was trained only on "Pick up the red cube and place it on the blue circle."
- **Cameras.** `cam0` is the external view and `cam1` the wrist. The checkpoint declares 640×480 inputs, so the nexus YAML captures at 640×480 and sends frames as captured (`image_size: null`), and the server's resize does nothing. Capturing at 320×240 also works; the server then upscales each frame 2×. In training, `cam0` was a low oblique view from a front corner: about 0.5 m in front of the arm, 0.25 m to the side and 14 cm high, aimed back at the workspace, with a 58° vertical field of view (the `front_low` preset in [realism.py](../src/vla_safety/pickplace/realism.py)). These numbers are approximate. The sim preset only roughly matches the real MolmoAct2 rigs, and a camera placed by eye on the real table will differ more. Place your camera about the same way if you can.

### Running it

1. **Calibrate the shield** once per arm and object, on the robot PC. The script guides you through it in about two minutes. Keep a hand on the arm, because the first steps run with torque off.
   ```powershell
   python client\calibrate_shield.py --object "foam egg"
   ```
   It measures three things:
   - The gripper's closed and open readings. These set the gripper scaling.
   - The forward-kinematics TCP height with the object resting in the jaws on the table. This height maps to the simulator's set-down height of 0.0295 m.
   - The gripper load when squeezing the object and when closing on nothing. These set the load sign and the `load_gripper` threshold, halfway between the two readings.

   Use the object you will run with, because a soft object squeezes with less load than the simulator's cube. Torque stays on at the end; cut power to release the arm. The script writes `server/shield_artifacts/calibration.json` (`--out` changes it), the path the server command below uses. Commit it and pull on the server, or copy it there.
2. **Start the server in shadow mode first.** In this mode it logs every decision and changes nothing:
   ```bash
   screen -S nexus
   cd ~/phd/robot_learning
   CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_XET=1 \
     .envs/lerobot/bin/python server/serve_molmoact2_nexus.py --host=127.0.0.1 --port=8080 --fps=8 \
     --no_similarity_filter --record_dir=artifacts/real_runs \
     --fragile --shield=shadow --shield_calibration=server/shield_artifacts/calibration.json
   ```
   Switch to `--shield=hold` or `--shield=abort` once the shadow decisions look right.
3. **Run the client** with the training prompt:
   ```powershell
   .\client\run_client.ps1 -Nexus -SshTarget weisoon@172.16.163.102 -Task "Pick up the red cube and place it on the blue circle."
   ```
   ```bash
   NEXUS=1 client/run_client.sh weisoon@172.16.163.102 "Pick up the red cube and place it on the blue circle."
   ```

### Shield options

| Option | Default | What it does |
| --- | --- | --- |
| `--shield` | `shadow` | `off`. `shadow`: decide and log, change nothing. `hold`: a refused action keeps the previous gripper command, while the arm joints move as proposed. `abort`: the first refusal latches; from then on every request gets an empty chunk, without inference, until the client reconnects. |
| `--shield_rule` | `server/shield_rules/v4_symbolic_k3.json` | The rule, in simulator units. [server/shield_rules/README.md](../server/shield_rules/README.md) lists the alternatives and their results in simulation. Only the default has a height condition, so the others also block safe set-downs. |
| `--shield_calibration` | none | JSON from `calibrate_shield.py`. `hold` and `abort` need one whose `measured` field is true. |
| `--fragile` | off | Marks the object as fragile; this is the rule's `fragile` input. Without it, a rule that has a `fragile` condition allows everything. |
| `--checkpoint` | see above | Server-side path or Hub repo of the nexus checkpoint. |

The default rule is `fragile AND tcp_z > 0.04924 AND load_gripper <= -0.09125`. With a calibration file, the height condition means about 2 cm above set-down height. The calibrated load threshold replaces the simulator's −0.09 N m, which is too fine for a real servo.

### How the shield decides

The server checks every action of a chunk. It checks action k against the state the arm should be in when that action runs: the observed state for k = 0, and action k−1's arm targets after that. The gripper position and the servo loads always come from the observation, because a held object stops the jaws short of their target. If a feature cannot be computed, the rule refuses (fail-closed). The server attaches the rule, calibration, mode and its decision to every action it sends.

The nexus client then does the following:

- **Sends servo loads.** Every observation includes the servo loads (`<motor>.load`, signed Present_Load). The shield reads them; the policy never sees them.
- **Checks again before sending.** Each queued action is checked again, right before it goes to the arm, against the joint positions and loads read at that moment. This also catches actions that went stale while the chunk travelled.
- **Latches.** This happens after a refusal in abort mode, or when the server sends an empty chunk (its shield latched). The client empties its queue, drops chunks still in flight, and stops sending. The arm holds its last command, so the gripper keeps its grip. Press Ctrl-C to exit; a new connection clears the server's latch.
- **Never blends chunks.** In hold and abort modes, each new chunk replaces the queue (`latest_only`), so a held chunk is never averaged with an earlier opening.
- **Keeps the grip on exit.** During a fragile run in hold or abort mode, homing on exit leaves the gripper at its last command, so the arm does not drop the object.

Use the nexus client with this server. The stock client sends no loads, so with the default rule every action more than about 2 cm above set-down height is refused as `unknown_feature`. It also executes whatever it has already queued.

### Logs and records

On each connection the server logs `Shield <mode> | fragile: <bool> | <rule>`. On each inference it logs `Shield #<timestep> [<mode>] allow` or `REFUSE <n>/<m> from step <k> (<reasons>)`, followed by the rule's feature values. When abort fires, it logs `Shield LATCHED`. With `--record_dir`, each line of `inferences.jsonl` also carries the decision, the reasons and the feature values. On the robot PC, runs record `shield`, `latch` and `blocked` events in `runs/<run_id>/events.jsonl`.

## 6. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ssh exited` / `Tunnel did not come up` | Wrong target, no key login, or you're not on the server's network or VPN. Try `ssh <target>` by itself first. |
| `Local port 8080 is already in use` | A tunnel from an earlier run is still open. Close it or use `-LocalPort 8081`. |
| `Failed to connect to policy server` | The tunnel is up but the server isn't running, or it's on another port. Check the `screen` session (`screen -r molmoact2`). |
| `Policy type molmoact2 not supported` | The server was started with the stock `lerobot.async_inference.policy_server` instead of `serve_molmoact2.py`. |
| Server log `KeyError: 'observation.images.<name>'` | A camera in the YAML isn't named `cam0` or `cam1`. |
| Long pause after connecting | Expected: the policy loads and CUDA graphs compile. A Hub checkpoint not yet on the server also downloads at this point. |
| Arm moves in jerks or pauses every second | Inference is slower than the remaining queue. Raise `chunk_size_threshold`, and check GPU load on the server. |
| Arm jumps to an old pose when the client connects | A stale stored target (see setup step 6). Run `check_torque_hold.py --sync` before starting. |
| Arm drifts the wrong way at the shoulder or elbow | Calibration doesn't match `robot.id`, or `use_degrees` is off. The checkpoint already applies the SO101 joint sign and offset correction. |

## 7. Expectations

The base SO100/101 checkpoint without fine-tuning will be hit-or-miss on a new scene. The usual next step is to record 50 or more demos with `lerobot-record` (leader arm) and LoRA fine-tune on the server; see the [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/main/en/molmoact2). The resulting checkpoint plugs in through `pretrained_name_or_path`.
