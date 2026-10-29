#!/usr/bin/env bash
# Opens an SSH tunnel to the MolmoAct2 policy server, runs the LeRobot robot client
# through it, and closes the tunnel when the client exits. Linux/macOS twin of run_client.ps1.
#
#   client/run_client.sh weisoon@172.16.163.102 "pick up the red cube" [extra robot_client args...]
#
# Environment overrides: CONFIG, LOCAL_PORT (8080), REMOTE_PORT (8080), PYTHON (the repo's venv from setup_client.sh, else python),
# IMAGE_SIZE (HxW frames are shrunk to before sending, empty sends full frames; default: image_size in the config),
# VIZ_PORT (8765, live viewer), NO_VIZ=1 (plain robot_client, no recording; frames still resized),
# NO_HOME=1 (leave the arm where it is on exit instead of returning it to its start pose),
# NEXUS=1 (client for server/serve_molmoact2_nexus.py: nexus_client.py, so101_molmoact2_nexus.yaml,
# see nexus_client.py).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0"; exit 2
fi
ssh_target=$1; task=$2; shift 2
if [ -n "${NEXUS:-}" ]; then
    [ -z "${NO_VIZ:-}" ] || { echo "NEXUS=1 needs the recording client; unset NO_VIZ." >&2; exit 2; }
    config=${CONFIG:-$(dirname "$0")/so101_molmoact2_nexus.yaml}
else
    config=${CONFIG:-$(dirname "$0")/so101_molmoact2.yaml}
fi
venv_python=$(dirname "$0")/../venv/bin/python
[ -x "$venv_python" ] || venv_python=python
local_port=${LOCAL_PORT:-8080}
remote_port=${REMOTE_PORT:-8080}

echo "Opening tunnel 127.0.0.1:$local_port -> $ssh_target 127.0.0.1:$remote_port"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L "$local_port:127.0.0.1:$remote_port" "$ssh_target" &
tunnel=$!
trap 'kill $tunnel 2>/dev/null; echo "Tunnel closed."' EXIT

for _ in $(seq 60); do
    (exec 3<>"/dev/tcp/127.0.0.1/$local_port") 2>/dev/null && break
    kill -0 $tunnel 2>/dev/null || { echo "ssh exited; check the SSH target and that the server is running." >&2; exit 1; }
    sleep 0.5
done

if [ -n "${NEXUS:-}" ]; then entry=("$(dirname "$0")/nexus_client.py" "--viz_port=${VIZ_PORT:-8765}")
elif [ -n "${NO_VIZ:-}" ]; then entry=("$(dirname "$0")/resize_client.py")
else entry=("$(dirname "$0")/viz_client.py" "--viz_port=${VIZ_PORT:-8765}"); fi
if [ -n "${IMAGE_SIZE+x}" ]; then entry+=("--image_size=$IMAGE_SIZE"); fi
if [ -n "${NO_HOME:-}" ]; then entry+=(--no_home); fi
"${PYTHON:-$venv_python}" "${entry[@]}" "--config_path=$config" \
    "--server_address=127.0.0.1:$local_port" "--task=$task" "$@"
