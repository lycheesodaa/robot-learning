#!/usr/bin/env bash
# Opens an SSH tunnel to the MolmoAct2 policy server, runs the LeRobot robot client
# through it, and closes the tunnel when the client exits. Linux/macOS twin of run_client.ps1.
#
#   client/run_client.sh weisoon@172.16.163.102 "pick up the red cube" [extra robot_client args...]
#
# Environment overrides: CONFIG, LOCAL_PORT (8080), REMOTE_PORT (8080), PYTHON (python).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,8p' "$0"; exit 2
fi
ssh_target=$1; task=$2; shift 2
config=${CONFIG:-$(dirname "$0")/so101_molmoact2.yaml}
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

"${PYTHON:-python}" -m lerobot.async_inference.robot_client "--config_path=$config" \
    "--server_address=127.0.0.1:$local_port" "--task=$task" "$@"
