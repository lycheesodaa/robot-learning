#!/usr/bin/env bash
# Starts the MolmoAct2 policy server on 127.0.0.1, so it is reachable only through the client's SSH tunnel.
# Run it inside screen or tmux so it survives logging out. Details: docs/so101_remote_inference.md.
#
#   ./run_server.sh [--nexus] [--gpu N] [--port N] [--fps N] [extra server args...]
#
#   --nexus   serve the so101-nexus fine-tune with the fragile-release shield (serve_molmoact2_nexus.py) in
#             shadow mode; add --shield=hold or --shield=abort to enforce it
#   --gpu N   GPU index (CUDA_VISIBLE_DEVICES); default $CUDA_VISIBLE_DEVICES, else 0
#   --port N  default 8080; the client's -RemotePort / REMOTE_PORT must match
#   --fps N   default 30 (8 with --nexus); must match the client config
#
# Extra args go to the server, e.g. --rtc. PYTHON overrides the interpreter (default .envs/lerobot/bin/python).
set -euo pipefail
cd "$(dirname "$0")"

nexus=; gpu=${CUDA_VISIBLE_DEVICES:-0}; port=8080; fps=; extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --nexus) nexus=1 ;;
        --gpu) gpu=$2; shift ;;
        --gpu=*) gpu=${1#*=} ;;
        --port) port=$2; shift ;;
        --port=*) port=${1#*=} ;;
        --fps) fps=$2; shift ;;
        --fps=*) fps=${1#*=} ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) extra+=("$1") ;;
    esac
    shift
done

python=${PYTHON:-.envs/lerobot/bin/python}
[ -x "$python" ] || { echo "$python not found; run ./setup_server.sh first." >&2; exit 1; }
[ -n "${STY:-}${TMUX:-}" ] || echo "Note: not inside screen or tmux; the server stops when this SSH session ends." >&2

if [ -n "$nexus" ]; then
    args=(server/serve_molmoact2_nexus.py --fps="${fps:-8}" --record_dir=artifacts/real_runs
          --fragile --shield=shadow --shield_calibration=server/shield_artifacts/calibration.json)
else
    args=(server/serve_molmoact2.py --fps="${fps:-30}" --record_dir=artifacts/inference_runs)
fi

echo "Serving on 127.0.0.1:$port, GPU $gpu. The policy loads when a client connects. Stop with Ctrl-C."
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_DISABLE_XET=1 exec "$python" "${args[@]}" \
    --host=127.0.0.1 --port="$port" --no_similarity_filter ${extra[@]+"${extra[@]}"}
