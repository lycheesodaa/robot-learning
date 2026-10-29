#!/usr/bin/env bash
# One-time GPU server setup: creates .envs/lerobot (Python 3.12+) and installs server/requirements.txt.
# Safe to re-run; an existing .envs/lerobot is reused.
#
#   ./setup_server.sh
#
# PYTHON overrides the interpreter used to create the env (default: the first of python3.12, python3.13, python3
# that is 3.12 or newer).
set -euo pipefail
cd "$(dirname "$0")"

env=.envs/lerobot
if [ ! -x "$env/bin/python" ]; then
    python=${PYTHON:-}
    if [ -z "$python" ]; then
        for candidate in python3.12 python3.13 python3; do
            if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
                python=$candidate; break
            fi
        done
    fi
    [ -n "$python" ] || { echo "Need Python 3.12 or newer (e.g. conda create -p .envs/lerobot python=3.12, then re-run)." >&2; exit 1; }
    echo "Creating $env with $python"
    "$python" -m venv "$env"
fi

"$env/bin/python" -m pip install --upgrade pip
# Linux PyPI torch is a CUDA build; install a matching torch first if it doesn't suit the driver.
"$env/bin/python" -m pip install -r server/requirements.txt
"$env/bin/python" -c 'import torch; print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")'

command -v screen >/dev/null || command -v tmux >/dev/null \
    || echo "Neither screen nor tmux is installed; the server will stop when your SSH session ends." >&2
echo "Done. Start the server with ./run_server.sh (inside screen or tmux)."
