#!/usr/bin/env bash
# One-time robot PC setup (Linux/macOS twin of setup_client.ps1): creates ./venv with Python 3.12+ and installs
# client/requirements.txt (LeRobot with CPU-only torch). run_client.sh then uses that venv without activating it.
# Safe to re-run; an existing ./venv is reused.
#
#   ./setup_client.sh
#
# PYTHON overrides the interpreter used to create the venv (default: the first of python3.12, python3.13, python3
# that is 3.12 or newer).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    python=${PYTHON:-}
    if [ -z "$python" ]; then
        for candidate in python3.12 python3.13 python3; do
            if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
                python=$candidate; break
            fi
        done
    fi
    [ -n "$python" ] || { echo "Need Python 3.12 or newer; install it and re-run (or set PYTHON)." >&2; exit 1; }
    echo "Creating venv with $python"
    "$python" -m venv venv
fi

venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r client/requirements.txt

command -v ssh >/dev/null || echo "ssh not found on PATH; install the OpenSSH client." >&2
cat <<'EOF'
Done. For the LeRobot tools (lerobot-find-port, lerobot-find-cameras, lerobot-calibrate), activate the venv first:
  source venv/bin/activate
Then start inference with ./run_client.sh (see README.md).
EOF
