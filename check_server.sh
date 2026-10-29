#!/usr/bin/env bash
# Robot-free check of a running server (base checkpoint), from a second shell on the GPU server: sends the
# official sample images and state and prints the returned action chunks. Expect a pause while the policy loads.
#
#   ./check_server.sh [port]        # default 8080
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-.envs/lerobot/bin/python}" server/smoke_client.py --server_address="127.0.0.1:${1:-8080}"
