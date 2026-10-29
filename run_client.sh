#!/usr/bin/env bash
# Runs the robot client: opens the SSH tunnel to the server, runs the client, and closes the tunnel on exit.
# Set the SSH target and task below. Everything else (arm port, cameras, checkpoint, chunking) is in
# client/so101_molmoact2.yaml, or client/so101_molmoact2_nexus.yaml for the NEXUS=1 line. Edit the YAML directly.
ssh_target=ihealthtech4   # Host alias from ~/.ssh/config, or user@host
task="pick up the red cube and place it on the blue circle"
client=$(dirname "$0")/client/run_client.sh

exec "$client" "$ssh_target" "$task" "$@"

# Nexus checkpoint with the fragile-release shield (server: ./run_server.sh --nexus); it only knows this prompt
# NEXUS=1 exec "$client" "$ssh_target" "Pick up the red cube and place it on the blue circle." "$@"

# No live viewer and no recording under runs/
# NO_VIZ=1 exec "$client" "$ssh_target" "$task" "$@"
