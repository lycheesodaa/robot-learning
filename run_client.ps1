<#
  Runs the robot client: opens the SSH tunnel to the server, runs the client, and closes the tunnel on exit.
  Set the SSH target and task below. Everything else (arm port, cameras, checkpoint, chunking) is in
  client\so101_molmoact2.yaml, or client\so101_molmoact2_nexus.yaml for the nexus line. Edit the YAML directly.
#>
$SshTarget = 'ihealthtech4'   # Host alias from ~/.ssh/config, or user@host
$Task = 'pick up the red cube and place it on the blue circle'
$client = Join-Path $PSScriptRoot 'client\run_client.ps1'

& $client -SshTarget $SshTarget -Task $Task @args

# Nexus checkpoint with the fragile-release shield (server: ./run_server.sh --nexus); it only knows this prompt
# & $client -Nexus -SshTarget $SshTarget -Task 'Pick up the red cube and place it on the blue circle.' @args

# No live viewer and no recording under runs\
# & $client -SshTarget $SshTarget -Task $Task -NoViz @args
