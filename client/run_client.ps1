<#
.SYNOPSIS
  Opens an SSH tunnel to the MolmoAct2 policy server, runs the LeRobot robot client
  through it, and closes the tunnel when the client exits.

.EXAMPLE
  .\client\run_client.ps1 -SshTarget weisoon@172.16.163.102 -Task "pick up the red cube"

.EXAMPLE
  # Different server port, robot port and checkpoint; extra args go straight to robot_client.
  .\client\run_client.ps1 -SshTarget gpu-box -RemotePort 8081 -Task "stack the blocks" `
      --robot.port=COM7 --pretrained_name_or_path=me/my-finetuned-molmoact2
#>
param(
    [Parameter(Mandatory)] [string] $SshTarget,   # user@host, or a Host alias from ~/.ssh/config
    [Parameter(Mandatory)] [string] $Task,
    [string] $Config = (Join-Path $PSScriptRoot 'so101_molmoact2.yaml'),
    [int] $LocalPort = 8080,
    [int] $RemotePort = 8080,
    [string] $Python = 'python',
    [Parameter(ValueFromRemainingArguments)] [string[]] $ClientArgs
)
$ErrorActionPreference = 'Stop'

$listening = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($listening) { throw "Local port $LocalPort is already in use; pick another -LocalPort or close the old tunnel." }

Write-Host "Opening tunnel 127.0.0.1:$LocalPort -> $SshTarget 127.0.0.1:$RemotePort"
$tunnel = Start-Process ssh -PassThru -NoNewWindow -ArgumentList @(
    '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
    '-L', "${LocalPort}:127.0.0.1:${RemotePort}", $SshTarget)
try {
    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-NetConnection 127.0.0.1 -Port $LocalPort -InformationLevel Quiet -WarningAction SilentlyContinue)) {
        if ($tunnel.HasExited) { throw "ssh exited (code $($tunnel.ExitCode)); check the SSH target and that the server is running." }
        if ((Get-Date) -gt $deadline) { throw 'Tunnel did not come up within 30 s.' }
        Start-Sleep -Milliseconds 500
    }
    & $Python -m lerobot.async_inference.robot_client "--config_path=$Config" `
        "--server_address=127.0.0.1:$LocalPort" "--task=$Task" @ClientArgs
    if ($LASTEXITCODE) { throw "robot_client exited with code $LASTEXITCODE" }
}
finally {
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id }
    Write-Host 'Tunnel closed.'
}
