<#
.SYNOPSIS
  One-time robot PC setup: creates .\venv with Python 3.12+ and installs client\requirements.txt
  (LeRobot with CPU-only torch). run_client.ps1 then uses that venv without activating it.
  Safe to re-run; an existing .\venv is reused.

.EXAMPLE
  .\setup_client.ps1

.EXAMPLE
  # A specific interpreter for creating the venv.
  .\setup_client.ps1 -Python C:\Python312\python.exe
#>
param(
    [string] $Python   # default: the first of python, py -3.13, py -3.12 that is 3.12 or newer
)
$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'

function Test-Python([string] $exe, [string[]] $pyArgs) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $false }
    $ErrorActionPreference = 'Continue'   # py.exe reports a missing version on stderr, which 'Stop' turns into a throw
    & $exe @pyArgs -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path $venvPython)) {
    $found = $null
    if ($Python) {
        if (Test-Python $Python @()) { $found = @($Python) }
    }
    else {
        foreach ($c in @(@('python'), @('py', '-3.13'), @('py', '-3.12'))) {
            if (Test-Python $c[0] @($c | Select-Object -Skip 1)) { $found = $c; break }
        }
    }
    if (-not $found) { throw 'Need Python 3.12 or newer: install it from python.org (or winget install Python.Python.3.12) and re-run.' }
    Write-Host "Creating venv with $($found -join ' ')"
    & $found[0] @($found | Select-Object -Skip 1) -m venv (Join-Path $PSScriptRoot 'venv')
    if ($LASTEXITCODE) { throw "venv creation failed (code $LASTEXITCODE)" }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE) { throw "pip upgrade failed (code $LASTEXITCODE)" }
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'client\requirements.txt')
if ($LASTEXITCODE) { throw "pip install failed (code $LASTEXITCODE)" }

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Warning 'ssh not found on PATH. Install it: Settings > Optional features > OpenSSH Client.'
}
Write-Host @'
Done. For the LeRobot tools (lerobot-find-port, lerobot-find-cameras, lerobot-calibrate), activate the venv first:
  .\venv\Scripts\Activate.ps1
Then start inference with .\run_client.ps1 (see README.md).
'@
