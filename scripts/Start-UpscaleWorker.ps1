[CmdletBinding()]
param(
    [ValidateRange(1, 300)]
    [int]$InitialRestartDelaySeconds = 5,

    [ValidateRange(1, 900)]
    [int]$MaximumRestartDelaySeconds = 60,

    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cliPath = (Resolve-Path (Join-Path $projectRoot "cli.py")).Path
$logDirectory = Join-Path $projectRoot "remote-worker-data\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$launcherLog = Join-Path $logDirectory "launcher.log"
$workerLog = Join-Path $logDirectory "worker.log"

# Stopping a scheduled PowerShell task can leave the Python/uv child process
# orphaned. Remove only workers whose command line points at this repository's
# exact cli.py before starting the replacement pool.
$escapedCliPath = [Regex]::Escape($cliPath)
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match $escapedCliPath -and
        $_.CommandLine -match '(?:^|\s)upscale-worker(?:\s|$)'
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

# Task Scheduler can retain an older environment block after variables are
# changed. Reload the worker's user-scoped settings without persisting or
# logging their values.
foreach ($name in @(
    "IMAGE_CLOUD_BASE_URL",
    "IMAGE_UPSCALE_WORKER_TOKEN",
    "IMAGE_UPSCALE_WORKER_ID",
    "IMAGE_UPSCALE_WORKER_CONCURRENCY",
    "IMAGE_UPSCALE_POLL_SECONDS",
    "IMAGE_UPSCALE_WORK_ROOT"
)) {
    $value = [Environment]::GetEnvironmentVariable($name, "User")
    if ($null -ne $value) {
        Set-Item -LiteralPath "Env:$name" -Value $value
    }
}

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ssK"
    Add-Content -LiteralPath $launcherLog -Value "[$timestamp] $Message" -Encoding UTF8
}

if ($PythonPath) {
    $pythonExe = (Resolve-Path -LiteralPath $PythonPath -ErrorAction Stop).Path
} else {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $pythonExe = $venvPython
    } else {
        $pythonCommand = Get-Command "python.exe" -ErrorAction Stop
        $pythonExe = $pythonCommand.Source
    }
}

$delaySeconds = $InitialRestartDelaySeconds
Write-LauncherLog "launcher started; python=$pythonExe"

while ($true) {
    $startedAt = Get-Date
    $exitCode = -1
    Write-LauncherLog "starting remote upscale worker"

    Push-Location $projectRoot
    try {
        & $pythonExe $cliPath upscale-worker *>> $workerLog
        $exitCode = $LASTEXITCODE
    } catch {
        Write-LauncherLog "worker launch error: $($_.Exception.GetType().Name): $($_.Exception.Message)"
    } finally {
        Pop-Location
    }

    $runtimeSeconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
    Write-LauncherLog "worker stopped; exit_code=$exitCode; runtime_seconds=$runtimeSeconds; restart_in_seconds=$delaySeconds"
    if ($runtimeSeconds -ge 300) {
        $delaySeconds = $InitialRestartDelaySeconds
    }

    Start-Sleep -Seconds $delaySeconds
    $delaySeconds = [Math]::Min($MaximumRestartDelaySeconds, $delaySeconds * 2)
}
