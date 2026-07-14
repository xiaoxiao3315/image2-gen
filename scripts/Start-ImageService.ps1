[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$ListenAddress = "0.0.0.0",

    [ValidateRange(1, 65535)]
    [int]$Port = 8012,

    [ValidateRange(1, 300)]
    [int]$InitialRestartDelaySeconds = 5,

    [ValidateRange(1, 900)]
    [int]$MaximumRestartDelaySeconds = 60,

    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDirectory = Join-Path $projectRoot "service-data\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$launcherLog = Join-Path $logDirectory "launcher.log"
$serviceLog = Join-Path $logDirectory "service.log"

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
Write-LauncherLog "launcher started; listen=$ListenAddress`:$Port; python=$pythonExe"

while ($true) {
    $startedAt = Get-Date
    $exitCode = -1
    Write-LauncherLog "starting image service"

    Push-Location $projectRoot
    try {
        & $pythonExe (Join-Path $projectRoot "cli.py") serve --host $ListenAddress --port $Port *>> $serviceLog
        $exitCode = $LASTEXITCODE
    } catch {
        Write-LauncherLog "service launch error: $($_.Exception.GetType().Name): $($_.Exception.Message)"
    } finally {
        Pop-Location
    }

    $runtimeSeconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
    Write-LauncherLog "service stopped; exit_code=$exitCode; runtime_seconds=$runtimeSeconds; restart_in_seconds=$delaySeconds"

    # A service that stayed healthy for five minutes gets the shortest retry delay again.
    if ($runtimeSeconds -ge 300) {
        $delaySeconds = $InitialRestartDelaySeconds
    }

    Start-Sleep -Seconds $delaySeconds
    $delaySeconds = [Math]::Min($MaximumRestartDelaySeconds, $delaySeconds * 2)
}
