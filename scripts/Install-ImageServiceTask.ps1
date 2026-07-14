[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$TaskName = "Image2GenService",

    [ValidatePattern("^[A-Za-z0-9.:-]+$")]
    [string]$ListenAddress = "0.0.0.0",

    [ValidateRange(1, 65535)]
    [int]$Port = 8012,

    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$startScript = (Resolve-Path (Join-Path $PSScriptRoot "Start-ImageService.ps1")).Path
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name

$actionArguments = @(
    "-NoProfile"
    "-NonInteractive"
    "-ExecutionPolicy Bypass"
    "-WindowStyle Hidden"
    "-File `"$startScript`""
    "-ListenAddress `"$ListenAddress`""
    "-Port $Port"
)
if ($PythonPath) {
    $resolvedPython = (Resolve-Path -LiteralPath $PythonPath -ErrorAction Stop).Path
    $actionArguments += "-PythonPath `"$resolvedPython`""
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument ($actionArguments -join " ") `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Image2Gen FastAPI service (hidden, auto-start and auto-restart)" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Scheduled task '$TaskName' installed and started for $currentUser."
Write-Host "Health check: http://127.0.0.1:$Port/health"
Write-Host "Logs: $projectRoot\service-data\logs"
