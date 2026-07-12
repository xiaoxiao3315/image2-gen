[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$TaskName = "Image2GenUpscaleWorker",

    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$startScript = (Resolve-Path (Join-Path $PSScriptRoot "Start-UpscaleWorker.ps1")).Path
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name

$actionArguments = @(
    "-NoProfile"
    "-NonInteractive"
    "-ExecutionPolicy Bypass"
    "-WindowStyle Hidden"
    "-File `"$startScript`""
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
    -Description "Image2Gen local Real-ESRGAN worker (hidden, auto-start and auto-restart)" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Scheduled task '$TaskName' installed and started for $currentUser."
Write-Host "Logs: $projectRoot\remote-worker-data\logs"
