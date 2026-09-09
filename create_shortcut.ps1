# Creates a desktop shortcut for the Newmeta Backtester.

$ErrorActionPreference = "Stop"

# Resolve target — this script lives in backtest_harness/
$scriptPath = $MyInvocation.MyCommand.Path
if (-not $scriptPath) {
    $scriptPath = $PSCommandPath
}
$projectRoot = Split-Path -Parent $scriptPath
$batPath = Join-Path $projectRoot "launch_backtester.bat"

if (-not (Test-Path $batPath)) {
    Write-Host "ERROR: launch_backtester.bat not found at $batPath" -ForegroundColor Red
    exit 1
}

# Shortcut path
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "Newmeta Backtester.lnk"

# Create the shortcut
$ws = New-Object -ComObject WScript.Shell
$shortcut = $ws.CreateShortcut($lnkPath)
$shortcut.TargetPath = $batPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "shell32.dll,238"
$shortcut.WindowStyle = 1
$shortcut.Description = "Newmeta Backtester - drop .mq5/.py/.pine/.txt and get pro backtest"
$shortcut.Save()

Write-Host "Shortcut created at: $lnkPath" -ForegroundColor Green
Write-Host "Double-click it to launch the backtester." -ForegroundColor Cyan