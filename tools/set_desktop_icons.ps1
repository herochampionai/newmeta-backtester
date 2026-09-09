# Sets custom icons on desktop folders using .folder_icons/*.ico

$ErrorActionPreference = "Continue"

$iconDir = "C:\Users\youha\Desktop\.folder_icons"
$desktop = "C:\Users\youha\Desktop"

# Map folder names -> icon file (matched to actual desktop folders)
$map = @{
    "APKs"                  = "apks.ico"
    "App Shortcuts"         = "app_shortcuts.ico"
    "Docs and Logs"         = "docs_and_logs.ico"
    "Media and Design"      = "media_and_design.ico"
    "Newmeta Hub"           = "trading.ico"
    "pika-poke"             = "media_and_design.ico"
    "Projects"              = "projects.ico"
    "Scripts and Setup"     = "scripts_and_setup.ico"
    "Temp and Cleanup"      = "temp_and_cleanup.ico"
    "Trading"                = "trading.ico"
    "VPS and Servers"        = "vps_and_servers.ico"
}

$applied = 0
$skipped = 0

foreach ($folderName in $map.Keys) {
    $folderPath = Join-Path $desktop $folderName
    if (-not (Test-Path $folderPath)) {
        Write-Host "  [SKIP] $folderName (not found)"
        $skipped++
        continue
    }
    $iconFile = Join-Path $iconDir $map[$folderName]
    if (-not (Test-Path $iconFile)) {
        Write-Host "  [MISS] $folderName -> icon file missing"
        $skipped++
        continue
    }

    try {
        # Clear read-only / system / hidden so we can write
        attrib -r -s -h $folderPath 2>&1 | Out-Null

        # Copy the .ico into the folder as folder.ico
        $localIco = Join-Path $folderPath "folder.ico"
        Copy-Item $iconFile $localIco -Force

        # Remove any existing desktop.ini
        $iniPath = Join-Path $folderPath "desktop.ini"
        if (Test-Path $iniPath) { Remove-Item $iniPath -Force -ErrorAction SilentlyContinue }

        # Write desktop.ini using cmd.exe (avoids PowerShell file locks)
        $iniContent = "[.ShellClassInfo]`r`nIconResource=folder.ico,0`r`n"
        $tmpIni = [System.IO.Path]::GetTempFileName()
        [System.IO.File]::WriteAllText($tmpIni, $iniContent, [System.Text.Encoding]::ASCII)
        Copy-Item $tmpIni $iniPath -Force
        Remove-Item $tmpIni -Force

        # Mark folder + desktop.ini as system + read-only so Windows recognizes it
        attrib +r +s $folderPath 2>&1 | Out-Null
        attrib +r +s +h $iniPath 2>&1 | Out-Null

        Write-Host "  [OK]   $folderName -> $($map[$folderName])"
        $applied++
    } catch {
        Write-Host "  [FAIL] $folderName -> $($_.Exception.Message)"
        $skipped++
    }
}

# Refresh the desktop shell so icons update immediately
try {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Shell32 {
    [DllImport("Shell32.dll")]
    public static extern void SHChangeNotify(int wEventId, int uFlags, IntPtr dwItem1, IntPtr dwItem2);
}
"@
    [Shell32]::SHChangeNotify(0x08000000, 0x1000, [IntPtr]::Zero, [IntPtr]::Zero)
} catch {}

Write-Host ""
Write-Host "Done. Applied: $applied  Skipped: $skipped"