# PowerShell Script to Create a Windows Desktop Shortcut for SignFlow Studio
$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)

$WorkspaceDir = $PSScriptRoot
$ExePath = Join-Path $WorkspaceDir "dist\SignFlowStudio\SignFlowStudio.exe"
$ShortcutPath = Join-Path $DesktopPath "SignFlow Studio.lnk"
$IconPath = Join-Path $WorkspaceDir "app\resources\app_icon.ico"

# If packaged exe does not exist, point to python launcher
if (Test-Path $ExePath) {
    $Target = $ExePath
    $Arguments = ""
} else {
    $Target = "pythonw.exe"
    $Arguments = "`"$WorkspaceDir\run_app.py`""
}

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Target
$Shortcut.Arguments = $Arguments
$Shortcut.WorkingDirectory = $WorkspaceDir
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = "$IconPath, 0"
}
$Shortcut.Description = "SignFlow Studio - Sign Language Recognition Desktop Application"
$Shortcut.Save()

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host " Shortcut Path: $ShortcutPath" -ForegroundColor Cyan
Write-Host " Target Path:   $Target $Arguments" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
