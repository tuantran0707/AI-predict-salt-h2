# Copy updated code to Raspberry Pi (run from Windows PowerShell in this folder).
# Usage: .\sync_to_pi.ps1
#        .\sync_to_pi.ps1 -PiHost 192.168.1.50

param(
    [string]$PiHost = "192.168.137.84",
    [string]$PiUser = "pi",
    [string]$RemoteDir = "~/Desktop/project/AI-predict-salt-h2"
)

$files = @(
    "version.py",
    "detect_salt.py",
    "run_camera.py",
    "thingsboard_mqtt.py",
    "thingsboard_service.py",
    "camera.py",
    "prototypes.npz"
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$paths = $files | ForEach-Object { Join-Path $here $_ } | Where-Object { Test-Path $_ }

Write-Host "Copying $($paths.Count) files to ${PiUser}@${PiHost}:${RemoteDir}/"
scp @paths "${PiUser}@${PiHost}:${RemoteDir}/"
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK. On Pi run:"
    Write-Host "  cd ~/Desktop/project/AI-predict-salt-h2"
    Write-Host "  bash check_pi_code.sh"
    Write-Host "  python3 thingsboard_service.py --token <TOKEN>"
}
