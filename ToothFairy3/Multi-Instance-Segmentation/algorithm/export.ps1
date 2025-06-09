$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multiinstance-algorithm"

Write-Host "Exporting ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."

docker save $DOCKER_TAG | gzip > "${DOCKER_TAG}.tar.gz"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Docker image exported to: ${DOCKER_TAG}.tar.gz"
} else {
    Write-Error "Export failed with exit code $LASTEXITCODE"
}
