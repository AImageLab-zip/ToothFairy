$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multiinstance-algorithm"

Write-Host "Building ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."

docker build $SCRIPT_DIR `
    --platform=linux/amd64 `
    --quiet `
    --tag $DOCKER_TAG

if ($LASTEXITCODE -eq 0) {
    Write-Host "Docker image built successfully: $DOCKER_TAG"
} else {
    Write-Error "Docker build failed with exit code $LASTEXITCODE"
}
