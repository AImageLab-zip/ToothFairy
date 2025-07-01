# Build script for ToothFairy3 Multi-Instance-Segmentation evaluation container

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multi-instance-evaluation"

# Find Docker executable
$DOCKER_EXE = "docker"
if (-Not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $DOCKER_PATH = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (Test-Path $DOCKER_PATH) {
        $DOCKER_EXE = $DOCKER_PATH
        Write-Host "Using Docker at: $DOCKER_PATH"
    } else {
        Write-Error "Docker not found. Please ensure Docker Desktop is installed and running."
        exit 1
    }
}

Write-Host "Building ToothFairy3 Multi-Instance-Segmentation evaluation container..."

# Build the docker container
& $DOCKER_EXE build $SCRIPT_DIR `
    --platform=linux/amd64 `
    --tag $DOCKER_TAG

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully built Docker image: $DOCKER_TAG"
} else {
    Write-Error "Failed to build Docker image"
    exit 1
}
