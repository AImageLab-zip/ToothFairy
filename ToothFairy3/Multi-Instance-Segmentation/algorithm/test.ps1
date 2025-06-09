$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multiinstance-algorithm"

Write-Host "Testing ToothFairy3 Multi-Instance-Segmentation algorithm..."

# Create test directories
$InputDir = Join-Path $SCRIPT_DIR "test\input\images\cbct"
$OutputDir = Join-Path $SCRIPT_DIR "test\output\images\oral-pharyngeal-segmentation"
$MetadataDir = Join-Path $SCRIPT_DIR "test\output\metadata"

New-Item -ItemType Directory -Path $InputDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $MetadataDir -Force | Out-Null

# Run the Docker container
docker run --rm `
    --memory=8g `
    -v "${SCRIPT_DIR}\test\input:/input" `
    -v "${SCRIPT_DIR}\test\output:/output" `
    $DOCKER_TAG

if ($LASTEXITCODE -eq 0) {
    Write-Host "Test completed successfully. Check output in test\output\"
    Write-Host "Instance metadata saved in test\output\metadata\"
} else {
    Write-Error "Test failed with exit code $LASTEXITCODE"
}
