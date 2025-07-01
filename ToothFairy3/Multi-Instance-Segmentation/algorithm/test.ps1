$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multiinstance-algorithm"

Write-Host "Testing ToothFairy3 Multi-Instance-Segmentation algorithm..."

# Define paths
$TEST_DATA_DIR = "C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-data\ToothFairy3"
$IMAGES_DIR = Join-Path $TEST_DATA_DIR "imagesTr"
$RESULTS_DIR = "C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-results"
$ALGORITHM_OUTPUT = Join-Path $RESULTS_DIR "algorithm-output"

# Create test directories
$InputDir = Join-Path $SCRIPT_DIR "test\input\images\cbct"
$OutputDir = Join-Path $SCRIPT_DIR "test\output\images\oral-pharyngeal-segmentation"
$MetadataDir = Join-Path $SCRIPT_DIR "test\output\metadata"

New-Item -ItemType Directory -Path $InputDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $MetadataDir -Force | Out-Null
New-Item -ItemType Directory -Path $ALGORITHM_OUTPUT -Force | Out-Null

# Run the Docker container
Write-Host "Running algorithm Docker container..."
docker run --rm `
    --memory=8g `
    -v "${SCRIPT_DIR}\test\input:/input" `
    -v "${SCRIPT_DIR}\test\output:/output" `
    $DOCKER_TAG

# Copy results to shared location for evaluation
Write-Host "Copying results to shared location: $ALGORITHM_OUTPUT"
Copy-Item -Path "${SCRIPT_DIR}\test\output\*" -Destination $ALGORITHM_OUTPUT -Recurse -Force

if ($LASTEXITCODE -eq 0) {
    Write-Host "Algorithm test completed successfully!"
    Write-Host "Results copied to: $ALGORITHM_OUTPUT"
    Write-Host "Ready for evaluation step."
} else {
    Write-Error "Algorithm test failed with exit code $LASTEXITCODE"
}
