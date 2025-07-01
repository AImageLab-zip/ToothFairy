# Test script for ToothFairy3 Multi-Instance-Segmentation evaluation container

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy3-multi-instance-evaluation"

# Define paths
$TEST_DATA_DIR = "C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-data\ToothFairy3"
$LABELS_DIR = Join-Path $TEST_DATA_DIR "labelsTr"
$RESULTS_DIR = "C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-results"
$ALGORITHM_OUTPUT = Join-Path $RESULTS_DIR "algorithm-output"
$EVALUATION_OUTPUT = Join-Path $RESULTS_DIR "evaluation-output"

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

Write-Host "Running ToothFairy3 Multi-Instance-Segmentation evaluation..."

# Cleanup function to fix permissions after container run
function Cleanup-Permissions {
    Write-Host "Cleaning up permissions..."
    # Ensure permissions are set correctly on the output
    # This allows the host user to access and handle these files
    & $DOCKER_EXE run --rm `
        --platform=linux/amd64 `
        --quiet `
        --volume "${EVALUATION_OUTPUT}:/output" `
        --entrypoint /bin/sh `
        $DOCKER_TAG `
        -c "chmod -R -f o+rwX /output/* || true"
}

# Create evaluation output directory
if (-Not (Test-Path $EVALUATION_OUTPUT)) {
    New-Item -ItemType Directory -Path $EVALUATION_OUTPUT -Force | Out-Null
}

# Check if required directories exist
if (-Not (Test-Path $ALGORITHM_OUTPUT)) {
    Write-Error "Algorithm output directory not found: $ALGORITHM_OUTPUT"
    Write-Host "Please run the algorithm test script first."
    exit 1
}

# Extract ground truth tarball
$TARBALL_PATH = Join-Path (Split-Path $SCRIPT_DIR -Parent) "evaluationgroundtruth.tar.gz"
$GT_EXTRACT_DIR = Join-Path $EVALUATION_OUTPUT "ground_truth"

if (-Not (Test-Path $TARBALL_PATH)) {
    Write-Error "Ground truth tarball not found: $TARBALL_PATH"
    exit 1
}

Write-Host "Extracting ground truth tarball..."
if (Test-Path $GT_EXTRACT_DIR) {
    Remove-Item -Recurse -Force $GT_EXTRACT_DIR
}
New-Item -ItemType Directory -Path $GT_EXTRACT_DIR -Force | Out-Null

# Extract tarball using tar command (available in Windows 10+)
tar -xzf "$TARBALL_PATH" -C "$GT_EXTRACT_DIR"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to extract ground truth tarball"
    exit 1
}

# Set permissions to allow Docker user to read input directories
Write-Host "Setting permissions for Docker access..."
icacls "$ALGORITHM_OUTPUT" /grant Everyone:R /T /Q
icacls "$GT_EXTRACT_DIR" /grant Everyone:R /T /Q

# Set permissions to allow Docker user to write to output directory
icacls "$EVALUATION_OUTPUT" /grant Everyone:F /T /Q

# Run the Docker container
Write-Host "Starting evaluation container..."
& $DOCKER_EXE run --rm `
    --platform=linux/amd64 `
    --shm-size=128m `
    --memory=4g `
    -v "${ALGORITHM_OUTPUT}:/input/:ro" `
    -v "${GT_EXTRACT_DIR}:/opt/ml/input/data/ground_truth/:ro" `
    -v "${EVALUATION_OUTPUT}:/output/" `
    $DOCKER_TAG

# Call cleanup function to fix permissions
Cleanup-Permissions

if ($LASTEXITCODE -eq 0) {
    Write-Host "Evaluation completed successfully!"
    $METRICS_FILE = Join-Path $EVALUATION_OUTPUT "metrics.json"
    if (Test-Path $METRICS_FILE) {
        Write-Host "Results saved to: $METRICS_FILE"
        Write-Host ""
        Write-Host "=== Evaluation Results Preview ==="
        try {
            $metrics = Get-Content $METRICS_FILE | ConvertFrom-Json
            if ($metrics.aggregates -and $metrics.aggregates.DiceCoefficient_Average_All) {
                Write-Host "Average Dice Coefficient: $($metrics.aggregates.DiceCoefficient_Average_All.mean)"
            }
            if ($metrics.aggregates -and $metrics.aggregates.HausdorffDistance95_Average_All) {
                Write-Host "Average HD95: $($metrics.aggregates.HausdorffDistance95_Average_All.mean)"
            }
        }
        catch {
            Write-Host "Metrics file created but couldn't parse preview: $_"
        }
        Write-Host "==================================="
    } else {
        Write-Warning "Metrics file not found at expected location: $METRICS_FILE"
    }
} else {
    Write-Error "Evaluation failed with exit code: $LASTEXITCODE"
    exit 1
}
