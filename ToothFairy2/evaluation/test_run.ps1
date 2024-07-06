# Stop at first error
$ErrorActionPreference = "Stop"

# Get the script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "example-evaluation-test-phase"

$INPUT_DIR = Join-Path $SCRIPT_DIR "test/input"
$OUTPUT_DIR = Join-Path $SCRIPT_DIR "test/output"

Write-Host "=+= Cleaning up any earlier output"
if (Test-Path $OUTPUT_DIR) {
    # Ensure permissions are setup correctly
    # This allows for the Docker user to write to this location
    Get-ChildItem -Path $OUTPUT_DIR -Recurse | Remove-Item -Force
    icacls $OUTPUT_DIR /grant '*S-1-1-0:(OI)(CI)F'
} else {
    New-Item -Path $OUTPUT_DIR -ItemType Directory
    icacls $OUTPUT_DIR /grant '*S-1-1-0:(OI)(CI)F'
}

Write-Host "=+= (Re)build the container"
docker build $SCRIPT_DIR `
    --platform=linux/amd64 `
    --quiet `
    --tag $DOCKER_TAG

Write-Host "=+= Doing an evaluation"
docker run --rm `
    --platform=linux/amd64 `
    --network none `
    --gpus all `
    --volume "${INPUT_DIR}:/input" `
    --volume "${OUTPUT_DIR}:/output" `
    $DOCKER_TAG

# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
$HOST_UID = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$HOST_GID = (New-Object System.Security.Principal.NTAccount((New-Object System.Security.Principal.WindowsIdentity]).User.Value)).Translate([System.Security.Principal.SecurityIdentifier]).Value

docker run --rm `
    --quiet `
    --env HOST_UID=$HOST_UID `
    --env HOST_GID=$HOST_GID `
    --volume "${OUTPUT_DIR}:/output" `
    alpine:latest `
    /bin/sh -c "chown -R ${HOST_UID}:${HOST_GID} /output"

Write-Host "=+= Wrote results to ${OUTPUT_DIR}"

