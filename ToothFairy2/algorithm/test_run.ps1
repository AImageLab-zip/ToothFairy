$ErrorActionPreference = "Stop"
$DOCKER_TAG = "toothfairy2-example-algorithm"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$INPUT_DIR = Join-Path $SCRIPT_DIR "test/input"
$OUTPUT_DIR = Join-Path $SCRIPT_DIR "test/output"

.\build.ps1

docker run --rm `
    --platform=linux/amd64 `
    --network none `
    --gpus all `
    --volume "${INPUT_DIR}:/input" `
    --volume "${OUTPUT_DIR}:/output" `
    $DOCKER_TAG
