$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOCKER_TAG = "toothfairy2-example-algorithm"

docker build $SCRIPT_DIR `
    --platform=linux/amd64 `
    --quiet `
    --tag $DOCKER_TAG
