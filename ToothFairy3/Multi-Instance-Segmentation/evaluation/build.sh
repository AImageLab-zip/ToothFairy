#!/bin/bash

# Build script for ToothFairy3 Multi-Instance-Segmentation evaluation container

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="toothfairy3-challenge1"

# Build the docker container
docker build "$SCRIPT_DIR" \
    --platform=linux/amd64 \
    --tag $DOCKER_TAG
