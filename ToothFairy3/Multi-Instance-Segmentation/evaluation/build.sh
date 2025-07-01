#!/bin/bash

# Build script for ToothFairy3 Multi-Instance-Segmentation evaluation container

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="toothfairy3-multi-instance-evaluation"

echo "Building ToothFairy3 Multi-Instance-Segmentation evaluation container..."

# Build the docker container
docker build "$SCRIPT_DIR" \
    --platform=linux/amd64 \
    --tag $DOCKER_TAG

if [ $? -eq 0 ]; then
    echo "Successfully built Docker image: $DOCKER_TAG"
else
    echo "Error: Failed to build Docker image"
    exit 1
fi
