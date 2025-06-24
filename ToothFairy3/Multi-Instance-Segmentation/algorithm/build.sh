#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

echo "Building ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."

docker build "$SCRIPTPATH" \
    --platform=linux/amd64 \
    --quiet \
    --tag $DOCKER_TAG

if [ $? -eq 0 ]; then
    echo "Docker image built successfully: $DOCKER_TAG"
else
    echo "Error: Docker build failed with exit code $?"
    exit 1
fi
