#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

echo "Testing ToothFairy3 Multi-Instance-Segmentation algorithm..."

# Create test directories
mkdir -p "$SCRIPTPATH/test/input/images/cbct"
mkdir -p "$SCRIPTPATH/test/output/images/oral-pharyngeal-segmentation"
mkdir -p "$SCRIPTPATH/test/output/metadata"

# Run the Docker container
docker run --rm \
    --memory=8g \
    -v "$SCRIPTPATH/test/input":/input \
    -v "$SCRIPTPATH/test/output":/output \
    $DOCKER_TAG

echo "Test completed. Check output in test/output/"
echo "Instance metadata saved in test/output/metadata/"
