#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="example-evaluation-test-phase"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"


echo "=+= Cleaning up any earlier output"
if [ -d "$OUTPUT_DIR" ]; then
  # Ensure permissions are setup correctly
  # This allows for the Docker user to write to this location
  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir --mode=o+rwx "$OUTPUT_DIR"
fi


echo "=+= (Re)build the container"
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --quiet \
  --tag $DOCKER_TAG


echo "=+= Doing an evaluation"
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "${INPUT_DIR}":/input \
    --volume "${OUTPUT_DIR}":/output \
    $DOCKER_TAG


# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
docker run --rm \
    --quiet \
    --env HOST_UID=`id --user` \
    --env HOST_GID=`id --group` \
    --volume ${OUTPUT_DIR}:/output \
    alpine:latest \
    /bin/sh -c 'chown -R ${HOST_UID}:${HOST_GID} /output'

echo "=+= Wrote results to ${OUTPUT_DIR}"
