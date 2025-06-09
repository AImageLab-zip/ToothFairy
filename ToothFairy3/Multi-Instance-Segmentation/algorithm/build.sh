#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Building ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."
docker build --platform=linux/amd64 -t toothfairy3-multiinstance-algorithm "$SCRIPTPATH"
