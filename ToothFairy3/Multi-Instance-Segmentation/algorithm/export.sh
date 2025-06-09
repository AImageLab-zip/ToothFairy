#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Exporting ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."
docker save toothfairy3-multiinstance-algorithm | gzip -c > toothfairy3-multiinstance-algorithm.tar.gz
echo "Docker image exported to: toothfairy3-multiinstance-algorithm.tar.gz"
