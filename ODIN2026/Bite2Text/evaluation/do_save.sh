#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="bite2text-eval-test"

echo "=+= (Re)build the container"
"$SCRIPT_DIR/do_build.sh"

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
  echo "Error: Failed to retrieve build information for $DOCKER_IMAGE_TAG" >&2
  exit 1
fi

formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
image_filename="$SCRIPT_DIR/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
ground_truth_filename="$SCRIPT_DIR/evaluationgroundtruth.tar.gz"

echo "=+= Saving container image to $image_filename"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$image_filename"

echo "=+= Creating ground-truth tarball at $ground_truth_filename"
"$SCRIPT_DIR/create_ground_truth_tarball.sh" "/mnt/c/Users/Kevin/Downloads/Bite2Text" "$ground_truth_filename"

echo "=+= Done"
