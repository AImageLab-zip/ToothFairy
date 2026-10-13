#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="toothfairy4-example-algorithm"

echo "=+= (Re)build the image"
export DOCKER_QUIET_BUILD=1
"$SCRIPT_DIR/do_build.sh"

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
  echo "Error: Failed to retrieve build information for $DOCKER_IMAGE_TAG" >&2
  exit 1
fi

formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="$SCRIPT_DIR/$output_filename"

echo "=+= Saving the image to $output_path"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"

echo "=+= Packing the model"
tar -czf "$SCRIPT_DIR/model.tar.gz" -C "$SCRIPT_DIR/model" .

echo "=+= Done"
