#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="odin2026-bite2text-ptv3-debug-v3"

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_path="$SCRIPT_DIR/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

echo "=+= Saving image to $output_path"
docker save "$DOCKER_IMAGE_TAG" | gzip -1 -c > "$output_path"
if [[ ! -f "$SCRIPT_DIR/model.tar.gz" ]]; then
  echo "=+= Saving model resource"
  tar -czf "$SCRIPT_DIR/model.tar.gz" -C "$SCRIPT_DIR/model" .
else
  echo "=+= Reusing existing model resource (v2 does not change model weights)"
fi
sha256sum "$output_path" "$SCRIPT_DIR/model.tar.gz"
echo "IMAGE_ARCHIVE=$output_path"
