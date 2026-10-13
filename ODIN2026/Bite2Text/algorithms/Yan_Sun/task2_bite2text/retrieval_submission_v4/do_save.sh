#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="odin2026-bite2text-retrieval-debug-v4"
ARTIFACT_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
image_path="$ARTIFACT_DIR/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
model_path="$ARTIFACT_DIR/model-retrieval-v4.tar.gz"

echo "=+= Saving image to $image_path"
docker save "$DOCKER_IMAGE_TAG" | gzip -1 -c > "$image_path"
echo "=+= Saving model resource to $model_path"
tar -czf "$model_path" -C "$SCRIPT_DIR/model" .
sha256sum "$image_path" "$model_path"
echo "IMAGE_ARCHIVE=$image_path"
echo "MODEL_ARCHIVE=$model_path"

