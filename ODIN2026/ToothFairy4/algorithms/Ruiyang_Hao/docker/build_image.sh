#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${1:-evident-cbct:latest}"
"$SCRIPT_DIR/prepare_build_context.sh"
docker build --platform linux/amd64 --progress plain \
  --file "$SCRIPT_DIR/Dockerfile" --tag "$IMAGE_NAME" "$SCRIPT_DIR/context"
echo "Built Docker image: $IMAGE_NAME"

