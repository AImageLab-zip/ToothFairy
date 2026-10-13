#!/usr/bin/env bash
# Produce the two Grand Challenge upload artefacts:
#   outputs/submission/bite2text-odin2026_<timestamp>.tar.gz   container image
#   outputs/submission/model.tar.gz                             contents of weights/ -> /opt/ml/model
# Upload both, and re-upload the model tarball whenever weights change.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
IMAGE=${IMAGE:-bite2text-odin2026}
DEST="$ROOT/outputs/submission"
mkdir -p "$DEST"

python "$ROOT/scripts/download_weights.py" --check

docker build --platform=linux/amd64 --tag "$IMAGE" "$ROOT"
stamp=$(date -u +%Y-%m-%d_%H-%M-%S)
echo "=+= saving image"
docker save "$IMAGE" | gzip -c > "$DEST/${IMAGE}_${stamp}.tar.gz"

echo "=+= packing weights"
tar -czhf "$DEST/model.tar.gz" -C "$ROOT/weights" --exclude='*.part' --exclude='README.md' .
tar -tzf "$DEST/model.tar.gz"
echo "=+= done: $DEST"
