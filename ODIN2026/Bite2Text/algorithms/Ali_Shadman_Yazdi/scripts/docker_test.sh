#!/usr/bin/env bash
# Build the container and run it on one case the way Grand Challenge does:
# no network, read-only /input holding the three sockets, weights at /opt/ml/model.
#
#   ./scripts/docker_test.sh                       # sample_data/F5535
#   ./scripts/docker_test.sh path/to/case          # dir with ios_upper.stl, ios_lower.stl, intraoral_*.jpg
#   GPUS=all ./scripts/docker_test.sh              # needs nvidia-container-toolkit
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
CASE=$(cd -- "${1:-$ROOT/sample_data/F5535}" && pwd)
IMAGE=${IMAGE:-bite2text-odin2026}
OUT="$ROOT/outputs/docker/$(basename "$CASE")"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=+= building $IMAGE"
docker build --platform=linux/amd64 --tag "$IMAGE" "$ROOT"

mkdir -p "$WORK/input/files/ios-upper" "$WORK/input/files/ios-lower" "$WORK/input/images/intraoral-photo"
cp "$CASE/ios_upper.stl" "$WORK/input/files/ios-upper/"
cp "$CASE/ios_lower.stl" "$WORK/input/files/ios-lower/"
find "$CASE" -maxdepth 1 -iname 'intraoral_*' -exec cp {} "$WORK/input/images/intraoral-photo/" \;
cat > "$WORK/input/inputs.json" <<'EOF'
[
  {"socket": {"slug": "3d-lower-teeth-scan", "relative_path": "files/ios-lower", "is_file_kind": true}},
  {"socket": {"slug": "3d-upper-teeth-scan", "relative_path": "files/ios-upper", "is_file_kind": true}},
  {"socket": {"slug": "2d-intraoral-photographs", "relative_path": "images/intraoral-photo", "is_image_kind": true}}
]
EOF
chmod -R a+rX "$WORK/input"
# the container runs as an unprivileged user with a different uid
mkdir -m 777 "$WORK/output"

echo "=+= running on $(basename "$CASE")"
docker run --rm --platform=linux/amd64 --network none ${GPUS:+--gpus "$GPUS"} \
  --volume "$WORK/input":/input:ro \
  --volume "$WORK/output":/output \
  --volume "$ROOT/weights":/opt/ml/model:ro \
  "$IMAGE"

mkdir -p "$OUT"
cp "$WORK/output/diagnostic-imaging-report.json" "$OUT/"
echo "=+= output: $OUT/diagnostic-imaging-report.json"
cat "$OUT/diagnostic-imaging-report.json"
