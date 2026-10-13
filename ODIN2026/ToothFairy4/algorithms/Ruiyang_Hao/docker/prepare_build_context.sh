#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTEXT_DIR="${DOCKER_CONTEXT:-$SCRIPT_DIR/context}"
UMAMBA2_DIR="${UMAMBA2_DIR:-$REPO_ROOT/third_party/U-Mamba2}"
WHEEL_DIR="${MAMBA_WHEEL_DIR:-$SCRIPT_DIR/wheels}"
MAMBA_WHEEL="$WHEEL_DIR/mamba_ssm-2.2.5-cp312-cp312-linux_x86_64.whl"
CAUSAL_WHEEL="$WHEEL_DIR/causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl"

for path in "$UMAMBA2_DIR/pyproject.toml" "$MAMBA_WHEEL" "$CAUSAL_WHEEL"; do
  [[ -e "$path" ]] || { echo "Missing Docker dependency: $path" >&2; exit 2; }
done

rm -rf "$CONTEXT_DIR"
mkdir -p "$CONTEXT_DIR/EviDent-CBCT" "$CONTEXT_DIR/U-Mamba2" "$CONTEXT_DIR/container" "$CONTEXT_DIR/wheels"

cp -a "$REPO_ROOT/evigre" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/setup.py" \
  "$REPO_ROOT/README.md" "$CONTEXT_DIR/EviDent-CBCT/"
cp -a "$UMAMBA2_DIR/." "$CONTEXT_DIR/U-Mamba2/"
rm -rf "$CONTEXT_DIR/U-Mamba2/.git"
cp -a "$SCRIPT_DIR/requirements-runtime.txt" "$CONTEXT_DIR/container/"
cp -a "$SCRIPT_DIR/entrypoint.sh" "$CONTEXT_DIR/container/"
cp -a "$MAMBA_WHEEL" "$CAUSAL_WHEEL" "$CONTEXT_DIR/wheels/"

echo "Docker context ready: $CONTEXT_DIR"
echo "No model artifact was copied."

