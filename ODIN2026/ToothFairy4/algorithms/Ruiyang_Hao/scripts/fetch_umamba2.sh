#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${UMAMBA2_DIR:-$ROOT/third_party/U-Mamba2}"
URL="${UMAMBA2_URL:-https://github.com/zhiqin1998/U-Mamba2.git}"
REVISION="${UMAMBA2_REVISION:-2046d29785087b656ca69fa02dd40e43e69cfb42}"

if [[ -e "$DEST" && ! -d "$DEST/.git" ]]; then
  echo "Refusing to replace non-Git path: $DEST" >&2
  exit 2
fi
if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
fi

git -C "$DEST" fetch --depth 1 origin "$REVISION"
git -C "$DEST" checkout --detach "$REVISION"

echo "U-Mamba2 ready at $DEST"
echo "Revision: $(git -C "$DEST" rev-parse HEAD)"
echo "License:  $DEST/LICENSE (CC BY-NC 4.0)"

