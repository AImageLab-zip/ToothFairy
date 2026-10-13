#!/usr/bin/env bash
# Copy the pipeline modules into vendor/, which is what the image ships.
#
# vendor/ is gitignored and was previously maintained by hand. It drifted: the
# shipped image was still emitting "Presence of prosthetic crowns." long after
# tf4/ had moved to per-site crown statements. tf4/ is the single source of
# truth; vendor/ is a build artefact and do_build.sh regenerates it every time.
#
# model.py and utils/ come from Trinity-Segment and are copied from
# third_party/trinity/, so a fresh clone builds without any external checkout.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SRC="$SCRIPT_DIR/../tf4"
DST="$SCRIPT_DIR/vendor"

mkdir -p "$DST"
for module in schema render mask_features trinity_infer; do
  cp "$SRC/$module.py" "$DST/$module.py"
done
TRINITY="$SCRIPT_DIR/../third_party/trinity"
cp "$TRINITY/model.py" "$DST/model.py"
mkdir -p "$DST/utils"
cp "$TRINITY/utils/"*.py "$DST/utils/"
touch "$DST/__init__.py"
find "$DST" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "=+= vendor/ synced: schema render mask_features trinity_infer (tf4/), model utils (third_party/trinity/)"
