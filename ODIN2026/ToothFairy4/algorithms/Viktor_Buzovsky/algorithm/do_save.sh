#!/usr/bin/env bash
# Produce the two artefacts Grand Challenge wants:
#   * <tag>_<timestamp>.tar.gz  -- the algorithm image
#   * model.tar.gz              -- the model tarball, mounted at /opt/ml/model
#
# The weights live in the model tarball rather than baked into the image, which
# is what keeps the image rebuildable without a 306 MB layer.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOCKER_IMAGE_TAG="toothfairy4-trinity-algorithm"

d() { sg docker -c "docker $*"; }

# vendor/ is a build artefact generated from tf4/. do_save.sh calls docker build
# directly rather than going through do_build.sh, so it has to sync too -- this
# is exactly how the shipped image drifted from tf4/ once already.
"$SCRIPT_DIR/sync_vendor.sh"

echo "=+= (Re)build the image"
export DOCKER_QUIET_BUILD=1
d "build --platform=linux/amd64 --quiet --tag $DOCKER_IMAGE_TAG $SCRIPT_DIR"

build_timestamp=$(d "inspect --format='{{ .Created }}' $DOCKER_IMAGE_TAG" | tr -d "'")
if [ -z "$build_timestamp" ]; then
  echo "Error: could not read build info for $DOCKER_IMAGE_TAG" >&2
  exit 1
fi

formatted=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_path="$SCRIPT_DIR/${DOCKER_IMAGE_TAG}_${formatted}.tar.gz"

echo "=+= Saving image to $output_path"
sg docker -c "docker save $DOCKER_IMAGE_TAG" | gzip -c > "$output_path"

echo "=+= Packing the model"
tar -czf "$SCRIPT_DIR/model.tar.gz" -C "$SCRIPT_DIR/model" .

echo "=+= Done"
ls -lh "$output_path" "$SCRIPT_DIR/model.tar.gz"
