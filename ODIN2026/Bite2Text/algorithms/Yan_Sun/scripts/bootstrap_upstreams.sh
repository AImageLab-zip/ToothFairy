#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=${BITE2TEXT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
VENDOR_ROOT=${BITE2TEXT_VENDOR_ROOT:-${PROJECT_ROOT}/.vendor}
PTV3_CONTEXT=${PROJECT_ROOT}/task2_bite2text/ptv3_submission

BITS2BITES_URL=https://github.com/AImageLab-zip/Bits2Bites.git
BITS2BITES_COMMIT=8c3c685160c9cabe2462e9e23d2ffcd9ca78c63a
IOS_NORMALIZER_URL=https://github.com/AImageLab-zip/IOS-Normalizer.git
IOS_NORMALIZER_COMMIT=ecebe110a15081ea435e5970bbe6cf472d8f2882

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

checkout_pinned() {
  local url=$1
  local commit=$2
  local destination=$3
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  local current_commit
  current_commit=$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)
  if [[ "${current_commit}" == "${commit}" && -e "${destination}/README.md" ]]; then
    return
  fi
  if [[ "${current_commit}" == "${commit}" && ! -e "${destination}/README.md" ]]; then
    echo "Incomplete upstream checkout: ${destination}; use a new BITE2TEXT_VENDOR_ROOT." >&2
    exit 1
  fi
  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    echo "Refusing to switch a modified upstream checkout: ${destination}" >&2
    echo "Use a clean BITE2TEXT_VENDOR_ROOT or preserve those edits first." >&2
    exit 1
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

apply_patch_once() {
  local repository=$1
  local patch_file=$2
  if git -C "${repository}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    return
  fi
  git -C "${repository}" apply --check "${patch_file}"
  git -C "${repository}" apply "${patch_file}"
}

require_command git
require_command rsync
mkdir -p "${VENDOR_ROOT}" "${PTV3_CONTEXT}"

BITS2BITES_DIR=${VENDOR_ROOT}/Bits2Bites
IOS_NORMALIZER_DIR=${VENDOR_ROOT}/IOS-Normalizer
checkout_pinned "${BITS2BITES_URL}" "${BITS2BITES_COMMIT}" "${BITS2BITES_DIR}"
apply_patch_once "${BITS2BITES_DIR}" "${PROJECT_ROOT}/bits2bites_repro_fixes.patch"
apply_patch_once "${BITS2BITES_DIR}" "${PROJECT_ROOT}/bits2bites_standalone_test_fix.patch"
apply_patch_once "${BITS2BITES_DIR}" "${PROJECT_ROOT}/bits2bites_bite2text_training.patch"
if [[ -e "${BITS2BITES_DIR}/pointcept/datasets/bite2text.py" ]] && \
  ! cmp -s "${PROJECT_ROOT}/task2_bite2text/ptv3_finetune/bite2text_dataset.py" \
    "${BITS2BITES_DIR}/pointcept/datasets/bite2text.py"; then
  echo "Refusing to overwrite a different local Bite2Text adapter; use a new vendor root." >&2
  exit 1
fi
cp "${PROJECT_ROOT}/task2_bite2text/ptv3_finetune/bite2text_dataset.py" \
  "${BITS2BITES_DIR}/pointcept/datasets/bite2text.py"
checkout_pinned "${IOS_NORMALIZER_URL}" "${IOS_NORMALIZER_COMMIT}" "${IOS_NORMALIZER_DIR}"
apply_patch_once "${IOS_NORMALIZER_DIR}" "${PROJECT_ROOT}/ios_normalizer_repro.patch"

test -d "${BITS2BITES_DIR}/pointcept"
test -d "${IOS_NORMALIZER_DIR}/src/scannormalizer"
rsync -a --delete "${BITS2BITES_DIR}/pointcept/" "${PTV3_CONTEXT}/pointcept/"
rsync -a --delete "${IOS_NORMALIZER_DIR}/src/scannormalizer/" "${PTV3_CONTEXT}/scannormalizer/"

for source_name in inference.py normalize_pair.py prepare_ptv3_dataset.py report_renderer.py; do
  cp "${PROJECT_ROOT}/task2_bite2text/ptv3_finetune/${source_name}" \
    "${PTV3_CONTEXT}/${source_name}"
done

echo "Prepared pinned upstream Docker context at ${PTV3_CONTEXT}"
