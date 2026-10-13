#!/usr/bin/env bash
# Build the Bite2Text (Task 2) int4 (NF4) MODEL tarball GC mounts at /opt/ml/model.
# Run in the DEV CONTAINER (needs the GPU) — NOT on the docker host.
#
#   1. merge b2t_chartqa_v1/adapter_ep{N} -> full bf16 model dir (CPU, 4GB shards)
#   2. quantize -> NF4 double-quant / bf16 compute, on GPU, saved sharded 2GB
#   3. tar with files AT ROOT (no top-level dir) — the layout /opt/ml/model expects
#   4. verify: reload the int4 dir with NO quant flags + generate one real val case,
#      under an optional VRAM cap (VRAM_CAP_GIB) to prove the deploy budget
#
# Sibling of submission/export_model_tarball.sh (Task 1 / fiveviews); same NF4 recipe and the
# same reused merge_export.py + quantize_int4.py. Only the checkpoint, the verify dataset path
# and the tarball name differ.
#
# Usage:  bite2text_pipeline/export_model_tarball.sh [EPOCH]        (default 10)
#         VRAM_CAP_GIB=16 bite2text_pipeline/export_model_tarball.sh 10
#         SKIP_MODULES="" ...   -> quantize the vision tower + lm_head too (smaller, ~-2.4 GB)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)
BASELINE="$REPO_ROOT/experiments/baseline_hu_v1/03_lora_train"   # merge/quantize scripts live here

EPOCH="${1:-10}"
RUN_NAME="${RUN_NAME:-b2t_chartqa_v1}"
CKPT="$SCRIPT_DIR/runs/${RUN_NAME}/adapter_ep${EPOCH}"
WORK="${WORK_DIR:-/path/to/cache/model_export}"
TAG="b2t_${RUN_NAME}_ep${EPOCH}"
MERGED="$WORK/merged_${TAG}"
INT4="$WORK/model_${TAG}_int4"
OUTDIR="${OUT_DIR:-/path/to/cache/model_tarballs}"
SKIP_MODULES="${SKIP_MODULES-visual,lm_head}"
SUFFIX=""; [ -z "$SKIP_MODULES" ] && SUFFIX="-allq"
TAR="$OUTDIR/odin-model-qwen7b-bite2text-ep${EPOCH}-int4${SUFFIX}.tar.gz"

[ -d "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$WORK" "$OUTDIR"

echo "### [1/4] merge adapter_ep${EPOCH} -> full model (CPU, bf16, 4GB shards) ###"
if [ -f "$MERGED/config.json" ]; then
    echo "  merged dir already present -> reuse $MERGED"
else
    CUDA_VISIBLE_DEVICES="" python "$BASELINE/merge_export.py" \
        --checkpoint "$CKPT" --out-dir "$MERGED" --max-shard-size 4GB
fi

echo "### [2/4] quantize -> int4 NF4 (GPU, 2GB shards); bf16-kept: '${SKIP_MODULES}' ###"
python "$BASELINE/quantize_int4.py" --src "$MERGED" --out "$INT4" \
    --max-shard-size 2GB --skip-modules "$SKIP_MODULES"

echo "### [3/4] tar (files at root) -> $TAR ###"
rm -f "$TAR"
if command -v pigz &> /dev/null; then
    tar cf - -C "$INT4" . | pigz > "$TAR"
else
    tar cf - -C "$INT4" . | gzip -1 > "$TAR"
fi
ls -lh "$TAR"

echo "### [4/4] verify: reload int4 payload + generate one real bite2text case ###"
python "$SCRIPT_DIR/verify_model.py" --model "$INT4" --split val --max-cases 2 \
    ${VRAM_CAP_GIB:+--vram-cap-gib "$VRAM_CAP_GIB"}

echo "### INT4 MODEL TARBALL COMPLETE — $TAR ###"
echo "Upload $TAR as a 'Model' on the GC algorithm; it mounts at /opt/ml/model."
