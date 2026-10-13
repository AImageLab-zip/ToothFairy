#!/usr/bin/env bash
# Task 1 (ToothFairy4), submission "two panoramics": chart-v2 target, tooth class weights,
# two panoramic stills (pano_union + pano_rgb), no cross-section video. Submitted epoch: 7.
#
#     bash run_chartv2_pano_v1.sh                 # add --wandb to log to Weights & Biases
set -euo pipefail
cd "$(dirname "$0")"

RUN="chartv2_pano_v1"
RUN_DIR="runs/$RUN"

VIEWS="pano_union,pano_rgb"   # two panoramic stills
FRAMES=0                      # no cross-section video
TILE=728                      # render height before the token-budget fit rescales it
MPX=1.59                      # per-image megapixel cap (renderer + processor max_pixels)
STILL_TOKENS=4800             # image-token ceiling across the stills
WEIGHT_POWER=0.5              # softening exponent of the inverse-frequency chart class weights
EPOCHS=8

python train.py \
    --split train --val-split val --canonical --chart-v2 \
    --chart-v2-class-weights --chart-v2-weight-power "$WEIGHT_POWER" \
    --epochs "$EPOCHS" --warmup-ratio 0.03 --batch 1 --grad-accum 8 \
    --lr 1e-4 --vision-lr 5e-5 \
    --frames "$FRAMES" --tile "$TILE" --views "$VIEWS" \
    --still-tokens "$STILL_TOKENS" --still-mpx "$MPX" --still-per-view \
    --workers 4 --seed 2026 \
    --family-weights "P=1.0,A=1.7,F=2.0,S=1.3" --sampler-seed 2026 \
    --augment --mirror-p 0.5 --z-jitter-mm 3.0 \
    --val-cases 62 --val-every 0 --save-each-epoch \
    --wandb-run-name "$RUN" \
    --out "$RUN_DIR" "$@"
