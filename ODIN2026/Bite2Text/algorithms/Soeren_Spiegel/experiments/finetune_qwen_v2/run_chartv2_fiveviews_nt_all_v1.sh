#!/usr/bin/env bash
# Task 1 (ToothFairy4), submission "five stills + video": chart-v2 target with tooth AND
# non-tooth class weights, five stills + a 64-frame cross-section video. Trained on all 622
# cases (train + val + test, `splits/all622.json`) for the final submission, so it has no local
# held-out data. Submitted epoch: 4 (77 steps/epoch -> 308 steps, the step-matched equivalent of
# epoch 5 of the same recipe trained on the 498-case train split).
#
#     bash run_chartv2_fiveviews_nt_all_v1.sh     # add --wandb to log to Weights & Biases
set -euo pipefail
cd "$(dirname "$0")"

RUN="chartv2_fiveviews_nt_all_v1"
RUN_DIR="runs/$RUN"

SPLIT="all622"                # splits/all622.json + splits/canonical/all622.jsonl
VIEWS="pano_union,pano_mandible,pano_rgb,occlusal_rgb,mpr_rgb"   # DEFAULT_VIEWS
FRAMES=64                     # cross-section video frames
TILE=728
MPX=1.59
STILL_TOKENS=4800
WEIGHT_POWER=0.5              # tooth class-weight softening exponent
NT_WEIGHT_POWER=0.5           # non-tooth class-weight softening exponent
NT_WEIGHT_CAP=8.0             # max inverse-frequency ratio for the non-tooth lines
EPOCHS=8

python train.py \
    --split "$SPLIT" --canonical --chart-v2 \
    --chart-v2-class-weights --chart-v2-weight-power "$WEIGHT_POWER" \
    --nontooth-class-weights --nontooth-weight-power "$NT_WEIGHT_POWER" \
    --nontooth-weight-cap "$NT_WEIGHT_CAP" \
    --epochs "$EPOCHS" --warmup-ratio 0.03 --batch 1 --grad-accum 8 \
    --lr 1e-4 --vision-lr 5e-5 \
    --frames "$FRAMES" --tile "$TILE" --views "$VIEWS" \
    --still-tokens "$STILL_TOKENS" --still-mpx "$MPX" --still-per-view \
    --workers 4 --seed 2026 \
    --family-weights "P=1.0,A=1.7,F=2.0,S=1.3" --sampler-seed 2026 \
    --augment --mirror-p 0.5 --z-jitter-mm 3.0 \
    --val-cases 0 --val-every 0 --save-each-epoch \
    --wandb-run-name "$RUN" \
    --out "$RUN_DIR" "$@"
