#!/usr/bin/env bash
# Task 2 (Bite2Text) submission: up to 7 rendered intraoral-scan views + up to 7 intraoral
# photos, target = chart block + "\n\nReport:\n" + canonical prose. Submitted epoch: 10.
#
#     bash bite2text_pipeline/run_b2t_chartqa_v1.sh    # add --wandb to log to Weights & Biases
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="b2t_chartqa_v1"

python -m bite2text_pipeline.train \
    --split train --chart-concat --chart-loss-weight 1.0 \
    --epochs 10 --batch 1 --accum 8 --warmup-ratio 0.05 \
    --lr 1e-4 --vision-lr 1e-4 --seed 2026 \
    --save-each-epoch \
    --wandb-run-name "$RUN" \
    --out "bite2text_pipeline/runs/$RUN" "$@"
