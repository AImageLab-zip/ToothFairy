#!/bin/bash
# =============================================================================
# scripts/run_eval.sh -- score a run: the official ranking, and the fact-level
# survey underneath it.
#
#   scripts/run_eval.sh outputs/aksssr_v7_trained_arm6_validate
#   STAGE=survey    scripts/run_eval.sh <run_dir>       # no judge needed
#   NO_RADFACT=1    scripts/run_eval.sh <run_dir>       # BLEU/METEOR only
#   ARM_DIR=outputs/training_results/vsft_arm5 SPLIT=training \
#                   scripts/run_eval.sh <run_dir>       # three surveys
#
# NO #SBATCH, ON PURPOSE. This runs on the LOGIN NODE.
# ----------------------------------------------------
# The RadFact judge is a separate, long-lived job:
#
#   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/judge_server.sh
#   tail -f logs/judge_server_*.log        # wait for "[PASS] Judge ready"
#   scripts/run_eval.sh <run_dir>          # as often as you like
#   scancel -n judge                       # it does NOT stop before --time
#
# That split is the point. judge_server.sh holds the model on a GPU node and
# advertises itself in .judge/{url,ready,jobid}; this script finds it and pays
# no model load. Folding them into one job -- which the evaluation script this
# supersedes did -- reloads a 14B model on every evaluation, 15 to 25 minutes
# to score a run that takes seconds. eval.py discovers the judge in this order:
# --judge-url, then .judge/url, then `squeue -n judge`.
#
# ARGUMENT ORDER CHANGED FROM THE eval_now.sh THIS SUPERSEDES
# ----------------------------------------------------------
#   was:  <training|validate> [synthesized_reports_dir]
#   now:  <run_dir> [training|validate]
# One RUN dir, because both halves read the same run -- the ranking scores
# <run_dir>/synthesized_reports, the survey reads <run_dir>/predictions and
# summaries. Passing a reports dir and a split separately made it possible to
# score one arm's reports against another arm's ground truth.
#
# WHY $PYTHON IS NOT JUST python3
# -------------------------------
# official_ranking.py needs nltk + radfact_lite, which live in cbct_base and
# not in the system python. eval.py invokes its two modules with
# sys.executable, so whichever interpreter starts it is the one they inherit --
# which is why picking it here is enough, and why the nltk preflight in eval.py
# is checking the interpreter that will actually do the work.
#
# The wordnet corpus is checked too, and it is the subtler of the two: nltk
# installs no corpora, a missing wordnet is not an error, and METEOR silently
# changes implementation AND SCALE on 10% of the Final Score. eval.py says so
# before a RadFact run that takes hours, while the fix is still cheap.
# =============================================================================
set -euo pipefail

# Python writes stdout in 4 KB blocks when it is redirected to a file, which
# for a 12 h job means [INFO] lines and Trainer metrics are invisible for hours
# after they happen. Job 561429 lost its first eval_loss that way: the eval
# demonstrably ran -- 478 s unaccounted against arm 6's 450 s eval_runtime --
# and the line sat in the buffer while the run looked stalled at step 295.
# vision_sft.sh, pool_infer.sh and draft_evidence.sh all carry this; the
# run_*.sh entry points that superseded them did not.
export PYTHONUNBUFFERED=1


RUN_DIR="${1:?usage: scripts/run_eval.sh <run_dir> [training|validate]}"
SPLIT="${2:-${SPLIT:-validate}}"
STAGE="${STAGE:-all}"

case "$SPLIT" in training|validate) ;; *)
    echo "[FAIL] split must be training or validate (got '$SPLIT')" >&2; exit 1 ;;
esac
case "$STAGE" in all|rank|survey) ;; *)
    echo "[FAIL] STAGE must be all, rank or survey (got '$STAGE')" >&2; exit 1 ;;
esac

PROJECT_DIR="${PROJECT_DIR:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; \
    while [ "$d" != "/" ] && [ ! -f "$d/schema/schema.json" ]; do d="$(dirname "$d")"; done; echo "$d")}"
[ -f "$PROJECT_DIR/schema/schema.json" ] || { echo "[FAIL] no schema/schema.json above $0" >&2; exit 1; }
[ -d "$RUN_DIR" ] || { echo "[FAIL] no such run dir: $RUN_DIR" >&2; exit 1; }

# cbct_base holds nltk + radfact_lite. Fall back to python3 rather than failing
# here: STAGE=survey needs neither, and eval.py's own preflight gives a better
# message than a guess at this level could.
PYTHON="${PYTHON:-$HOME/miniconda3/envs/cbct_base/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

ARGS=("$RUN_DIR" --split "$SPLIT" --stage "$STAGE"
      --schema "$PROJECT_DIR/schema/schema.json")

[ -n "${OUT_DIR:-}"         ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "${GT_DIR:-}"          ] && ARGS+=(--gt-dir "$GT_DIR")
[ -n "${REPORTS_DIR:-}"     ] && ARGS+=(--reports-dir "$REPORTS_DIR")
[ -n "${SYNTH_DIR:-}"       ] && ARGS+=(--synthesized-dir "$SYNTH_DIR")
[ -n "${CASE_IDS:-}"        ] && ARGS+=(--case-ids ${CASE_IDS})
[ -n "${NO_RADFACT:-}"      ] && ARGS+=(--no-radfact)
[ -n "${JUDGE_URL:-}"       ] && ARGS+=(--judge-url "$JUDGE_URL")
[ -n "${SERVED_NAME:-}"     ] && ARGS+=(--served-name "$SERVED_NAME")
[ -n "${RESUME:-}"          ] && ARGS+=(--resume)
[ -n "${BATCH_SIZE:-}"      ] && ARGS+=(--batch-size "$BATCH_SIZE")
[ -n "${MULTI_REFERENCE:-}" ] && ARGS+=(--multi-reference)
[ -n "${NO_FILTER_NORMAL:-}" ] && ARGS+=(--no-filter-normal)
[ -n "${ARM_DIR:-}"         ] && ARGS+=(--survey-splits "$ARM_DIR")

echo "[INFO] python3: $PYTHON"
exec "$PYTHON" "$PROJECT_DIR/code/eval/eval.py" "${ARGS[@]}"
