#!/bin/bash
# =============================================================================
# scripts/run_infer.sh -- the inference path: CBCT + mask -> a finished report.
#
#   STAGE=images sbatch scripts/run_infer.sh validate    # CPU partition
#   STAGE=infer  sbatch scripts/run_infer.sh validate    # a100, the only GPU stage
#   STAGE=post         scripts/run_infer.sh validate     # login node, no queue
#   STAGE=all    sbatch scripts/run_infer.sh validate    # one job, one case
#
# All the orchestration is in code/pipeline/infer.py. This file is the SLURM directives,
# the pyxis invocation and the environment -- the three things a Python entry
# point genuinely cannot do for itself. If you are looking for what the pipeline
# DOES, read infer.py; if you are looking for where it runs, read this.
#
# WHY THE STAGES EXIST, WHICH IS NOT MODULARITY
# ---------------------------------------------
# They want different hardware, and the difference is expensive:
#
#   images   Four generators per case, all CPU. On the gpu partition this holds
#            an a100 for ~50 minutes over 40 cases while never touching it.
#            gen_images_cpu.sh existed for exactly this reason and is folded in
#            here: a batch run holds its allocation from the first case to the
#            last, so the rendering half belongs on the cpu partition where the
#            queue is short and the card is not idle.
#
#   infer    vLLM + the VLM calls. The only stage that needs a GPU, and the only
#            one that must run INSIDE the container -- vLLM is not installed on
#            the host. Folded in from pool_infer.sh.
#
#   post     postprocess_pred.py -> synthesize_report.py -> the fact survey.
#            No GPU, no model, so it runs on the login node in seconds per case.
#            This is the postprocess_now.sh tuning loop, and it is the reason
#            that loop was worth having: a flag change is re-measured without a
#            queue wait. Folded in from postprocess_now.sh.
#
# STAGE=all is the single-case path. Splitting one case into three queue waits
# would cost more than it saves, and infer.py already overlaps the vLLM load
# against the rendering, which is the whole trick that makes one case fast.
#
# DO NOT RUN TWO COPIES OVER THE SAME OUT_DIR
# -------------------------------------------
# Inherited from gen_images_cpu.sh and still true. The per-generator completion
# signal is written at the END of each generator, so two workers that start the
# same case both see "not done" and race on the same PNG paths.
#
# WHAT --gt-dir DOES, AND WHY IT IS OPTIONAL
# ------------------------------------------
# It turns on the fact-level survey at the end of STAGE=post, written to
# <out>/survey/survey_facts_<stamp>.{txt,json} and never overwritten, so
# successive runs diff against each other. That diff is how a postprocess flag
# change is read -- `git diff` cannot show it, because outputs/ is gitignored.
# Without it the summaries and reports are still built and only the measurement
# is skipped; a missing ground truth is not a pipeline failure.
#
# THE SOURCE RULES ARE ON BY DEFAULT, AND SILENT WHEN THEY ARE NOT
# ---------------------------------------------------------------
# infer.py passes --facts-dir to postprocess_pred.py whenever audited facts are
# present, which is what turns on the ten source rules worth +0.06 official. In
# STAGE=post the facts come from <out>/facts, written by STAGE=images. Run post
# against an out-dir that never had an images stage and the rules are OFF and
# the summaries are the pre-2026-08-16 shape -- infer.py warns, because nothing
# else about the output says so. NO_FACTS=1 is the deliberate ablation.
#
# THE ARM IS A CONFIG FILE -- POSTPROCESS_CONFIG
# ----------------------------------------------
# Which source rules, cross-source gates and FOV policies run is
# configs/postprocess/<arm>.yaml, not a constant edited in postprocess_pred.py
# and not a flag added here. Unset, infer.py uses configs/postprocess/default.yaml
# -- the arm-6 settings, identical to what the code carries as constants, so
# the default costs nothing and every summary gains a `postprocess_config`
# stamp naming the arm that built it.
#
#   STAGE=post POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml \
#       scripts/run_infer.sh validate
#
# An experiment is a new file in configs/postprocess/ and one line here. It is
# NOT a code edit -- that is the whole point of the split, and it is what makes
# the survey diff between two runs attributable to a named arm rather than to
# whatever the working tree happened to contain when the job was scheduled.
#
# The config and NO_FACTS=1 are DIFFERENT ablations. NO_FACTS drops
# --facts-dir, which also takes away the maxilla-FOV override; a config with
# the eleven rules off keeps the facts file and isolates the rules alone.
# configs/postprocess/no_source_rules.yaml says so in its own header.
# =============================================================================
#SBATCH --job-name=run_infer
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_infer_%j.log
#SBATCH --error=logs/run_infer_%j.err
#
# NOTE: --partition/--qos/--gres are NOT here on purpose. They differ per stage
# (images wants cpu, infer wants an a100) and per site, so they come from
# env/cluster.env via the submit line rather than being baked in:
#
#   sbatch $(STAGE=images bash scripts/run_infer.sh --sbatch-args) ...
#
# Until env/cluster.env lands, pass them on the sbatch command line:
#   STAGE=images sbatch --partition=cpu --qos=cpu scripts/run_infer.sh validate
#   STAGE=infer  sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 \
#                       scripts/run_infer.sh validate

set -euo pipefail

# Python writes stdout in 4 KB blocks when it is redirected to a file, which
# for a 12 h job means [INFO] lines and Trainer metrics are invisible for hours
# after they happen. Job 561429 lost its first eval_loss that way: the eval
# demonstrably ran -- 478 s unaccounted against arm 6's 450 s eval_runtime --
# and the line sat in the buffer while the run looked stalled at step 295.
# vision_sft.sh, pool_infer.sh and draft_evidence.sh all carry this; the
# run_*.sh entry points that superseded them did not.
export PYTHONUNBUFFERED=1


SPLIT="${1:-${SPLIT:-validate}}"
STAGE="${STAGE:-all}"

case "$STAGE" in
    all|images|infer|post) ;;
    *) echo "[FAIL] STAGE must be one of: all images infer post (got '$STAGE')" >&2
       exit 1 ;;
esac

# ── where things are ─────────────────────────────────────────────────────────
# Resolved by walking up for schema/schema.json, so this script does not care
# how deep scripts/ sits, and PROJECT_DIR still overrides for an out-of-tree run.
# Repo root, by walking up for schema/schema.json. TWO starting points, and
# the order matters: under `sbatch` SLURM copies this file to
# /var/spool/slurmd/job*/slurm_script, so $BASH_SOURCE walks up out of the
# SPOOL directory and finds nothing. $SLURM_SUBMIT_DIR is where sbatch was
# run and is set only under SLURM, so it is tried first there and is absent
# everywhere else. Job 561426 died on this four seconds in, holding an a100 --
# and all three `run_*.sh` entry points had the same defect, because the
# restructure only ever exercised them by path on the login node.
_find_root() {
    local d
    for d in "$@"; do
        [ -n "$d" ] || continue
        d="$(cd "$d" 2>/dev/null && pwd)" || continue
        while [ "$d" != "/" ]; do
            [ -f "$d/schema/schema.json" ] && { echo "$d"; return 0; }
            d="$(dirname "$d")"
        done
    done
    return 1
}
# `|| true`, or `set -e` kills the assignment on a failed search and the guard
# below never gets to say why.
PROJECT_DIR="${PROJECT_DIR:-$(_find_root "${SLURM_SUBMIT_DIR:-}" \
                                         "$(dirname "${BASH_SOURCE[0]}")" || true)}"
[ -f "${PROJECT_DIR:-}/schema/schema.json" ] || {
    echo "[FAIL] no schema/schema.json above ${SLURM_SUBMIT_DIR:-<unset>} or $0" >&2
    echo "[HINT] set PROJECT_DIR=/path/to/project_ToothFairy4" >&2; exit 1; }

CONTAINER="${CONTAINER:-$HOME/containers/extraction.sqsh}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B-AWQ-dental-cbct-sft}"
CONDA_ENV="${CONDA_ENV-cbct_base}"

RUN_NAME="${RUN_NAME:-aksssr_v7}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/${RUN_NAME}_${SPLIT}}"
GT_DIR="${GT_DIR:-$PROJECT_DIR/dataset/$SPLIT/outputs/ground_truth}"

# Unset on purpose rather than defaulted here: infer.py falls back to the
# shipped configs/postprocess/default.yaml and says so in one place. Defaulting
# it in this file too would give the default two homes and let them drift.
POSTPROCESS_CONFIG="${POSTPROCESS_CONFIG:-}"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-0}"

# ── what to run over ─────────────────────────────────────────────────────────
ARGS=(--stage "$STAGE" --out-dir "$OUT_DIR" --schema "$PROJECT_DIR/schema/schema.json")

if [ -n "${CASE_ID:-}" ]; then
    ARGS+=(--case-id "$CASE_ID"
           --volume "$PROJECT_DIR/dataset/$SPLIT/images/${CASE_ID}_0000.nii.gz"
           --mask   "$PROJECT_DIR/dataset/$SPLIT/masks/${CASE_ID}.nii.gz")
else
    ARGS+=(--dataset-dir "$PROJECT_DIR/dataset/$SPLIT")
fi
[ -n "${CASE_IDS:-}" ] && ARGS+=(--case-ids ${CASE_IDS})
[ -n "${LIMIT:-}"    ] && ARGS+=(--limit "$LIMIT")
[ -n "${SEED:-}"     ] && ARGS+=(--seed "$SEED")
[ -n "${RESUME:-}"   ] && ARGS+=(--resume)
[ -n "${NO_FACTS:-}" ] && ARGS+=(--no-facts)
# The survey belongs to post, so only those two stages ask for it. A missing
# GT dir is not an error here: infer.py skips the survey and says so.
if { [ "$STAGE" = post ] || [ "$STAGE" = all ]; } && [ -d "$GT_DIR" ]; then
    ARGS+=(--gt-dir "$GT_DIR")
fi
# Fail here rather than three stages in: a mistyped arm path that reaches
# infer.py only after the images and the inference have run has cost an a100.
if [ -n "$POSTPROCESS_CONFIG" ]; then
    [ -f "$POSTPROCESS_CONFIG" ] || {
        echo "[FAIL] POSTPROCESS_CONFIG=$POSTPROCESS_CONFIG does not exist." >&2
        echo "       Shipped arms:" >&2
        ls "$PROJECT_DIR/configs/postprocess/"*.yaml 2>/dev/null | sed 's/^/         /' >&2
        exit 1; }
    ARGS+=(--postprocess-config "$POSTPROCESS_CONFIG")
fi

echo "[INFO] stage=$STAGE split=$SPLIT out=$OUT_DIR arm=${POSTPROCESS_CONFIG:-configs/postprocess/default.yaml}"
mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"

# ── run it ───────────────────────────────────────────────────────────────────
# Only the infer stage enters the container: vLLM lives there and nowhere else
# on this cluster. images and post are pure host Python -- nibabel, PIL and the
# schema -- and putting them in the container would buy nothing and cost a
# squashfs mount.
if [ "$STAGE" = infer ] || [ "$STAGE" = all ]; then
    [ -f "$CONTAINER" ] || { echo "[FAIL] no container: $CONTAINER" >&2; exit 1; }
    [ -d "$MODEL_DIR/$MODEL_NAME" ] || { echo "[FAIL] no model: $MODEL_DIR/$MODEL_NAME" >&2; exit 1; }

    # $PROJECT_DIR is mounted at BOTH /project and its own path: qa_pairs.jsonl
    # stores image paths relative to --project-dir so one payload resolves on
    # host and in container, and the second mount keeps an absolute path handed
    # in from outside working too.
    srun --container-image="$CONTAINER" \
         --container-mounts="$MODEL_DIR:/models,$PROJECT_DIR:/project,$PROJECT_DIR:$PROJECT_DIR" \
         --container-workdir=/project \
         python3 /project/code/pipeline/infer.py \
            "${ARGS[@]}" \
            --model "/models/$MODEL_NAME" \
            --project-dir /project \
            --port "$PORT" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-concurrency "$MAX_CONCURRENCY"
else
    # No network is needed and none is wanted: an HF-hub lookup with no route
    # blocks on a timeout rather than failing fast.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1

    # Do NOT reach conda through `source ~/.bashrc`. A batch job's environment
    # is whatever the SUBMITTING shell exported, and a non-interactive
    # submission (ssh host 'sbatch ...', a cron hook) never sourced ~/.bashrc,
    # so the conda shell FUNCTION does not exist on the compute node even
    # though it does on an interactive login node. Job 550476 ran that way and
    # every generator in all 40 cases died on a missing nibabel. Source a real
    # conda.sh by path instead. CONDA_ENV= (empty) skips activation entirely,
    # for a shell that is already in the right env -- which is the normal case
    # for STAGE=post on the login node.
    if [ -n "${CONDA_ENV:-}" ]; then
        _conda_sh=""
        command -v conda >/dev/null 2>&1 && \
            _conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
        for _cand in "$_conda_sh" "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
                     "$HOME/miniconda3/etc/profile.d/conda.sh" \
                     "$HOME/anaconda3/etc/profile.d/conda.sh" \
                     "$HOME/miniforge3/etc/profile.d/conda.sh" \
                     "/opt/conda/etc/profile.d/conda.sh"; do
            # shellcheck disable=SC1090
            [ -n "$_cand" ] && [ -f "$_cand" ] && { source "$_cand"; break; }
        done
        command -v conda >/dev/null 2>&1 \
            || { echo "[FAIL] no conda.sh on this node. Set CONDA_BASE=/path/to/conda, or CONDA_ENV= to use the current environment." >&2; exit 1; }
        conda activate "$CONDA_ENV" \
            || { echo "[FAIL] conda activate $CONDA_ENV failed" >&2; exit 1; }
    fi
    python3 "$PROJECT_DIR/code/pipeline/infer.py" "${ARGS[@]}" --project-dir "$PROJECT_DIR"
fi

echo "[PASS] stage=$STAGE done -> $OUT_DIR"
