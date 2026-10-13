#!/bin/bash
# =============================================================================
# scripts/run_gen_gt.sh -- reference reports -> the labels everything is scored
# against.
#
#   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 \
#          scripts/run_gen_gt.sh validate          # the full run, needs a GPU
#
#   STAGE=derive scripts/run_gen_gt.sh validate    # login node, no GPU, no queue
#   DRY_RUN=1    scripts/run_gen_gt.sh validate    # group the reports, call nothing
#
# All the orchestration is in code/ground_truth/gen_gt.py -- the server
# lifecycle, the port proofs, the flag marshalling and the output audit. This
# file is the SLURM directives, the conda activation and the container path.
#
# CONSENSUS IS THE DEFAULT, OR THE RESULT IS UNUSABLE FOR 26 OF 40 VALIDATE CASES
# -------------------------------------------------------------------------------
#   (default)             merge every radiologist of a case, field by field
#   FIRST_REPORT_ONLY=1   one report per case, the lowest-numbered radiologist
#   CONSENSUS=0           neither -- per-reader files only, see below
#
# structured_findings_evaluation.py looks up exactly one file per case,
# {case}_gt.json. With neither flag, a multi-report case writes only
# {case}_{radiologist}_gt.json -- so that case is skipped with NO ERROR and the
# run reports a plausible score over a subset. gen_gt.py counts both filename
# shapes when it finishes and warns by name, but the flag is the fix.
#
# The two are mutually exclusive: consensus needs every radiologist's report.
#
# NO HUMAN IN THE GROUND TRUTH -- on by default, AUTO_RESOLVE=0 turns it off
# -------------------------------------------------------------------------
# After stage 2: the audit's mechanical repairs, then every claim they leave
# unsettled is masked (its labels nulled), stage 2 is replayed, and the job
# exits 1 unless the audit is clean. Output goes straight to
# dataset/<split>/outputs/ground_truth/ -- which a full run OVERWRITES, so set
# OUT_DIR to build somewhere else first if you want to compare. SCHEMA=
# builds against another schema file than schema/schema.json:
#   SCHEMA=/path/to/schema.json \
#       sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/run_gen_gt.sh training
#
# TWO STAGES, AND ONLY THE FIRST WANTS THIS JOB'S GPU
# ---------------------------------------------------
#   STAGE=all     (default) LLM extraction -> *_report_facts.json, then the
#                 deterministic derivation -> *_gt.json. Needs the server.
#   STAGE=derive  the second half alone, from report_facts already on disk.
#                 Pure CPU, so run it on the login node -- which is what makes
#                 a change to the deterministic half cheap to test. Verified
#                 to reproduce dataset/validate's live *_gt.json byte for byte.
#
# WHY THE PORT DEFAULT IS 8011 AND NOT 8001
# -----------------------------------------
# scripts/judge_server.sh defaults to 8001 and takes --qos=a100 too, so it can
# hold the same node. vLLM binds with SO_REUSEPORT, so a second server does NOT
# fail with "address already in use" -- both bind and the kernel round-robins
# between them. The judge serves `qwen3-14b-text` and this job asks for
# `qwen3-text`, so about half of every extraction call 404s: see
# logs/gen_gt_549316.log (342 x 200) against logs/judge_server_549303.log
# (379 x 404) at the same timestamps. gen_gt.py scans for a free port, waits on
# /v1/models rather than /health, and then probes eight times before trusting
# it. PORT= overrides the start of the scan.
# =============================================================================
#SBATCH --job-name=gen_gt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_gt_%j.log
#SBATCH --error=logs/gen_gt_%j.err
#
# NOTE: --partition/--qos/--gres are deliberately not here. They are site
# facts, and STAGE=derive wants no GPU at all, so they belong on the submit
# line until env/cluster.env lands:
#   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/run_gen_gt.sh validate
#
# On an 80GB card there is room for the full context:
#   MAX_MODEL_LEN=32768 sbatch --qos=a100-sxm4-80gb \
#       --gres=gpu:a100-sxm4-80gb:1 scripts/run_gen_gt.sh validate

set -euo pipefail

# Python writes stdout in 4 KB blocks when it is redirected to a file, which
# for a 12 h job means [INFO] lines and Trainer metrics are invisible for hours
# after they happen. Job 561429 lost its first eval_loss that way: the eval
# demonstrably ran -- 478 s unaccounted against arm 6's 450 s eval_runtime --
# and the line sat in the buffer while the run looked stalled at step 295.
# vision_sft.sh, pool_infer.sh and draft_evidence.sh all carry this; the
# run_*.sh entry points that superseded them did not.
export PYTHONUNBUFFERED=1


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SPLIT="${1:-${SPLIT:-validate}}"
STAGE="${STAGE:-all}"

case "$SPLIT" in training|validate) ;; *)
    echo "[FAIL] SPLIT must be training or validate (got '$SPLIT')" >&2; exit 1 ;;
esac
case "$STAGE" in all|derive) ;; *)
    echo "[FAIL] STAGE must be all or derive (got '$STAGE')" >&2; exit 1 ;;
esac

# Resolved by walking up for schema/schema.json, so this script does not care
# how deep scripts/ sits.
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

# SCHEMA= builds against another schema than the working copy's -- e.g. the
# one a trained arm used, while schema/schema.json is mid-edit for the next.
SCHEMA="${SCHEMA:-$PROJECT_DIR/schema/schema.json}"
[ -f "$SCHEMA" ] || { echo "[FAIL] no schema: $SCHEMA" >&2; exit 1; }
CONTAINER="${CONTAINER:-${SIF_PATH:-$HOME/containers/extraction.sqsh}}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
MODEL_NAME="${QWEN_MODEL_NAME:-${MODEL_NAME:-Qwen3-14B}}"
CONDA_ENV="${CONDA_ENV-cbct_base}"

ARGS=(--split "$SPLIT" --stage "$STAGE"
      --schema "$SCHEMA"
      --model-dir "$MODEL_DIR" --model-name "$MODEL_NAME")

[ -n "${OUT_DIR:-}"      ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "${REPORTS_DIR:-}"  ] && ARGS+=(--reports-dir "$REPORTS_DIR")
[ -n "${LIMIT:-}"        ] && ARGS+=(--limit "$LIMIT")
[ -n "${CASE_IDS:-}"     ] && ARGS+=(--case-ids ${CASE_IDS})
[ -n "${RESUME:-}"       ] && ARGS+=(--resume)
[ -n "${DRY_RUN:-}"      ] && ARGS+=(--dry-run)
if [ -z "${FIRST_REPORT_ONLY:-}" ] && [ "${CONSENSUS:-1}" != 0 ]; then
    ARGS+=(--consensus)
fi
[ "${AUTO_RESOLVE:-1}" = 0 ] && ARGS+=(--no-auto-resolve)
[ -n "${FIRST_REPORT_ONLY:-}" ] && ARGS+=(--first-report-only)
[ -n "${PORT:-}"         ] && ARGS+=(--port "$PORT")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "${VLLM_URL:-}"     ] && ARGS+=(--vllm-url "$VLLM_URL")

# --consensus is also accepted positionally, which is how this job has always
# been invoked: `sbatch scripts/run_gen_gt.sh validate --consensus`.
shift || true
for arg in "$@"; do
    [ "$arg" = "--consensus" ] && ARGS+=(--consensus)
done

# The container is only needed to START vLLM, and only STAGE=all does that.
# gen_gt.py then calls the server over HTTP from the HOST: extraction makes
# network calls and loads no model of its own, so there is nothing to gain from
# running the client inside the image.
if [ "$STAGE" = all ] && [ -z "${DRY_RUN:-}" ] && [ -z "${VLLM_URL:-}" ]; then
    [ -f "$CONTAINER" ] || { echo "[FAIL] no container: $CONTAINER" >&2; exit 1; }
    ARGS+=(--container "$CONTAINER")
fi

# Do NOT reach conda through `conda info --base` alone. A non-interactive
# submission (ssh host 'sbatch ...') never sourced ~/.bashrc, so the conda shell
# function does not exist on the compute node, and on this cluster `conda` is
# not on PATH even on the login node -- a naive guard silently skips activation
# and leaves python3 as /usr/bin/python3. Job 551011 is what that costs.
# CONDA_ENV= (empty) skips activation for a shell already in the right env.
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
echo "[INFO] python3: $(command -v python3)"

mkdir -p "$PROJECT_DIR/logs"
python3 "$PROJECT_DIR/code/ground_truth/gen_gt.py" "${ARGS[@]}"

echo "[PASS] stage=$STAGE split=$SPLIT done"
