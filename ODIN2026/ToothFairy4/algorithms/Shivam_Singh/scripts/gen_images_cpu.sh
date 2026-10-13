#!/usr/bin/env bash
# =============================================================================
# scripts/gen_images_cpu.sh -- steps 1-4 of the main pipeline, on the CPU partition
#
# Why this exists
# ---------------
# A batch run holds its a100 allocation from the first case to the last, but
# only the vLLM half (preflight / server / inference) actually touches the GPU.
# Steps 1-4 are nibabel + numpy + PIL + marching cubes -- ~4 min/case, so
# ~3 h of a 40-case run, all of it spent with an idle A100 attached. When the
# gpu/a100 queue is deep, that time is paid twice: once waiting in PD for a GPU
# the image generation never needed, and again generating images while holding
# one.
#
# This job does steps 1-4 ONLY, on --partition=cpu (whose queue is normally far
# shorter), writing into the SAME outputs/${RUN_NAME}_${SPLIT}/images directory
# the pipeline uses. It is not a separate arm: same four generators, same facts
# arguments, same output filenames, so the images are the ones the pipeline
# would have produced itself. It also writes qa_pairs.jsonl (step 5), so what
# follows on the GPU is inference and nothing else:
#
#   sbatch scripts/gen_images_cpu.sh validate     # now, on the short cpu queue
#   STAGE=infer sbatch scripts/run_infer.sh validate  # later; the GPU step alone
#
# A batch driver that renders as well can be queued at the same time -- it
# re-checks each completion signal per case, so whatever this job has finished
# by then is skipped and the rest is generated on the GPU node as before.
# Nothing breaks if they overlap on a case; the worst case is that one case's
# images get written twice (see PARALLELISM below for the one real caveat).
#
# PARALLELISM
# -----------
# The pipeline walks cases serially. Here they are independent -- each
# generator reads only this case's volume + mask + facts and writes only this
# case's files -- so WORKERS cases run concurrently via xargs -P. WORKERS=4 on
# --cpus-per-task=8 / --mem=32G means 2 threads and ~8 GB per case, which is
# what the thread-count exports below are divided for. That turns ~3 h into
# ~45 min for a 40-case split.
#
# Raise WORKERS only together with --cpus-per-task and --mem: a CBCT volume
# plus its mask is several GB resident, and the failure mode of over-committing
# memory here is the whole job being OOM-killed, not one slow case.
#
# The one thing NOT to do is run two copies of this job over the same split and
# images dir at once. The completion signals are written at the END of each
# generator, so two workers that start the same case both see "not done" and
# race on the same PNG paths.
#
# Per-case output goes to outputs/${RUN_NAME}_${SPLIT}/imagegen_logs/<case>.log
# rather than the job log: with WORKERS>1 the generators' stdout would otherwise
# interleave into something unreadable. The job log keeps one line per case.
#
# FACTS
# -----
# Identical semantics to aksssr_pipeline.sh, because the images have to be the
# same ones:
#   default    step 1 gets --facts-file (REQUIRED -- create_panoramic.py filters
#              its outlines/tags by facts.teeth_present and hard-errors without
#              it), steps 2-3 get it optionally, step 4 never takes it. A case
#              with no facts file is skipped, not hard-failed.
#   NO_FACTS=1 step 1 takes --no-facts (every SEGMENTED tooth outlined -- this
#              changes the PIXELS, not just the caption), steps 2-3 are handed
#              no --facts-file, and the missing-facts skip is lifted.
# If you generate here with a different NO_FACTS setting than you later run the
# pipeline with, the pipeline will NOT correct it -- it skips on file existence,
# not on how the file was made. Keep the two invocations' NO_FACTS in sync.
#
# LIMIT/SEED match aksssr_pipeline.sh exactly (same seeded shuf), so LIMIT=5
# here generates images for the same 5 cases LIMIT=5 there will ask for.
#
# qa_pairs.jsonl (step 5) is built here too, but only on a full run: it is the
# real check that every image the schema asks for exists and is named right,
# and the pipeline rebuilds it in seconds anyway. Under LIMIT it is skipped
# rather than clobbering a full-split qa_pairs.jsonl with a 5-case one, it is
# skipped when no case succeeded, and it is written via a temp file that only
# moves into place once it is non-empty -- it is a COMMITTED artefact, so a
# build over an empty images dir must leave the good one alone.
#
# ENVIRONMENT: this activates cbct_base by sourcing a conda.sh found BY PATH,
# not via the `conda` command, so it works under a non-interactive submission
# (ssh host 'sbatch ...') where ~/.bashrc was never sourced and the conda shell
# function does not exist. Override the search with CONDA_BASE=/path/to/conda
# or the env name with CONDA_ENV=. Imports are preflighted once before any case
# runs, so a broken env exits in two seconds instead of failing 582 times.
#
# Usage:
#   sbatch scripts/gen_images_cpu.sh validate
#   sbatch scripts/gen_images_cpu.sh training
#   LIMIT=5 sbatch scripts/gen_images_cpu.sh validate            # same 5 cases as the pipeline
#   WORKERS=8 sbatch scripts/gen_images_cpu.sh training          # see PARALLELISM first
#   RUN_NAME=aksssr_v7 sbatch scripts/gen_images_cpu.sh validate # different output dir
#   NO_FACTS=1 sbatch scripts/gen_images_cpu.sh validate         # must match the pipeline run
# =============================================================================

#SBATCH --job-name=gen_images_cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_images_cpu_%j.log
#SBATCH --error=logs/gen_images_cpu_%j.err

set -euo pipefail

SPLIT="${1:?Usage: sbatch scripts/gen_images_cpu.sh <training|validate>}"
if [ "$SPLIT" != "training" ] && [ "$SPLIT" != "validate" ]; then
    echo "[FAIL] SPLIT must be 'training' or 'validate', got: $SPLIT"
    exit 1
fi

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
CODE_DIR="$PROJECT_DIR/code"
RUN_NAME="${RUN_NAME:-aksssr_v6}"
SCHEMA="$PROJECT_DIR/schema/schema.json"

BASE_DIR="$PROJECT_DIR/dataset/$SPLIT"
CBCT_DIR="$BASE_DIR/images"
MASK_DIR="$BASE_DIR/masks"
FACTS_DIR="$BASE_DIR/facts"

OUT_DIR="$PROJECT_DIR/outputs/${RUN_NAME}_${SPLIT}"
OUT_IMAGES="$OUT_DIR/images"
CASE_LOG_DIR="$OUT_DIR/imagegen_logs"
QA_JSONL="$OUT_DIR/qa_pairs.jsonl"

NO_FACTS="${NO_FACTS:-}"
LIMIT="${LIMIT:-}"
SEED="${SEED-42}"
WORKERS="${WORKERS:-4}"

# STEPS -- which generators to run, comma-separated from {pano,3d,tooth,sinus}.
#
# Default "all", which is the only correct setting for a pipeline run: the
# schema asks for every image and build_vqa_pairs.py drops a call whose images
# are missing, so a partial images dir silently produces a smaller payload.
#
# STEPS=tooth exists for docs/vision_sft_plan.md, whose target (§3.0) is the per-tooth
# composite call alone. Training targets are built from the tooth rows only, so
# the other three generators would cost ~3.5 of every 4 minutes per case to
# render images nothing in that experiment reads. On the 144-case SFT pool that
# is the difference between a ~2.5 h job and a ~25 min one.
#
# Do NOT then point a rendering batch driver at that images dir: it would find
# the tooth sidecars present, skip step 3, and generate the other three itself
# on the GPU node -- which works, but pays for the A100 to do CPU work, the
# exact thing this job exists to avoid.
STEPS="${STEPS:-all}"
want_step() {
    case ",$STEPS," in
        ,all,) return 0 ;;
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}
export -f want_step
export STEPS
for _s in ${STEPS//,/ }; do
    case "$_s" in
        all|pano|3d|tooth|sinus) ;;
        *) echo "[FAIL] STEPS: unknown step '$_s' (want pano,3d,tooth,sinus or all)"; exit 1 ;;
    esac
done

# Per-worker thread caps. Without these every worker's BLAS would try to grab
# all 8 allocated cores, and WORKERS=4 would oversubscribe the allocation 4x --
# slower than running serially, not faster.
CPUS_TOTAL="${SLURM_CPUS_PER_TASK:-8}"
THREADS_PER_WORKER=$(( CPUS_TOTAL / WORKERS ))
[ "$THREADS_PER_WORKER" -lt 1 ] && THREADS_PER_WORKER=1
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"

# Activate cbct_base -- all four generators are host python3 against it
# (nibabel/numpy/scipy/PIL). Nothing here goes near the container.
#
# Do NOT reach this through `conda info --base`. A batch job's environment is
# whatever the SUBMITTING shell exported, and a non-interactive submission
# (ssh host 'sbatch ...', a cron hook) never sourced ~/.bashrc, so the conda
# shell function does not exist on the compute node even though it does on an
# interactive login node. Job 550476 ran that way and every generator in all 40
# cases died on a missing nibabel. Source a real conda.sh by path instead, and
# hard-fail if none is found -- CONDA_BASE overrides the search.
CONDA_ENV="${CONDA_ENV:-cbct_base}"
_conda_sh=""
if command -v conda >/dev/null 2>&1; then
    _conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
for _cand in "$_conda_sh" "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -n "$_cand" ] && [ -f "$_cand" ]; then
        # shellcheck disable=SC1090
        source "$_cand"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] no conda.sh found on this node. Set CONDA_BASE=/path/to/conda and resubmit."
    exit 1
fi
conda activate "$CONDA_ENV" \
    || { echo "[FAIL] conda activate $CONDA_ENV failed"; exit 1; }

# Preflight the imports ONCE, before any case runs.
#
# gen_case tolerates a failing generator on purpose, so that one bad case
# cannot cost the other 581 theirs. The cost of that choice is that an
# ENVIRONMENT fault is not a per-case problem but looks exactly like one: it
# reports itself 582 times, after the whole allocation has been spent. This
# check turns that into a two-second exit, which is what 550476 needed.
echo "[INFO] python3: $(command -v python3)"
if ! python3 -c "import nibabel, numpy, scipy, PIL" 2>&1; then
    echo "[FAIL] '$CONDA_ENV' cannot import the image-generation dependencies"
    echo "[FAIL] (nibabel / numpy / scipy / PIL). Nothing would be generated."
    exit 1
fi

mkdir -p logs "$OUT_IMAGES" "$CASE_LOG_DIR"

# Status markers, one file per case, written by the workers. Needed because
# xargs -P runs each case in a subshell: a plain counter variable there would
# not survive back to this shell.
STATUS_DIR=$(mktemp -d)
cleanup_status() { rm -rf "$STATUS_DIR"; }
trap cleanup_status EXIT

# ── Per-case work (run by the xargs workers) ────────────────────────────────
#
# Deliberately mirrors steps 1-4 of aksssr_pipeline.sh line for line, including
# each step's completion signal, so an interrupted run of EITHER job resumes
# cheaply from the other's output:
#   1 panoramic -> {case}_panoramic.png            (may legitimately not exist:
#                                                   create_panoramic.py returns
#                                                   None when there is no jaw
#                                                   tissue at all)
#   2 3D        -> {case}_3d_captions.json         (sidecar, written last)
#   3 tooth     -> {case}_tooth_captions.json      (sidecar, written after the
#                                                   final tooth, so a half-done
#                                                   case redoes itself)
#   4 sinus     -> {case}_sinus_{right,left}_detail.png
#
# Failures are recorded and swallowed rather than propagated: one bad case must
# not take down the other 39. The tally after the loop is what reports them.
gen_case() {
    local case_id="$1"
    local log="$CASE_LOG_DIR/${case_id}.log"
    local volume_file="$CBCT_DIR/${case_id}_0000.nii.gz"
    local mask_file="$MASK_DIR/${case_id}.nii.gz"
    local facts_file="$FACTS_DIR/${case_id}.json"
    local -a pano_facts_arg case_facts_arg

    : > "$log"

    if [ ! -f "$volume_file" ]; then
        echo "[SKIP] $case_id -- no volume: $volume_file"
        touch "$STATUS_DIR/${case_id}.skip"
        return 0
    fi
    if [ ! -f "$mask_file" ]; then
        echo "[SKIP] $case_id -- no mask: $mask_file"
        touch "$STATUS_DIR/${case_id}.skip"
        return 0
    fi
    # Same guard as the pipeline: create_panoramic.py calls ap.error() without
    # --facts-file, so a missing facts file costs this case rather than the job.
    if [ -z "$NO_FACTS" ] && [ ! -f "$facts_file" ]; then
        echo "[SKIP] $case_id -- no facts: $facts_file"
        touch "$STATUS_DIR/${case_id}.skip"
        return 0
    fi

    if [ -n "$NO_FACTS" ]; then
        pano_facts_arg=(--no-facts)
        case_facts_arg=()
    else
        pano_facts_arg=(--facts-file "$facts_file")
        case_facts_arg=(--facts-file "$facts_file")
    fi

    local failed=0 did=""

    # Step 1: panoramic
    # Keyed on the caption sidecar, not the .png -- see the matching comment in
    # aksssr_pipeline.sh. The sidecar is what build_vqa_pairs.py consumes, and
    # it degrades silently (captions:{}) rather than failing when absent, so a
    # .png-keyed skip could hand the VLM an uncaptioned arm without a word.
    if ! want_step pano; then
        did="$did pano:off"
    elif [ -f "$OUT_IMAGES/${case_id}_panoramic_caption.json" ]; then
        did="$did pano:skip"
    else
        if python3 "$CODE_DIR/pipeline/preprocess/create_panoramic.py" \
                --volume "$volume_file" --mask "$mask_file" \
                "${pano_facts_arg[@]}" \
                --out-dir "$OUT_IMAGES" --case-id "$case_id" >>"$log" 2>&1; then
            did="$did pano"
        else
            did="$did pano:ERR"; failed=1
        fi
    fi

    # Step 2: 3D renders (left/frontal/right + captions sidecar)
    if ! want_step 3d; then
        did="$did 3d:off"
    elif [ -f "$OUT_IMAGES/${case_id}_3d_captions.json" ]; then
        did="$did 3d:skip"
    else
        if python3 "$CODE_DIR/pipeline/preprocess/create_3d_renders.py" \
                --mask "$mask_file" \
                "${case_facts_arg[@]}" \
                --out-dir "$OUT_IMAGES" --case-id "$case_id" >>"$log" 2>&1; then
            did="$did 3d"
        else
            did="$did 3d:ERR"; failed=1
        fi
    fi

    # Step 3: tooth composites (one per present tooth)
    if ! want_step tooth; then
        did="$did tooth:off"
    elif [ -f "$OUT_IMAGES/${case_id}_tooth_captions.json" ]; then
        did="$did tooth:skip"
    else
        if python3 "$CODE_DIR/pipeline/preprocess/create_tooth_detail.py" \
                --volume "$volume_file" --mask "$mask_file" \
                "${case_facts_arg[@]}" \
                --out-dir "$OUT_IMAGES" --case-id "$case_id" >>"$log" 2>&1; then
            did="$did tooth"
        else
            did="$did tooth:ERR"; failed=1
        fi
    fi

    # Step 4: sinus detail (no facts input, so NO_FACTS changes nothing here)
    # Keyed on the caption sidecar, as in step 1. A case with 0 sinus voxels
    # gets no sidecar and so re-runs every pass, the same cheap redundancy
    # step 3 already accepts for a case with no tooth labels.
    if ! want_step sinus; then
        did="$did sinus:off"
    elif [ -f "$OUT_IMAGES/${case_id}_sinus_captions.json" ]; then
        did="$did sinus:skip"
    else
        if python3 "$CODE_DIR/pipeline/preprocess/create_sinus_detail.py" \
                --volume "$volume_file" --mask "$mask_file" \
                --out-dir "$OUT_IMAGES" --case-id "$case_id" >>"$log" 2>&1; then
            did="$did sinus"
        else
            did="$did sinus:ERR"; failed=1
        fi
    fi

    if [ "$failed" -eq 0 ]; then
        touch "$STATUS_DIR/${case_id}.ok"
        echo "[OK]   $case_id --$did"
    else
        touch "$STATUS_DIR/${case_id}.fail"
        echo "[FAIL] $case_id --$did  (see $log)"
    fi
    return 0
}
export -f gen_case
export CODE_DIR OUT_IMAGES CASE_LOG_DIR CBCT_DIR MASK_DIR FACTS_DIR
export NO_FACTS STATUS_DIR

# ── Main ───────────────────────────────────────────────────────────────────

{
    echo "[INFO] ========== Image generation (CPU only, steps 1-4) =========="
    echo "[INFO] Start      : $(date)"
    echo "[INFO] Job ID     : ${SLURM_JOB_ID:-local}"
    echo "[INFO] Split      : $SPLIT"
    echo "[INFO] Run name   : $RUN_NAME"
    echo "[INFO] Volumes    : $CBCT_DIR"
    echo "[INFO] Masks      : $MASK_DIR"
    echo "[INFO] Out images : $OUT_IMAGES"
    echo "[INFO] Case logs  : $CASE_LOG_DIR/<case>.log"
    echo "[INFO] Workers    : $WORKERS (x $THREADS_PER_WORKER thread(s), $CPUS_TOTAL cpu(s) allocated)"
    if [ -n "$NO_FACTS" ]; then
        echo "[INFO] NO_FACTS   : ON -- facts/<case>.json is never opened."
        echo "[INFO]              panoramic outlines every SEGMENTED tooth (different"
        echo "[INFO]              PIXELS, not just a shorter caption). The pipeline run"
        echo "[INFO]              that consumes these images must ALSO set NO_FACTS=1."
    else
        echo "[INFO] Facts      : $FACTS_DIR"
    fi
    echo ""

    # ── Discover cases ──────────────────────────────────────────────────────
    CASE_IDS=()
    for volume_file in "$CBCT_DIR"/*_0000.nii.gz; do
        [ -f "$volume_file" ] || continue
        CASE_IDS+=("$(basename "$volume_file" _0000.nii.gz)")
    done
    if [ ${#CASE_IDS[@]} -eq 0 ]; then
        echo "[FAIL] No volumes found in $CBCT_DIR"
        exit 1
    fi
    echo "[INFO] Found ${#CASE_IDS[@]} case(s)"

    # Byte-for-byte the same sampling as aksssr_pipeline.sh, so LIMIT=N here
    # picks the same N cases that run will ask for. Diverging on this would
    # leave the GPU job generating images for cases this job never touched --
    # exactly the wait this job exists to avoid.
    # CASE_LIST -- one case id per line, blank lines and #comments ignored.
    #
    # A named set of cases rather than a sample of them: the SFT pools written
    # by code/train/select_sft_pool.py are chosen by GT quality and prefix balance,
    # which no seed reproduces. Passed as a FILE and not as a variable on
    # purpose -- `--export=ALL,CASE_IDS=A004,A019` is read by Slurm as
    # `CASE_IDS=A004`, because --export is itself comma-separated, and that
    # silently ran a whole-pool job over one case. A file has no
    # such edge, and the count is printed below either way.
    if [ -n "${CASE_LIST:-}" ]; then
        if [ ! -f "$CASE_LIST" ]; then
            echo "[FAIL] CASE_LIST not found: $CASE_LIST"
            exit 1
        fi
        mapfile -t WANTED < <(grep -vE '^\s*(#|$)' "$CASE_LIST" | tr -d '\r' | sort -u)
        if [ ${#WANTED[@]} -eq 0 ]; then
            echo "[FAIL] CASE_LIST is empty: $CASE_LIST"
            exit 1
        fi
        # Intersect with what actually has a volume, and say what was lost --
        # a case list naming a case this split does not have is a mistake
        # worth seeing, not worth silently rendering 143 of 144 for.
        mapfile -t FOUND < <(printf '%s\n' "${CASE_IDS[@]}" | sort -u)
        mapfile -t CASE_IDS < <(comm -12 <(printf '%s\n' "${WANTED[@]}") \
                                        <(printf '%s\n' "${FOUND[@]}"))
        missing=$(( ${#WANTED[@]} - ${#CASE_IDS[@]} ))
        echo "[INFO] CASE_LIST $CASE_LIST: ${#WANTED[@]} requested, ${#CASE_IDS[@]} present in $SPLIT"
        if [ "$missing" -gt 0 ]; then
            echo "[WARN] $missing case(s) in the list have no volume in $CBCT_DIR:"
            comm -23 <(printf '%s\n' "${WANTED[@]}") <(printf '%s\n' "${FOUND[@]}") | head -20
        fi
        if [ ${#CASE_IDS[@]} -eq 0 ]; then
            echo "[FAIL] no case in the list exists in $SPLIT"
            exit 1
        fi
    elif [ -n "$LIMIT" ]; then
        if [ -n "$SEED" ]; then
            mapfile -t CASE_IDS < <(printf '%s\n' "${CASE_IDS[@]}" \
                | shuf -n "$LIMIT" --random-source=<(yes "$SEED") | sort)
            echo "[INFO] Limited to ${#CASE_IDS[@]} random case(s) (SEED=$SEED): ${CASE_IDS[*]}"
        else
            mapfile -t CASE_IDS < <(printf '%s\n' "${CASE_IDS[@]}" | shuf -n "$LIMIT" | sort)
            echo "[INFO] Limited to ${#CASE_IDS[@]} random case(s) (unseeded): ${CASE_IDS[*]}"
        fi
    fi
    echo ""

    if [ ! -f "$SCHEMA" ]; then
        echo "[FAIL] Schema not found: $SCHEMA"
        exit 1
    fi

    # ── STEP 1-4, WORKERS cases at a time ───────────────────────────────────
    echo "[INFO] Generating images for ${#CASE_IDS[@]} case(s), $WORKERS at a time..."
    echo ""
    printf '%s\n' "${CASE_IDS[@]}" \
        | xargs -P "$WORKERS" -I{} bash -c 'gen_case "$@"' _ {}
    echo ""

    n_ok=$(find "$STATUS_DIR" -name '*.ok'   | wc -l)
    n_skip=$(find "$STATUS_DIR" -name '*.skip' | wc -l)
    n_fail=$(find "$STATUS_DIR" -name '*.fail' | wc -l)
    echo "[INFO] Cases: $n_ok ok, $n_skip skipped, $n_fail failed"
    if [ "$n_fail" -gt 0 ]; then
        echo "[WARN] Failed case(s):"
        find "$STATUS_DIR" -name '*.fail' -exec basename {} .fail \; | sort | sed 's/^/[WARN]   /'
        echo "[WARN] Per-case detail is in $CASE_LOG_DIR/<case>.log."
        echo "[WARN] Re-running this job retries exactly those cases: every"
        echo "[WARN] completed step is skipped on its existence check."
    fi
    echo ""

    # ── STEP 5: qa_pairs.jsonl (full runs only) ─────────────────────────────
    #
    # Built here as the real completeness check -- build_vqa_pairs.py resolves
    # every images_needed name in the schema against this directory, so it
    # catches a missing or misnamed image that the file counts below would not.
    # Skipped under LIMIT so a smoke run cannot replace a full-split
    # qa_pairs.jsonl with a 5-case one; the pipeline rebuilds it either way.
    if [ -n "$LIMIT" ]; then
        echo "[INFO] LIMIT set -- skipping qa_pairs.jsonl (would clobber a full-split one)."
    elif [ "$n_ok" -eq 0 ]; then
        echo "[WARN] no case succeeded -- NOT touching qa_pairs.jsonl."
        echo "[WARN] Rebuilding it from an empty images dir writes a 0-case file"
        echo "[WARN] over a good one, which is what 550476 did."
    else
        # Build to a temp file and move it into place only once it is non-empty.
        # $QA_JSONL is a COMMITTED artefact that the GPU run consumes; a build
        # that produces nothing (no images resolved, schema mismatch) must leave
        # the previous one alone rather than replace it with 0 records.
        echo "[STEP 5] Building QA pairs from schema..."
        qa_tmp="${QA_JSONL}.tmp.$$"
        if ! python3 "$CODE_DIR/pipeline/preprocess/build_vqa_pairs.py" \
                --schema "$SCHEMA" \
                --images-dir "$OUT_IMAGES" \
                --project-dir "$PROJECT_DIR" \
                --cases "${CASE_IDS[@]}" \
                --out "$qa_tmp"; then
            rm -f "$qa_tmp"
            echo "[FAIL] build_vqa_pairs.py failed; $QA_JSONL left unchanged"
            exit 1
        fi
        qa_lines=$( { wc -l < "$qa_tmp" ; } 2>/dev/null || echo 0 )
        if [ ! -s "$qa_tmp" ] || [ "$qa_lines" -eq 0 ]; then
            rm -f "$qa_tmp"
            echo "[FAIL] build_vqa_pairs.py produced 0 case record(s);"
            echo "[FAIL] $QA_JSONL left unchanged. Check that $OUT_IMAGES is populated."
            exit 1
        fi
        mv "$qa_tmp" "$QA_JSONL"
        echo "[INFO] QA pairs: $QA_JSONL ($qa_lines case record(s))"
    fi
    echo ""

    # ── Summary ─────────────────────────────────────────────────────────────
    n_pano=$(find "$OUT_IMAGES" -maxdepth 1 -name "*_panoramic.png" | wc -l)
    n_3d_cases=$(find "$OUT_IMAGES" -maxdepth 1 -name "*_3d_captions.json" | wc -l)
    n_3d=$(find "$OUT_IMAGES" -maxdepth 1 \( -name "*_3d_left.png" -o -name "*_3d_frontal.png" -o -name "*_3d_right.png" \) | wc -l)
    n_tooth=$(find "$OUT_IMAGES" -maxdepth 1 -name "*_tooth*_composite.png" | wc -l)
    n_sinus=$(find "$OUT_IMAGES" -maxdepth 1 \( -name "*_sinus_right_detail.png" -o -name "*_sinus_left_detail.png" \) | wc -l)

    echo "[INFO] ========== Done =========="
    echo "[INFO] End: $(date)"
    echo "[INFO] Panoramic       : $n_pano case(s)"
    echo "[INFO] 3D renders      : $n_3d_cases case(s), $n_3d image(s) (3/case)"
    echo "[INFO] Tooth composites: $n_tooth image(s) (1/tooth, up to 32/case)"
    echo "[INFO] Sinus detail    : $n_sinus image(s) (up to 2/case)"
    echo ""
    echo "[INFO] Next -- images and qa_pairs.jsonl are done; the GPU step is inference:"
    echo "  STAGE=infer MODEL_NAME=<served-model-name> RUN_NAME=${RUN_NAME} \\"
    echo "    sbatch scripts/run_infer.sh $SPLIT"

} 2>&1
