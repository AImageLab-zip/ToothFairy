#!/usr/bin/env bash
# =============================================================================
# judge_server.sh — a PERSISTENT RadFact judge (text-only Qwen3-14B on vLLM).
#
# WHY THIS EXISTS
# ---------------
# A judge that starts its own vLLM, evaluates and then kills it pays the 15-25
# min model load on EVERY evaluation. This script does ONLY the serving half.
# Start it once, then run scripts/run_eval.sh as many times as you like against
# it; each run costs just the RadFact calls.
#
# THE ONE THING THAT TRIPS PEOPLE UP
# -----------------------------------
# The server runs on a GPU COMPUTE node (s0-n0x), not on the login node where
# you run the evaluation. So http://localhost:8001 will NOT reach it -- the
# client has to use this node's hostname. That is exactly what
# scripts/run_eval.sh resolves for you, and this script prints the ready-to-paste
# URL into its log.
#
# Usage:
#   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/judge_server.sh
#
#   # then, from the login node, any number of times:
#   scripts/run_eval.sh outputs/<arm>_validate
#
#   # when finished (it does NOT stop by itself before --time):
#   scancel -n judge
#
# Runs for 8 hours, then exits and releases the GPU. Raise --time if you want
# longer, but do not leave a GPU parked on a shared cluster unattended -- the
# gpu partition has no time limit of its own, so this cap is the only thing
# stopping it.
# =============================================================================

#SBATCH --job-name=judge
#SBATCH --partition=gpu
#SBATCH --qos=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/judge_server_%j.log
#SBATCH --error=logs/judge_server_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_ToothFairy4}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
CONTAINER="${SIF_PATH:-$HOME/containers/extraction.sqsh}"
QWEN_TEXT_MODEL_NAME="${QWEN_TEXT_MODEL_NAME:-Qwen3-14B}"
SERVED_NAME="${SERVED_NAME:-qwen3-14b-text}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Weights first, KV cache with what is left. 0.85 of an 80GB card is 68G,
# which is comfortable for the 28G Qwen3-14B default and tight for a 62G
# 32B -- raise it there (0.92 -> 73G) rather than discovering the OOM after
# the load. Override per job, do not edit the default.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

mkdir -p "$PROJECT_DIR/logs"

# ── Preflight: fail loudly now, not after a 20-minute model load ────────────
if [ ! -d "$MODEL_DIR/$QWEN_TEXT_MODEL_NAME" ]; then
    echo "[FAIL] Judge model not found: $MODEL_DIR/$QWEN_TEXT_MODEL_NAME"; exit 1
fi
if [ ! -f "$CONTAINER" ]; then
    echo "[FAIL] Container not found: $CONTAINER"; exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[FAIL] No GPU visible on $(hostname). Submit with:"
    echo "[FAIL]   sbatch --partition=gpu --qos=a100 --gres=gpu:a100:1 scripts/judge_server.sh"
    exit 1
fi

NODE="$(hostname -s)"
echo "============================================================"
echo "  RadFact judge server"
echo "  node       : $NODE"
echo "  model      : $QWEN_TEXT_MODEL_NAME  (served as '$SERVED_NAME')"
echo "  model dir  : $MODEL_DIR"
echo "  gpu mem    : $GPU_MEM_UTIL"
echo "  port       : $PORT"
echo "  time limit : $(squeue -j "${SLURM_JOB_ID:-0}" -h -o %l 2>/dev/null || echo '8:00:00')"
echo "============================================================"
nvidia-smi | head -10 || true
echo ""

# The URL the CLIENT must use -- not localhost, this node.
JUDGE_URL="http://${NODE}:${PORT}/v1"

# Advertise the address so run_eval.sh (and you) can find it without parsing
# squeue. Written before the server is up; the readiness marker is separate.
mkdir -p "$PROJECT_DIR/.judge"
printf '%s\n' "$JUDGE_URL" > "$PROJECT_DIR/.judge/url"
printf '%s\n' "${SLURM_JOB_ID:-unknown}" > "$PROJECT_DIR/.judge/jobid"
rm -f "$PROJECT_DIR/.judge/ready"

cleanup() {
    echo "[INFO] Judge server shutting down; clearing $PROJECT_DIR/.judge"
    rm -f "$PROJECT_DIR/.judge/url" "$PROJECT_DIR/.judge/ready" "$PROJECT_DIR/.judge/jobid"
}
trap cleanup EXIT

echo "[INFO] Starting vLLM (A100 load takes 15-25 min)..."
srun \
    --overlap \
    --container-image="$CONTAINER" \
    --container-mounts="$MODEL_DIR:/models" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/$QWEN_TEXT_MODEL_NAME" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" &
VLLM_PID=$!

# ── Wait for readiness, then hold the allocation open ───────────────────────
for i in $(seq 1 180); do          # 180 x 10s = 30 min
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[FAIL] vLLM died during startup (see the .err log)."; exit 1
    fi
    if curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        touch "$PROJECT_DIR/.judge/ready"
        echo ""
        echo "[PASS] Judge ready after $((i * 10))s"
        echo "============================================================"
        echo "  Evaluate now, from the LOGIN node:"
        echo "    scripts/run_eval.sh outputs/<arm>_validate"
        echo ""
        echo "  Or point any client at:"
        echo "    --judge-backend vllm --vllm-url $JUDGE_URL --model $SERVED_NAME"
        echo ""
        echo "  Stop it with:  scancel -n judge"
        echo "============================================================"
        break
    fi
    [ $((i % 6)) -eq 0 ] && echo "  ...still loading ($((i * 10))s)"
    sleep 10
done

if [ ! -f "$PROJECT_DIR/.judge/ready" ]; then
    echo "[FAIL] vLLM never answered /health within 30 min."; exit 1
fi

# Block until vLLM exits or SLURM hits --time. Nothing else to do.
wait "$VLLM_PID"
