#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=${BITE2TEXT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
TASK_ROOT=${PROJECT_ROOT}/task2_bite2text
REPO_DIR=${TASK_ROOT}/Bits2Bites
CONFIG_DIR=${REPO_DIR}/configs/dental
STATUS_LOG=${TASK_ROOT}/bite2text_ptv3_staged_status.log

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH:-}
export WANDB_MODE=disabled

mkdir -p "${REPO_DIR}/logs"
exec 9>"${TASK_ROOT}/bite2text_ptv3_staged.lock"
if ! flock -n 9; then
  echo "Another Bite2Text PTv3 staged workflow holds the lock." >&2
  exit 1
fi

status() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${STATUS_LOG}"
}

checkpoint_epoch() {
  "${REPO_DIR}/.venv/bin/python" - "$1" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
print(torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1) if path.is_file() else -1)
PY
}

train_config() {
  local config=$1
  local run_name=$2
  local expected_epoch=$3
  local checkpoint="${REPO_DIR}/exp/dental/${run_name}/model/model_last.pth"
  if [[ "$(checkpoint_epoch "${checkpoint}")" == "${expected_epoch}" ]]; then
    status "${run_name} already complete; reusing epoch ${expected_epoch}"
    return
  fi
  status "Starting ${run_name} (${expected_epoch} epochs)"
  cd "${REPO_DIR}"
  .venv/bin/python -u tools/train.py \
    --config-file "${CONFIG_DIR}/${config}" \
    --num-gpus 1 --num-machines 1 --dist-url auto \
    2>&1 | tee "logs/${run_name}.log"
  local actual
  actual=$(checkpoint_epoch "${checkpoint}")
  if [[ "${actual}" != "${expected_epoch}" ]]; then
    status "ERROR: ${run_name} checkpoint epoch=${actual}, expected=${expected_epoch}"
    exit 1
  fi
  status "Completed ${run_name}"
}

status "Workflow started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
train_config bite2text_ptv3_stage1_frozen.py bite2text_ptv3_stage1_frozen_seed2026 15
train_config bite2text_ptv3_stage2_joint.py bite2text_ptv3_stage2_joint_seed2026 85
status "WORKFLOW COMPLETE"
