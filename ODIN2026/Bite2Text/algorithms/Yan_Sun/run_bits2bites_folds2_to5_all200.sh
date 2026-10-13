#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=${BITE2TEXT_PROJECT_ROOT:-${SCRIPT_DIR}}
TASK_ROOT=${PROJECT_ROOT}/task2_bite2text
REPO_DIR=${BITE2TEXT_BITS2BITES_ROOT:-${BITE2TEXT_VENDOR_ROOT:-${PROJECT_ROOT}/.vendor}/Bits2Bites}
DATA_ROOT=${REPO_DIR}/data/dental_landmarks_mesh
ALL_ROOT=${REPO_DIR}/data/dental_landmarks_mesh_all200
EXP_ROOT=${REPO_DIR}/exp/dental
STATUS_LOG=${TASK_ROOT}/bits2bites_cv_all200_status.log
UPSTREAM_COMMIT=8c3c685160c9cabe2462e9e23d2ffcd9ca78c63a

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH:-}
export WANDB_MODE=disabled

mkdir -p "${TASK_ROOT}" "${REPO_DIR}/logs"
exec 9>"${TASK_ROOT}/bits2bites_cv_all200.lock"
if ! flock -n 9; then
  echo "Another Bits2Bites CV/all-200 workflow holds the lock." >&2
  exit 1
fi

status() {
  local message=$1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}" | tee -a "${STATUS_LOG}"
}

checkpoint_epoch() {
  local checkpoint=$1
  "${REPO_DIR}/.venv/bin/python" - "${checkpoint}" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
if not path.is_file():
    print(-1)
else:
    print(torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1))
PY
}

train_fold() {
  local fold=$1
  local run_name="ptv3_mesh_mtl_fold${fold}_seed2026"
  local save_dir="${EXP_ROOT}/${run_name}"
  local train_log="${REPO_DIR}/logs/${run_name}.log"

  status "Preparing Fold ${fold}: 160 train / 40 validation"
  cd "${REPO_DIR}"
  .venv/bin/python tools/dental_fold.py \
    --fold-val "${fold}" \
    --data-root "${DATA_ROOT}"

  local epoch
  epoch=$(checkpoint_epoch "${save_dir}/model/model_last.pth")
  if [[ "${epoch}" == "200" ]]; then
    status "Fold ${fold} training already complete; reusing epoch-200 checkpoint"
  else
    status "Starting Fold ${fold} training"
    .venv/bin/python -u tools/train.py \
      --config-file configs/dental/cls-ptv3-base.py \
      --num-gpus 1 \
      --num-machines 1 \
      --dist-url auto \
      --options \
      save_path="exp/dental/${run_name}" \
      data.train.data_root="data/dental_landmarks_mesh" \
      data.val.data_root="data/dental_landmarks_mesh" \
      data.test.data_root="data/dental_landmarks_mesh" \
      fold_val="${fold}" \
      seed=2026 \
      epoch=200 \
      eval_epoch=200 \
      batch_size=8 \
      batch_size_val=8 \
      batch_size_test=8 \
      num_worker=4 \
      enable_wandb=False \
      2>&1 | tee "${train_log}"
  fi

  epoch=$(checkpoint_epoch "${save_dir}/model/model_last.pth")
  if [[ "${epoch}" != "200" ]]; then
    status "ERROR: Fold ${fold} checkpoint epoch is ${epoch}, expected 200"
    exit 1
  fi

  status "Testing Fold ${fold} Accuracy-selected checkpoint"
  sh scripts/test.sh \
    -p .venv/bin/python \
    -d dental \
    -n "${run_name}" \
    -w model_best \
    -g 1 \
    2>&1 | tee "logs/${run_name}_test_best.log"

  status "Testing Fold ${fold} final-epoch checkpoint"
  .venv/bin/python -u tools/test.py \
    --config-file "exp/dental/${run_name}/config.py" \
    --num-gpus 1 \
    --num-machines 1 \
    --dist-url auto \
    --options \
    save_path="exp/dental/${run_name}_eval_last" \
    weight="exp/dental/${run_name}/model/model_last.pth" \
    2>&1 | tee "logs/${run_name}_test_last.log"

  test -s "${save_dir}/result/metrics.json"
  test -s "${EXP_ROOT}/${run_name}_eval_last/result/metrics.json"
  status "Fold ${fold} complete"
}

prepare_all200() {
  status "Preparing all-200 training root"
  mkdir -p "${ALL_ROOT}/train" "${ALL_ROOT}/val"
  ln -sfn "$(realpath "${DATA_ROOT}/labels.csv")" "${ALL_ROOT}/labels.csv"
  for fold in 1 2 3 4 5; do
    for sample in "${DATA_ROOT}/fold_${fold}"/dental_*.json; do
      ln -sfn "$(realpath "${sample}")" "${ALL_ROOT}/train/$(basename "${sample}")"
    done
  done
  local count
  count=$(find "${ALL_ROOT}/train" -maxdepth 1 -name 'dental_*.json' | wc -l)
  if [[ "${count}" != "200" ]]; then
    status "ERROR: all-200 root contains ${count} JSON files"
    exit 1
  fi
  status "All-200 root verified: ${count} training cases"
}

train_all200() {
  local run_name=ptv3_mesh_mtl_all200_seed2026
  local save_dir="${EXP_ROOT}/${run_name}"
  local epoch
  epoch=$(checkpoint_epoch "${save_dir}/model/model_last.pth")
  if [[ "${epoch}" == "200" ]]; then
    status "All-200 training already complete; reusing epoch-200 checkpoint"
  else
    status "Starting unified all-200 PTv3 encoder training"
    cd "${REPO_DIR}"
    .venv/bin/python -u tools/train.py \
      --config-file configs/dental/cls-ptv3-base.py \
      --num-gpus 1 \
      --num-machines 1 \
      --dist-url auto \
      --options \
      save_path="exp/dental/${run_name}" \
      data.train.data_root="data/dental_landmarks_mesh_all200" \
      data.val.data_root="data/dental_landmarks_mesh_all200" \
      data.test.data_root="data/dental_landmarks_mesh_all200" \
      seed=2026 \
      epoch=200 \
      eval_epoch=200 \
      evaluate=False \
      batch_size=8 \
      batch_size_val=8 \
      batch_size_test=8 \
      num_worker=4 \
      enable_wandb=False \
      2>&1 | tee "logs/${run_name}.log"
  fi

  epoch=$(checkpoint_epoch "${save_dir}/model/model_last.pth")
  if [[ "${epoch}" != "200" ]]; then
    status "ERROR: all-200 checkpoint epoch is ${epoch}, expected 200"
    exit 1
  fi

  status "Extracting portable encoder-only checkpoint"
  "${REPO_DIR}/.venv/bin/python" "${PROJECT_ROOT}/extract_bits2bites_encoder.py" \
    --checkpoint "${save_dir}/model/model_last.pth" \
    --output "${save_dir}/model/ptv3_encoder_all200_seed2026.pth" \
    --upstream-commit "${UPSTREAM_COMMIT}" \
    --data-count 200 \
    2>&1 | tee "${REPO_DIR}/logs/${run_name}_extract_encoder.log"
  test -s "${save_dir}/model/ptv3_encoder_all200_seed2026.pth"
  status "Unified all-200 encoder complete"
}

status "Workflow started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if [[ "${BITE2TEXT_BITS_SKIP_CV:-0}" != "1" ]]; then
  for fold in ${BITE2TEXT_BITS_FOLDS:-2 3 4 5}; do
    [[ "${fold}" =~ ^[1-5]$ ]] || { echo "Invalid fold: ${fold}" >&2; exit 1; }
    train_fold "${fold}"
  done

  status "Aggregating five-fold metrics"
  "${REPO_DIR}/.venv/bin/python" "${PROJECT_ROOT}/aggregate_bits2bites_cv.py" \
    --exp-root "${EXP_ROOT}" \
    --output-json "${TASK_ROOT}/Bits2Bites_5fold_summary.json" \
    --output-md "${TASK_ROOT}/Bits2Bites_5fold_summary.md" \
    2>&1 | tee "${REPO_DIR}/logs/ptv3_mesh_mtl_5fold_aggregate.log"
fi

prepare_all200
train_all200
status "WORKFLOW COMPLETE"
