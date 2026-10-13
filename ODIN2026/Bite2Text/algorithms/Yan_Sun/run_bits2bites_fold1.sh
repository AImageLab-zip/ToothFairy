#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=${BITE2TEXT_PROJECT_ROOT:-${SCRIPT_DIR}}
REPO_DIR=${BITE2TEXT_BITS2BITES_ROOT:-${BITE2TEXT_VENDOR_ROOT:-${PROJECT_ROOT}/.vendor}/Bits2Bites}
RUN_NAME=ptv3_mesh_mtl_fold1_seed2026

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH:-}
export WANDB_MODE=disabled

cd "${REPO_DIR}"
mkdir -p logs
.venv/bin/python tools/dental_fold.py --fold-val 1 --data-root data/dental_landmarks_mesh

.venv/bin/python -u tools/train.py \
  --config-file configs/dental/cls-ptv3-base.py \
  --num-gpus 1 \
  --num-machines 1 \
  --dist-url auto \
  --options \
  save_path=exp/dental/${RUN_NAME} \
  data.train.data_root=data/dental_landmarks_mesh \
  data.val.data_root=data/dental_landmarks_mesh \
  data.test.data_root=data/dental_landmarks_mesh \
  seed=2026 \
  epoch=200 \
  eval_epoch=200 \
  batch_size=8 \
  batch_size_val=8 \
  batch_size_test=8 \
  num_worker=4 \
  enable_wandb=False \
  2>&1 | tee logs/${RUN_NAME}.log
