#!/bin/zsh

set -u

eval_dir="${0:A:h}"
cd "$eval_dir" || exit 1

cleanup() {
  unset RADFACT_API_KEY
}
trap cleanup EXIT INT TERM

echo "ODIN Bite2Text RadFact-Lite / GLM-5.2"
read -rs "RADFACT_API_KEY?请输入智谱 API Key（输入不回显）: "
echo

if [[ -z "${RADFACT_API_KEY:-}" ]]; then
  echo "未输入 API Key，评测已取消。"
  exit 2
fi
export RADFACT_API_KEY

echo "[1/2] 正在探测 API 与结构化输出兼容性……"
if ! .venv313/bin/python run_radfact_glm.py \
  --probe \
  --run-dir runs/probe_glm52; then
  echo "API 探测失败，未启动病例评测。"
  exit 1
fi

echo "[2/2] 正在评测固定 10 例……"
.venv313/bin/python run_radfact_glm.py \
  --sample-size 10 \
  --seed 20260813 \
  --run-dir runs/v7_glm52_pilot10

status=$?
if [[ $status -eq 0 ]]; then
  echo "评测完成，结果已保存到 runs/v7_glm52_pilot10。"
else
  echo "评测失败，退出码：$status"
fi
exit $status
