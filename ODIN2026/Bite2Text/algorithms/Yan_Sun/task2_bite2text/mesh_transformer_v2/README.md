# Bite2Text IOS Mesh Transformer v2

第二代 IOS-only 基线，保留 v1 的患者级 split、严格跨报告共识标签和 masked multi-task loss，升级三个部分：

1. **面积加权表面采样**：按 STL 三角面面积进行重心坐标采样，避免 v1 的重复顶点密度偏差。
2. **上下颌接触特征**：每个点计算到对颌采样点的最近距离和相对垂直间隙，并同时使用面法线。
3. **点云 token Transformer**：表面点先按空间邻近关系聚合为每颌 128 个 token，再以 Transformer 做跨颌建模。

这不是 Pointcept/PTv3 的复现；它是无额外编译依赖、可在现有 CUDA Docker 镜像上运行的强一点的对照模型。

## 冒烟训练

```bash
export BITE2TEXT_PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "${BITE2TEXT_PROJECT_ROOT}/task2_bite2text/mesh_transformer_v2"
docker build --network none -t odin2026-bite2text-mesh-transformer-v2:latest .

docker run --rm --gpus 'device=1' --network none --user "$(id -u):$(id -g)" \
  -v "${BITE2TEXT_PROJECT_ROOT}/data/Bite2Text_raw:/data:ro" \
  -v "${BITE2TEXT_PROJECT_ROOT}/task2_bite2text/mesh_baseline/data/manifest_v1:/manifest:ro" \
  -v "$PWD":/workspace \
  odin2026-bite2text-mesh-transformer-v2:latest /opt/mesh_transformer_v2/train.py \
  --manifest /manifest/manifest.jsonl --head-vocabs /manifest/head_vocabs.json \
  --data-root /data --output-dir /workspace/runs/smoke_v2 \
  --epochs 1 --batch-size 2 --num-points 512 --tokens-per-jaw 32 --num-workers 0 --max-samples 8
```

正式首轮建议：`--epochs 30 --batch-size 4 --num-points 2048 --tokens-per-jaw 128 --num-workers 4`，完成后使用 `evaluate.py` 对 `best.pt` 进行独立测试。
