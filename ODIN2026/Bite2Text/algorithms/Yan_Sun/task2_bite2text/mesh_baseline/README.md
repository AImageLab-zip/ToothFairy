# Bite2Text Mesh-only 多头基线

这是一个轻量、可复现的 **Bits2Bites 风格**基线：上、下颌 STL 分别采样为点云，经共享 PointNet 编码器和融合层，预测 7 个报告弱标签头。

它刻意不复刻 Pointcept/PTv3 的编译依赖，也不使用牙齿 landmark 或照片；目标是尽快验证本任务的 IOS 信号、患者级切分和 masked multi-task loss。参考实现固定在 `references/Bits2Bites`，但本目录是独立的起跑基线。

## 头与标签

- `right_molar_relation`、`right_canine_relation`
- `left_molar_relation`、`left_canine_relation`
- `overjet`、`vertical_relation`、`midline_relation`

当一个患者的多个 IOS 报告对于某字段不一致、报告内部对应字段有冲突，或字段缺失时，`prepare_manifest.py` 将其目标编码为 `-100`。训练时该字段不会产生 loss。

`prepare_manifest.py` 默认仍生成上述 7 个主要头。`--heads` 可显式选择其他经过数据覆盖审计的辅助报告头：`crossbite`、`upper_crowding`、`lower_crowding`、`curve_spee`、`curve_wilson`。该机制用于训练独立的报告扩展模型，避免改变已经冻结的主模型与提交镜像。

## 构建与运行

在仓库根目录设置 `BITE2TEXT_PROJECT_ROOT`，然后进入本目录执行。Docker 镜像使用 PyTorch CUDA 基础镜像；训练和推理均可在禁用网络的容器中运行。

```bash
export BITE2TEXT_PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "${BITE2TEXT_PROJECT_ROOT}/task2_bite2text/mesh_baseline"
docker build --network none -t odin2026-bite2text-mesh-baseline:latest .

# 创建一次训练清单；output 目录必须不存在
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "${BITE2TEXT_PROJECT_ROOT}/data/Bite2Text_raw:/data:ro" \
  -v "${BITE2TEXT_PROJECT_ROOT}/task2_bite2text/structured_labels_v2:/labels:ro" \
  -v "$PWD":/workspace \
  odin2026-bite2text-mesh-baseline:latest /opt/baseline/prepare_manifest.py \
  --labels-csv /labels/report_labels.csv --data-root /data --output-dir /workspace/data/manifest_v1
```

GPU 1 冒烟训练（8 个病例、1 epoch）：

```bash
docker run --rm --gpus 'device=1' --network none --user "$(id -u):$(id -g)" \
  -v "${BITE2TEXT_PROJECT_ROOT}/data/Bite2Text_raw:/data:ro" \
  -v "$PWD":/workspace \
  odin2026-bite2text-mesh-baseline:latest /opt/baseline/train.py \
  --manifest /workspace/data/manifest_v1/manifest.jsonl \
  --head-vocabs /workspace/data/manifest_v1/head_vocabs.json \
  --data-root /data --output-dir /workspace/runs/smoke_v1 \
  --epochs 1 --batch-size 2 --num-points 512 --num-workers 0 --max-samples 8 --test-after-train
```

改进版正式对照建议：`--epochs 100 --warmup-epochs 5 --patience 20 --batch-size 8 --num-points 4096 --num-workers 4`。训练集每个 epoch 会重新采样点云并重新进行旋转/扰动；验证集保持确定性。先检查验证集的每头样本数和 macro-F1，再用下方独立评估命令在测试集运行最佳 checkpoint。

类别失衡消融可在其他参数不变的前提下加入 `--loss focal --focal-gamma 2`。该实现保留原有平方根逆频率类别权重；仅改变训练 loss，方便与默认 `weighted_ce` 做单变量比较。

跨颌几何消融可加入 `--geometry-features --contact-points 256`。该分支在 joint-normalized 的上下颌点云中均匀选择 256 点，融合双向最近邻距离分位数、水平/垂直偏移统计、上下颌高度和质心差；旧 checkpoint 默认不启用该分支，评估脚本会从 checkpoint 配置自动恢复正确架构。

若几何探针显示统计本身有信号，可额外使用 `--direct-geometry-heads`：经过运行统计标准化的 34 维几何向量将经线性层直接产生各预测头的残差 logits，再与 PointNet logits 相加。该选项必须与 `--geometry-features` 同时启用。

可用 `--direct-geometry-exclude-heads head_a,head_b` 让指定头保留 PointNet logits，不叠加直接几何残差；该选择必须只按验证集的逐头结果决定。

如需判断这些 34 维统计本身是否具有标签信号，可用 `probe_geometry.py` 训练一个只看几何统计的线性或小 MLP 多头探针；它不使用 PointNet 表面特征，适合在新增几何架构前做诊断。

如需独立、可重跑的测试评估（推荐），使用：

```bash
docker run --rm --gpus 'device=1' --network none --user "$(id -u):$(id -g)" \
  -v "${BITE2TEXT_PROJECT_ROOT}/data/Bite2Text_raw:/data:ro" \
  -v "$PWD":/workspace \
  odin2026-bite2text-mesh-baseline:latest /opt/baseline/evaluate.py \
  --manifest /workspace/data/manifest_v1/manifest.jsonl \
  --head-vocabs /workspace/data/manifest_v1/head_vocabs.json \
  --data-root /data --checkpoint /workspace/runs/<run>/best.pt \
  --output-dir /workspace/runs/<run>/test_metrics --split test \
  --batch-size 8 --num-points 1024 --num-workers 0
```

同配置 checkpoint 可用概率平均集成：每个预测头先对各模型的 softmax 概率取均值，再输出 argmax。`evaluate_ensemble.py` 也接受单个 checkpoint，用于评估单模型的确定性采样 TTA。先在 `--split val` 验证策略，再在独立 `--split test` 上报告指标；两次均应固定同一 `--seed` 和 `--num-points 4096`。

```bash
docker run --rm --gpus 'device=1' --network none --user "$(id -u):$(id -g)" \
  -v "${BITE2TEXT_PROJECT_ROOT}/data/Bite2Text_raw:/data:ro" \
  -v "$PWD":/workspace \
  odin2026-bite2text-mesh-baseline:latest /opt/baseline/evaluate_ensemble.py \
  --manifest /workspace/data/manifest_v1/manifest.jsonl \
  --head-vocabs /workspace/data/manifest_v1/head_vocabs.json \
  --data-root /data \
  --checkpoints /workspace/runs/<run_a>/best.pt /workspace/runs/<run_b>/best.pt /workspace/runs/<run_c>/best.pt \
  --output-dir /workspace/runs/ensemble_<split> --split val \
  --batch-size 8 --num-points 4096 --num-workers 0 --seed 20260807
```

`evaluate_ensemble.py` 的 `--tta-samples N` 会对每个病例使用 `N` 个确定性的点采样视角，并同时对模型和视角的 softmax 概率做平均。`N>1` 只能在验证集有明确收益后再固定到官方推理入口的 `BITE2TEXT_TTA_SAMPLES`。

## 官方容器推理入口

`inference.py` 遵循官方 Bite2Text socket 合约：读取 `/input/files/ios-upper/*.stl`、`/input/files/ios-lower/*.stl` 与 `inputs.json`，并写出 `/output/diagnostic-imaging-report.json`。当前模型只使用 IOS；照片 socket 被接受但尚未纳入模型。它以三个 checkpoint 的类别概率平均生成七项已验证标签对应的英文报告。

镜像内默认从 `/opt/ml/model/checkpoints/*.pt` 读取模型。开发/冒烟时可用以冒号分隔的绝对容器路径覆盖它。当前验证集选择的默认值是 `BITE2TEXT_TTA_SAMPLES=4`；可显式设为 `1` 做消融或设为其他正整数复现实验：

```bash
docker run --rm --gpus 'device=1' --network none --user "$(id -u):$(id -g)" \
  -e BITE2TEXT_CHECKPOINTS=/workspace/runs/<seed_a>/best.pt:/workspace/runs/<seed_b>/best.pt:/workspace/runs/<seed_c>/best.pt \
  -v "$PWD":/workspace:ro \
  -v /path/to/one/official/case:/input:ro -v /tmp/bite2text_output:/output \
  odin2026-bite2text-mesh-baseline:latest /opt/baseline/inference.py
```

正式上传时将三个经验证 checkpoint 放到算法镜像的 `/opt/ml/model/checkpoints/` 中即可；输出 JSON 保持严格的 `{"report": "..."}` 结构。

用于镜像封装的权重应先经 `export_inference_checkpoints.py` 导出。该脚本只保留模型权重、词表和架构配置，移除 optimizer 等训练状态；`Dockerfile.submission` 会将输出目录 `models/direct_geometry_ensemble/` 直接复制到 `/opt/ml/model/checkpoints/`。这保证上传镜像不依赖服务器上的 `runs/` 挂载。

如果后续的报告扩展模型经验证可以纳入，可将它的 checkpoint 用 `BITE2TEXT_EXTENSION_CHECKPOINTS`（冒号分隔路径）传入同一入口。扩展模型的头必须与主模型不重叠；已支持 crossbite、上下颌 crowding、Curve of Spee 和 Curve of Wilson 的英文渲染。未设置该变量时，入口行为与主七头提交完全一致。
