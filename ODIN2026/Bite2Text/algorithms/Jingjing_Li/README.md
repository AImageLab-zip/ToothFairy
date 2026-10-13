# Bite2Text

输入上下颌三维扫描和五帧口内照片，输出 23 项结构化牙齿分析结果及英文报告。

已在现有 Ubuntu 22.04 / Python 3.11 容器环境、RTX 3090 上，使用仓库任务权重及准备脚本新下载的官方基础模型完成真实病例推理：得到 23 项结果和通过核验的 Qwen 报告。

## 安装

需要 Linux x86_64、Python 3.11、NVIDIA GPU 和支持 CUDA 12.6 的驱动。建议使用 Ubuntu 22.04 或更新版本。

```bash
sudo apt-get install -y libgl1 libegl1 libglib2.0-0 libgomp1 libxrender1 libxext6 libsm6
python -m pip install -r requirements.txt
```

仓库已包含 DADSN 源码、分割权重及我们训练的任务权重和运行配置，位于 `model/dadsn/`、`model/frozen/`；安装依赖以根目录 `requirements.txt` 为准。克隆后只需下载官方 Qwen 和 ResNet 基础模型：

```bash
python prepare_models.py
```

脚本固定 Qwen3.5-2B 版本 `15852e8c16360a2fea060d615a32b45270f8a8fc` 和 ResNet-18 ImageNet V1 权重，校验下载文件，并自动转换 ResNet 保存格式。约需下载 4.6 GB，不需要手动创建和整理模型目录。

## 运行

准备输入文件：

```text
input/
  upper.stl
  lower.stl
  intraoral.tiff
```

- 上下颌 STL 必须保留共同坐标系和原始咬合关系。
- TIFF 的五帧顺序：正面、左侧、右侧、上颌咬合面、下颌咬合面。

```bash
python run.py
# 可选：指定输入、输出目录及 GPU
python run.py --input-dir /path/to/input --output-dir /path/to/output --gpu 0
```

默认目录为项目根目录，输出目录自动创建。

输出：`output/structured_23.json`（23 项结果）、`output/report.txt`（英文报告）、`output/result.json`（完整结果和报告生成状态）。

## 许可与引用

项目自有代码、文档及自有任务模型贡献采用 [MIT](LICENSE)。DADSN 代码和权重、其他第三方内容、官方基础模型、数据及运行环境不由本项目授权。

- **DADSN**：源码和分割权重随仓库提供，项目负责人已确认作者允许公开，保留原署名。[Oculins/DADSN](https://github.com/Oculins/DADSN)。Yuxian Jiang et al., *Morphology Prior Enhanced Teeth Segmentation for High-Resolution Oral Scans*, IEEE Journal of Biomedical and Health Informatics, 2025。
- **Qwen3.5-2B**：[官方模型](https://huggingface.co/Qwen/Qwen3.5-2B)。通过 Transformers 加载，本仓库不包含 Qwen 权重或官方实现；另行获取模型时遵守其使用条件。
- **torchvision / ResNet-18**：[官方项目](https://github.com/pytorch/vision)。通过安装的 torchvision 库调用，本仓库不包含该库或 ImageNet 基础权重。
