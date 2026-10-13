# Bite2Text 口内照五视图流水线

该目录包含最终 v9 的照片训练与评估流水线。完整可执行命令见
[复现指南 B6–B8](../../reproducibility/README.md#b6-photos-view-selection-cv-final-model)，
包括环境安装、视图分类、照片缓存、五折训练和 867 例全量训练。
历史最终配置的 structural3 视图分类器实际训练 **6 epochs**（最佳为 epoch 2），
不是命令行默认的 8 epochs。最终多视图模型训练 10 epochs。
视图分类器并未按外层五折重新训练，因此不能把整条流水线称为完全嵌套验证。

标准视图顺序为：

1. `frontal`：正面咬合
2. `right_buccal`：右侧咬合
3. `left_buccal`：左侧咬合
4. `lower_occlusal`：下颌咬合面
5. `upper_occlusal`：上颌咬合面

视图分类器只使用“恰有五张且文件名为 `intraoral_1` 至 `intraoral_5`”的高置信病例构造弱标签。默认保留五分类审计模式；实际选择推荐使用更稳健的 `structural3` 模式，将容易互换的左右侧合并为 `lateral`、上下咬合面合并为 `occlusal`，最终仍选择 1 张正面、2 张侧面和 2 张咬合面。训练不使用水平翻转，因为翻转会交换左右侧。推理时使用 Hungarian assignment，在一个病例内保证五个槽位不会选中同一张照片；多余照片被过滤，少于五张时保留显式缺失槽位。

主要文件：

- `audit_intraoral_photos.py`：生成原始照片清单、统计报告和抽样联系图。
- `train_view_classifier.py`：训练 ImageNet 预训练 ResNet-18 五分类器，并按病例划分训练/验证集。
- `select_five_views.py`：全量预测、唯一分配并生成固定 Montage。
- `prepare_photo_cache.py`：把最终选中的原图只读转换为紧凑 JPEG 训练缓存。
- `train_multiview_12head.py`：共享 ResNet-18 编码五张照片，以 masked multi-head loss 训练 12 项结构化诊断标签；按既有五折病例划分输出 OOF logits。
- `train_multiview_full.py`：用五折确定的固定 10 轮配置在全部 867 个训练病例上生成最终照片模型。
- `evaluate_photo_ptv3_fusion.py`：合并照片五折 OOF，并用外层 Fold 留出的 cross-fit 权重评估与 PTv3 的逐槽位概率融合。
- `make_crossfit_fused_oof_jsonl.py`：把严格 cross-fit 融合概率写回现有 OOF JSONL，供同一 hybrid retrieval 网格进行 BLEU/METEOR 对比。
- `evaluate_photo_augmented_hybrid.py`：冻结 v5 的 PTv3 hard agreement，仅增加照片 soft agreement，并对照片系数做外层 Fold cross-fit 评估。

原始数据只读，所有派生文件写入单独的输出目录。
