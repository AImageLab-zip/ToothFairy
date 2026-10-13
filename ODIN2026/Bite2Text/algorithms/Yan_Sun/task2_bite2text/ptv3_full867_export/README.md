# Bite2Text PTv3 v3 全量模型导出

该目录对应 867 例全量数据的最终模型。训练采用两阶段方案：冻结 encoder 训练 10 epochs，随后联合训练 47 epochs。47 来自五折最佳 epoch（17、59、47、38、48）的中位数。

由于全量训练没有留出验证集，最终权重固定使用 Stage 2 的 `model_last.pth`，在本目录中命名为 `model_final.pth`。checkpoint 内的 `best_metric_value = -inf` 是关闭验证的预期结果，不代表训练失败。

文件用途：

- `model_final.pth`：最终 12-head PTv3 权重。
- `config.py`：模型和预处理配置。
- `head_vocabs.json`：12 个任务头的类别顺序。
- `class_weights.json`：全量训练类别权重。
- `full_dataset_audit.json`：867 例数据构建审计。
- `export_audit.json`：导出元数据、文件大小和 SHA-256。

在将模型放入提交容器前，应先以 `export_audit.json` 校验哈希，并通过 OOF 实验确定它用于直接结构化报告，还是仅用于约束检索候选排序。
