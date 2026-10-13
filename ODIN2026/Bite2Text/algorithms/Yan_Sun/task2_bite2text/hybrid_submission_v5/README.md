# Bite2Text PTv3 + retrieval hybrid v5

该版本使用 867 例全量训练的 12-head PTv3 预测咬合结构标签，并在几何检索 top-50 候选中按五折 Macro-F1 加权计算硬标签一致性。最终分数为 `cosine_similarity + 0.5 * label_agreement`，输出最高分候选的原始英文报告。

严格五折 OOF 官方 evaluator（867 例，每个查询排除同 fold 候选）结果：

- 纯检索：BLEU-4 0.251328，METEOR 0.452335。
- 混合 v5：BLEU-4 0.264565，METEOR 0.465952。

模型资源目录包含 `model_final.pth`、`config.py`、`head_vocabs.json`、`ios_normalizer_best.pt`、`retrieval_index.npz`、`retrieval_reports.json` 和 `retrieval_labels.json`。
