# Bite2Text PTv3 + photo + retrieval hybrid v6

该版本冻结 v5 的 PTv3 硬标签检索分数，再加入五视图照片模型的概率软证据。最终分数为 `cosine_similarity + 0.5 * PTv3_agreement + 0.2 * photo_agreement`。照片只参与五折 OOF 证实有用的 8 个槽位；照片缺失或解析失败时自动回退到 v5。

严格五折 OOF 官方 evaluator（867 例，每个查询排除同 fold 候选）结果：

- 混合 v5：BLEU-4 0.264565，METEOR 0.465952，combined 0.365258。
- 照片增强 v6：BLEU-4 0.265290，METEOR 0.466686，combined 0.365988。

模型资源目录在 v5 文件基础上新增 `photo_model_final.pt` 和 `photo_view_classifier.pt`。
