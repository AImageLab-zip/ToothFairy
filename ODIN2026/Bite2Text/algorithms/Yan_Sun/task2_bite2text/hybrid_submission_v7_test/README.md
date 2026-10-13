# Bite2Text PTv3 + photo + retrieval + midline correction v7

该版本在 v6 基础上增加唯一通过外层五折选择的事实纠错：当 PTv3 中线置信度不低于 0.45 时，用确定性中线句替换检索报告中的中线句。其他事实类别均不修改。

严格五折 OOF 官方 evaluator（867 例，每个查询排除同 fold 候选）结果：

- 混合 v5：BLEU-4 0.264565，METEOR 0.465952，combined 0.365258。
- 照片增强 v6：BLEU-4 0.265290，METEOR 0.466686，combined 0.365988。
- 中线纠错 v7：BLEU-4 0.267678，METEOR 0.469383，combined 0.368531。

模型资源目录在 v5 文件基础上新增 `photo_model_final.pt` 和 `photo_view_classifier.pt`。
