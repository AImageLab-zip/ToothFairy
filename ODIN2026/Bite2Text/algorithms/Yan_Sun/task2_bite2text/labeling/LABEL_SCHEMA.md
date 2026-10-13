# Bite2Text 英文报告弱标签规范（v2）

本目录的解析器将英文报告转成**可追溯的弱监督标签**。它不是人工标注的临床真值：训练前应根据 `parse_warnings`、标签覆盖率和抽样人工复核继续清洗。

## 核心字段

| 字段 | 标签值 |
|---|---|
| `overjet` | `normal`, `increased`, `reduced`, `negative`, `edge_to_edge` |
| `vertical_relation` | `normal`, `increased`, `reduced`, `deep_bite`, `open_bite` |
| `midline_relation` | `coincident`, `slightly_deviated`, `deviated` |
| `crossbite` | `none`, `anterior`, `posterior`, `present_unspecified` |
| `maxillary_constriction` | `mild`, `present`, `severe` |
| `*_molar_relation`, `*_canine_relation` | `class_i`, `class_ii_edge_to_edge`, `class_ii_full`, `class_ii_unspecified`, `class_iii` |
| `upper_crowding`, `lower_crowding` | `none`, `mild`, `mild_to_moderate`, `moderate`, `moderate_to_severe`, `severe` |
| `upper_spacing`, `lower_spacing` | `present` |
| `curve_spee`, `curve_wilson` | `normal`, `increased` |

`null` 表示文本中没有被规则可靠解析，不代表临床正常或阴性。

## 文件

- `report_records.jsonl`：每条英文报告的文本、原始相对路径、SHA-256、标签、切分及警告。
- `report_labels.csv`：适合数据分析和训练读取的扁平标签表。
- `patient_splits.csv`：按患者而非报告划分的可复现 train/val/test 划分。
- `case_report_index.csv`：病例与每种报告数的索引。
- `parse_audit.json`：覆盖率、类别分布、警告计数和数据范围。

## 训练使用规则

1. 仅使用 `split` 为 `train`、`val`、`test` 的记录；`excluded_incomplete` 不参与训练或模型选择。
2. 一个任务只使用该任务有明确标签的记录；例如没有 `right_molar_relation` 的记录应从对应头的损失中 mask。
3. 先抽样复核高频标签和 `conflict_*` 记录，再将规则标签作为监督信号。
4. 拆分严格按 `patient_id`，不要把同一患者的 IOS 和照片报告放入不同集合。
