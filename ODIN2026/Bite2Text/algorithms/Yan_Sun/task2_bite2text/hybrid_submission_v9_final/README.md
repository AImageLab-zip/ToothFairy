# Bite2Text v9 final：conservative fact-risk reranking

v9 冻结 v8a.2 的全量 PTv3、口内照模型、检索库、中线纠错和 precision sanitizer，仅在检索决策层增加保守的事实风险重排。

原始 v8a.2 分数仍为：

```text
cosine + 0.5 * PTv3 hard agreement + 0.2 * photo soft agreement
```

当原始最高分报告的 unsupported-pattern 句数至少为 `5`，或者高置信矛盾风险至少为 `0.01` 时，触发风险重排。矛盾证据仅累计 PTv3 置信度超出 `0.65` 的部分。
候选来自几何 top-50，并要求原始融合分数落后不超过 `0.02`，不是仅比较前两名。风险分数为：

```text
original_score - 0.005 * unsupported_sentences - 0.5 * contradiction_risk
```

最终门控还要求：

- 在 unsupported 触发路径下降低 unsupported-pattern 句数，或者在 contradiction 触发路径使矛盾风险至少降低 `0.015`；
- 替代报告不能增加 unsupported sentence 数；
- 其余病例保持 v8a.2 选择不变；
- `BITE2TEXT_RISK_RERANK=0` 可精确回退至 v8a.2。

这里的 unsupported 是预定义规则匹配，不是独立临床事实核验，不能保证报告每句话都正确。照片缺失槽位会被掩码处理；无可用照片或照片推理失败时，默认回退到几何评分。

## 历史五折开发评估（867 例）

完整复现步骤与验证记录见[复现指南](../../reproducibility/README.md)。
任务头训练和检索候选按病例分折，但共享预训练、视图分类器和描述符统计量，
因此并非整条流水线完全嵌套的独立验证。以下历史分数不等于本次已重新训练获得的结果。

以下是用官方文本 evaluator 对开发集 OOF 输出计算的历史结果，不是隐藏测试集成绩：

| 版本 | BLEU-4 | METEOR | Combined |
|---|---:|---:|---:|
| v8a.2 | 0.267154 | 0.469131 | 0.368143 |
| v9 | 0.268426 | 0.470040 | 0.369233 |
| 增量 | +0.001272 | +0.000909 | +0.001090 |

v9 仅改变 8/867 例，五个 OOF fold 的 Combined 全部提升。

RadFact-Lite/GLM-5.2 对全部 8 个受影响病例的成对评估：

- v8a.2：Precision 0.2973，Recall 0.2364，F1 0.2634；
- v9：Precision 0.3700，Recall 0.3659，F1 0.3680；
- 病例级：7 胜、0 平、1 负，0 个 LLM failure。

这只是受影响子集上的本地代理评估，不是全部 867 例的 RadFact 均值，也不能替代主办方最终 RadFact 成绩。

## 镜像

镜像标签：`odin2026-bite2text-hybrid-photo-test-v9:latest`。
