# Bite2Text v8a：precision sanitizer

v8a 保留 v7 的 PTv3、照片融合、检索和中线纠错，新增确定性高精度报告过滤层。模型权重不变。为控制 BLEU/METEOR 损失，过滤器只在一份报告检测到至少5个独立高风险句子时启用；包含核心咬合槽位的复合句留给后续结构化渲染处理，不做破坏性整句删除。

过滤层删除当前12头几何/照片模型无法定位验证的高风险检索事实：

- 牙号级修复、牙冠、龋坏、缺牙、萌出和附件；
- 牙号级白斑、脱矿和交叉咬合；
- 无牙号但仍属于修复、龋坏、缺牙、附件等高风险事实；
- 完全重复的句子。

保留结构化咬合、牙弓/曲线以及牙龈退缩、plaque、calculus、gingival inflammation 等牙周卫生描述。首次 RadFact 消融证明整类删除牙周事实会损失 Recall，因此 v8a.2 明确保留它们。每次删除的句子和原因写入 stdout 的 `precision_sanitizer` 字段，不进入官方报告 JSON。

环境变量 `BITE2TEXT_PRECISION_SANITIZER=0` 可关闭过滤，用于严格回归 v7。

## 离线验证

```bash
python test_report_sanitizer.py

python ../photo_pipeline/build_precision_sanitizer_oof.py \
  --input ../photo_pipeline/fact_correction_crossfit_v1/crossfit_predictions.jsonl \
  --output-dir ../photo_pipeline/precision_sanitizer_v8a_oof
```

## 镜像

镜像标签：`odin2026-bite2text-hybrid-photo-test-v8a2:latest`。
