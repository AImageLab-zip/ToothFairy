#!/usr/bin/env python3
"""Aggregate Bits2Bites five-fold metrics for two checkpoint strategies."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("accuracy", "precision", "recall", "f1")
TASKS = ("right_occ", "left_occ", "anterior_bite", "transverse_bite", "midline")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_strategy(exp_root: Path, folds: list[int], strategy: str) -> dict:
    fold_results = {}
    for fold in folds:
        run = exp_root / f"ptv3_mesh_mtl_fold{fold}_seed2026"
        if strategy == "accuracy_best":
            metric_path = run / "result" / "metrics.json"
        elif strategy == "final_epoch":
            metric_path = Path(f"{run}_eval_last") / "result" / "metrics.json"
        else:
            raise ValueError(strategy)
        with metric_path.open() as fp:
            fold_results[str(fold)] = json.load(fp)

    aggregate = {
        "average": {
            metric: summarize(
                [fold_results[str(fold)]["average"][metric] for fold in folds]
            )
            for metric in METRICS
        },
        "per_task": {
            task: {
                metric: summarize(
                    [
                        fold_results[str(fold)]["per_task"][task][metric]
                        for fold in folds
                    ]
                )
                for metric in METRICS
            }
            for task in TASKS
        },
    }
    return {"folds": fold_results, "aggregate": aggregate}


def render_markdown(payload: dict) -> str:
    lines = [
        "# Bits2Bites PTv3 Mesh-only 五折汇总",
        "",
        f"> 生成时间：{payload['generated_at']}",
        "",
        "## 配置",
        "",
        "- PT-v3m1 encoder + 五个多任务分类头",
        "- Mesh-only，200 epochs，batch size 8，seed 2026",
        "- AdamW，初始学习率 1e-4，CosineAnnealingLR，AMP FP16",
        "- 每折 160 例训练、40 例验证",
        "",
        "## 五折总体结果",
        "",
        "| Checkpoint 策略 | Accuracy | Precision | Recall | macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "accuracy_best": "Accuracy 最佳",
        "final_epoch": "最终 epoch",
    }
    for strategy in ("accuracy_best", "final_epoch"):
        avg = payload["strategies"][strategy]["aggregate"]["average"]
        cells = [labels[strategy]] + [
            f"{avg[m]['mean']:.4f} ± {avg[m]['std']:.4f}" for m in METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    recommended = payload["recommended_checkpoint_strategy"]
    lines += [
        "",
        f"按五折平均 macro-F1，推荐策略：**{labels[recommended]}**。",
        "",
        "## 每折 macro-F1",
        "",
        "| Fold | Accuracy 最佳 | 最终 epoch |",
        "|---:|---:|---:|",
    ]
    for fold in payload["folds"]:
        best = payload["strategies"]["accuracy_best"]["folds"][str(fold)]["average"]["f1"]
        last = payload["strategies"]["final_epoch"]["folds"][str(fold)]["average"]["f1"]
        lines.append(f"| {fold} | {best:.4f} | {last:.4f} |")

    task_labels = {
        "right_occ": "右侧矢状向咬合",
        "left_occ": "左侧矢状向咬合",
        "anterior_bite": "前牙覆合/开合",
        "transverse_bite": "横向咬合",
        "midline": "中线",
    }
    lines += [
        "",
        "## 推荐策略的逐任务结果",
        "",
        "| 任务 | Accuracy | Precision | Recall | macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    per_task = payload["strategies"][recommended]["aggregate"]["per_task"]
    for task in TASKS:
        cells = [task_labels[task]] + [
            f"{per_task[task][m]['mean']:.4f} ± {per_task[task][m]['std']:.4f}"
            for m in METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "标准差使用五折样本标准差（n-1）。详细逐折原始值见同目录 JSON。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    strategies = {
        strategy: load_strategy(args.exp_root, args.folds, strategy)
        for strategy in ("accuracy_best", "final_epoch")
    }
    recommended = max(
        strategies,
        key=lambda strategy: strategies[strategy]["aggregate"]["average"]["f1"]["mean"],
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "folds": args.folds,
        "config": {
            "model": "PT-v3m1 multi-task classifier",
            "modality": "mesh-only",
            "epochs": 200,
            "batch_size": 8,
            "seed": 2026,
        },
        "strategies": strategies,
        "recommended_checkpoint_strategy": recommended,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_md.write_text(render_markdown(payload))
    print(json.dumps(payload["strategies"][recommended]["aggregate"]["average"], indent=2))
    print(f"recommended_checkpoint_strategy={recommended}")


if __name__ == "__main__":
    main()
