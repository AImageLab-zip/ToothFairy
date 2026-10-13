#!/usr/bin/env python3
"""Summarize the best validation epoch from each Bite2Text CV training log."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TASK_PATTERN = re.compile(
    r"Task (?P<index>\d+) \((?P<name>[^)]+)\): "
    r"Acc: (?P<accuracy>[0-9.]+) \| Prec: (?P<precision>[0-9.]+) \| "
    r"Rec: (?P<recall>[0-9.]+) \| F1: (?P<f1>[0-9.]+)"
)
AGG_PATTERN = re.compile(
    r"Aggregated metrics: Acc: (?P<accuracy>[0-9.]+) \| "
    r"Prec: (?P<precision>[0-9.]+) \| Rec: (?P<recall>[0-9.]+) \| "
    r"F1: (?P<f1>[0-9.]+)"
)


def parse_log(path: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        task_match = TASK_PATTERN.search(line)
        if task_match:
            row: dict[str, Any] = {
                "index": int(task_match.group("index")),
                "name": task_match.group("name"),
            }
            row.update(
                {name: float(task_match.group(name)) for name in ("accuracy", "precision", "recall", "f1")}
            )
            tasks.append(row)
            continue
        aggregate_match = AGG_PATTERN.search(line)
        if aggregate_match:
            if not tasks:
                raise RuntimeError(f"Aggregate metric without task rows in {path}")
            blocks.append(
                {
                    "epoch": len(blocks) + 1,
                    "tasks": tasks,
                    "aggregate": {
                        name: float(aggregate_match.group(name))
                        for name in ("accuracy", "precision", "recall", "f1")
                    },
                }
            )
            tasks = []
    if tasks:
        raise RuntimeError(f"Incomplete evaluation block in {path}")
    if not blocks:
        raise RuntimeError(f"No evaluation blocks found in {path}")
    return blocks


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    folds: list[dict[str, Any]] = []
    for fold in range(1, args.folds + 1):
        matches = sorted(
            exp_root.glob(f"{args.experiment_prefix}_fold{fold}_stage2_joint_seed*")
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one stage-2 directory for fold {fold}, found {matches}")
        log_path = matches[0] / "train.log"
        blocks = parse_log(log_path)
        best = max(blocks, key=lambda block: block["aggregate"]["f1"])
        folds.append(
            {
                "fold": fold,
                "experiment": matches[0].name,
                "evaluated_epochs": len(blocks),
                "best_epoch": best["epoch"],
                "best": best["aggregate"],
                "tasks": best["tasks"],
                "final": blocks[-1]["aggregate"],
                "model_best": str(matches[0] / "model" / "model_best.pth"),
            }
        )

    best_epochs = [row["best_epoch"] for row in folds]
    selected_stage2_epochs = int(statistics.median(best_epochs))
    aggregate_summary = {
        metric: mean_std([row["best"][metric] for row in folds])
        for metric in ("accuracy", "precision", "recall", "f1")
    }
    task_names = [row["name"] for row in folds[0]["tasks"]]
    per_task = []
    for index, task_name in enumerate(task_names):
        per_task.append(
            {
                "index": index,
                "name": task_name,
                **{
                    metric: mean_std([row["tasks"][index][metric] for row in folds])
                    for metric in ("accuracy", "precision", "recall", "f1")
                },
            }
        )
    summary = {
        "experiment_prefix": args.experiment_prefix,
        "folds": folds,
        "best_epoch_statistics": {
            "values": best_epochs,
            "mean": statistics.mean(best_epochs),
            "median": statistics.median(best_epochs),
            "selected_full_stage2_epochs": selected_stage2_epochs,
        },
        "aggregate": aggregate_summary,
        "per_task": per_task,
        "selected_full_training": {
            "stage1_frozen_epochs": args.stage1_epochs,
            "stage2_joint_epochs": selected_stage2_epochs,
            "selection_basis": "median of five best stage-2 validation epochs",
            "evaluation_during_full_training": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Bite2Text PTv3 v3 五折汇总与全量配置选择",
        "",
        "## 五折最佳结果",
        "",
        "| Fold | 最佳 epoch | Accuracy | Precision | Recall | Macro-F1 | 最终 epoch F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['fold']} | {row['best_epoch']} | {row['best']['accuracy']:.4f} | "
            f"{row['best']['precision']:.4f} | {row['best']['recall']:.4f} | "
            f"{row['best']['f1']:.4f} | {row['final']['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"五折最佳 Macro-F1：**{aggregate_summary['f1']['mean']:.4f} ± "
            f"{aggregate_summary['f1']['std']:.4f}**。",
            "",
            "## 各任务五折最佳 checkpoint 汇总",
            "",
            "| 任务 | Accuracy | Macro-F1 |",
            "|---|---:|---:|",
        ]
    )
    for row in per_task:
        lines.append(
            f"| {row['name']} | {row['accuracy']['mean']:.4f} ± {row['accuracy']['std']:.4f} | "
            f"{row['f1']['mean']:.4f} ± {row['f1']['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 全量训练选择",
            "",
            f"- 五折最佳 epoch：{' / '.join(str(value) for value in best_epochs)}。",
            f"- Stage 1：冻结 encoder，训练 {args.stage1_epochs} epochs。",
            f"- Stage 2：联合训练 {selected_stage2_epochs} epochs（五折最佳 epoch 中位数）。",
            "- 全部 867 个至少有一个有效官方标签的病例均进入训练集。",
            "- 全量阶段关闭验证，不用训练集指标选择 checkpoint；最终使用固定 epoch 的 model_last。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "selected_stage2_epochs": selected_stage2_epochs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

