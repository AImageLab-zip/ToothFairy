#!/usr/bin/env python3
"""Generate reproducible Pointcept configs for staged Bite2Text fine-tuning."""

from __future__ import annotations

import argparse
import json
import pprint
from pathlib import Path


DEFAULT_TASK_NAMES = [
    "right_molar_relation",
    "right_canine_relation",
    "left_molar_relation",
    "left_canine_relation",
    "overjet",
    "vertical_relation",
    "midline_relation",
]


def config_text(
    *,
    data_root: Path,
    weight: str,
    save_path: str,
    class_weights: list[list[float]],
    num_classes: list[int],
    freeze_backbone: bool,
    epochs: int,
    batch_size: int,
    max_samples: int,
    stage2: bool,
    task_names: list[str],
    seed: int,
    evaluate: bool = True,
) -> str:
    hooks = [
        {
            "type": "CheckpointLoader",
            "keywords": "module.",
            "replacement": "module." if stage2 else "module.backbone.",
            "strict": bool(stage2),
        },
        {"type": "IterationTimer"},
        {"type": "InformationWriter"},
        {"type": "MultiClsEvaluator"},
        {"type": "CheckpointSaver", "save_freq": None},
    ]
    train_transforms = [
        {"type": "NormalizeCoord"},
        {"type": "RandomScale", "scale": [0.95, 1.05]},
        {"type": "RandomShift", "shift": ((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))},
        {"type": "RandomRotate", "angle": [-0.1, 0.1], "axis": "z", "center": [0, 0, 0], "p": 0.5},
        {"type": "RandomDropout", "dropout_ratio": 0.35, "dropout_application_ratio": 0.5},
        {"type": "GridSample", "grid_size": 0.01, "hash_type": "fnv", "mode": "train", "return_grid_coord": True},
        {"type": "ShufflePoint"},
        {"type": "ToTensor"},
        {
            "type": "Collect",
            "keys": ("coord", "grid_coord", *[f"label_{i}" for i in range(len(task_names))]),
            "feat_keys": ["coord", "point_label_onehot"],
        },
    ]
    eval_transforms = [
        {"type": "NormalizeCoord"},
        {"type": "GridSample", "grid_size": 0.01, "hash_type": "fnv", "mode": "train", "return_grid_coord": True},
        {"type": "ToTensor"},
        {
            "type": "Collect",
            "keys": ("coord", "grid_coord", "name", *[f"label_{i}" for i in range(len(task_names))]),
            "feat_keys": ["coord", "point_label_onehot"],
        },
    ]
    data = {
        "names": task_names,
        "train": {
            "type": "Bite2TextDataset",
            "split": "train",
            "data_root": str(data_root),
            "transform": train_transforms,
            "loop": 1,
            "max_samples": max_samples,
        },
        "val": {
            "type": "Bite2TextDataset",
            "split": "val",
            "data_root": str(data_root),
            "test_mode": False,
            "transform": eval_transforms,
            "max_samples": max_samples,
        },
        "test": {
            "type": "Bite2TextDataset",
            "split": "test",
            "data_root": str(data_root),
            "test_mode": False,
            "transform": eval_transforms,
            "max_samples": max_samples,
        },
    }
    model = {
        "type": "MultiTaskClassifier",
        "num_classes_list": num_classes,
        "class_weights": class_weights,
        "loss_type": "ce",
        "backbone_embed_dim": 128,
        "freeze_backbone": freeze_backbone,
        "backbone": {
            "type": "PT-v3m1",
            "in_channels": 9,
            "enc_channels": (16, 32, 48, 64, 128),
            "enc_num_head": (1, 2, 3, 4, 8),
            "dec_channels": (32, 32, 64, 96),
            "dec_num_head": (2, 2, 4, 6),
            "enable_flash": False,
            "enc_mode": True,
        },
    }
    variables = {
        "weight": weight,
        "resume": False,
        "evaluate": evaluate,
        "test_only": False,
        "seed": seed,
        "save_path": save_path,
        # Pointcept enables persistent DataLoader workers unconditionally.
        # Keep one worker even for the tiny smoke configuration.
        "num_worker": 4 if not max_samples else 1,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "batch_size_val": batch_size,
        "batch_size_test": batch_size,
        "epoch": epochs,
        "eval_epoch": epochs,
        "clip_grad": 1.0,
        "sync_bn": False,
        "enable_amp": True,
        "amp_dtype": "float16",
        "empty_cache": False,
        "empty_cache_per_epoch": False,
        "find_unused_parameters": False,
        "enable_wandb": False,
        "wandb_project": "bite2text-ptv3",
        "wandb_key": None,
        "mix_prob": 0,
        "selection_metric": "f1",
        "param_dicts": ([{"keyword": "backbone", "lr": 1e-5}] if stage2 else None),
        "hooks": hooks,
        "train": {"type": "DefaultTrainer"},
        "test": {"type": "MultiClsTester", "verbose": True},
        "data": data,
        "model": model,
        "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 0.01},
        "scheduler": {"type": "CosineAnnealingLR", "total_steps": epochs},
    }
    return "\n\n".join(f"{name} = {pprint.pformat(value, width=110, sort_dicts=False)}" for name, value in variables.items()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=85)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--config-prefix",
        default="bite2text_ptv3",
        help="Prefix used for generated config filenames and experiment directories.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Disable validation during fixed-length full-data training.",
    )
    parser.add_argument(
        "--stage2-from-last",
        action="store_true",
        help="Initialize stage 2 from the fixed stage-1 model_last checkpoint.",
    )
    args = parser.parse_args()

    data_root = args.dataset_root.resolve()
    encoder = args.encoder.resolve()
    output_dir = args.output_dir.resolve()
    if not encoder.is_file():
        raise SystemExit(f"Missing encoder checkpoint: {encoder}")
    vocabs: dict[str, list[str]] = json.loads((data_root / "head_vocabs.json").read_text(encoding="utf-8"))
    task_names = list(vocabs)
    if not task_names:
        raise SystemExit("head_vocabs.json must contain at least one task")
    class_weights: list[list[float]] = json.loads((data_root / "class_weights.json").read_text(encoding="utf-8"))
    num_classes = [len(vocabs[name]) for name in task_names]
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_run = f"{args.config_prefix}_stage1_frozen_seed{args.seed}"
    stage2_run = f"{args.config_prefix}_stage2_joint_seed{args.seed}"
    stage1_checkpoint_name = "model_last.pth" if args.stage2_from_last else "model_best.pth"
    configs = {
        f"{args.config_prefix}_smoke.py": config_text(
            data_root=data_root,
            weight=str(encoder),
            save_path=f"exp/dental/{args.config_prefix}_smoke_seed{args.seed}",
            class_weights=class_weights,
            num_classes=num_classes,
            freeze_backbone=True,
            epochs=1,
            batch_size=2,
            max_samples=8,
            stage2=False,
            task_names=task_names,
            seed=args.seed,
            evaluate=not args.no_evaluate,
        ),
        f"{args.config_prefix}_stage1_frozen.py": config_text(
            data_root=data_root,
            weight=str(encoder),
            save_path=f"exp/dental/{stage1_run}",
            class_weights=class_weights,
            num_classes=num_classes,
            freeze_backbone=True,
            epochs=args.stage1_epochs,
            batch_size=8,
            max_samples=0,
            stage2=False,
            task_names=task_names,
            seed=args.seed,
            evaluate=not args.no_evaluate,
        ),
        f"{args.config_prefix}_stage2_joint.py": config_text(
            data_root=data_root,
            weight=f"exp/dental/{stage1_run}/model/{stage1_checkpoint_name}",
            save_path=f"exp/dental/{stage2_run}",
            class_weights=class_weights,
            num_classes=num_classes,
            freeze_backbone=False,
            epochs=args.stage2_epochs,
            batch_size=8,
            max_samples=0,
            stage2=True,
            task_names=task_names,
            seed=args.seed,
            evaluate=not args.no_evaluate,
        ),
    }
    for filename, content in configs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
        print(output_dir / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
