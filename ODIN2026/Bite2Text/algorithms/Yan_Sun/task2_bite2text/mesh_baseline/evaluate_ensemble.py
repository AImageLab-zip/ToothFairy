#!/usr/bin/env python3
"""Evaluate a uniform probability ensemble of paired-IOS checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from train import IGNORE_INDEX, IOSMeshDataset, PairedPointNet, class_weights, metrics_from_counts, read_manifest, save_json, set_seed


def load_models(checkpoint_paths: list[Path], head_vocabs: dict[str, list[str]], device: torch.device) -> tuple[list[PairedPointNet], dict[str, int | bool]]:
    models: list[PairedPointNet] = []
    expected_model_config: dict[str, int | bool] | None = None
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_vocabs = checkpoint.get("head_vocabs")
        if checkpoint_vocabs != head_vocabs:
            raise ValueError(f"Head vocabularies do not match in checkpoint: {checkpoint_path}")
        checkpoint_config = checkpoint.get("config", {})
        model_config: dict[str, int | bool] = {
            "geometry_features": bool(checkpoint_config.get("geometry_features", False)),
            "contact_points": int(checkpoint_config.get("contact_points", 256)),
            "direct_geometry_heads": bool(checkpoint_config.get("direct_geometry_heads", False)),
            "direct_geometry_exclude_heads": str(checkpoint_config.get("direct_geometry_exclude_heads", "")),
        }
        if expected_model_config is None:
            expected_model_config = model_config
        elif model_config != expected_model_config:
            raise ValueError(f"Checkpoint model configurations do not match: {checkpoint_path}")
        model = PairedPointNet(head_vocabs, **model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    if expected_model_config is None:
        raise ValueError("No checkpoints supplied")
    return models, expected_model_config


def run_ensemble_epoch(
    models: list[nn.Module],
    loader: DataLoader[dict[str, object]],
    device: torch.device,
    head_vocabs: dict[str, list[str]],
    weights: dict[str, Tensor],
    tta_samples: int = 1,
) -> dict[str, object]:
    """Compute metrics after averaging model and deterministic sampling views."""
    if tta_samples < 1:
        raise ValueError("tta_samples must be at least one")
    matrices = {head: np.zeros((len(vocab), len(vocab)), dtype=np.int64) for head, vocab in head_vocabs.items()}
    device_weights = {head: weight.to(device) for head, weight in weights.items()}
    dataset = loader.dataset
    if not isinstance(dataset, IOSMeshDataset):
        raise TypeError("TTA evaluation requires IOSMeshDataset")
    probability_sums: dict[str, list[Tensor]] | None = None
    targets_by_batch: dict[str, list[Tensor]] | None = None
    for view in range(tta_samples):
        dataset.set_epoch(view)
        batch_probabilities: dict[str, list[Tensor]] = {head: [] for head in head_vocabs}
        batch_targets: dict[str, list[Tensor]] = {head: [] for head in head_vocabs}
        for batch in loader:
            upper = batch["upper"].to(device, non_blocking=True)  # type: ignore[union-attr]
            lower = batch["lower"].to(device, non_blocking=True)  # type: ignore[union-attr]
            targets = {head: values.to(device, non_blocking=True) for head, values in batch["targets"].items()}  # type: ignore[union-attr]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits_by_model = [model(upper, lower) for model in models]
            probabilities = {
                head: torch.stack([logits[head].float().softmax(dim=1) for logits in logits_by_model], dim=0).mean(dim=0)
                for head in head_vocabs
            }
            for head in head_vocabs:
                batch_probabilities[head].append(probabilities[head].cpu())
                if view == 0:
                    batch_targets[head].append(targets[head].cpu())
        if probability_sums is None:
            probability_sums = batch_probabilities
            targets_by_batch = batch_targets
        else:
            for head in head_vocabs:
                probability_sums[head] = [old + new for old, new in zip(probability_sums[head], batch_probabilities[head])]
    if probability_sums is None or targets_by_batch is None:
        raise RuntimeError("No evaluation batches")

    total_loss = 0.0
    steps = 0
    for batch_index in range(len(next(iter(probability_sums.values())))):
        probabilities = {
            head: probability_sums[head][batch_index].to(device) / tta_samples
            for head in head_vocabs
        }
        targets = {head: targets_by_batch[head][batch_index].to(device) for head in head_vocabs}
        losses = []
        for head, prediction in probabilities.items():
            if (targets[head] != IGNORE_INDEX).any():
                losses.append(
                    F.nll_loss(
                        prediction.clamp_min(torch.finfo(prediction.dtype).tiny).log(),
                        targets[head],
                        weight=device_weights[head],
                        ignore_index=IGNORE_INDEX,
                    )
                )
        if not losses:
            continue
        total_loss += float(torch.stack(losses).mean().cpu())
        steps += 1
        for head, prediction in probabilities.items():
            target = targets[head]
            valid = target != IGNORE_INDEX
            if valid.any():
                truth = target[valid].cpu().numpy()
                guessed = prediction[valid].argmax(dim=1).cpu().numpy()
                np.add.at(matrices[head], (truth, guessed), 1)
    result = metrics_from_counts(matrices)
    result["loss"] = total_loss / max(steps, 1)
    result["steps"] = steps
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--tta-samples", type=int, default=1, help="Average this many deterministic point-sampling views per case.")
    args = parser.parse_args()

    if not args.checkpoints:
        raise SystemExit("At least one checkpoint is required")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    missing = [path for path in args.checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"Checkpoint files do not exist: {missing}")

    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    records = read_manifest(args.manifest)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] == args.split]
    if not eval_records:
        raise SystemExit(f"No records in split {args.split!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = IOSMeshDataset(eval_records, args.data_root, args.num_points, args.seed + 300000, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    models, model_config = load_models(args.checkpoints, head_vocabs, device)
    with torch.no_grad():
        metrics = run_ensemble_epoch(models, loader, device, head_vocabs, class_weights(train_records, head_vocabs), args.tta_samples)
    result = {
        "checkpoints": [str(path) for path in args.checkpoints],
        "ensemble": "uniform_probability_average",
        "model_config": model_config,
        "split": args.split,
        "samples": len(eval_records),
        "seed": args.seed,
        "tta_samples": args.tta_samples,
        "device": str(device),
        "metrics": metrics,
    }
    save_json(args.output_dir / "metrics.json", result)
    print(
        json.dumps(
            {
                "split": args.split,
                "samples": len(eval_records),
                "models": len(models),
                "loss": metrics["loss"],
                "mean_macro_f1": metrics["mean_macro_f1"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
