#!/usr/bin/env python3
"""Probe how much label signal exists in the handcrafted cross-jaw features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    IOSMeshDataset,
    PairedPointNet,
    class_weights,
    masked_classification_loss,
    metrics_from_counts,
    read_manifest,
    save_json,
    set_seed,
)


class GeometryProbe(nn.Module):
    def __init__(self, feature_dim: int, head_vocabs: dict[str, list[str]], model_type: str, hidden_dim: int) -> None:
        super().__init__()
        if model_type == "linear":
            self.stem = nn.Identity()
            head_features = feature_dim
        elif model_type == "mlp":
            self.stem = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.1))
            head_features = hidden_dim
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.heads = nn.ModuleDict({head: nn.Linear(head_features, len(vocab)) for head, vocab in head_vocabs.items()})

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        embedded = self.stem(features)
        return {head: classifier(embedded) for head, classifier in self.heads.items()}


def extract_features(
    records: list[dict[str, Any]],
    data_root: Path,
    head_vocabs: dict[str, list[str]],
    num_points: int,
    contact_points: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    dataset = IOSMeshDataset(records, data_root, num_points, seed + 300000, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    extractor = PairedPointNet(head_vocabs, geometry_features=True, contact_points=contact_points).to(device).eval()
    feature_batches: list[Tensor] = []
    target_batches: dict[str, list[Tensor]] = {head: [] for head in head_vocabs}
    with torch.no_grad():
        for batch in loader:
            upper = batch["upper"].to(device, non_blocking=True)
            lower = batch["lower"].to(device, non_blocking=True)
            feature_batches.append(extractor._cross_jaw_geometry(upper, lower).cpu())
            for head, values in batch["targets"].items():
                target_batches[head].append(values.cpu())
    return torch.cat(feature_batches), {head: torch.cat(values) for head, values in target_batches.items()}


def evaluate(
    model: GeometryProbe,
    features: Tensor,
    targets: dict[str, Tensor],
    head_vocabs: dict[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    matrices = {head: np.zeros((len(vocab), len(vocab)), dtype=np.int64) for head, vocab in head_vocabs.items()}
    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device))
    for head, logits in predictions.items():
        target = targets[head]
        valid = target >= 0
        if valid.any():
            truth = target[valid].numpy()
            guessed = logits[valid.to(device)].argmax(dim=1).cpu().numpy()
            np.add.at(matrices[head], (truth, guessed), 1)
    return metrics_from_counts(matrices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--contact-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_manifest(args.manifest)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    splits = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
    extracted = {
        split: extract_features(rows, args.data_root, head_vocabs, args.num_points, args.contact_points, args.seed, device)
        for split, rows in splits.items()
    }
    train_features, train_targets = extracted["train"]
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    extracted = {
        split: ((features - feature_mean) / feature_std, targets)
        for split, (features, targets) in extracted.items()
    }
    train_features, train_targets = extracted["train"]
    model = GeometryProbe(train_features.shape[1], head_vocabs, args.model, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device_weights = {head: weight.to(device) for head, weight in class_weights(splits["train"], head_vocabs).items()}
    dataset = TensorDataset(train_features, *[train_targets[head] for head in head_vocabs])
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            features = batch[0].to(device)
            targets = {head: batch[index + 1].to(device) for index, head in enumerate(head_vocabs)}
            logits = model(features)
            losses = [
                loss
                for head, prediction in logits.items()
                if (loss := masked_classification_loss(prediction, targets[head], device_weights[head])) is not None
            ]
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            optimizer.step()
        val_metrics = evaluate(model, *extracted["val"], head_vocabs, device)
        score = val_metrics["mean_macro_f1"]
        if score is not None and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif args.patience and epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Probe did not produce a validation score")
    model.load_state_dict(best_state)
    val_metrics = evaluate(model, *extracted["val"], head_vocabs, device)
    test_metrics = evaluate(model, *extracted["test"], head_vocabs, device)
    result = {
        "model": args.model,
        "feature_dim": int(train_features.shape[1]),
        "feature_normalization": "train_split_zscore",
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "best_val_mean_macro_f1": best_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    save_json(args.output_dir / "metrics.json", result)
    print(json.dumps({"model": args.model, "feature_dim": result["feature_dim"], "best_epoch": best_epoch, "val_macro_f1": val_metrics["mean_macro_f1"], "test_macro_f1": test_metrics["mean_macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
