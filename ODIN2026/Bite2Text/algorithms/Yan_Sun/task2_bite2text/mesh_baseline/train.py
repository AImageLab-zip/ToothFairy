#!/usr/bin/env python3
"""Train a compact paired-IOS, PointNet-style multi-task baseline.

This is intentionally a lightweight Bits2Bites-style baseline: two sampled
STL point clouds (upper/lower), a shared point encoder, pair fusion, and
masked classification heads.  It has no tooth-landmark input or photo branch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100
GEOMETRY_FEATURE_DIM = 34


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_stl_triangles(path: Path) -> np.ndarray:
    """Read binary or ASCII STL triangles without an external mesh dependency."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangles = struct.unpack_from("<I", payload, 80)[0]
        expected_size = 84 + 50 * triangles
        if expected_size == len(payload):
            dtype = np.dtype([("normal", "<f4", (3,)), ("vectors", "<f4", (3, 3)), ("attribute", "<u2")])
            mesh = np.frombuffer(payload, dtype=dtype, offset=84, count=triangles)
            return np.asarray(mesh["vectors"], dtype=np.float32).copy()
    text = payload.decode("utf-8", errors="ignore")
    values = re.findall(r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text)
    if not values or len(values) % 3:
        raise ValueError(f"Unsupported or empty STL: {path}")
    return np.asarray(values, dtype=np.float32).reshape(-1, 3, 3)


def load_stl_vertices(path: Path) -> np.ndarray:
    """Return the STL triangle vertices as a flat array for legacy sampling."""
    return load_stl_triangles(path).reshape(-1, 3)


def sample_points(vertices: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(vertices) == 0:
        raise ValueError("Cannot sample an empty mesh")
    indices = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
    return vertices[indices].astype(np.float32, copy=False)


def sample_surface_points(triangles: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points uniformly by triangle area using barycentric coordinates."""
    if len(triangles) == 0:
        raise ValueError("Cannot sample an empty mesh")
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    valid = areas > np.finfo(np.float32).eps
    if not valid.any():
        return sample_points(triangles.reshape(-1, 3), num_points, rng)
    valid_triangles = triangles[valid]
    probabilities = areas[valid] / areas[valid].sum()
    chosen = valid_triangles[rng.choice(len(valid_triangles), size=num_points, p=probabilities)]
    first = rng.random(num_points, dtype=np.float32)
    second = rng.random(num_points, dtype=np.float32)
    reflected = first + second > 1.0
    first[reflected] = 1.0 - first[reflected]
    second[reflected] = 1.0 - second[reflected]
    return (chosen[:, 0] + first[:, None] * (chosen[:, 1] - chosen[:, 0]) + second[:, None] * (chosen[:, 2] - chosen[:, 0])).astype(np.float32)


def sample_mesh_points(triangles: np.ndarray, num_points: int, rng: np.random.Generator, sampling_mode: str) -> np.ndarray:
    if sampling_mode == "vertices":
        return sample_points(triangles.reshape(-1, 3), num_points, rng)
    if sampling_mode == "surface":
        return sample_surface_points(triangles, num_points, rng)
    raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def normalize_pair(upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([upper, lower], axis=0)
    center = combined.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(combined - center, axis=1).max()
    scale = max(float(scale), 1e-6)
    return (upper - center) / scale, (lower - center) / scale


class IOSMeshDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], data_root: Path, num_points: int, seed: int, augment: bool, sampling_mode: str = "vertices") -> None:
        self.records = records
        self.data_root = data_root
        self.num_points = num_points
        self.seed = seed
        self.augment = augment
        self.sampling_mode = sampling_mode
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic sampling stream for the next DataLoader pass."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # A new epoch gets a new, reproducible surface-vertex sample and
        # augmentation.  The old v1 implementation reused exactly the same
        # sampled points each epoch, which weakened augmentation substantially.
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        upper = sample_mesh_points(load_stl_triangles(self.data_root / record["upper_stl"]), self.num_points, rng, self.sampling_mode)
        lower = sample_mesh_points(load_stl_triangles(self.data_root / record["lower_stl"]), self.num_points, rng, self.sampling_mode)
        upper, lower = normalize_pair(upper, lower)
        if self.augment:
            theta = rng.uniform(-math.pi, math.pi)
            rotation = np.asarray(
                [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            upper = upper @ rotation.T
            lower = lower @ rotation.T
            jitter = rng.normal(0.0, 0.005, size=upper.shape).astype(np.float32)
            upper = upper + jitter
            lower = lower + rng.normal(0.0, 0.005, size=lower.shape).astype(np.float32)
        return {
            "upper": torch.from_numpy(upper),
            "lower": torch.from_numpy(lower),
            "targets": {head: torch.tensor(value, dtype=torch.long) for head, value in record["targets"].items()},
            "patient_id": record["patient_id"],
        }


class PointEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )

    def forward(self, points: Tensor) -> Tensor:
        features = self.layers(points.transpose(1, 2))
        return torch.cat([features.amax(dim=2), features.mean(dim=2)], dim=1)


class PairedPointNet(nn.Module):
    def __init__(
        self,
        head_vocabs: dict[str, list[str]],
        geometry_features: bool = False,
        contact_points: int = 256,
        direct_geometry_heads: bool = False,
        direct_geometry_exclude_heads: str = "",
    ) -> None:
        super().__init__()
        if direct_geometry_heads and not geometry_features:
            raise ValueError("direct_geometry_heads requires geometry_features")
        self.geometry_features = geometry_features
        self.contact_points = contact_points
        self.direct_geometry_heads = direct_geometry_heads
        excluded = {name.strip() for name in direct_geometry_exclude_heads.split(",") if name.strip()}
        unknown_excluded = excluded.difference(head_vocabs)
        if unknown_excluded:
            raise ValueError(f"Unknown direct-geometry excluded heads: {sorted(unknown_excluded)}")
        self.direct_geometry_exclude_heads = excluded
        self.encoder = PointEncoder()
        fusion_features = 2048 + (GEOMETRY_FEATURE_DIM if geometry_features else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_features, 512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.15),
        )
        self.heads = nn.ModuleDict({head: nn.Linear(256, len(vocab)) for head, vocab in head_vocabs.items()})
        if direct_geometry_heads:
            # The probe uses train-split z-scored geometry values.  This
            # running-stat normalization approximates that behavior while
            # accommodating the training-time point resampling stream.
            self.geometry_normalizer = nn.BatchNorm1d(GEOMETRY_FEATURE_DIM, affine=False)
            self.geometry_heads = nn.ModuleDict(
                {head: nn.Linear(GEOMETRY_FEATURE_DIM, len(vocab)) for head, vocab in head_vocabs.items() if head not in excluded}
            )

    @staticmethod
    def _summary(values: Tensor, quantiles: bool = False) -> Tensor:
        """Return per-sample rotation-stable scalar statistics for [B, N]."""
        output = [
            values.mean(dim=1, keepdim=True),
            values.std(dim=1, unbiased=False, keepdim=True),
            values.amin(dim=1, keepdim=True),
        ]
        if quantiles:
            count = values.shape[1]
            for fraction in (0.10, 0.50, 0.90):
                kth = max(1, min(count, int(round((count - 1) * fraction)) + 1))
                output.append(values.kthvalue(kth, dim=1).values.unsqueeze(1))
            output.append(values.amax(dim=1, keepdim=True))
        return torch.cat(output, dim=1)

    def _cross_jaw_geometry(self, upper: Tensor, lower: Tensor) -> Tensor:
        """Summarize bidirectional nearest-opposing-jaw geometry.

        Inputs are already jointly normalized and only receive yaw augmentation.
        The feature design intentionally uses distances, horizontal magnitudes,
        and z offsets rather than x/y directions so it remains stable under
        that augmentation while retaining inter-arch separation information.
        """
        count = min(self.contact_points, upper.shape[1], lower.shape[1])
        indices_upper = torch.linspace(0, upper.shape[1] - 1, count, device=upper.device).round().long()
        indices_lower = torch.linspace(0, lower.shape[1] - 1, count, device=lower.device).round().long()
        upper_sample = upper.index_select(1, indices_upper).float()
        lower_sample = lower.index_select(1, indices_lower).float()
        pair_distances = torch.cdist(upper_sample, lower_sample)

        upper_distance, upper_index = pair_distances.min(dim=2)
        lower_distance, lower_index = pair_distances.min(dim=1)
        upper_match = lower_sample.gather(1, upper_index.unsqueeze(-1).expand(-1, -1, 3))
        lower_match = upper_sample.gather(1, lower_index.unsqueeze(-1).expand(-1, -1, 3))
        upper_delta = upper_match - upper_sample
        lower_delta = lower_match - lower_sample

        def directional_features(distance: Tensor, delta: Tensor) -> Tensor:
            horizontal = delta[..., :2].norm(dim=2)
            vertical = delta[..., 2]
            return torch.cat(
                [self._summary(distance, quantiles=True), self._summary(horizontal), self._summary(vertical)],
                dim=1,
            )

        upper_height = self._summary(upper_sample[..., 2])
        lower_height = self._summary(lower_sample[..., 2])
        centroid_delta = upper_sample.mean(dim=1) - lower_sample.mean(dim=1)
        centroid_features = torch.cat([centroid_delta[..., :2].norm(dim=1, keepdim=True), centroid_delta[..., 2:3]], dim=1)
        return torch.cat(
            [directional_features(upper_distance, upper_delta), directional_features(lower_distance, lower_delta), upper_height, lower_height, centroid_features],
            dim=1,
        )

    def forward(self, upper: Tensor, lower: Tensor) -> dict[str, Tensor]:
        upper_features = self.encoder(upper)
        lower_features = self.encoder(lower)
        pair = torch.cat([upper_features, lower_features, torch.abs(upper_features - lower_features), upper_features * lower_features], dim=1)
        geometry: Tensor | None = None
        if self.geometry_features:
            geometry = self._cross_jaw_geometry(upper, lower)
            pair = torch.cat([pair, geometry], dim=1)
        features = self.fusion(pair)
        output = {head: classifier(features) for head, classifier in self.heads.items()}
        if self.direct_geometry_heads:
            if geometry is None:
                raise RuntimeError("Missing geometry for direct geometry heads")
            normalized_geometry = self.geometry_normalizer(geometry)
            output = {
                head: output[head] + self.geometry_heads[head](normalized_geometry) if head in self.geometry_heads else output[head]
                for head in output
            }
        return output


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def class_weights(records: list[dict[str, Any]], head_vocabs: dict[str, list[str]]) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for head, vocab in head_vocabs.items():
        counts = np.zeros(len(vocab), dtype=np.float32)
        for record in records:
            target = int(record["targets"][head])
            if target >= 0:
                counts[target] += 1
        # A tiny smoke subset can contain a validation-only class.  Keep its
        # loss weight at one instead of zero, otherwise PyTorch's weighted CE
        # can divide by a zero total weight and return NaN.
        weights = np.ones_like(counts)
        observed = counts > 0
        weights[observed] = 1.0 / np.sqrt(counts[observed])
        if observed.any():
            weights[observed] /= weights[observed].mean()
        output[head] = torch.tensor(weights, dtype=torch.float32)
    return output


def masked_classification_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    loss_name: str = "weighted_ce",
    focal_gamma: float = 2.0,
) -> Tensor | None:
    """Return weighted CE or focal loss while preserving masked-target semantics.

    The denominator is the sum of the selected class weights, matching
    ``torch.nn.functional.cross_entropy(..., reduction='mean')`` for the
    weighted-CE case.  Setting ``focal_gamma=0`` therefore recovers the
    existing loss exactly (up to floating-point roundoff).
    """
    valid = target != IGNORE_INDEX
    if not valid.any():
        return None
    selected_target = target[valid]
    log_probabilities = F.log_softmax(prediction[valid], dim=1)
    log_probability = log_probabilities.gather(1, selected_target.unsqueeze(1)).squeeze(1)
    sample_weights = weight.to(prediction.device)[selected_target]
    loss = -sample_weights * log_probability
    if loss_name == "focal":
        probability = log_probability.exp()
        loss = loss * (1.0 - probability).pow(focal_gamma)
    elif loss_name != "weighted_ce":
        raise ValueError(f"Unsupported loss: {loss_name}")
    return loss.sum() / sample_weights.sum().clamp_min(torch.finfo(loss.dtype).eps)


def metrics_from_counts(counts: dict[str, np.ndarray]) -> dict[str, Any]:
    per_head: dict[str, Any] = {}
    macro_scores: list[float] = []
    for head, matrix in counts.items():
        support = matrix.sum(axis=1)
        total = int(support.sum())
        accuracy = float(matrix.trace() / total) if total else None
        f1_values = []
        for cls, cls_support in enumerate(support):
            if cls_support == 0:
                continue
            tp = float(matrix[cls, cls])
            fp = float(matrix[:, cls].sum() - tp)
            fn = float(matrix[cls, :].sum() - tp)
            denom = 2 * tp + fp + fn
            f1_values.append((2 * tp / denom) if denom else 0.0)
        macro_f1 = float(np.mean(f1_values)) if f1_values else None
        if macro_f1 is not None:
            macro_scores.append(macro_f1)
        per_head[head] = {"samples": total, "accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix": matrix.tolist()}
    return {"per_head": per_head, "mean_macro_f1": float(np.mean(macro_scores)) if macro_scores else None}


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    head_vocabs: dict[str, list[str]],
    weights: dict[str, Tensor],
    loss_name: str = "weighted_ce",
    focal_gamma: float = 2.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    matrices = {head: np.zeros((len(vocab), len(vocab)), dtype=np.int64) for head, vocab in head_vocabs.items()}
    total_loss = 0.0
    steps = 0
    for batch in loader:
        upper = batch["upper"].to(device, non_blocking=True)
        lower = batch["lower"].to(device, non_blocking=True)
        targets = {head: values.to(device, non_blocking=True) for head, values in batch["targets"].items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(upper, lower)
            losses = []
            for head, prediction in logits.items():
                loss = masked_classification_loss(prediction, targets[head], weights[head], loss_name, focal_gamma)
                if loss is not None:
                    losses.append(loss)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach().cpu())
        steps += 1
        for head, prediction in logits.items():
            target = targets[head]
            valid = target != IGNORE_INDEX
            if valid.any():
                truth = target[valid].detach().cpu().numpy()
                guessed = prediction[valid].argmax(dim=1).detach().cpu().numpy()
                np.add.at(matrices[head], (truth, guessed), 1)
    result = metrics_from_counts(matrices)
    result["loss"] = total_loss / max(steps, 1)
    result["steps"] = steps
    return result


def save_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--sampling-mode", choices=("vertices", "surface"), default="vertices", help="Sample raw STL vertices or uniform points over triangle area.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("weighted_ce", "focal"), default="weighted_ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal exponent when --loss focal; gamma=0 recovers weighted CE.")
    parser.add_argument("--geometry-features", action="store_true", help="Fuse cross-jaw nearest-neighbor and arch-position statistics after PointNet pooling.")
    parser.add_argument("--contact-points", type=int, default=256, help="Points per jaw used by the geometry branch when --geometry-features is enabled.")
    parser.add_argument("--direct-geometry-heads", action="store_true", help="Add normalized linear geometry logits directly to each prediction head.")
    parser.add_argument("--direct-geometry-exclude-heads", type=str, default="", help="Comma-separated heads that keep PointNet logits only when direct geometry heads are enabled.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20, help="Stop after this many epochs without validation improvement; 0 disables it.")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-samples", type=int, default=0, help="Per split cap for a quick smoke test; 0 uses all samples.")
    parser.add_argument("--test-after-train", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.focal_gamma < 0:
        raise SystemExit("--focal-gamma must be non-negative")
    if args.contact_points < 2:
        raise SystemExit("--contact-points must be at least 2")
    if args.direct_geometry_heads and not args.geometry_features:
        raise SystemExit("--direct-geometry-heads requires --geometry-features")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_manifest(args.manifest)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    splits = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
    if args.max_samples:
        splits = {split: rows[: args.max_samples] for split, rows in splits.items()}
    if not splits["train"] or not splits["val"]:
        raise SystemExit("Manifest must contain non-empty train and val splits")

    datasets = {
        "train": IOSMeshDataset(splits["train"], args.data_root, args.num_points, args.seed, augment=True, sampling_mode=args.sampling_mode),
        "val": IOSMeshDataset(splits["val"], args.data_root, args.num_points, args.seed + 100000, augment=False, sampling_mode=args.sampling_mode),
        "test": IOSMeshDataset(splits["test"], args.data_root, args.num_points, args.seed + 200000, augment=False, sampling_mode=args.sampling_mode),
    }
    # Keep worker lifetimes scoped to each pass.  Three persistent pools
    # (train/val/test) are unnecessary here and can make end-of-run cleanup
    # brittle on a shared GPU server.
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }
    weights = class_weights(splits["train"], head_vocabs)
    model = PairedPointNet(
        head_vocabs,
        args.geometry_features,
        args.contact_points,
        args.direct_geometry_heads,
        args.direct_geometry_exclude_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({"head_vocabs": head_vocabs, "device": str(device), "split_sizes": {name: len(rows) for name, rows in splits.items()}})
    save_json(args.output_dir / "config.json", config)

    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            epoch_lr = args.lr * epoch / max(args.warmup_epochs, 1)
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            epoch_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(model, loaders["train"], optimizer, scaler, device, head_vocabs, weights, args.loss, args.focal_gamma)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], None, scaler, device, head_vocabs, weights, args.loss, args.focal_gamma)
        score = val_metrics["mean_macro_f1"] if val_metrics["mean_macro_f1"] is not None else -float("inf")
        epoch_result = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(epoch_result)
        print(json.dumps({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], "val_mean_macro_f1": score}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "head_vocabs": head_vocabs, "config": config, "val": val_metrics}, args.output_dir / "best.pt")
        elif args.patience and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch, "patience": args.patience}), flush=True)
            break
    save_json(args.output_dir / "history.json", history)

    summary: dict[str, Any] = {"best_val_mean_macro_f1": best_score, "best_epoch": best_epoch, "epochs_requested": args.epochs, "epochs_completed": len(history), "device": str(device)}
    if args.test_after_train and splits["test"]:
        checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            summary["test"] = run_epoch(model, loaders["test"], None, scaler, device, head_vocabs, weights, args.loss, args.focal_gamma)
    save_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
