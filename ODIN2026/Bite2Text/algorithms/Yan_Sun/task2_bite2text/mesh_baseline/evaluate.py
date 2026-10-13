#!/usr/bin/env python3
"""Evaluate a saved paired-IOS baseline checkpoint on one manifest split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import IOSMeshDataset, PairedPointNet, class_weights, read_manifest, run_epoch, save_json, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    records = read_manifest(args.manifest)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] == args.split]
    if not eval_records:
        raise SystemExit(f"No records in split {args.split!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    model_config = {
        "geometry_features": bool(checkpoint_config.get("geometry_features", False)),
        "contact_points": int(checkpoint_config.get("contact_points", 256)),
        "direct_geometry_heads": bool(checkpoint_config.get("direct_geometry_heads", False)),
        "direct_geometry_exclude_heads": str(checkpoint_config.get("direct_geometry_exclude_heads", "")),
    }
    sampling_mode = str(checkpoint_config.get("sampling_mode", "vertices"))
    dataset = IOSMeshDataset(
        eval_records,
        args.data_root,
        args.num_points,
        args.seed + 300000,
        augment=False,
        sampling_mode=sampling_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = PairedPointNet(head_vocabs, **model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    weights = class_weights(train_records, head_vocabs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    with torch.no_grad():
        metrics = run_epoch(model, loader, None, scaler, device, head_vocabs, weights)
    result = {"checkpoint": str(args.checkpoint), "model_config": model_config, "sampling_mode": sampling_mode, "split": args.split, "samples": len(eval_records), "device": str(device), "metrics": metrics}
    save_json(args.output_dir / "metrics.json", result)
    print(json.dumps({"split": args.split, "samples": len(eval_records), "loss": metrics["loss"], "mean_macro_f1": metrics["mean_macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
