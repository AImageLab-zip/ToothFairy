#!/usr/bin/env python3
"""Evaluate a v2 checkpoint using surface sampling and contact features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import SurfacePairDataset, read_manifest, set_seed
from model import ContactPointTransformer
from train import class_weights, run_epoch, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--tokens-per-jaw", type=int, default=128)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SurfacePairDataset(eval_records, args.data_root, args.num_points, args.seed + 200000, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ContactPointTransformer(head_vocabs, tokens_per_jaw=args.tokens_per_jaw).to(device)
    model.load_state_dict(checkpoint["model"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    with torch.no_grad():
        metrics = run_epoch(model, loader, None, scaler, device, head_vocabs, class_weights(train_records, head_vocabs))
    result = {"checkpoint": str(args.checkpoint), "split": args.split, "samples": len(eval_records), "device": str(device), "metrics": metrics}
    save_json(args.output_dir / "metrics.json", result)
    print(json.dumps({"split": args.split, "samples": len(eval_records), "loss": metrics["loss"], "mean_macro_f1": metrics["mean_macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
