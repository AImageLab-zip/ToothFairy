#!/usr/bin/env python3
"""Extract a portable PTv3 backbone checkpoint from a Bits2Bites run."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--data-count", type=int, default=200)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    if not backbone:
        raise RuntimeError("No backbone.* tensors found in checkpoint")

    payload = {
        "state_dict": backbone,
        "meta": {
            "architecture": "PT-v3m1",
            "input_channels": 9,
            "modality": "Bits2Bites mesh-only",
            "training_cases": args.data_count,
            "source_epoch": checkpoint.get("epoch"),
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "upstream_commit": args.upstream_commit,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(f"encoder_tensors={len(backbone)}")
    print(f"output={args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
