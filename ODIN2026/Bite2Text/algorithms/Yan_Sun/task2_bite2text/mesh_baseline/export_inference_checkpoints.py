#!/usr/bin/env python3
"""Export self-contained inference checkpoints without optimizer state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {missing}")

    args.output_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    for index, source in enumerate(args.checkpoints, start=1):
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        required = {"model", "head_vocabs", "config"}
        absent = sorted(required.difference(checkpoint))
        if absent:
            raise RuntimeError(f"{source} lacks checkpoint fields: {absent}")
        target = args.output_dir / f"model_{index:02d}.pt"
        payload = {
            "export_format": "bite2text-inference-checkpoint-v1",
            "source_checkpoint": str(source),
            "epoch": checkpoint.get("epoch"),
            "model": checkpoint["model"],
            "head_vocabs": checkpoint["head_vocabs"],
            "config": checkpoint["config"],
        }
        torch.save(payload, target)
        manifest.append(
            {
                "source": str(source),
                "target": target.name,
                "bytes": target.stat().st_size,
                "sha256": file_sha256(target),
                "epoch": checkpoint.get("epoch"),
            }
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "models": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
