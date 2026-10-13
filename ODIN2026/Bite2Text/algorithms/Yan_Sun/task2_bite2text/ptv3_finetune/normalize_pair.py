#!/usr/bin/env python3
"""Minimal paired IOS-Normalizer entrypoint for production inference.

This intentionally bypasses ``orient_scan.py`` because that research CLI
imports the optional ``debugpy`` package unconditionally.  The underlying
normalization API and paired transformation are unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lower", required=True, type=Path)
    parser.add_argument("--upper", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str((args.normalizer_root / "src").resolve()))
    from scannormalizer.scan_inference import (  # pylint: disable=import-outside-toplevel
        load_normalizer,
        normalize_scan,
        transform_scan,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lower_output = args.output_dir / "ios_lower_oriented.stl"
    upper_output = args.output_dir / "ios_upper_oriented.stl"
    normalizer = load_normalizer(
        args.checkpoint,
        args.device,
        points=None,
        seed=args.seed,
    )
    result = normalize_scan(
        args.lower,
        lower_output,
        normalizer,
        orient_only=False,
        center_and_orient=True,
    )
    transform_scan(
        args.upper,
        upper_output,
        result.matrix,
        center=result.center,
        scale=result.scale,
        orient_only=False,
        center_and_orient=True,
    )
    print(
        json.dumps(
            {
                "rotation_index": int(result.rotation_index),
                "lower_output": str(lower_output),
                "upper_output": str(upper_output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
