"""Command-line interface for the synthetic, weight-free demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_from_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitized Bitemark public skeleton")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unfitted-placeholder", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.features.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    result = run_from_features(
        payload,
        load_config(args.config),
        allow_unfitted=args.allow_unfitted_placeholder,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
