#!/usr/bin/env python3
"""Create a compact, read-only-derived cache for multi-view CNN training."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def cache_one(job: tuple[dict[str, str], Path, int, int]) -> dict[str, Any]:
    row, output_path, max_side, quality = job
    register_heif()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(row["source_path"]) as image:
            image.draft("RGB", (max_side, max_side))
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(output_path, format="JPEG", quality=quality, optimize=True)
        return {
            **row,
            "cached_path": str(output_path),
            "cache_width": image.width,
            "cache_height": image.height,
            "cache_error": "",
        }
    except Exception as exc:
        return {
            **row,
            "cached_path": "",
            "cache_width": "",
            "cache_height": "",
            "cache_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.selection.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    jobs: list[tuple[dict[str, str], Path, int, int]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["missing"].lower() == "true":
            missing_rows.append(
                {
                    **row,
                    "cached_path": "",
                    "cache_width": "",
                    "cache_height": "",
                    "cache_error": "missing_source_view",
                }
            )
            continue
        output_path = args.output_dir / "images" / row["patient_id"] / f"{row['view']}.jpg"
        jobs.append((row, output_path, args.max_side, args.quality))

    cached_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(executor.map(cache_one, jobs), start=1):
            cached_rows.append(result)
            if completed % 500 == 0 or completed == len(jobs):
                print(
                    json.dumps({"cached": completed, "total": len(jobs)}),
                    flush=True,
                )
    all_rows = sorted(
        cached_rows + missing_rows,
        key=lambda row: (row["patient_id"], int(row["slot_index"])),
    )
    manifest_path = args.output_dir / "cached_selection.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "selection": str(args.selection),
        "rows": len(all_rows),
        "cached_images": sum(bool(row["cached_path"]) for row in all_rows),
        "missing_views": sum(row["cache_error"] == "missing_source_view" for row in all_rows),
        "cache_errors": sum(
            bool(row["cache_error"]) and row["cache_error"] != "missing_source_view"
            for row in all_rows
        ),
        "max_side": args.max_side,
        "quality": args.quality,
    }
    (args.output_dir / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
