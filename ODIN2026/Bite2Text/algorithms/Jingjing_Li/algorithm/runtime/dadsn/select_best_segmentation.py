#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select the better segmentation result between the original run and retry1.

Priority:
1. Geometry available
2. No retry recommendation
3. Fewer hard failures
4. Fewer critical reasons
5. Higher selection score
6. Higher mean core confidence
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def number(row: pd.Series, key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def rank(row: pd.Series) -> tuple[float, ...]:
    return (
        number(row, "geometry_available_v2", 0.0),
        1.0 - number(row, "retry_recommended_v2", 1.0),
        -number(row, "hard_failure_count", 999.0),
        -number(row, "critical_reason_count", 999.0),
        number(row, "selection_score_v2", -999.0),
        number(row, "mean_core_confidence", -999.0),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-qc",
        type=Path,
        default=Path(
            "work/segmentation_qc_v2/segmentation_qc_v2_summary.csv"
        ),
    )
    parser.add_argument(
        "--retry-qc",
        type=Path,
        default=Path(
            "work/segmentation_qc_retry1_v2/segmentation_qc_v2_summary.csv"
        ),
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=Path("work/output2"),
    )
    parser.add_argument(
        "--retry-output",
        type=Path,
        default=Path("work/output2_retry1"),
    )
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=Path("work/output2_selected"),
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--clean", action="store_true")
    return parser


def place_file(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()

    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def main() -> int:
    args = build_parser().parse_args()

    original_qc = pd.read_csv(
        args.original_qc.resolve(),
        dtype={"case": str},
    )
    retry_qc = pd.read_csv(
        args.retry_qc.resolve(),
        dtype={"case": str},
    )

    original_rows = {
        str(row["case"]): row
        for _, row in original_qc.iterrows()
    }
    retry_rows = {
        str(row["case"]): row
        for _, row in retry_qc.iterrows()
    }

    original_output = args.original_output.resolve()
    retry_output = args.retry_output.resolve()
    selected_output = args.selected_output.resolve()

    if args.clean and selected_output.exists():
        shutil.rmtree(selected_output)

    selected_output.mkdir(parents=True, exist_ok=True)
    manifest = []

    for case in sorted(original_rows):
        original_row = original_rows[case]
        retry_row = retry_rows.get(case)

        if retry_row is None:
            source_name = "original"
            source_dir = original_output
            selected_row = original_row
            reason = "no_retry_candidate"
        elif rank(retry_row) > rank(original_row):
            source_name = "retry1"
            source_dir = retry_output
            selected_row = retry_row
            reason = "retry_rank_better"
        else:
            source_name = "original"
            source_dir = original_output
            selected_row = original_row
            reason = "original_rank_equal_or_better"

        case_files = sorted(
            path
            for path in source_dir.glob(f"{case}*")
            if path.is_file() or path.is_symlink()
        )

        for source_file in case_files:
            place_file(
                source_file,
                selected_output / source_file.name,
                args.mode,
            )

        manifest.append(
            {
                "case": case,
                "selected_source": source_name,
                "selection_reason": reason,
                "selected_geometry_available": int(
                    number(selected_row, "geometry_available_v2", 0.0)
                ),
                "selected_retry_recommended": int(
                    number(selected_row, "retry_recommended_v2", 1.0)
                ),
                "selected_score": number(
                    selected_row,
                    "selection_score_v2",
                    float("nan"),
                ),
                "selected_mean_core_confidence": number(
                    selected_row,
                    "mean_core_confidence",
                    float("nan"),
                ),
                "selected_file_count": len(case_files),
            }
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_path = selected_output / "selection_manifest.csv"
    manifest_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 72)
    print("Best segmentation selection")
    print("=" * 72)
    print(f"Cases: {len(manifest_df)}")
    print(
        "Selected: "
        f"{manifest_df['selected_source'].value_counts().to_dict()}"
    )
    print(f"Output: {selected_output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
