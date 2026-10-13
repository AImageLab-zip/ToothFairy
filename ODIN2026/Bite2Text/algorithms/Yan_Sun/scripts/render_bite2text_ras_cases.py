#!/usr/bin/env python3
"""Render selected Bite2Text upper/lower STL pairs from the dataset ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from audit_bite2text_ras import build_archive_index, render_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patients", required=True, help="Comma-separated patient IDs")
    parser.add_argument("--max-triangles", type=int, default=3000)
    parser.add_argument("--cases-per-page", type=int, default=6)
    args = parser.parse_args()

    selected = [item.strip() for item in args.patients.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path) as archive:
        patients, _ = build_archive_index(archive)
        missing = [patient for patient in selected if patient not in patients]
        if missing:
            raise SystemExit(f"Patients not found: {missing}")
        manifest = render_cases(
            archive,
            patients,
            selected,
            args.out_dir,
            args.max_triangles,
            args.cases_per_page,
        )
    (args.out_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
