#!/usr/bin/env python3
"""Package a retrieval index and its aligned official-reference reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    index_path = args.index.resolve()
    with np.load(index_path, allow_pickle=False) as payload:
        patient_ids = [str(value) for value in payload["patient_ids"]]
        descriptor_shape = list(payload["descriptors"].shape)
    records = {
        record["patient_id"]: record
        for record in (
            json.loads(line)
            for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        )
    }
    reports = []
    report_paths = []
    for patient_id in patient_ids:
        record = records[patient_id]
        relative_path = record.get("target_report_path")
        if not relative_path:
            raise RuntimeError(f"Index patient {patient_id} has no official report")
        report_path = args.raw_data_root.resolve() / relative_path
        reports.append(report_path.read_text(encoding="utf-8", errors="replace").strip())
        report_paths.append(relative_path)

    packaged_index = output_dir / "retrieval_index.npz"
    shutil.copy2(index_path, packaged_index)
    corpus = {
        "version": "bite2text-geometry-retrieval-v1",
        "patient_ids": patient_ids,
        "report_paths": report_paths,
        "reports": reports,
        "descriptor_shape": descriptor_shape,
        "index_sha256": sha256_file(packaged_index),
    }
    (output_dir / "retrieval_reports.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "cases": len(patient_ids),
        "descriptor_shape": descriptor_shape,
        "index_bytes": packaged_index.stat().st_size,
        "reports_bytes": (output_dir / "retrieval_reports.json").stat().st_size,
        "index_sha256": corpus["index_sha256"],
    }
    (output_dir / "retrieval_assets_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
