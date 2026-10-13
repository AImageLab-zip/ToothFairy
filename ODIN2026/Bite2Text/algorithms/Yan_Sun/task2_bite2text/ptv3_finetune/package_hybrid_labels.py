#!/usr/bin/env python3
"""Package report-derived structured labels in retrieval-index order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports_path = args.retrieval_reports.resolve()
    metadata = json.loads(reports_path.read_text(encoding="utf-8"))
    patient_ids = metadata["patient_ids"]
    manifest = {}
    for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            manifest[record["patient_id"]] = record
    missing = [patient_id for patient_id in patient_ids if patient_id not in manifest]
    if missing:
        raise RuntimeError(f"Missing manifest patients: {missing[:10]}")
    target_values = [manifest[patient_id]["target_values"] for patient_id in patient_ids]
    payload = {
        "version": "bite2text-hybrid-labels-v1",
        "patient_ids": patient_ids,
        "target_values": target_values,
        "retrieval_reports_sha256": sha256(reports_path),
    }
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"patients": len(patient_ids), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
