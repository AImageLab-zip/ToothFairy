#!/usr/bin/env python3
"""Select and extract a stratified Bite2Text IOS-Normalizer pilot."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path", type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=40)
    return parser.parse_args()


def numeric_patient(row):
    text = row["patient"]
    return int(text[1:]) if text[:1].isalpha() and text[1:].isdigit() else text


def category(row):
    prefix = "xy" if row["axis_layout"] == "XY_arch_Z_vertical" else "xz"
    angle = abs(float(row["front_angle_degrees"]))
    if angle <= 30.0:
        direction = "positive"
    elif angle >= 150.0:
        direction = "negative"
    else:
        direction = "oblique"
    return f"{prefix}_{direction}"


def choose_rows(rows, count):
    rows = sorted(rows, key=numeric_patient)
    selected = {}
    reasons = defaultdict(set)

    def add(row, reason):
        if len(selected) >= count and row["patient"] not in selected:
            return False
        selected[row["patient"]] = row
        reasons[row["patient"]].add(reason)
        return True

    # Keep all of the highest-value regression cases first.
    for row in rows:
        if float(row["upper_minus_lower_vertical"]) <= 0:
            add(row, "upper_not_superior")
        if float(row["front_upper_lower_agreement"]) < 0.50:
            add(row, "upper_lower_front_disagreement")
        if "upper_lower_x_center_gap" in row["flags"]:
            add(row, "x_center_gap_outlier")

    grouped = defaultdict(list)
    for row in rows:
        grouped[category(row)].append(row)

    # Rare and oblique categories are filled before common orientations.
    targets = [
        ("xz_negative", 2),
        ("xy_oblique", 5),
        ("xz_oblique", 5),
        ("xy_positive", 5),
        ("xy_negative", 5),
        ("xz_positive", 5),
    ]
    for name, target in targets:
        current = sum(category(row) == name for row in selected.values())
        for row in grouped[name]:
            if current >= target or len(selected) >= count:
                break
            if row["patient"] not in selected:
                add(row, f"stratum:{name}")
                current += 1
            else:
                reasons[row["patient"]].add(f"stratum:{name}")

    # Deterministic round-robin fill if anomaly overlap leaves free slots.
    group_names = [name for name, _ in targets]
    positions = {name: 0 for name in group_names}
    while len(selected) < count:
        progress = False
        for name in group_names:
            candidates = grouped[name]
            while positions[name] < len(candidates):
                row = candidates[positions[name]]
                positions[name] += 1
                if row["patient"] in selected:
                    continue
                add(row, f"fill:{name}")
                progress = True
                break
            if len(selected) >= count:
                break
        if not progress:
            break

    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of requested {count} cases")
    return sorted(selected.values(), key=numeric_patient), reasons


def build_ios_index(archive):
    index = defaultdict(dict)
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".stl"):
            continue
        parts = info.filename.split("/", 1)
        if len(parts) != 2 or "/ios/" not in info.filename.lower():
            continue
        name = Path(info.filename).name.lower()
        if "upper" in name:
            index[parts[0]]["upper"] = info
        elif "lower" in name:
            index[parts[0]]["lower"] = info
    return index


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.audit_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected, reasons = choose_rows(rows, args.count)

    manifest = []
    with zipfile.ZipFile(args.zip_path) as archive:
        ios_index = build_ios_index(archive)
        for row in selected:
            patient = row["patient"]
            files = ios_index.get(patient, {})
            if set(files) != {"upper", "lower"}:
                raise RuntimeError(f"Expected one standard upper/lower pair for {patient}: {files}")
            patient_dir = args.output_dir / patient
            patient_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "patient": patient,
                "category": category(row),
                "reasons": sorted(reasons[patient]),
                "source_orientation_signature": row["orientation_signature"],
                "source_front_angle_degrees": float(row["front_angle_degrees"]),
                "source_upper_minus_lower_vertical": float(row["upper_minus_lower_vertical"]),
                "files": {},
            }
            for arch in ("lower", "upper"):
                info = files[arch]
                if info.file_size <= 84:
                    raise RuntimeError(f"Invalid {patient} {arch}: {info.file_size} bytes")
                output_path = patient_dir / f"ios_{arch}.stl"
                output_path.write_bytes(archive.read(info))
                record["files"][arch] = {
                    "zip_member": info.filename,
                    "size_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "output_path": str(output_path),
                }
            manifest.append(record)

    manifest_path = args.output_dir / "pilot_selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_zip": str(args.zip_path),
                "source_audit_csv": str(args.audit_csv),
                "count": len(manifest),
                "cases": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(manifest), "patients": [item["patient"] for item in manifest]}, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
