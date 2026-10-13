#!/usr/bin/env python3
"""Build a patient-level manifest for an extracted Bite2Text release.

The script is dependency-light on purpose. Pillow is used when available for
image verification; STL files are checked with a small standard-library parser.
No dataset files are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mha", ".dzi"}
PATIENT_PATTERN = re.compile(r"^F\d+$", re.IGNORECASE)


def normalize_report(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect_stl(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "format": "missing",
        "triangles": 0,
        "readable": False,
        "error": "",
    }
    if not path.is_file():
        result["error"] = "missing"
        return result

    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(84)
        if len(header) < 15:
            raise ValueError("file is too small to be an STL")

        ascii_header = header[:80].lstrip().lower().startswith(b"solid")
        binary_match = False
        binary_triangles = 0
        if len(header) >= 84:
            binary_triangles = struct.unpack("<I", header[80:84])[0]
            binary_match = size == 84 + 50 * binary_triangles

        if binary_match:
            result.update(
                format="binary_stl",
                triangles=binary_triangles,
                readable=binary_triangles > 0,
            )
        elif ascii_header:
            # Full ASCII parsing would reread many gigabytes. Integrity is
            # already covered by unzip -t; here we verify recognizable tokens.
            with path.open("rb") as handle:
                sample = handle.read(min(size, 1024 * 1024)).lower()
            facets = sample.count(b"facet normal")
            vertices = sample.count(b"vertex")
            result.update(
                format="ascii_stl",
                triangles=facets,
                readable=facets > 0 and vertices >= 3,
            )
        else:
            result.update(format="unknown_stl", readable=False)
            result["error"] = "unrecognized STL encoding or inconsistent binary size"
    except Exception as exc:  # audit must continue after one malformed sample
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_image(path: Path) -> tuple[bool, str, int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
        return True, image_format, width, height
    except ImportError:
        # The zip integrity test already validates bytes. Without Pillow, use
        # standard-library parsers for the JPEG/PNG formats in this release.
        try:
            with path.open("rb") as handle:
                data = handle.read(32)
            if data.startswith(b"\xff\xd8\xff"):
                with path.open("rb") as handle:
                    handle.read(2)
                    while True:
                        byte = handle.read(1)
                        while byte == b"\xff":
                            byte = handle.read(1)
                        if not byte:
                            break
                        marker = byte[0]
                        if marker in {0xD8, 0xD9}:
                            continue
                        length_raw = handle.read(2)
                        if len(length_raw) != 2:
                            break
                        segment_length = struct.unpack(">H", length_raw)[0]
                        if marker in {
                            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
                        }:
                            precision_height_width = handle.read(5)
                            if len(precision_height_width) != 5:
                                break
                            height, width = struct.unpack(">HH", precision_height_width[1:])
                            return width > 0 and height > 0, "JPEG", width, height
                        handle.seek(max(0, segment_length - 2), 1)
                return False, "JPEG_NO_SOF", 0, 0
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                if len(data) < 24 or data[12:16] != b"IHDR":
                    return False, "PNG_NO_IHDR", 0, 0
                width, height = struct.unpack(">II", data[16:24])
                return width > 0 and height > 0, "PNG", width, height
            if data[:4] in {b"II*\x00", b"MM\x00*"}:
                return True, "TIFF", 0, 0
            return False, "UNKNOWN", 0, 0
        except Exception as exc:
            return False, f"ERROR:{type(exc).__name__}:{exc}", 0, 0
    except Exception as exc:
        return False, f"ERROR:{type(exc).__name__}:{exc}", 0, 0


def report_files(patient_dir: Path, folder: str) -> list[Path]:
    root = patient_dir / folder
    return sorted(root.glob("*.txt")) if root.is_dir() else []


def read_report(path: Path) -> tuple[str, str]:
    try:
        return normalize_report(path.read_text(encoding="utf-8")), ""
    except UnicodeDecodeError:
        try:
            return normalize_report(path.read_text(encoding="utf-8", errors="replace")), "decode_replaced"
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    patient_dirs = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and PATIENT_PATTERN.match(path.name)
    )
    if not patient_dirs:
        raise SystemExit(f"No patient directories found under {dataset_root}")

    rows: list[dict[str, Any]] = []
    report_index: list[dict[str, Any]] = []
    image_errors: list[str] = []
    mesh_errors: list[str] = []
    report_errors: list[str] = []
    photo_count_distribution: Counter[int] = Counter()
    total_en_distribution: Counter[int] = Counter()
    unique_en_distribution: Counter[int] = Counter()
    image_format_distribution: Counter[str] = Counter()
    mesh_format_distribution: Counter[str] = Counter()

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        upper_path = patient_dir / "ios" / "ios_upper.stl"
        lower_path = patient_dir / "ios" / "ios_lower.stl"
        upper = inspect_stl(upper_path)
        lower = inspect_stl(lower_path)
        mesh_format_distribution[upper["format"]] += 1
        mesh_format_distribution[lower["format"]] += 1
        if not upper["readable"]:
            mesh_errors.append(f"{patient_id}\tupper\t{upper['error']}")
        if not lower["readable"]:
            mesh_errors.append(f"{patient_id}\tlower\t{lower['error']}")

        photo_dir = patient_dir / "intraoral-photo"
        photos = sorted(
            path
            for path in photo_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if photo_dir.is_dir() else []
        photo_count_distribution[len(photos)] += 1
        valid_photo_count = 0
        photo_shapes: list[str] = []
        for photo in photos:
            valid, image_format, width, height = inspect_image(photo)
            image_format_distribution[image_format] += 1
            if valid:
                valid_photo_count += 1
            else:
                image_errors.append(f"{patient_id}\t{photo}\t{image_format}")
            photo_shapes.append(f"{photo.name}:{width}x{height}:{image_format}")

        report_groups = {
            "ios_en": report_files(patient_dir, "reports_ios_en"),
            "photo_en": report_files(patient_dir, "reports_intraoral-photo_en"),
            "ios_it": report_files(patient_dir, "reports_ios_it"),
            "photo_it": report_files(patient_dir, "reports_intraoral-photo_it"),
        }
        unique_en: dict[str, dict[str, Any]] = {}
        for source in ("ios_en", "photo_en"):
            for report_path in report_groups[source]:
                normalized, error = read_report(report_path)
                report_hash = sha256_text(normalized.lower()) if normalized else ""
                if error:
                    report_errors.append(f"{patient_id}\t{report_path}\t{error}")
                entry = {
                    "patient_id": patient_id,
                    "source": source,
                    "path": str(report_path),
                    "text": normalized,
                    "normalized_sha256": report_hash,
                    "error": error,
                }
                report_index.append(entry)
                if normalized:
                    unique_en.setdefault(report_hash, entry)

        total_en = len(report_groups["ios_en"]) + len(report_groups["photo_en"])
        total_en_distribution[total_en] += 1
        unique_en_distribution[len(unique_en)] += 1

        reasons: list[str] = []
        if not upper["readable"]:
            reasons.append("upper_ios_missing_or_unreadable")
        if not lower["readable"]:
            reasons.append("lower_ios_missing_or_unreadable")
        if not photos:
            reasons.append("photos_missing")
        elif valid_photo_count != len(photos):
            reasons.append("one_or_more_photos_unreadable")
        if total_en == 0:
            reasons.append("english_report_missing")
        if len(unique_en) == 0:
            reasons.append("english_report_empty_or_unreadable")

        rows.append(
            {
                "patient_id": patient_id,
                "upper_ios_path": str(upper_path) if upper_path.exists() else "",
                "lower_ios_path": str(lower_path) if lower_path.exists() else "",
                "upper_ios_readable": upper["readable"],
                "lower_ios_readable": lower["readable"],
                "upper_ios_bytes": upper["bytes"],
                "lower_ios_bytes": lower["bytes"],
                "upper_triangles": upper["triangles"],
                "lower_triangles": lower["triangles"],
                "upper_stl_format": upper["format"],
                "lower_stl_format": lower["format"],
                "photo_count": len(photos),
                "valid_photo_count": valid_photo_count,
                "photo_paths": json.dumps([str(path) for path in photos], ensure_ascii=False),
                "photo_shapes": json.dumps(photo_shapes, ensure_ascii=False),
                "ios_report_en_count": len(report_groups["ios_en"]),
                "photo_report_en_count": len(report_groups["photo_en"]),
                "total_report_en_count": total_en,
                "unique_report_en_count": len(unique_en),
                "ios_report_it_count": len(report_groups["ios_it"]),
                "photo_report_it_count": len(report_groups["photo_it"]),
                "english_report_paths": json.dumps(
                    [str(path) for source in ("ios_en", "photo_en") for path in report_groups[source]],
                    ensure_ascii=False,
                ),
                "is_complete": not reasons,
                "exclusion_reason": ";".join(reasons),
            }
        )

    fieldnames = list(rows[0].keys())
    with (output_dir / "data_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "report_index.jsonl").open("w", encoding="utf-8") as handle:
        for entry in report_index:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    valid_ids = [row["patient_id"] for row in rows if row["is_complete"]]
    invalid_rows = [row for row in rows if not row["is_complete"]]
    (output_dir / "valid_patient_ids.txt").write_text("\n".join(valid_ids) + "\n", encoding="utf-8")
    (output_dir / "invalid_cases.txt").write_text(
        "\n".join(f"{row['patient_id']}\t{row['exclusion_reason']}" for row in invalid_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "mesh_errors.txt").write_text("\n".join(mesh_errors) + "\n", encoding="utf-8")
    (output_dir / "image_errors.txt").write_text("\n".join(image_errors) + "\n", encoding="utf-8")
    (output_dir / "report_errors.txt").write_text("\n".join(report_errors) + "\n", encoding="utf-8")

    stats = {
        "dataset_root": str(dataset_root),
        "patients": len(rows),
        "complete_patients": len(valid_ids),
        "invalid_patients": len(invalid_rows),
        "photos": sum(int(row["photo_count"]) for row in rows),
        "valid_photos": sum(int(row["valid_photo_count"]) for row in rows),
        "english_reports": sum(int(row["total_report_en_count"]) for row in rows),
        "unique_english_reports_within_patient": sum(int(row["unique_report_en_count"]) for row in rows),
        "photo_count_distribution": dict(sorted(photo_count_distribution.items())),
        "total_en_report_distribution": dict(sorted(total_en_distribution.items())),
        "unique_en_report_distribution": dict(sorted(unique_en_distribution.items())),
        "image_format_distribution": dict(sorted(image_format_distribution.items())),
        "mesh_format_distribution": dict(sorted(mesh_format_distribution.items())),
        "mesh_error_count": len(mesh_errors),
        "image_error_count": len(image_errors),
        "report_error_count": len(report_errors),
    }
    (output_dir / "report_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
