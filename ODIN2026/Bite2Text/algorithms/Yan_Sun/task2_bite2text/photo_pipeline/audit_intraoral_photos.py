#!/usr/bin/env python3
"""Audit Bite2Text intraoral photographs and create labelled contact sheets.

The script is deliberately read-only with respect to the source dataset.  It
creates a machine-readable manifest, aggregate statistics, a Markdown report,
and optional contact sheets under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def natural_key(value: str) -> list[Any]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def image_files(photo_dir: Path) -> list[Path]:
    return sorted(
        (path for path in photo_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: natural_key(path.name),
    )


def inspect_image(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "filename": path.name,
        "extension": path.suffix.lower().lstrip("."),
        "bytes": path.stat().st_size,
        "valid": False,
        "width": None,
        "height": None,
        "mode": None,
        "exif_orientation": None,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            row.update(
                width=image.width,
                height=image.height,
                mode=image.mode,
                exif_orientation=image.getexif().get(274),
            )
        with Image.open(path) as image:
            image.verify()
        row["valid"] = True
    except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def choose_contact_sheet_patients(
    patient_rows: dict[str, list[dict[str, Any]]], requested: list[str], max_auto: int
) -> list[str]:
    if requested:
        return [patient_id for patient_id in requested if patient_id in patient_rows]

    by_count: dict[int, list[str]] = defaultdict(list)
    for patient_id, rows in patient_rows.items():
        by_count[len(rows)].append(patient_id)

    selected: list[str] = []
    for count in sorted(by_count):
        selected.extend(sorted(by_count[count], key=natural_key)[:2])
    return selected[:max_auto]


def make_contact_sheet(
    patient_id: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    source_root: Path,
    thumb_size: tuple[int, int] = (420, 315),
    columns: int = 3,
) -> None:
    font = ImageFont.load_default()
    label_height = 36
    padding = 12
    cell_width = thumb_size[0] + 2 * padding
    cell_height = thumb_size[1] + label_height + 2 * padding
    n_rows = max(1, (len(rows) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_width, n_rows * cell_height + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 12), f"Patient: {patient_id} | images: {len(rows)}", fill="black", font=font)

    for index, row in enumerate(rows):
        grid_row, grid_col = divmod(index, columns)
        x0 = grid_col * cell_width + padding
        y0 = 42 + grid_row * cell_height + padding
        source_path = source_root / patient_id / "intraoral-photo" / row["filename"]
        try:
            with Image.open(source_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                x = x0 + (thumb_size[0] - image.width) // 2
                y = y0 + (thumb_size[1] - image.height) // 2
                canvas.paste(image, (x, y))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            draw.rectangle((x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]), fill="#dddddd")
            draw.text((x0 + 8, y0 + 8), f"Cannot open\n{type(exc).__name__}", fill="#aa0000", font=font)
        draw.text(
            (x0, y0 + thumb_size[1] + 8),
            f"{index + 1}. {row['filename']}  {row.get('width')}x{row.get('height')}",
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def percent(value: int, total: int) -> str:
    return f"{100.0 * value / total:.1f}%" if total else "0.0%"


def counter_rows(counter: Counter[Any]) -> Iterable[str]:
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        yield f"| {key} | {value} |"


def write_markdown_report(
    output_path: Path,
    raw_root: Path,
    patient_rows: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    total_patients = summary["total_patients"]
    count_distribution = Counter({int(k): v for k, v in summary["images_per_patient"].items()})
    non_five = [
        (patient_id, len(rows), ", ".join(row["filename"] for row in rows))
        for patient_id, rows in patient_rows.items()
        if len(rows) != 5
    ]
    invalid = [
        (patient_id, row["filename"], row["error"])
        for patient_id, rows in patient_rows.items()
        for row in rows
        if not row["valid"]
    ]
    lines = [
        "# Bite2Text 口内照数据审计",
        "",
        f"- 数据根目录：`{raw_root}`",
        f"- 病例数：**{total_patients}**",
        f"- 图像文件数：**{summary['total_images']}**",
        f"- 恰有 5 张照片的病例：**{summary['patients_with_five']} / {total_patients}** "
        f"（{percent(summary['patients_with_five'], total_patients)}）",
        f"- 非 5 张照片的病例：**{len(non_five)}**",
        f"- 无法解码的图像：**{len(invalid)}**",
        "",
        "## 每例照片数量",
        "",
        "| 照片数 | 病例数 |",
        "|---:|---:|",
        *counter_rows(count_distribution),
        "",
        "## 文件格式",
        "",
        "| 格式 | 文件数 |",
        "|---|---:|",
        *counter_rows(Counter(summary["extensions"])),
        "",
        "## 常见尺寸",
        "",
        "| 宽×高 | 文件数 |",
        "|---|---:|",
        *counter_rows(Counter(summary["dimensions"])),
        "",
        "## 非 5 张病例",
        "",
        "| 病例 | 数量 | 文件名 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {patient_id} | {count} | {names} |" for patient_id, count, names in non_five)
    lines.extend(["", "## 无法解码的文件", ""])
    if invalid:
        lines.extend(["| 病例 | 文件 | 错误 |", "|---|---|---|"])
        lines.extend(f"| {patient_id} | {filename} | {error} |" for patient_id, filename, error in invalid)
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "不能假设每例都严格提供五张照片，也不能直接截取排序后的前五张。下一步应先通过联系图确认命名与视角的稳定对应关系，再制定缺失视角、额外照片和 HEIC 的确定性处理规则。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patients", nargs="*", default=[])
    parser.add_argument("--max-auto-contact-sheets", type=int, default=16)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    args = parser.parse_args()

    register_heif()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_rows: dict[str, list[dict[str, Any]]] = {}
    for patient_dir in sorted((path for path in raw_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name)):
        photo_dir = patient_dir / "intraoral-photo"
        if not photo_dir.is_dir():
            continue
        rows: list[dict[str, Any]] = []
        for position, path in enumerate(image_files(photo_dir), start=1):
            row = inspect_image(path)
            row.update(patient_id=patient_dir.name, source_position=position, source_path=str(path))
            rows.append(row)
        patient_rows[patient_dir.name] = rows

    all_rows = [row for rows in patient_rows.values() for row in rows]
    count_distribution = Counter(len(rows) for rows in patient_rows.values())
    extension_distribution = Counter(row["extension"] for row in all_rows)
    dimension_distribution = Counter(
        f"{row['width']}×{row['height']}" for row in all_rows if row["valid"]
    )
    sizes = [row["bytes"] for row in all_rows]
    summary = {
        "raw_root": str(raw_root),
        "total_patients": len(patient_rows),
        "total_images": len(all_rows),
        "patients_with_five": count_distribution[5],
        "images_per_patient": dict(sorted(count_distribution.items())),
        "extensions": dict(sorted(extension_distribution.items())),
        "dimensions": dict(dimension_distribution.most_common(30)),
        "distinct_dimensions": len(dimension_distribution),
        "valid_images": sum(bool(row["valid"]) for row in all_rows),
        "invalid_images": sum(not row["valid"] for row in all_rows),
        "file_size_bytes": {
            "min": min(sizes) if sizes else None,
            "median": int(statistics.median(sizes)) if sizes else None,
            "max": max(sizes) if sizes else None,
        },
    }

    fieldnames = [
        "patient_id",
        "source_position",
        "filename",
        "extension",
        "bytes",
        "valid",
        "width",
        "height",
        "mode",
        "exif_orientation",
        "error",
        "source_path",
    ]
    with (output_dir / "photo_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "photo_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(output_dir / "PHOTO_DATA_AUDIT.md", raw_root, patient_rows, summary)

    if not args.skip_contact_sheets:
        selected = choose_contact_sheet_patients(
            patient_rows, args.patients, args.max_auto_contact_sheets
        )
        for patient_id in selected:
            make_contact_sheet(
                patient_id,
                patient_rows[patient_id],
                output_dir / "contact_sheets" / f"{patient_id}.jpg",
                raw_root,
            )
        (output_dir / "contact_sheet_patients.txt").write_text(
            "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
