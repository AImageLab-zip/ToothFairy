#!/usr/bin/env python3
"""Build an official-layout multi-page TIFF smoke fixture from raw photos."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    case_source = args.raw_root / args.case_id
    photo_source = case_source / "intraoral-photo"
    output = args.output_root / args.case_id
    lower_dir = output / "files" / "ios-lower"
    upper_dir = output / "files" / "ios-upper"
    photo_dir = output / "images" / "intraoral-photo"
    lower_dir.mkdir(parents=True, exist_ok=True)
    upper_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(case_source / "ios" / "ios_lower.stl", lower_dir / "ios_lower.stl")
    shutil.copy2(case_source / "ios" / "ios_upper.stl", upper_dir / "ios_upper.stl")
    sources = sorted(
        path
        for path in photo_source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    frames: list[Image.Image] = []
    for source in sources:
        try:
            with Image.open(source) as image:
                frame = ImageOps.exif_transpose(image).convert("RGB")
                frame.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                frames.append(frame.copy())
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"No decodable photos for {args.case_id}")
    tiff_path = photo_dir / "intraoral-photo.tiff"
    frames[0].save(tiff_path, save_all=True, append_images=frames[1:], compression="tiff_lzw")

    manifest = [
        {
            "socket": {
                "slug": "3d-lower-teeth-scan",
                "relative_path": "files/ios-lower",
                "is_image_kind": False,
                "is_panimg_kind": False,
                "is_file_kind": True,
            },
            "file": {"name": "ios_lower.stl"},
            "image": None,
            "value": None,
        },
        {
            "socket": {
                "slug": "3d-upper-teeth-scan",
                "relative_path": "files/ios-upper",
                "is_image_kind": False,
                "is_panimg_kind": False,
                "is_file_kind": True,
            },
            "file": {"name": "ios_upper.stl"},
            "image": None,
            "value": None,
        },
        {
            "socket": {
                "slug": "2d-intraoral-photographs",
                "relative_path": "images/intraoral-photo",
                "is_image_kind": True,
                "is_panimg_kind": True,
                "is_file_kind": False,
            },
            "file": None,
            "image": {"name": "intraoral-photo.tiff"},
            "value": None,
        },
    ]
    (output / "inputs.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"case_id": args.case_id, "frames": len(frames), "output": str(output)}))


if __name__ == "__main__":
    main()
