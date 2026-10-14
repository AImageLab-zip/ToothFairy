"""Grand Challenge-style input discovery without private dataset paths."""

from __future__ import annotations

from pathlib import Path

from .schema import CaseInput

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_case(input_dir: str | Path, case_id: str | None = None) -> CaseInput:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    images = tuple(
        sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    )
    if len(images) < 5:
        raise ValueError("Bite2Text requires at least five intraoral photographs")

    stls = tuple(sorted(root.rglob("*.stl")))
    upper = [path for path in stls if "upper" in path.name.lower()]
    lower = [path for path in stls if "lower" in path.name.lower()]
    if len(upper) != 1 or len(lower) != 1:
        raise ValueError("Expected exactly one upper and one lower STL mesh")

    return CaseInput(
        case_id=case_id or root.name,
        image_paths=images,
        upper_stl=upper[0],
        lower_stl=lower[0],
    )
