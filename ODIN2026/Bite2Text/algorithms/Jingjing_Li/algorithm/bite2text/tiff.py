"""Canonical five-view intraoral TIFF decoding shared by every photo task."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


# Grand Challenge physical frame order:
# 1=frontal, 2=left, 3=right, 4=upper occlusal, 5=lower occlusal.
CANONICAL_VIEW_ORDER = ("frontal", "left", "right", "upper_occlusal", "lower_occlusal")
# Frozen training slots: 1=frontal, 2=right, 3=left, 4=lower occlusal, 5=upper occlusal.
TRAINING_SLOT_ORDER = ("frontal", "right", "left", "lower_occlusal", "upper_occlusal")
# Some ODIN cases include auxiliary intraoral_11 / intraoral_111 views. Grand
# Challenge stacks source filenames lexicographically, so the canonical
# intraoral_1..5 views occupy these frame positions.
CANONICAL_FRAME_INDICES = {
    5: (0, 1, 2, 3, 4),
    6: (0, 2, 3, 4, 5),
    7: (0, 3, 4, 5, 6),
    8: (0, 3, 4, 5, 6),
}



def load_five_view_tiff(path: Path | str, layout: str | None = None) -> dict[str, Image.Image]:
    """Decode the canonical five views from supported ODIN TIFF stacks."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frames = _load_frames(source)
    if len(frames) == 1:
        frames = _split_contact_sheet(
            frames[0],
            layout or os.environ.get("B2T_SINGLE_TIFF_LAYOUT", "auto"),
        )
    frame_indices = CANONICAL_FRAME_INDICES.get(len(frames))
    if frame_indices is None:
        raise RuntimeError(
            f"Expected a supported 5-8 frame TIFF stack, found {len(frames)} in {source}"
        )
    if len(frames) != len(CANONICAL_VIEW_ORDER):
        print(
            "B2T_TIFF_FRAME_SELECTION "
            f"total={len(frames)} selected={','.join(map(str, frame_indices))} "
            f"source={source}",
            flush=True,
        )
    selected_frames = [frames[index] for index in frame_indices]
    return dict(zip(CANONICAL_VIEW_ORDER, selected_frames, strict=True))


def training_slot_views(photos: dict[str, Image.Image]) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """Return frozen training slots intraoral_1 through intraoral_5 from semantic views."""
    return tuple(photos[view] for view in TRAINING_SLOT_ORDER)


def _load_frames(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as image:
        for index in range(int(getattr(image, "n_frames", 1))):
            image.seek(index)
            frames.append(image.convert("RGB").copy())
    return frames


def _split_contact_sheet(image: Image.Image, layout: str) -> list[Image.Image]:
    width, height = image.size
    normalized = layout.strip().lower()
    if normalized == "auto":
        ratio = width / max(height, 1)
        if ratio >= 4.0:
            normalized = "horizontal5"
        elif ratio <= 0.25:
            normalized = "vertical5"
        else:
            raise RuntimeError(
                "Single-frame TIFF layout is ambiguous; set B2T_SINGLE_TIFF_LAYOUT to "
                "horizontal5, vertical5, grid_2x3, or grid_3x2"
            )

    if normalized == "horizontal5":
        edges = [round(index * width / 5) for index in range(6)]
        boxes = [(edges[index], 0, edges[index + 1], height) for index in range(5)]
    elif normalized == "vertical5":
        edges = [round(index * height / 5) for index in range(6)]
        boxes = [(0, edges[index], width, edges[index + 1]) for index in range(5)]
    elif normalized == "grid_2x3":
        x_edges = [0, width // 2, width]
        y_edges = [0, height // 3, 2 * height // 3, height]
        boxes = [
            (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
            for row in range(3)
            for column in range(2)
        ][:5]
    elif normalized == "grid_3x2":
        x_edges = [0, width // 3, 2 * width // 3, width]
        y_edges = [0, height // 2, height]
        boxes = [
            (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
            for row in range(2)
            for column in range(3)
        ][:5]
    else:
        raise RuntimeError(f"Unsupported single-frame TIFF layout: {layout!r}")
    return [image.crop(box).convert("RGB") for box in boxes]
