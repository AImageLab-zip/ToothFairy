#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load the competition five-frame TIFF in canonical view order."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


CANONICAL_ORDER = ("frontal", "left", "right", "upper", "lower")


def _rgb_copy(image: Image.Image) -> Image.Image:
    return image.convert("RGB").copy()


def _load_frames(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as image:
        count = int(getattr(image, "n_frames", 1))
        for index in range(count):
            image.seek(index)
            frames.append(_rgb_copy(image))
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


def load_five_view_tiff(path: str | Path, layout: str | None = None) -> dict[str, Image.Image]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frames = _load_frames(source)
    if len(frames) == 1:
        frames = _split_contact_sheet(
            frames[0],
            layout or os.environ.get("B2T_SINGLE_TIFF_LAYOUT", "auto"),
        )
    if len(frames) != 5:
        raise RuntimeError(f"Expected exactly five TIFF frames, found {len(frames)} in {source}")
    return dict(zip(CANONICAL_ORDER, frames, strict=True))
