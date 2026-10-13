"""Composite contact sheets.

Only composites are kept under `plots/` — per-case files are written to a temp dir, tiled,
and deleted. Comparing renderers means seeing them next to each other; a directory of a
hundred single-case PNGs is not reviewable and was how the geometry failures went unnoticed.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

BAR = 16


def tile_sheet(panels: list[tuple[str, Path | Image.Image]], out: Path, cols: int = 4,
               tile: tuple[int, int] = (430, 180), bg: tuple[int, int, int] = (10, 10, 10)) -> Path:
    """Tile ``(label, image)`` panels into one sheet, letterboxed to a common tile size."""
    tw, th = tile
    cells: list[Image.Image] = []
    for label, src in panels:
        im = src if isinstance(src, Image.Image) else Image.open(src)
        im = im.convert("RGB")
        w = max(1, min(int(im.width * th / im.height), tw))
        cell = Image.new("RGB", (tw, th + BAR), (16, 16, 16))
        cell.paste(im.resize((w, th)), ((tw - w) // 2, BAR))
        ImageDraw.Draw(cell).text((4, 3), f"{label}  {im.width}x{im.height}",
                                  fill=(255, 210, 110))
        cells.append(cell)
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 4) + 4, rows * (th + BAR + 4) + 4), bg)
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * (tw + 4) + 4, (i // cols) * (th + BAR + 4) + 4))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def figure_sheet(figs: list[Path], out: Path, cols: int = 2,
                 width: int = 1100, bg: tuple[int, int, int] = (10, 10, 10)) -> Path:
    """Tile whole matplotlib figures (already labelled) into one sheet at a common width."""
    ims = []
    for p in figs:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((width, max(1, int(im.height * width / im.width)))))
    if not ims:
        raise ValueError("no figures to tile")
    h = max(im.height for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (width + 4) + 4, rows * (h + 4) + 4), bg)
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (width + 4) + 4, (i // cols) * (h + 4) + 4))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
