"""Reading intraoral photographs into the five named views.

Two input layouts are supported:

1. Separate files named ``intraoral_1`` ... ``intraoral_5`` (any image
   extension). This is the Bites2Text layout; the name carries the view:

       1 front   2 right buccal   3 left buccal   4 lower occlusal   5 upper occlusal

2. Anything else, e.g. the single stacked ``.tif`` / ``.mha`` Grand Challenge
   delivers. Frames are read in file-name order and assigned to the views in
   the order above. Fewer than five frames fill the leading slots and the rest
   are masked out of the model's pooling.

Grand Challenge files can be multi-gigapixel pyramidal TIFFs, so page geometry
is always read from headers first and no page above ``MAX_DECODE_PIXELS`` is
ever decoded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from .photo_findings import VIEW_ORDER

log = logging.getLogger("bite2text")

PHOTO_SUFFIXES = (".mha", ".mhd", ".nrrd", ".nii", ".tif", ".tiff",
                  ".jpg", ".jpeg", ".png", ".bmp")

VIEW_BY_STEM = {f"intraoral_{i}": view for i, view in enumerate(VIEW_ORDER, start=1)}

MAX_DECODE_PIXELS = 40_000_000   # ~360 MB as uint8 RGB
MIN_VIEW_SIDE = 224              # network input size; larger pyramid levels buy nothing


def list_photos(paths: Iterable) -> List[Path]:
    """Expand directories and keep files with an image suffix (case-insensitive)."""
    found: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in PHOTO_SUFFIXES)
        elif p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES:
            found.append(p)
        else:
            raise FileNotFoundError(f"not an image file or directory: {p}")
    return found


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def _to_uint8(a):
    import numpy as np

    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    a = a.astype("float64")
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not hi > lo:
        return np.zeros(a.shape, dtype="uint8")
    return (((a - lo) / (hi - lo)) * 255.0).round().astype("uint8")


def _frames_from_array(arr) -> List[Image.Image]:
    arr = _to_uint8(arr)
    if arr.ndim == 4:                                    # (frames, H, W, C)
        return [Image.fromarray(a).convert("RGB") for a in arr]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):        # one colour image
        return [Image.fromarray(arr).convert("RGB")]
    if arr.ndim == 3:                                    # (frames, H, W) grey
        return [Image.fromarray(a).convert("RGB") for a in arr]
    if arr.ndim == 2:
        return [Image.fromarray(arr).convert("RGB")]
    return []


def _page_hw(shape, axes=None):
    dims = [d for d in shape if d > 1]
    if axes:
        yx = [d for d, a in zip(shape, axes) if a in ("Y", "X")]
        if len(yx) == 2:
            return yx[0], yx[1]
    spatial = [d for d in dims if d not in (3, 4)]
    if len(spatial) >= 2:
        return spatial[-2], spatial[-1]
    return (dims[-1], 1) if dims else (0, 0)


def _looks_like_pyramid(areas) -> bool:
    """True if areas step down ~4x per level (one image at several scales).

    Five photographs of one mouth are within a small factor of each other, so
    the two regimes are easy to tell apart.
    """
    ordered = sorted((a for a in areas if a > 0), reverse=True)
    if len(ordered) < 3:
        return False
    return all(2.0 <= ordered[i] / ordered[i + 1] <= 8.0 for i in range(len(ordered) - 1))


def _frames_pil(path) -> List[Image.Image]:
    """Pillow pages. Handles tiled BigTIFF with JPEG tiles, which tifffile may refuse."""
    from PIL import ImageSequence

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None          # size-checked per page below instead
    try:
        with Image.open(path) as im:
            meta = []
            for index, frame in enumerate(ImageSequence.Iterator(im)):
                width, height = frame.size
                try:                       # TIFF tag 254 bit 0 = reduced-resolution copy
                    reduced = bool(int(frame.tag_v2.get(254, 0) or 0) & 1)
                except Exception:
                    reduced = False
                meta.append((index, width, height, reduced))

            chosen = [(i, w, h) for i, w, h, r in meta if not r and 0 < w * h <= MAX_DECODE_PIXELS]
            if not chosen:
                usable = sorted(((i, w, h) for i, w, h, _ in meta
                                 if min(w, h) >= MIN_VIEW_SIDE and 0 < w * h <= MAX_DECODE_PIXELS),
                                key=lambda t: t[1] * t[2])
                chosen = usable[:1]

            frames = []
            for index, _, _ in chosen:
                im.seek(index)
                frames.append(im.convert("RGB").copy())
            return frames
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def _frames_tifffile(path) -> List[Image.Image]:
    """Page-wise TIFF decoding that never allocates more than MAX_DECODE_PIXELS."""
    import tifffile

    with tifffile.TiffFile(str(path)) as handle:
        pages = list(handle.pages)

        def area(page):
            h, w = _page_hw(tuple(page.shape), getattr(page, "axes", None))
            return h * w

        chosen = [p for p in pages
                  if not getattr(p, "is_reduced", False) and 0 < area(p) <= MAX_DECODE_PIXELS]
        if len(chosen) >= len(VIEW_ORDER) and _looks_like_pyramid([area(p) for p in chosen]):
            raise RuntimeError(f"the readable pages of {Path(path).name} are resolution levels "
                               f"of one image, not {len(VIEW_ORDER)} views")

        frames: List[Image.Image] = []
        for page in chosen:
            frames.extend(_frames_from_array(page.asarray()))
            if len(frames) >= len(VIEW_ORDER):
                break
        if frames:
            return frames

        # One image stored as a pyramid: decode the smallest page that still covers the input size.
        for size, _, page in sorted(((area(p), i, p) for i, p in enumerate(pages) if area(p) > 0),
                                    key=lambda t: t[0]):
            h, w = _page_hw(tuple(page.shape), getattr(page, "axes", None))
            if min(h, w) >= MIN_VIEW_SIDE and size <= MAX_DECODE_PIXELS:
                return _frames_from_array(page.asarray())

        raise MemoryError(f"no page of {Path(path).name} is both under {MAX_DECODE_PIXELS:,} "
                          f"pixels and at least {MIN_VIEW_SIDE}px on its short side")


def _frames_sitk(path) -> List[Image.Image]:
    import SimpleITK as sitk

    return _frames_from_array(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))


def read_frames(path) -> List[Image.Image]:
    """Every usable frame of an image file; tries Pillow, tifffile and SimpleITK."""
    suffix = Path(path).suffix.lower()
    readers = ([_frames_sitk, _frames_tifffile, _frames_pil]
               if suffix in (".mha", ".mhd", ".nii", ".gz", ".nrrd")
               else [_frames_pil, _frames_tifffile, _frames_sitk])

    problems = []
    for reader in readers:
        try:
            frames = reader(path)
        except Exception as exc:
            problems.append(f"{reader.__name__}: {type(exc).__name__}: {exc}")
            continue
        if len(frames) > 1 and _looks_like_pyramid([f.size[0] * f.size[1] for f in frames]):
            usable = [f for f in frames if min(f.size) >= MIN_VIEW_SIDE]
            frames = [min(usable, key=lambda f: f.size[0] * f.size[1]) if usable
                      else max(frames, key=lambda f: f.size[0] * f.size[1])]
            log.info("%s: pyramid levels of one image; using a single frame", Path(path).name)
        if frames:
            return frames
        problems.append(f"{reader.__name__}: 0 frames")

    raise RuntimeError(f"no decoder could read {Path(path).name} "
                       f"({os.path.getsize(path)} bytes): {problems}")


# ---------------------------------------------------------------------------
# View assignment
# ---------------------------------------------------------------------------

def resolve_photo_views(photos: Iterable) -> Dict[str, Image.Image]:
    """Photo files -> ``{view name: PIL image}`` (possibly a partial set)."""
    photos = [Path(p) for p in photos]

    named: Dict[str, Path] = {}
    for path in photos:
        view = VIEW_BY_STEM.get(path.stem.lower())
        if view and view not in named:
            named[view] = path
    if len(named) == len(VIEW_ORDER):
        return {view: Image.open(path) for view, path in named.items()}

    frames: List[Image.Image] = []
    for path in sorted(photos, key=lambda p: p.name):
        frames.extend(read_frames(path))
    if len(frames) > len(VIEW_ORDER):
        log.warning("%d frames found; using the first %d", len(frames), len(VIEW_ORDER))
    views = dict(zip(VIEW_ORDER, frames))
    if views:
        log.info("photo views from frame order: %s", list(views))
    return views
