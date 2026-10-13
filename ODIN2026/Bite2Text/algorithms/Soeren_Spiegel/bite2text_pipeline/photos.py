"""Intraoral photo loading for Bite2Text.

Quirks handled (measured on the training zip): EXIF Orientation incl. tag=2
horizontal MIRROR on 4 files (silent L/R corruption without exif_transpose);
HEIC (case F5003) via pillow-heif when available; MPO Nikon multi-picture JPEGs
(PIL reads frame 0); nonstandard `intraoral-photo_<id>_<ts>` names in 17 cases;
2-9 photos per case (mode 5). Photo index -> view mapping is NOT consistent
across cases — never assume an order.
"""
import io
import re

from PIL import Image, ImageOps

try:  # HEIC support is optional; F5003 is the only HEIC case in the training set
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

MAX_SIDE = 1008          # ~860 Qwen tokens per photo at 28px patches
MAX_PHOTOS = 7

_IDX = re.compile(r"intraoral_(\d+)\.", re.IGNORECASE)


def _sort_key(name):
    m = _IDX.search(name)
    return (0, int(m.group(1))) if m else (1, name)


def load_photos_from_zip(zf, case_id, max_side=MAX_SIDE, max_n=MAX_PHOTOS):
    """-> list[PIL.Image] (RGB, EXIF-transposed, downscaled). Never raises;
    unreadable files are skipped."""
    names = sorted((n for n in zf.namelist()
                    if n.startswith(f"{case_id}/intraoral-photo/")
                    and not n.endswith("/")), key=_sort_key)
    return _load([io.BytesIO(zf.read(n)) for n in names[:max_n]],
                 max_side=max_side)


def load_photos_from_files(paths, max_side=MAX_SIDE, max_n=MAX_PHOTOS):
    return _load(list(paths)[:max_n], max_side=max_side)


def load_photos_from_tiff(path, max_side=MAX_SIDE, max_n=MAX_PHOTOS):
    """Container path: the photo set arrives as one (multi-frame) TIFF."""
    imgs = []
    try:
        tif = Image.open(path)
        for i in range(getattr(tif, "n_frames", 1)):
            if i >= max_n:
                break
            tif.seek(i)
            imgs.append(_prep(tif.copy(), max_side))
    except Exception:
        pass
    return imgs


def load_photos_from_mha(path, max_side=MAX_SIDE, max_n=MAX_PHOTOS):
    """Container path: Grand Challenge re-encodes 2-D image sockets as MetaImage, and the
    organizers' template globs `*.mha` FIRST for the intraoral-photo socket, so this is a
    likely delivery form for the photo set even though the training corpus ships jpgs.

    One .mha may hold one photo or a stack of them; both are handled:
      (H,W)        -> 1 grayscale         (H,W,3|4)   -> 1 RGB(A)
      (N,H,W)      -> N grayscale         (N,H,W,3|4) -> N RGB(A)
    SimpleITK is imported lazily so a container without it still runs every other path.
    Never raises -- an unreadable .mha yields [] and the caller falls through to its other
    sources, matching load_photos_from_tiff().
    """
    try:
        import SimpleITK as sitk
        import numpy as np
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    except Exception:
        return []
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        frames = [arr]
    elif arr.ndim in (3, 4):
        frames = list(arr)
    else:
        return []
    out = []
    for fr in frames[:max_n]:
        try:
            if fr.dtype != "uint8":     # 16-bit / float payloads -> 8-bit for PIL
                fr = fr.astype("float64")
                lo, hi = float(fr.min()), float(fr.max())
                fr = ((fr - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(fr)
                fr = fr.astype("uint8")
            out.append(_prep(Image.fromarray(fr), max_side))
        except Exception:
            continue
    return out


def _load(sources, max_side):
    out = []
    for src in sources:
        try:
            out.append(_prep(Image.open(src), max_side))
        except Exception:
            continue
    return out


def _prep(img, max_side):
    img = ImageOps.exif_transpose(img)   # tag=2 mirror flags MUST be applied
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img
