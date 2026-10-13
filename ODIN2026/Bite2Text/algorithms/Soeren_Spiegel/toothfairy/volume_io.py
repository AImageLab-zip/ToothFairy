"""Zip-streamed + local-disk NIfTI volume I/O for the ToothFairy4 Qwen pipeline.

Port of the volume side of HEADLINE's ``ct_utils.load_volume``,
but wired to ODIN's data stack:
  - reads a ``.nii.gz`` either from inside the 44 GB dataset zip (``read_volume_zip``,
    the exploration/baseline path) or from a local extracted file (``read_volume_path``,
    the local-cache /cache fast path) — same fast struct-based pixel read either way;
  - carries the NIfTI **affine** (parsed by nibabel from the header) so we can
    **reorient every family to a canonical orientation** — HEADLINE's LAS reorient
    analogue. ToothFairy orientation is NOT verified uniform across the 4 families, and a
    flipped L<->R panoramic swaps every FDI quadrant, so this is
    load-bearing, not cosmetic;
  - resamples to an isotropic grid before any arch/cross-section geometry;
  - windows CBCT by **per-volume percentile**.

The fast pixel read mirrors experiments/medgemma4b_v0/volume_utils.read_volume (NIfTI-1
is Fortran order); we only additionally decode the affine for the reorient step.
"""
from __future__ import annotations

import gzip
import struct
import zipfile
from pathlib import Path

import numpy as np
from nibabel import Nifti1Header
from nibabel.orientations import (
    aff2axcodes, apply_orientation, axcodes2ornt, io_orientation, ornt_transform,
)
from PIL import Image
from scipy.ndimage import zoom

from .config import CANONICAL_AXCODES, RESAMPLE_MM, WINDOW_PCT

# NIfTI-1 datatype code -> numpy dtype letter (matches the baseline volume_utils table).
_NP_DTYPE = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
    256: "i1", 512: "u2", 768: "u4",
}


# ── Raw NIfTI parsing (bytes -> array + spacing + affine) ────────────────────────

def _parse_nifti(content: bytes) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """Parse decompressed .nii bytes -> (arr[nx,ny,nz], spacing_mm, affine 4x4).

    Fast struct-based voxel read (no float upcast) + nibabel for the affine so the
    sform/qform selection and quaternion math are the battle-tested implementation.
    """
    sizeof_hdr = struct.unpack_from("<i", content, 0)[0]
    little = sizeof_hdr == 348
    end = "<" if little else ">"

    dim = struct.unpack_from(end + "8h", content, 40)
    datatype = struct.unpack_from(end + "h", content, 70)[0]
    pixdim = struct.unpack_from(end + "8f", content, 76)
    vox_offset = int(struct.unpack_from(end + "f", content, 108)[0])

    ndim = min(int(dim[0]), 3)
    shape = tuple(int(dim[i + 1]) for i in range(ndim))
    spacing = tuple(round(float(pixdim[i + 1]), 4) for i in range(3))

    np_dtype = np.dtype(end + _NP_DTYPE.get(datatype, "i2"))
    n_vox = int(np.prod(shape))
    start = max(vox_offset, 352)
    raw = content[start: start + n_vox * np_dtype.itemsize]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape, order="F").copy()

    hdr = Nifti1Header(binaryblock=bytes(content[:348]),
                       endianness=("<" if little else ">"))
    affine = np.asarray(hdr.get_best_affine(), dtype=np.float64)
    return arr, spacing, affine  # type: ignore[return-value]


def read_volume_zip(zf: zipfile.ZipFile, member: str):
    """Read a .nii.gz volume from inside the dataset zip -> (arr, spacing, affine)."""
    with zf.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
        content = gz.read()
    return _parse_nifti(content)


def read_volume_path(path: str | Path):
    """Read a local (extracted) .nii.gz volume -> (arr, spacing, affine).

    The local-cache fast path: reads from /cache instead of the NFS zip. .nii.gz is gzip, so
    the byte-copied cache file is identical to the in-zip member (see extract_cache)."""
    with gzip.open(path, "rb") as gz:
        content = gz.read()
    return _parse_nifti(content)


# ── Orientation ────────────────────────────────────────────────────────────

def orientation_axcodes(affine: np.ndarray) -> tuple[str, str, str]:
    """Anatomical axis codes of the data axes, e.g. ('L','A','S'). See check_orientation."""
    return tuple(aff2axcodes(affine))  # type: ignore[return-value]


def reorient_to_canonical(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    affine: np.ndarray,
    target: tuple[str, str, str] = CANONICAL_AXCODES,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Reorient ``arr`` (and permute ``spacing``) to ``target`` axcodes.

    Identity when the volume is already canonical. Guarantees axis 2 = inferior->superior
    and consistent L/R, A/P across families so laterality (FDI quadrants) and arch geometry
    are stable — the ToothFairy analogue of HEADLINE's LAS reorient."""
    src = io_orientation(affine)
    dst = axcodes2ornt(target)
    transform = ornt_transform(src, dst)
    reoriented = apply_orientation(arr, transform)
    # transform[i] = (old_axis, flip): new axis i comes from old axis old_axis.
    new_spacing = tuple(spacing[int(transform[i, 0])] for i in range(3))
    return reoriented, new_spacing  # type: ignore[return-value]


# ── Geometry / windowing ─────────────────────────────────────────────────────────

def resample_isotropic(
    arr: np.ndarray,
    spacing: tuple[float, ...],
    target_mm: float = RESAMPLE_MM,
) -> np.ndarray:
    """Resample to isotropic ``target_mm`` (mirrors baseline volume_utils.resample_isotropic)."""
    factors = tuple(s / target_mm for s in spacing[:3])
    if all(abs(f - 1.0) < 0.02 for f in factors):
        return arr
    resampled = zoom(arr.astype(np.float32), factors, order=1, prefilter=False)
    return resampled.astype(arr.dtype)


def z_extent_mm(arr: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Superior-inferior extent of the volume in mm (arr must be canonical: axis 2 = S)."""
    return arr.shape[2] * spacing[2]


def percentile_window(arr: np.ndarray, pct: tuple[float, float] = WINDOW_PCT):
    """Per-volume percentile window bounds (lo, hi). CBCT has no HU, so no fixed window."""
    lo = float(np.percentile(arr, pct[0]))
    hi = float(np.percentile(arr, pct[1]))
    return lo, max(hi, lo + 1.0)


def window_to_uint8(arr2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip a 2D slice to [lo, hi] and map to uint8 [0, 255]."""
    a = np.clip(arr2d.astype(np.float32), lo, hi)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def to_wide_rgb(gray_u8: np.ndarray, height: int, max_width: int | None = None) -> Image.Image:
    """Resize a uint8 grayscale image to ``height`` keeping aspect ratio (no square pad).

    For the panoramic, which is wide (arc length × Z) and must NOT be squished. Both sides
    are rounded to a multiple of 28 so Qwen's ViT does not re-resize (image-token no-op)."""
    h, w = gray_u8.shape
    new_h = max(28, int(round(height / 28)) * 28)
    new_w = max(28, int(round(w * new_h / h)))
    if max_width:
        new_w = min(new_w, max_width)
    new_w = max(28, int(round(new_w / 28)) * 28)
    return Image.fromarray(gray_u8).resize((new_w, new_h), Image.LANCZOS).convert("RGB")


def to_square_rgb(gray_u8: np.ndarray, size: int) -> Image.Image:
    """Pad a uint8 grayscale slice to square (black), resize to size×size, return RGB.

    Padding preserves aspect ratio (no anatomical stretch); an integer area-reduce
    precedes LANCZOS on downscales to suppress aliasing (mirrors HEADLINE _to_square)."""
    img = Image.fromarray(gray_u8)                 # mode "L"
    w, h = img.size
    side = max(w, h)
    if (w, h) != (side, side):
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    if side != size:
        if side > size:
            factor = side // size
            if factor > 1:
                img = img.reduce(factor)
        img = img.resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


# ── Case loading: local /cache first, zip fallback ──────────────────────────────

def load_canonical_volume(
    volume_member: str,
    zf: zipfile.ZipFile | None = None,
    local_path: str | Path | None = None,
    resample: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a case volume, reorient to canonical, (optionally) resample to isotropic.

    Prefers ``local_path`` (extracted /cache file — no NFS, worker-friendly), else reads
    ``volume_member`` from the open zip ``zf``. Returns (arr, spacing) on the canonical grid.
    """
    if local_path is not None and Path(local_path).exists():
        arr, spacing, affine = read_volume_path(local_path)
    elif zf is not None:
        arr, spacing, affine = read_volume_zip(zf, volume_member)
    else:
        raise ValueError("need either an existing local_path or an open zip handle")
    arr, spacing = reorient_to_canonical(arr, spacing, affine)
    if resample:
        arr = resample_isotropic(arr, spacing, RESAMPLE_MM)
        spacing = (RESAMPLE_MM, RESAMPLE_MM, RESAMPLE_MM)
    return arr, spacing
