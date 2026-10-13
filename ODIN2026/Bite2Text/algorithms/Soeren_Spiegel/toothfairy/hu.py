"""Per-volume HU calibration for ToothFairy4 CBCTs.

ToothFairy4 volumes are already stored on an **HU-like** scale (air pins near
−1000, soft tissue peaks near 0, bone spans +500…+3000), but with per-scanner /
per-family offsets and gains, plus metal-artifact outliers (raw max up to
+31767). To make **fixed** HU windows (Task 2) consistent across families
A/P/S/F we harmonise every volume onto a common reference scale via a per-volume
affine estimated from robust intensity landmarks.

Landmark logic (see ``estimate_landmarks``)
-------------------------------------------
Three landmarks are estimated per volume directly from its intensity histogram:

* **air peak** ``air_in`` — the histogram mode inside ``[-1100, -800]``. CBCT
  background/air is the tallest, sharpest peak; its position is the volume's
  additive offset. (Observed range across families: ~−1024 … −998.)
* **soft-tissue peak** ``soft_in`` — the histogram mode inside ``[-200, +300]``
  (muscle / soft tissue / fluid). This is the second anchor; together with air
  it fixes both the offset and the gain. (Observed: ~−28 … +92.)
* **bone reference** ``bone_in`` — a robust high percentile (default 99.5) of the
  in-body voxels, reported for logging/QC only. Bone HU is not used to fit the
  affine (it is contaminated by metal and by per-scanner saturation, e.g. some
  P volumes are hard-clipped at their max), but it is mapped through the fitted
  affine so downstream code can see where bone lands after calibration.

Affine fit
----------
A 1-D affine ``out = scale * in + shift`` is fit so the two robust anchors map to
the reference::

    air_in  -> AIR_REF   (default −1000)
    soft_in -> SOFT_REF  (default    0)

giving ``scale = (SOFT_REF - AIR_REF) / (soft_in - air_in)`` and
``shift = AIR_REF - scale * air_in``. Air therefore stays anchored at exactly
−1000 by construction, and soft tissue is pulled to 0 across all families.

Metal outliers
--------------
After the affine, values are clipped to ``[AIR_REF, clip_hi]`` with
``clip_hi = CLIP_HI_HU`` (default +3100 HU). This caps dense metal / beam-hardening
streaks (raw up to +31767) at a physically-sane dense-bone/restoration ceiling
without touching real anatomy (cortical bone ~ +1800, enamel ~ +2500–3000).

The module is import-safe and side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# ── Reference calibrated HU scale (Task 2 defines windows on this) ───────────────
AIR_REF: float = -1000.0     # calibrated air landmark
SOFT_REF: float = 0.0        # calibrated soft-tissue landmark
CLIP_HI_HU: float = 3100.0   # metal / dense-restoration ceiling after calibration

# Landmark search windows (raw HU-like intensities).
_AIR_LO, _AIR_HI = -1100.0, -800.0
_SOFT_LO, _SOFT_HI = -200.0, 300.0
_BONE_PCT: float = 99.5


@dataclass
class HUCalibration:
    """Fitted per-volume HU-calibration parameters (for logging / QC).

    ``out = clip(scale * in + shift, AIR_REF, clip_hi)``.
    """
    air_in: float       # detected raw air peak
    soft_in: float      # detected raw soft-tissue peak
    bone_in: float      # detected raw bone reference (percentile), pre-affine
    bone_out: float     # bone reference after the affine (QC: where bone lands)
    scale: float        # affine gain
    shift: float        # affine offset
    clip_hi: float      # high clip applied after affine
    air_ref: float = AIR_REF
    soft_ref: float = SOFT_REF

    def as_dict(self) -> dict:
        return asdict(self)


def _hist_mode(values: np.ndarray, lo: float, hi: float, bins: int) -> float | None:
    """Modal intensity inside ``[lo, hi]`` from a histogram, or ``None`` if empty."""
    m = (values >= lo) & (values <= hi)
    sel = values[m]
    if sel.size < 100:
        return None
    h, edges = np.histogram(sel, bins=bins)
    if h.sum() == 0:
        return None
    i = int(h.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def estimate_landmarks(arr: np.ndarray) -> tuple[float, float, float]:
    """Estimate (air_in, soft_in, bone_in) raw landmarks from a volume.

    * air_in  — histogram mode in [-1100, -800] (fallback: AIR_REF).
    * soft_in — histogram mode in [-200, +300] (fallback: SOFT_REF).
    * bone_in — 99.5th percentile of in-body (> soft-ish) voxels (fallback: soft_in).

    Uses a subsample for large volumes to keep percentile/histogram cost bounded;
    landmarks are stable under subsampling because they are modes / high percentiles
    of a very large voxel population.
    """
    a = arr.ravel()
    # Subsample for speed on big grids; modes/percentiles are subsample-stable.
    if a.size > 20_000_000:
        step = a.size // 20_000_000 + 1
        a = a[::step]
    a = a.astype(np.float32, copy=False)

    air = _hist_mode(a, _AIR_LO, _AIR_HI, bins=60)
    air_in = AIR_REF if air is None else air

    soft = _hist_mode(a, _SOFT_LO, _SOFT_HI, bins=100)
    soft_in = SOFT_REF if soft is None else soft

    # Bone reference: high percentile of voxels clearly above soft tissue, so the
    # (huge) air background does not drag the percentile down.
    body = a[a > (soft_in + 100.0)]
    if body.size >= 100:
        bone_in = float(np.percentile(body, _BONE_PCT))
    else:
        bone_in = soft_in
    return air_in, soft_in, bone_in


def fit_calibration(arr: np.ndarray, clip_hi: float = CLIP_HI_HU) -> HUCalibration:
    """Estimate landmarks and fit the affine (air, soft) -> (AIR_REF, SOFT_REF)."""
    air_in, soft_in, bone_in = estimate_landmarks(arr)

    denom = soft_in - air_in
    if abs(denom) < 1.0:            # degenerate; fall back to pure offset
        scale = 1.0
    else:
        scale = (SOFT_REF - AIR_REF) / denom
    shift = AIR_REF - scale * air_in
    bone_out = scale * bone_in + shift
    return HUCalibration(
        air_in=air_in, soft_in=soft_in, bone_in=bone_in, bone_out=bone_out,
        scale=scale, shift=shift, clip_hi=clip_hi,
    )


def apply_calibration(arr: np.ndarray, cal: HUCalibration) -> np.ndarray:
    """Apply a fitted affine + clip, returning float32 calibrated HU."""
    out = arr.astype(np.float32) * np.float32(cal.scale) + np.float32(cal.shift)
    np.clip(out, cal.air_ref, cal.clip_hi, out=out)
    return out


def calibrate_to_hu(
    arr: np.ndarray,
    spacing=None,
    clip_hi: float = CLIP_HI_HU,
    return_params: bool = False,
):
    """Calibrate a CBCT volume onto the common reference HU scale.

    Estimates the air peak (mode near −1000), the soft-tissue peak (mode in
    ~[−200, +300]) and a bone reference (99.5th percentile), fits a per-volume
    affine mapping (air, soft) -> (−1000, 0) HU, applies it, and clips metal
    outliers at ``clip_hi`` (~+3100 HU). Air stays anchored at −1000.

    Parameters
    ----------
    arr : np.ndarray
        Raw HU-like volume (any shape); returned dtype is float32.
    spacing : optional
        Accepted and ignored (landmarks are intensity-only). Present so callers
        can pass ``(arr, spacing)`` from ``load_canonical_volume`` uniformly.
    clip_hi : float
        High clip cap after calibration (default ``CLIP_HI_HU`` = +3100 HU).
    return_params : bool
        If True, return ``(calibrated_arr, HUCalibration)`` for logging.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, HUCalibration]
    """
    cal = fit_calibration(arr, clip_hi=clip_hi)
    out = apply_calibration(arr, cal)
    if return_params:
        return out, cal
    return out


# ── RGB windows (Task 2) ─────────────────────────────────────────────────────────
# Three pathology-derived (window_level, window_width) pairs on the CALIBRATED HU
# scale (air = AIR_REF = -1000, soft = SOFT_REF = 0, clip ceiling = CLIP_HI_HU =
# +3100). Each pair maps to one channel of an RGB axial frame (R, G, B). Windows
# were empirically validated against calibrated volumes across all four families
# (A/P/S/F).
#
#   R = soft-tissue / lucent-lesion window     L=150   W=1400  band [-550, +850]
#       Reveals: soft tissue, maxillary-sinus mucosa/air, cysts & periapical/
#       radicular lucencies, edentulous ridge soft profile, early cancellous bone.
#   G = dentin / periodontal / IAN window      L=1100  W=2200  band [   0, +2200]
#       Reveals (workhorse channel): the soft→cancellous→dentin→cortical gradient —
#       root morphology, periodontal/alveolar bone level & resorption, the cortical
#       outline of the mandibular (IAN) canal within cancellous bone, endodontic fill.
#   B = cortical / enamel / restoration window L=2200  W=2600  band [+900, +3500]
#       Reveals: dense cortical bone, enamel, prosthetic crowns / implants / metal
#       restorations (saturating toward the +3100 clip), osteosclerosis.
RGB_WINDOWS: list[tuple[float, float]] = [
    (150.0, 1400.0),   # R — soft tissue / lucent lesion
    (1100.0, 2200.0),  # G — dentin / periodontal / IAN canal
    (2200.0, 2600.0),  # B — cortical bone / enamel / restorations
]


# ── Tissue BANDS for cross-sections (not projections) ───────────────────────────
# RGB_WINDOWS above are *cumulative*: each is a monotone ramp, and enamel (~2600 HU) is above
# the ceiling of all three, so in a thin slice a tooth renders (255, 255, 166) — white-yellow.
# That is fine for a projection, where R and G are ray *means* and stay mid-range while B takes
# the ray max, so a tooth still comes out blue. It is wrong for a cross-section, where every
# channel sees the same voxel: measured on a buccolingual frame, the median voxel is +42 HU and
# 75% of the frame is under +210 HU, so R alone carries the image and the video reads red.
#
# These are band-*pass* instead — a tissue class per channel, with a soft rolloff so the
# boundaries are not hard edges. Measured tissue -> colour on F038:
#
#     soft tissue ~0 HU    (255,   0,   0)      cancellous ~600      (  0, 255,   0)
#     cortical ~1500       (  0, 255,   0)      enamel ~2600         (  0,   0, 163)
#     metal ~3100          (  0,   0, 255)
#
# The top channel is a RAMP, not a band: a band's upper rolloff made metal (213) darker than
# enamel (255), which inverts the density ordering the whole point is to preserve.
TISSUE_BANDS: list[tuple[float, float]] = [
    (-500.0, 500.0),     # R — air/fat/muscle/fluid: soft tissue and lucent lesions
    (300.0, 1900.0),     # G — cancellous through cortical bone, the IAN canal wall
    (1700.0, 3100.0),    # B — ramp (no upper rolloff): enamel -> restoration -> metal
]
# ⚠ The rolloff is what separates a *radiograph* from a *segmentation map*. At 250 HU the bands
# are effectively hard class labels: teeth are flat blue, bone flat green, and the trabecular
# texture, the PDL space and the dentin gradient — all of which the grayscale video shows plainly
# — are quantised away. A wide rolloff keeps a gradient inside each tissue class, so the channel
# says "how bony" and not just "bone".
BAND_SOFT_HU = 700.0     # rolloff width at a band edge


def to_tissue_rgb(cal_hu: np.ndarray,
                  bands: list[tuple[float, float]] = TISSUE_BANDS) -> np.ndarray:
    """Calibrated HU -> uint8 RGB by tissue class. Last channel is a ramp; the others band-pass."""
    out = []
    for k, (lo, hi) in enumerate(bands):
        up = np.clip((cal_hu - lo) / BAND_SOFT_HU, 0.0, 1.0)
        if k == len(bands) - 1:
            c = np.clip((cal_hu - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            c = np.minimum(up, np.clip((hi - cal_hu) / BAND_SOFT_HU, 0.0, 1.0))
        out.append((c * 255.0).astype(np.uint8))
    return np.stack(out, axis=-1)


def window_to_uint8(cal_hu_2d: np.ndarray, level: float, width: float) -> np.ndarray:
    """Apply one (level, width) window to a calibrated-HU 2D slice -> uint8 [0, 255].

    ``lo = level - width/2``, ``hi = level + width/2``; values are clipped to
    ``[lo, hi]`` and linearly mapped to ``[0, 255]``.
    """
    lo = level - width / 2.0
    hi = level + width / 2.0
    a = np.clip(cal_hu_2d.astype(np.float32), lo, hi)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def to_rgb_windowed(
    cal_hu_2d: np.ndarray,
    windows: list[tuple[float, float]] = RGB_WINDOWS,
) -> np.ndarray:
    """Render a calibrated-HU 2D slice to an ``uint8`` ``HxWx3`` RGB image.

    Each of the three ``(level, width)`` pairs in ``windows`` (R, G, B) is applied
    to the same calibrated slice via :func:`window_to_uint8` and stacked into the
    corresponding colour channel. Input must already be on the calibrated HU scale
    (e.g. from :func:`calibrate_to_hu`). Task 3 calls this once per axial slice to
    build the Qwen video frames.

    Parameters
    ----------
    cal_hu_2d : np.ndarray
        A 2-D calibrated-HU slice (float; any HU range, typically [-1000, +3100]).
    windows : list of (level, width), optional
        Exactly three windows for the R, G, B channels (default ``RGB_WINDOWS``).

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(H, W, 3)``.
    """
    if len(windows) != 3:
        raise ValueError(f"expected 3 (level, width) windows for RGB, got {len(windows)}")
    if cal_hu_2d.ndim != 2:
        raise ValueError(f"expected a 2D slice, got shape {cal_hu_2d.shape}")
    chans = [window_to_uint8(cal_hu_2d, lvl, wid) for (lvl, wid) in windows]
    return np.stack(chans, axis=-1)
