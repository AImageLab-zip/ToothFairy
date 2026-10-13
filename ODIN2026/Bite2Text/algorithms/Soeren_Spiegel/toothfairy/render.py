"""Renderers for the two ToothFairy4 views.

The clinical reading model maps 1:1 onto two rendered views, both
defined relative to the detected dental arch (arch.detect_arch_curve_dentition):

  - **panoramic (2D)** — the global map: FDI inventory, gross pathology, bone level, IAN
    course. Fills the HEADLINE bone-MIP image slot. Rendered by ``drr.pseudo_opg``, which
    simulates the radiograph rather than reformatting a slab; ``panoramic_curved`` below is the
    superseded curved-MPR version, kept for the exploration probes that A/B against it.
  - **cross-section stack (3D)** — buccolingual planes perpendicular to the arch at a fixed
    mm budget, fed as a Qwen video. The depth detail: bone width/height, IAN proximity, root
    morphology. Fills the HEADLINE soft-tissue video slot. → ``arch_cross_sections``.
  - **stills** — a coronal MPR through the maxillary sinuses + L/R sagittal MPR through the
    condyles. → ``sinus_coronal_mpr`` / ``condyle_sagittal_mpr``.

Robustness: windowing is per-volume percentile (CBCT has no HU); z-extent and
superior truncation are exposed so the model says "not included" rather than hallucinating.
Fallbacks: no trough passes ``arch.trough_quality`` → straight thick-slab coronal MIP ("flat
panoramic") + axial cross-sections; total render fail → a never-empty single-slice template.
A bad trough is worse than none, so the geometry is gated on a *measured* score rather than on
which detector happened to return something.

Input ``arr`` MUST be the canonical (L, A, S) isotropic grid from
volume_io.load_canonical_volume (x=left, y=anterior, z=superior; voxel distance = mm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage

from .arch import (
    ARCH_FAST_PATH_COVERAGE, ARCH_SWITCH_MARGIN, ARCH_Z_HI, ARCH_Z_LO, ArchCurve, detect_arch_curve,
    detect_arch_curve_dentition, detect_arch_curve_polar, trough_quality,
)
from .config import FRAME_TILE, FOV_LIMITED_MM, N_CROSS_SECTIONS
from .volume_io import (
    percentile_window, to_square_rgb, to_wide_rgb, window_to_uint8, z_extent_mm,
)

# Curved-MPR / cross-section geometry (mm; the canonical grid is isotropic so mm≈vox·spacing).
PANO_DEPTH_MM   = 12.0      # thick-slab half-depth along the buccolingual normal for the unroll
CROSS_STEP_MM   = 5.0       # arc spacing between buccolingual cross-sections
EMPTY_FRAME_FG_MIN = 0.20   # a frame with less foreground than this is outside the acquired FOV
STRETCH_PCT = (1.0, 99.5)   # per-channel stretch for the HU-windowed video (mirrors views._stack_u8)
CROSS_HALF_W_MM = 16.0      # buccolingual half-width of each cross-section
PANO_HEIGHT     = FRAME_TILE     # panoramic image height (px) for a full OPG band; width follows arc
PANO_MAX_WIDTH  = 1568           # cap width (multiple of 28) so image tokens stay bounded
# Crop the unroll to a clinical OPG band around the occlusal plane (occlusal ± this), so the
# dentition fills the frame at a *fixed* mm/px on every family — a tall full-head volume no
# longer packs the teeth into a sliver above the neck/below the orbits. Full Z if not localised.
PANO_HALF_HEIGHT_MM = 35.0


def _occlusal_zrange(occlusal_z: int | None, nz: int,
                     spacing: tuple[float, float, float]) -> tuple[int, int]:
    """Voxel [z0, z1) OPG band: occlusal ± PANO_HALF_HEIGHT_MM, clamped; full Z if the
    occlusal plane was not localised."""
    if occlusal_z is None:
        return 0, nz
    half = int(round(PANO_HALF_HEIGHT_MM / float(spacing[2])))
    return max(occlusal_z - half, 0), min(occlusal_z + half, nz)


def _dental_zrange(curve: ArchCurve, nz: int) -> tuple[int, int]:
    """OPG band for an ArchCurve (spline path)."""
    return _occlusal_zrange(curve.occlusal_z, nz, curve.spacing)


@dataclass
class RenderedCase:
    """What a case renders to. ``frames`` = the cross-section video (soft-tissue slot);
    ``stills`` = labeled images (bone-MIP slot: panoramic + sinus/condyle MPR).

    ``regime`` records which detector produced the geometry ('dentition' | 'pseudopano' |
    'arch' | 'flat' | 'template'). It is **not** a quality signal and must not be read as one —
    that mistake is why 41 of 43 broken val_ood cases reported success. ``arch_coverage`` is
    the quality signal: the fraction of the dentition the panoramic actually images, and a low
    value (not the regime) is what drives the flat fallback.

    ``z_mm`` + ``superior_truncated`` thread the FOV note so the model declines unseen anatomy
    rather than hallucinating."""
    frames: list[Image.Image]
    stills: list[tuple[str, Image.Image]] = field(default_factory=list)
    z_mm: float = 0.0
    regime: str = "arch"
    superior_truncated: bool = False
    arch_coverage: float | None = None


# ── Sampling helper ─────────────────────────────────────────────────────────────

def _sample(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
            cval: float) -> np.ndarray:
    """Trilinear sample arr at (xs, ys, zs) (same broadcast shape) -> that shape."""
    shape = np.broadcast_shapes(xs.shape, ys.shape, zs.shape)
    X = np.broadcast_to(xs, shape).ravel()
    Y = np.broadcast_to(ys, shape).ravel()
    Z = np.broadcast_to(zs, shape).ravel()
    vals = ndimage.map_coordinates(arr, np.stack([X, Y, Z]), order=1,
                                   mode="constant", cval=cval)
    return vals.reshape(shape)


# ── Panoramic (2D) — curved-MPR unroll along the arch ───────────────────────────

def panoramic_curved(
    arr: np.ndarray,
    curve: ArchCurve,
    lo: float,
    hi: float,
    depth_mm: float = PANO_DEPTH_MM,
    height: int = PANO_HEIGHT,
    reduce: str = "mean",
) -> Image.Image:
    """Thick-slab curved-MPR: for each arc position and each Z in the OPG band, project the
    volume over the buccolingual normal ±depth. ``reduce='mean'`` is a ray-sum (the classic
    semi-transparent pseudo-panoramic radiograph — roots/canals/sinuses show through);
    ``reduce='max'`` is a bone MIP. Output is [Z (superior up) × arc length], cropped to
    occlusal ± PANO_HALF_HEIGHT_MM at a fixed mm/px so scale is comparable across families.

    **Superseded by ``drr.pseudo_opg`` as the shipped panoramic** and kept only for the
    exploration probes that A/B against it. Two reasons not to revive it as-is: the ``lo``/``hi``
    percentile window is the content-dependent tone map that made contrast swing case to case,
    and the occlusal ± 35 mm crop is anchored on ``arch._occlusal_z``, which is biased ~23 mm
    inferior on val_ood and therefore cuts the maxilla."""
    nz = arr.shape[2]
    z0, z1 = _dental_zrange(curve, nz)
    cval = float(arr.min())
    depth_vox = depth_mm / curve.spacing[0]
    d = np.linspace(-depth_vox, depth_vox, max(int(2 * depth_vox) + 1, 3))
    P, N = curve.points, curve.normals                     # (M,2)
    xy = P[:, None, :] + d[None, :, None] * N[:, None, :]  # (M, nd, 2)
    xs = xy[:, :, 0][:, :, None]                           # (M, nd, 1)
    ys = xy[:, :, 1][:, :, None]
    zs = np.arange(z0, z1, dtype=np.float32)[None, None, :]  # (1, 1, nz_band)
    vals = _sample(arr, xs, ys, zs, cval)                  # (M, nd, nz_band)
    reduced = vals.mean(axis=1) if reduce == "mean" else vals.max(axis=1)
    pano = reduced.T[::-1, :]                              # -> (nz_band, M), superior up
    mm_per_px = (2.0 * PANO_HALF_HEIGHT_MM) / height       # fixed vertical scale
    out_h = max(28, int(round((z1 - z0) * float(curve.spacing[2]) / mm_per_px)))
    return to_wide_rgb(window_to_uint8(pano, lo, hi), height=out_h, max_width=PANO_MAX_WIDTH)


# ── Panoramic (2D) — star / radial unwrap from a focal point ────────────────────

STAR_R_MIN_MM = 3.0        # start radius (skip the focal neighbourhood)
STAR_R_MAX_MM = 78.0       # MIP out to here along each ray (arch wall + a margin)
STAR_N_COLS   = 720        # angular columns (one per ~0.25° over the arch span)
STAR_MARGIN_DEG = 6.0      # widen the detected angular span slightly at each end


def panoramic_star(
    arr: np.ndarray,
    focal: tuple[float, float],
    theta_min: float,
    theta_max: float,
    lo: float,
    hi: float,
    z0: int,
    z1: int,
    spacing: tuple[float, float, float] = (0.3, 0.3, 0.3),
    r_min_mm: float = STAR_R_MIN_MM,
    r_max_mm: float = STAR_R_MAX_MM,
    n_cols: int = STAR_N_COLS,
    height: int = PANO_HEIGHT,
    flip: bool = True,
) -> Image.Image:
    """Star/radial panoramic: for each angle θ (column) from ``focal`` and each Z in the OPG
    band, MIP the volume along the ray radius r. Output is [Z (superior up) × θ]. Needs only
    a focal point + angular span (no spline), so it survives cases where the arch spline fails.
    ``flip`` mirrors the angular axis to match the curved-MPR left→right reading order."""
    fx, fy = focal
    cval = float(arr.min())
    rs = np.arange(r_min_mm / spacing[0], r_max_mm / spacing[0], 1.0)       # (nr,)
    margin = math.radians(STAR_MARGIN_DEG)
    thetas = np.linspace(theta_min - margin, theta_max + margin, n_cols)    # (n_cols,)
    zs = np.arange(z0, z1, dtype=np.float32)                                # (nzb,)
    nzb = int(zs.shape[0])
    pano = np.empty((n_cols, nzb), dtype=np.float32)
    for c0 in range(0, n_cols, 48):                                        # chunk to bound memory
        th = thetas[c0:c0 + 48]
        xs = (fx + rs[None, :] * np.cos(th)[:, None])[:, :, None]           # (C, nr, 1)
        ys = (fy + rs[None, :] * np.sin(th)[:, None])[:, :, None]
        vals = _sample(arr, xs, ys, zs[None, None, :], cval)               # (C, nr, nzb)
        pano[c0:c0 + 48] = vals.max(axis=1)                                # MIP over r
    img = pano.T[::-1, :]                                                   # (nzb, n_cols), sup up
    if flip:
        img = img[:, ::-1]
    mm_per_px = (2.0 * PANO_HALF_HEIGHT_MM) / height                        # match curved scale
    out_h = max(28, int(round((z1 - z0) * float(spacing[2]) / mm_per_px)))
    return to_wide_rgb(window_to_uint8(img, lo, hi), height=out_h, max_width=PANO_MAX_WIDTH)


# ── Cross-section stack (3D) — star/radial buccolingual planes ──────────────────

def cross_sections_star(
    arr: np.ndarray,
    pa,                       # arch.PolarArch
    lo: float,
    hi: float,
    z0: int,
    z1: int,
    spacing: tuple[float, float, float] = (0.3, 0.3, 0.3),
    half_width_mm: float = CROSS_HALF_W_MM,
    tile: int = FRAME_TILE,
    n_max: int = N_CROSS_SECTIONS,
) -> list[Image.Image]:
    """Buccolingual cross-sections from the *star* geometry: at evenly-spaced angles θ, the
    ray direction (cosθ, sinθ) is (approximately) the buccolingual normal, so the (radial ×
    Z) plane centered on the arch radius r(θ) is a buccolingual section — same basis as
    ``panoramic_star``, so no dependency on a smooth spline. Superior up, even-padded."""
    fx, fy = pa.focal
    cval = float(arr.min())
    hw_vox = half_width_mm / spacing[0]
    w = np.linspace(-hw_vox, hw_vox, max(int(2 * hw_vox) + 1, 3))           # (n_w,)
    thetas = np.linspace(float(pa.thetas[0]), float(pa.thetas[-1]), min(n_max, len(pa.thetas)))
    r_at = np.interp(thetas, pa.thetas, pa.r_theta)                         # arch radius per θ
    zs = np.arange(z0, z1, dtype=np.float32)[None, :]                       # (1, nzb)
    frames: list[Image.Image] = []
    for th, r0 in zip(thetas, r_at):
        rr = r0 + w                                                        # (n_w,) radii across the wall
        xs = (fx + rr * math.cos(th))[:, None]                             # (n_w, 1)
        ys = (fy + rr * math.sin(th))[:, None]
        cs = _sample(arr, xs, ys, zs, cval).T[::-1, :]                     # (nzb, n_w), superior up
        frames.append(to_square_rgb(window_to_uint8(cs, lo, hi), tile))
    if len(frames) % 2 == 1:
        frames.append(frames[-1].copy())
    return frames


def _rgb_wide_u8(rgb_u8: np.ndarray, height: int, max_width: int | None = None) -> Image.Image:
    """`volume_io.to_wide_rgb` for an already-3-channel frame (the HU-windowed video path)."""
    h, w = rgb_u8.shape[:2]
    new_h = max(28, int(round(height / 28)) * 28)
    new_w = max(28, int(round(w * new_h / h)))
    if max_width:
        new_w = min(new_w, max_width)
    new_w = max(28, int(round(new_w / 28)) * 28)
    return Image.fromarray(rgb_u8, mode="RGB").resize((new_w, new_h), Image.LANCZOS)


# ── Cross-section stack (3D) — buccolingual planes ⊥ arch (spline) ──────────────

def arch_cross_sections(
    arr: np.ndarray,
    curve: ArchCurve,
    lo: float,
    hi: float,
    step_mm: float = CROSS_STEP_MM,
    half_width_mm: float = CROSS_HALF_W_MM,
    tile: int = FRAME_TILE,
    n_max: int = N_CROSS_SECTIONS,
    slab_mm: float = 0.0,
    arr_hu: np.ndarray | None = None,
    drop_empty: bool = False,
) -> list[Image.Image]:
    """Buccolingual cross-sections perpendicular to the arch every ``step_mm``, as Qwen video
    frames (superior up), count padded to even for the pair-merge.

Three fixes so the video and the panoramic image the SAME anatomy:
      1. walk the **extended** trough (``drr.flared_extend``, the 170 mm arc the panoramic
         renders) rather than ``curve.length_mm`` (the fitted ~83 mm), so video frame k and
         panoramic column k are the same arch position;
      2. use ``drr.jaw_zband`` (occlusal +42/-45) — the panoramic's band — not ``_dental_zrange``
         (occlusal +/-35);
      3. aspect-preserving frames (``to_wide_rgb``, ~448x168) instead of ``to_square_rgb``,
         which padded ~54% of every frame black. Frame size is uniform per case, and
         ``dataset.video_content`` reads it back so the processor does not re-stretch to square.

    Three v2 levers, all defaulting OFF (the container calls this path):

    ``slab_mm`` — **the big one.** Each frame is otherwise a single trilinear plane: zero
    thickness against a ``step_mm`` stride, so the arc duty cycle is voxel/step = **5.8%** and
    94% of the arch is never imaged at all. Set ``slab_mm=step_mm`` and each frame
    becomes a maximum over its own share of the arc, so the video covers the arch continuously.
    A max, not a mean: these are cross-sections, and the point is to keep the 1 mm gutta-percha
    line that a mean over 5 mm would wash out. Zero token cost.

    ``arr_hu`` — a calibrated-HU volume. Renders through ``hu.RGB_WINDOWS`` instead of the
    per-volume percentile grey, which is what every frame is today (grayscale on 62/62 val
    cases, and half the token budget). Zero token cost.

    ``drop_empty`` — the flared trough ends run past the acquired volume on short-FOV scans, so
    those frames are pure fill value; 6/62 val cases carry them, up to 20.6% of A005's frames.
    """
    from .drr import jaw_zband, flared_extend
    from .hu import to_tissue_rgb
    nz = arr.shape[2]
    z0, z1 = jaw_zband(curve.occlusal_z, nz, curve.spacing)
    cval = float(arr.min())
    hw_vox = half_width_mm / curve.spacing[0]
    w = np.linspace(-hw_vox, hw_vox, max(int(2 * hw_vox) + 1, 3))
    # Extended geometry (same as the panoramic) + its own arc-length parametrisation.
    P, N = flared_extend(curve)
    sp = float(curve.spacing[0])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]) * sp  # (M,) mm
    length_mm = float(arc[-1])
    positions = np.arange(0.0, length_mm, step_mm)
    if len(positions) > n_max:                              # keep the mm step; cap coverage evenly
        positions = np.linspace(0.0, length_mm, n_max)
    idxs = [int(np.argmin(np.abs(arc - pm))) for pm in positions]
    zs = np.arange(z0, z1, dtype=np.float32)[None, :]       # (1, nz_band)
    src = arr_hu if arr_hu is not None else arr
    fill = -1000.0 if arr_hu is not None else cval
    raw: list[np.ndarray] = []          # HU-windowed channels, before the per-CASE stretch
    # Offsets ALONG the arc, so the frame integrates its own slice of the stride, not across it.
    offs = (np.linspace(-slab_mm / 2.0, slab_mm / 2.0, max(int(slab_mm / sp) + 1, 3)) / sp
            if slab_mm > 0 else np.zeros(1, np.float32))
    frames: list[Image.Image] = []
    for i in idxs:
        p, nrm = P[i], N[i]
        tan = np.array([-nrm[1], nrm[0]], dtype=np.float32)
        xy = p[None, :] + w[:, None] * nrm[None, :]         # (n_w, 2)
        stack = [_sample(src, (xy[:, 0] + o * tan[0])[:, None],
                         (xy[:, 1] + o * tan[1])[:, None], zs, fill) for o in offs]
        cs = (stack[0] if len(stack) == 1 else np.max(stack, axis=0)).T[::-1, :]
        if arr_hu is not None:
            raw.append(to_tissue_rgb(cs).astype(np.float32) / 255.0)
        else:
            frames.append(to_wide_rgb(window_to_uint8(cs, lo, hi), tile))
    if arr_hu is not None:
        # ⚠ No percentile stretch here, deliberately. `hu.to_tissue_rgb` is band-pass: a channel
        # value already means "this voxel is that tissue", and stretching per channel would
        # re-normalise that meaning per case — the content-dependent tone map the HU calibration
        # exists to avoid. The stills stretch because their channels are ray *reductions*, which
        # have no absolute meaning left to protect.
        frames = [_rgb_wide_u8((f * 255.0).astype(np.uint8), tile) for f in raw]
    if drop_empty:
        keep = [f for f in frames
                if float((np.asarray(f.convert("L")) > 8).mean()) > EMPTY_FRAME_FG_MIN]
        if len(keep) >= 4:                                  # never empty the video
            frames = keep
    if len(frames) % 2 == 1:                                # even count for temporal-pair merge
        frames.append(frames[-1].copy())
    return frames


# ── Targeted orthogonal MPR stills (sinus / condyles) ───────────────────────────

def sinus_coronal_mpr(arr: np.ndarray, lo: float, hi: float, tile: int = FRAME_TILE) -> Image.Image:
    """Coronal thick-mean MPR through the posterior maxilla (maxillary sinuses sit above the
    posterior teeth). Plane = (L × S) meaned over an anterior-posterior slab; superior up."""
    ny = arr.shape[1]
    y0, y1 = int(0.38 * ny), int(0.60 * ny)
    coronal = arr[:, y0:y1, :].mean(axis=1).T[::-1, :]      # (nz, nx), superior up
    return to_square_rgb(window_to_uint8(coronal, lo, hi), tile)


def condyle_sagittal_mpr(arr: np.ndarray, lo: float, hi: float, side: str,
                         tile: int = FRAME_TILE) -> Image.Image:
    """Sagittal thick-mean MPR through a condyle. Plane = (A × S) meaned over a thin L-slab at
    the lateral extreme (x high = patient left in LAS). Empty when the FOV crops the condyle
    (P/S) — which is exactly the 'not included' signal, handled by the truncation note."""
    nx = arr.shape[0]
    xc = int((0.83 if side == "l" else 0.17) * nx)
    half = max(int(0.02 * nx), 4)
    sag = arr[max(xc - half, 0): xc + half, :, :].mean(axis=0).T[::-1, :]   # (nz, ny), superior up
    return to_square_rgb(window_to_uint8(sag, lo, hi), tile)


# ── Superior-truncation detection ──────────────────────────────────────────

TRUNC_NEAR_ARCH_MM = 20.0   # column the occupancy is measured in
TRUNC_RATIO = 0.25          # top-slice bone / occlusal-plane bone above which anatomy continues


def detect_superior_truncation(
    arr: np.ndarray,
    z_mm: float,
    bone_thr: float,
    cloud: np.ndarray | None = None,
    thresholds: tuple[float, float, float] | None = None,
    occlusal_z: int | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> bool:
    """True if the upper maxilla / condyles are cut off by the scan's Z-FOV.

    The measured test (``toothfairy/probe_fov.py``): at the volume's **last slice**, is the
    scanner still cutting through jaw-region bone? Occupancy is normalised by the same
    measurement at the occlusal plane, so it does not depend on a scanner's absolute values,
    and it is restricted to within 20 mm of the dental arch so the cranial vault and the spine
    cannot answer for the maxilla. Over the 128 cached volumes: 111 truncated, 10 tight,
    **7 complete**.

    The previous heuristic — ``z_mm < FOV_LIMITED_MM`` or almost no bone in the top 10 % — is
    kept only as the fallback when no dentition cloud is available. It cannot separate the
    seven complete cases from the rest: the median volume is 78.8 mm tall against a 100 mm
    threshold, so its first clause fires on essentially every case."""
    if cloud is not None and thresholds is not None and occlusal_z is not None:
        nz = arr.shape[2]
        sp = float(spacing[0]) if spacing is not None else 1.0
        near = ndimage.distance_transform_edt(~(cloud > 0)) * sp <= TRUNC_NEAR_ARCH_MM
        if near.any():
            occ = (arr > thresholds[1])[near].reshape(-1, nz).mean(axis=0)
            ref = float(occ[max(occlusal_z - 3, 0):occlusal_z + 4].mean())
            if ref > 1e-6:
                return bool(float(occ[-3:].mean()) / ref > TRUNC_RATIO)
    if z_mm < FOV_LIMITED_MM:
        return True
    nz = arr.shape[2]
    top = arr[:, :, int(0.90 * nz):]
    return float((top > bone_thr).mean()) < 0.004


# ── Fallback renderers (arch-detect fail) ───────────────────────────────────────

def flat_panoramic(arr: np.ndarray, lo: float, hi: float, height: int = PANO_HEIGHT) -> Image.Image:
    """Straight thick-slab coronal MIP of the arch band — the fallback when arch detection
    fails. Projects along A over the arch band -> an (L × S) frontal view. Superior up."""
    nz = arr.shape[2]
    z0, z1 = int(ARCH_Z_LO * nz), int(ARCH_Z_HI * nz)
    frontal = arr[:, :, z0:z1 + 1].max(axis=1).T[::-1, :]   # MIP along A -> (S, L) -> superior up
    return to_wide_rgb(window_to_uint8(frontal, lo, hi), height=height, max_width=PANO_MAX_WIDTH)


def axial_cross_sections(arr: np.ndarray, lo: float, hi: float, n: int = N_CROSS_SECTIONS,
                         tile: int = FRAME_TILE) -> list[Image.Image]:
    """Dense axial arch-band slices — the cross-section fallback (== v0 baseline slices), used
    when there is no arch curve to take buccolingual sections along. Even-padded."""
    nz = arr.shape[2]
    z_lo, z_hi = max(int(ARCH_Z_LO * nz), 0), min(int(ARCH_Z_HI * nz), nz - 1)
    idx = np.round(np.linspace(z_lo, z_hi, max(n, 2))).astype(int)
    frames = [to_square_rgb(window_to_uint8(arr[:, :, z], lo, hi), tile) for z in idx]
    if len(frames) % 2 == 1:
        frames.append(frames[-1].copy())
    return frames


# ── Compose the two views for a case (with graceful degradation) ────────────────

def render_case(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    n_cross: int = N_CROSS_SECTIONS,
    tile: int = FRAME_TILE,
    with_mpr: bool = True,
    z_jitter_vox: int = 0,
    video_slab_mm: float = 0.0,
    video_rgb: bool = False,
    video_drop_empty: bool = False,
    with_video: bool = True,
) -> RenderedCase:
    """Render a canonical isotropic volume -> RenderedCase.

    Primary geometry is the **dentition-driven** trough (arch.detect_arch_curve_dentition):
    the arch is derived from the densest tissue in the scan rather than from a flattened
    projection of everything, which is what breaks on full-FOV scans (val_ood 21/63 -> 60/63).
    The panoramic still is the pseudo-OPG (drr.pseudo_opg), replacing the curved-MPR reformat

    **Selection, not a fallback chain.** Each candidate trough is scored against the dentition
    it is supposed to image (arch.trough_quality) and the better one wins; only when none
    passes does the flat panoramic take over. The dentition trough is taken outright when its
    coverage clears ARCH_FAST_PATH_COVERAGE, so the much slower polar detector runs only on the
    cases where it might actually help — it exists here because it recovered a handful of
    val_id cases the dentition path regressed. Never returns empty."""
    # Deferred: drr imports _sample/_dental_zrange/PANO_MAX_WIDTH from this module, so importing
    # it at module scope would be circular.
    from .drr import DRR_DEPTH_MM, pseudo_opg

    lo, hi = percentile_window(arr)
    bone_thr = lo + 0.5 * (hi - lo)
    z_mm = z_extent_mm(arr, spacing)

    curve = None
    cloud = None
    regime = "flat"
    quality: dict | None = None
    try:
        curve, cloud = detect_arch_curve_dentition(arr, spacing)
    except Exception:  # noqa: BLE001 — detection must never crash the pipeline
        curve, cloud = None, None
    if curve is not None and cloud is not None:
        quality = trough_quality(curve, cloud, DRR_DEPTH_MM)
        regime = "dentition"

    # The fast path must require `ok`, not just coverage: A019 and P497 score 0.99/0.93 on a
    # cloud of two restorations, so a coverage-only shortcut skips the comparison exactly where
    # the alternative detector is most likely to help.
    fast = quality is not None and quality["ok"] and quality["coverage"] >= ARCH_FAST_PATH_COVERAGE
    if cloud is not None and not fast:
        try:
            alt = detect_arch_curve_polar(arr, spacing)
        except Exception:  # noqa: BLE001
            alt = None
        if alt is not None:
            alt_q = trough_quality(alt, cloud, DRR_DEPTH_MM)
            better = alt_q["ok"] and (quality is None or not quality["ok"]
                                      or alt_q["coverage"] > quality["coverage"] + ARCH_SWITCH_MARGIN)
            if better:
                curve, quality, regime = alt, alt_q, "pseudopano"

    if quality is not None and not quality["ok"]:
        curve, regime = None, "flat"                       # a bad trough is worse than none
    if curve is None and cloud is None:                    # detection crashed -> legacy spline
        try:
            curve = detect_arch_curve(arr, spacing)
        except Exception:  # noqa: BLE001
            curve = None
        if curve is not None:
            regime = "arch"

    # occlusal-plane jitter (augmentation). Shift the detected band by a few voxels so the
    # video/panoramic image a slightly different Z window each epoch — orthogonal to laterality.
    if z_jitter_vox and curve is not None and curve.occlusal_z is not None:
        nzc = arr.shape[2]
        curve.occlusal_z = int(min(max(curve.occlusal_z + z_jitter_vox, 0), nzc - 1))

    truncated = detect_superior_truncation(
        arr, z_mm, bone_thr, cloud=cloud,
        thresholds=curve.thresholds if curve is not None else None,
        occlusal_z=curve.occlusal_z if curve is not None else None, spacing=spacing)
    coverage = quality["coverage"] if quality is not None else None

    stills: list[tuple[str, Image.Image]] = []
    try:
        if curve is not None:
            panoramic = pseudo_opg(arr, curve, height=tile)
            # The calibrated volume is recomputed here rather than threaded from five_views:
            # one histogram pass (~0.2 s), against keeping two call sites in sync.
            hu = None
            if video_rgb:
                from .hu import calibrate_to_hu
                hu = calibrate_to_hu(arr)
            # with_video=False still runs the arch fit — z_mm, regime and the truncation
            # flag all come from it and the prompt needs them — it only skips the frames.
            frames = arch_cross_sections(arr, curve, lo, hi, tile=tile, n_max=n_cross,
                                         slab_mm=video_slab_mm, arr_hu=hu,
                                         drop_empty=video_drop_empty) if with_video else []
        else:                                              # regime already "flat"
            panoramic = flat_panoramic(arr, lo, hi, height=tile)
            frames = axial_cross_sections(arr, lo, hi, n=n_cross,
                                          tile=tile) if with_video else []
        stills.append(("panoramic", panoramic))
        if with_mpr:
            stills.append(("sinus_mpr", sinus_coronal_mpr(arr, lo, hi, tile)))
            stills.append(("condyle_r", condyle_sagittal_mpr(arr, lo, hi, "r", tile)))
            stills.append(("condyle_l", condyle_sagittal_mpr(arr, lo, hi, "l", tile)))
    except Exception:  # noqa: BLE001 — never-empty guarantee
        z = arr.shape[2] // 2
        template = to_square_rgb(window_to_uint8(arr[:, :, z], lo, hi), tile)
        return RenderedCase(frames=[template, template.copy()] if with_video else [],
                            stills=[("panoramic", template.copy())],
                            z_mm=z_mm, regime="template", superior_truncated=truncated)

    return RenderedCase(frames=frames, stills=stills, z_mm=z_mm, regime=regime,
                        superior_truncated=truncated, arch_coverage=coverage)
