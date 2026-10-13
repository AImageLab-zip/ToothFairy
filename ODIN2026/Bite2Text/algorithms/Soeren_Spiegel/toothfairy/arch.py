"""Dental arch-curve detection — the make-or-break geometry.

The panoramic (2D) and the buccolingual cross-sections (3D) are both defined relative to
the dental arch. This module recovers the arch centerline from the volume with a heuristic:

  1. thick-mean axial slab through the arch band (20-80% of Z, where the dentition sits);
  2. threshold bone/teeth (bright on CBCT), keep the largest connected component (drops
     isolated bright noise);
  3. per-column (left-right x) weighted-centroid of the anterior-posterior y -> points that
     trace the middle of the arch band;
  4. weighted degree-4 polynomial fit y = P(x) (the arch is single-valued in x over the
     dental region after LAS reorient: incisors anterior at center x, molars posterior at
     the x extremes -> a concave arc), resampled to equal arc-length with tangents+normals.

Input ``arr`` MUST be the canonical (L, A, S) isotropic grid from
volume_io.load_canonical_volume, so x=left, y=anterior, z=superior and voxel distance = mm.

Returns an ``ArchCurve`` or ``None`` when detection is untrustworthy (too little bone, a
degenerate fit) — the caller then falls back to the straight thick-slab "flat panoramic"
(render.flat_panoramic). A bad spline is worse than none, so the guards are deliberate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ARCH_Z_LO, ARCH_Z_HI = 0.20, 0.80     # dental-arch band along S for detection (matches v0 baseline)
# Keep the top ~12% of slab intensity as "bone/teeth". This self-adapts to field of view:
# on a full-head FOV (F family) it isolates the mandible+dentition and drops brain/soft
# tissue; on a jaw-only FOV (P/S) it isolates the densest teeth/cortical bone — both trace
# the arch. A lower percentile (e.g. 70) swallows soft tissue on full-FOV scans and collapses
# the fit to a horizontal line (verified on F001). Gated visually across all 4 families.
_BONE_PCT = 88                        # slab percentile above which is "bone/teeth"
_POLY_DEG = 4                         # arch fit degree
_MIN_BONE_VOX = 200                   # too little bone in the slab -> untrustworthy
_MIN_VALID_COLS = 15                  # too few columns spanned -> untrustworthy


@dataclass
class ArchCurve:
    """The detected arch centerline, sampled at equal arc length.

    points   : (M, 2) arch centerline in (x, y) voxel coords on the canonical grid
    normals  : (M, 2) unit buccolingual direction (perpendicular to the arch tangent)
    tangents : (M, 2) unit arch-tangent direction
    arc_mm   : (M,)   cumulative arc length in mm (isotropic grid, so vox*spacing)
    slab     : (nx, ny) the axial mean slab used for detection (for overlays/inspection)
    mask     : (nx, ny) the bone mask (largest CC) used for the fit
    midline_xy : a point on the head's mid-sagittal plane (``head_midline``). The parabola's
                 vertex sits on it, and the posterior extension is balanced about it, so the two
                 halves of the frame carry equal arc. The arch's *angle* is not stored because it
                 is not fitted: it is the canonical LAS grid's, i.e. the coronal view's.
    sym_score  : how symmetric the jaw-level bone actually was about that plane (reporting only).
    occlusal_z : the occlusal-plane Z index (crown-band peak), or None if not localised.
                 The renderers crop the panoramic/cross-section unroll to a band around it
                 so every family is framed at the same mm scale (else tall full-head volumes
                 pack the dentition into a sliver of the frame). None -> render full Z.
    thresholds : the per-volume cascaded-Otsu ladder (t_head, t_bone, t_tooth) the curve was
                 detected with, or None for the legacy detectors. ``drr.pseudo_opg`` needs it
                 to render on a tissue-anchored scale instead of a percentile window; when it
                 is None the renderer recomputes it.
    coverage   : fraction of the dentition the trough's integration slab images, set by
                 ``trough_quality``. This — not ``RenderedCase.regime`` — is the signal that a
                 panoramic is usable.
    """
    points: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    arc_mm: np.ndarray
    spacing: tuple[float, float, float]
    slab: np.ndarray
    mask: np.ndarray
    occlusal_z: int | None = None
    thresholds: tuple[float, float, float] | None = None
    coverage: float | None = None
    midline_xy: tuple[float, float] | None = None
    sym_score: float | None = None
    parabola: tuple[float, float, float] | None = None   # (v0 mm, c 1/mm, half-width mm)

    @property
    def length_mm(self) -> float:
        return float(self.arc_mm[-1])


def _occlusal_z(arr: np.ndarray, pts: np.ndarray, spacing: tuple[float, float, float]) -> int:
    """The occlusal-plane Z index: where the volume is brightest *along the fitted arch*.

    Sampling along the detected arch (not an independent brightest-voxel scan) ties the crop
    to whichever arch the fit actually found — so bright maxillary metalwork can't drag the
    plane off the mandibular arch it was detected on (verified on F001, whose implants pull a
    global enamel-peak scan ~25 mm too superior). The arch passes through teeth/alveolar bone,
    which are dense (bright); soft tissue / sinus above and below are not."""
    nx, ny, nz = arr.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, nx - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ny - 1)
    zprof = arr[xi, yi, :].mean(axis=0).astype(np.float32)             # (nz,) mean intensity / slice
    zprof = ndimage.gaussian_filter1d(zprof, max(1.0, 2.0 / float(spacing[2])))  # ~2 mm smooth
    # Search the interior only: FOV-edge slices carry reconstruction-rim brightness that can
    # spike the profile (verified on A003, a bright bottom slice) — the occlusal plane is never
    # in the outer ~8% of a scan.
    lo = int(0.08 * nz)
    return lo + int(np.argmax(zprof[lo:nz - lo]))


def detect_arch_curve(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    z_band: tuple[float, float] | None = None,
) -> ArchCurve | None:
    """Detect the dental arch centerline, or None if untrustworthy (-> flat fallback).

    ``z_band`` (Z-fraction) overrides the detection slab; default is the fixed 20-80% band,
    which reliably traces the arch across families. The renderers additionally crop to a band
    around ``occlusal_z`` (derived from the fitted arch) for consistent framing.
    """
    nx, ny, nz = arr.shape
    zb = z_band if z_band is not None else (ARCH_Z_LO, ARCH_Z_HI)
    z0, z1 = int(zb[0] * nz), int(zb[1] * nz)
    if z1 <= z0:
        return None
    slab = arr[:, :, z0:z1].mean(axis=2).astype(np.float32)     # (x=L, y=A)

    thr = float(np.percentile(slab, _BONE_PCT))
    mask = slab > thr
    if int(mask.sum()) < _MIN_BONE_VOX:
        return None

    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.float32), lab, index=np.arange(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))                    # largest connected component

    # Per-column (x) weighted centroid of y, weighted by bone count in the column.
    y_idx = np.arange(ny, dtype=np.float32)
    weights = mask.sum(axis=1).astype(np.float32)                # (nx,) bone count per column
    col_y = (mask * y_idx[None, :]).sum(axis=1) / np.maximum(weights, 1.0)
    valid = weights > (0.05 * weights.max())
    if int(valid.sum()) < _MIN_VALID_COLS:
        return None

    xv = np.arange(nx, dtype=np.float32)[valid]
    yv = col_y[valid]
    wv = weights[valid]
    try:
        coeff = np.polyfit(xv, yv, _POLY_DEG, w=wv)
    except Exception:  # noqa: BLE001 — singular fit -> untrustworthy
        return None
    poly = np.poly1d(coeff)

    # Dense sample over the spanned x-range, then re-parametrise by equal arc length so
    # cross-section spacing is uniform in mm along the arch (not in x).
    x_lo, x_hi = float(xv.min()), float(xv.max())
    xs = np.linspace(x_lo, x_hi, 1024)
    ys = poly(xs)
    d = np.diff(np.stack([xs, ys], axis=1), axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg)])                # arc length in voxels
    if arc[-1] < _MIN_VALID_COLS:
        return None

    m = max(int(arc[-1]), 32)                                    # ~1 sample / voxel along arch
    s_new = np.linspace(0.0, arc[-1], m)
    xr = np.interp(s_new, arc, xs)
    yr = np.interp(s_new, arc, ys)
    pts = np.stack([xr, yr], axis=1)

    t = np.gradient(pts, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn == 0] = 1.0
    tang = t / tn[:, None]
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)        # perpendicular = buccolingual

    return ArchCurve(
        points=pts, normals=normals, tangents=tang,
        arc_mm=s_new * float(spacing[0]),
        spacing=spacing, slab=slab, mask=mask,
        occlusal_z=_occlusal_z(arr, pts, spacing),
    )


# ── Star / radial arch detection ─────────────────
#
# The polynomial fit y = P(x) above assumes the arch is single-valued in x. It is not:
# at the posterior molars/ramus the arch curls back (two y per x), and on full-FOV scans
# soft tissue drags the fit flat. A ray cast *outward from a focal point inside the arch*
# crosses the dentition exactly once, so the arch becomes r(θ) — always single-valued.
# We recover r(θ) per angle (peak-bone radius of the nearest bone the ray hits), which is
# far more robust than the spline, then re-parametrise to equal arc length exactly like
# detect_arch_curve so the downstream renderers (panoramic_curved, arch_cross_sections)
# consume it unchanged.

_POLAR_R_MIN_MM = 3.0                  # ignore the immediate neighbourhood of the focal point
_POLAR_R_MAX_MM = 80.0                 # search out to here for the arch wall
_POLAR_R_STEP_MM = 0.5                 # radial sampling step
_POLAR_N_ANGLES = 720                  # full-circle angular resolution (0.5°) for r(θ)
_POLAR_SMOOTH_DEG = 4.0                # gaussian smoothing of r(θ), in degrees
_POLAR_MIN_ARC_DEG = 90.0             # need at least this angular span of arch to trust it


@dataclass
class PolarArch:
    """Intermediate polar description of the arch, before equal-arc resampling.

    focal      : (fx, fy) focal point in axial (x, y) voxel coords, inside the arch "U".
    thetas     : (K,) angles (rad, unwrapped, monotonic) of the bone-bearing arch span.
    r_theta    : (K,) arch radius (voxels) at each angle = peak-bone radius of the nearest
                 bone the ray from ``focal`` hits.
    points     : (K, 2) = focal + r_theta·(cosθ, sinθ) — the arch centerline in (x, y).
    slab, mask : the detection slab and bone mask (largest CC), as in ArchCurve.
    occlusal_z : occlusal-plane Z index (via _occlusal_z along ``points``), or None.
    """
    focal: tuple[float, float]
    thetas: np.ndarray
    r_theta: np.ndarray
    points: np.ndarray
    slab: np.ndarray
    mask: np.ndarray
    occlusal_z: int | None = None


def _bone_slab_mask(
    arr: np.ndarray, spacing: tuple[float, float, float], z_band: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Shared slab → threshold → largest-CC bone mask (the arch.py:107-118 pattern).

    Returns (slab, mask, thr) or None if there is too little bone / a degenerate band."""
    nz = arr.shape[2]
    zb = z_band if z_band is not None else (ARCH_Z_LO, ARCH_Z_HI)
    z0, z1 = int(zb[0] * nz), int(zb[1] * nz)
    if z1 <= z0:
        return None
    slab = arr[:, :, z0:z1].mean(axis=2).astype(np.float32)
    thr = float(np.percentile(slab, _BONE_PCT))
    mask = slab > thr
    if int(mask.sum()) < _MIN_BONE_VOX:
        return None
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.float32), lab, index=np.arange(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
    return slab, mask, thr


def focal_point(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    z_band: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray] | None:
    """Focal point for the star projection: intensity-weighted centroid of the arch bone
    mask. The arch is a ∩ (incisors anterior/central, molars posterior/lateral), so its
    centroid lands *inside* the U — the natural centre to cast rays from. Returns
    ((fx, fy), slab, mask) or None. Robust where detect_arch_curve returns None: needs no
    polynomial fit, only a centroid."""
    got = _bone_slab_mask(arr, spacing, z_band)
    if got is None:
        return None
    slab, mask, thr = got
    nx, ny = slab.shape
    w = np.where(mask, slab - thr, 0.0).astype(np.float32)   # weight by how bright over thr
    tot = float(w.sum())
    if tot <= 0.0:
        return None
    fx = float((w.sum(axis=1) * np.arange(nx, dtype=np.float32)).sum() / tot)
    fy = float((w.sum(axis=0) * np.arange(ny, dtype=np.float32)).sum() / tot)
    return (fx, fy), slab, mask


def _longest_circular_run(valid: np.ndarray) -> np.ndarray:
    """Indices of the longest contiguous run of True in a *circular* boolean array,
    in traversal order (handles wraparound at ±π)."""
    n = len(valid)
    if bool(valid.all()):
        return np.arange(n)
    if not bool(valid.any()):
        return np.array([], dtype=int)
    v2 = np.concatenate([valid, valid])
    best_len = best_start = cur_len = cur_start = 0
    for i in range(2 * n):
        if v2[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if best_len < cur_len <= n:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return (np.arange(best_start, best_start + best_len) % n).astype(int)


def polar_arch(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    z_band: tuple[float, float] | None = None,
) -> PolarArch | None:
    """Recover the arch as r(θ) by casting rays from the focal point (see module note).

    For each angle θ we sample the bone mask along the ray and take the *nearest* bone
    segment the ray hits; r(θ) is the intensity-weighted radius within that segment (the
    densest point of the arch wall). We keep the longest bone-bearing angular span, smooth
    r(θ), and lift to (x, y) points. Returns a PolarArch or None if untrustworthy."""
    res = focal_point(arr, spacing, z_band)
    if res is None:
        return None
    (fx, fy), slab, mask = res
    nx, ny = slab.shape
    maskf = mask.astype(np.float32)

    rs = np.arange(_POLAR_R_MIN_MM / spacing[0], _POLAR_R_MAX_MM / spacing[0],
                   max(_POLAR_R_STEP_MM / spacing[0], 0.5))
    thetas_full = np.linspace(-math.pi, math.pi, _POLAR_N_ANGLES, endpoint=False)
    r_at = np.full(_POLAR_N_ANGLES, np.nan, dtype=np.float32)

    for k, th in enumerate(thetas_full):
        xs = fx + rs * math.cos(th)
        ys = fy + rs * math.sin(th)
        inb = (xs >= 0) & (xs < nx - 1) & (ys >= 0) & (ys < ny - 1)
        if not inb.any():
            continue
        coords = np.stack([xs, ys])
        mvals = ndimage.map_coordinates(maskf, coords, order=0, mode="constant", cval=0.0) > 0.5
        mvals &= inb
        if not mvals.any():
            continue
        i0 = int(np.argmax(mvals))                       # first (nearest) masked sample
        end = i0
        while end + 1 < len(mvals) and mvals[end + 1]:   # extend the nearest contiguous run
            end += 1
        seg = slice(i0, end + 1)
        ivals = ndimage.map_coordinates(slab, coords, order=1, mode="constant",
                                        cval=float(slab.min()))
        w = np.maximum(ivals[seg], 0.0)
        wsum = float(w.sum())
        r_at[k] = float((rs[seg] * w).sum() / wsum) if wsum > 0 else float(rs[seg].mean())

    valid = ~np.isnan(r_at)
    run = _longest_circular_run(valid)
    if run.size < 4:
        return None
    thetas = np.unwrap(thetas_full[run])
    r_theta = r_at[run].astype(np.float32)
    if math.degrees(float(thetas[-1] - thetas[0])) < _POLAR_MIN_ARC_DEG:
        return None
    # Smooth r(θ) to denoise the per-ray peak-pick (sigma in samples from the angular step).
    sigma = max(_POLAR_SMOOTH_DEG / (360.0 / _POLAR_N_ANGLES), 0.5)
    r_theta = ndimage.gaussian_filter1d(r_theta, sigma, mode="nearest")

    points = np.stack([fx + r_theta * np.cos(thetas),
                       fy + r_theta * np.sin(thetas)], axis=1).astype(np.float32)
    return PolarArch(
        focal=(fx, fy), thetas=thetas, r_theta=r_theta, points=points,
        slab=slab, mask=mask, occlusal_z=_occlusal_z(arr, points, spacing),
    )


def _arch_from_points(
    pts: np.ndarray,
    spacing: tuple[float, float, float],
    slab: np.ndarray,
    mask: np.ndarray,
    occlusal_z: int | None,
) -> ArchCurve | None:
    """Build an ArchCurve from an ordered (x, y) centerline: equal-arc-length resample +
    tangents/normals (same tail as detect_arch_curve, so the renderers are shared)."""
    d = np.diff(pts, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < _MIN_VALID_COLS:
        return None
    m = max(int(arc[-1]), 32)
    s_new = np.linspace(0.0, arc[-1], m)
    xr = np.interp(s_new, arc, pts[:, 0])
    yr = np.interp(s_new, arc, pts[:, 1])
    p = np.stack([xr, yr], axis=1)
    t = np.gradient(p, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn == 0] = 1.0
    tang = t / tn[:, None]
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return ArchCurve(
        points=p, normals=normals, tangents=tang,
        arc_mm=s_new * float(spacing[0]), spacing=spacing,
        slab=slab, mask=mask, occlusal_z=occlusal_z,
    )


_PARABOLA_DEG = 2                     # focal-trough fit degree (2 = parabola; the OPG shape)


def _fit_parabola(pts: np.ndarray, deg: int = _PARABOLA_DEG) -> np.ndarray:
    """Fit a smooth low-degree focal-trough curve to the (noisy) polar arch points.

    Fits v = P(u) in the arch's *principal* frame (PCA: u = major/left-right axis, v =
    minor/anterior-posterior axis) so a slight head rotation doesn't break single-valuedness.
    Degree 2 is the classic parabolic dental-arch shape — smoothing away the per-ray r(θ)
    jitter and the rami curl-back that made the raw polar curve's lower edges look weird."""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    # u must be the LEFT-RIGHT axis: after LAS reorient the dental arch is always widest
    # L-R, and v = P(u) needs the arch single-valued in u. Plain PCA takes the *largest-
    # variance* axis as major, which on full-FOV scans is the ANTERIOR-POSTERIOR spread of
    # spine/skull/ramus bone, not the arch — flipping the unroll into a side view (verified
    # F002 87°, S0000 48°). So pick whichever principal axis is more aligned with x (L-R).
    e_a, e_b = vt[0], vt[1]
    e1, e2 = (e_a, e_b) if abs(e_a[0]) >= abs(e_b[0]) else (e_b, e_a)
    if e1[0] < 0:                                          # orient u along +x (stable L→R columns)
        e1 = -e1
    u = q @ e1
    v = q @ e2
    coeff = np.polyfit(u, v, deg)
    p = np.poly1d(coeff)
    m = max(int(u.max() - u.min()), 32)
    ug = np.linspace(float(u.min()), float(u.max()), m)
    out = c[None, :] + ug[:, None] * e1[None, :] + p(ug)[:, None] * e2[None, :]
    return out.astype(np.float32)


def detect_arch_curve_polar(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
    z_band: tuple[float, float] | None = None,
    smooth_deg: int | None = _PARABOLA_DEG,
) -> ArchCurve | None:
    """Star/radial arch detector as a drop-in for detect_arch_curve: recover r(θ) via
    polar_arch, then (by default) fit a smooth parabolic focal trough and re-parametrise to
    equal arc length. ``smooth_deg=None`` keeps the raw polar points. Returns an ArchCurve
    (consumed unchanged by panoramic_curved / arch_cross_sections) or None."""
    pa = polar_arch(arr, spacing, z_band)
    if pa is None:
        return None
    pts = pa.points if smooth_deg is None else _fit_parabola(pa.points, smooth_deg)
    if not _arch_shape_ok(pts):                      # skull-tracing oval -> untrustworthy
        return None
    return _arch_from_points(pts, spacing, pa.slab, pa.mask, pa.occlusal_z)


_MAX_ARCH_ASPECT = 0.80    # A-P depth / L-R width. A real dental arch is ~0.55 (35mm deep, 63mm
                           # wide); a full-skull-FOV oval traced by the polar rays is ~1.0. Reject
                           # above this so those cases fall back to the flat panoramic (verified
                           # F002/S0000 traced the skull, not the dentition) — a bad trough is
                           # worse than none.


def _arch_shape_ok(pts: np.ndarray) -> bool:
    """True if ``pts`` look like a dental arch (wider L-R than deep A-P), False for the
    near-circular skull outline the polar detector latches onto on cluttered full-FOV scans."""
    q = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    ea, eb = vt[0], vt[1]
    e1, e2 = (ea, eb) if abs(ea[0]) >= abs(eb[0]) else (eb, ea)   # e1 = L-R axis
    u, v = q @ e1, q @ e2
    width = float(u.max() - u.min())
    depth = float(v.max() - v.min())
    return width > 1e-6 and (depth / width) <= _MAX_ARCH_ASPECT


# ── Dentition-driven arch detection ───────
#
# The three detectors above all derive geometry from a flattened projection of *everything*:
# a ~50 mm-thick axial slab, thresholded at a fixed percentile, largest connected component.
# On a jaw-only FOV that component is the dentition; on a full-FOV scan it is ramus + zygoma +
# skull base + C-spine fused into one blob that dwarfs the tooth arch, and the trough is fitted
# to the skull (measured: 21/63 usable on val_ood).
#
# This detector derives geometry from the densest tissue in the scan instead — the dentition.
# Teeth are the brightest natural structure in a dental CBCT, which is a stronger and cheaper
# prior than connectedness of a percentile-thresholded slab, and it needs no segmentation model
# (val_ood 21/63 -> 60/63). Everything here was prototyped and ablated in `probe_arch2` /
# `probe_unroll`; the measurements behind each constant are documented with them.

OCCL_SLAB_MM = 8.0       # half-thickness of the tooth-cloud slab about the occlusal plane
Z_GUARD_FRAC = 0.04      # ignore the outermost Z slices (FOV rim / reconstruction artefacts)
N_ANGLES = 360
TOOTH_CAP_PCT = 88.0     # keep at least the top 12% of the bone class as enamel (metal cap)
SEED_CANDIDATES = 4      # largest components tried as the arch seed; best arch-likeness wins
R_PERCENTILE = 35.0      # radius percentile per angular bin; see fit_arch_cloud
ARCH_GAP_MM = 20.0       # merge cloud components within this of the seed. Swept on rendered
                         # images: 6 mm fragments a gappy dentition (A012 collapses to a 9 mm
                         # arc), no removal lets the rami drag the trough off both tooth rows;
                         # 14-20 mm works on every case tried.
MIN_RADIUS_MM = 18.0     # curvature bound: a dental arch's tightest real curvature is at the
                         # incisors (~14-18 mm). Below the integration half-depth the trough
                         # normals cross inside the slab and the anterior teeth fan out.

# ── Fixed tissue landmarks ─────────────────
# ToothFairy4 volumes share one intensity scale: the air mode is at -1000 on 125/128 cached
# volumes, and the three exceptions have no air in the FOV at all (their floor is still -1024
# and their soft-tissue mode is +30..+90 like everyone else) rather than a shifted scale. Soft
# tissue sits at +10..+130 on all 128.
#
# The old justification for cascaded Otsu — "t_tooth ranges 979 -> 2265, so a fixed threshold
# would not transfer" — was inferred from the spread of the *Otsu thresholds*, which is
# circular: Otsu's split point moving is not the same as the tissue moving. Measured against
# the tissue, fixed thresholds transfer *better* than Otsu (120/128 usable troughs vs 117/128)
# and rescue every metal case, because Otsu is what metal corrupts:
# corr(metal_frac, t_bone) = 0.73, and on a heavily restored mouth t_head/t_bone are dragged to
# +94/1748 against a typical -323/779.
HU_AIR       = -1000.0   # the floor every volume in this dataset shares
HU_SOFT      = 0.0       # soft-tissue mode (measured +10..+130)
HU_T_HEAD    = -500.0    # air|soft
HU_T_BONE    = 300.0     # soft|bone
HU_T_TOOTH   = 1500.0    # bone|enamel — best single level over 128 cases (120/128)
HU_METAL     = 3000.0    # above any natural dental tissue; 61/128 cases have exactly zero
# Levels tried in order when the first does not yield a usable trough. The union over this
# ladder covers 127/128 (appending the Otsu rung takes it to 128/128 — P010 passes under Otsu
# and under no fixed level), so the residual is a *selection* problem, not a case for
# per-volume thresholding. Ordered by single-level score (1500 > 1300/1700 > 1100 > 900).
HU_TOOTH_LADDER = (1500.0, 1700.0, 1300.0, 1100.0, 900.0)

# ── Enamel core: the threshold that separates teeth from the ramus ───────────────
# records that intensity *cannot* separate the dentition
# from the mandibular ramus, citing enamel at 2774 against ramus cortex at 2358 on F002. Measured
# again in **calibrated pseudo-HU** rather than in raw Otsu units, that is wrong -- and the reason
# the original measurement said otherwise is that it compared raw values across volumes whose
# offset and gain differ by 40 %:
#
#   over 10 cases, of the >1500 HU voxels in each cloud component
#     kept (dentition)          p50 1903   p90 2935 (min 2235)   p99 3648
#     dropped (ramus/cortex)    p50 1566   p90 1719 (max 1954)   p99 1919
#     dropped p90 < kept p90 on 10/10
#
# So a level around 2100 HU keeps crowns and drops cortical bone. It is used only for the *arch
# fit*, never for `occlusal_z` (which needs the mass of the whole band to find the plane at all)
# and never as a gate: when the core cloud is too small -- a clipped reconstruction like F056 tops
# out at 1660 HU and has no voxels above it -- the fit falls back to the full cloud.
HU_TOOTH_CORE = 2100.0
CORE_MIN_MM2 = 250.0     # below this the core is not a dentition; fall back to the full cloud

# Anchor sanity check. Fixed thresholds are a claim about *this dataset*, not about CBCT in
# general: whatever produced these volumes applied a consistent scale, and a new source family
# could break it. If the measured anchors are not where they should be, fall back to Otsu.
HU_AIR_TOL   = 120.0     # how far the air mode may sit from HU_AIR
HU_SOFT_TOL  = 700.0     # ... and the soft-tissue mode from HU_SOFT. Generous, because an
                         # offset within this range is *corrected* by the calibration rather
                         # than rejected; the check only has to catch scales this affine cannot
                         # represent at all.
HU_MIN_AIR_FRAC = 0.02   # below this there is no air peak to test (tight-FOV scans: A032/A071/
                         # A091). Those are not miscalibrated, so the check passes on soft alone
                         # and the gain falls back to DEFAULT_SCALE.
SOFT_SEARCH  = (-400.0, 1200.0)   # window the soft-tissue mode is sought in; see
                                  # `intensity_anchors` for why it must be wider than HU_T_BONE
SOFT_VALLEY_LO = 200.0            # search for the air|tissue valley + soft peak this far above
SOFT_VALLEY_HI = 2200.0           # ... to this far above the air mode
DEFAULT_SCALE = 1080.0   # dataset median of (soft - air); used when there is no air in the FOV
SCALE_RANGE   = (700.0, 1800.0)   # plausibility bound on the measured gain


def _otsu(vals: np.ndarray, lo: float, hi: float, nbins: int = 256) -> float:
    """Between-class-variance threshold of ``vals`` restricted to [lo, hi]."""
    h, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
    p = h.astype(np.float64)
    tot = p.sum()
    if tot <= 0:
        return float(0.5 * (lo + hi))
    p /= tot
    c = np.arange(nbins) + 0.5
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    m1 = np.cumsum((p * c)[::-1])[::-1] / np.maximum(w1, 1e-12)
    var = w0 * w1 * (m0 - m1) ** 2
    k = int(np.nanargmax(var[:-1]))
    return float(edges[0] + (k + 1) * (edges[1] - edges[0]))


def otsu_thresholds(arr: np.ndarray) -> tuple[float, float, float]:
    """Cascaded Otsu -> (t_head, t_bone, t_tooth): air|soft, soft|bone, bone|enamel.

    **Now the fallback, not the primary** — see `hu_thresholds` The premise this was built on ("CBCT values are not
    calibrated across scanners, so a fixed threshold would not transfer") does not hold for
    ToothFairy4, and Otsu is the component metal corrupts.

    Per-volume and parameter-free. Sub-sampled; the top 0.1% is excluded so metal cannot
    stretch the histogram range.

    **Metal cap.** In a heavily restored mouth the bone-class histogram is bimodal *bone|metal*
    rather than *bone|enamel*, so Otsu splits off the metal and t_tooth lands absurdly high —
    A019 5344, A091 2480, P042 2134 against a typical 1000-1800. The dentition cloud then holds
    only crowns and implants and the arch is fitted to two blobs. Flooring t_tooth at a
    percentile of the bone class keeps a fixed share of it as "enamel"."""
    a = arr[::2, ::2, ::2].astype(np.float32).ravel()
    p_lo = float(np.percentile(a, 0.5))
    p_hi = float(np.percentile(a, 99.9))
    t_head = _otsu(a, p_lo, p_hi)
    head = a[a > t_head]
    t_bone = _otsu(head, t_head, p_hi) if head.size > 1000 else t_head
    bone = head[head > t_bone]
    t_tooth = _otsu(bone, t_bone, p_hi) if bone.size > 1000 else t_bone
    if bone.size > 1000:
        t_tooth = min(t_tooth, float(np.percentile(bone, TOOTH_CAP_PCT)))
    return t_head, t_bone, max(t_tooth, t_bone)


def intensity_anchors(arr: np.ndarray) -> tuple[float, float, float]:
    """(air mode, soft-tissue mode, air fraction) — the two calibration anchors.

    Both are set by the bulk of the histogram, so metal cannot move either. Deliberately
    Otsu-free: anchoring the soft window on `t_head` is what makes the equivalent estimator in
    `probe_calib` report a wrong soft mode on restored mouths.

    The soft window is **wide** (`SOFT_SEARCH`) rather than clipped at the nominal bone landmark.
    A window that stops at HU_T_BONE cannot represent a volume whose scale is genuinely offset —
    it just returns its own upper edge, which then passes any tolerance check. A072 is the case:
    its soft-tissue mode is at **+415**, a narrow window reported 298, the volume was treated as
    on-scale, and every ray picked up ~0.16 of baseline attenuation from soft tissue that should
    have been transparent.

    **A plain argmax over that wide window is not enough either.** On a volume with a lot of air
    and a broad air-to-tissue transition, the largest bin in the window sits on that shoulder
    rather than on the soft-tissue peak: A003 reported -275 against a true ~+195, so its measured
    gain came out 720 instead of 1190 and the affine over-attenuated *everything* — band median
    166/255 against a survey median of 104, the washed-out frame. So when there is an air peak to
    walk away from, the soft mode is taken as the first real peak *past the valley* that separates
    it from air. Volumes with no air in the FOV (A032, A071, A091) have no valley and no air peak;
    forcing the walk on them lands on bone (A091 +990), so they keep the plain argmax."""
    a = arr[::4, ::4, ::2].astype(np.float32).ravel()
    air_frac = float((a < HU_AIR + 100.0).mean())
    lo, hi = float(np.percentile(a, 0.02)), float(np.percentile(a, 99.5))
    edges = np.arange(lo, hi + 10.0, 10.0)
    if edges.size < 5:
        return HU_AIR, HU_SOFT, air_frac
    h, _ = np.histogram(a, bins=edges)
    c = 0.5 * (edges[:-1] + edges[1:])
    hs = ndimage.gaussian_filter1d(h.astype(np.float64), 3.0)
    below = c < HU_T_HEAD
    air = float(c[below][np.argmax(hs[below])]) if below.any() else HU_AIR

    win = (c > air + SOFT_VALLEY_LO) & (c < air + SOFT_VALLEY_HI)
    idx = np.nonzero(win)[0]
    if air_frac >= HU_MIN_AIR_FRAC and idx.size >= 5:
        seg = hs[idx]
        j = int(np.argmin(seg[:max(seg.size // 2, 2)]))       # the air|tissue valley
        return air, float(c[idx[j + int(np.argmax(seg[j:]))]]), air_frac
    m = (c > SOFT_SEARCH[0]) & (c < SOFT_SEARCH[1])
    soft = float(c[m][np.argmax(hs[m])]) if m.any() else HU_SOFT
    return air, soft, air_frac


def calibrate(arr: np.ndarray) -> tuple[float, float, bool]:
    """(soft, scale, on_scale) — the two-point affine plus whether it can be trusted.

    ``on_scale`` is False when the measured anchors are not where this dataset puts them, i.e.
    a volume from some future source family on a genuinely different scale. Callers should then
    use `otsu_thresholds` and treat the returned (soft, scale) as a best effort only.

    Kept as one call because both the thresholds and the tone map need the same numbers, and
    each extra call is another subsampled histogram over the volume."""
    air, soft, air_frac = intensity_anchors(arr)
    scale = (soft - air) if air_frac >= HU_MIN_AIR_FRAC else DEFAULT_SCALE
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        scale = DEFAULT_SCALE
    air_ok = air_frac < HU_MIN_AIR_FRAC or abs(air - HU_AIR) <= HU_AIR_TOL
    return soft, scale, bool(air_ok and abs(soft - HU_SOFT) <= HU_SOFT_TOL)


def hu_to_raw(hu: float, soft: float, scale: float) -> float:
    """Canonical pseudo-HU landmark -> this volume's raw intensity."""
    return soft + hu * (scale / 1000.0)


def intensity_calibration(arr: np.ndarray) -> tuple[float, float]:
    """(soft, scale) mapping raw intensity -> pseudo-HU by ``(v - soft) * 1000 / scale``,
    i.e. air -> -1000 and soft tissue -> 0.

    This is the two-point affine the fixed thresholds are *really* expressed in. Treating the
    raw values as already-canonical works on the ~95 % of volumes whose anchors sit at
    (-1000, ~+85), and fails on the rest — A072's soft tissue at +415 is a 40 % gain error, not
    a rounding difference.

    Both anchors are metal-immune by mass, which is the whole point: this is a *calibration*,
    not a content-dependent normalisation like the percentile windows v1 used or the Otsu ladder
    v2 used. Volumes with no air in the FOV (A032, A071, A091) get the dataset's median gain,
    since their offset is still measurable and only the gain is not."""
    soft, scale, _ = calibrate(arr)
    return soft, scale


def hu_thresholds(arr: np.ndarray, t_tooth: float = HU_T_TOOTH) -> tuple[float, float, float]:
    """Fixed tissue landmarks in pseudo-HU, mapped back to this volume's raw units.

    Returns raw-intensity thresholds equivalent to (HU_T_HEAD, HU_T_BONE, t_tooth) after the
    two-point calibration, so the *tissue* the thresholds select is the same on every volume
    even when the raw scale is offset or differently gained.

    Falls back to `otsu_thresholds` when the calibration is implausible — a future source family
    on a genuinely different scale degrades to the old behaviour rather than rendering nonsense.
    Tight-FOV scans contain no air at all; they are not miscalibrated, so the air half of the
    check is skipped when there is no air peak to test."""
    soft, scale, on_scale = calibrate(arr)
    if not on_scale:
        return otsu_thresholds(arr)
    return (hu_to_raw(HU_T_HEAD, soft, scale), hu_to_raw(HU_T_BONE, soft, scale),
            hu_to_raw(t_tooth, soft, scale))


def occlusal_z(arr: np.ndarray, t_tooth: float, spacing) -> tuple[int, np.ndarray]:
    """Occlusal plane = argmax of the per-slice **enamel**-voxel count.

    `_occlusal_z` above takes the argmax of mean *intensity*, which the mandible body wins — a
    large volume of cortical + trabecular bone outweighs the thin enamel band, so the plane
    lands in the mandible, the hyoid or the neck (median disagreement 22.8 mm on val_ood, p90
    38.9). Enamel is denser than any bone, so counting only above-t_tooth voxels fixes it."""
    nz = arr.shape[2]
    prof = (arr > t_tooth).sum(axis=(0, 1)).astype(np.float32)
    prof = ndimage.gaussian_filter1d(prof, max(2.0 / float(spacing[2]), 1.0))
    g = max(int(Z_GUARD_FRAC * nz), 1)
    return g + int(np.argmax(prof[g:nz - g])), prof


ROW2_MIN_SEP_MM = 8.0    # the second crown band must be at least this far from the first
ROW2_MIN_FRAC = 0.0      # ... and that is the only condition: the second band is *always*
                         # taken when the profile has one, because a trough fitted to a
                         # single jaw drops the other jaw's anterior teeth. On an
                         # edentulous jaw the second band is alveolar bone, which widens
                         # the cloud but does not move the arch (F012 verified).


def occlusal_rows(arr: np.ndarray, t_tooth: float, spacing) -> tuple[int, int | None]:
    """(first crown band, second crown band or None) from the enamel-count Z profile.

    `occlusal_z` returns one plane, and a cloud built in a +/-8 mm slab about it therefore holds
    **one jaw**. When the bite is open — or simply when the mandible hangs low — the other row is
    tens of millimetres away and never enters the cloud, so the parabola is fitted to the maxillary
    arch alone. The mandibular arch is narrower and sits lingual to it, so its anterior teeth then
    fall outside the integration slab: F007's crown bands are z = 214 and z = 122 (27.6 mm apart)
    and its lower incisors rendered as a black gap in the middle of the frame.

    The second band is the strongest remaining peak of the same profile once a guard around the
    first is suppressed. It is optional: an edentulous jaw has no second peak, which is the correct
    answer rather than a failure."""
    z1, prof = occlusal_z(arr, t_tooth, spacing)
    nz = arr.shape[2]
    spz = float(spacing[2])
    g = max(int(Z_GUARD_FRAC * nz), 1)
    guard = max(int(round(ROW2_MIN_SEP_MM / spz)), 1)
    p = prof.copy()
    p[max(z1 - guard, 0):min(z1 + guard + 1, nz)] = 0.0
    interior = p[g:nz - g]
    if interior.size == 0 or float(interior.max()) < ROW2_MIN_FRAC * float(prof[z1]):
        return z1, None
    return z1, g + int(np.argmax(interior))


def tooth_cloud(arr: np.ndarray, z_occ: int, t_tooth: float, spacing,
                dedup_ramus: bool = True, z_occ2: int | None = None) -> np.ndarray:
    """(nx, ny) weight map of enamel voxels in a slab about the occlusal plane — the dentition.

    No largest-CC step: at the occlusal plane teeth are *separate objects*, so the largest
    component is one tooth or one metal blob.

    ``dedup_ramus`` drops the mandibular ramus / coronoid, which also clears the enamel
    threshold at the maxillary occlusal plane. Left in, they drag the fitted trough posteriorly
    until it sits *between* the two arches and both tooth rows fall outside the integration
    slab — the dropped-maxilla failure. Two natural fixes were measured and do not work:
    intensity (enamel and ramus cortex overlap: F002 2774 vs 2358), and a radius percentile
    (teeth occupy the anterior angular bins and the rami the lateral ones, so they are disjoint
    in theta). What works is connectivity — in a thin slab the dental arch is one horseshoe
    component and the rami are separate posterior blobs.

    **Seed by arch-likeness, not by size.** The largest enamel-threshold component is not always
    the dentition: on F028 it is the cervical spine, whose cortical bone is dense at the same Z,
    and the trough is then fitted to the spine and unrolls a column of vertebrae. Scoring
    candidates by (L-R + A-P extent) x (minor/major singular-value ratio) — wide *and*
    two-dimensional, unlike a vertebral body or a metal cluster — picks the arch."""
    half = max(int(round(OCCL_SLAB_MM / float(spacing[2]))), 1)

    def band(zc: int) -> np.ndarray:
        a, b = max(zc - half, 0), min(zc + half + 1, arr.shape[2])
        return (arr[:, :, a:b] > t_tooth).sum(axis=2).astype(np.float32)

    # Union over both crown bands, so the trough is fitted to *both* arches rather than to
    # whichever one happened to win the Z profile (see `occlusal_rows`).
    w = band(z_occ) if z_occ2 is None else band(z_occ) + band(z_occ2)
    if not dedup_ramus:
        return w
    lab, n = ndimage.label(w > 0)
    if n < 2:
        return w
    sizes = ndimage.sum(np.ones_like(lab, np.float32), lab, np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:SEED_CANDIDATES]
    best, best_score = None, -1.0
    for k in order:
        seed = lab == (1 + int(k))
        gap = ndimage.distance_transform_edt(~seed) * float(spacing[0]) <= ARCH_GAP_MM
        keep = np.zeros(n + 1, bool)
        keep[np.unique(lab[gap & (lab > 0)])] = True
        keep[0] = False
        merged = np.where(keep[lab], w, 0.0)
        xs, ys = np.nonzero(merged > 0)
        if xs.size < 200:
            continue
        q = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
        _, sv, _ = np.linalg.svd(q, full_matrices=False)
        span = float(np.ptp(xs) + np.ptp(ys)) * float(spacing[0])
        score = span * float(sv[1] / max(sv[0], 1e-6))
        if score > best_score:
            best, best_score = merged, score
    return best if best is not None else w


# ── The arch's frame is the canonical grid; only the midline is measured ──────────
#
# `_fit_parabola` takes its frame from the *cloud's* PCA. That frame is ill-conditioned here by
# construction: over the 128 cached volumes sv[1]/sv[0] has a median of 0.66 and a p90 of 0.82,
# so the two principal axes are nearly interchangeable and a lopsided or sparse dentition tilts
# the frame by tens of degrees — F003 -42, F005 -45, F004 -37, S0020 +35, F012's chord -35. That
# tilt is what the survey sheet reported as "distorted yaw"; it is a fit artefact, not anatomy.
#
# **There is nothing to fit.** `volume_io.load_canonical_volume` reorients every volume to LAS, so
# x *is* the patient's left-right axis and y *is* anterior — the same axes the coronal MIP is taken
# along. A panoramic radiograph is the coronal view with the lateral anatomy stretched out, so its
# angle is the coronal one by definition. Estimating it can only introduce error, and did: a
# mirror-symmetry search over yaw picks the wrong lobe on A003, whose jaw-level bone mask is nearly
# round and therefore scores 0.815 at -38 deg, 0.807 at +40 deg and a *minimum* 0.677 at 0 deg.
#
# The one thing the grid does not give is *where* the mid-sagittal plane sits in x, and the vertex
# has to sit on it or the two halves of the arch come out different lengths. That is a 1-D
# measurement (`head_midline`) and cannot be ambiguous in angle.
#
# The old `axis_deg > 30` failure criterion was dropped
# because it "punished genuinely rotated heads" — P210 was the example. P210's arch is aligned with
# the canonical axis; what `axis_deg` reported was this same fit artefact, not chin-rest yaw.
SYM_BAND_MM = 30.0        # Z half-band about the occlusal plane the midline is measured over
SYM_SEARCH_MM = 25.0      # how far either side of the bone centroid the midline is searched
SYM_STEP_MM = 0.5
RING_GAP_DEG = 40.0       # a real posterior opening; below this the span is a closed ring
POST_CUT_DEG = 50.0       # half-width of the posterior wedge cut when the span is a ring

# The canonical axes: x = patient left, y = anterior. Not fitted, not estimated.
E_LR = np.array([1.0, 0.0])
E_AP = np.array([0.0, 1.0])


def head_midline(arr: np.ndarray, spacing, z_occ: int,
                 t_bone: float) -> tuple[tuple[float, float], float]:
    """(mid-sagittal point (x, y), symmetry score) — a 1-D search for the midline's x.

    Reflecting the jaw-level bone mask about x = x0 should reproduce it when x0 is the
    mid-sagittal plane; scored as the normalised overlap and maximised over x0 near the mask's
    centroid. **Bone, not the skin silhouette** — an axial head outline is close to an ellipse and
    an ellipse is symmetric about any axis through its centre, so it carries almost no information;
    mandible + maxilla + spine do. Metal cannot dominate a binary mask of ~10^5 voxels.

    Deliberately **only** the offset, not the angle. The angle comes from the canonical LAS grid
    (see the section note): searching it as well is what fails on A003, where the same score is
    bimodal at -38 deg and +40 deg with a minimum at the truth.

    Unlike the tooth cloud this works on an edentulous jaw, which is the point: the midline is a
    property of the scan and should not depend on whether the patient has teeth."""
    nz, nx, ny = arr.shape[2], arr.shape[0], arr.shape[1]
    hz = max(int(round(SYM_BAND_MM / float(spacing[2]))), 1)
    z0, z1 = max(z_occ - hz, 0), min(z_occ + hz + 1, nz)
    m = (arr[:, :, z0:z1] > t_bone).any(axis=2)
    xs, ys = np.nonzero(m)
    if xs.size < 500:
        return (float(nx / 2), float(ny / 2)), 0.0
    prof = m.sum(axis=1).astype(np.float64)            # bone per x column, summed over y
    c = float((prof * np.arange(nx)).sum() / max(prof.sum(), 1e-9))
    step = max(SYM_STEP_MM / float(spacing[0]), 0.5)
    best_x, best_sc = c, -1.0
    grid = np.arange(nx, dtype=np.float64)
    for x0 in np.arange(c - SYM_SEARCH_MM / float(spacing[0]),
                        c + SYM_SEARCH_MM / float(spacing[0]) + 1e-9, step):
        mirrored = np.interp(2.0 * x0 - grid, grid, prof, left=0.0, right=0.0)
        sc = float(np.minimum(prof, mirrored).sum() / max(prof.sum(), 1e-9))
        if sc > best_sc:
            best_x, best_sc = float(x0), sc
    return (best_x, float(ys.mean())), best_sc


def _span_posterior_cut(n: int, half_deg: float = POST_CUT_DEG) -> np.ndarray:
    """Dental span for a cloud with **no posterior opening** — the ring case.

    `_dental_span` finds the opening as the widest run of *empty* angular bins, which requires
    the opening to be empty. On 8/128 cached volumes it is not (metal bloom, or a mandible whose
    rami and the C-spine bridge behind the arch): the widest gap is 1-2 bins, the span wraps the
    full circle, the centreline closes into a loop, and a curve fitted to a loop collapses onto a
    near-flat chord — F056 renders 46 mm of trough for a 111 mm arch, S0000 a 40 mm sliver.

    The canonical grid supplies the fix for free: after the LAS reorient **posterior is -y**, so
    the opening's direction is known a priori and does not have to be discovered. Cut a wedge of
    +/-``half_deg`` about it. An anatomical constant, not a fitted one."""
    th = -180.0 + (np.arange(n) + 0.5) * (360.0 / n)
    cut = np.nonzero(np.abs(((th + 90.0 + 180.0) % 360.0) - 180.0) <= half_deg)[0]
    if cut.size == 0 or cut.size >= n - 8 or np.ptp(cut) >= n - 1:
        return np.arange(n)          # wedge wraps the array ends: leave the span alone
    return (int(cut[-1]) + 1 + np.arange(n - cut.size)) % n


def _dental_span(valid: np.ndarray) -> np.ndarray:
    """Angular-bin indices of the dental span: everything except the widest circular gap.

    A dental arch is an open horseshoe — the single widest run of *empty* angular bins is its
    posterior opening. Taking the complement (rather than the longest filled run) tolerates the
    small dropouts a missing tooth or a diastema leaves in the middle of the arch. Without this
    the centreline closes into a loop and the unroll repeats the anatomy."""
    n = len(valid)
    filled = np.nonzero(valid)[0]
    if filled.size == 0:
        return filled
    gaps = np.diff(np.concatenate([filled, filled[:1] + n]))
    k = int(np.argmax(gaps))
    start = int(filled[(k + 1) % filled.size])
    return (start + np.arange(int(n - gaps[k]) + 1)) % n


def limit_curvature(curve: ArchCurve, min_radius_mm: float = MIN_RADIUS_MM) -> ArchCurve | None:
    """Smooth the trough until its radius of curvature stays above ``min_radius_mm``.

    A curved reformat integrates along the trough normal; where the radius of curvature drops
    below the integration half-depth the normals **cross inside the slab** and the anterior
    teeth smear into a fan. Measured on the prototype troughs: S0001 10.2 mm, A012 11.0 mm,
    F001 14.3 mm against a 14 mm half-depth — this really happens. Gaussian-smoothing a
    centreline strictly lowers |kappa|, so iterate until the tightest point clears the bound."""
    pts = curve.points.astype(np.float64).copy()
    sp = float(curve.spacing[0])
    for _ in range(24):
        d1 = np.gradient(pts, axis=0)
        d2 = np.gradient(d1, axis=0)
        num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        den = np.power(np.hypot(d1[:, 0], d1[:, 1]), 3) + 1e-9
        if float(np.percentile(num / den / sp, 98)) <= 1.0 / min_radius_mm:
            break
        pts = ndimage.gaussian_filter1d(pts, 3.0, axis=0, mode="nearest")
    out = _arch_from_points(pts.astype(np.float32), curve.spacing, curve.slab,
                            curve.mask, curve.occlusal_z)
    if out is not None:
        out.thresholds = curve.thresholds
        out.midline_xy, out.sym_score = curve.midline_xy, curve.sym_score
        out.parabola = curve.parabola
    return out


CENTRELINE_SMOOTH = 8.0      # Gaussian sigma, in angular bins, applied to the r(theta) centreline
                             # before the curvature bound. In mm that is ~3.5 anteriorly and ~5.6
                             # laterally (sigma * r * dtheta), i.e. more smoothing exactly where
                             # the centreline is sampled more sparsely. Swept 8/16/24 over 128
                             # cases: 8 is best on centreline deviation (p90 2.8 / 4.3 / 6.2 mm)
                             # and on how far short of the last molar the trough stops
                             # (12.4 / 17.1 / 21.9 mm); the gate passes 124/128 at all three.


def arch_centreline(w: np.ndarray, spacing) -> np.ndarray | None:
    """Angular-median r(theta) about the dentition centroid -> the smoothed arch centreline.

    A *low* radius percentile, not the median: laterally the ray leaving the arch crosses the
    maxillary molar first and the ramus/coronoid second, and both pass the enamel threshold.
    The median lands between them, so the trough is pulled posteriorly and both tooth rows fall
    outside the slab. Taking the nearest structure keeps the dentition and skips the ramus,
    which is the principle `polar_arch` already uses.

    This is an *observation* of where the dentition is, not the trough. It is the input both
    `fit_arch_symmetric` (the shipped shape model) and `fit_arch_cloud` (the unconstrained
    comparison) are fitted to, and — because it is measured from the cloud rather than from any
    curve — it is also the yardstick the fits are scored against in
    `toothfairy/probe_arch_fit.py`. `coverage` cannot play that role: with a 14 mm integration
    slab it is nearly blind to a trough that stops 25 mm short at each end, which is why three
    rounds of work scored a badly short trough as a pass.

    Smoothing (`CENTRELINE_SMOOTH`) is what the old parabola fit was really contributing —
    suppression of per-crown r(theta) jitter — and it is kept here so the shape model is fitted to
    a clean curve. The traversal order is fixed on the canonical L-R axis rather than in the
    cloud's ill-conditioned PCA frame, so the panoramic's columns run patient-right to
    patient-left on every case."""
    xs, ys = np.nonzero(w > 0)
    if xs.size < 200:
        return None
    ww = w[xs, ys].astype(np.float32)
    cx = float((xs * ww).sum() / ww.sum())
    cy = float((ys * ww).sum() / ww.sum())
    dx, dy = xs - cx, ys - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    bins = np.clip(((th + np.pi) / (2 * np.pi) * N_ANGLES).astype(int), 0, N_ANGLES - 1)
    r_med = np.full(N_ANGLES, np.nan, np.float32)
    order = np.argsort(bins)
    b_sorted, r_sorted = bins[order], r[order]
    edges = np.searchsorted(b_sorted, np.arange(N_ANGLES + 1))
    for k in range(N_ANGLES):
        seg = r_sorted[edges[k]:edges[k + 1]]
        if seg.size >= 3:
            r_med[k] = np.percentile(seg, R_PERCENTILE)
    valid = ~np.isnan(r_med)
    if valid.sum() < N_ANGLES // 4:
        return None
    # The posterior opening is normally found as the widest run of *empty* angular bins. On 8/128
    # cached volumes there is no empty run — metal bloom or a mandible bridged to the C-spine
    # closes the ring — and the span then wraps the full circle. Fall back to cutting the wedge
    # about the known posterior direction (see `_span_posterior_cut`).
    filled = np.nonzero(valid)[0]
    gaps = np.diff(np.concatenate([filled, filled[:1] + N_ANGLES]))
    ring = float(gaps.max()) * 360.0 / N_ANGLES < RING_GAP_DEG
    idx = _span_posterior_cut(N_ANGLES) if ring else _dental_span(valid)
    if idx.size < N_ANGLES // 6:
        return None
    thetas = np.unwrap(-np.pi + (idx + 0.5) * (2 * np.pi / N_ANGLES))
    # The span is the complement of the posterior opening, so it can contain interior dropouts
    # (a missing tooth, a diastema). Interpolate them rather than dropping the bin — the trough
    # must span an edentulous stretch, because an empty ridge is a finding the report needs.
    rspan = r_med[idx].astype(np.float32)
    good = ~np.isnan(rspan)
    if good.sum() < 8:
        return None
    rspan = np.interp(np.arange(rspan.size), np.nonzero(good)[0], rspan[good])
    rr = ndimage.gaussian_filter1d(rspan, 3.0, mode="nearest")
    pts = np.stack([cx + rr * np.cos(thetas), cy + rr * np.sin(thetas)], axis=1)
    pts = ndimage.gaussian_filter1d(pts, CENTRELINE_SMOOTH, axis=0, mode="nearest")
    # Traverse patient-right -> patient-left, so the panoramic's column order is the coronal
    # view's on every case (a panoramic is the coronal view with the lateral anatomy stretched
    # out; the view angle has to be the same).
    if float(pts[-1, 0] - pts[0, 0]) < 0.0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def fit_arch_cloud(w: np.ndarray, spacing, slab: np.ndarray) -> ArchCurve | None:
    """The centreline as an ArchCurve, with no shape model imposed. Kept for the probes that
    A/B against `fit_arch_symmetric`; `detect_arch_curve_dentition` uses the parabola."""
    pts = arch_centreline(w, spacing)
    if pts is None:
        return None
    return _arch_from_points(pts, spacing, slab, w > 0, None)


# ── The arch as a parabola posed on the canonical axes ───────────────────────────
#
# The trough is a **parabola**, level with the coronal axes, described by four numbers and nothing
# else: the vertex (x, y, z) and the spread. x is the mid-sagittal plane (`head_midline`), z is the
# occlusal plane (`occlusal_z`), and y and the spread are all that is fitted to the dentition.
#
# The value of that restriction is what it makes impossible. `v = v0 - c·u²` on the canonical axes
# cannot come out lopsided or S-shaped, cannot put both posterior ends on the same side, cannot
# rotate away from the coronal view, and cannot have a vertex off the midline. Every failure of the
# old fit was of exactly that kind, and each was a degree of freedom that should not have existed:
#
#   - the frame from the cloud's PCA, ill-conditioned (see the section note above);
#   - a free linear term, which tilts a symmetric arch (measured chord tilt up to 58 deg);
#   - a free u-range, which let the two halves of the frame have different lengths;
#   - a searched yaw, which picks the wrong lobe on a near-round bone mask (A003).
#
# `c` is solved by non-negative least squares, so the parabola is guaranteed concave-posterior with
# its single vertex on the midline.
ARCH_HALF_WIDTH_MM = (22.0, 35.0)  # half-width of a real dental arch at the occlusal plane
                                   # (intermolar width ~55-65 mm). 46 let ramus-contaminated clouds
                                   # set W to 40-46 and the parabola then reached depth 60-80 mm.
ARCH_WIDTH_PCT = 97.0              # percentile of |u| over the centreline that sets W
ARCH_MIN_DEPTH_MM = 8.0            # a flatter "arch" than this is a degenerate fit
ARCH_FALLBACK_DEPTH_MM = 26.0      # canonical arch depth at W, used when the fit is degenerate
# Curvature is bounded at the *parameter*, not by smoothing the sampled curve afterwards. A
# parabola's tightest point is its vertex, where the radius is 1/(2c), so one clamp guarantees the
# whole trough clears the integration half-depth and the normals can never cross inside the slab.
# `limit_curvature` did this by iterated Gaussian smoothing, which works on an arbitrary curve but
# would turn a parabola into something that is no longer a parabola — so the dentition path does
# not call it any more.
# Vertex radius floor. The bound that matters is the one that stops the trough normals
# crossing inside the integration slab, and `drr.adaptive_depth` already ties depth to
# 0.55 * R with a 9 mm floor -- so crossing needs R < 9 mm, not R < MIN_RADIUS_MM (18).
# Clamping at 18 saturated the curvature on 24/24 spot-checked cases and cost coverage
# (F002 1.00 -> 0.86, F026 -> 0.83); the natural fitted vertex radius has a median of
# 11.3 mm.
ARCH_MIN_VERTEX_R_MM = 10.0
ARCH_MAX_VERTEX_R_MM = 44.0
ARCH_MAX_C = 1.0 / (2.0 * ARCH_MIN_VERTEX_R_MM)
ARCH_MIN_C = 1.0 / (2.0 * ARCH_MAX_VERTEX_R_MM)
# `c` is fitted on the anterior/premolar window only. A real dental arch is not a parabola -- it is
# flatter anteriorly and turns harder posteriorly -- so a single quadratic cannot match both ends,
# and a plain least-squares fit over the whole centreline resolves that conflict in favour of the
# posterior (u^2 leverage) and over-curves the front, which distorts exactly the anterior teeth an
# FDI inventory needs. Measured over 14 cases, p90 deviation of the centreline in |u| <= 18 mm:
# fitted anteriorly 3.96 mm, fitted over everything 4.39 mm, and 5.3-6.4 mm if `c` is additionally
# forced above a 14-16 mm vertex radius. The posterior residual it trades away (13.7 mm) is
# absorbed by the +/-14-18 mm integration slab; an anterior one is not.
#
# The weighting is a Gaussian in u rather than a hard window, which measured better at *both* ends
# (16 cases, centreline p90 deviation, anterior | posterior): hard 18 mm 4.37 | 11.66, hard 24 mm
# 4.90 | 10.49, Gaussian sigma 10 mm 4.24 | 10.61, sigma 12-20 mm 4.3-4.5 | 10.5-10.7. A hard cut
# throws the posterior information away entirely and puts a cliff at the window edge; the Gaussian
# keeps it at low weight, so the curvature is still anchored anteriorly without the fit going blind
# past the premolars. The optimum is flat over sigma 10-20 mm.
ARCH_FIT_SIGMA_MM = 12.0


def parabola_points(midline_xy: tuple[float, float], v0: float, c: float, half_width_mm: float,
                    spacing) -> np.ndarray:
    """Sample `v = v0 - c·u²` over `u ∈ [-W, +W]` back onto the canonical (x, y) voxel grid.

    Shared by the fit and by the posterior extension, so the extended trough is *the same
    parabola* over a wider `u` — not a different curve grafted onto the ends."""
    sp = float(spacing[0])
    # ~1 output column per voxel of *arc*, not of u: the parabola's arc grows much faster than its
    # width once it turns posteriorly, and sampling in u under-samples the ramus by ~1.4x, which
    # showed up as the rendered panoramic losing half its width.
    n = max(int(2.0 * half_width_mm * math.hypot(1.0, c * half_width_mm) / sp), 64)
    ug = np.linspace(-half_width_mm, half_width_mm, n)
    vg = v0 - c * ug ** 2
    m = np.asarray(midline_xy, np.float64)
    return (m[None, :] + (ug / sp)[:, None] * E_LR[None, :]
            + (vg / sp)[:, None] * E_AP[None, :]).astype(np.float32)


def fit_arch_symmetric(w: np.ndarray, spacing, slab: np.ndarray,
                       midline_xy: tuple[float, float]) -> ArchCurve | None:
    """Pose a parabola on the canonical axes: `v = v0 - c·u²`, `u = 0` on the mid-sagittal plane.

    Four parameters — vertex x (the midline), vertex y (`v0`), vertex z (the occlusal plane, set by
    the caller) and the spread (half-width `W` plus the curvature `c`). See the section note for
    why nothing else is fitted."""
    pts = arch_centreline(w, spacing)
    if pts is None:
        return None
    sp = float(spacing[0])
    m = np.asarray(midline_xy, np.float64)
    q = pts.astype(np.float64) - m
    u, v = (q @ E_LR) * sp, (q @ E_AP) * sp                 # mm, u = 0 on the mid-sagittal plane
    W = float(np.clip(np.percentile(np.abs(u), ARCH_WIDTH_PCT),
                      ARCH_HALF_WIDTH_MM[0], ARCH_HALF_WIDTH_MM[1]))
    # v = v0 - c·u², c >= 0. v0 is free in sign, so it is split into two non-negative columns.
    # Weighted toward the vertex: u^2 gives the outermost points enormous leverage, so a ramus tail
    # -- or simply the posterior arch, which a parabola cannot match at the same time as the front
    # -- would otherwise set the curvature for the whole trough. See ARCH_FIT_SIGMA_MM.
    sel = np.abs(u) <= W
    u, v = u[sel], v[sel]
    if u.size < 16:
        return None
    wt = np.exp(-0.5 * (u / ARCH_FIT_SIGMA_MM) ** 2)
    A = np.stack([np.ones_like(u), -np.ones_like(u), -(u ** 2)], axis=1) * wt[:, None]
    try:
        from scipy.optimize import nnls
        coef = nnls(A, v * wt)[0]
    except Exception:  # noqa: BLE001 — fall back to an unconstrained solve, then clamp
        coef = np.abs(np.linalg.lstsq(A, v * wt, rcond=None)[0])
    v0, c = float(coef[0] - coef[1]), float(coef[2])
    if c * W ** 2 < ARCH_MIN_DEPTH_MM:
        # Degenerate (a ring cloud, or a cloud with no anterior-posterior extent): fall back to a
        # canonical arch depth rather than rendering a straight line through the jaw.
        c = ARCH_FALLBACK_DEPTH_MM / (W ** 2)
    c = float(np.clip(c, max(ARCH_MIN_C, ARCH_MIN_DEPTH_MM / W ** 2), ARCH_MAX_C))
    out = _arch_from_points(parabola_points(midline_xy, v0, c, W, spacing), spacing, slab,
                            w > 0, None)
    if out is not None:
        out.midline_xy, out.parabola = (float(m[0]), float(m[1])), (v0, c, W)
    return out


def detect_arch_curve_dentition(
    arr: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[ArchCurve | None, np.ndarray | None]:
    """Primary detector: fixed HU thresholds -> enamel occlusal plane -> ramus-free dentition
    cloud -> angular-median parabola -> curvature bound. Returns (curve, cloud); curve is None
    when there is no usable dentition, which the caller degrades from.

    **Thresholds are fixed, and the enamel level is selected over a ladder.** No single level
    works everywhere — 1500 is the best single choice at 120/128, and the union over
    HU_TOOTH_LADDER covers 127/128 — so the residual is a selection problem. Levels are tried in
    order and the first to yield a trough that passes `trough_quality` wins; if none passes, the
    best-scoring one is returned so the caller can still see (and reject) it. This is the same
    score-and-select principle `render.render_case` applies across *detectors*, applied here
    across thresholds.

    **Otsu is the last rung, not a dead branch.** P010 is the one case that passes under
    cascaded Otsu and under no fixed level, so appending it takes the union to 128/128 and makes
    the fallback load-bearing rather than theoretical. It costs nothing on the 127 cases that
    never reach it."""
    soft, scale, on_scale = calibrate(arr)     # once: each call is a histogram over the volume
    # Off this dataset's scale -> the fixed ladder is meaningless, so run the Otsu rung alone.
    rungs = (HU_TOOTH_LADDER + (None,)) if on_scale else (None,)

    best: tuple[float, ArchCurve, np.ndarray] | None = None
    cloud: np.ndarray | None = None
    pose: tuple[tuple[float, float], float] | None = None
    for t_tooth in rungs:                      # None = the Otsu rung, tried last
        if t_tooth is None:
            thr = otsu_thresholds(arr)
        else:
            thr = (hu_to_raw(HU_T_HEAD, soft, scale), hu_to_raw(HU_T_BONE, soft, scale),
                   hu_to_raw(t_tooth, soft, scale))
        z_occ, z_occ2 = occlusal_rows(arr, thr[2], spacing)
        # The midline is a rigid property of the scan, so it is measured once and reused across
        # rungs — it does not depend on which enamel level the ladder is on.
        if pose is None:
            pose = head_midline(arr, spacing, z_occ, thr[1])
        cloud = tooth_cloud(arr, z_occ, thr[2], spacing, z_occ2=z_occ2)
        slab = arr[:, :, max(z_occ - 20, 0):min(z_occ + 20, arr.shape[2])].mean(axis=2)
        # Fit on the enamel *core* where there is one: crowns only, no cortical bone. The full
        # cloud is still what the trough is scored against, so the gate keeps measuring how much
        # of the dentition the panoramic images.
        fit_cloud = cloud
        if on_scale:
            core = tooth_cloud(arr, z_occ, hu_to_raw(HU_TOOTH_CORE, soft, scale), spacing,
                               z_occ2=z_occ2)
            if float((core > 0).sum()) * float(spacing[0]) * float(spacing[1]) >= CORE_MIN_MM2:
                fit_cloud = core
        curve = fit_arch_symmetric(fit_cloud, spacing, slab, pose[0])
        if curve is None:
            continue
        out = curve                      # curvature is bounded at ARCH_MAX_C; no smoothing needed
        out.occlusal_z, out.thresholds = z_occ, thr
        out.midline_xy, out.sym_score = pose
        q = trough_quality(out, cloud, ARCH_SELECT_DEPTH_MM)
        if q["ok"]:
            return out, cloud
        if best is None or q["coverage"] > best[0]:
            best = (q["coverage"], out, cloud)
    if best is None:
        return None, cloud
    return best[1], best[2]


# ── Trough quality ──────────────────

ARCH_MIN_COVERAGE = 0.60    # fraction of the dentition the integration slab must image
ARCH_MIN_CLOUD_MM2 = 400.0  # is there a dentition to image at all? This — not arc length — is
                            # what separates a metal-only mouth from a small real one; an
                            # arc-length floor cannot, because F013 fits a legitimate 57 mm
                            # half-arch, shorter than some junk fits.
                            #
                            # The value is anatomical, not fitted: a molar crown sections at
                            # ~90-110 mm2 at the occlusal plane, a premolar ~50, an incisor ~25,
                            # so 400 mm2 is roughly four molars' worth of dentition.
                            #
                            # **The thin margin was a symptom of Otsu, and fixed thresholds
                            # removed it**. The old
                            # separation was junk at A091 190 / A019 194 / P497 294 against
                            # usable at A071 350 / F061 450 — a 56 mm2 gap. Those "junk" clouds
                            # were starved *by the metal-corrupted t_tooth*, not by the anatomy:
                            # at a fixed 1500 they measure 453 / 908 / 1537 mm2 and render as
                            # real arches. The smallest cloud over 128 cases is now 268 mm2
                            # (P526) and it is also real.
                            #
                            # The value therefore stays at its anatomical justification rather
                            # than being re-fitted. It no longer has to separate metal-only
                            # clouds — there are none — and the cases it still rejects (P526,
                            # A071) are recovered by the HU_TOOTH_LADDER sweep instead of
                            # falling to the flat MIP.
                            #
                            # Note the gate is *not* trivially satisfiable by descending the
                            # ladder: a lower threshold inflates the cloud with cortical bone,
                            # which the parabola cannot cover, so `coverage` falls (fixed 900
                            # scores 93/128 against 1500's 120/128). Cloud size and coverage
                            # guard opposite failure directions and both are load-bearing.
ARCH_MIN_LENGTH_MM = 35.0   # degenerate-fit guard only (A091 fits 5 mm). Measured on the
                            # *fitted* curve, before the posterior extension — measuring after
                            # it is what let A019's 41 mm sliver report 90 mm and pass.
ARCH_MIN_SPAN_DEG = 120.0   # angular span about the trough centroid: a dental arch subtends
                            # ~180 deg, a vertebral column or a pair of restorations far less.
ARCH_FAST_PATH_COVERAGE = 0.85   # a passing dentition trough this good is taken without also
                                 # running the (much slower) polar detector to compare
ARCH_SELECT_DEPTH_MM = 14.0      # slab half-depth the HU_TOOTH_LADDER selection scores against.
                                 # Mirrors drr.DRR_DEPTH_MM, duplicated because drr imports from
                                 # render which imports this module — importing it back would be
                                 # circular. Keep the two in sync.
ARCH_SWITCH_MARGIN = 0.05        # coverage the polar trough must beat the dentition one by
                                 # before it is preferred. The dentition detector is the
                                 # validated primary (val_ood 21/63 -> 60/63), so ties keep it.


def trough_quality(curve: ArchCurve, cloud: np.ndarray, depth_mm: float) -> dict:
    """Score a trough against the dentition it is supposed to image.

    ``coverage`` is the fraction of dentition mass within ``depth_mm`` of the curve. It is
    necessary but **not sufficient**, and that was learned the hard way: it scores the trough
    against the *cloud*, not the cloud against the anatomy, so when the cloud itself is wrong
    the metric is happily high — F028 rendered a column of cervical vertebrae at 0.95, A019 a
    sliver of two restorations at 0.99. ``cloud_mm2`` and ``span_deg`` are the anatomy-anchored
    companions that catch exactly those two failures.

    Note ``cloud_mm2`` is a property of the *cloud*, so it is identical for every candidate
    trough scored against it — deliberately: it asks whether there is a dentition to image at
    all, which no choice of curve can fix."""
    P = curve.points
    sp = float(curve.spacing[0])
    rail = np.zeros(cloud.shape, bool)
    rail[np.clip(np.round(P[:, 0]).astype(int), 0, cloud.shape[0] - 1),
         np.clip(np.round(P[:, 1]).astype(int), 0, cloud.shape[1] - 1)] = True
    dist = ndimage.distance_transform_edt(~rail) * sp
    tot = float(cloud.sum())
    cov = float(cloud[dist <= depth_mm].sum() / tot) if tot > 0 else 0.0

    c = P.mean(axis=0)
    th = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    span = math.degrees(float(np.abs(th[-1] - th[0])))
    cloud_mm2 = float((cloud > 0).sum()) * sp * float(curve.spacing[1])

    return {"coverage": round(cov, 3), "length_mm": round(curve.length_mm, 1),
            "span_deg": round(span, 1), "cloud_mm2": round(cloud_mm2),
            "ok": bool(cov >= ARCH_MIN_COVERAGE and cloud_mm2 >= ARCH_MIN_CLOUD_MM2
                       and curve.length_mm >= ARCH_MIN_LENGTH_MM
                       and span >= ARCH_MIN_SPAN_DEG)}
