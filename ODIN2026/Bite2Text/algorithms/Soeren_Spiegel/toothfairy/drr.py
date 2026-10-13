"""Pseudo-panoramic radiograph via acquisition simulation.

Where ``render.panoramic_curved`` *reformats* the arch (a ±12 mm mean ray-average — a
tomographic slice), this simulates the *radiograph*: a Beer–Lambert line integral of a
bone-emphasised pseudo-attenuation along the buccolingual beam, read out column-by-column
along the dental arch. The output reads as a dental OPG (bone bright, semi-transparent
superimposition) instead of a CT reformat — the bet being domain-match for the frozen VLM.

Design decisions:
  - **Legibility over realism** → Tier 1: parallel beam, *limited* buccolingual depth (no
    full-head ghosting). No orbiting-source magnification yet.
  - **Replace** ``panoramic_curved`` as the panoramic still.
  - **Full jaw height** (no occlusal ± crop) and the arch extended posteriorly, so the
    rami/last molars are not cut.

Three things changed in v2, all measured over 128 cases:

**Tone.** Tier 1 normalised twice, both times against image *content* — p99.5 of the volume,
then a p2/p99.5 window of the rendered panoramic. A row of implants covers well over 0.5 % of
the frame, so it owned the white point and crushed everything else: **121/128 frames had
saturated highlights**. v2 renders a bounded transmittance ``1 - exp(-tau)``, which fixed that
(saturated frames 121/128 → 2/128, band p95 IQR 33 → 15) — but it read the attenuation off the
per-volume **Otsu ladder**, and that half was a regression: metal drags the ladder
(corr(metal_frac, t_bone) = 0.73) and restored mouths rendered black. v3 anchors the same
transmittance on **fixed tissue landmarks** instead, so nothing per-volume enters the tone map
at all. See the attenuation-model header below

**Posterior extension.** Tier 1 walked each end along its straight tangent until it left the
*volume*. The length was therefore set by how much air the scanner captured (F001: 304 mm of
arc for 101 mm of dentition), and the direction was worse — the parabola's end tangents point
postero-*medially*, so the two extensions converged behind the jaw and drove straight through
the cervical vertebral bodies, which is why the spine appeared *lateral* to the dental arch
instead of behind it. `flared_extend` replaces it. Dead arc p90 58 % → 43 %.

**Depth.** A constant ±14 mm slab is wrong at both ends of the trough: anteriorly the incisor
curvature is tighter than 14 mm so the normals cross inside the slab and the front teeth fan,
posteriorly it is too thin for a molar row the parabola only approximates. Depth now follows
the local radius of curvature, and the ray is **averaged** rather than summed so a wider slab
does not read brighter.

``drr._trough_follow`` is gone. It translated the whole trough by a per-Z
(dx, dy) of up to ±10 mm, so every output row was sampled at a different (x, y) and every
vertical structure — tooth axes, cortical borders, the skin outline — was sheared into an S.
It could not be tuned away, because a rigid translation cannot represent two arches that differ
in *shape*, which is what it was introduced to do. The measured need for it is also small: the
buccolingual offset between the two tooth rows exceeds 8 mm in only 6/128 cases.

Input ``arr`` MUST be the canonical (L, A, S) isotropic grid (volume_io.load_canonical_volume).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .arch import HU_METAL as HU_METAL_RAW
from .arch import ArchCurve, hu_thresholds, intensity_calibration
from .config import FRAME_TILE
from .render import PANO_MAX_WIDTH, _dental_zrange, _sample
from .volume_io import to_wide_rgb

# ── Attenuation model ───────────────────────────────────────────────────────────
# Pseudo-attenuation is anchored on **fixed tissue landmarks** in the dataset's HU-like raw
# units. No per-volume quantity enters the tone map, so a
# given tissue class renders to the same gray in every case *by construction* and metal cannot
# move the scale at all.
#
# **This replaces an Otsu ladder x = (v - t_head)/(t_bone - t_head) with a fixed ENAMEL_X = 2,
# which was a regression on exactly the cases it was meant to fix.** That design argued the two
# low thresholds were "metal-free by construction, decided by the bulk of the histogram".
# Measured over 128 volumes they are not: corr(metal_frac, t_bone) = 0.73, and on a heavily
# restored mouth t_head/t_bone are dragged to +94/1748 against a typical -323/779. Real enamel
# (~2200) then lands at x = 0.72-0.90 instead of 2.0, where the map returns BONE_SIGMA * x
# ~ 0.08 rather than 1.0 — a ~12x under-attenuation of enamel and bone. Dentition-band medians
# on plots/survey_v2.png: A019 3, P497 0, P544 4, A045 10, S0000 24 out of 255 — black frames
# whose *geometry* was fine. It survived review because v2's tone metrics were saturation and
# p95, deliberately not the band median, so nothing watched the axis it broke.
HU_SOFT = 0.0          # soft-tissue mode; measured +10..+130 on all 128 cached volumes
HU_BONE = 300.0        # bone onset: mu = BONE_SIGMA here
HU_ENAMEL = 2000.0     # enamel: mu = 1.0 here
BONE_SIGMA = 0.10      # sigma at the soft|bone boundary: soft tissue stays nearly transparent
MU_CAP = 1.25          # clip sigma just above enamel: metal must not set the scale
METAL_TRIM_Q = 0.90    # per-ray winsorising quantile; see `trim_metal` in `pseudo_opg`
METAL_HALO_MM = 5.0    # radius around metal whose streak artefacts are clamped, see `damp_metal`
METAL_HALO_HU = 1000.0 # pseudo-HU the halo is clamped to (cortical bone), so streaks stop reading
                       # as enamel while the bone that is really there still shows
DEPTH_REF_MM = 28.0    # nominal path length the averaged sigma is scaled by
TAU0_MM = 5.4          # transmittance scale: sets where mid-gray lands. One global constant.
                       # Re-swept after `trim_metal` went in: winsorising each ray removes real
                       # attenuation as well as streaks, which cost ~8/255 of band median (6.0 ->
                       # median 94.5, 5.4 -> 102.5, 5.0 -> 108.5 over 24 stratified cases).

# ── Focal trough ────────────────────────────────────────────────────────────────
DRR_DEPTH_MM  = 14.0   # nominal buccolingual half-depth (fixed-depth path, and the quality gate)
DEPTH_K       = 0.55   # adaptive depth as a fraction of the local radius of curvature
DEPTH_MIN_MM  = 9.0    # anterior: below the incisor curvature radius, so normals cannot cross
DEPTH_MAX_MM  = 18.0   # posterior: thick enough for a molar row the parabola approximates
# The extension is driven by a *target total arc*, not by a fixed length per side, and the two
# sides are balanced about the head's mid-sagittal plane. A fixed 38 mm per side made the output
# width a function of how long the fit happened to be — and since the old degree-2 fit was 26 %
# too short at the median, ~12 of 48 survey cases rendered a 40-60 mm arc against a real OPG's
# 150-180 mm. Balancing about the midline is what makes
# the frame read like the coronal view it is derived from: without it, a trough whose fit ran
# further on one side puts the midline off-centre and the panoramic looks rotated even when the
# geometry is right.
TARGET_ARC_MM = 170.0  # a real OPG spans ~150-180 mm end to end
EXTEND_MM     = 38.0   # per-side extension when no target is used (the ablation path)
EXTEND_MAX_MM = 45.0   # cap per side. Past this the arc leaves the ramus and enters the skull
                       # base, and — more to the point — every millimetre of extension is a
                       # constant-curvature *guess*, so a short fit must not be papered over with
                       # 60 mm of invented geometry (S0020's 67 mm fit is the case that shows it).
MAX_ARC_MM    = 200.0  # hard cap
FLARE_R_MM    = 34.0   # radius the extension curves *outward* on; see `flared_extend`
N_DEPTH_SAMPLES = 33   # rays per column, fixed so per-column depth does not change the sampling
# Vertical framing. `full_height=True` rendered the entire volume Z, which on a tall FOV spends most
# of the frame on cranium and neck and pushes the maxilla to the very top edge (survey rows 4 and 8).
# The band is asymmetric on purpose: superiorly it has to keep the maxillary sinus, which the reports
# comment on; inferiorly it only has to keep the mandibular border.
BAND_SUP_MM = 42.0
BAND_INF_MM = 45.0


def jaw_zband(z_occ: int | None, nz: int, spacing) -> tuple[int, int]:
    """Voxel [z0, z1) of the OPG band: occlusal + BAND_SUP_MM / - BAND_INF_MM, clamped."""
    if z_occ is None:
        return 0, nz
    spz = float(spacing[2])
    return (max(int(round(z_occ - BAND_INF_MM / spz)), 0),
            min(int(round(z_occ + BAND_SUP_MM / spz)) + 1, nz))


def damp_metal(arr: np.ndarray, spacing, calib: tuple[float, float],
               halo_mm: float = METAL_HALO_MM) -> np.ndarray:
    """Clamp the streak halo around metal to cortical bone, returning a corrected volume.

    Restorations and implants throw bright in-plane streak artefacts. Those streaks sit *between*
    bone and metal in value, so `MU_CAP` does not touch them, and averaging them along a 36 mm
    buccolingual ray spreads them over the neighbouring columns and rows — the aura reported on
    F001, F002, F003, F012, S0000. Winsorising each ray does not help, because the halo is not on
    the rays that pass through the metal; it is in the reconstruction, next to it.

    This is content-dependent, which the tone map otherwise avoids on purpose. The distinction is
    the same one draws: it corrects a known *acquisition*
    artefact in a region identified by a fixed threshold, rather than normalising to image
    content. Metal itself is left alone — a restoration should read as opaque."""
    soft, scale = calib
    t_metal = soft + HU_METAL_RAW * (scale / 1000.0)
    m = arr[::2, ::2, ::2] > t_metal
    if not m.any():
        return arr
    it = max(int(round(halo_mm / (2.0 * float(spacing[0])))), 1)
    halo = ndimage.binary_dilation(m, iterations=it)
    halo = np.repeat(np.repeat(np.repeat(halo, 2, 0), 2, 1), 2, 2)[
        :arr.shape[0], :arr.shape[1], :arr.shape[2]]
    if halo.shape != arr.shape:                      # odd dimension: pad by edge replication
        pad = [(0, arr.shape[i] - halo.shape[i]) for i in range(3)]
        halo = np.pad(halo, pad, mode="edge")
    cap = soft + METAL_HALO_HU * (scale / 1000.0)
    out = arr.copy()
    sel = halo & (arr > cap) & (arr <= t_metal)       # streaks only; real metal is untouched
    out[sel] = cap
    return out


def pseudo_mu(vals: np.ndarray, calib: tuple[float, float] = (0.0, 1000.0)) -> np.ndarray:
    """Raw CBCT intensity -> bounded pseudo-attenuation, identical per tissue class on every
    volume. Piecewise linear through (HU_SOFT, 0), (HU_BONE, BONE_SIGMA), (HU_ENAMEL, 1), then
    linear to the cap so restorations stay distinguishable from enamel without being able to set
    the scale.

    ``calib`` is ``arch.intensity_calibration``'s (soft, scale): the two-point affine that puts
    this volume's air mode at -1000 and its soft tissue at 0. It is a *calibration*, measured
    from two anchors metal cannot move, not a content-dependent normalisation — which is the
    distinction that matters, since content-dependent normalisation is what broke v1 (percentile
    windows owned by implants) and v2 (an Otsu ladder dragged by metal)."""
    soft, scale = calib
    hu = (vals - soft) * (1000.0 / max(scale, 1e-6))
    s1 = BONE_SIGMA / (HU_BONE - HU_SOFT)
    s2 = (1.0 - BONE_SIGMA) / (HU_ENAMEL - HU_BONE)
    mu = np.where(hu < HU_BONE, s1 * (hu - HU_SOFT), BONE_SIGMA + s2 * (hu - HU_BONE))
    return np.clip(mu, 0.0, MU_CAP)


def _curvature_radius_mm(P: np.ndarray, sp: float) -> np.ndarray:
    """Local radius of curvature of the trough, in mm (large on straight runs)."""
    d1 = np.gradient(P, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(np.hypot(d1[:, 0], d1[:, 1]), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(num > 1e-9, den / np.maximum(num, 1e-9), 1e6)
    return ndimage.gaussian_filter1d(np.minimum(r, 1e6), 8.0, mode="nearest") * sp


def _parabola_extend(curve: ArchCurve, target_arc_mm: float,
                     max_ext_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Continue the *same* parabola to a wider half-width until the arc reaches the target.

    The arch is one parabola, so its posterior extension must be that parabola over a larger |u|,
    not a constant-radius arc grafted onto the ends — that graft is what made the trough read as a
    quartic (A091) and put a curvature discontinuity at the last molar."""
    from .arch import parabola_points
    v0, c, W = curve.parabola
    sp = float(curve.spacing[0])

    def arc_of(half_w: float) -> float:
        u = np.linspace(0.0, half_w, 512)
        return 2.0 * float(np.trapezoid(np.sqrt(1.0 + (2.0 * c * u) ** 2), u))

    hi = W + max_ext_mm
    if arc_of(W) >= target_arc_mm:
        half_w = W
    elif arc_of(hi) <= target_arc_mm:
        half_w = hi
    else:
        lo = W
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if arc_of(mid) < target_arc_mm:
                lo = mid
            else:
                hi = mid
        half_w = 0.5 * (lo + hi)
    P = parabola_points(curve.midline_xy, v0, c, half_w, curve.spacing)
    t = np.gradient(P.astype(np.float64), axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn == 0] = 1.0
    t /= tn[:, None]
    N = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return P.astype(np.float32), N.astype(np.float32)


def _balanced_extensions(curve: ArchCurve, target_arc_mm: float) -> tuple[float, float]:
    """(pre_mm, post_mm): how far to extend each end so the frame is centred and OPG-sized.

    The midline splits the fitted arc into a right and a left half. Extend each side so the two
    halves come out equal and the total reaches ``target_arc_mm``, capped at ``EXTEND_MAX_MM``
    per side. With no midline available (legacy detectors) both sides get the same length, which
    reduces to the old fixed-extension behaviour."""
    sp = float(curve.spacing[0])
    P = curve.points
    need = max(target_arc_mm - curve.length_mm, 0.0)
    if curve.midline_xy is None:
        half = min(0.5 * need, EXTEND_MAX_MM)
        return half, half
    u = P[:, 0] - float(curve.midline_xy[0])        # signed L-R offset from the mid-sagittal plane
    k = int(np.argmin(np.abs(u)))                   # where the trough crosses it
    seg = np.hypot(*np.diff(P, axis=0).T) * sp
    pre_arc, post_arc = float(seg[:k].sum()), float(seg[k:].sum())
    # want (pre_arc + pre) == (post_arc + post) and pre + post == need
    pre = float(np.clip(0.5 * (need + post_arc - pre_arc), 0.0, min(need, EXTEND_MAX_MM)))
    post = float(np.clip(need - pre, 0.0, EXTEND_MAX_MM))
    return pre, post


def flared_extend(
    curve: ArchCurve,
    extend_mm: float | None = None,
    flare_r_mm: float = FLARE_R_MM,
    max_arc_mm: float = MAX_ARC_MM,
    target_arc_mm: float = TARGET_ARC_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the trough past the last molar the way a real focal trough does: a bounded arc
    that curves **outward** toward the condyle.

    Each end is continued as a constant-curvature arc bending away from the arch centroid, which
    tracks the ramus and never points at the midline. The *length* is set by how much arc is
    still needed to reach ``target_arc_mm``, split so the midline ends up in the middle of the
    frame — pass ``extend_mm`` to force a fixed length per side instead (the ablation path). See
    the module docstring for what this replaces and why."""
    # A parabola-posed trough extends as itself; the flared arc below is the legacy path for the
    # polar/spline detectors, which do not carry a shape model.
    if extend_mm is None and curve.parabola is not None and curve.midline_xy is not None:
        return _parabola_extend(curve, target_arc_mm, EXTEND_MAX_MM)
    sp = float(curve.spacing[0])
    P0, N0 = curve.points, curve.normals
    centre = P0.mean(axis=0)
    pre_mm, post_mm = ((extend_mm, extend_mm) if extend_mm is not None
                       else _balanced_extensions(curve, target_arc_mm))

    def sense(anchor: np.ndarray, tang: np.ndarray) -> float:
        """Turn direction that moves away from the arch centroid."""
        a = np.array([-tang[1], tang[0]]) + anchor
        b = np.array([tang[1], -tang[0]]) + anchor
        return 1.0 if np.hypot(*(a - centre)) > np.hypot(*(b - centre)) else -1.0

    def arc(anchor: np.ndarray, tang: np.ndarray, nrm: np.ndarray, length_mm: float):
        out_n = nrm if np.dot(nrm, anchor - centre) > 0 else -nrm
        dtheta = sense(anchor, tang) * (sp / flare_r_mm)
        t, p = tang.copy(), anchor.copy()
        pts, nrms = [], []
        for _ in range(max(int(length_mm / sp), 0)):
            c, s = np.cos(dtheta), np.sin(dtheta)
            t = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
            p = p + t
            nn = np.array([-t[1], t[0]])
            pts.append(p.copy())
            nrms.append(nn if np.dot(nn, out_n) > 0 else -nn)
        return np.array(pts, np.float32), np.array(nrms, np.float32)

    t0 = -(P0[1] - P0[0]);  t0 = t0 / max(float(np.hypot(*t0)), 1e-6)   # outward at the start
    t1 = (P0[-1] - P0[-2]); t1 = t1 / max(float(np.hypot(*t1)), 1e-6)
    pre_p, pre_n = arc(P0[0], t0, N0[0], pre_mm)
    post_p, post_n = arc(P0[-1], t1, N0[-1], post_mm)
    parts_p = [x for x in (pre_p[::-1] if len(pre_p) else pre_p, P0, post_p) if len(x)]
    parts_n = [x for x in (pre_n[::-1] if len(pre_n) else pre_n, N0, post_n) if len(x)]
    P = np.concatenate(parts_p, axis=0)
    N = np.concatenate(parts_n, axis=0)

    if float(np.hypot(*np.diff(P, axis=0).T).sum() * sp) > max_arc_mm:
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))] * sp
        keep = np.abs(s - 0.5 * (s[0] + s[-1])) <= 0.5 * max_arc_mm    # trim about the centre
        P, N = P[keep], N[keep]
    return P, N


def pseudo_opg(
    arr: np.ndarray,
    curve: ArchCurve,
    height: int = FRAME_TILE,
    extend_arch: bool = True,
    full_height: bool = True,
    adaptive_depth: bool = True,
    trim_metal: bool = True,
    damp_halo: bool = True,
    depth_mm: float = DRR_DEPTH_MM,
    thresholds: tuple[float, float, float] | None = None,
    calib: tuple[float, float] | None = None,
) -> Image.Image:
    """Pseudo-OPG: transmittance of a bone-emphasis line integral along the arch.

    For each arc position (output column) and each Z (output row), average ``pseudo_mu`` over
    the buccolingual normal, scale to an optical depth and read out ``1 - exp(-tau)``. Bone
    bright, soft tissue transparent, output [Z (superior up) × arc] and bounded in [0, 1] with
    no display window.

    ``thresholds`` defaults to ``curve.thresholds`` (set by ``arch.detect_arch_curve_dentition``)
    and is recomputed only if neither is available — the legacy detectors do not carry it.
    ``calib`` is the (soft, scale) affine the tone map is expressed in; measured here when not
    supplied. It is deliberately *not* derived from ``thresholds``, which may have come from the
    Otsu fallback and would then invert to a meaningless soft-tissue level."""
    nz = arr.shape[2]
    spacing = curve.spacing
    sp = float(spacing[0])
    z0, z1 = jaw_zband(curve.occlusal_z, nz, spacing) if full_height else _dental_zrange(curve, nz)
    thr = thresholds or curve.thresholds or hu_thresholds(arr)
    cal = calib or intensity_calibration(arr)
    if damp_halo:
        arr = damp_metal(arr, spacing, cal)

    P, N = flared_extend(curve) if extend_arch else (curve.points, curve.normals)

    if adaptive_depth:
        depth = np.clip(DEPTH_K * _curvature_radius_mm(P, sp), DEPTH_MIN_MM, DEPTH_MAX_MM)
    else:
        depth = np.full(len(P), float(depth_mm))
    off = (depth[:, None] / sp) * np.linspace(-1.0, 1.0, N_DEPTH_SAMPLES)[None, :]  # (M, nd)
    xs = (P[:, 0:1] + off * N[:, 0:1])[:, :, None]
    ys = (P[:, 1:2] + off * N[:, 1:2])[:, :, None]
    zs = np.arange(z0, z1, dtype=np.float32)[None, None, :]

    # Outside the volume reads as air, never as bone — with a percentile cval the border used
    # to contribute attenuation and lift the frame edges. Clamped to <= HU_SOFT so this still
    # holds on the Otsu fallback path, where a metal-corrupted t_head can be as high as +909
    # (that would map to mu ~ 0.42 and light up the frame edges).
    vals = _sample(arr, xs, ys, zs, min(float(thr[0]), cal[0]))   # (M, nd, nzb)
    mu = pseudo_mu(vals, cal)
    if trim_metal:
        # Metal aura. A restoration and its streak artefacts sit at the MU_CAP on a handful of the
        # 33 samples in a ray, and because the ray is *averaged* that handful lifts every
        # neighbouring column and row -- the halo reported on S0000, F001, F003, F012. Winsorising
        # the ray at a high quantile drops the few extreme samples without touching a tissue that
        # actually occupies the beam: a real 2 mm enamel layer spans ~5 of 33 samples, so the cap is
        # set above that fraction.
        hi = np.quantile(mu, METAL_TRIM_Q, axis=1, keepdims=True)
        mu = np.minimum(mu, hi)
    tau = mu.mean(axis=1) * (DEPTH_REF_MM / TAU0_MM)
    pano = (1.0 - np.exp(-tau)).T[::-1, :]                        # (nzb, M), superior up
    u8 = (np.clip(pano, 0.0, 1.0) * 255.0).astype(np.uint8)

    return to_wide_rgb(u8, height=min(z1 - z0, 4 * height), max_width=PANO_MAX_WIDTH)
