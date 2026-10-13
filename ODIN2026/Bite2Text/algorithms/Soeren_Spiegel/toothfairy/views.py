"""Five views per case — coverage by construction, not by one trough being right.

Entry point: `five_views`.

The shipped representation is a single pseudo-OPG built on a single planar trough, and on an
open bite that trough loses the anterior mandible.

The mechanism, measured: on F028 and F001 the anterior mandibular **bone** sits a median 24 mm
lingual of the trough while the posterior sits within 2 mm. The front of the frame is therefore
imaging lip and vestibule — the dark gap — with the symphysis smeared along the slab edge under
it. The cause is upstream of the curve: those jaws have no erupted anterior crowns, so the
enamel cloud has nothing in the anterior sextant and the parabola's vertex is *extrapolated*
from the molars, landing well labial of the real symphysis. Fitting that jaw to bone instead
(`anterior_starved`) recovers it.

Two things this is NOT, both checked: it is not the *shape* of the curve (fixed depths of 8 and
20 mm, the un-extended arch, and per-jaw single-band troughs all leave the hole), and it is not
the Z crop (`enamel_above_band` is 0.0% on 7 of 8 cases; rendering the whole volume Z changes
nothing). An earlier version of this note claimed capture was 0.97-1.00 and the geometry
therefore innocent — that measurement counted *enamel* voxels, which anteriorly in these cases
are maxillary incisor tips dipping below the split plane, not mandibular teeth. Measured on
bone it is 24 mm off.

A third thing it is not: a fixable rendering fault on P216 and S0020, whose maxillae look cut.
Both reports say so themselves — P216 "Maxilla: not included in the scan volume", S0020
"Maxilla: partially included in the scan volume" from both annotators. The band discards none
of it (`enamel_above_band` 0.0%); the scanner never acquired it. Those cases are the model's
"not assessable" answer, not a geometry problem.

DEFAULT_VIEWS — the shipped set:

  1. ``pano_union``    the pseudo-OPG on the union trough — the global map, both rows in one
                       frame, the view that matches what a VLM saw in pretraining. Grayscale
                       transmittance (Beer-Lambert, metal-halo damped).
  2. ``pano_mandible`` OPG on a trough fitted to the mandibular dentition alone, from a
                       Z-projected cloud so a dropped or rotated jaw still contributes. This is
                       the one that answers "what if the two arches disagree".
  3. ``pano_rgb``      the same curved slab as 1, read out through hu.RGB_WINDOWS instead of a
                       transmittance: R soft/lucent, G dentin/periodontal/IAN, B cortical/
                       enamel/restoration. Pixel-aligned with 1, so they read together.
  4. ``occlusal_rgb``  axial projection per jaw, side by side, RGB-windowed. **No trough, no
                       fit** — every crown in the arch is in it by construction, so this is the
                       view that cannot have a trough-induced hole.
  5. ``mpr_rgb``       mid-sagittal + coronal thick-MPR, RGB-windowed. The Z story the
                       panoramics flatten away: bite opening, overbite, curve of Spee, and
                       whether the FOV cut the condyles or the upper maxilla.

``pano_maxilla``, ``hemiarch``, and the grayscale ``occlusal`` / ``mpr`` are built but off by
default — see the DEFAULT_VIEWS block below for the measurements that decided it.

Run:  python -m toothfairy.views --cases F028 F001 P216 A044 S0020
      python -m toothfairy.views --views pano_rgb occlusal_rgb mpr_rgb   # the RGB-only arm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .arch import (
    ARCH_GAP_MM, SEED_CANDIDATES, ArchCurve, detect_arch_curve_dentition, fit_arch_symmetric,
    occlusal_rows,
)
from .config import FRAME_TILE, REPO, VOLUME_CACHE_DIR
from .drr import (
    DEPTH_K, DEPTH_MAX_MM, DEPTH_MIN_MM, N_DEPTH_SAMPLES, _curvature_radius_mm,
    flared_extend, jaw_zband, pseudo_opg,
)
from .hu import AIR_REF, RGB_WINDOWS, calibrate_to_hu
from .render import _sample
from .plotting import tile_sheet
from .volume_io import load_canonical_volume, percentile_window, to_wide_rgb, window_to_uint8

SAG_HALF_MM = 4.0      # half-thickness of the mid-sagittal thick-MPR slab
COR_HALF_MM = 25.0     # half-thickness of the coronal slab, about the arch's A-P centre


# ── RGB windowing (toothfairy.hu / baseline_hu_v1 Task 2) ────────────────────────
#
# The grayscale views spend their whole dynamic range separating bone from air, which is the
# one distinction a dental report almost never turns on. RGB_WINDOWS spends it on three
# *pathology* axes instead, on the calibrated HU scale:
#
#   R  L=150  W=1400  [-550, +850]   soft tissue, sinus mucosa, cysts, periapical lucencies
#   G  L=1100 W=2200  [   0, +2200]  the workhorse: cancellous->dentin->cortical gradient,
#                                    root morphology, alveolar bone level, the IAN canal wall
#   B  L=2200 W=2600  [+900, +3500]  cortical bone, enamel, crowns/implants/metal
#
# So a periapical lucency that is "slightly darker grey" in the panoramic is a red-channel
# feature here, and bone level — the thing periodontal findings are graded on — is a green edge
# rather than a texture gradient. The three channels are the same voxels, not new data; what
# changes is which contrasts survive the 8-bit quantisation.
#
# Calibration is per volume and costs a histogram pass, so it is computed once per case and
# threaded through. CBCT is not HU-calibrated out of the scanner; `hu.calibrate_to_hu` fits the
# (air, soft) -> (-1000, 0) affine that makes these fixed windows transfer across families.
def _rgb_wide(rgb_u8: np.ndarray, height: int, max_width: int = 1568) -> Image.Image:
    """Resize an HxWx3 uint8 image to ``height``, aspect kept, both sides multiples of 28.

    ⚠ ``max_width`` is a hard clamp, not a scale: once it binds, the aspect ratio is silently
    wrong (a 2.7:1 panoramic rendered at 1568 x 728 is squashed to 2.15:1). It must therefore
    rise with ``height`` -- see MAX_WIDTH_V2 -- and stay under the processor's 1.64 Mpx budget
    or the whole frame is downsampled again on the other side."""
    img = Image.fromarray(rgb_u8, mode="RGB")
    aspect = img.width / max(img.height, 1)
    h = max(28, int(round(height / 28)) * 28)
    w = int(round(h * aspect))
    if w > max_width:
        # Give up HEIGHT, not aspect. Clamping width alone silently squashes the frame -- a
        # 3.6:1 panoramic asked for at 728 px tall becomes 3.0:1, i.e. the anatomy is stretched
        # vertically by 20% relative to every other case, which is exactly the kind of
        # inconsistency the fixed-mm/px framing work was for.
        w = max_width
        h = max(28, int(round(w / max(aspect, 1e-6))))
    w = max(28, int(round(w / 28)) * 28)
    h = max(28, int(round(h / 28)) * 28)
    return img.resize((w, h), Image.LANCZOS)


def _window01(hu, level: float, width: float):
    """One (level, width) window applied to a HU array -> float32 in [0, 1]."""
    lo = level - width / 2.0
    return np.clip((hu.astype(np.float32) - lo) / width, 0.0, 1.0)


STRETCH_PCT = (1.0, 99.5)


def _stack_u8(chans: list[np.ndarray], stretch: bool = True) -> np.ndarray:
    """Stack three [0,1] channels to uint8 RGB, each stretched to its own percentile range.

    The stretch is what makes these frames legible rather than amber. Per *voxel* the windows
    are well behaved — R clips on only 2.6-6.6% of voxels, B on 0.0% — but a projection averages
    33 samples, and a channel's mean rises with how much of the ray sits above its band's floor.
    R's floor is -550 HU, so every ray through tissue averages high; B's is +900, so only enamel
    and metal lift it. The result is R >> G >> B everywhere and the frame reads orange even
    though no channel is individually saturated.

    Stretching each projected channel over ``STRETCH_PCT`` restores its own dynamic range. This
    trades the absolute HU meaning of a pixel for per-channel contrast — deliberate: after a
    ray-average the pixel is not a tissue's HU anyway, it is a path average, and what the model
    needs from the channel is which structures differ within it. The *slice*-based views in
    baseline_hu_v1 keep the unstretched absolute mapping, where it is still meaningful."""
    out = []
    for c in chans:
        if stretch:
            lo, hi = np.percentile(c, STRETCH_PCT)
            c = np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out.append((c * 255.0).astype(np.uint8))
    return np.stack(out, axis=-1)


# **Window first, then project.** The obvious order — project to a 2-D image, then apply the
# three windows — is wrong here and looked it: every channel came out saturated and the frames
# rendered amber. A ray-mean over an 18-36 mm slab, or a MIP over a whole jaw, lands at a HU
# well above the R band's ceiling (+850) and near the G band's (+2200), so R and G both clip to
# 255 and only B carries structure. RGB_WINDOWS was validated on axial *slices*, where each
# pixel is one voxel's HU; a projection is not on that scale.
#
# Windowing the volume first and projecting each channel separately keeps each channel's
# reduction inside the tissue class that channel is for: the R projection is of a soft-tissue
# contrast field, the B projection of an enamel/metal one. Costs three reductions instead of
# one, and for the panoramic the expensive part (trilinear sampling of the curved slab) is
# still done once and shared.
# ── Reduction along the ray ──────────────────────────────────────────────────────
#
# `mean` is the original: it is a projection, it looks like a radiograph, and it is what makes
# soft tissue and cancellous bone legible. It is also where restorations go. Measured over the
# 62 val cases, classing each ray by
# whether it contains a voxel above 2500 HU (a dense restoration) against rays holding only
# enamel, and reading the FINAL 8-bit pixel:
#
#     channel   mean (shipped)   max (MIP)
#     R           +7.8            +0.0     <- both classes clip R's ceiling under a max
#     G          +14.5            +0.0     <- and G's
#     B          +48.2           +84.4     <- the enamel/restoration window
#     d' in B     0.94            4.51
#
# So a full MIP wins in B and *destroys* R and G: a maximum over a 36 mm ray is above the R and
# G window ceilings almost everywhere, which is the white-silhouette failure the grayscale
# occlusal view has. `hybrid` takes the reduction each window was designed for -- mean for the
# two soft/medium windows, max for the dense one -- so the frame keeps its tissue context and
# gains the 4.8x separability in the one channel that carries crowns, fillings, posts and
# implants. Same three channels, same geometry, same token count, one line of code.
REDUCTIONS = ("mean", "max", "hybrid")

# A window designed for a MEAN is the wrong window for a MAX. RGB_WINDOWS[2] spans [900, 3500]
# because a 36 mm ray *average* through the dentition lands low in it; take the ray's maximum
# through the same band and plain enamel already renders at 151/255, so a titanium implant at
# 255 has nothing to stand out against -- it is numerically brighter and visually invisible,
# which is what "the implants disappeared from pano_rgb" looks like. Re-banding to start at the
# 2100 HU enamel shoulder restores the contrast (measured on F038, B channel, max reduction):
#
#     window            enamel   dense restoration   metal    metal - enamel
#     [900, 3500]        151.0        227.6          255.0        +104
#     [2100, 3200]        26.3        194.8          255.0        +229
#
# The band deliberately starts BELOW the 3000 HU metal constant and overlaps the enamel range:
# F045 and P379 are clipped at reconstruction and hold zero voxels above 3000 (Representation.md
# item 4), so anything hung on a single hard cutoff reads "no metal" on those two cases.
B_MAX_WINDOW = (2650.0, 1100.0)   # (level, width) -> [2100, 3200] on the calibrated HU scale


def _reduce_rgb(vals: np.ndarray, axis: int, reduce: str = "hybrid") -> list[np.ndarray]:
    """Window `vals` through RGB_WINDOWS and reduce each channel along `axis`.

    ⚠ Window FIRST, reduce second. The other order is a different and much worse render: a
    36 mm ray mean lands near +250 HU, far below the B window's +900 HU floor, so B would be
    identically zero (measured: +0.0 pixel difference between a restoration and plain enamel)."""
    if reduce not in REDUCTIONS:
        raise ValueError(f"reduce must be one of {REDUCTIONS}, got {reduce!r}")
    out = []
    for k, (lvl, wid) in enumerate(RGB_WINDOWS):
        use_max = reduce == "max" or (reduce == "hybrid" and k == 2)
        if use_max and k == 2:
            lvl, wid = B_MAX_WINDOW
        w = _window01(vals, lvl, wid)
        out.append(w.max(axis=axis) if use_max else w.mean(axis=axis))
    return out


CROP_THR = 6.0            # luminance below this is air/outside the reconstruction FOV


def crop_to_content(rgb_u8: np.ndarray, thr: float = CROP_THR,
                    margin_px: int = 4) -> np.ndarray:
    """Trim the black border a reconstruction FOV leaves in a rectangular frame.

    Measured: 25% of `occlusal_rgb` and ~half of `mpr_rgb` is background, and every background
    pixel still costs its share of a ViT patch. Cropping first and resizing after means the
    same token count buys more px/mm on the anatomy -- the only way to raise patch density
    without raising the token bill."""
    lum = rgb_u8.astype(np.float32).mean(axis=2) if rgb_u8.ndim == 3 else rgb_u8.astype(np.float32)
    m = lum > thr
    if not m.any():
        return rgb_u8
    ys, xs = np.nonzero(m)
    y0 = max(int(ys.min()) - margin_px, 0)
    y1 = min(int(ys.max()) + 1 + margin_px, rgb_u8.shape[0])
    x0 = max(int(xs.min()) - margin_px, 0)
    x1 = min(int(xs.max()) + 1 + margin_px, rgb_u8.shape[1])
    return rgb_u8[y0:y1, x0:x1]


def curved_reformat_rgb(arr_hu: np.ndarray, curve, spacing,
                        height: int = FRAME_TILE, reduce: str = "mean",
                        crop: bool = False, max_width: int = 1568) -> Image.Image:
    """The panoramic geometry, windowed per channel then reduced along the ray.

    Deliberately not `pseudo_opg`'s transmittance: that maps attenuation through a Beer-Lambert
    exponential onto one grey axis, which is what makes it read as a radiograph and what
    destroys the HU value the windows need. Same trough, same adaptive depth, same Z band as
    view 1, so the two are pixel-aligned and can be read together.

    ``reduce`` defaults to "mean" -- the shipped behaviour, unchanged, because the submission
    container calls this path. See REDUCTIONS above for what "hybrid" buys."""
    nz = arr_hu.shape[2]
    sp = float(spacing[0])
    z0, z1 = jaw_zband(curve.occlusal_z, nz, spacing)
    P, N = flared_extend(curve)
    depth = np.clip(DEPTH_K * _curvature_radius_mm(P, sp), DEPTH_MIN_MM, DEPTH_MAX_MM)
    off = (depth[:, None] / sp) * np.linspace(-1.0, 1.0, N_DEPTH_SAMPLES)[None, :]
    xs = (P[:, 0:1] + off * N[:, 0:1])[:, :, None]
    ys = (P[:, 1:2] + off * N[:, 1:2])[:, :, None]
    zs = np.arange(z0, z1, dtype=np.float32)[None, None, :]
    vals = _sample(arr_hu, xs, ys, zs, AIR_REF)      # (M, nd, nzb); sampled once, shared
    chans = [c.T[::-1, :] for c in _reduce_rgb(vals, 1, reduce)]
    out = _stack_u8(chans)
    if crop:
        out = crop_to_content(out)
    return _rgb_wide(out, height, max_width=max_width)


def occlusal_rgb_view(arr_hu: np.ndarray, spacing, z_split: int,
                      height: int = FRAME_TILE, max_width: int = 1568,
                      bands: tuple[tuple[int, int] | None, tuple[int, int]] | None = None,
                      reduce: str = "mean", crop: bool = False) -> Image.Image:
    """`occlusal_view`'s geometry, windowed per channel then reduced per jaw.

    ``bands`` (from `occlusal_bands`) replaces the whole-jaw-half Z ranges; ``reduce="hybrid"``
    keeps the mean in R and G and takes the max in B, which is what stops a crown-band maximum
    from rendering as a white silhouette in every channel. Both default off: the submission
    container calls this with the shipped signature."""
    nz = arr_hu.shape[2]
    zs = bands if bands is not None else ((min(z_split, nz - 1), nz), (0, max(z_split, 1)))
    zs = [b for b in zs if b is not None]          # a mandible-only scan emits ONE tile
    tiles = []
    for z0, z1 in zs:
        sub = arr_hu[:, :, max(z0, 0):min(max(z1, z0 + 1), nz)]
        # Mean, not the grayscale view's MIP: a maximum over a whole jaw half is 1.0 in every
        # channel almost everywhere once windowed, which is a white silhouette, not a tissue map.
        chans = [c.T[::-1, :] for c in _reduce_rgb(sub, 2, reduce)]
        t = _stack_u8(chans)
        tiles.append(crop_to_content(t) if crop else t)
    h = max(t.shape[0] for t in tiles)
    pad = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    if len(pad) == 1:
        return _rgb_wide(pad[0], height, max_width=max_width)
    return _rgb_wide(np.concatenate([pad[0], np.zeros((h, 6, 3), np.uint8), pad[1]], axis=1),
                     height, max_width=max_width)


def mpr_rgb_view(arr_hu: np.ndarray, spacing, curve, height: int = FRAME_TILE,
                 crop: bool = False, max_width: int = 1568) -> Image.Image:
    """`mpr_view`'s geometry, windowed per channel then reduced.

    The coronal is a mean here, not the grayscale view's MIP: a maximum over a 50 mm slab
    reports the densest thing anywhere in it, which after windowing is 1.0 for every channel
    across most of the frame."""
    nx, ny = arr_hu.shape[0], arr_hu.shape[1]
    mx = int(round(curve.midline_xy[0])) if curve.midline_xy else nx // 2
    hx = max(int(round(SAG_HALF_MM / float(spacing[0]))), 1)
    sag = arr_hu[max(mx - hx, 0):mx + hx + 1, :, :]
    cy = int(round(np.average(np.arange(ny),
                              weights=arr_hu.mean(axis=(0, 2)) - arr_hu.min() + 1e-3)))
    hy = max(int(round(COR_HALF_MM / float(spacing[1]))), 1)
    cor = arr_hu[:, max(cy - hy, 0):cy + hy + 1, :]
    a = _stack_u8([_window01(sag, l, w).mean(axis=0).T[::-1, :] for l, w in RGB_WINDOWS])
    b = _stack_u8([_window01(cor, l, w).mean(axis=1).T[::-1, :] for l, w in RGB_WINDOWS])
    if crop:                                  # ~half of this view is background
        a, b = crop_to_content(a), crop_to_content(b)
    h = max(a.shape[0], b.shape[0])
    a, b = (np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in (a, b))
    return _rgb_wide(np.concatenate([a, np.zeros((h, 6, 3), np.uint8), b], axis=1), height,
                     max_width=max_width)


def jaw_cloud_projected(arr: np.ndarray, spacing, t_tooth: float,
                        z_lo: int, z_hi: int) -> np.ndarray:
    """Enamel anywhere in ``[z_lo, z_hi)``, summed along Z, then reduced to the arch component.

    ``arch.tooth_cloud`` samples a ±8 mm slab about one plane, which is correct for a closed bite
    and wrong for an open one: it catches whichever teeth happen to be coplanar with the enamel
    peak. Projecting the jaw's whole Z range is blind to how the jaw is tilted, so a lower incisor
    sitting 30 mm above the lower molars still lands in the cloud. The connectivity/arch-likeness
    seed is `tooth_cloud`'s, unchanged — it is what keeps the spine and the rami out."""
    w = (arr[:, :, max(z_lo, 0):min(z_hi, arr.shape[2])] > t_tooth).sum(axis=2).astype(np.float32)
    lab, n = ndimage.label(w > 0)
    if n < 2:
        return w
    sizes = ndimage.sum(np.ones_like(lab, np.float32), lab, np.arange(1, n + 1))
    best, best_score = None, -1.0
    for k in np.argsort(sizes)[::-1][:SEED_CANDIDATES]:
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
        score = float(np.ptp(xs) + np.ptp(ys)) * float(spacing[0]) * float(sv[1] / max(sv[0], 1e-6))
        if score > best_score:
            best, best_score = merged, score
    return best if best is not None else w


ANT_HALF_MM = 22.0        # canine-to-canine half-width about the midline
ANT_MIN_FRAC = 0.04       # below this share of the jaw cloud in the anterior sextant, the
                          # enamel cloud has no front teeth to pose the vertex on


def anterior_starved(w: np.ndarray, spacing, mid_x: float) -> bool:
    """True when a jaw's enamel cloud carries almost nothing in the anterior sextant.

    This is the open-bite / unerupted-anterior case, and it is why fitting the mandible to
    enamel fails there: with no anterior points the parabola's vertex is *extrapolated* from the
    molars, and it lands tens of mm labial of the real symphysis. Measured on F028/F001 the
    anterior mandibular **bone** sits a median 24 mm lingual of the trough while the posterior
    sits within 2 mm — so the front of the frame images lip and vestibule, which is the dark gap,
    with the symphysis smeared along the slab edge underneath it."""
    xs, _ = np.nonzero(w > 0)
    if xs.size == 0:
        return True
    ant = np.abs(xs - mid_x) * float(spacing[0]) <= ANT_HALF_MM
    return float(ant.sum()) / float(xs.size) < ANT_MIN_FRAC


HU_CROWN = 2300.0         # calibrated-HU floor for "this slice contains crowns"
ROW2_ENAMEL_FRAC = 0.10   # a second crown band must hold at least this share of the first's


def jaw_split_z(arr: np.ndarray, spacing, curve, arr_hu: np.ndarray | None = None,
                validate: bool = False) -> tuple[int, int, int]:
    """(z_maxilla, z_mandible, z_split) from the two enamel crown bands. Falls back to a split at
    the single band when the profile has only one (edentulous jaw, or a closed bite whose rows
    merge) so the caller always gets two halves to work with.

    ``validate=True`` checks the second band against **calibrated HU** before believing it, and
    that check matters: `arch.occlusal_rows` profiles at ``curve.thresholds[2]`` (1500-1930 HU
    on this cohort), which counts the mandibular **cortical shell** as well as enamel, and
    ``arch.ROW2_MIN_FRAC = 0.0`` makes the second band unconditional -- the argmax of whatever
    survives an 8 mm guard around the first. Measured over the 62 val cases, the returned second
    band holds **under 10% of the first band's enamel on 24 of 62 (39%)**, at separations up to
    44 mm: A005's "mandibular crown row" is the inferior border of the mandible, 31 enamel
    voxels against 3449 at the real row. Downstream that puts `z_split` inside the mandibular
    body, so the "maxilla" tile of `occlusal_rgb` holds both arches and the "mandible" tile
    holds none, and `pano_mandible` is Z-centred on cortical bone.

    Left off by default: `arch.occlusal_rows` also feeds the arch fit itself, and
    Panoramic_Rendering.md records four `occlusal_z` fixes that each regressed other cases.
    This validates the *consumer*, not the detector."""
    t_tooth = float(curve.thresholds[2])
    z1, z2 = occlusal_rows(arr, t_tooth, spacing)
    if z2 is not None and validate and arr_hu is not None:
        nz = arr.shape[2]
        h = max(int(round(3.0 / float(spacing[2]))), 1)
        prof = (arr_hu >= HU_CROWN).sum(axis=(0, 1)).astype(np.float64)
        e1 = prof[max(z1 - h, 0):z1 + h + 1].sum()
        e2 = prof[max(z2 - h, 0):z2 + h + 1].sum()
        if e2 < ROW2_ENAMEL_FRAC * max(e1, 1.0):
            z2 = None
    if z2 is None:
        return z1, z1, z1
    return max(z1, z2), min(z1, z2), int(round(0.5 * (z1 + z2)))


DUP_TROUGH_MM = 2.0       # two troughs closer than this everywhere are the same view


def _same_trough(a, b, spacing) -> bool:
    """True when two fitted troughs are the same curve to within DUP_TROUGH_MM.

    A validated second crown band is necessary but not sufficient for `pano_mand_rgb` to earn
    its ~900 tokens: on a normal closed bite the mandibular refit converges on the union trough
    anyway (the two arches occlude, so they share a horseshoe), and the second render is then a
    near-copy. Compare the curves, not the labels."""
    pa, pb = a.points, b.points
    if pa is None or pb is None or len(pa) < 4 or len(pb) < 4:
        return False
    d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1]).min(axis=1)
    return float(np.median(d) * float(spacing[0])) < DUP_TROUGH_MM


OCCL_BAND_MM = 13.0       # crown-band thickness per jaw, measured from the occlusal contact
OCCL_PEAK_FRAC = 0.20     # a slice is "in the dentition" above this share of the enamel peak
OCCL_ROW2_MASS_FRAC = 0.15   # an opposing arch carries at least this share of the main one's enamel


def is_single_jaw(arr_hu: np.ndarray, spacing) -> bool:
    """⚠ **MEASURED AND IT DOES NOT WORK. Do not wire this up.** Kept as a recorded negative.

    The idea was sound -- a two-arch scan wants its split at the occlusal contact (both occlusal
    tiles full) while a single-jaw scan wants it above the one arch (upper tile near-empty, which
    is what "maxilla not included" should look like) -- and separating the two populations *does*
    show the contact-plane detector helping on both (see `occlusal_contact_z`). The problem is
    identifying which population a case is in.

    This implementation thresholds the calibrated-HU enamel profile and asks whether a second run
    carries a comparable mass. Scored against the annotators' own `fov.maxilla.status` on the 34
    val cases that state it: **TP 12 / TN 4 / FP 4 / FN 14, accuracy 0.47** -- worse than chance.
    P216 (annotator: maxilla excluded) is called two-arch; A005 (maxilla included) is called
    single-jaw. Enamel mass above a plane does not separate "the other arch" from "the top half
    of this arch", which is the same confusion that makes `arch.occlusal_rows` return the
    mandibular cortex as a crown row.

    An FOV-based test (is the scanner still cutting bone at the top slice) is the obvious next
    candidate -- `render.detect_superior_truncation` already computes something close -- but it
    was not tried, and note it fires on 111/128 volumes, so it is unlikely to separate these two
    populations either."""
    prof = (arr_hu >= HU_CROWN).sum(axis=(0, 1)).astype(np.float64)
    if prof.max() <= 0:
        return True
    prof = ndimage.uniform_filter1d(prof, 5)
    lab, n = ndimage.label(prof > OCCL_PEAK_FRAC * prof.max())
    if n < 2:
        return True
    mass = ndimage.sum(prof, lab, np.arange(1, n + 1))
    top = np.sort(mass)[::-1]
    return bool(top[1] < OCCL_ROW2_MASS_FRAC * top[0])


def occlusal_contact_z(arr_hu: np.ndarray, spacing, fallback: int) -> int:
    """The Z of the occlusal contact plane, from the **calibrated-HU enamel profile**.

    `arch.occlusal_rows` cannot be used for this and it is worth being explicit about why: it
    profiles at ``curve.thresholds[2]`` (1500-1930 HU here), which counts the mandibular
    cortical shell, so on A005 it returns the inferior border of the mandible (z=64) as the
    "second crown row" over the real mandibular crowns at z~163. Rejecting that bogus row then
    collapses the split onto the maxillary peak (z=212) -- above the whole lower arch -- and the
    "mandible" tile ends up holding both jaws while the "maxilla" tile holds alveolar bone. That
    is the blur.

    At >=2300 HU the profile is enamel and dense restorations only, and the two arches are two
    humps with a gap between them:

        A005  ... 157:2059  163:2825  169:2537 ... 187:901 ... 205:3532  211:3583 ...
        F038  ... 156:1417  162:573   168:60   174:666  180:2369 ... 192:2685 ...

    So: threshold the profile at ``OCCL_PEAK_FRAC`` of its peak; if that leaves two or more
    runs, the contact is the middle of the largest gap between the two biggest; if it leaves
    one, the contact is the deepest valley inside it (a closed bite whose arches touch). Falls
    back to the caller's row when the profile has no usable structure."""
    prof = (arr_hu >= HU_CROWN).sum(axis=(0, 1)).astype(np.float64)
    if prof.max() <= 0:
        return int(fallback)
    prof = ndimage.uniform_filter1d(prof, 5)
    mask = prof > OCCL_PEAK_FRAC * prof.max()
    lab, n = ndimage.label(mask)
    if n >= 2:
        # Mass, not count: on P545 (a mandible-only 50 mm scan) the "second arch" was a 4-slice
        # fragment at the FOV rim, and splitting on it put the contact above 97% of the enamel.
        # A real opposing arch carries a comparable amount of enamel, so require it.
        mass = ndimage.sum(prof, lab, np.arange(1, n + 1))
        a, b = (np.argsort(mass)[::-1][:2] + 1)
        if mass[b - 1] >= OCCL_ROW2_MASS_FRAC * mass[a - 1]:
            za = np.nonzero(lab == a)[0]
            zb = np.nonzero(lab == b)[0]
            if za.min() > zb.max():
                za, zb = zb, za
            if za.max() < zb.min():                   # a genuine gap between two arches
                return int(round(0.5 * (za.max() + zb.min())))
    z = np.nonzero(mask)[0]
    if z.size >= 12:                                   # one merged mass -> its interior valley
        lo, hi = int(z.min()), int(z.max())
        pad = int(0.25 * (hi - lo))
        seg = prof[lo + pad:hi - pad + 1]
        if seg.size:
            return int(lo + pad + int(np.argmin(seg)))
    return int(fallback)


def occlusal_bands(spacing, z_max: int, z_mand: int, nz: int,
                   band_mm: float = OCCL_BAND_MM) -> tuple[tuple[int, int], tuple[int, int]]:
    """((z0, z1) maxilla, (z0, z1) mandible) -- the crown slabs, **disjoint**, not jaw halves.

    Two things this has to get right, and the first version got the second one wrong.

    1. *Thickness.* The shipped `occlusal_rgb` means over everything above / below `z_split`:
       56 mm and 41 mm of Z on A005. Crowns are ~8 mm of that, so they are diluted ~5x while the
       mandibular cortical shell -- which spans the full height -- accumulates into the bright
       horseshoe that dominates the tile. `band_mm` is a crown band, so it is ~13 mm, not 20+:
       thicker just re-admits the roots and the alveolar bone that caused the blur.

    2. *Disjointness.* Bands centred on each row **overlap** whenever the rows are closer than
       the band thickness, and then the same crowns are projected into BOTH tiles -- two arches
       superimposed in one frame, which reads as ghosting rather than as a maxilla and a
       mandible. Split at the contact plane (the midpoint of the two rows) and grow away from
       it, so no slice can be in both.

    3. *The rule is uniform even when the anatomy is not.* Always two bands, always split at
       the occlusal contact -- never a case-dependent number of tiles. A single-jaw scan (P545:
       50.4 mm, annotator marks the maxilla absent, chart carries 4 mandibular teeth and **zero**
       maxillary ones) then shows the upper half of the lower crowns in the upper tile, which is
       honest: it is what is above the contact plane. Emitting one tile there instead would let
       the model read "maxilla not included" off the *number of images* rather than off the
       pixels -- the same shortcut `_FOV_NOTE` taught when it fired on 99% of prompts. With one
       detected row, that row IS the contact plane.

    Returns ``(maxilla_band, mandible_band)``, both always present."""
    sz = float(spacing[2])
    thick = max(int(round(band_mm / sz)), 1)
    mid = (int(round(0.5 * (z_max + z_mand))) if z_max != z_mand else int(z_max))
    mid = int(min(max(mid, 0), nz - 1))
    return ((mid, min(mid + thick, nz)), (max(mid - thick, 0), mid))


def _gray(a2d: np.ndarray, lo: float, hi: float, height: int) -> Image.Image:
    return to_wide_rgb(window_to_uint8(a2d, lo, hi), height=height, max_width=1568)


def occlusal_view(arr: np.ndarray, spacing, z_split: int, lo: float, hi: float,
                  height: int = FRAME_TILE) -> Image.Image:
    """Axial MIP of each jaw, maxilla left, mandible right, anterior up.

    A maximum along Z, so it is defined for every (x, y) regardless of what the dentition does in
    Z — the property the panoramic lacks. Crowns read as the bright arch; this is the view that
    still shows a tooth the trough missed."""
    nz = arr.shape[2]
    tiles = []
    for z0, z1 in ((min(z_split, nz - 1), nz), (0, max(z_split, 1))):
        mip = arr[:, :, z0:z1].max(axis=2).T[::-1, :]        # (y=A, x=L) -> anterior up
        tiles.append(window_to_uint8(mip, lo, hi))
    h = max(t.shape[0] for t in tiles)
    pad = [np.pad(t, ((0, h - t.shape[0]), (0, 0))) for t in tiles]
    return _gray(np.concatenate([pad[0], np.zeros((h, 6), np.uint8), pad[1]], axis=1),
                 0, 255, height)


def mpr_view(arr: np.ndarray, spacing, curve, lo: float, hi: float,
             height: int = FRAME_TILE) -> Image.Image:
    """Mid-sagittal thick-MPR + coronal thick-MPR, side by side, superior up.

    The sagittal is where an open bite, an overbite and the curve of Spee are directly visible —
    all three are invisible in an unrolled panoramic, and all three are why a planar trough drops
    a row. The coronal carries the FOV story (condyles, sinus floor, superior truncation)."""
    nx, ny = arr.shape[0], arr.shape[1]
    mx = int(round(curve.midline_xy[0])) if curve.midline_xy else nx // 2
    hx = max(int(round(SAG_HALF_MM / float(spacing[0]))), 1)
    sag = arr[max(mx - hx, 0):mx + hx + 1, :, :].mean(axis=0).T[::-1, :]     # (S, A) sup up
    cy = int(round(np.average(np.arange(ny), weights=arr.mean(axis=(0, 2)) - arr.min() + 1e-3)))
    hy = max(int(round(COR_HALF_MM / float(spacing[1]))), 1)
    cor = arr[:, max(cy - hy, 0):cy + hy + 1, :].max(axis=1).T[::-1, :]      # (S, L) sup up
    a = window_to_uint8(sag, lo, hi)
    b = window_to_uint8(cor, lo, hi)
    h = max(a.shape[0], b.shape[0])
    a, b = (np.pad(t, ((0, h - t.shape[0]), (0, 0))) for t in (a, b))
    return _gray(np.concatenate([a, np.zeros((h, 6), np.uint8), b], axis=1), 0, 255, height)


# The shipped set. `pano_maxilla` and `hemiarch` are built but off by default:
#
#   pano_maxilla — near-duplicate of `pano_union` on every case measured. The union trough is
#     already posed on the maxillary crowns (they dominate the enamel cloud), so refitting to
#     the maxilla alone moves it by a few mm. It is the *mandible* that needs its own trough.
#   hemiarch — the magnified half-arches. Same pixels as `pano_union`, only cropped and
#     resampled, so it adds tokens without adding information; worth switching on only if a
#     downstream read needs the resolution.
#
# Both stay in the code because the argument for dropping them is a measurement, not a
# deletion — switch them back on with `views=`.
# Grayscale is kept ONLY where it is a different function of the data, not a channel-collapsed
# RGB. Measured on F001:
#
#   * A grayscale view costs exactly as many image tokens as an RGB one — `to_wide_rgb` already
#     emits 3-channel images with R=G=B, and Qwen's ViT patches on pixels, not channels. So
#     `occlusal` and `occlusal_rgb` are both 512 tokens. There is no cheap-grayscale argument.
#   * `pano_union` vs `pano_rgb` luminance correlates at r = 0.93, not 1.0 — the pseudo-OPG is a
#     Beer-Lambert transmittance with metal-halo damping and per-ray winsorising, the RGB pano a
#     linear windowed mean. Neither contains the other, and the grayscale one is the one that
#     matches what the VLM was pretrained on (real panoramic radiographs), which is the whole
#     premise of the pseudo-OPG.
#   * `occlusal` and `mpr`, by contrast, are plain reformats. Their RGB twins are the same
#     geometry over three pathology windows instead of one percentile window, so the grayscale
#     adds a fourth view of the same field. Dropped: -1408 tokens/case.
#
# Whether the frozen encoder actually needs the grayscale panoramic is an ablation, not a
# theory question — run `--views pano_rgb occlusal_rgb mpr_rgb` against this default.
DEFAULT_VIEWS = ("pano_union", "pano_mandible", "pano_rgb", "occlusal_rgb", "mpr_rgb")
ALL_VIEWS = ("pano_union", "pano_maxilla", "pano_mandible", "occlusal", "mpr", "hemiarch",
             "pano_rgb", "pano_mand_rgb", "occlusal_rgb", "mpr_rgb")
RGB_VIEWS = ("pano_rgb", "pano_mand_rgb", "occlusal_rgb", "mpr_rgb")

# ── The v2 set (2026-08-10) ──────────────────────────────────────────────────────
#
# Four stills, all RGB, all cropped to anatomy, all hybrid-reduced, at TILE_V2 instead of 448.
# The reasoning, all of it measured:
#
#   * the two grayscale panoramics go. `pano_union` is R^2 = 0.937 linearly predictable from
#     `pano_rgb`'s three channels (and the relation is a clean saturating curve, so a nonlinear
#     read is higher still), while `pano_rgb` is NOT predictable from it (R^2 0.58 in B). The
#     mandible-fitted trough was the part worth keeping, so it comes back as `pano_mand_rgb` --
#     same geometry, three windows instead of one transmittance.
#   * `hybrid` reduction on both panoramics: d' 0.94 -> 4.51 for dense restoration vs enamel.
#   * `occlusal_rgb` on validated crown bands instead of whole jaw halves.
#   * everything cropped: 25% of the occlusal frame and ~half of the MPR frame was background,
#     and a background pixel costs its share of a ViT patch like any other.
#
# Freed by the two dropped grayscale views + the crops, and spent on TILE_V2: the point is
# px/mm on anatomy, not fewer tokens.
#   * `mpr_rgb` goes. It is the emptiest view in the set on the v2 renders -- **45% flat
#     patches**, B-channel entropy 1.8-3.4 bits of 8 -- and its unique content is the Z story
#     (bite opening, FOV truncation), which the prompt already states in words from the measured
#     `superior_truncated` flag. None of the five failing chart classes is read from a
#     mid-sagittal MPR. It stays in the code, one `--views mpr_rgb` away, because dropping it is
#     a measurement and not a deletion.
#   * `pano_mand_rgb` goes too, for ALL cases. Refitting the trough to the sub-contact cloud
#     converges on the union trough on a closed bite -- A005 renders a complete duplicate at
#     ~1600 tokens. It is a genuinely different view on the open-bite minority (r < 0.5 on
#     13/62 with the grayscale pair), but the alternative to dropping it everywhere is emitting
#     it only where it differs, and that makes the *view count* a function of the bite -- a
#     shortcut the model can read instead of the pixels. Uniform beats opportunistic:
#     `occlusal_rgb` is the no-trough backstop for the mandibular crowns.
#   * `pano_union` is BACK, and this is a reversal. The audit's information argument against it
#     stands (R^2 0.937 predictable from `pano_rgb`), but it is an argument about a linear fit,
#     not about legibility: on the implant cases the grayscale transmittance shows metal as the
#     single brightest object in the frame, while in `pano_rgb` every dense structure — enamel,
#     crown, post, implant — renders as the same violet. The B channel does separate them
#     numerically (enamel 17 / crown 157 / metal 255) but that ordering lives in hue on a bright
#     background, and it did not survive visual inspection on F038 or A068. Until the representation probe
#     can rank them, the cheap grayscale view stays: it is ~1000 tokens and it is the one view
#     whose tone map is monotone in density.
VIEWS_V2 = ("pano_union", "pano_rgb", "occlusal_rgb")
TILE_V2 = 728            # 26 x 28. 448 -> 728 is 1.63x linear, ~2.6x the patches per view.
STILL_TOKEN_BUDGET_V2 = 4800   # per case, across whatever stills survived the duplicate gates
# panoramic-only: panoramic-only. Drop the cross-section video AND occlusal_rgb and spend the
# whole budget on the two panoramics. The point is NOT more pixels — pano_rgb already renders at
# 2.2x the native 0.3 mm voxel grid — it is ViT patch density: at ~6 k tokens on two views a
# 28-px patch covers ~1.3 mm instead of ~3.9 mm, the regime where a crown margin or a root
# filling is several patches wide rather than a fraction of one. It is equally an ABLATION: if
# panoramic-only ties the full set, the video's ~2.5 k tokens are free budget, and the
# "panoramic vs video vs both" question is finally answered.
VIEWS_PANO = ("pano_union", "pano_rgb")
STILL_TOKEN_BUDGET_PANO = 6000   # ~= reprv2_1's total (stills 4800 + video 1440-2992) on 2 views
MAX_MPX = 1.59e6         # per image, under the processor's 1280^2 = 1.64 Mpx downsample threshold


def fit_token_budget(views: list[tuple[str, Image.Image]], budget: int,
                     max_mpx: float = MAX_MPX,
                     per_view: bool = False) -> list[tuple[str, Image.Image]]:
    """Scale the stills by a common factor so they spend ``budget`` image tokens between them.

    ``per_view=True`` scales each still to **its own** cap instead — ``min(max_mpx, an equal
    share of budget)``. The common factor is right when the views are near the same size and the
    budget is tight, because it preserves their relative scale; it is wrong when one view is at
    the per-image cap and another is a thumbnail, because the factor is then bounded by the
    view with no headroom and the small one can never grow. That is not hypothetical: measured
    2026-08-11 on F001, `pano_union` emitted **532 x 196 = 133 tokens** next to `pano_rgb`'s
    2044, i.e. 6% of the budget, for the whole `fiveviews_pano_v1` run. Relative scale is not
    worth preserving between a pseudo-OPG and an RGB reformat of the same trough — they are
    different tone maps of one geometry, not a scale pair.

    The view set is now *case-dependent*: `pano_mand_rgb` is dropped when it duplicates
    `pano_rgb`, and `occlusal_rgb` is one tile instead of two on a single-jaw scan. Without this
    the cases that emit fewer views also get a smaller total input -- A005, a closed-bite mouth
    with 4 crowns and 7 fillings, came out at 2 stills and *fewer* tokens than the five-view v1
    render. A budget spends the freed tokens on resolution instead, which is the point of
    dropping a redundant view.

    Upsampling adds no data -- `pano_rgb` is already ~7 px/mm against a 0.3 mm voxel grid, i.e.
    2.2x above it. What it adds is **ViT patch density**: 28 px covers ~3.9 mm at the base scale
    and ~2.6 mm at 1.5x, so the encoder spends more compute per mm^2 on the same information.
    That is the patch-density lever, and it is an empirical bet, not a free lunch.

    Never scales past ``max_mpx`` per image, and never scales *down* (a case whose views already
    exceed the budget keeps them; the cap that matters there is the per-image one)."""
    if not views or budget <= 0:
        return views
    if per_view:
        share = budget / len(views)
        out = []
        for n, im in views:
            area = max(im.width * im.height, 1)
            f = min((share * 784 / area) ** 0.5, (max_mpx / area) ** 0.5)
            if f <= 1.01:
                out.append((n, im))
                continue
            w = max(28, int(round(im.width * f / 28)) * 28)
            h = max(28, int(round(im.height * f / 28)) * 28)
            out.append((n, im.resize((w, h), Image.LANCZOS)))
        return out
    have = sum((im.width // 28) * (im.height // 28) for _, im in views)
    if have <= 0 or have >= budget:
        return views
    f = min((budget / have) ** 0.5,
            min((max_mpx / max(im.width * im.height, 1)) ** 0.5 for _, im in views))
    if f <= 1.01:
        return views
    out = []
    for n, im in views:
        w = max(28, int(round(im.width * f / 28)) * 28)
        h = max(28, int(round(im.height * f / 28)) * 28)
        out.append((n, im.resize((w, h), Image.LANCZOS)))
    return out
MAX_WIDTH_V2 = 2184      # 78 x 28. 2184 x 728 = 1.59 Mpx, just inside the 1.64 Mpx budget, so
                         # a 2.7:1 panoramic reaches its true aspect instead of being clamped.

# ── Panoramic tiling: the only way past 2028 tokens on one field ─────────────────
#
# `MAX_MPX` caps ONE image at 1.59 Mpx = 2028 image tokens, and `fit_token_budget` scales the
# whole set by a COMMON factor bounded by the smallest per-view headroom -- so with `pano_rgb`
# already at the cap, `--still-tokens 6000` bought nothing (measured 2026-08-11, the
# panoramic-only run landed at ~2.2 k tokens/case whatever the budget said). More tokens on the
# same anatomy therefore requires more IMAGES, which is what this does: render the trough into
# one wide frame at `k x MAX_WIDTH_V2`, then cut it into `k` tiles that each sit at the per-image
# cap. `k` tiles = `k x 2028` tokens over the same field, i.e. `k`-fold ViT patch density.
#
# ⚠ **These tokens are patch density, not data, and the margin is smaller than it looks.**
# Measured on F001: the native trough raster is **544 x 206** (the trough is stepped at one voxel,
# ~0.3 mm, and the Z band at the voxel rate), while the shipped `pano_rgb` already emits
# 1932 x 728 -- a **3.5x LANCZOS upsample of the render**. At k=3 the anterior tile is ~10x the
# native raster. Nothing below the 0.3 mm voxel grid can appear; what changes is how many 28-px
# ViT patches the encoder spends per mm^2 (3.9 mm/patch at k=1 -> ~1.3 mm at k=3), which is
# exactly the panoramic-only hypothesis and has never been tested. If it fails, patch density is refuted
# and the remaining lever on the panoramic is denser trough SAMPLING (sub-voxel steps in
# `flared_extend`), which is renderer surgery this round did not buy.
#
# The cut is by ARC POSITION, so tile i of k is always the same anatomical segment: left of the
# image is the patient's right (LAS, unrolled low-x -> high-x), so k=2 is R | L and k=3 is
# R molars | anterior | L molars. Tiles overlap by `PANO_TILE_OVERLAP` of their width so a
# finding on a seam is whole in one of them.
PANO_TILE_OVERLAP = 0.06
PANO_TILEABLE = ("pano_union", "pano_rgb", "pano_mand_rgb", "pano_maxilla", "pano_mandible")


def split_wide(img: Image.Image, k: int, overlap: float = PANO_TILE_OVERLAP,
               max_mpx: float = MAX_MPX) -> list[Image.Image]:
    """Cut a wide frame into ``k`` overlapping column tiles, each scaled to the per-image cap.

    Returned in image order, i.e. patient right -> left. Each tile keeps the source aspect of
    its own segment and is upscaled (never downscaled below its own pixels) to just under
    ``max_mpx``; both sides stay multiples of 28 so no patch is half empty."""
    if k <= 1:
        return [img]
    step = img.width / k
    pad = step * overlap
    out: list[Image.Image] = []
    for i in range(k):
        x0 = int(max(0, round(i * step - pad)))
        x1 = int(min(img.width, round((i + 1) * step + pad)))
        t = img.crop((x0, 0, x1, img.height))
        f = (max_mpx / max(t.width * t.height, 1)) ** 0.5
        w = max(28, int(round(t.width * f / 28)) * 28)
        h = max(28, int(round(t.height * f / 28)) * 28)
        out.append(t.resize((w, h), Image.LANCZOS))
    return out


def five_views(arr: np.ndarray, spacing, tile: int = FRAME_TILE,
               views: tuple[str, ...] = DEFAULT_VIEWS,
               z_jitter_vox: int = 0, reduce: str = "mean",
               crop: bool = False, validate_rows: bool = False,
               max_width: int | None = None, still_token_budget: int = 0,
               pano_split: int = 1, max_mpx: float = MAX_MPX,
               still_per_view: bool = False,
               ) -> list[tuple[str, Image.Image]]:
    """The requested views for one canonical volume, in ``views`` order.

    A view that cannot be built is omitted rather than faked, so the caller can count what it
    got. Only the requested views are computed — a skipped panoramic costs no arch fit.
    ``z_jitter_vox`` offsets the occlusal plane so the stills track the jittered video.

    ``reduce`` / ``crop`` / ``validate_rows`` are the v2 levers and every one of them defaults
    to the shipped behaviour — `submission/entrypoint.py` calls this path, and the container
    ships a checkpoint trained on the old render. Use ``VIEWS_V2`` + ``reduce="hybrid"`` +
    ``crop=True`` + ``validate_rows=True`` (+ ``tile=TILE_V2``) for the new set.

    ``pano_split`` > 1 emits each panoramic as that many overlapping arc tiles instead of one
    frame (``split_wide``), which is the only way past the 2028-token per-image cap. The
    panoramics are rendered ``pano_split``x wider before the cut, so every tile lands at the
    cap. See ``split_wide`` for why those tokens are patch density and not data."""
    k = max(int(pano_split), 1)
    # Keep the frame under the processor's per-image pixel budget at whatever tile was asked
    # for, rather than clamping width at a constant and silently changing the aspect ratio.
    # ⚠ `max_mpx` must match the `max_pixels` the PROCESSOR was built with, or the processor
    # silently downsamples everything this function worked to render.
    mw = max_width if max_width is not None else max(28, int(max_mpx / max(height_of := tile, 1)
                                                            // 28) * 28)
    del height_of
    # The pre-cut panoramic may be k x wider than one image is allowed to be: that frame is
    # never sent, only sliced, and each slice lands back under MAX_MPX.
    mw_pano = mw * k
    lo, hi = percentile_window(arr)
    out: list[tuple[str, Image.Image]] = []
    curve, _cloud = detect_arch_curve_dentition(arr, spacing)
    if curve is None:
        return out
    if z_jitter_vox and curve.occlusal_z is not None:      # match render_case's z jitter
        curve.occlusal_z = int(min(max(curve.occlusal_z + z_jitter_vox, 0), arr.shape[2] - 1))
    if "pano_union" in views:
        out.append(("pano_union", pseudo_opg(arr, curve, height=tile)))

    nz = arr.shape[2]
    hu = calibrate_to_hu(arr) if (any(v in views for v in RGB_VIEWS) or validate_rows) else None
    z_max, z_mand, z_split = jaw_split_z(arr, spacing, curve, arr_hu=hu, validate=validate_rows)
    t_tooth = float(curve.thresholds[2])
    slab = arr[:, :, max(curve.occlusal_z - 20, 0):min(curve.occlusal_z + 20, nz)].mean(axis=2)

    def _jaw_curve(z_lo: int, z_hi: int, z_ref: int):
        """Refit the trough to one jaw's own cloud. None when that jaw has no fittable arch."""
        if z_hi - z_lo < 4:
            return None
        w = jaw_cloud_projected(arr, spacing, t_tooth, z_lo, z_hi)
        # No anterior crowns to pose the vertex on -> pose it on the jaw's *bone* instead. The
        # mandibular body is a horseshoe in its own Z half, so it is a legitimate fit target;
        # the enamel cloud is preferred wherever it has a front, because crowns localise the
        # arch better than a cortical outline does.
        if anterior_starved(w, spacing, curve.midline_xy[0]):
            w = jaw_cloud_projected(arr, spacing, float(curve.thresholds[1]), z_lo, z_hi)
        c = fit_arch_symmetric(w, spacing, slab, curve.midline_xy)
        if c is None:
            return None
        c.occlusal_z, c.thresholds = z_ref, curve.thresholds
        return c

    for name, z_lo, z_hi, z_ref in (("pano_maxilla", z_split, nz, z_max),
                                    ("pano_mandible", 0, z_split, z_mand)):
        if name not in views:
            continue
        c = _jaw_curve(z_lo, z_hi, z_ref)
        if c is not None:
            out.append((name, pseudo_opg(arr, c, height=tile)))

    if "occlusal" in views:
        out.append(("occlusal", occlusal_view(arr, spacing, z_split, lo, hi, tile)))
    if "mpr" in views:
        out.append(("mpr", mpr_view(arr, spacing, curve, lo, hi, tile)))
    if "hemiarch" in views:
        hemi = hemiarch_view(arr, spacing, curve, tile)
        if hemi is not None:
            out.append(("hemiarch", hemi))
    if hu is not None:
        if "pano_rgb" in views:
            out.append(("pano_rgb",
                        curved_reformat_rgb(hu, curve, spacing, tile, reduce=reduce, crop=crop,
                                            max_width=mw_pano)))
        # ⚠ Emitted for EVERY case, even when the refit converges on the union trough and the
        # two panoramics come out near-identical (a closed bite shares one horseshoe, so
        # r(pano_union, pano_mandible) > 0.9 on 14/62 val cases; it is a genuinely different
        # view on 13/62). Gating it on "is it different?" would make the *view count* a
        # function of the bite, and the model would learn to read occlusion off how many images
        # it received rather than off the images — the `_FOV_NOTE` failure mode. A redundant
        # view costs tokens; a case-dependent view set costs a shortcut.
        if "pano_mand_rgb" in views:
            c = _jaw_curve(0, z_split if z_split > 0 else nz, z_mand)
            if c is not None:
                out.append(("pano_mand_rgb",
                            curved_reformat_rgb(hu, c, spacing, tile, reduce=reduce, crop=crop,
                                                max_width=mw_pano)))
        if "occlusal_rgb" in views:
            if validate_rows:
                zc = occlusal_contact_z(hu, spacing, z_split)
                bands = occlusal_bands(spacing, zc, zc, nz)
            else:
                bands = None
            out.append(("occlusal_rgb",
                        occlusal_rgb_view(hu, spacing, z_split, tile, max_width=mw,
                                          bands=bands, reduce=reduce, crop=crop)))
        if "mpr_rgb" in views:
            out.append(("mpr_rgb",
                        mpr_rgb_view(hu, spacing, curve, tile, crop=crop, max_width=mw)))
    out = [(n, im) for n, im in sorted(out, key=lambda kv: views.index(kv[0]))]
    if k > 1:
        # Cut each panoramic into k arc tiles, in place, keeping view order. Names carry the
        # position so the prompt can say which segment of the arch a tile is, and so a per-view
        # count in a log still reads.
        tiled: list[tuple[str, Image.Image]] = []
        for n, im in out:
            if n in PANO_TILEABLE:
                tiled += [(f"{n}_{i + 1}of{k}", t) for i, t in enumerate(split_wide(im, k))]
            else:
                tiled.append((n, im))
        out = tiled
    return (fit_token_budget(out, still_token_budget, max_mpx=max_mpx,
                             per_view=still_per_view)
            if still_token_budget else out)


def _upscale_to(img: Image.Image, height: int) -> Image.Image:
    """Resize to ``height`` keeping aspect, both sides a multiple of 28.

    Needed because ``pseudo_opg`` caps its output height at the number of Z rows in the band
    (``min(z1 - z0, 4 * height)``). On a short FOV that cap binds — S0020's band is 181 rows — so
    asking for a taller render silently returns the same size and a half-arch comes out at half
    the *pixels* rather than twice the detail. Upsampling adds no information, but it does give
    the half-arch a token budget comparable to the full frame, which is the point of a
    magnified view for a VLM."""
    h = max(28, int(round(height / 28)) * 28)
    w = max(28, int(round(img.width * h / img.height / 28)) * 28)
    return img.resize((w, h), Image.LANCZOS)


def hemiarch_view(arr: np.ndarray, spacing, curve, tile: int = FRAME_TILE) -> Image.Image | None:
    """Both hemiarches in one frame: patient RIGHT | patient LEFT, each magnified ~2x.

    Same trough and same tone map as ``pano_union`` — only the arc is restricted — so a finding
    here sits where it sits in the full frame, just larger. The cut is at ``curve.midline_xy``,
    the *measured* mid-sagittal plane rather than the middle of the image, so an off-centre head
    still puts quadrants 1+4 in the left tile. Reading order is R then L, matching how the full
    panoramic is hung.

    Returned as one image rather than two so the pair is a single item in the message and the
    two halves are directly comparable; a lone narrow tile also letterboxes badly on a sheet."""
    if curve.midline_xy is None:
        return None
    P, N = flared_extend(curve)
    mid_x = float(curve.midline_xy[0])
    tiles: list[np.ndarray] = []
    for side in ("right", "left"):
        keep = (P[:, 0] <= mid_x) if side == "right" else (P[:, 0] >= mid_x)
        if int(keep.sum()) < 32:
            return None
        n = int(keep.sum())
        half = ArchCurve(points=P[keep], normals=N[keep], tangents=N[keep],
                         arc_mm=np.arange(n, dtype=np.float32) * float(spacing[0]),
                         spacing=curve.spacing, slab=curve.slab, mask=curve.mask,
                         occlusal_z=curve.occlusal_z, thresholds=curve.thresholds,
                         midline_xy=curve.midline_xy, parabola=None)
        # extend_arch=False: P is already the extended trough; re-extending would double it.
        img = _upscale_to(pseudo_opg(arr, half, height=tile, extend_arch=False), 2 * tile)
        tiles.append(np.asarray(img.convert("L")))
    h = max(t.shape[0] for t in tiles)
    pad = [np.pad(t, ((0, h - t.shape[0]), (0, 0))) for t in tiles]
    joined = np.concatenate([pad[0], np.zeros((h, 6), np.uint8), pad[1]], axis=1)
    return Image.fromarray(joined).convert("RGB")


DEFAULT_CASES = ["F028", "F001", "P216", "A044", "S0020"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--out", default="plots/five_views")
    ap.add_argument("--tile", type=int, default=FRAME_TILE)
    ap.add_argument("--views", nargs="*", default=list(DEFAULT_VIEWS),
                    help=f"subset/order of {ALL_VIEWS}")
    args = ap.parse_args()

    for case in args.cases:
        arr, spacing = load_canonical_volume(
            f"{case}/cbct/volume.nii.gz", local_path=VOLUME_CACHE_DIR / f"{case}.nii.gz")
        arr = arr.astype(np.float32, copy=False)
        views = five_views(arr, spacing, tile=args.tile, views=tuple(args.views))
        panels = [(f"{case}  {i + 1}. {n}", im) for i, (n, im) in enumerate(views)]
        out = REPO / f"{args.out}_{case}.png"
        tile_sheet(panels, out, cols=1, tile=(1100, 300))
        print(f"{case}: {len(views)} views {[n for n, _ in views]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
