#!/usr/bin/env python3
"""
code/pipeline/preprocess/create_panoramic.py

Generate a standardized curved dental panoramic reconstruction (CPR) from
CBCT volume + segmentation mask.

Algorithm:
  1. Detect arch curve   — per-FDI-pair tooth centroids (or, when too few
                           teeth are present -- edentulous/sparse dentition --
                           a jawbone centerline instead) → ramus-extended,
                           fold-safe parametric cubic spline, then RESAMPLED TO
                           UNIFORM ARC LENGTH in millimetres so one output
                           column always spans the same physical distance
                           along the arch.
  2. Compute normals     — smooth finite-difference normals to prevent seam
                           artefacts caused by jumpy per-point normals.
  3. Curved slab sample  — chunked map_coordinates across positions × depths × Z,
                           at a SQUARE PHYSICAL PIXEL (same mm along the arch as
                           down Z), so the panoramic's aspect ratio is the
                           anatomy's own rather than a forced constant.
                           Slab depth is specified in mm and additionally shrunk
                           wherever the curve's own local radius of curvature is
                           tighter than the slab -- a safety margin on top of the
                           fold-proofed curve above, since an offset ray can
                           still cross even along a curve that doesn't fold on
                           itself if it bends tightly enough.
  4. MIP projection      — max-intensity along slab depth → one column per arch pt.
  5. Window/level        — the volume is first clipped at an ENAMEL CEILING
                           estimated per case from the tooth labels themselves
                           (median of the per-tooth 98th percentile, so a few
                           metal-crowned teeth cannot move it), which stops
                           metal restorations from smearing through the MIP and
                           dragging the window ceiling far above real anatomy --
                           the reason panoramics of restored cases came out
                           dark. Percentile windowing then runs over FOREGROUND
                           pixels only, followed by a mild CLAHE blend that
                           leaves near-black air untouched. Flip Z
                           superior-at-top.
  6. Overlay             — vectorised label vote, restricted to whichever FDI
                           numbers the case's facts.json declares present
                           (facts.structured.teeth_present) so a segmentation
                           false-positive can never draw an outline/tag for a
                           tooth that isn't actually there; FDI numbers at centroids.

Output geometry: the panoramic body is rendered directly at its final pixel
size (default ~900 px tall, width set by the arch's true length), so the PNG
is never an upscale of a smaller raster and the 1-3 px tooth outlines are
never resampled. Pass --canvas WxH to letterbox into a fixed canvas instead.

A facts.json is REQUIRED for every case (--facts-file / --facts-dir) -- it's
the source of truth for which teeth actually get outlined and tagged, not
just whatever the segmentation mask happens to contain.

Usage:
    python code/pipeline/preprocess/create_panoramic.py \
        --volume  dataset/validate/images/F006_0000.nii.gz \
        --mask    dataset/validate/masks/F006.nii.gz \
        --facts-file dataset/validate/facts/F006.json \
        --out-dir outputs/panoramic/

    toothless (facts.json still required -- typically declares an
    empty/near-empty teeth_present list for an edentulous case; the arch
    curve itself falls back to a jawbone-centerline seed automatically):
    python code/pipeline/preprocess/create_panoramic.py \
            --volume  dataset/validate/images/A022_0000.nii.gz \
            --mask    dataset/validate/masks/A022.nii.gz \
            --facts-file dataset/validate/facts/A022.json \
            --out-dir outputs/panoramic/
"""

import argparse, os, sys
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    sys.exit("pip install nibabel")
try:
    from scipy.ndimage   import (map_coordinates, uniform_filter1d,
                                  binary_erosion, binary_dilation,
                                  binary_opening, binary_closing,
                                  binary_fill_holes)
    from scipy.interpolate import splprep, splev
except ImportError:
    sys.exit("pip install scipy")

from PIL import Image, ImageDraw

# ── Label map ────────────────────────────────────────────────────────────────
UPPER_FDI_ORDER = [18, 17, 16, 15, 14, 13, 12, 11,
                   21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI_ORDER = [48, 47, 46, 45, 44, 43, 42, 41,
                   31, 32, 33, 34, 35, 36, 37, 38]

_TOOTH_PALETTE_RGBA = [
    (220, 215, 175, 130), ( 90, 170, 220, 130), (235, 130,  55, 130),
    (175,  95, 185, 130), ( 65, 185, 155, 130), (230,  75,  75, 130),
    (240, 215,  60, 130), (105, 110, 215, 130), (100, 215, 165, 130),
    (215,  95, 165, 130), (165, 215,  90, 130), ( 95, 165, 215, 130),
    (215, 165,  90, 130), (100, 215, 215, 130), (165,  95, 215, 130),
    (195, 195,  65, 130), (215, 190, 185, 130), ( 70, 145, 195, 130),
    (195, 120,  65, 130), (145,  75, 155, 130), ( 55, 155, 135, 130),
    (195,  60,  60, 130), (200, 185,  55, 130), ( 80,  90, 185, 130),
    ( 80, 185, 135, 130), (185,  75, 140, 130), (140, 185,  75, 130),
    ( 75, 140, 185, 130), (185, 140,  75, 130), ( 85, 185, 185, 130),
    (140,  75, 185, 130), (185,  75,  75, 130),
]
_ALL_TOOTH_TIDS = (list(range(11, 19)) + list(range(21, 29)) +
                   list(range(31, 39)) + list(range(41, 49)))
TOOTH_COLORS = {tid: _TOOTH_PALETTE_RGBA[i % len(_TOOTH_PALETTE_RGBA)]
                for i, tid in enumerate(_ALL_TOOTH_TIDS)}

_ALL_JAW_IDS = ([1, 2] + list(range(11, 19)) + list(range(21, 29)) +
                list(range(31, 39)) + list(range(41, 49)))
_JAW_BODY_IDS = [1, 2]   # mandible, maxilla


# ── Side anchors (orientation-marker fallback only -- NOT used by arch
#    curve detection below, which has its own independent right→left
#    traversal via descending-X sort) ─────────────────────────────────────────
#
# These exist purely so process_case can decide, in the rare case where
# _needs_lr_flip has no quadrant labels to vote with (e.g. a genuinely
# edentulous mask with zero tooth-crown labels at all), whether there's
# enough independent side information to still trust placing the 'L' side
# marker, rather than guessing.

RIGHT_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 48, 47, 46, 45, 44, 43, 42, 41]
LEFT_FDI  = [28, 27, 26, 25, 24, 23, 22, 21, 38, 37, 36, 35, 34, 33, 32, 31]


def _side_anchor(mask, fdi_list, fallback_side):
    """
    Locate an anchor point for one side of the arch. Tries the tooth mask
    first (most posterior/distal tooth present in fdi_list, since that's
    the most reliable reference point for that side); falls back to the
    lateral extremity of the jawbone mask if no teeth on that side are
    present -- so a fully edentulous case still gets a well-defined anchor
    instead of failing outright.
    """
    for fdi in fdi_list:
        voxels = np.argwhere(mask == fdi)
        if len(voxels) >= 5:
            return np.median(voxels[:, :2], axis=0)

    jaw_vox = np.argwhere(np.isin(mask, _JAW_BODY_IDS))
    if len(jaw_vox) < 20:
        return None
    x = jaw_vox[:, 0]
    pct = 95 if fallback_side == 'right' else 5
    x_edge = np.percentile(x, pct)
    extreme = jaw_vox[x >= x_edge] if fallback_side == 'right' else jaw_vox[x <= x_edge]
    if len(extreme) < 5:
        return None
    return np.median(extreme[:, :2], axis=0)


def arch_side_anchors(mask):
    """(right_anchor, left_anchor) for this mask, or None each if undetectable."""
    return (_side_anchor(mask, RIGHT_FDI, fallback_side='right'),
            _side_anchor(mask, LEFT_FDI,  fallback_side='left'))


# ── Step 1 — Arch curve detection ────────────────────────────────────────────

def _collect_jaw_centerline(mask, n_bins=60):
    """
    Fallback/seed curve for edentulous or very-sparse-dentition cases: bin
    jawbone voxels (mandible + maxilla combined) by x, and take the midpoint
    of each column's 5th-95th percentile y-extent as that column's
    centerline point. Vectorised, no per-slice looping.

    Used as the initial `centroids` seed in detect_arch_curve whenever fewer
    than 4 tooth-pair centroids are found, so the exact same
    ramus-extension / fold-trimming / spline-fit pipeline below still
    produces a usable arch curve even with no teeth (or very few) to anchor
    on -- rather than detect_arch_curve simply giving up.
    """
    vox = np.argwhere(np.isin(mask, _JAW_BODY_IDS))
    if len(vox) < 50:
        return np.empty((0, 2))
    x, y = vox[:, 0], vox[:, 1]
    x_lo, x_hi = x.min(), x.max()
    if x_hi <= x_lo:
        return np.empty((0, 2))
    edges = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    pts = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < 5:
            continue
        y_lo, y_hi = np.percentile(y[sel], [5, 95])
        pts.append([(edges[b] + edges[b + 1]) / 2, (y_lo + y_hi) / 2])
    return np.array(pts) if pts else np.empty((0, 2))


def _order_arch_points(pts, min_pts=6, straight_frac=0.15, min_gap_deg=45.0,
                       radial_smooth=3):
    """
    Put the arch points in traversal order, right → left ALONG THE ARCH, and
    damp the radial jitter between them. Both steps work in polar coordinates
    about a centre inside the arch's concavity, and together they are what
    keeps the polyline from doubling back on itself.

    1. ORDER BY ANGLE, not by X. Sorting by X is only correct where the arch
       actually runs left-right: at the posterior ends it runs almost straight
       anterior-posteriorly, so X barely separates neighbouring molars and
       their order there is decided by a few voxels of segmentation noise.
       Two consecutive points coming out in the wrong order double the
       polyline back on itself -- a fold -- which the spline fit then has to
       reproduce. Angle follows the U-shape instead, so the traversal stays
       correct all the way round into the ramus whatever the posterior teeth's
       exact X values.

    2. SMOOTH THE RADIUS along that angular sequence. Angular order alone is
       not enough, because the points are not all measured the same way: an
       arch point is the mean of the upper and the lower tooth of one FDI
       pair, EXCEPT where only one of the two is segmented, where it is that
       single tooth's own centroid. The maxillary arch is wider than the
       mandibular one, so those single-jaw points sit systematically 10-20
       voxels off the arch the paired ones describe. A point that jumps
       outward and back in again is a fold just as much as a mis-ordered one
       (this is what put a visible loop in A019's right posterior), and it is
       measurement noise rather than anatomy. Smoothing the radius along the
       sequence removes it and leaves the arch's shape alone.

    Falls back to the plain descending-X sort, with no radial smoothing, when
    polar treatment has no well-defined centre to work from: too few points,
    or a near-straight point set (the jawbone-centerline seed can be almost
    flat), or no clear angular gap on the open side of the arch to start the
    traversal from.
    """
    def _by_x(p):
        return p[np.argsort(p[:, 0])[::-1]]

    n = len(pts)
    if n < min_pts:
        return _by_x(pts)

    centred = pts - pts.mean(axis=0)
    # Principal axis of the point cloud = the arch's left-right axis; the
    # perpendicular is its anterior-posterior axis. Kept pointing toward +X
    # so "right" below still means high X, as everywhere else in this file.
    _u, _sv, vt = np.linalg.svd(centred, full_matrices=False)
    e1 = vt[0] if vt[0][0] >= 0 else -vt[0]
    e2 = np.array([-e1[1], e1[0]])
    a  = centred @ e1
    b  = centred @ e2

    a_rng, b_rng = a.max() - a.min(), b.max() - b.min()
    if a_rng < 1e-6 or b_rng < straight_frac * a_rng:
        return _by_x(pts)          # too flat to have a concavity to sort around

    # Which way the arch OPENS along e2: the two lateral ends sit on the open
    # side, the incisor apex on the other. Read off the data rather than
    # assumed, so it holds whatever orientation the source volume used.
    k       = max(2, n // 6)
    ord_a   = np.argsort(a)
    b_ends  = float(np.concatenate([b[ord_a[:k]], b[ord_a[-k:]]]).mean())
    b_mid   = 0.5 * (b.min() + b.max())
    if abs(b_ends - b_mid) < 0.10 * b_rng:
        return _by_x(pts)          # ends vs apex indistinguishable -- don't guess
    if b_ends < b_mid:
        # Make +b the open side. b_mid flips with it -- it is a coordinate in
        # this frame, and `centre` below is rebuilt from it.
        e2, b, b_mid = -e2, -b, -b_mid

    # Centre midway between apex and ends, i.e. inside the concavity.
    da, db = a, b - b_mid
    theta  = np.arctan2(db, da)
    radius = np.hypot(da, db)

    # Start the traversal at the open side: the arch occupies one continuous
    # angular band and leaves a gap where it opens, so the traversal must
    # begin at the point just after that gap. Only gaps facing the open side
    # (+b) qualify -- an edentulous ANTERIOR region also leaves a big gap, but
    # cutting there would start the sequence mid-arch.
    order = np.argsort(theta)
    th    = theta[order]
    gaps  = np.diff(np.concatenate([th, th[:1] + 2 * np.pi]))
    mids  = th + gaps / 2.0
    open_side = np.sin(mids) > 0
    if not open_side.any():
        return _by_x(pts)
    cand = np.where(open_side)[0]
    g    = cand[np.argmax(gaps[cand])]
    if np.degrees(gaps[g]) < min_gap_deg:
        return _by_x(pts)          # no clear opening -- don't trust the cut

    start = (g + 1) % n
    seq   = np.concatenate([order[start:], order[:start]])
    th_s  = theta[seq]
    r_s   = radius[seq]
    if radial_smooth and radial_smooth > 1:
        r_s = uniform_filter1d(r_s, size=radial_smooth, mode='nearest')

    # Back to XY: same centre, same angles, damped radii.
    centre  = pts.mean(axis=0) + b_mid * e2
    ordered = (centre
               + r_s[:, None] * np.cos(th_s)[:, None] * e1[None, :]
               + r_s[:, None] * np.sin(th_s)[:, None] * e2[None, :])
    if ordered[0, 0] < ordered[-1, 0]:
        ordered = ordered[::-1]    # enforce the right → left convention
    return ordered


def _reject_arch_outliers(pts, k=4.0, max_drop=3, min_keep=5):
    """
    Drop seed points that sit nowhere near the rest of the arch.

    A mis-segmented tooth label puts a centroid a long way from the dentition,
    and `detect_arch_curve` averages the upper and lower tooth of an FDI pair
    into ONE seed point -- so a single bad label drags that pair's seed to a
    meaningless midpoint between the real tooth and the bogus one. F014 is the
    worked example: label 28 covers 4266 voxels at (97, 76) while every other
    tooth in the case lies in x 170-356, y 274-412, and the 28/38 pair seeds
    the arch at (226, 180), 200+ voxels off the curve in both directions
    against a median step of 31.

    Nothing else in this file removes it. `_trim_fold` inspects only a few
    segments at each END, and this lands mid-sequence; `_order_arch_points`
    smooths the radius with a size-3 MEAN, which spreads a lone spike over
    three points instead of rejecting it. The fitted spline then has to travel
    out to the stray point and back, which on F014 added 110 mm to a 180 mm
    arch -- a third of the panoramic's width spent on empty space, with every
    tooth shrunk to pay for it.

    Test: a point's ISOLATION is its distance to the nearer of its two
    neighbours along the traversal (its only neighbour, at the ends). A real
    arch point always has a close neighbour, even across a gap left by a
    missing tooth -- that gap is about two tooth widths, ~2x the median step.
    `k=4` therefore sits well clear of any legitimate spacing while catching
    the pathological case, and `max_drop`/`min_keep` bound the damage if some
    future case trips it for a reason not anticipated here.
    """
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(max_drop):
        if len(pts) <= min_keep:
            break
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if len(steps) < 2:
            break
        med = float(np.median(steps))
        if not np.isfinite(med) or med <= 0:
            break
        isolation        = np.empty(len(pts))
        isolation[0]     = steps[0]
        isolation[-1]    = steps[-1]
        isolation[1:-1]  = np.minimum(steps[:-1], steps[1:])
        worst = int(np.argmax(isolation))
        if isolation[worst] <= k * med:
            break
        print(f"  [info] dropping arch seed point at "
              f"({pts[worst][0]:.0f}, {pts[worst][1]:.0f}) -- "
              f"{isolation[worst]:.0f} vox from its nearest neighbour vs a "
              f"median step of {med:.0f}; a mis-segmented tooth label would "
              f"otherwise bend the arch curve out to it", file=sys.stderr)
        pts = np.delete(pts, worst, axis=0)
    return pts


def detect_arch_curve(mask, n_points=500, smooth_tol_vox=2.0):
    """
    Fit a cubic spline through per-position XY centroids of detected teeth,
    then extend the curve into the ramus/condyle region using the mandible
    mask — adapted from Kwon et al. (Sci Rep 2023) which blends a tooth-arch
    parabola with a jawbone parabola to cover the full panoramic field of view.

    Point source: per-FDI-pair tooth centroids when at least 4 are found (the
    normal full/partial-dentition case). When fewer than 4 are available --
    edentulous, or only a couple of remaining teeth -- falls back to
    `_collect_jaw_centerline` (mandible+maxilla, binned by x) as the initial
    seed instead, so the rest of this function (ramus extension, fold
    trimming, spline fit) runs unchanged and still produces a usable curve.

    Instead of a fixed horizontal extension, we find the centroid of the
    lateral mandible voxels on each side (the ramus region) and add smooth
    intermediate waypoints from the outermost seed point to the ramus
    centre.
    """
    # ── Collect centroids in FDI arch order ───────────────────────────────────
    arch_pts = []
    for fdi_up, fdi_lo in zip(UPPER_FDI_ORDER, LOWER_FDI_ORDER):
        pts = []
        for fdi in (fdi_up, fdi_lo):
            voxels = np.argwhere(mask == fdi)
            if len(voxels) >= 5:
                pts.append(np.median(voxels[:, :2], axis=0))
        if pts:
            arch_pts.append(np.mean(pts, axis=0))

    if len(arch_pts) >= 4:
        centroids = np.array(arch_pts, dtype=np.float64)
        # Order along the arch, right → left. FDI order ≠ spatial order for
        # impacted or displaced teeth, so the order is taken from geometry --
        # but by ANGLE about the arch's centre rather than by X, because X
        # stops separating neighbouring teeth at the posterior ends where the
        # arch turns. See _order_arch_points.
        centroids = _order_arch_points(centroids)
        # Reject seed points that a mis-segmented label put far off the arch.
        # Done AFTER ordering, because the test is on distance to traversal
        # neighbours; if anything is dropped the ordering is redone, since the
        # outlier also skewed the polar centre that ordering was measured from.
        cleaned = _reject_arch_outliers(centroids)
        if len(cleaned) < len(centroids):
            centroids = _order_arch_points(cleaned)
    else:
        # Edentulous / very sparse dentition: no reliable tooth centroids to
        # anchor on -- seed the curve from the jawbone centerline instead so
        # the case still gets a panoramic rather than being skipped outright.
        jaw_pts = _collect_jaw_centerline(mask)
        if len(jaw_pts) < 4:
            return None   # no jaw tissue at all -- nothing to build a curve from
        # Descending-X here, not _order_arch_points: the centerline is BUILT as
        # one point per x-bin, so it is already a function of x and X-sorting
        # it is exact by construction.
        centroids = jaw_pts[np.argsort(jaw_pts[:, 0])[::-1]]

    # Remove near-duplicates
    dists     = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    keep      = np.concatenate([[True], dists > 0.5])
    centroids = centroids[keep]
    if len(centroids) < 4:
        return None

    # ── Ramus extension using mandible mask (label 1) ─────────────────────────
    # Adapted from Kwon et al.: fit a "jawbone curve" through the molar tips
    # and the posterior jaw extremes, then blend with the tooth arch.
    # Here we find the XY centroid of lateral mandible voxels on each side
    # and insert smooth waypoints from the outermost molar toward the ramus.
    mandible_vox = np.argwhere(mask == 1)   # all mandible XYZ voxels

    def _ramus_centroid(man_vox, x_thresh, side='right',
                        anchor=None, tangent=None, max_turn_deg=60):
        """
        Find the ramus centroid on one side, constrained to continue
        roughly along the arch's local direction of travel.

        Without this constraint, the raw lateral-voxel centroid can sit
        medial/posterior relative to the last molar (a real feature of
        ramus anatomy), which makes the straight-line extension from molar
        to ramus point bend sharply back over the tooth arch instead of
        continuing outward. splprep(..., s=0) then fits an exact
        interpolating spline through that folded polyline -- i.e. the fit
        curve itself loops back on itself near the last molar. When
        curved_mip() samples perpendicular slabs along a curve that folds,
        the slab at the fold sweeps back over the same anatomical region
        with a flipped in-plane normal, which is exactly what produces a
        mirrored duplicate of the last molar/last few teeth at the lateral
        edges of the panoramic.

        `max_turn_deg` caps how far the extension is allowed to turn away
        from the arch's incoming tangent: past that, we keep the ramus
        point's distance but clamp its direction, rather than let it bend
        back toward the arch.
        """
        if side == 'right':
            lateral = man_vox[man_vox[:, 0] > x_thresh]
        else:
            lateral = man_vox[man_vox[:, 0] < x_thresh]
        if len(lateral) < 20:
            return None
        # Use the most extreme (lateral) 10% of those voxels
        pct = 90 if side == 'right' else 10
        x_edge = np.percentile(lateral[:, 0], pct)
        if side == 'right':
            extreme = lateral[lateral[:, 0] >= x_edge]
        else:
            extreme = lateral[lateral[:, 0] <= x_edge]
        if len(extreme) < 5:
            return None
        target = np.array([extreme[:, 0].mean(), extreme[:, 1].mean()])

        if anchor is not None and tangent is not None:
            offset = target - anchor
            dist   = np.linalg.norm(offset)
            if dist > 1e-6:
                direction = offset / dist
                cos_turn  = np.clip(np.dot(direction, tangent), -1.0, 1.0)
                turn_deg  = np.degrees(np.arccos(cos_turn))
                if turn_deg > max_turn_deg:
                    # Rotate the tangent toward `direction` by at most
                    # max_turn_deg (2D rotation; sign from the cross
                    # product picks which way to rotate) instead of
                    # allowing the full turn.
                    max_turn_rad = np.radians(max_turn_deg)
                    cross = tangent[0] * direction[1] - tangent[1] * direction[0]
                    sign  = 1.0 if cross >= 0 else -1.0
                    rot   = sign * max_turn_rad
                    c, s  = np.cos(rot), np.sin(rot)
                    clamped_dir = np.array([
                        c * tangent[0] - s * tangent[1],
                        s * tangent[0] + c * tangent[1],
                    ])
                    target = anchor + clamped_dir * dist
        return target

    def _local_tangent(pts_near, pts_far):
        """Unit direction of travel from pts_far toward/through pts_near,
        i.e. the direction the arch is heading as it approaches this end
        -- used so the extension continues outward rather than doubling
        back."""
        v = pts_near - pts_far
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else None

    n_waypoints = 4   # interpolation points from molar to ramus

    if len(mandible_vox) > 50:
        # Right ramus (beyond rightmost detected molar, high X)
        tangent_r = (_local_tangent(centroids[0], centroids[1])
                    if len(centroids) >= 2 else None)
        ramus_r = _ramus_centroid(mandible_vox, centroids[0, 0], 'right',
                                  anchor=centroids[0], tangent=tangent_r)
        if ramus_r is not None:
            ts = np.linspace(0.0, 1.0, n_waypoints + 1)[1:]   # exclude molar
            ext_r = np.column_stack([
                centroids[0, 0] + ts * (ramus_r[0] - centroids[0, 0]),
                centroids[0, 1] + ts * (ramus_r[1] - centroids[0, 1]),
            ])[::-1]  # prepend in far→near order
            centroids = np.vstack([ext_r, centroids])

        # Left ramus (beyond leftmost detected molar, low X) -- note the
        # left-end anchor/tangent are computed from the tooth-only
        # centroids, unaffected by the right-side prepend above.
        left_anchor = centroids[-1]
        left_prev   = centroids[-2]
        tangent_l   = _local_tangent(left_anchor, left_prev)
        ramus_l = _ramus_centroid(mandible_vox, centroids[-1, 0], 'left',
                                  anchor=left_anchor, tangent=tangent_l)
        if ramus_l is not None:
            ts = np.linspace(0.0, 1.0, n_waypoints + 1)[1:]
            ext_l = np.column_stack([
                centroids[-1, 0] + ts * (ramus_l[0] - centroids[-1, 0]),
                centroids[-1, 1] + ts * (ramus_l[1] - centroids[-1, 1]),
            ])
            centroids = np.vstack([centroids, ext_l])
    else:
        # Fallback: horizontal extension to jaw bbox when mandible not segmented
        jaw_2d = np.isin(mask, _ALL_JAW_IDS).any(axis=2)
        jx, _  = np.where(jaw_2d)
        if len(jx) > 0:
            x_hi = float(jx.max())
            x_lo = float(jx.min())
        else:
            span = centroids[:, 0].max() - centroids[:, 0].min()
            x_hi = centroids[:, 0].max() + span * 0.15
            x_lo = centroids[:, 0].min() - span * 0.15

        if x_hi > centroids[0, 0]:
            xs  = np.linspace(x_hi, centroids[0, 0], n_waypoints + 1)[:-1]
            centroids = np.vstack([
                np.column_stack([xs, np.full(n_waypoints, centroids[0, 1])]),
                centroids
            ])
        if x_lo < centroids[-1, 0]:
            xs  = np.linspace(centroids[-1, 0], x_lo, n_waypoints + 1)[1:]
            centroids = np.vstack([
                centroids,
                np.column_stack([xs, np.full(n_waypoints, centroids[-1, 1])])
            ])

    # Remove near-duplicates after extension
    dists     = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    keep      = np.concatenate([[True], dists > 0.5])
    centroids = centroids[keep]
    if len(centroids) < 4:
        return None

    # ── Fold-detection safety net ──────────────────────────────────────────────
    # Backstop beyond the tangent clamp above: if a ramus extension still
    # reverses sharply (cosine similarity < -0.5 -- well past any legitimate
    # anatomical turn like the canine bend), the spline fit would run through
    # the fold and curved_mip would sample it with a flipped normal,
    # producing a mirrored duplicate tooth. Trim the offending extension
    # points off that end rather than let them reach the fit.
    #
    # Only the `window` segments at THIS end are inspected, and only points
    # OUTSIDE the fold are dropped. The previous version scanned the whole
    # polyline and cut at the first fold wherever it sat, so a single kink in
    # the middle of the arch -- which two adjacent molar centroids a few
    # voxels apart are enough to produce -- discarded every point on one side
    # of it; the second call then did the same to the remainder, leaving
    # fewer than 4 points, and detect_arch_curve returned None. That is
    # exactly why A034/A092/F006/S0027 produced no panoramic at all. A fold in
    # the MIDDLE of the arch is not a bad extension and trimming cannot fix
    # it: it is an ordering artefact, handled by _order_arch_points above and
    # by the smoothing fit below. Trimming stays for what it was for -- a bad
    # extension, always within a few points of an end -- and it never cuts
    # the point set below `min_keep`, so it can no longer turn a recoverable
    # case into a skipped one.
    def _trim_fold(pts, from_end, window=n_waypoints + 2, min_keep=4):
        if len(pts) < 3:
            return pts
        seq = pts if from_end == 'start' else pts[::-1]
        tang  = np.diff(seq, axis=0)
        norms = np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-8)
        tang  = tang / norms
        dots  = np.sum(tang[:-1] * tang[1:], axis=1)[:window]
        bad   = np.where(dots < -0.5)[0]
        if len(bad) == 0:
            return pts
        # bad[-1] is the tangent index before the LAST fold inside the window
        # (an extension can kink more than once); keep everything inward of it.
        cut = bad[-1] + 1
        if len(seq) - cut < min_keep:
            return pts
        seq = seq[cut:]
        return seq if from_end == 'start' else seq[::-1]

    centroids = _trim_fold(centroids, from_end='start')
    centroids = _trim_fold(centroids, from_end='end')
    if len(centroids) < 4:
        return None

    # ── Fit spline through extended centroids ─────────────────────────────────
    # SMOOTHING fit (s > 0), not exact interpolation (s = 0). An exact fit
    # reproduces every wobble of the centroid polyline, and two neighbouring
    # molar centroids a few voxels apart are enough to put a near-zero-radius
    # kink in the result. That kink is segmentation noise, not anatomy, and it
    # costs real image quality: _slab_depth_scale then has to collapse the slab
    # there to stop the offset rays crossing (down to 0.01x of slab_depth on
    # the worst observed case, i.e. that stretch of the panoramic gets almost
    # no MIP depth). `s` follows splprep's convention of n * tol^2 -- the fit
    # may sit up to ~tol voxels off each centroid, well inside the uncertainty
    # of a centroid taken from a segmentation mask in the first place.
    # Falls back to the exact fit if the smoothed one won't fit.
    k = min(3, len(centroids) - 1)
    tck = None
    for s_try in (len(centroids) * float(smooth_tol_vox) ** 2, 0.0):
        try:
            tck, _ = splprep([centroids[:, 0], centroids[:, 1]], k=k, s=s_try)
            break
        except Exception:
            continue
    if tck is None:
        return None

    u_fine = np.linspace(0.0, 1.0, n_points)
    return np.column_stack(splev(u_fine, tck))


# ── Arc-length resampling ─────────────────────────────────────────────────────

def resample_arch_arclength(arch_curve, in_plane_mm, step_mm):
    """
    Resample the arch curve so consecutive points are `step_mm` apart in TRUE
    PHYSICAL ARC LENGTH. Returns (resampled_curve, total_arc_length_mm).

    This is what makes the output's aspect ratio correct. `detect_arch_curve`
    returns points evenly spaced in the SPLINE PARAMETER u, which is not the
    same as evenly spaced along the curve -- splprep's parameterisation runs
    faster through the straighter stretches -- so a column of the panoramic
    covered a varying physical distance along the arch, stretching some
    regions relative to others within one image.

    On top of that, the old code chose the number of arch points as
    `z_height * target_aspect`, i.e. it forced every case's raster to a fixed
    3:1 whatever that case's actual arch-length-to-jaw-height ratio was. A
    short-Z scan got few columns and its teeth came out horizontally
    squashed; a tall-Z scan got the opposite. Fixing the column spacing to a
    physical `step_mm` (the same mm the rows are sampled at, see `z_step` in
    process_case) makes the pixel square in millimetres, so the image's aspect
    ratio is whatever the anatomy's actually is -- which is the only aspect
    that is "right".

    `in_plane_mm` is (x_mm_per_voxel, y_mm_per_voxel); the curve's coordinates
    are voxel indices, so the two axes have to be scaled into mm before the
    segment lengths mean anything.
    """
    scale = np.asarray(in_plane_mm, dtype=np.float64).reshape(1, 2)
    d     = np.diff(arch_curve, axis=0) * scale
    seg   = np.hypot(d[:, 0], d[:, 1])

    # Drop zero-length segments: np.interp needs a strictly increasing xp, and
    # a duplicated spline point would otherwise make the lookup ambiguous.
    keep      = np.concatenate([[True], seg > 1e-9])
    curve     = arch_curve[keep]
    if len(curve) < 2:
        return arch_curve, 0.0
    d   = np.diff(curve, axis=0) * scale
    seg = np.hypot(d[:, 0], d[:, 1])
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0 or not np.isfinite(total):
        return arch_curve, 0.0

    n_out = max(2, int(round(total / float(step_mm))) + 1)
    s_out = np.linspace(0.0, total, n_out)
    return np.column_stack([np.interp(s_out, s, curve[:, 0]),
                            np.interp(s_out, s, curve[:, 1])]), total


# ── Shared normal computation ─────────────────────────────────────────────────

def _smooth_normals(arch_curve, smooth_pts=5):
    """
    Compute per-point in-plane normals for the arch curve and smooth them.

    Raw finite-difference normals from an interpolating spline can be very
    jumpy at irregularly-spaced knots, creating bright/dark seam artefacts in
    the curved MIP.  A small box smooth over `smooth_pts` arch points removes
    the spike without meaningfully shifting the normals.

    `smooth_pts` is a POINT COUNT, and the caller has to set it from the arch
    point spacing rather than leave it at the default: since arch points are
    now spaced by a physical `step_mm` that shrinks as output resolution
    rises, a fixed count would smooth over a shrinking physical distance and
    the normals would get noisier the sharper the image was asked to be.
    process_case converts a fixed ~1.6 mm of arc into the right count.
    """
    tangents = np.gradient(arch_curve, axis=0)
    norms    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / norms

    # Perpendicular (in-plane normal)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Smooth both components independently
    smooth_pts = max(1, int(smooth_pts))
    normals[:, 0] = uniform_filter1d(normals[:, 0], size=smooth_pts)
    normals[:, 1] = uniform_filter1d(normals[:, 1], size=smooth_pts)

    # Re-normalise after smoothing
    norms   = np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return normals / norms


# ── Curvature-adaptive slab depth ────────────────────────────────────────────
#
# Extra safety net on top of the fold-proofed arch curve above: even a curve
# that never folds on itself can still bend tightly enough (radius smaller
# than slab_depth) that the OFFSET curve used for slab sampling crosses
# itself. Where that happens, shrink the offset locally (tapered smoothly, so
# there's no visible seam) rather than let it fold.

def _local_curvature_radius(arch_curve, smooth_pts=9):
    """
    Local radius of curvature at each arch point, from smoothed first/second
    finite differences. This formula (|x'y'' - y'x''| / (x'^2+y'^2)^1.5) is
    invariant to how the curve is parameterised, so the radius comes out in
    the curve's own coordinate units -- voxels -- matching slab_depth. Smoothed
    on both derivative components so noisy per-point differences don't produce
    a spuriously tiny radius.

    `smooth_pts` is a POINT COUNT and must be scaled by the caller with the
    arch point spacing, for the same reason as in `_smooth_normals`. It
    matters more here: too short a window reports a spuriously tiny radius,
    `_slab_depth_scale` then collapses the slab there, and that stretch of
    the panoramic loses almost all of its MIP depth.
    """
    smooth_pts = max(1, int(smooth_pts))
    d1 = np.gradient(arch_curve, axis=0)
    d1x = uniform_filter1d(d1[:, 0], size=smooth_pts)
    d1y = uniform_filter1d(d1[:, 1], size=smooth_pts)
    d2x = uniform_filter1d(np.gradient(d1x), size=smooth_pts)
    d2y = uniform_filter1d(np.gradient(d1y), size=smooth_pts)

    speed2 = (d1x ** 2 + d1y ** 2).clip(1e-8)
    curvature = np.abs(d1x * d2y - d1y * d2x) / np.power(speed2, 1.5)
    return 1.0 / curvature.clip(1e-6)


def _slab_depth_scale(arch_curve, slab_depth, safety=0.85, smooth_pts=9):
    """
    Per-arch-point scale factor in (0, 1] applied to the ±slab_depth offset
    range, so the slab shrinks wherever the local radius of curvature is
    smaller than slab_depth (times a safety margin) and stays at full width
    everywhere else. Tapered with a further smoothing pass so the shrink
    comes in gradually instead of as a hard seam.

    Computed once per case and shared between curved_mip and build_overlay,
    so the grayscale image and the label mask it's annotated with are
    always sampled from identical geometry.
    """
    smooth_pts = max(1, int(smooth_pts))
    radius = _local_curvature_radius(arch_curve, smooth_pts=smooth_pts)
    scale = np.clip((radius * safety) / slab_depth, 0.0, 1.0)
    return uniform_filter1d(scale, size=smooth_pts)


# ── Step 2-3 — Curved slab sampling + MIP ────────────────────────────────────

def _slab_offsets(arch_curve, slab_depth, depth_scale, normals, depth_step=1):
    """
    Shared geometry for curved_mip and build_overlay: the (n_pts, n_d) XY
    voxel coordinates of every sample in the curved slab. Factored out so the
    grayscale image and the label mask can never drift apart, and so both
    samplers can chunk over the same arrays.
    """
    d     = np.arange(-slab_depth, slab_depth + 1, depth_step, dtype=np.float32)
    d_eff = depth_scale[:, np.newaxis] * d[np.newaxis, :]     # (n_pts, n_d)
    slab_x = arch_curve[:, 0:1] + d_eff * normals[:, 0:1]
    slab_y = arch_curve[:, 1:2] + d_eff * normals[:, 1:2]
    return slab_x.astype(np.float32), slab_y.astype(np.float32)


def _z_samples(shape_z, z_lo, z_hi, z_step):
    """Row sample positions down Z, in (possibly fractional) voxel units."""
    z_lo = 0.0     if z_lo is None else float(z_lo)
    z_hi = shape_z if z_hi is None else float(z_hi)
    step = max(1e-3, float(z_step))
    return np.arange(z_lo, z_hi, step, dtype=np.float32)


def curved_mip(volume, arch_curve, slab_depth=28, z_lo=None, z_hi=None,
               depth_scale=None, normals=None, z_step=1.0, chunk_pts=128):
    """
    Sample a curved slab along the arch and compute a MIP.

    `depth_scale` / `normals`, if given, are the per-arch-point arrays from
    `_slab_depth_scale` / `_smooth_normals` -- pass the same ones used for
    `build_overlay` on this case so the image and label mask stay
    geometrically identical. If omitted they're computed here from
    `arch_curve` directly, but only at the DEFAULT smoothing window, which is
    wrong once arch points are spaced finer than ~0.3 mm; process_case always
    passes them explicitly.

    `z_step` is the row spacing in voxel units and may be fractional -- that
    is how the output gets a square physical pixel (process_case sets it to
    px_mm / z_voxel_mm) instead of being locked to one row per Z voxel.

    Sampling is chunked over arch points (`chunk_pts` at a time) because the
    coordinate arrays are the memory bottleneck: at the resolutions this now
    renders at, materialising all of n_pts × n_d × n_z at once would need
    over a gigabyte for the coordinates alone.

    Returns (n_z, n_arch_pts) float32, Z-flipped so superior is at row 0.
    """
    n_pts = len(arch_curve)
    n_d   = 2 * slab_depth + 1
    z_idx = _z_samples(volume.shape[2], z_lo, z_hi, z_step)
    n_z   = len(z_idx)

    if normals is None:
        normals = _smooth_normals(arch_curve)
    if depth_scale is None:
        depth_scale = _slab_depth_scale(arch_curve, slab_depth)
    slab_x, slab_y = _slab_offsets(arch_curve, slab_depth, depth_scale, normals)

    out  = np.empty((n_pts, n_z), dtype=np.float32)
    cval = float(volume.min())
    for a in range(0, n_pts, chunk_pts):
        b  = min(a + chunk_pts, n_pts)
        nc = b - a
        cx = np.repeat(slab_x[a:b].ravel(), n_z)
        cy = np.repeat(slab_y[a:b].ravel(), n_z)
        cz = np.tile(z_idx, nc * n_d)
        vals = map_coordinates(
            volume, [cx, cy, cz], order=1, mode='constant', cval=cval,
        ).reshape(nc, n_d, n_z)
        out[a:b] = vals.max(axis=1)

    return out.T[::-1, :]   # (n_z, n_pts), superior at top


# ── Step 4 — Window/level ─────────────────────────────────────────────────────

def enamel_ceiling(volume, mask, headroom=1.15):
    """
    Estimate the intensity of the brightest NATURAL tooth tissue in this
    volume, so everything above it (metal crowns, posts, implants, amalgam,
    and the streak artefacts they throw) can be clipped before projection.

    Why this is the fix for the dark panoramics
    -------------------------------------------
    The MIP takes the maximum along ~8 mm of slab depth, so a single metal
    restoration does not stay the size of the restoration: it wins the max for
    every ray that passes anywhere near it, and its streaks win over a wider
    area still. In a restored case that puts a bright population across a
    non-trivial fraction of the image -- easily more than the 0.5% that the
    old `np.percentile(arr, 99.5)` window ceiling was chosen to discard. The
    ceiling therefore landed inside the metal rather than at enamel, and
    every real structure got mapped into the bottom of the 0-255 range. That
    is the "too dark" case, and it is worst exactly on the cases with the most
    to read.

    Estimator: the MEDIAN over teeth of each tooth's own 98th percentile. A
    per-tooth peak is enamel for a natural tooth and metal for a restored one,
    and taking the median across teeth means the estimate only moves if MOST
    of the dentition is restored -- unlike a percentile pooled over all tooth
    voxels at once, which a handful of metal crowns would drag upward (metal
    is far denser than enamel, so those voxels sit at the very top of the
    pooled distribution and a few teeth's worth is enough to occupy the top
    percentile).

    The estimate is then capped against the jaw bone (see `_cap_against_bone`),
    which catches the one case a median across teeth cannot: a dentition where
    the restored teeth are the MAJORITY, so the median peak is metal itself.

    Falls back to the jaw bone's own high percentile with extra headroom when
    there are too few teeth to take a median over (edentulous cases), and to a
    plain global percentile when there is no usable segmentation at all.
    Returns None only if the volume itself is degenerate.
    """
    jaw_sel = np.isin(mask, _JAW_BODY_IDS)
    bone_p99 = (float(np.percentile(volume[jaw_sel], 99.0))
                if int(jaw_sel.sum()) >= 200 else None)

    tooth_sel = np.isin(mask, _ALL_TOOTH_TIDS)
    n_tooth   = int(tooth_sel.sum())
    if n_tooth >= 200:
        # Compact once, then group -- one pass over the volume rather than one
        # boolean pass per FDI label.
        ids  = mask[tooth_sel]
        vals = volume[tooth_sel]
        peaks = []
        for tid in np.unique(ids):
            v = vals[ids == tid]
            if v.size >= 50:
                peaks.append(np.percentile(v, 98.0))
        est = None
        if len(peaks) >= 4:
            est = float(np.median(peaks)) * headroom
        elif peaks:
            # Too few teeth to trust a median: any one of them could be the
            # restored one, so take the lowest peak rather than the middle.
            est = float(np.min(peaks)) * headroom
        if est is not None:
            return _cap_against_bone(est, bone_p99)

    if bone_p99 is not None:
        # Cortical bone peaks well below enamel, so this needs headroom to
        # avoid clipping real crown tissue. 1.6 is the middle of the measured
        # enamel/bone-p99 range (see _cap_against_bone).
        return bone_p99 * 1.6

    hi = float(np.percentile(volume, 99.9))
    return hi if np.isfinite(hi) else None


# Measured across the 19-case validate split: ceiling / jaw-bone-p99 sits at
# 1.41-2.15 for 17 of 19 cases, and 1.4-1.85 for the metal-free ones. The two
# exceptions are the failure mode the median across teeth is documented not to
# catch -- a dentition where the RESTORED teeth are the majority, so the median
# peak is itself metal: A022 (10 teeth left, nearly all crowned) at 2.60, and
# A041 (18 teeth, most of them restored) at 5.48, whose ceiling landed above
# the volume's own maximum and therefore clipped nothing at all.
#
# Jaw bone is immune to that: it is a large structure, it is never restored,
# and its p99 is cortical bone in every case. So it makes a sound upper bound
# on what enamel can plausibly be. 2.5 clears the highest observed
# genuinely-enamel ratio by a wide margin, so it is inert on normal cases and
# only engages where the tooth-based estimate has demonstrably run away.
_BONE_CAP_RATIO = 2.5


def _cap_against_bone(estimate, bone_p99):
    """Bound a tooth-derived enamel ceiling by what the jaw bone implies."""
    if bone_p99 is None or not np.isfinite(bone_p99) or bone_p99 <= 0:
        return estimate
    return float(min(estimate, _BONE_CAP_RATIO * bone_p99))


def window_to_uint8(arr, low_pct=1.0, high_pct=99.5, fg_frac=0.10):
    """
    Percentile window to uint8, with the percentiles taken over FOREGROUND
    pixels only.

    The old version ran them over the whole image. A panoramic is mostly air:
    the region outside the head, the space between the jaws, and everything
    beyond the arch's lateral ends. Those pixels dominate the histogram, so
    the low percentile sat in air and a large part of the 0-255 range was
    spent separating shades of nothing, leaving real tissue compressed into
    what was left. Restricting the percentiles to pixels above `fg_frac` of
    the image's own range puts the black point at the bottom of actual tissue
    instead; air falls below it and clips to black, which is where it belongs.

    Pair this with `enamel_ceiling` clipping on the volume BEFORE projection
    (see process_case) -- that handles the top of the range, this handles the
    bottom, and neither substitutes for the other.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    air    = np.percentile(finite, 1.0)
    top    = np.percentile(finite, 99.9)
    thresh = air + fg_frac * (top - air)
    fg     = finite[finite > thresh]
    if fg.size < 0.02 * finite.size:
        fg = finite          # almost nothing above threshold -- don't over-fit

    lo  = np.percentile(fg, low_pct)
    hi  = np.percentile(fg, high_pct)
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def apply_clahe(img_u8, clip_limit=0.01, blend=0.6, tiles=(6, 12),
                dark_floor=0.12):
    """
    Blend in contrast-limited adaptive histogram equalisation, so root apices
    and the ramus read as well as the crowns do rather than sitting in a flat
    low-contrast band -- a linear window can only place ONE contrast slope
    across a projection whose density varies by an order of magnitude from
    enamel to cancellous bone.

    Three deliberate restraints, because unrestrained CLAHE on a projection
    radiograph looks wrong and invents texture:
      * `clip_limit` is low (0.01), the mild end of skimage's range;
      * the result is BLENDED with the linear window rather than replacing it,
        so the image is still mostly a straight windowing of real values;
      * the blend is faded out below `dark_floor` of the range, so the air
        background stays black instead of being equalised up into grey noise
        (CLAHE's usual failure mode on images with a large empty region).

    Degrades to a no-op, with a warning, if skimage isn't importable --
    everything else about the panoramic is unaffected.
    """
    if blend <= 0:
        return img_u8
    try:
        from skimage.exposure import equalize_adapthist
    except ImportError:
        print("  [warn] skimage not available -- skipping CLAHE "
              "(linear windowing only)", file=sys.stderr)
        return img_u8

    h, w = img_u8.shape
    kernel = (max(8, h // tiles[0]), max(8, w // tiles[1]))
    eq = equalize_adapthist(img_u8, kernel_size=kernel, clip_limit=clip_limit)
    eq = np.clip(eq, 0.0, 1.0) * 255.0

    base = img_u8.astype(np.float32)
    # Fade the effect in over the darkest part of the range instead of
    # switching it on at a hard threshold, which would draw a visible contour
    # around the anatomy at exactly the dark_floor level.
    fade = np.clip(base / (255.0 * max(dark_floor, 1e-3)), 0.0, 1.0)
    out  = base + (blend * fade) * (eq.astype(np.float32) - base)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Segmentation overlay ──────────────────────────────────────────────────────

def build_overlay(mask, arch_curve, slab_depth, volume_shape,
                  z_lo=None, z_hi=None, depth_scale=None, normals=None,
                  z_step=1.0, depth_step=2, chunk_pts=64):
    """
    Sample segmentation labels along the curved slab.

    `depth_scale` / `normals` / `z_step`, if given, are the per-case geometry
    from `_slab_depth_scale` / `_smooth_normals` / process_case -- pass the
    same ones used for `curved_mip` so the label mask stays geometrically
    identical to the grayscale image it's annotating. If omitted they're
    computed here from `arch_curve` directly.

    `depth_step` subsamples the slab depth (every 2nd offset by default). The
    vote only needs to know which label OCCUPIES each ray, and a tooth spans
    far more than two voxels of depth, so halving the samples changes the
    winner essentially never while halving the cost of the most expensive
    step in the file. `curved_mip` deliberately does NOT do this -- a MIP's
    maximum can genuinely live in a single voxel.

    Returns (n_z, n_arch_pts) int32 label array (Z-flipped, superior at row 0).

    The raw mask is the FULL multi-class segmentation (teeth + pulp + jaw
    bone), not just tooth crowns. We restrict it to the 32 tooth-CROWN FDI
    labels (_ALL_TOOTH_TIDS) with one simple filter up front -- anything
    else (background, jaw bone, pulp, any other class) becomes 0/background
    before sampling even starts. That keeps the majority-vote step itself
    simple: every remaining label is already <= 48, so it can be used
    directly as a bin index with no separate lookup/remapping logic.
    """
    n_pts = len(arch_curve)
    z_idx = _z_samples(volume_shape[2], z_lo, z_hi, z_step)
    n_z   = len(z_idx)

    if normals is None:
        normals = _smooth_normals(arch_curve)
    if depth_scale is None:
        depth_scale = _slab_depth_scale(arch_curve, slab_depth)
    slab_x, slab_y = _slab_offsets(arch_curve, slab_depth, depth_scale, normals,
                                   depth_step=max(1, int(depth_step)))
    n_d = slab_x.shape[1]

    # Keep only tooth-crown labels; everything else -> 0/background.
    tooth_only_mask = np.where(np.isin(mask, _ALL_TOOTH_TIDS), mask, 0).astype(np.float32)

    MAX_LBL = 49   # FDI labels are 11-48; 49 is a safe upper bound + 1
    out = np.zeros((n_pts, n_z), dtype=np.int32)

    for a in range(0, n_pts, chunk_pts):
        b  = min(a + chunk_pts, n_pts)
        nc = b - a
        cx = np.repeat(slab_x[a:b].ravel(), n_z)
        cy = np.repeat(slab_y[a:b].ravel(), n_z)
        cz = np.tile(z_idx, nc * n_d)
        labels = map_coordinates(
            tooth_only_mask, [cx, cy, cz], order=0, mode='constant', cval=0,
        ).reshape(nc, n_d, n_z).astype(np.int32)

        # ── Majority vote over slab depth ─────────────────────────────────────
        # One bincount over a (point, z, label) code replaces the previous
        # per-depth np.add.at loop. np.add.at is an unbuffered scatter and
        # costs per ELEMENT; at these resolutions that loop was the dominant
        # cost of the whole run.
        pi, di, zi = np.nonzero(labels)
        if len(pi) == 0:
            continue
        codes = (pi.astype(np.int64) * n_z + zi) * MAX_LBL + labels[pi, di, zi]
        counts = np.bincount(codes, minlength=nc * n_z * MAX_LBL)
        out[a:b] = counts.reshape(nc, n_z, MAX_LBL).argmax(axis=2)

    return out.T[::-1, :]   # (n_z, n_pts), Z-flip: superior at top


def _load_font(size):
    """
    Bold sans at `size`, falling back through the usual system paths, then
    matplotlib's bundled DejaVu, and only then to PIL's built-in font.

    The matplotlib fallback matters now that the tags are drawn at ~3x the
    old point size: PIL's built-in is a fixed-size bitmap font that does not
    scale, so on a host with no system TTFs the tags would come out as tiny
    aliased digits on a large image. matplotlib ships DejaVuSans-Bold and is
    already a transitive dependency here (skimage).
    """
    from PIL import ImageFont
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
               "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
               "C:/Windows/Fonts/arialbd.ttf"]:
        try:   return ImageFont.truetype(fp, size)
        except (IOError, OSError): pass
    try:
        import matplotlib
        return ImageFont.truetype(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" /
                "DejaVuSans-Bold.ttf"), size)
    except Exception:
        pass
    try:    return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def _canonical_lr_rank(fdi):
    """
    Anatomical left-to-right rank for a tooth's FDI number, independent of
    its (possibly noisy) rasterized pixel column -- a safety net so tag
    order can never come out wrong even in some future case with residual
    curvature. FDI numbering is a fixed, known, monotonic sequence along
    each quadrant, so ranking by it can never be thrown off by per-case
    geometric noise the way sorting by raw pixel `cx` could be.

    Standard convention (post orientation-fix, patient's right on image
    left): quadrants 1/4 (11-18, 41-48) run distal-to-mesial LEFT to RIGHT,
    i.e. HIGHER FDI is further left -- so rank = -fdi. Quadrants 2/3
    (21-28, 31-38) run mesial-to-distal LEFT to RIGHT, i.e. HIGHER FDI is
    further right -- so rank = fdi.
    """
    quadrant = fdi // 10
    return -fdi if quadrant in (1, 4) else fdi


def apply_overlay(pano_rgb, label_pano, draw_outline=True, outline_width=None,
                  mask_smooth_iterations=None, left_marker_edge=None,
                  font_size=None):
    """
    Draw a colored outline around each tooth directly on the grayscale
    image, and place its FDI tag in a blank margin band above (maxilla) or
    below (mandible) the image. All tags within a band sit in a SINGLE
    horizontally-aligned row (no vertical stagger), coloured to match that
    tooth's own outline colour, with NO leader line connecting tag to
    tooth.

    Correspondence between a tag and its tooth is carried by two
    independent cues instead of a line: (1) horizontal position -- the tag
    sits at approximately the same x as its tooth's centroid, and (2)
    colour -- the tag's text matches the tooth's outline colour
    exactly. This keeps the image visually simpler (no crossing/overlapping
    lines to untangle, especially for tilted/impacted teeth where a line
    would have to travel at an angle) while still giving an unambiguous,
    redundant way to match tag to tooth.

    draw_outline=True by default now (final design keeps the outline on the
    same single image rather than a separate overlay panel). Each tooth's
    mask is smoothed first (binary_opening + binary_closing) so imperfect/
    jagged segmentation doesn't get traced as a falsely precise boundary.

    `left_marker_edge` ('left'/'right', or None to omit) says which edge of
    the image is the PATIENT's left, and draws a white 'L' side marker in
    that corner of the upper tag band -- the standard radiographic way to
    state laterality on the film itself rather than leaving it implicit.

    `outline_width`, `mask_smooth_iterations` and `font_size` are all in
    PIXELS / pixel-iterations, so process_case scales them with the case's
    px_mm; left at None they are derived from the image's own dimensions.
    They cannot be left as the old fixed constants: at the resolutions this
    now renders at, a 2 px outline is a hairline and a 1-iteration
    binary_opening smooths a tenth of the physical distance it used to,
    which would put the jagged edges of the raw segmentation back on the
    image.
    """
    n_z, n_pts = label_pano.shape

    if font_size is None:
        # Tie the tag size to the image height, capped by what the width can
        # actually fit. The cap counts the tags THIS case will draw in its
        # busiest band rather than assuming a full 32-tooth dentition: a case
        # with eight remaining teeth has room for far larger tags, and holding
        # back for teeth that aren't there just makes them harder to read.
        present = [int(v) for v in np.unique(label_pano) if int(v) in TOOTH_COLORS]
        per_band = [sum(1 for f in present if 11 <= f <= 28),
                    sum(1 for f in present if 31 <= f <= 48)]
        busiest = max(max(per_band), 1)
        # ~2 digits at 0.6 em each, plus an em of breathing room between tags.
        width_cap = int(n_pts / (busiest * 2.2))
        font_size = int(np.clip(round(n_z * 0.050), 16, max(16, width_cap)))
    fsize = int(max(9, font_size))
    if outline_width is None:
        outline_width = max(2, int(round(n_z / 350.0)))
    if mask_smooth_iterations is None:
        mask_smooth_iterations = max(1, int(round(n_z / 300.0)))
    font  = _load_font(fsize)
    cw    = fsize * 6 // 10

    # ── Reserve ONE single-row blank margin above and below (root side) ───────
    # The band grows WITH the tag rather than by a fixed 8 px: the padding has
    # to stay proportional or a larger tag ends up jammed against the image
    # edge. Only the canvas height changes -- the panoramic body itself is
    # untouched, so nothing about the rendered anatomy or its scale moves.
    row_h    = fsize + max(8, fsize // 2)  # single row -- tags are horizontally aligned
    canvas_h = n_z + 2 * row_h
    arr      = np.zeros((canvas_h, n_pts, 3), dtype=np.uint8)
    arr[row_h:row_h + n_z, :, :] = np.array(pano_rgb, dtype=np.uint8)

    tooth_jobs = []  # per-tooth: side, cx, color, text, tw

    for lbl in np.unique(label_pano):
        lbl = int(lbl)
        if lbl not in TOOTH_COLORS:
            continue
        tooth_mask = (label_pano == lbl)
        rows, cols = np.where(tooth_mask)
        if len(rows) == 0:
            continue

        r, g, b, _a = TOOTH_COLORS[lbl]
        color = (r, g, b)

        if draw_outline:
            smoothed = tooth_mask
            if mask_smooth_iterations > 0:
                smoothed = binary_opening(smoothed, iterations=mask_smooth_iterations)
                smoothed = binary_closing(smoothed, iterations=mask_smooth_iterations)
                if smoothed.sum() == 0:
                    smoothed = tooth_mask
            # Force outer boundary only: fill any interior hole (e.g. a gap
            # where a different class -- pulp, a segmentation dropout --
            # sits inside the tooth) before finding the boundary, so erosion
            # never has an inner ring to detect in the first place.
            smoothed = binary_fill_holes(smoothed)
            eroded  = binary_erosion(smoothed, iterations=1)
            outline = smoothed & ~eroded
            if outline_width > 1:
                outline = binary_dilation(outline, iterations=outline_width - 1)
            oy, ox = np.where(outline)
            arr[row_h + oy, ox] = color

        is_upper = 11 <= lbl <= 28
        # Anchor the tag at the tooth's MOST DISTAL point (dental terminology:
        # away from the midline, toward the back of the arch) rather than its
        # centroid. Quadrants I/IV (11-18, 41-48) sit on the image's LEFT
        # after the mirror fix, and within them FDI number increases distally
        # moving further left -- so distal = the tooth's leftmost (min)
        # column. Quadrants II/III (21-28, 31-38) sit on the image's RIGHT,
        # where FDI number increases distally moving further right -- so
        # distal = the tooth's rightmost (max) column. A centroid can land
        # in a visually misleading spot for an irregular/tilted tooth (e.g.
        # a rotated wisdom tooth); the distal extreme is a more consistent,
        # shape-independent reference point, and it's also the natural edge
        # to align against the NEXT tooth going toward the back of the arch.
        quadrant = lbl // 10
        if quadrant in (1, 4):
            cx = int(cols.min())
        else:  # quadrant in (2, 3)
            cx = int(cols.max())
        text = str(lbl)
        tw   = len(text) * cw
        tooth_jobs.append(dict(side="upper" if is_upper else "lower",
                               cx=cx, tw=tw, text=text, color=color, fdi=lbl))

    img    = Image.fromarray(arr, mode="RGB")
    draw_r = ImageDraw.Draw(img)

    # ── Patient-side marker ──────────────────────────────────────────────────
    # Radiographs are annotated with the PATIENT's side, not the viewer's, so
    # the letter has to follow whichever edge the patient's left actually
    # landed on. That side comes from the caller's `_side_anchor` / quadrant
    # determination -- the same one used for the orientation fallback -- so
    # it is read off this case's own data rather than assumed from the dataset.
    #
    # It goes in the upper tag band, not over the image: the band is
    # guaranteed-blank, so a white letter there can never be mistaken for
    # anatomy or obscure a tooth the way an in-image corner marker would.
    # Its footprint is reserved BEFORE the tags are laid out (below), so the
    # outermost FDI tag is nudged clear of it rather than drawn underneath.
    draw_marker = left_marker_edge in ("left", "right")
    if draw_marker:
        m_size = fsize + 4                      # bigger than a tag, still fits the band
        m_font = _load_font(m_size)
        m_pad  = max(3, fsize // 3)
        try:
            m_w = int(draw_r.textlength("L", font=m_font))
        except AttributeError:                  # Pillow < 8
            m_w = int(m_size * 0.6)
        m_reserve = m_w + 2 * m_pad
    else:
        m_reserve = 0

    # ── Place tags: ONE horizontally-aligned row per band, nudged only
    #    horizontally to avoid overlapping a neighbour -- no stagger, no
    #    leader lines, no box -- just the colored digits directly on the
    #    black margin, colour matching the tooth's outline exactly ─────────────
    #
    # Two passes, not one. A single left-to-right pass can only push tags
    # RIGHT, so crowding accumulates toward the right edge and the last tags
    # arrive with nowhere to go; clamping them back inside the border then
    # undoes the very nudge that was keeping them apart, and they overprint
    # each other (the most distal tag, 28/38, and its neighbour). The
    # backward pass distributes that pressure back leftward instead, which is
    # where the free space actually is.
    for side in ("upper", "lower"):
        # Sort by canonical anatomical rank (FDI-based), NOT raw pixel cx --
        # see _canonical_lr_rank's docstring for why. cx is still used below
        # for each tag's actual on-image x-position.
        side_jobs = sorted([j for j in tooth_jobs if j["side"] == side],
                           key=lambda j: _canonical_lr_rank(j["fdi"]))
        if not side_jobs:
            continue

        # _canonical_lr_rank encodes the STANDARD convention, which holds only
        # once the L-R flip has been applied. `_needs_lr_flip` returns None
        # when the segmentation has no quadrant pair to vote with (a case with
        # teeth on one side only), and then no flip happens and the band runs
        # the other way. Rank order would then be the exact reverse of pixel
        # order, and the two passes below -- which assume they agree -- would
        # sweep every tag to one edge in a heap.
        #
        # So take the ORDER from the FDI rank (per-tooth robustness, the whole
        # point of _canonical_lr_rank) but the DIRECTION from the aggregate of
        # the columns, which is a majority vote over the whole band and so is
        # just as immune to one tooth's noisy centroid.
        if len(side_jobs) >= 2:
            steps = np.diff([j["cx"] for j in side_jobs])
            if int((steps < 0).sum()) > int((steps > 0).sum()):
                side_jobs = side_jobs[::-1]

        label_cy = row_h // 2 if side == "upper" else canvas_h - row_h // 2

        # Only the upper band carries the marker, so only it loses room.
        res = m_reserve if side == "upper" else 0
        x_lo = 2 + (res if left_marker_edge == "left"  else 0)
        x_hi = n_pts - 2 - (res if left_marker_edge == "right" else 0)

        gap    = max(3, fsize // 3)
        widths = [j["tw"] for j in side_jobs]
        left   = [j["cx"] - j["tw"] / 2.0 for j in side_jobs]   # desired left edges
        n_tag  = len(side_jobs)

        # Forward: honour the left border and keep each tag clear of the one
        # before it.
        left[0] = max(left[0], x_lo)
        for i in range(1, n_tag):
            left[i] = max(left[i], left[i - 1] + widths[i - 1] + gap)

        # Backward: honour the right border and push the overflow back left.
        left[-1] = min(left[-1], x_hi - widths[-1])
        for i in range(n_tag - 2, -1, -1):
            left[i] = min(left[i], left[i + 1] - widths[i] - gap)

        # If the band is genuinely too full to fit every tag at this size,
        # the backward pass will have pushed the first tag past the left
        # border. Keep them on the canvas and accept the tightness: a tag
        # that is slightly close to its neighbour is still readable, and its
        # colour still identifies its tooth unambiguously, whereas a dropped
        # tag loses that tooth's identity entirely.
        if left[0] < x_lo:
            shift = x_lo - left[0]
            left  = [l + shift for l in left]

        for job, lx in zip(side_jobs, left):
            draw_r.text((int(round(lx)), label_cy - fsize // 2),
                        job["text"], fill=job["color"], font=font)

    if draw_marker:
        m_x = m_pad if left_marker_edge == "left" else n_pts - m_w - m_pad
        draw_r.text((m_x, row_h // 2 - m_size // 2), "L",
                    fill=(255, 255, 255), font=m_font)

    return img


def _needs_lr_flip(label_pano):
    """
    Determine, FROM THE SEGMENTATION ITSELF, whether the panoramic needs a
    left-right flip to match standard dental convention: quadrant I
    (11-18, upper RIGHT) and quadrant IV (41-48, lower RIGHT) should sit on
    the image's LEFT (as if facing the patient); quadrant II (21-28, upper
    LEFT) and quadrant III (31-38, lower LEFT) should sit on the image's
    RIGHT.

    This replaces a single fixed, dataset-specific flip: different source
    datasets/scanners are not guaranteed to share the same raw orientation
    convention, so hard-coding "always flip" (or "never flip") breaks the
    moment a differently-oriented dataset is mixed in. Instead, we read off
    each quadrant's actual mean column position from the FDI labels that
    are already in label_pano and let that vote on whether a flip is
    needed -- correct regardless of whatever orientation convention the
    source volume happened to use.

    Returns True/False, or None when no quadrant pair is present to vote
    with (edentulous cases) -- distinct from False so the caller can fall
    back to the `_side_anchor` determination instead of mistaking "can't
    tell" for "already correct".
    """
    def _mean_col(fdi_range):
        cols = np.where(np.isin(label_pano, list(fdi_range)))[1]
        return float(cols.mean()) if len(cols) else None

    q1 = _mean_col(range(11, 19))   # upper right -- should be LEFT  (low column)
    q2 = _mean_col(range(21, 29))   # upper left  -- should be RIGHT (high column)
    q3 = _mean_col(range(31, 39))   # lower left  -- should be RIGHT (high column)
    q4 = _mean_col(range(41, 49))   # lower right -- should be LEFT  (low column)

    votes = []
    if q1 is not None and q2 is not None:
        votes.append(q2 - q1)   # positive -> already correctly oriented
    if q3 is not None and q4 is not None:
        votes.append(q3 - q4)   # positive -> already correctly oriented

    if not votes:
        return None   # undecidable here -- caller falls back to the arch anchors

    return sum(votes) < 0   # net negative -> quadrants are swapped -> flip needed


# ── Caption ────────────────────────────────────────────────────────────────
#
# Per schema.json (v6.1), `panoramic` is the images_needed target for
# absent_teeth, implants, fixed_bridges, and (implicitly) teeth_present /
# crowns among the six fields currently in scope: teeth_present,
# teeth_absent, crowns, implants, bridge, ian_close_teeth. ian_close_teeth
# maps to the 3D views instead (handled in create_3d_renders.py) -- it does
# not appear here.
#
# The caption is written at generation time (same pattern as
# create_3d_renders.py): a static description of what the image itself
# shows (grayscale CPR, color-coded tooth outlines, FDI tags, L marker),
# plus terse fragments for whichever facts are actually present. A field
# only produces a fragment when it's informative (non-empty list / truthy
# bridge); empty/false/missing fields are silently skipped, never padded
# with "none" text.
#
# CROWNS AND IMPLANTS NO LONGER PRODUCE A FRAGMENT (v7). They used to, and
# the caption then read "Crown on teeth 24 and 45. Endosseous implant in
# position 25." -- which hands the model the answer to the very questions
# the panoramic call asks it (dental_arch_findings_{arch} and
# implants_{arch}). Present/absent stays, because it is not a finding the
# schema scores: it identifies WHICH teeth the tags name, and the outlines
# themselves are already filtered by teeth_present, so the caption only
# says out loud what the pixels already show.

PANORAMIC_STATIC_CAPTION = (
    "Panoramic image reconstructed from the CBCT volume (curved panoramic "
    "reconstruction). Each tooth is outlined in its own distinct color, with "
    "its FDI tag in the margin band above (maxillary) or below (mandibular), "
    "colored to match that tooth's outline. "
    "CHECK EVERY TOOTH NUMBER BEFORE YOU USE IT: match a tag to its tooth by "
    "COLOR, not by horizontal position -- neighbouring tags are shifted "
    "sideways to avoid overlap, so a tag can sit over the wrong tooth. If "
    "position and color disagree, THE COLOR IS CORRECT. "
    "A white 'L' in the margin band marks the patient's left side."
)


def load_case_facts(facts_path):
    """Load a case's facts JSON. Returns {} if no path given or file missing."""
    if not facts_path:
        return {}
    p = Path(facts_path)
    if not p.exists():
        print(f"  [warn] facts file not found: {p}", file=sys.stderr)
        return {}
    import json
    return json.loads(p.read_text())


def _normalize_fdi_list(raw):
    """
    Normalize a structured field representing a set of FDI tooth numbers,
    regardless of whether it's given as a flat list of ints (e.g. [37, 47])
    or a list of objects (e.g. [{"fdi": 14}, ...] / [{"tooth": 14}, ...]).
    Returns an empty set for None / empty / anything that doesn't parse.
    """
    if not raw:
        return set()
    out = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.add(item)
        elif isinstance(item, dict):
            for key in ("fdi", "tooth", "fdi_number", "location"):
                if key in item and isinstance(item[key], int):
                    out.add(item[key])
                    break
    return out


def _extract_present_teeth(facts):
    """
    FDI numbers this case's facts.json declares present
    (structured.teeth_present -- `present_teeth` also accepted as an alias
    in case a facts file uses that key name instead).

    Returns None, not an empty set, when the field itself isn't in the
    facts at all -- callers need to tell "field missing, nothing to filter
    by" apart from "field present but an empty list, this case genuinely
    has zero present teeth" (e.g. edentulous), since those two cases call
    for different behaviour (skip filtering vs. filter down to nothing).
    """
    structured = facts.get("structured", {})
    for key in ("teeth_present", "present_teeth"):
        if key in structured:
            return _normalize_fdi_list(structured.get(key))
    return None


def _join_teeth(fdis):
    """'14, 15 and 18' style join for a sorted list of FDI numbers."""
    nums = sorted(fdis)
    if not nums:
        return ""
    if len(nums) == 1:
        return str(nums[0])
    return ", ".join(str(n) for n in nums[:-1]) + f" and {nums[-1]}"


def _normalize_bridge_spans(structured):
    """
    Look for richer bridge span data under either `bridges` or
    `fixed_bridges` (matching schema.json's fixed_bridges fact shape:
    list[object] with span_start / span_end). Returns a sorted, deduplicated
    list of (span_start, span_end) tuples, or [] if no span data is
    available even when `bridge_present` is true -- caller handles the
    generic fallback in that case.
    """
    spans = []
    for key in ("bridges", "fixed_bridges"):
        raw = structured.get(key)
        if not raw:
            continue
        for item in raw:
            if isinstance(item, dict) and "span_start" in item and "span_end" in item:
                spans.append((item["span_start"], item["span_end"]))
    return sorted(set(spans))


def _fragment_teeth_present(teeth):
    if not teeth:
        return None
    return f"teeth {_join_teeth(teeth)} present"


def _fragment_teeth_absent(teeth):
    if not teeth:
        return None
    return f"teeth {_join_teeth(teeth)} absent (refer to FDI tags on image)"


def _fragment_bridge(bridge_present, spans):
    if spans:
        return " ".join(f"{start}-{end} fixed bridge." for start, end in spans)
    if bridge_present:
        return "Fixed bridge present."
    return None


def build_panoramic_caption(facts):
    """
    Combine the static panoramic description with terse fact-based
    fragments for teeth_present, teeth_absent and bridge. Only informative
    fields contribute text; an empty facts dict returns just the static
    description. crowns and implants are deliberately NOT captioned -- see
    the section comment above.
    """
    structured = facts.get("structured", {})

    teeth_present = _extract_present_teeth(facts) or set()
    teeth_absent = _normalize_fdi_list(structured.get("teeth_absent"))
    bridge_present = bool(structured.get("bridge_present"))
    bridge_spans = _normalize_bridge_spans(structured)

    fragments = [PANORAMIC_STATIC_CAPTION]

    present_absent_parts = [
        p for p in (_fragment_teeth_present(teeth_present), _fragment_teeth_absent(teeth_absent)) if p
    ]
    if present_absent_parts:
        clause = "; ".join(present_absent_parts) + "."
        fragments.append(clause[0].upper() + clause[1:])

    bridge_fragment = _fragment_bridge(bridge_present, bridge_spans)
    if bridge_fragment:
        fragments.append(bridge_fragment)

    return " ".join(fragments)


def letterbox_to_size(img, target_w, target_h, bg_color=(0, 0, 0)):
    """
    Fit `img` into a fixed (target_w, target_h) canvas without distorting
    its aspect ratio: scale down/up to the largest size that fits inside
    the target box (LANCZOS), then center it on a bg_color canvas of
    exactly (target_w, target_h), padding whichever axis falls short.

    OPT-IN ONLY now (--canvas WxH). By default process_case renders the
    panoramic directly at its final pixel size and writes it untouched:
    every resample here costs real detail, and it cost the most where it
    hurt most -- LANCZOS over 2-3 px coloured tooth outlines blurs them into
    the grayscale underneath, and the old default canvas was small enough
    that most cases were being DOWNscaled into it after being rendered.
    """
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


# ── Per-case entry point ──────────────────────────────────────────────────────

def process_case(volume_path, mask_path, out_dir, case_id,
                 n_arch_points=1200, slab_mm=6.0, slab_depth=None,
                 facts_path=None, out_height=900, canvas=None,
                 clahe_blend=0.6):
    """
    Generate panoramic reconstruction for one case.
    Output: {case_id}_panoramic.png -- ONE image: the grayscale panoramic
    with a colored outline drawn around each tooth (smoothed first so
    imperfect segmentation doesn't read as a falsely precise boundary),
    plus its FDI tag placed in a blank margin band above (maxilla) or
    below (mandible). Tags within a band are all horizontally aligned in a
    single row and colored to match their tooth's outline exactly --
    position + color together identify which tag belongs to which tooth,
    with no leader line needed.

    `facts_path` is required (enforced by the CLI, see main()): the case's
    facts.json is the source of truth for which teeth actually get an
    outline + FDI tag, via facts.structured.teeth_present -- any tooth the
    segmentation mask happens to find that ISN'T in that list is
    suppressed rather than drawn, so a segmentation false-positive can
    never appear as a labeled tooth on the image. The case-specific
    findings (teeth_present, teeth_absent, crowns, implants, bridge -- see
    build_panoramic_caption) are also appended to the static image
    description and written to {case_id}_panoramic_caption.json alongside
    the PNG, so the caption is produced at generation time rather than a
    separate post-hoc captioning pass.

    Geometry parameters
    -------------------
    `out_height` is the panoramic BODY height in pixels (tag bands add to it);
    it sets the physical pixel size px_mm, which is then used for BOTH the
    row spacing and the arch column spacing, so the pixel is square in
    millimetres and the image's aspect ratio is the anatomy's own.
    `slab_mm` is the slab HALF-thickness in millimetres (`slab_depth`
    overrides it with an explicit voxel count). Expressing it in mm rather
    than voxels is what makes it mean the same thing across scans acquired at
    different resolutions.
    `canvas` is an optional (w, h) to letterbox into; None writes the
    natively-rendered image with no resampling at all.
    """
    print(f"[{case_id}] loading ...", file=sys.stderr)
    vol_nib  = nib.as_closest_canonical(nib.load(str(volume_path)))
    mask_nib = nib.as_closest_canonical(nib.load(str(mask_path)))
    volume   = vol_nib.get_fdata().astype(np.float32)[::-1, :, :]
    mask     = mask_nib.get_fdata().astype(np.int32)[::-1, :, :]

    # Voxel size in mm along each axis, post-canonical-reorient. Everything
    # geometric below is expressed in mm and converted through these, so a
    # scan at 0.2 mm and one at 0.4 mm produce panoramics at the same
    # physical scale instead of the same voxel count.
    zooms = vol_nib.header.get_zooms()[:3]
    sx, sy, sz = (float(z) if (np.isfinite(z) and z > 0) else 1.0 for z in zooms)
    in_plane_mm = 0.5 * (sx + sy)
    print(f"  voxel spacing: {sx:.3f} × {sy:.3f} × {sz:.3f} mm", file=sys.stderr)

    facts = load_case_facts(facts_path)

    # ── Auto-detect dental Z range from TEETH ONLY ───────────────────────────
    # Using the full jaw mask (including mandible ramus) gives a Z range that
    # is too wide inferiorly, showing neck/vertebral anatomy.
    # Tooth labels give a tighter range centred on the crowns and root apices.
    _TOOTH_IDS  = (list(range(11, 19)) + list(range(21, 29)) +
                   list(range(31, 39)) + list(range(41, 49)))
    tooth_z_any = np.isin(mask, _TOOTH_IDS).any(axis=(0, 1))
    tz_where    = np.where(tooth_z_any)[0]

    if len(tz_where) > 0:
        # Generous margin (15%) so root apices and alveolar bone are included
        z_span   = tz_where[-1] - tz_where[0]
        z_margin = max(12, int(z_span * 0.15))
        z_lo     = max(0,               tz_where[0]  - z_margin)
        z_hi     = min(volume.shape[2], tz_where[-1] + z_margin)
    else:
        # Fallback: jaw mask Z range
        jaw_z_any = np.isin(mask, _ALL_JAW_IDS).any(axis=(0, 1))
        z_where   = np.where(jaw_z_any)[0]
        if len(z_where) > 0:
            z_margin = max(8, int((z_where[-1] - z_where[0]) * 0.08))
            z_lo     = max(0,               z_where[0]  - z_margin)
            z_hi     = min(volume.shape[2], z_where[-1] + z_margin)
        else:
            z_lo, z_hi = 0, volume.shape[2]

    # ── Output pixel size: ONE physical scale for both axes ──────────────────
    # px_mm is chosen so the panoramic body comes out `out_height` rows tall,
    # then used for the arch column spacing too -- that square physical pixel
    # is what makes the aspect ratio correct (see resample_arch_arclength).
    #
    # Clamped so we neither pretend to resolve finer than a quarter of a voxel
    # (pure interpolation, no new information, just cost) nor sample coarser
    # than the voxel grid (which would throw detail away).
    z_span_mm = max(1e-3, (z_hi - z_lo) * sz)
    finest_mm = min(sx, sy, sz)
    px_mm     = float(np.clip(z_span_mm / max(64, int(out_height)),
                              finest_mm / 4.0, finest_mm))
    z_step    = px_mm / sz          # row spacing, in (fractional) Z voxels
    print(f"  render scale: {px_mm:.3f} mm/px "
          f"(z span {z_span_mm:.1f} mm → {int(z_span_mm / px_mm)} rows)",
          file=sys.stderr)

    # Slab half-thickness in mm → voxels. `slab_depth` overrides if given.
    if slab_depth is None:
        slab_depth = max(4, int(round(float(slab_mm) / max(in_plane_mm, 1e-6))))
    print(f"  slab: ±{slab_depth} vox (±{slab_depth * in_plane_mm:.1f} mm)",
          file=sys.stderr)

    # Used later only for the L-marker fallback decision (see below), not by
    # detect_arch_curve itself -- that has its own independent right→left
    # traversal via descending-X sort plus jaw-centerline fallback.
    anchors = arch_side_anchors(mask)

    print(f"[{case_id}] detecting arch curve ...", file=sys.stderr)
    arch_dense = detect_arch_curve(mask, n_points=max(600, int(n_arch_points)))
    if arch_dense is None:
        print(f"  [warn] insufficient tooth/jaw geometry detected, skipping",
              file=sys.stderr)
        return []

    # Uniform ARC LENGTH, not uniform spline parameter, and at the same px_mm
    # the rows use. This is the aspect fix.
    arch_curve, arch_len_mm = resample_arch_arclength(
        arch_dense, (sx, sy), px_mm)
    n_arch_pts = len(arch_curve)
    print(f"  arch length {arch_len_mm:.1f} mm → {n_arch_pts} columns "
          f"at {px_mm:.3f} mm/px", file=sys.stderr)

    # Smoothing windows are POINT COUNTS but must span a FIXED physical
    # distance, so they're derived from px_mm rather than left at the old
    # constants -- those were tuned for ~0.32 mm between arch points and
    # would now cover a few tenths of a millimetre, making the normals noisy
    # and the curvature radius spuriously tiny.
    norm_win = max(3, int(round(1.6 / px_mm)))   # ~1.6 mm of arc
    curv_win = max(5, int(round(2.9 / px_mm)))   # ~2.9 mm of arc

    # Computed once here and passed into BOTH curved_mip and build_overlay,
    # so the grayscale image and the label mask are always sampled from
    # identical geometry -- see `_slab_depth_scale`'s docstring for why this
    # locally shrinks the slab wherever the arch curve bends more tightly
    # than slab_depth allows (belt-and-suspenders on top of the fold-proofed
    # curve itself).
    normals     = _smooth_normals(arch_curve, smooth_pts=norm_win)
    depth_scale = _slab_depth_scale(arch_curve, slab_depth, smooth_pts=curv_win)
    if depth_scale.min() < 1.0:
        radii = _local_curvature_radius(arch_curve, smooth_pts=curv_win)
        min_radius = float(radii.min())
        print(f"  [info] tight arch curvature detected (min radius "
              f"≈{min_radius:.1f}vox vs slab_depth={slab_depth}) -- slab "
              f"depth locally scaled down to {depth_scale.min():.2f}x to "
              f"avoid offset self-intersection", file=sys.stderr)

    # ── Clip metal before projecting ─────────────────────────────────────────
    # See enamel_ceiling: without this the MIP hands a few restorations enough
    # of the image for them to own the window's top end, and everything real
    # gets squeezed into the dark end. Metal still renders -- saturated white,
    # as it does on a real panoramic -- it just stops setting the scale.
    ceiling = enamel_ceiling(volume, mask)
    if ceiling is not None and np.isfinite(ceiling):
        n_clipped = int((volume > ceiling).sum())
        if n_clipped:
            print(f"  [info] clipping {n_clipped} voxels "
                  f"({100.0 * n_clipped / volume.size:.3f}%) above the enamel "
                  f"ceiling ≈{ceiling:.0f} -- metal/streak, kept out of the "
                  f"window ceiling", file=sys.stderr)
        volume = np.minimum(volume, np.float32(ceiling))

    print(f"[{case_id}] curved MIP "
          f"({n_arch_pts} pts × {2*slab_depth+1} depth × "
          f"{int(z_span_mm / px_mm)} Z) ...", file=sys.stderr)
    panoramic = curved_mip(volume, arch_curve, slab_depth=slab_depth,
                           z_lo=z_lo, z_hi=z_hi, depth_scale=depth_scale,
                           normals=normals, z_step=z_step)

    pano_u8  = window_to_uint8(panoramic)
    if clahe_blend > 0:
        pano_u8 = apply_clahe(pano_u8, blend=clahe_blend)
    pano_rgb = np.stack([pano_u8] * 3, axis=-1)
    h, w     = pano_rgb.shape[:2]
    print(f"  size: {w}×{h}  ({w/h:.2f}:1, anatomical)", file=sys.stderr)

    print(f"[{case_id}] locating teeth for FDI tags ...", file=sys.stderr)
    label_pano = build_overlay(mask, arch_curve, slab_depth,
                               volume.shape, z_lo=z_lo, z_hi=z_hi,
                               depth_scale=depth_scale, normals=normals,
                               z_step=z_step)

    # ── Left-right orientation check (data-driven, not a fixed assumption) ───
    # Standard dental panoramic convention shows the patient's RIGHT side on
    # the LEFT of the image (as if facing the patient). Different source
    # datasets can use different raw volume orientations, so rather than
    # always applying (or never applying) a fixed flip, we read the actual
    # quadrant positions off the segmentation itself and only flip when the
    # data says it's actually needed for THIS case.
    print(f"[{case_id}] checking left-right orientation from segmentation ...",
          file=sys.stderr)
    flip = _needs_lr_flip(label_pano)

    if flip is None:
        # No quadrant pair to vote with (edentulous / single-quadrant cases).
        # Fall back to the arch anchors computed above.
        oriented = all(a is not None for a in anchors)
        print("  no quadrant coverage to vote with -- relying on the arch "
              f"side anchors ({'both found' if oriented else 'INCOMPLETE'})",
              file=sys.stderr)
    else:
        oriented = True
        if flip:
            print("  orientation is reversed relative to convention -- flipping",
                  file=sys.stderr)
            pano_rgb   = pano_rgb[:, ::-1, :]
            label_pano = label_pano[:, ::-1]
        else:
            print("  orientation already matches convention -- no flip needed",
                  file=sys.stderr)

    # After the step above the image is in standard convention (patient's
    # right on the image's left), so the patient's LEFT is the right-hand
    # edge. If neither the quadrant vote nor the anchors could establish the
    # side, the marker is omitted rather than guessed.
    left_marker_edge = "right" if oriented else None
    if left_marker_edge is None:
        print("  [warn] laterality undetermined -- omitting the 'L' side marker",
              file=sys.stderr)

    # ── Restrict outlines/tags to facts.json's declared present teeth ────────
    # The segmentation mask can false-positive (or, less often, miss a tooth
    # facts.json says is there); facts.json is treated as the source of truth
    # for which teeth actually exist, so anything the mask found that ISN'T
    # in facts.structured.teeth_present is suppressed here before drawing.
    # Done on the (already orientation-corrected) label_pano, AFTER the L-R
    # flip check above -- that check benefits from every quadrant-labeled
    # pixel the segmentation found, so it deliberately runs on the
    # unfiltered label_pano rather than this filtered one.
    present_teeth = _extract_present_teeth(facts)
    if present_teeth is None:
        print("  [warn] facts.json has no teeth_present field -- outlining/"
              "tagging every tooth the segmentation found, unfiltered",
              file=sys.stderr)
    else:
        mask_found = {int(l) for l in np.unique(label_pano) if int(l) in TOOTH_COLORS}
        extra   = sorted(mask_found - present_teeth)
        missing = sorted(present_teeth - mask_found)
        if extra:
            print(f"  [info] suppressing teeth {extra} -- segmentation found "
                  "them but facts.teeth_present does not list them",
                  file=sys.stderr)
        if missing:
            print(f"  [warn] facts.teeth_present lists {missing} but no "
                  "matching segmentation was found for them", file=sys.stderr)
        keep = np.isin(label_pano, sorted(present_teeth))
        label_pano = np.where(keep, label_pano, 0)

    # Annotation weights are in pixels, so they get scaled with px_mm rather
    # than left at the constants that suited a ~0.3 mm/px raster: a fixed 2 px
    # outline would be a hairline here, and a fixed 1-iteration opening would
    # barely touch the segmentation's jagged edges.
    combined = apply_overlay(
        pano_rgb, label_pano, draw_outline=True,
        left_marker_edge=left_marker_edge,
        outline_width=max(2, int(round(0.30 / px_mm))),           # ~0.30 mm
        mask_smooth_iterations=int(np.clip(round(0.30 / px_mm), 1, 4)),
    )

    # ── Output ───────────────────────────────────────────────────────────────
    # By default the image is written exactly as rendered. It was already
    # produced at its final pixel size by the px_mm logic above, so there is
    # nothing to gain from a resample here and real detail to lose -- LANCZOS
    # over the thin coloured outlines blurs them into the grayscale, and the
    # old fixed 1536×512 canvas was small enough that most cases got
    # DOWNscaled into it. Pass --canvas W H to opt back into a constant size.
    if canvas is not None:
        cw_, ch_ = int(canvas[0]), int(canvas[1])
        print(f"[{case_id}] letterboxing to {cw_}×{ch_} ...", file=sys.stderr)
        combined = letterbox_to_size(combined, cw_, ch_)
    else:
        print(f"[{case_id}] writing at native render size "
              f"{combined.size[0]}×{combined.size[1]} (no resample)",
              file=sys.stderr)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_id}_panoramic.png")
    combined.save(out_path)
    print(f"  → {out_path}", file=sys.stderr)

    caption = build_panoramic_caption(facts)
    caption_path = os.path.join(out_dir, f"{case_id}_panoramic_caption.json")
    import json
    with open(caption_path, "w") as f:
        json.dump({"panoramic": caption}, f, indent=2)
    print(f"  captions → {caption_path}", file=sys.stderr)

    return [out_path, caption_path]


# ── Helpers ───────────────────────────────────────────────────────────────────

def case_id_from_path(path):
    stem = Path(path).name
    for s in (".nii.gz", ".nii"):
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem.replace("ToothFairy", "").replace("_0000", "")


def find_volume_for_mask(mask_path, volumes_dir):
    import glob
    stem     = case_id_from_path(mask_path)
    patterns = [os.path.join(volumes_dir, f"*{stem}*.nii.gz"),
                os.path.join(volumes_dir, f"*{stem}*_0000.nii.gz")]
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sg = ap.add_argument_group("single case")
    sg.add_argument("--volume",  help="CBCT volume .nii.gz")
    sg.add_argument("--mask",    help="Segmentation mask .nii.gz")
    sg.add_argument("--case-id", default=None)

    bg = ap.add_argument_group("batch")
    bg.add_argument("--volumes-dir")
    bg.add_argument("--masks-dir")

    ap.add_argument("--out-dir",       required=True)
    ap.add_argument("--n-arch-points", type=int,   default=1200,
                    help="Points used to sample the fitted spline BEFORE "
                         "arc-length resampling (default 1200). This is a "
                         "working resolution only -- the number of output "
                         "columns comes from the arch's length in mm divided "
                         "by the render pixel size, not from this.")
    ap.add_argument("--out-height",    type=int,   default=900,
                    help="Panoramic body height in pixels (default 900). Sets "
                         "the physical pixel size, which is then used for the "
                         "arch columns too, so the pixel is square in mm and "
                         "the output's aspect ratio is the anatomy's own. "
                         "Width follows from the arch length. Clamped so the "
                         "sampling stays between 1x and 4x the voxel grid.")
    ap.add_argument("--slab-mm",       type=float, default=6.0,
                    help="Slab half-thickness in MILLIMETRES (default 6.0). In "
                         "mm rather than voxels so it means the same thing "
                         "across scans acquired at different resolutions -- the "
                         "old fixed 28 VOXELS silently meant ±4.2 mm on the "
                         "0.15 mm scans and ±8.4 mm on the 0.3 mm ones. "
                         "Compared on A019: ±4.2 mm is crispest through the "
                         "roots, ±8 mm is brighter but visibly superimposed; "
                         "±6 mm keeps nearly all of the former's tooth detail "
                         "with more jaw context. Lower it for tooth clarity, "
                         "raise it to capture more buccally/lingually "
                         "displaced roots.")
    ap.add_argument("--slab-depth",    type=int,   default=None,
                    help="Override --slab-mm with an explicit half-thickness "
                         "in voxels.")
    ap.add_argument("--clahe-blend",   type=float, default=0.6,
                    help="How much mild CLAHE to blend over the linear window, "
                         "0-1 (default 0.6). 0 disables it, leaving a purely "
                         "linear windowing. Near-black air is excluded from "
                         "the blend regardless.")
    ap.add_argument("--canvas",        type=int, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="Letterbox every output into a fixed W×H canvas. "
                         "Off by default: the image is already rendered at its "
                         "final size, so resampling it only costs detail.")
    ap.add_argument("--facts-file",    default=None,
                    help="Path to this case's facts JSON (single-case mode). "
                         "REQUIRED in single-case mode -- facts.json is the "
                         "source of truth for which teeth get outlined/tagged.")
    ap.add_argument("--facts-dir",     default=None,
                    help="Directory of {case_id}.json facts files (batch mode). "
                         "REQUIRED in batch mode, matched by case_id -- a case "
                         "with no matching facts file is skipped with a warning "
                         "rather than run unfiltered.")
    ap.add_argument("--limit",         type=int,   default=None)
    ap.add_argument("--no-facts", action="store_true",
                    help="Render facts-free: read no facts.json at all. Two "
                         "consequences, both deliberate. (1) EVERY tooth the "
                         "segmentation found is outlined and tagged, because "
                         "_extract_present_teeth({}) returns None and the "
                         "teeth_present filter below is skipped. (2) The caption "
                         "is PANORAMIC_STATIC_CAPTION alone, with no findings "
                         "fragments. Use this for arms that must not be handed "
                         "ground truth through their inputs -- facts.json is "
                         "derived from the reference reports, so the tags and "
                         "the caption otherwise encode the answer. Do NOT point "
                         "this at an existing aksssr images dir: it overwrites "
                         "that run's panoramics with a different image.")
    args = ap.parse_args()

    import glob as _glob
    if args.volume and args.mask:
        if not args.facts_file and not args.no_facts:
            ap.error("--facts-file is required in single-case mode -- "
                     "facts.json is the source of truth for which teeth "
                     "get outlined/tagged (or pass --no-facts to render "
                     "facts-free, unfiltered)")
        cid = args.case_id or case_id_from_path(args.mask)
        process_case(args.volume, args.mask, args.out_dir, cid,
                     n_arch_points=args.n_arch_points,
                     slab_mm=args.slab_mm, slab_depth=args.slab_depth,
                     facts_path=None if args.no_facts else args.facts_file,
                     out_height=args.out_height, canvas=args.canvas,
                     clahe_blend=args.clahe_blend)
    elif args.volumes_dir and args.masks_dir:
        if not args.facts_dir and not args.no_facts:
            ap.error("--facts-dir is required in batch mode -- facts.json "
                     "is the source of truth for which teeth get "
                     "outlined/tagged (or pass --no-facts to render "
                     "facts-free, unfiltered)")
        masks = sorted(_glob.glob(os.path.join(args.masks_dir, "*.nii.gz")))
        if args.limit:
            masks = masks[:args.limit]
        for mp in masks:
            cid = case_id_from_path(mp)
            vp  = find_volume_for_mask(mp, args.volumes_dir)
            if vp is None:
                print(f"  [skip] {cid}: no matching volume", file=sys.stderr)
                continue
            if args.no_facts:
                fp = None
            else:
                fp = os.path.join(args.facts_dir, f"{cid}.json")
                if not os.path.exists(fp):
                    print(f"  [skip] {cid}: no matching facts file at {fp} "
                          "(facts.json is required, not optional)", file=sys.stderr)
                    continue
            process_case(vp, mp, args.out_dir, cid,
                         n_arch_points=args.n_arch_points,
                         slab_mm=args.slab_mm, slab_depth=args.slab_depth,
                         facts_path=fp,
                         out_height=args.out_height, canvas=args.canvas,
                         clahe_blend=args.clahe_blend)
    else:
        ap.error("Provide (--volume + --mask) OR (--volumes-dir + --masks-dir)")
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()