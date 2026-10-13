"""Orthographic shaded renders + RGB occlusogram from a bite-registered IOS pair.

All views share the case's anatomical frame (geometry.case_frame) and fixed
physical scales — never per-case intensity or size normalization.

View set (mirrors the photo protocol, plus what photos cannot show):
  frontal        both jaws occluded, seen from anterior; image-left = patient-RIGHT
  buccal_right   seen from the patient's right; anterior at image-RIGHT
  buccal_left    seen from the patient's left;  anterior at image-LEFT
  occlusal_upper upper jaw alone, biting surface (view from below, un-mirrored grid)
  occlusal_lower lower jaw alone, biting surface (view from above)
  cutaway        mid-sagittal +-2 mm slab, seen from patient's right: overjet/overbite
  occlusogram    R=upper heightfield, G=lower, B=|gap| proximity, edge engraving

Shaded views use vertex splatting with a z-buffer, flat normals, headlight +
top-light Lambertian, subtle jaw tint (upper warm / lower cool, consistent with
the occlusogram's R/G identity), and 3x3 hole closing.
"""
import numpy as np

from .geometry import CaseFrame, case_frame

RES = 0.1            # mm / px, shaded views (28-px ViT patch = 2.8 mm; class
                     # shifts are 2-4 mm — the TF4 "patching is the limit" lesson)
OCCLU_RES = 0.15     # mm / px, occlusogram (relational shapes are larger-scale)
OCCLU_SIZE = 640     # px -> 96 mm field
MARGIN_MM = 4.0
H_RANGE = 15.0       # occlusogram height range (mm)
GAP_RANGE = 3.0      # occlusogram |gap| range (mm)
EDGE_TAU = 0.8       # mm/px full-edge depth step
EDGE_STRENGTH = 0.55
CUTAWAY_HALF_MM = 3.0
TINT_UPPER = np.array([1.00, 0.88, 0.84])   # warm
TINT_LOWER = np.array([0.86, 1.00, 0.88])   # cool
AMBIENT = 0.30


def _close(buf, fill_from, invalid, it=2):
    """Fill invalid pixels from 3x3 neighborhood aggregate of fill_from."""
    H, W = buf.shape[:2]
    out = buf.copy()
    for _ in range(it):
        bad = invalid(out)
        if not bad.any():
            break
        pad = np.pad(out, [(1, 1), (1, 1)] + [(0, 0)] * (out.ndim - 2), mode="edge")
        stack = np.stack([pad[r:r + H, c:c + W] for r in range(3) for c in range(3)])
        out = np.where(bad[..., None] if out.ndim == 3 else bad, fill_from(stack), out)
    return out


def _splat(px, py, depth, values, size_wh, footprint=1):
    """Nearest-depth vertex splat with an odd NxN pixel sprite footprint.
    All sprite copies enter ONE global depth sort, so nearer surfaces win
    everywhere (no far-side shine-through at sub-vertex pixel density)."""
    W, H = size_wh
    cols = (px / RES).astype(np.int64)
    rows = (py / RES).astype(np.int64)
    if footprint > 1:
        r = footprint // 2
        offs = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)]
        rows = np.concatenate([rows + dr for dr, _ in offs])
        cols = np.concatenate([cols + dc for _, dc in offs])
        depth = np.tile(depth, len(offs))
        values = np.tile(values, (len(offs),) + (1,) * (values.ndim - 1))
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols, depth, values = rows[ok], cols[ok], depth[ok], values[ok]
    flat = rows * W + cols
    order = np.argsort(depth)            # nearest last -> wins the scatter write
    flat, depth, values = flat[order[::-1]], depth[order[::-1]], values[order[::-1]]
    dbuf = np.full(H * W, np.inf)
    vbuf = np.zeros((H * W,) + values.shape[1:])
    dbuf[flat] = depth
    vbuf[flat] = values
    return vbuf.reshape((H, W) + values.shape[1:]), dbuf.reshape(H, W)


def _shade(normals, view_dir, up):
    headlight = np.clip(normals @ (-view_dir), 0, 1)
    toplight = np.clip(normals @ up, 0, 1)
    return AMBIENT + 0.55 * headlight + 0.15 * toplight


def _render_shaded(jaws, view_dir, screen_x, screen_y, frame, z_window=None,
                   footprint=3, blend=None):
    """jaws: list of (verts, normals, tint). Orthographic along view_dir.

    blend=None: hard z-test across jaws (opaque). blend=alpha in (0,1): each jaw
    is rendered opaquely against itself, then the jaws are alpha-composited per
    pixel — nearer jaw at `alpha`, farther showing through at 1-alpha. Lets the
    occluded jaw (e.g. lower incisors behind a deep bite) stay visible while
    within-jaw surfaces remain crisp."""
    allpx, allpy = [], []
    parts = []
    for verts, normals, tint in jaws:
        a, l, z = frame.coords(verts)
        if z_window is not None:
            keep = (z - frame.z0 > z_window[0]) & (z - frame.z0 < z_window[1])
            verts, normals = verts[keep], normals[keep]
        px = verts @ screen_x
        py = verts @ screen_y
        parts.append((px, py, verts, normals, tint))
        allpx.append(px)
        allpy.append(py)
    allpx, allpy = np.concatenate(allpx), np.concatenate(allpy)
    x0, y0 = allpx.min() - MARGIN_MM, allpy.min() - MARGIN_MM
    W = int((allpx.max() - x0 + MARGIN_MM) / RES) + 1
    H = int((allpy.max() - y0 + MARGIN_MM) / RES) + 1

    buffers = []
    for px, py, verts, normals, tint in parts:
        depth = verts @ view_dir
        shade = _shade(normals, view_dir, frame.occ)
        rgb = shade[:, None] * tint[None, :]
        buffers.append(_splat(px - x0, py - y0, depth, rgb, (W, H),
                              footprint=footprint))

    if blend is not None and len(buffers) == 2:
        (v1, d1), (v2, d2) = buffers
        f1, f2 = np.isfinite(d1), np.isfinite(d2)
        img = np.zeros((H, W, 3))
        img[f1 & ~f2] = v1[f1 & ~f2]
        img[f2 & ~f1] = v2[f2 & ~f1]
        both = f1 & f2
        near1 = both & (d1 <= d2)
        near2 = both & ~(d1 <= d2)
        img[near1] = blend * v1[near1] + (1 - blend) * v2[near1]
        img[near2] = blend * v2[near2] + (1 - blend) * v1[near2]
        valid = f1 | f2
    else:
        img = np.zeros((H, W, 3))
        depth_all = np.full((H, W), np.inf)
        for vbuf, dbuf in buffers:
            closer = dbuf < depth_all
            img = np.where(closer[..., None], vbuf, img)
            depth_all = np.minimum(depth_all, dbuf)
        valid = np.isfinite(depth_all)
    img = _close(img, lambda s: s.mean(0), lambda o: (~valid) & (
        _grow(valid)), it=2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _grow(mask, it=2):
    m = mask.copy()
    H, W = m.shape
    for _ in range(it):
        pad = np.pad(m, 1)
        m = np.stack([pad[r:r + H, c:c + W] for r in range(3) for c in range(3)]).any(0)
    return m


def _close_zbuf(buf, mode, it=2):
    H, W = buf.shape
    for _ in range(it):
        pad = np.pad(buf, 1, constant_values=np.inf if mode == "min" else -np.inf)
        stack = np.stack([pad[r:r + H, c:c + W] for r in range(3) for c in range(3)])
        agg = stack.min(0) if mode == "min" else stack.max(0)
        buf = np.where(~np.isfinite(buf), agg, buf)
    return buf


def occlusogram(vu, vl, frame: CaseFrame, size=OCCLU_SIZE, res=OCCLU_RES):
    au, lu_, zu = frame.coords(vu)
    al, ll_, zl = frame.coords(vl)

    def ras(a, l, z, mode):
        rows = ((frame.a_c - a) / res + size / 2).astype(np.int64)
        cols = ((l - frame.l_c) / res + size / 2).astype(np.int64)
        ok = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
        buf = np.full((size, size), np.inf if mode == "min" else -np.inf)
        (np.minimum if mode == "min" else np.maximum).at(
            buf, (rows[ok], cols[ok]), z[ok])
        return _close_zbuf(buf, mode)

    zu_b, zl_b = ras(au, lu_, zu, "min"), ras(al, ll_, zl, "max")
    fu, fl = np.isfinite(zu_b), np.isfinite(zl_b)
    R = np.where(fu, np.clip(1 - (zu_b - frame.z0) / H_RANGE, 0, 1) ** 0.8, 0)
    G = np.where(fl, np.clip(1 - (frame.z0 - zl_b) / H_RANGE, 0, 1) ** 0.8, 0)
    B = np.zeros_like(R)
    both = fu & fl
    B[both] = np.clip(1 - np.abs((zu_b - zl_b)[both]) / GAP_RANGE, 0, 1)

    def edge(buf, fin):
        z = np.where(fin, buf, np.nan)
        g = np.hypot(*np.gradient(z))
        g = np.where(np.isfinite(g), g, EDGE_TAU)
        g[~fin] = 0.0
        return np.clip(g / EDGE_TAU, 0, 1)

    shade = 1.0 - EDGE_STRENGTH * np.maximum(edge(zu_b, fu), edge(zl_b, fl))
    rgb = np.stack([R, G, B], -1) * shade[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render_views(vu, nu, vl, nl, frame: CaseFrame | None = None):
    """All mesh views for one case -> dict name -> uint8 HxWx3.
    Never raises on geometry degeneracy at view level; callers handle empties."""
    if frame is None:
        frame = case_frame(vu, vl)
    up, ant, left = frame.occ, frame.a_dir, frame.l_left
    U = (vu, nu, TINT_UPPER)
    L = (vl, nl, TINT_LOWER)
    views = {}
    # crown-band z-window: cuts the vestibular gingival shells that otherwise
    # bury the crowns in lateral orthographic views
    ZW = (-16.0, 16.0)
    BLEND = 0.65   # two-jaw views: nearer jaw 65%, farther ghosts through
    # frontal: viewer anterior, looking posterior; image-left = patient-RIGHT
    views["frontal"] = _render_shaded([U, L], view_dir=-ant, screen_x=left,
                                      screen_y=-up, frame=frame, z_window=ZW,
                                      blend=BLEND)
    # right buccal: viewer at patient's right; anterior at image-RIGHT
    views["buccal_right"] = _render_shaded([U, L], view_dir=left, screen_x=ant,
                                           screen_y=-up, frame=frame, z_window=ZW,
                                           blend=BLEND)
    # left buccal: viewer at patient's left; anterior at image-LEFT
    views["buccal_left"] = _render_shaded([U, L], view_dir=-left, screen_x=-ant,
                                          screen_y=-up, frame=frame, z_window=ZW,
                                          blend=BLEND)
    # occlusal per jaw: biting surface, un-mirrored shared grid
    # (top = anterior, image-left = patient-RIGHT, same convention as occlusogram)
    views["occlusal_upper"] = _render_shaded([U], view_dir=up, screen_x=left,
                                             screen_y=-ant, frame=frame)
    views["occlusal_lower"] = _render_shaded([L], view_dir=-up, screen_x=left,
                                             screen_y=-ant, frame=frame)
    # mid-sagittal cutaway: +-3 mm slab zoomed to the incisal region, viewed from
    # patient's right — the direct overjet/overbite panel (thin shells edge-on
    # need the fat sprite to read as a solid profile)
    a_u, a_l = vu @ ant, vl @ ant
    a_front = max(a_u.max(), a_l.max())
    keep_u = ((np.abs(vu @ left - frame.l_c) < CUTAWAY_HALF_MM)
              & (a_u > a_front - 35.0))
    keep_l = ((np.abs(vl @ left - frame.l_c) < CUTAWAY_HALF_MM)
              & (a_l > a_front - 35.0))
    if keep_u.sum() > 200 and keep_l.sum() > 200:
        views["cutaway"] = _render_shaded(
            [(vu[keep_u], nu[keep_u], np.array([1.0, 0.55, 0.5])),
             (vl[keep_l], nl[keep_l], np.array([0.5, 1.0, 0.55]))],
            view_dir=left, screen_x=ant, screen_y=-up, frame=frame,
            z_window=(-18, 18), footprint=5, blend=0.7)
    views["occlusogram"] = occlusogram(vu, vl, frame)
    return views
