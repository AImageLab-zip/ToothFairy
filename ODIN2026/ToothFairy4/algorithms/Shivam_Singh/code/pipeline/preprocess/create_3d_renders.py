#!/usr/bin/env python3
"""
create_3d_renders.py

Generate high-quality 3D surface renders from a CBCT segmentation mask.
Shows mandible + maxilla + all 32 teeth + IAC nerves + maxillary sinuses
from left / frontal / right viewpoints.

Quality pipeline per structure:
  1. Gaussian pre-smoothing (scipy) on the binary mask  → smooth isosurface
  2. Marching cubes                                      → triangle mesh
  3. Laplacian post-smoothing (numpy)                    → removes voxel staircase
  4. PIL painter's algorithm render                      → correct depth ordering
  5. Two-pass alpha composite for bone transparency      → teeth visible inside bone
  6. LANCZOS downsample from 2× supersampled image      → anti-aliased edges

Output (MedThinkVQA-style separate images + captions, NOT combined):
  {case_id}_3d_left.png
  {case_id}_3d_frontal.png
  {case_id}_3d_right.png
  {case_id}_3d_captions.json   -- {"3d_left": "<caption>", "3d_frontal": "...", "3d_right": "..."}

`posterior` is no longer rendered -- only left / frontal / right are produced,
each kept as its own file with its own caption, instead of being merged into
a single 2×2 combined grid (the old `3d_combined.png` + combine() step has
been removed).

Each saved PNG is stamped with radiology-style R/L corner markers indicating
which anatomical side is visible: "3d_right" gets "R" (top-left; the camera
sits at the patient's right and sees the patient's RIGHT side), "3d_left" gets
"L" (top-right), and "3d_frontal" gets both "R" (top-left) and "L" (top-right).

Usage:
    python code/pipeline/preprocess/create_3d_renders.py --mask dataset/predictions/A004.nii.gz --out-dir outputs/3d_renders --rot-deg 60

    python code/pipeline/preprocess/create_3d_renders.py \
        --mask    dataset/training/masks/A003.nii.gz \
        --facts-file dataset/training/facts/A003.json \
        --out-dir outputs/3d_renders/ \
        --step-size 3
"""

import argparse, json, os, sys
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    sys.exit("pip install nibabel")
try:
    from skimage.measure import marching_cubes
except ImportError:
    sys.exit("pip install scikit-image")
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not found — surfaces will be blocky. pip install scipy",
          file=sys.stderr)

from PIL import Image, ImageDraw

# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

# ── Label map ─────────────────────────────────────────────────────────────────
MANDIBLE_ID = 1
MAXILLA_ID  = 2
BONE_IDS    = {MANDIBLE_ID, MAXILLA_ID}
SINUS_IDS   = {5, 6}    # Left and Right Maxillary Sinuses
CANAL_IDS   = {3, 4}    # Left and Right mandibular canals (IAC)

# The canal's colour, named once. create_tooth_detail.py paints the canal on
# its 2D composites with this same RGB, so a reader who learns "yellow = the
# mandibular canal" from a 3D render reads the tooth composites the same way,
# and CANAL_COLOR_NAME below stays the true word for both.
CANAL_COLOR = (240, 200, 60)

TOOTH_IDS = (list(range(11, 19)) + list(range(21, 29)) +
             list(range(31, 39)) + list(range(41, 49)))

_TOOTH_PALETTE = [
    (220, 215, 175), ( 90, 170, 220), (235, 130,  55), (175,  95, 185),
    ( 65, 185, 155), (230,  75,  75), (240, 215,  60), (105, 110, 215),
    (100, 215, 165), (215,  95, 165), (165, 215,  90), ( 95, 165, 215),
    (215, 165,  90), (100, 215, 215), (165,  95, 215), (195, 195,  65),
    (215, 190, 185), ( 70, 145, 195), (195, 120,  65), (145,  75, 155),
    ( 55, 155, 135), (195,  60,  60), (200, 185,  55), ( 80,  90, 185),
    ( 80, 185, 135), (185,  75, 140), (140, 185,  75), ( 75, 140, 185),
    (185, 140,  75), ( 85, 185, 185), (140,  75, 185), (185,  75,  75),
]

# Rendering order matters for two-pass alpha:
# teeth and nerves (opaque) first so bone transparency reveals them.
STRUCTURES = (
    [(tid, _TOOTH_PALETTE[i % len(_TOOTH_PALETTE)])
     for i, tid in enumerate(TOOTH_IDS)]
    + [
        (3, CANAL_COLOR),       # Left  mandibular canal (IAC) — yellow
        (4, CANAL_COLOR),       # Right mandibular canal (IAC)
        (5, ( 80, 160, 220)),   # Left  Maxillary Sinus — blue
        (6, ( 80, 160, 220)),   # Right Maxillary Sinus
        (MANDIBLE_ID, (100, 195, 110)),  # bone last → transparent overlay
        (MAXILLA_ID,  (220, 195, 130)),
    ]
)

# Human-readable color names for the same RGB tuples used above, kept next to
# STRUCTURES so caption text and render colors can't silently drift apart.
# Caption terminology uses "mandibular canal" (schema.json's term for facts
# mandible_canal_right/left) rather than "IAC nerve", to avoid mismatched
# vocabulary between the image caption and the schema's question text.
MANDIBLE_COLOR_NAME = "green"
MAXILLA_COLOR_NAME  = "tan"
CANAL_COLOR_NAME    = "yellow"
SINUS_COLOR_NAME    = "blue"

BG_COLOR = (200, 205, 218)

# ── What the mask actually contains ───────────────────────────────────────────
#
# A label being NON-EMPTY is not the same as that structure being in the
# acquisition. nnU-Net leaves slivers: S0002's mask carries 9,327 maxilla
# voxels = 252 mm^3, 0.5% of its segmentation, and its rendered 3d_frontal
# shows no tan whatsoever -- yet its facts file says "fov": {"maxilla":
# "partial"} and its caption duly announced "maxilla (tan) ... Acquisition
# FOV: maxilla partially included" over a mandible-only picture, then went on
# to call the maxilla "fully edentulous" because the report listed 11-28 as
# absent. Three false claims about anatomy nobody scanned, all from trusting
# a label id instead of a volume.
#
# Calibrated by rendering across the band and looking at the result:
#
#     S0002    252 mm^3  -- no tan anywhere in the frontal render
#     P437     428 mm^3  -- one thin wedge, easily read as noise
#     P025   1,554 mm^3  -- two small alveolar caps, clearly present
#     F056   3,731 mm^3  -- an unmistakable band
#
# 500 mm^3 sits in the gap. It is deliberately a VISIBILITY threshold, not an
# agreement-with-the-report one: the caption's job is to describe the picture
# the VLM is looking at. (For reference, the reports themselves are noisy
# here -- of 622 cases, 37 whose report says "partially included" have a mask
# with zero maxilla voxels, and no threshold in 0-2000 mm^3 agrees with the
# report corpus better than ~81%.)
#
# The full distribution is strongly bimodal, so the exact cut matters little:
# 264 of 622 cases have EXACTLY zero maxilla voxels, another 90 fall under
# 500 mm^3, and 137 are over 10,000.
MIN_ARCH_BONE_MM3 = 500.0


def mask_label_volumes(mask, zooms=None):
    """
    {label_id: volume in mm^3} for every non-zero label in the mask.

    `zooms` is the voxel size; without it the values are voxel COUNTS, which
    is wrong by up to ~50x between a 0.15 mm A-case and a 0.3 mm P-case --
    so callers that have the header should always pass it.
    """
    ids, counts = np.unique(mask, return_counts=True)
    scale = float(np.prod(zooms[:3])) if zooms is not None else 1.0
    return {int(i): float(c) * scale for i, c in zip(ids, counts) if int(i) != 0}


def arch_coverage(volumes, structured=None):
    """
    What the acquisition actually covers, per arch, as
    {"maxilla": {"bone": bool, "teeth": bool}, "mandible": {...}}.

    `volumes` is mask_label_volumes' output and is AUTHORITATIVE -- it is
    measured from the very voxels the renderer turns into meshes. `structured`
    (a facts file's structured block) is only the fallback for callers with no
    mask in hand, and then only through present_label_ids, whose bare label
    inventory cannot express "252 mm^3 of nothing"; that path keeps the old
    presence-means-included behaviour rather than guessing.

    Returns None when neither source says anything, so callers make no
    "not included" claim in either direction.
    """
    if volumes:
        return {arch: {"bone": volumes.get(ARCH_BONE_ID[arch], 0.0) >= MIN_ARCH_BONE_MM3,
                       "teeth": any(f in volumes for f in ARCH_FDI[arch])}
                for arch in ARCH_ORDER}

    raw = (structured or {}).get("present_label_ids")
    if not isinstance(raw, list):
        return None
    ids = {i for i in raw if isinstance(i, int) and not isinstance(i, bool)}
    return {arch: {"bone": ARCH_BONE_ID[arch] in ids,
                   "teeth": bool(ARCH_FDI[arch] & ids)}
            for arch in ARCH_ORDER}


def drawn_structures(volumes):
    """
    Which of the four NAMED structures the color key can honestly promise:
    {"mandible", "maxilla", "canals", "sinuses"}. None when there is no mask
    to go on, which is the signal to name all four as before.

    Bones use the same MIN_ARCH_BONE_MM3 cut as arch_coverage so the key and
    the FOV sentence can never disagree with each other. Canals and sinuses
    are small structures whose whole visible extent is legitimately a few
    hundred mm^3, so for them any voxels at all count.
    """
    if not volumes:
        return None
    drawn = set()
    for arch, name in (("mandible", "mandible"), ("maxilla", "maxilla")):
        if volumes.get(ARCH_BONE_ID[arch], 0.0) >= MIN_ARCH_BONE_MM3:
            drawn.add(name)
    if any(volumes.get(i) for i in CANAL_IDS):
        drawn.add("canals")
    if any(volumes.get(i) for i in SINUS_IDS):
        drawn.add("sinuses")
    return drawn


# ── Per-view captions ─────────────────────────────────────────────────────────
# MedThinkVQA-style: each separately-saved image gets a short caption
# describing what it shows (viewpoint / orientation), not case-specific
# findings -- the VLM still has to derive findings from the image itself.
#
# NOTE on left/right: the camera named "left" sits at the patient's LEFT and
# therefore sees the patient's LEFT side. This read the other way round until
# 2026-08-11 -- the comment claimed camera-left brought the patient's RIGHT
# side into the foreground, and the marker, the colour key and schema.json's
# images_needed were all built on that sentence, so every side-specific 3D
# fact was asked on a picture of the other side. Measured three ways before
# fixing: depth along the eye vector puts the left arch nearer in the "left"
# view for every case tested; the near lower third molar in 3d_left renders
# slate blue (38) rather than brick red (48); and P543's 3d_left shows an
# isolated molar with a gap either side, a pattern only its left arch has.
# Captions name the anatomical side actually visible, which is now also the
# camera-position name.
#
# THE COLOR KEY NAMES ONLY WHAT IS DRAWN. It used to promise all four
# structures in every caption, with "a structure absent from the segmentation
# is simply not drawn" as the escape hatch -- but a caption that says "maxilla
# (tan)" over a mandible-only picture is an expected-but-unseen structure, and
# the VLM describes those anyway. When a mask is available (view_caption's
# `drawn` argument, from drawn_structures) the key lists exactly the
# structures with geometry in it; without one it names all four, as before.
_VIEW_CAMERA = {
    "frontal": ("at the patient's front", "FRONTAL-view", False),
    "left":    ("at the patient's left",  "LEFT-side",    True),
    "right":   ("at the patient's right", "RIGHT-side",   True),
}

# (key in `drawn`, singular noun, plural noun, color)
_KEY_STRUCTURES = (
    ("mandible", "mandible",        "mandible",         MANDIBLE_COLOR_NAME),
    ("maxilla",  "maxilla",         "maxilla",          MAXILLA_COLOR_NAME),
    ("canals",   "mandibular canal", "mandibular canals", CANAL_COLOR_NAME),
    ("sinuses",  "maxillary sinus",  "maxillary sinuses", SINUS_COLOR_NAME),
)


def view_caption(view, drawn=None):
    """
    The static part of one view's caption. `drawn` is drawn_structures()'s set
    -- None names all four structures (no mask to check against).

    A view whose mask contains none of the four still gets a well-formed
    sentence ("...bringing no segmented bone, canal or sinus into view"),
    which cannot happen in practice but beats emitting "bringing the patient's
    FRONTAL-view  into view."
    """
    where, side, side_view = _VIEW_CAMERA[view]
    names = [(sing if side_view else plur, colour)
             for key, sing, plur, colour in _KEY_STRUCTURES
             if drawn is None or key in drawn]

    if names:
        listed = [f"{n} ({c})" for n, c in names]
        if len(listed) == 1:
            subject = listed[0]
        elif len(listed) == 2:
            subject = f"{listed[0]} and {listed[1]}"
        else:
            subject = ", ".join(listed[:-1]) + f", and {listed[-1]}"
        subject = f"the patient's {side} {subject}"
    else:
        subject = "no segmented bone, canal or sinus"

    caption = ("3D reconstruction from CBCT segmentation masks: camera "
               f"positioned {where}, bringing {subject} into view. Color key "
               "only -- a structure absent from the segmentation is simply "
               "not drawn.")
    wisdom = wisdom_color_key(view)
    return f"{caption} {wisdom}" if wisdom else caption

# FDI arch membership, used both for L-R flip auto-detection in render_view()
# and for splitting ian_close_teeth by side when building fact-based captions.
RIGHT_ARCH_FDI = frozenset(range(11, 19)) | frozenset(range(41, 49))
LEFT_ARCH_FDI  = frozenset(range(21, 29)) | frozenset(range(31, 39))

# Upper/lower membership, for the per-arch tooth-presence caption fragment.
ARCH_FDI = {
    "maxilla":  frozenset(range(11, 19)) | frozenset(range(21, 29)),
    "mandible": frozenset(range(31, 39)) | frozenset(range(41, 49)),
}
ARCH_BONE_ID   = {"maxilla": MAXILLA_ID, "mandible": MANDIBLE_ID}
ARCH_ADJECTIVE = {"maxilla": "maxillary", "mandible": "mandibular"}
ARCH_ORDER     = ("maxilla", "mandible")

# The wisdom-tooth facts (upper/lower_left/right_wisdom_tooth) are the only
# per-tooth questions asked on the 3D side views, so those captions name the
# third molars -- and ONLY them. Naming all 32 tooth colors would bury the two
# actually being asked about; the frontal view names no tooth color at all
# (see build_tooth_presence_fragment).
WISDOM_TEETH = {
    "left":  ((28, "upper"), (38, "lower")),   # camera-left  -> patient's LEFT
    "right": ((18, "upper"), (48, "lower")),   # camera-right -> patient's RIGHT
}

# Color names for those four teeth only, each pinned to the palette entry the
# renderer will actually use, so caption wording and render color cannot
# silently drift apart -- the same guarantee MANDIBLE_COLOR_NAME and friends
# give for the bones.
WISDOM_COLOR_NAMES = {
    18: ("indigo",     (105, 110, 215)),
    28: ("olive",      (195, 195,  65)),
    38: ("slate blue", ( 80,  90, 185)),
    48: ("brick red",  (185,  75,  75)),
}
for _fdi, (_name, _rgb) in WISDOM_COLOR_NAMES.items():
    assert _TOOTH_PALETTE[TOOTH_IDS.index(_fdi) % len(_TOOTH_PALETTE)] == _rgb, (
        f"WISDOM_COLOR_NAMES[{_fdi}] = {_name} {_rgb} no longer matches the "
        f"palette color the renderer assigns tooth {_fdi}")

# Which side of the mouth each wisdom tooth is, in words. The schema asks these
# four facts by FDI number alone ("tooth 18"), which is the hardest possible
# handle for a reader that has to FIND the tooth in a 3D render first: 18 vs 28
# is one digit, and the two sit on opposite sides of the head. Naming the tooth
# the way a radiologist would say it out loud, plus the colour it is drawn in,
# gives three independent ways to land on the right one.
WISDOM_TOOTH_WORDS = {
    18: "upper right",
    28: "upper left",
    38: "lower left",
    48: "lower right",
}


def wisdom_color_key(view: str) -> str:
    """
    "The upper right wisdom tooth (third molar, FDI 18) is drawn in indigo; the
    lower right wisdom tooth (FDI 48) in brick red." -- for one side view.

    STATIC, and deliberately so: a colour is a property of the RENDERER, not of
    this case's findings, so unlike build_wisdom_fragment (which reports what
    the facts file says is absent) it carries no ground truth and belongs in
    the caption whether or not a facts file was given. It was previously only
    reachable through that fact fragment, which meant --no-facts silently took
    the colour key away with the answers -- leaving the model asked about
    "tooth 18" in a picture of 30-odd similarly-sized teeth with no way to
    pick it out.

    A tooth that is not segmented is simply not drawn, and the caption's own
    "a structure absent from the segmentation is simply not drawn" already
    covers that -- so this says which colour to LOOK for, never that the tooth
    is there.
    """
    pairs = WISDOM_TEETH.get(view)
    if not pairs:
        return ""
    parts = []
    for fdi, _ in pairs:
        colour = WISDOM_COLOR_NAMES[fdi][0]
        parts.append(f"the {WISDOM_TOOTH_WORDS[fdi]} wisdom tooth (third "
                     f"molar, FDI {fdi}) is drawn in {colour}")
    return ("Wisdom-tooth colour key -- " + "; ".join(parts) +
            ". If you cannot find that colour, that tooth is not segmented.")


# Folded into the caption itself by view_caption, not appended by the caller,
# so every consumer gets it: render_case and rebuild_captions.py both go
# through that one function. The frontal view gets nothing -- no wisdom-tooth
# fact is asked of it, so wisdom_color_key returns "".
#
# Every-structure captions, kept as a module-level dict because that is what
# rebuild_captions.py imports and what a caller with no mask should get.
VIEW_CAPTIONS = {view: view_caption(view) for view in _VIEW_CAMERA}

# Only these views are rendered/saved now (posterior dropped).
ACTIVE_VIEWS = ("left", "frontal", "right")

# ── Shading ───────────────────────────────────────────────────────────────────

def shade_faces(verts, faces):
    """Multi-light two-sided Lambertian shading → (N_faces,) in [0,1]."""
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    ns = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(ns, axis=1, keepdims=True)
    ns /= np.where(norms < 1e-8, 1.0, norms)
    lights = [
        (np.array([ 0.4,  0.3, 1.0], np.float32), 0.55),
        (np.array([-0.6,  0.2, 0.5], np.float32), 0.22),
        (np.array([ 0.3, -0.5, 0.3], np.float32), 0.12),
    ]
    diffuse = np.full(len(faces), 0.28, np.float32)
    for ldir, strength in lights:
        ldir = ldir / np.linalg.norm(ldir)
        diffuse += strength * np.abs(ns @ ldir)
    return np.clip(diffuse, 0.0, 1.0)


# ── Mesh smoothing ────────────────────────────────────────────────────────────

def laplacian_smooth(verts, faces, n_iter=10, lam=0.5):
    """Vectorized Laplacian smoothing (numpy-only)."""
    n = len(verts)
    v = verts.copy()
    fi, fj, fk = faces[:,0], faces[:,1], faces[:,2]
    for _ in range(n_iter):
        adj_sum   = np.zeros_like(v)
        adj_count = np.zeros(n, np.float32)
        for a, b in [(fi,fj),(fi,fk),(fj,fi),(fj,fk),(fk,fi),(fk,fj)]:
            np.add.at(adj_sum,   a, v[b])
            np.add.at(adj_count, a, 1)
        v += lam * (adj_sum / np.maximum(adj_count[:,None], 1) - v)
    return v


# ── Camera ────────────────────────────────────────────────────────────────────

def look_at(eye_dir, world_up=(0., 0., 1.)):
    """3×3 camera matrix."""
    e = np.asarray(eye_dir, np.float64); e /= np.linalg.norm(e)
    u = np.asarray(world_up, np.float64)
    r = np.cross(e, u)
    if np.linalg.norm(r) < 1e-6:
        u = np.array([0., 1., 0.]); r = np.cross(e, u)
    r /= np.linalg.norm(r)
    u2 = np.cross(r, e); u2 /= np.linalg.norm(u2)
    return np.stack([r, u2, e])


_ELEV = 0.18

def make_viewpoints(rot_deg: float = 45.0):
    """
    Build camera viewpoints for a given oblique rotation angle.

    Only ACTIVE_VIEWS (left, frontal, right) are actually rendered/saved by
    render_case() -- "posterior" is kept here (harmless, unused) only so the
    camera geometry stays documented in one place if it's ever needed again.
    """
    rot = np.deg2rad(rot_deg)
    return [
        ("left",      look_at((-np.sin(rot),  np.cos(rot), _ELEV)), ["L"]),
        ("frontal",   look_at(( 0.,            1.,          _ELEV)), ["R", "L"]),
        ("right",     look_at(( np.sin(rot),   np.cos(rot), _ELEV)), ["R"]),
        ("posterior", look_at(( 0.,           -1.,          _ELEV)), ["R", "L"]),
    ]


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_view(all_verts, all_faces, all_colors_u8, face_struct_ids,
                cam_matrix, img_size=512, supersample=2, bone_alpha=0.55,
                tag_ids=None):
    """
    Two-pass painter's algorithm render with bone transparency.

    Pass 1: all faces fully opaque in depth order  → teeth/nerves under bone
    Pass 2: bone faces redrawn as semi-transparent overlay → reveals interior

    `tag_ids` writes each of those FDI numbers into the image beside its own
    tooth, in that tooth's colour. A whole-jaw side view renders a third molar
    about 30 px across at img_size=512, and both judgements these views exist
    for -- how much crown is clear of bone, how far the axis is tipped off the
    neighbour -- start with finding the tooth at all. The panoramic has done
    this from the start ("its FDI tag in the margin band, colored to match
    that tooth's outline"); this is the same device, and it costs one text
    draw rather than a second render.
    """
    sz = img_size * supersample
    margin = 0.05

    cv = (cam_matrix @ all_verts.T).T
    xs, ys, zs = cv[:,0], cv[:,1], cv[:,2]

    # ── Auto-detect L-R flip from segmentation ────────────────────────────
    # Standard radiology: patient-right (teeth 11-18, 41-48) on image-LEFT.
    # Project face centres and compare mean x of right-arch vs left-arch.
    _rm = np.isin(face_struct_ids, sorted(RIGHT_ARCH_FDI))
    _lm = np.isin(face_struct_ids, sorted(LEFT_ARCH_FDI))
    x_flipped = False
    if _rm.any() and _lm.any():
        _rx = xs[all_faces[_rm]].mean()
        _lx = xs[all_faces[_lm]].mean()
        if _rx > _lx:          # right teeth are on image-right → flip
            xs = -xs
            x_flipped = True

    span  = max(xs.max()-xs.min(), ys.max()-ys.min()) + 1e-6
    scale = sz * (1 - 2*margin) / span
    sx = ((xs - (xs.min()+xs.max())/2) * scale + sz/2).astype(np.float32)
    sy = (-(ys - (ys.min()+ys.max())/2) * scale + sz/2).astype(np.float32)

    f_sx = sx[all_faces]; f_sy = sy[all_faces]
    f_sz = zs[all_faces].mean(axis=1)
    order = np.argsort(f_sz)

    px0=f_sx[order,0]; py0=f_sy[order,0]
    px1=f_sx[order,1]; py1=f_sy[order,1]
    px2=f_sx[order,2]; py2=f_sy[order,2]
    cols = all_colors_u8[order]
    sids = face_struct_ids[order]


    # Structures rendered with alpha: bone + sinuses.
    # Teeth and IAC nerves are always fully opaque.
    # Sinuses are transparent so tooth roots inside the sinus are visible.
    _ALPHA_IDS = BONE_IDS | set(SINUS_IDS)

    def _draw_pass(include_alpha):
        im  = Image.new("RGB", (sz, sz), BG_COLOR)
        drw = ImageDraw.Draw(im)
        for i in range(len(order)):
            if (not include_alpha) and sids[i] in _ALPHA_IDS:
                continue
            drw.polygon(
                [(px0[i],py0[i]),(px1[i],py1[i]),(px2[i],py2[i])],
                fill=(int(cols[i,0]), int(cols[i,1]), int(cols[i,2])),
            )
        return np.array(im, dtype=np.float32)

    # Pass 1 (all opaque) establishes correct depth order for teeth vs bone/sinus.
    # Pass 2 blends bone + sinus as transparent over the opaque layer.
    arr_full     = _draw_pass(include_alpha=True)
    arr_no_alpha = _draw_pass(include_alpha=False)
    blended      = (bone_alpha * arr_full + (1.0 - bone_alpha) * arr_no_alpha).astype(np.uint8)
    img          = Image.fromarray(blended)

    # Compute normalised centroid x of right/left teeth for label placement
    _r_sx = float(sx[all_faces[_rm]].mean() / sz) if _rm.any() else None
    _l_sx = float(sx[all_faces[_lm]].mean() / sz) if _lm.any() else None

    if supersample > 1:
        img = img.resize((img_size, img_size), Image.LANCZOS)

    # FDI tags, drawn AFTER the downsample so the glyphs are crisp rather than
    # resampled. Anchor: the tooth's own projected centroid, offset clear of
    # it -- up for a maxillary tooth, down for a mandibular one -- with a
    # leader line back, because a number floating near two teeth names
    # neither. A tag is drawn only for a tooth that is IN THE MASK; the
    # absence of a tag is itself the answer to "is it segmented", and the
    # caption says so in the same words.
    for fdi in sorted(tag_ids or ()):
        fm = face_struct_ids == fdi
        if not fm.any():
            continue
        vi = np.unique(all_faces[fm])
        cx = float(sx[vi].mean()) / supersample
        cy = float(sy[vi].mean()) / supersample
        # Which way is "away from the arch" in THIS image? Read it off the
        # rest of the dentition rather than assuming, since the side views
        # are mirrored by the L-R auto-flip on a per-case basis.
        om = np.isin(face_struct_ids, TOOTH_IDS) & (face_struct_ids != fdi)
        if om.any():
            ax = float(sx[np.unique(all_faces[om])].mean()) / supersample
            out_dir = 1.0 if cx >= ax else -1.0
        else:
            out_dir = 1.0
        img = draw_fdi_tag(img, fdi, cx, cy, out_dir)
    return img, _r_sx, _l_sx


# ── R/L margin labels ─────────────────────────────────────────────────────────


# Camera position → patient face shown in foreground
_VIEW_DISPLAY = {
    "left":      "left",      # camera on patient's left  → patient's left face
    "frontal":   "frontal",
    "right":     "right",     # camera on patient's right → patient's right face
    "posterior": "posterior",
}
def draw_view_label(img, view_name):
    """Draw the view name centred at the bottom of the image."""
    from PIL import ImageFont
    draw  = ImageDraw.Draw(img)
    w, h  = img.size
    fsize = max(16, round(h * 0.05))
    font  = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:   font = ImageFont.truetype(fp, fsize); break
        except (IOError, OSError): pass
    if font is None:
        try:    font = ImageFont.load_default(size=fsize)
        except TypeError: font = ImageFont.load_default()

    label = _VIEW_DISPLAY.get(view_name, view_name)
    draw.text((w // 2, h - fsize - 6), label,
              fill=(0, 0, 0), font=font, anchor="mt")
    return img


def _load_marker_font(fsize):
    from PIL import ImageFont
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(fp, fsize)
        except (IOError, OSError):
            pass
    try:
        return ImageFont.load_default(size=fsize)
    except TypeError:
        return ImageFont.load_default()


def draw_fdi_tag(img, fdi, cx, cy, out_dir, colour=None):
    """
    Write one FDI number beside its tooth, in that tooth's own colour.

    The colour is the link: the caption already tells the model which colour
    the third molar is drawn in, so a tag in the same colour identifies the
    tooth twice over and survives the case where two teeth of similar hue sit
    next to each other. Black stroke for legibility over both the green bone
    and the grey background, the same treatment the R/L markers get.

    `out_dir` is +1/-1 along the image x axis, pointing AWAY from the rest of
    the dentition, and the tag goes there rather than under the tooth. Under a
    lower molar is where the mandibular canal runs, and whether a root apex
    reaches that canal is one of the four things these views are asked --
    mandible_canal_{side}.adjacent_teeth. A tag or a leader line lying over
    the yellow canal hides the evidence for one question while answering
    another. Posterior to the third molar there is only ramus.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    rgb = colour or _TOOTH_PALETTE[TOOTH_IDS.index(fdi) % len(_TOOTH_PALETTE)]
    # 6.5% of the image height -- 33 px at the default 512. The R/L side
    # markers are 7%; a tag much below that stops being readable once the
    # image is downsampled into the prompt, and the stroke and leader-line
    # widths below are derived from it, so they scale with it.
    fsize = max(20, round(h * 0.065))
    font = _load_marker_font(fsize)

    tx, ty = cx + out_dir * w * 0.11, cy
    # Keep the tag on canvas: a third molar can sit close to the frame edge.
    pad = fsize
    tx = min(max(tx, pad), w - pad)
    ty = min(max(ty, pad), h - pad)

    draw.line([(tx, ty), (cx, cy)], fill=(0, 0, 0), width=max(1, fsize // 12))
    draw.text((tx, ty), str(fdi), fill=rgb, font=font, anchor="mm",
              stroke_width=max(2, fsize // 9), stroke_fill=(0, 0, 0))
    return img


def draw_direction_markers(img, labels):
    """
    Stamp radiology-style R/L side markers in the image corners, per view:
      - "left"     view -> labels=["L"]      -> "L" only, top-right
      - "frontal"  view -> labels=["R","L"]  -> "R" top-left + "L" top-right
      - "right"    view -> labels=["R"]      -> "R" only, top-left

    Convention: image-left corresponds to the patient's RIGHT side (standard
    radiographic display convention), so "R" always goes top-left and "L"
    always goes top-right, regardless of which single letter (or both) a
    given view needs.
    """
    draw  = ImageDraw.Draw(img)
    w, h  = img.size
    fsize = max(22, round(h * 0.07))
    font  = _load_marker_font(fsize)
    margin = round(h * 0.03)

    if "R" in labels:
        draw.text((margin, margin), "R", fill=(255, 235, 0), font=font,
                  anchor="la", stroke_width=2, stroke_fill=(0, 0, 0))
    if "L" in labels:
        draw.text((w - margin, margin), "L", fill=(255, 235, 0), font=font,
                  anchor="ra", stroke_width=2, stroke_fill=(0, 0, 0))
    return img



# ── Mesh builder ──────────────────────────────────────────────────────────────

def build_meshes(mask, step_size=2, smooth_sigma=1.5, laplacian_iter=10):
    """Extract, smooth and shade meshes. Returns verts, faces, colors, struct_ids."""
    all_verts, all_faces, all_colors, all_sids = [], [], [], []
    v_offset = 0

    for train_id, base_rgb in STRUCTURES:
        bm = (mask == train_id).astype(np.float32)
        if bm.sum() < 10:
            continue

        # Thin structures (IAC, sinuses) need smaller sigma so the peak stays above 0.5
        if train_id in (3, 4, 5, 6):
            _sigma = min(smooth_sigma, 0.6)
        else:
            _sigma = smooth_sigma

        field = gaussian_filter(bm, sigma=_sigma) if HAS_SCIPY else bm

        if field.max() < 0.5:
            # Retry with raw binary — happens when structure is too thin for Gaussian
            field = bm

        if field.max() < 0.5:
            print(f"  [skip] id={train_id}: field max={field.max():.3f} < 0.5",
                  file=sys.stderr)
            continue

        try:
            verts, faces, _, _ = marching_cubes(field, level=0.5, step_size=step_size)
        except Exception as e:
            print(f"  [skip] id={train_id}: {e}", file=sys.stderr)
            continue
        if laplacian_iter > 0:
            verts = laplacian_smooth(verts, faces, n_iter=laplacian_iter)

        shading = shade_faces(verts, faces)
        rgb     = np.array(base_rgb, np.float32) / 255.
        fc_u8   = (np.clip(rgb * shading[:,None], 0, 1) * 255).astype(np.uint8)

        all_verts.append(verts.astype(np.float32))
        all_faces.append(faces.astype(np.int32) + v_offset)
        all_colors.append(fc_u8)
        all_sids.append(np.full(len(faces), train_id, dtype=np.int32))
        v_offset += len(verts)

        name = ("mandible" if train_id == MANDIBLE_ID
                else "maxilla" if train_id == MAXILLA_ID
                else f"id_{train_id}")
        print(f"  {name:20s}: {len(faces):>7,} faces", file=sys.stderr)

    if not all_verts:
        return None, None, None, None
    return (np.vstack(all_verts),
            np.vstack(all_faces),
            np.vstack(all_colors),
            np.concatenate(all_sids))


# ── Case-level ────────────────────────────────────────────────────────────────

def render_case(mask_path, out_dir, case_id,
                step_size=2, smooth_sigma=1.5, laplacian_iter=10,
                img_size=512, supersample=2, bone_alpha=0.55,
                rot_deg=45.0, facts_path=None):
    """
    Render the ACTIVE_VIEWS (left, frontal, right) as separate PNGs and write
    a per-case caption sidecar JSON alongside them. No combined/grid image is
    produced anymore.

    If `facts_path` is given, the corresponding case-specific findings (only
    ian_close_teeth is in scope for 3D views -- see build_3d_view_addenda)
    are appended to each view's static caption at generation time, so the
    caption sidecar is written with facts baked in rather than requiring a
    separate post-hoc captioning pass.

    Returns dict: {view_name: {"path": <png path>, "caption": <str>}}
    """

    print(f"[{case_id}] loading ...", file=sys.stderr)
    raw_nib  = nib.load(str(mask_path))
    can_nib  = nib.as_closest_canonical(raw_nib)
    mask     = can_nib.get_fdata().astype(np.int32)

    # Measured from the canonical mask, which is the array build_meshes turns
    # into the geometry -- so the caption describes this exact picture.
    volumes = mask_label_volumes(mask, can_nib.header.get_zooms())
    drawn = drawn_structures(volumes)

    facts = load_case_facts(facts_path)
    view_addenda = build_3d_view_addenda(facts, volumes)

    print(f"[{case_id}] building meshes ...", file=sys.stderr)
    av, af, ac, asids = build_meshes(mask, step_size, smooth_sigma, laplacian_iter)
    if av is None:
        print("  [warn] no structures found", file=sys.stderr)
        return {}

    print(f"  Total: {len(af):,} faces", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)

    saved = {}
    for view_name, cam, labels in make_viewpoints(rot_deg):
        if view_name not in ACTIVE_VIEWS:
            continue
        print(f"[{case_id}] rendering {view_name} ...", file=sys.stderr)
        # Tag only the two third molars this view is asked about: 28/38 on
        # 3d_left, 18/48 on 3d_right, nothing on 3d_frontal. Tagging all 32
        # would bury the two the side-view questions are about, which is the
        # same reason view_caption names only these two colours.
        tag_ids = {fdi for fdi, _ in WISDOM_TEETH.get(view_name, ())}
        img, r_sx, l_sx = render_view(av, af, ac, asids, cam,
                                      img_size=img_size, supersample=supersample,
                                      bone_alpha=bone_alpha, tag_ids=tag_ids)
        # img = draw_view_label(img, view_name)
        img = draw_direction_markers(img, labels)
        fpath = os.path.join(out_dir, f"{case_id}_3d_{view_name}.png")
        img.save(fpath)
        print(f"  → {fpath}", file=sys.stderr)

        caption = view_caption(view_name, drawn)
        addendum = view_addenda.get(view_name)
        if addendum:
            caption = f"{caption} {addendum}"
        saved[view_name] = {"path": fpath, "caption": caption}

    # Write per-case caption sidecar so downstream (build_vqa_pairs.py) can
    # attach captions without importing this module's VIEW_CAPTIONS directly.
    if saved:
        captions_path = os.path.join(out_dir, f"{case_id}_3d_captions.json")
        with open(captions_path, "w") as f:
            json.dump(
                {f"3d_{view_name}": info["caption"] for view_name, info in saved.items()},
                f, indent=2
            )
        print(f"  captions → {captions_path}", file=sys.stderr)

    return saved


def case_id_from_mask(path):
    stem = Path(path).name
    for s in (".nii.gz", ".nii"):
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem.replace("ToothFairy", "").replace("_0000", "")


# ── Fact-based caption addenda ────────────────────────────────────────────────
#
# Per schema.json (v6.1), the 3D views cover the mandibular-canal facts
# (mandible_canal_right/left, images_needed: 3d_left / 3d_right respectively)
# among the six in-scope fields -- teeth_present/absent/crowns/implants/bridge
# all belong to `panoramic` instead, so `ian_close_teeth` and the case's
# acquisition `fov` are what apply here.
#
# ian_close_teeth in the facts file is a flat list of FDI numbers (not
# pre-split by side), so it's split here using the same RIGHT_ARCH_FDI /
# LEFT_ARCH_FDI membership used for L-R flip detection above. Per the
# schema's view mapping: LEFT-side canal findings surface on 3d_left, which
# is the view that shows the patient's left side, and right-side findings on
# 3d_right.
#
# fov is not side-specific -- it states how much of the maxilla/condyles the
# scan volume covers at all -- so its fragment goes on every view, 3d_frontal
# included (which is where mandible_scope/maxilla_scope are answered).
#
# These are ONLY ever surfaced via image captions, never injected anywhere
# else in the prompt. A field only produces text when non-empty -- an empty
# ian_close_teeth and no fov (or a missing/absent facts file) leaves the
# caption as the static viewpoint description alone.

def load_case_facts(facts_path):
    """Load a case's facts JSON. Returns {} if no path given or file missing."""
    if not facts_path:
        return {}
    p = Path(facts_path)
    if not p.exists():
        print(f"  [warn] facts file not found: {p}", file=sys.stderr)
        return {}
    return json.loads(p.read_text())


def _normalize_int_list(raw):
    """Flat list of FDI ints -> sorted list, tolerating None/missing/non-int junk."""
    if not raw:
        return []
    out = set()
    for item in raw:
        if isinstance(item, int) and not isinstance(item, bool):
            out.add(item)
    return sorted(out)


def _join_teeth(fdis):
    """'14, 15 and 18' style join for a sorted list of FDI numbers."""
    nums = sorted(fdis)
    if not nums:
        return ""
    if len(nums) == 1:
        return str(nums[0])
    return ", ".join(str(n) for n in nums[:-1]) + f" and {nums[-1]}"


# facts.structured.fov describes how much of the anatomy the acquisition
# actually covers. Unlike ian_close_teeth it is NOT side-specific -- it is a
# property of the volume, so the same fragment goes on all three views (it
# backs mandible_scope/maxilla_scope on 3d_frontal and the condyle facts on
# 3d_left/3d_right alike). A key that is absent from fov says nothing about
# that structure, so it contributes no text -- only the values actually
# present are stated.
FOV_PHRASES = {
    "maxilla": {
        "partial":  "maxilla partially included",
        "complete": "maxilla fully included",
        "full":     "maxilla fully included",
        "excluded": "maxilla not included",
        # Aliases so a hand-edited or externally-produced facts file spelling
        # this the way the sentence reads still lands on the same phrase
        # instead of being passed through verbatim.
        "not included":  "maxilla not included",
        "not_included":  "maxilla not included",
        "none":          "maxilla not included",
    },
    "condyles": {
        "excluded": "mandibular condyles excluded",
        "partial":  "mandibular condyles partially included",
        "complete": "mandibular condyles fully included",
        "full":     "mandibular condyles fully included",
        "included": "mandibular condyles included",
    },
}
FOV_KEY_ORDER = ("maxilla", "condyles")


def build_fov_fragment(facts, coverage=None):
    """
    'Acquisition FOV: maxilla partially included, mandibular condyles
    excluded.' -- or None when the case carries no fov information. Unknown
    enum values are passed through verbatim rather than dropped, so a new
    value shows up in the caption instead of silently vanishing.

    THE MASK OVERRULES THE FACTS on the maxilla. `coverage` is
    arch_coverage()'s verdict, measured from the voxels this very image was
    rendered from; the facts file's fov is a property of the ACQUISITION as
    someone recorded it, and the two disagree often enough to matter -- 37 of
    622 cases say "partially included" over a mask with no maxilla voxels at
    all, S0002 included. The picture is what the VLM has to answer from, so
    when the mask says the maxillary bone is not there the caption says so,
    whatever fov claims. The condyles have no mask label and are left alone.

    The maxilla line is stated whenever `coverage` is known, so a mask-only
    case (no facts file, or a facts file with no fov key) still gets it.
    """
    fov = facts.get("structured", {}).get("fov")
    fov = fov if isinstance(fov, dict) else {}
    maxilla_imaged = (coverage or {}).get("maxilla", {}).get("bone")

    parts = []
    for key in FOV_KEY_ORDER:
        if key == "maxilla" and maxilla_imaged is not None:
            parts.append("maxilla not included" if not maxilla_imaged
                         else FOV_PHRASES["maxilla"].get(
                             str(fov.get(key, "")).strip().lower(),
                             "maxilla partially included"))
            continue
        value = fov.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip().lower()
        parts.append(FOV_PHRASES[key].get(value, f"{key} {value}"))
    if not parts:
        return None
    return f"Acquisition FOV: {', '.join(parts)}."


# ── Tooth-presence caption fragments ─────────────────────────────────────────
#
# The renderer draws every segmented tooth (STRUCTURES, above) opaque inside
# the translucent bone, but the static caption names only the four non-tooth
# colors -- so to the VLM an arch with no teeth in it is indistinguishable
# from an arch whose teeth simply were not rendered. P123 is the worked
# example: a fully edentulous mandible with the canal breaching the crest,
# answered alveolar_bone_atrophy_mandible.present = false on the evidence
# "a continuous U-shaped tooth-bearing arch with no edentulous regions" --
# teeth it had no reason to believe would have been drawn at all.
#
# alveolar_bone_atrophy_{mandible,maxilla} gate on exactly that ("Only answer
# if at least one edentulous area is present") and both are asked on
# 3d_frontal, so the frontal caption states which teeth are missing. Same
# facts field and same wording style as create_panoramic.py's absent-teeth
# fragment, which has always done this for the panoramic view.
#
# Stated per arch, and only for arches the acquisition actually covers:
# P123's facts list all 32 teeth as absent, but its maxilla is outside the
# FOV entirely, and "the maxilla is edentulous" would be a fresh false
# positive on alveolar_bone_atrophy_maxilla. present_label_ids -- the mask's
# label inventory -- is what decides that.

def build_arch_coverage_fragment(facts, coverage=None):
    """
    'The maxilla is outside this acquisition and is not rendered.' for an
    arch the mask has no bone for. Goes on EVERY view: P123 shows what its
    absence costs -- a mandible-only volume whose caption still announced a
    "maxilla (tan)" that is nowhere in the image, which is exactly the kind
    of expected-but-unseen structure the VLM then describes anyway.

    Now largely belt-and-braces: view_caption drops the missing structure from
    the color key outright and build_fov_fragment states "maxilla not
    included", so this only speaks when the fov line does not already cover
    that arch. Repeating it would be noise.
    """
    structured = facts.get("structured", {})
    if coverage is None:
        coverage = arch_coverage(None, structured)
    if coverage is None:
        return None
    fov = structured.get("fov")
    fov_keys = set(fov) if isinstance(fov, dict) else set()
    # build_fov_fragment now states the maxilla whenever coverage is known,
    # not only when the facts carry an fov key.
    spoken_for = fov_keys | {"maxilla"}

    parts = [f"The {arch} is outside this acquisition and is not rendered."
             for arch in ARCH_ORDER
             if not coverage.get(arch, {}).get("bone") and arch not in spoken_for]
    return " ".join(parts) if parts else None


def build_partial_upper_teeth_fragment(coverage):
    """
    'The upper teeth are cut across by the volume boundary...' -- for the one
    configuration the caption otherwise describes misleadingly: no maxillary
    BONE in the volume, but maxillary TEETH in it.

    256 of 622 cases are in this state, and what is segmented there is the
    crowns, not the teeth. Median volume of one segmented upper tooth is
    476 mm^3 when the maxillary bone is in the volume and 222 mm^3 (47%) when
    it is not, while mandible bone barely moves between the two groups
    (42,180 vs 39,880 mm^3) -- so this is the FOV cutting the upper arch, not
    a different population. P217's render shows it directly: the upper teeth
    are flat sliced-off stubs with the cut plane visible through them.

    Without this the caption's own inventory oversells them -- P217 read "All
    16 maxillary teeth are present" with nothing to say those sixteen are
    crowns sheared off at the boundary.

    It also matches how the radiologists write these cases. Of the 33
    reference reports in this group that describe upper teeth at all, every
    one hedges: "as far as can be visualized", "as far as visible", "only the
    dental elements are partially visualized" (P477). The mask cannot say
    whether a given finding survived the crop, so the caption states the
    limitation and leaves the reading to the model.

    NOT for the arch with no upper teeth at all -- build_tooth_presence
    _fragment already says "Maxillary teeth are not included in the view",
    and there is nothing partial about it.

    Goes on ALL THREE views, like the FOV line and unlike the tooth
    inventory: it is a property of the acquisition, and the side views are
    where upper_left/right_wisdom_tooth are asked -- 18 and 28 are exactly
    the teeth a short volume shears.
    """
    maxilla = (coverage or {}).get("maxilla", {})
    if not coverage or maxilla.get("bone") or not maxilla.get("teeth"):
        return None
    return ("The upper teeth are cut across by the volume boundary and only "
            "partially segmented, so maxillary findings can be judged only as "
            "far as visible -- roots, apices and periapical bone are not "
            "evaluable.")


def build_tooth_presence_fragment(facts, coverage=None):
    """
    'Every segmented tooth is drawn inside the translucent bone... the
    mandible is fully edentulous.' -- or None when the case records no
    tooth inventory at all (nothing recorded means nothing claimed).

    EDENTULOUS vs NOT SCANNED is the distinction this has to keep straight,
    and it is the one that produced S0002's caption: the report lists 11-28 as
    absent because the maxilla was never in the volume, the facts file records
    that verbatim, and the arch came out described as "fully edentulous" -- a
    clinical claim, and a false one, about anatomy nobody imaged. An arch with
    NO tooth labels in the mask therefore gets "maxillary teeth are not
    included in the view", which says only what the picture shows.

    `coverage` is arch_coverage()'s verdict; the mask's tooth inventory is
    what decides, not the facts file's list.
    """
    structured = facts.get("structured", {})
    present = set(_normalize_int_list(structured.get("teeth_present")))
    absent  = set(_normalize_int_list(structured.get("teeth_absent")))
    if coverage is None:
        coverage = arch_coverage(None, structured)
    if not present and not absent:
        return None

    parts = ["Every segmented tooth is drawn opaque inside the translucent "
             "bone, so a tooth you cannot see in this image is not present."]
    for arch in ARCH_ORDER:
        adj = ARCH_ADJECTIVE[arch]
        arch_cov = (coverage or {}).get(arch, {})
        if coverage is not None and not arch_cov.get("teeth"):
            # No tooth of this arch is in the mask. Whether the patient has
            # none or the scan simply stopped short is not visible here, and
            # only the arch whose BONE is also missing can be called out --
            # for the other, silence beats guessing.
            if not arch_cov.get("bone"):
                parts.append(f"{adj.capitalize()} teeth are not included in "
                             f"the view.")
            continue
        arch_present = sorted(present & ARCH_FDI[arch])
        arch_absent  = sorted(absent  & ARCH_FDI[arch])
        if not arch_present and not arch_absent:
            continue
        if not arch_absent:
            parts.append(f"All {len(arch_present)} {adj} teeth are present.")
        elif not arch_present:
            # "Fully edentulous" is a claim about the ALVEOLAR RIDGE, so it
            # needs the bone in the volume to be sayable at all. Seven cases
            # (P063, P076, P140, P330, P370, P505, P512) have zero maxillary
            # bone, a single upper tooth segmented, and a facts file listing
            # all sixteen as absent -- which rendered as "no maxillary tooth
            # is present" over a picture containing one. Without the bone the
            # caption states only what the segmentation shows.
            if arch_cov.get("bone", True):
                parts.append(f"No {adj} tooth is present -- the {arch} is "
                             f"fully edentulous.")
            else:
                parts.append(f"No {adj} tooth is segmented in this volume.")
        else:
            word = "Tooth" if len(arch_absent) == 1 else "Teeth"
            parts.append(f"{word} absent from the {arch}: "
                         f"{_join_teeth(arch_absent)}.")

    return " ".join(parts) if len(parts) > 1 else None


def build_wisdom_fragment(view, facts, coverage=None):
    """
    Name the two third molars on the side this view faces, and nothing else
    -- 28/38 on 3d_left, which faces the patient's left side, and 18/48 on
    3d_right. Returns None for the frontal view, for
    cases with no tooth inventory, or when neither arch on this side is
    imaged.
    """
    if view not in WISDOM_TEETH:
        return None
    structured = facts.get("structured", {})
    present = set(_normalize_int_list(structured.get("teeth_present")))
    absent  = set(_normalize_int_list(structured.get("teeth_absent")))
    if not present and not absent:
        return None

    if coverage is None:
        coverage = arch_coverage(None, structured)

    clauses = []
    for fdi, position in WISDOM_TEETH[view]:
        arch = "maxilla" if fdi in ARCH_FDI["maxilla"] else "mandible"
        # "no tooth 18 is segmented (upper third molar absent)" is the same
        # edentulous-vs-not-scanned confusion as in
        # build_tooth_presence_fragment: an arch with neither bone nor teeth
        # in the mask supports no claim about its third molar either way.
        arch_cov = (coverage or {}).get(arch, {})
        if coverage is not None and not (arch_cov.get("bone") or arch_cov.get("teeth")):
            continue
        color = WISDOM_COLOR_NAMES[fdi][0]
        if fdi in present:
            clauses.append(f"{fdi} is the most posterior {position} tooth on "
                           f"this side, rendered {color} and tagged {fdi} in "
                           f"the image, the tag drawn in the same colour as "
                           f"the tooth and joined to it by a line")
        elif fdi in absent:
            clauses.append(f"no tooth {fdi} is segmented ({position} third "
                           f"molar absent)")
    if not clauses:
        return None
    return "Third molars (wisdom teeth): " + "; ".join(clauses) + "."


def build_3d_view_addenda(facts, volumes=None):
    """
    Returns {"left": <str or None>, "frontal": <str or None>, "right": <str
    or None>} -- terse fragment addenda to append to the view caption.
    Sources: the case's fov (same text on every view), ian_close_teeth
    (side-specific, the one per-side fact that maps to 3D views), the third
    molars named on the side views only, and the tooth inventory stated on
    the frontal view only.

    `volumes` is mask_label_volumes()'s {label: mm^3}. Passing it makes every
    claim about what is or is not in the acquisition measured from the mask
    the images were rendered from; omitting it falls back to the facts file's
    present_label_ids, which is all a caller without a mask has.
    """
    structured = facts.get("structured", {})
    ian_close_teeth = _normalize_int_list(structured.get("ian_close_teeth"))

    right_side = [t for t in ian_close_teeth if t in RIGHT_ARCH_FDI]
    left_side  = [t for t in ian_close_teeth if t in LEFT_ARCH_FDI]

    coverage = arch_coverage(volumes, structured)
    fov_fragment = build_fov_fragment(facts, coverage)

    def _fragment(teeth):
        if not teeth:
            return None
        word = "Tooth" if len(teeth) == 1 else "Teeth"
        return f"{word} {_join_teeth(teeth)} close to mandibular canal."

    coverage_fragment = build_arch_coverage_fragment(facts, coverage)
    # Sits with the FOV line rather than with the tooth inventory: it
    # qualifies the acquisition, and it belongs on the side views too, where
    # the upper wisdom-tooth facts are asked.
    partial_upper = build_partial_upper_teeth_fragment(coverage)

    def _addendum(*fragments):
        parts = [p for p in (fov_fragment, coverage_fragment, partial_upper,
                             *fragments) if p]
        return " ".join(parts) if parts else None

    # left-side canal findings show up in the 3d_left view, which is the one
    # that faces the patient's left side; right-side findings -> 3d_right.
    return {
        "left":    _addendum(_fragment(left_side),
                             build_wisdom_fragment("left", facts, coverage)),
        "frontal": _addendum(build_tooth_presence_fragment(facts, coverage)),
        "right":   _addendum(_fragment(right_side),
                             build_wisdom_fragment("right", facts, coverage)),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def assert_schema_wisdom_colors(schema_path=None) -> None:
    """
    schema.json's four wisdom-tooth questions name the render colour of the
    tooth they ask about ("FDI 18, drawn in indigo"), so the colour now lives
    in two files. The palette assert above keeps WISDOM_COLOR_NAMES honest
    against the renderer; this keeps the SCHEMA honest against
    WISDOM_COLOR_NAMES, so re-ordering the palette cannot leave the prompt
    telling the model to look for a colour that is no longer drawn.

    Warns rather than raises, and silently returns if the schema is missing or
    unreadable: rendering images is useful even when the schema has moved on,
    and a hard failure here would stop a whole batch over prompt wording.
    """
    path = Path(schema_path) if schema_path else (
        REPO_ROOT / "schema" / "schema.json")
    try:
        schema = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    facts = {f.get("fact_id"): f
             for section in ("mandible", "maxilla")
             for f in schema.get(section, [])}
    fact_for_fdi = {18: "upper_right_wisdom_tooth", 28: "upper_left_wisdom_tooth",
                    38: "lower_left_wisdom_tooth", 48: "lower_right_wisdom_tooth"}
    for fdi, fact_id in fact_for_fdi.items():
        fact = facts.get(fact_id)
        if not fact:
            continue
        colour = WISDOM_COLOR_NAMES[fdi][0]
        text = f"{fact.get('question_text', '')} {fact.get('description', '')}"
        if colour not in text:
            print(f"  [warn] {path.name}: {fact_id} does not mention the colour "
                  f"tooth {fdi} is rendered in ({colour!r}) -- the prompt and "
                  f"the render may have drifted apart", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--mask",      help="Single mask .nii.gz")
    grp.add_argument("--masks-dir", help="Directory of mask .nii.gz")
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--case-id",        default=None)
    ap.add_argument("--facts-file",     default=None,
                    help="Path to this case's facts JSON (single-mask mode only)")
    ap.add_argument("--facts-dir",      default=None,
                    help="Directory of {case_id}.json facts files (batch mode; "
                         "matched by case_id, missing files are skipped with a warning)")
    ap.add_argument("--step-size",      type=int,   default=2)
    ap.add_argument("--smooth-sigma",   type=float, default=1.5)
    ap.add_argument("--laplacian-iter", type=int,   default=10)
    ap.add_argument("--img-size",       type=int,   default=512)
    ap.add_argument("--supersample",    type=int,   default=2)
    ap.add_argument("--rot-deg",        type=float, default=60.0,
                    help="Oblique view rotation angle in degrees")
    ap.add_argument("--bone-alpha",     type=float, default=0.75,
                    help="Bone opacity 0–1 (default 0.75; 1.0 = fully opaque)")
    ap.add_argument("--limit",          type=int,   default=None)
    args = ap.parse_args()

    assert_schema_wisdom_colors()

    import glob as _glob
    if args.mask:
        cid = args.case_id or case_id_from_mask(args.mask)
        render_case(args.mask, args.out_dir, cid,
                    args.step_size, args.smooth_sigma, args.laplacian_iter,
                    args.img_size, args.supersample, args.bone_alpha, args.rot_deg,
                    facts_path=args.facts_file)
    else:
        masks = sorted(_glob.glob(os.path.join(args.masks_dir, "*.nii.gz")))
        if args.limit:
            masks = masks[:args.limit]
        for mp in masks:
            cid = case_id_from_mask(mp)
            fp = (os.path.join(args.facts_dir, f"{cid}.json")
                  if args.facts_dir else None)
            render_case(mp, args.out_dir, cid,
                        args.step_size, args.smooth_sigma, args.laplacian_iter,
                        args.img_size, args.supersample, args.bone_alpha, args.rot_deg,
                        facts_path=fp)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


"""
python code/pipeline/preprocess/create_3d_renders.py --mask dataset/training/masks/A004.nii.gz --out-dir outputs/3d_renders --rot-deg 60
"""