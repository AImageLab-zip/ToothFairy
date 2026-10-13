#!/usr/bin/env python3
"""
code/pipeline/preprocess/create_sinus_detail.py

Multi-view CBCT detail image for each maxillary sinus.

Layout  (n_slices columns per row, default 3)
──────
Row 1 — CORONAL  (constant-Y slices, posterior → anterior through sinus bbox)
  X crop : midline → ipsilateral border  (half the total X width)
  Z crop : upper half of volume
  Slice positions: evenly spaced through sinus Y bbox

Row 2 — SAGITTAL (constant-X slices, midline → distal through sinus bbox)
  Y crop : full volume extent
  Z crop : upper half of volume
  Slice positions: evenly spaced through sinus X bbox

Overlays on every sinus slice:
  • The sinus itself — bright green cross-marker ('+') centred on the region,
    marking the area to judge. No outline.
  • Teeth of the IPSILATERAL MAXILLARY QUADRANT ONLY that the slice cuts
    through — FDI tag only, in its own colour, no outline. Right sinus tags
    18-11, left sinus tags 28-21; not restricted to x6/x7/x8 within that
    quadrant. Mandibular and contralateral teeth the crop happens to contain
    stay untagged (see SINUS_TAG_TIDS), and the maxilla itself is neither
    marked nor tagged.

Output:
  {case_id}_sinus_{right,left}_detail.png
  {case_id}_sinus_captions.json   -- {"sinus_right_detail": "<caption>", ...}

Captions are static descriptions of what each image shows, written at
generation time (see the Caption section below); no facts.json is involved,
since none of the case-level fields map to these images.

Usage:
    python code/pipeline/preprocess/create_sinus_detail.py \
        --volume  dataset/training/images/A004_0000.nii.gz \
        --mask    dataset/training/masks/A004.nii.gz \
        --out-dir outputs/sinus_detail/ \
        --case-id A004
"""

import os, sys, json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── Label / colour constants ──────────────────────────────────────────────────
SINUS_RIGHT = 6
SINUS_LEFT  = 5

SINUS_MARKER_RGB = (0, 255, 0)   # bright green — the region-of-interest marker

_TOOTH_PALETTE_RGBA = [
    (220, 215, 175, 180), ( 90, 170, 220, 180), (235, 130,  55, 180),
    (175,  95, 185, 180), ( 65, 185, 155, 180), (230,  75,  75, 180),
    (240, 215,  60, 180), (105, 110, 215, 180), (100, 215, 165, 180),
    (215,  95, 165, 180), (165, 215,  90, 180), ( 95, 165, 215, 180),
    (215, 165,  90, 180), (100, 215, 215, 180), (165,  95, 215, 180),
    (195, 195,  65, 180), (215, 190, 185, 180), ( 70, 145, 195, 180),
    (195, 120,  65, 180), (145,  75, 155, 180), ( 55, 155, 135, 180),
    (195,  60,  60, 180), (200, 185,  55, 180), ( 80,  90, 185, 180),
    ( 80, 185, 135, 180), (185,  75, 140, 180), (140, 185,  75, 180),
    ( 75, 140, 185, 180), (185, 140,  75, 180), ( 85, 185, 185, 180),
    (140,  75, 185, 180), (185,  75,  75, 180),
]
_ALL_TOOTH_TIDS = (list(range(11, 19)) + list(range(21, 29)) +
                   list(range(31, 39)) + list(range(41, 49)))
TOOTH_COLORS = {tid: _TOOTH_PALETTE_RGBA[i % len(_TOOTH_PALETTE_RGBA)]
                for i, tid in enumerate(_ALL_TOOTH_TIDS)}

# Which teeth may be tagged on each sinus image. ONLY the ipsilateral MAXILLARY
# quadrant -- a maxillary sinus has no anatomical relationship to a mandibular
# tooth or to the opposite side's.
#
# The tag loop used to walk _ALL_TOOTH_TIDS, i.e. every segmented tooth the
# slice happened to cut through, and the sagittal crop takes the FULL Y extent
# with only the upper half of Z -- which on a tall FOV still reaches the lower
# arch. So A075's RIGHT sinus image came out tagging 48, and A073's tagged
# 44/45. The VQA fact asks whether a root projects into the sinus floor and
# answers with an FDI list, so a mandibular number sitting in the picture is
# not cosmetic: it is a wrong answer offered to the model in its own prompt.
# Restricting the loop is the fix; the crop is left alone, since the extra
# anatomy is useful context as long as it is not labelled as this sinus's.
SINUS_TAG_TIDS = {"right": list(range(11, 19)),    # FDI 18-11, upper right
                  "left":  list(range(21, 29))}    # FDI 28-21, upper left



# ── Windowing ─────────────────────────────────────────────────────────────────

def _apply_window(arr, lo_val, hi_val):
    out = np.clip(
        (arr.astype(np.float32) - lo_val) / max(hi_val - lo_val, 1e-6),
        0., 1.
    )
    return (out * 255).astype(np.uint8)


# ── Region marker ─────────────────────────────────────────────────────────────

def _draw_cross_marker(draw, disp_mask, rgb=SINUS_MARKER_RGB):
    """
    Mark a region with a '+' centred on it instead of tracing its border.

    The arms leave a gap in the middle so the tissue under the centre stays
    visible, and both arm length and stroke width scale with the region so the
    marker survives the later rescale to tile width.
    """
    ys, xs = np.where(disp_mask)
    if len(ys) == 0:
        return
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    # A concave slice can put the centroid outside the region — snap to the
    # nearest region pixel so the marker always sits on the sinus.
    if not disp_mask[cy, cx]:
        i = int(np.argmin((ys - cy)**2 + (xs - cx)**2))
        cy, cx = int(ys[i]), int(xs[i])
    span = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    arm  = max(8, int(0.35 * span))
    gap  = max(2, arm // 3)
    lw   = max(2, int(round(span / 20)))
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.line([(cx + dx*gap, cy + dy*gap), (cx + dx*arm, cy + dy*arm)],
                  fill=rgb, width=lw)


def _fdi_tag(draw, disp_mask, fdi, font, rgb):
    ys, xs = np.where(disp_mask)
    if len(ys) == 0:
        return
    cy, cx = int(ys.mean()), int(xs.mean())
    text = str(fdi)
    fh   = font.size if hasattr(font, 'size') else 11
    tw   = len(text) * (fh * 6 // 10)
    draw.rectangle([cx - tw//2 - 2, cy - fh//2 - 1,
                    cx + tw//2 + 2, cy + fh//2 + 1], fill=(0, 0, 0))
    draw.text((cx - tw//2, cy - fh//2), text, fill=rgb, font=font)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _get_font(size=14):
    from PIL import ImageFont
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_arrow(draw, x0, y, x1, left_label, right_label,
                color=(220, 40, 40), font_size=14):
    font = _get_font(font_size)
    cw   = font_size * 7 // 10
    lw   = len(left_label)  * cw
    rw   = len(right_label) * cw
    draw.text((x0,          y - font_size - 2), left_label,  fill=color, font=font)
    draw.text((x1 - rw - 2, y - font_size - 2), right_label, fill=color, font=font)
    ax0, ax1 = x0 + lw + 6, x1 - rw - 8
    if ax1 > ax0 + 20:
        draw.line([(ax0, y), (ax1, y)], fill=color, width=2)
        aw, ah = 10, 5
        draw.polygon([(ax1, y), (ax1-aw, y-ah), (ax1-aw, y+ah)], fill=color)


def _draw_rotated_label(canvas, text, cx, cy, color=(220, 40, 40), font_size=16):
    from PIL import Image as _PIL
    font = _get_font(font_size)
    cw   = font_size * 7 // 10
    tw, th = len(text) * cw, font_size + 6
    tmp = _PIL.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((0, 2), text, fill=(*color, 255), font=font)
    rot = tmp.rotate(90, expand=True)
    canvas.paste(rot, (max(0, cx - rot.width//2), max(0, cy - rot.height//2)), rot)


def _scale_to_width(img, w):
    sw, sh = img.size
    nh = max(1, round(sh * w / max(sw, 1)))
    return img.resize((w, nh), Image.LANCZOS)


def _autocrop(img, threshold=15):
    """Remove black margins by cropping to the bounding box of non-black pixels."""
    arr    = np.array(img)
    bright = arr.max(axis=2) > threshold
    rows   = np.where(bright.any(axis=1))[0]
    cols   = np.where(bright.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img.crop((int(cols[0]), int(rows[0]),
                     int(cols[-1]) + 1, int(rows[-1]) + 1))


def _detect_orientation(mask):
    """
    Detect spatial orientation from tooth segmentation masks.
    Returns:
        r_high_x : True if right teeth (FDI 11-18) are at HIGHER X than left (21-28)
        a_high_y : True if anterior teeth (11-13, 21-23) are at HIGHER Y than posterior
    """
    rt   = np.argwhere(np.isin(mask, list(range(11, 19))))
    lt   = np.argwhere(np.isin(mask, list(range(21, 29))))
    ant  = np.argwhere(np.isin(mask, [11, 12, 13, 21, 22, 23]))
    post = np.argwhere(np.isin(mask, [16, 17, 18, 26, 27, 28]))
    r_high_x = (rt[:, 0].mean() > lt[:, 0].mean()
                if len(rt) > 0 and len(lt) > 0 else True)
    a_high_y = (ant[:, 1].mean() > post[:, 1].mean()
                if len(ant) > 0 and len(post) > 0 else True)
    return bool(r_high_x), bool(a_high_y)


# ── Main composite builder ────────────────────────────────────────────────────

def make_sinus_composite(volume, mask, sinus_id,
                          n_slices=3, out_w=1344, pad_frac=0.10,
                          window_center=None, window_width=None,
                          lo_pct=0.5, hi_pct=99.5):
    sinus_vox = np.argwhere(mask == sinus_id)
    if len(sinus_vox) < 10:
        return None

    side = "right" if sinus_id == SINUS_RIGHT else "left"
    tag_tids = SINUS_TAG_TIDS[side]

    sx0, sy0, _ = sinus_vox.min(axis=0)
    sx1, sy1, _ = sinus_vox.max(axis=0)
    py = max(3, round((sy1 - sy0) * pad_frac))

    Z_MID = mask.shape[2] // 2
    z_lo, z_hi = Z_MID, mask.shape[2]-1
    y_lo_sag, y_hi_sag = 0, mask.shape[1]-1

    # ── Detect orientation from tooth masks (data-driven, not assumed) ────────
    # r_high_x: True  → right teeth (11-18) are at HIGH X in the mask array
    # a_high_y: True  → anterior teeth are at HIGH Y
    r_high_x, a_high_y = _detect_orientation(mask)

    # Coronal X crop: ipsilateral half of the volume
    # "Ipsilateral" = the X half where THIS sinus's teeth actually are.
    X_MID = mask.shape[0] // 2
    if side == "right":
        if r_high_x:                   # right teeth at HIGH X → crop [MID, MAX]
            x_lo_cor, x_hi_cor = X_MID, mask.shape[0]-1
        else:                          # right teeth at LOW  X → crop [0, MID]
            x_lo_cor, x_hi_cor = 0, X_MID
    else:                              # left sinus
        if r_high_x:                   # left teeth at LOW  X → crop [0, MID]
            x_lo_cor, x_hi_cor = 0, X_MID
        else:                          # left teeth at HIGH X → crop [MID, MAX]
            x_lo_cor, x_hi_cor = X_MID, mask.shape[0]-1

    # ── Slice position helper ─────────────────────────────────────────────────
    def _pos(lo, hi, n):
        return [int(lo + (i+1)/(n+1)*(hi-lo)) for i in range(n)]

    y_positions = _pos(sy0 - py, sy1 + py, n_slices)

    # Sagittal X positions: midline → distal
    # "Midline side" = sinus boundary facing the centre; "distal" = outer edge.
    if side == "right":
        # Right sinus midline = LOWER X end of sinus bbox (toward centre)
        # when right teeth are at high X, and UPPER X end otherwise.
        if r_high_x:
            x_positions = _pos(sx0, sx1, n_slices)   # low→high X (mid→distal)
        else:
            x_positions = _pos(sx1, sx0, n_slices)   # high→low X (mid→distal)
    else:
        # Left sinus midline = UPPER X end when left teeth are at LOW X.
        if r_high_x:                   # left teeth at LOW X → midline=sx1
            x_positions = _pos(sx1, sx0, n_slices)   # high→low X (mid→distal)
        else:                          # left teeth at HIGH X → midline=sx0
            x_positions = _pos(sx0, sx1, n_slices)   # low→high X (mid→distal)

    # Display transforms — always put patient-right on image-left, anterior on
    # image-left for sagittal, so L/R and A/P markers are deterministic.
    # Coronal (X,Z): flip X cols if right=high-X (high→low col = right on left)
    #                no col flip if right=low-X  (low col = right already on left)
    def cor_xform(m):   # (X, Z) array → display bool
        return (m.T[::-1, ::-1] if r_high_x else m.T[::-1, :]).astype(bool)

    # Sagittal (Y,Z): flip Y cols if anterior=high-Y; no flip if anterior=low-Y
    def sag_xform(m):   # (Y, Z) array → display bool
        return (m.T[::-1, ::-1] if a_high_y else m.T[::-1, :]).astype(bool)

    # Fixed HU window when both bounds are given, else percentiles of this
    # volume. Percentiles remain the DEFAULT here, unlike create_tooth_detail:
    # this composite is read for mucosal thickening, which lives in a narrow
    # soft-tissue band, so the right fixed window is a soft-tissue one
    # (roughly --window-center 400 --window-width 2000), not the wide
    # bone/enamel window that script needs. Switching the default blind would
    # be a regression, so it is opt-in until the sinus renders are reviewed.
    if window_center is not None and window_width is not None:
        lo_val = float(window_center) - float(window_width) / 2.0
        hi_val = float(window_center) + float(window_width) / 2.0
    else:
        lo_val = float(np.percentile(volume, lo_pct))
        hi_val = float(np.percentile(volume, hi_pct))

    tag_font = _get_font(max(10, out_w // (n_slices * 30)))

    # ── Coronal slices ────────────────────────────────────────────────────────
    def coronal_slice(y_pos):
        yp  = int(np.clip(y_pos, 0, mask.shape[1]-1))
        vc  = volume[x_lo_cor:x_hi_cor+1, yp, z_lo:z_hi+1].astype(np.float32)
        mc  = mask  [x_lo_cor:x_hi_cor+1, yp, z_lo:z_hi+1]
        u8  = _apply_window(vc, lo_val, hi_val)
        disp_u8 = u8.T[::-1, ::-1] if r_high_x else u8.T[::-1, :]
        arr = np.stack([disp_u8] * 3, axis=-1)
        img  = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # Cross-marker on the sinus — points at the region of interest without
        # tracing (and so without hiding) its border.
        sd = cor_xform(mc == sinus_id)
        if sd.any():
            _draw_cross_marker(draw, sd)
        # Teeth: FDI tag only, no outline. Ipsilateral maxillary quadrant only.
        for fdi in tag_tids:
            td = cor_xform(mc == fdi)
            if not td.any():
                continue
            r, g, b, _ = TOOTH_COLORS.get(fdi, (200, 200, 200, 180))
            _fdi_tag(draw, td, fdi, tag_font, (r, g, b))
        return img.convert("RGB")

    # ── Sagittal slices ───────────────────────────────────────────────────────
    def sagittal_slice(x_pos):
        xp  = int(np.clip(x_pos, 0, mask.shape[0]-1))
        vc  = volume[xp, y_lo_sag:y_hi_sag+1, z_lo:z_hi+1].astype(np.float32)
        mc  = mask  [xp, y_lo_sag:y_hi_sag+1, z_lo:z_hi+1]
        u8  = _apply_window(vc, lo_val, hi_val)
        disp_u8 = u8.T[::-1, ::-1] if a_high_y else u8.T[::-1, :]
        arr = np.stack([disp_u8] * 3, axis=-1)
        img  = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        sd = sag_xform(mc == sinus_id)
        if sd.any():
            _draw_cross_marker(draw, sd)
        # Teeth: FDI tag only, no outline. Ipsilateral maxillary quadrant only.
        for fdi in tag_tids:
            td = sag_xform(mc == fdi)
            if not td.any():
                continue
            r, g, b, _ = TOOTH_COLORS.get(fdi, (200, 200, 200, 180))
            _fdi_tag(draw, td, fdi, tag_font, (r, g, b))
        return img.convert("RGB")

    cor_imgs = [coronal_slice(y)  for y in y_positions]
    sag_imgs = [sagittal_slice(x) for x in x_positions]

    LABEL_W = 80
    ARROW_H = 40   # strip at TOP of each row
    LR_H    = 30   # strip at BOTTOM of each row for orientation markers
    DIV     = 2

    tile_w  = (out_w - LABEL_W - (n_slices-1)*DIV) // n_slices
    cor_s   = [_scale_to_width(_autocrop(img), tile_w) for img in cor_imgs]
    sag_s   = [_scale_to_width(_autocrop(img), tile_w) for img in sag_imgs]
    cor_h   = cor_s[0].size[1]
    sag_h   = sag_s[0].size[1]
    row_cor = ARROW_H + cor_h + LR_H
    row_sag = ARROW_H + sag_h + LR_H
    total_h = row_cor + DIV + row_sag

    canvas = Image.new("RGB", (out_w, total_h), (10, 10, 10))
    draw   = ImageDraw.Draw(canvas)

    # Paste tiles below their arrow strips
    for i, img in enumerate(cor_s):
        canvas.paste(img, (LABEL_W + i*(tile_w+DIV), ARROW_H))
    y_sag = row_cor + DIV
    for i, img in enumerate(sag_s):
        canvas.paste(img, (LABEL_W + i*(tile_w+DIV), y_sag + ARROW_H))

    draw.line([(0, row_cor), (out_w, row_cor)], fill=(60,60,60), width=DIV)

    # Arrows at TOP of each row
    _draw_arrow(draw, LABEL_W, ARROW_H//2,          out_w, "posterior", "anterior")
    _draw_arrow(draw, LABEL_W, y_sag + ARROW_H//2,  out_w, "midline",   "distal")

    _draw_rotated_label(canvas, "coronal",  LABEL_W//2, row_cor//2)
    _draw_rotated_label(canvas, "sagittal", LABEL_W//2, y_sag + row_sag//2)

    title = f"{'Right' if side == 'right' else 'Left'} Maxillary Sinus"
    font  = _get_font(14)
    tw    = len(title) * 9
    draw.rectangle([4, 4, tw+12, 22], fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(220, 220, 80), font=font)

    # ── Orientation markers — deterministic because transforms enforce convention ─
    # Coronal:  patient-right always on image-LEFT  → "R" left, "L" right
    # Sagittal: anterior     always on image-LEFT   → "A" left, "P" right
    lr_font  = _get_font(max(16, out_w // 55))
    lr_color = (255, 255, 255)

    def _place_markers(row_y, row_tile_h, left_lbl, right_lbl):
        by = row_y + ARROW_H + row_tile_h + LR_H // 2 -17
        if left_lbl:
            draw.text((LABEL_W + 8, by), left_lbl,  fill=lr_color, font=lr_font)
        if right_lbl:
            draw.text((out_w - 28,  by), right_lbl, fill=lr_color, font=lr_font)

    _place_markers(0,     cor_h, "", "L")   # coronal: L only on right
    _place_markers(y_sag, sag_h, "", "P")   # sagittal: P only on right

    return canvas


# ── Caption ────────────────────────────────────────────────────────────────
#
# Per schema.json (v6.1) these images are the images_needed targets for
# maxilla_sinus_right (sinus_right_detail) and maxilla_sinus_left
# (sinus_left_detail). Those are the
# facts the VLM has to read off the image, and none of the six case-level
# fields that feed the panoramic / 3D captions (teeth_present, teeth_absent,
# crowns, implants, bridge, ian_close_teeth) map here -- so unlike those two
# scripts these captions are static, with no facts.json input at all. They
# describe the picture: which slice family each row holds, which way the
# slices step, which way each axis points, and what the overlays mean.
#
# Written at generation time and dropped next to the PNGs as
# {case_id}_sinus_captions.json, the sidecar build_vqa_pairs.py already
# looks for.

_SINUS_SHARED_TAIL = (
    "Tiles are cropped individually, so their heights differ and the black "
    "gaps around them are padding, not anatomy."
)

SINUS_DETAIL_CAPTION = (
    "CBCT detail view of the {side} maxillary sinus (title '{title}', "
    "yellow, top-left). Two rows of grayscale slices; in every tile a bright "
    "green cross-marker ('+', arms pointing up/down/left/right with a gap at "
    "its centre) is centred on the {side} maxillary sinus. The marker points "
    "at the air space to judge -- it does not trace its border, so the sinus "
    "is the dark cavity surrounding the cross. "

    "Top row 'coronal': slices stepping posterior to anterior, cropped to "
    "the {side} half of the volume; superior is up, the patient's right is "
    "on the image's left, white 'L' marks the patient's left. The sinus sits "
    "above the posterior upper teeth. The other large dark space beside it, "
    "toward the midline edge of the tile, is the NASAL CAVITY -- it carries "
    "no cross-marker, so it is not the sinus, ignore it. "

    "Bottom row 'sagittal': slices stepping midline to distal; superior is "
    "up, anterior on the image's left, white 'P' marks posterior. Shows the "
    "sinus floor above the root apices. "

    "Judge only the cavity the cross-marker sits in (aeration, mucosal "
    "thickening, fluid, opacification, floor); anything else is context. "
    "Teeth are not marked -- each is tagged with its FDI number "
    "in its own color. Only upper teeth on the {side} ({tag_range}) are "
    "tagged; any other tooth visible in the crop is unlabelled context and "
    "cannot be the answer to a question about this sinus. " + _SINUS_SHARED_TAIL
)


def build_sinus_caption(side):
    """Static caption for the right/left maxillary sinus detail image."""
    title = f"{'Right' if side == 'right' else 'Left'} Maxillary Sinus"
    tag_range = "FDI 18-11" if side == "right" else "FDI 28-21"
    return SINUS_DETAIL_CAPTION.format(side=side, title=title, tag_range=tag_range)


# ── Case-level entry point ────────────────────────────────────────────────────

def process_case(volume_path, mask_path, out_dir, case_id, n_slices=3, out_w=1344,
                 window_center=None, window_width=None):
    import nibabel as nib
    import nibabel.orientations as nibo

    print(f"[{case_id}] loading ...", file=sys.stderr)
    vol_nib  = nib.load(str(volume_path))
    mask_nib = nib.load(str(mask_path))
    ornt = nibo.io_orientation(vol_nib.affine)
    ras  = nibo.axcodes2ornt(('R', 'A', 'S'))
    xfm  = nibo.ornt_transform(ornt, ras)
    print(f"  orientation: {nibo.ornt2axcodes(ornt)} → RAS+", file=sys.stderr)
    vol_nib  = vol_nib.as_reoriented(xfm)
    mask_nib = mask_nib.as_reoriented(xfm)
    volume   = vol_nib.get_fdata().astype(np.float32)
    mask     = mask_nib.get_fdata().astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    captions = {}
    for sinus_id, side in [(SINUS_RIGHT, "right"), (SINUS_LEFT, "left")]:
        n_vox = int((mask == sinus_id).sum())
        if n_vox < 10:
            print(f"  [skip] {side} sinus: {n_vox} voxels", file=sys.stderr)
            continue
        img = make_sinus_composite(volume, mask, sinus_id,
                                   n_slices=n_slices, out_w=out_w,
                                   window_center=window_center,
                                   window_width=window_width)
        if img is None:
            continue
        out_path = os.path.join(out_dir, f"{case_id}_sinus_{side}_detail.png")
        img.save(out_path)
        print(f"  → {out_path}", file=sys.stderr)
        saved.append(out_path)
        captions[f"sinus_{side}_detail"] = build_sinus_caption(side)

    # Caption sidecar, keyed by schema.json's images_needed names. Only
    # images that were actually written get an entry -- a skipped sinus
    # (too few voxels) must not leave a caption pointing at nothing.
    if captions:
        cap_path = os.path.join(out_dir, f"{case_id}_sinus_captions.json")
        with open(cap_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"  captions → {cap_path}", file=sys.stderr)
        saved.append(cap_path)

    return saved


def _case_id_from_path(path):
    stem = Path(path).name
    for s in (".nii.gz", ".nii"):
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem.replace("ToothFairy", "").replace("_0000", "")


def _find_volume(mask_path, volumes_dir):
    import glob
    stem = _case_id_from_path(mask_path)
    for pat in [f"*{stem}*_0000.nii.gz", f"*{stem}*.nii.gz"]:
        hits = sorted(glob.glob(os.path.join(volumes_dir, pat)))
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    import argparse, glob

    ap = argparse.ArgumentParser(
        description="CBCT sinus detail: coronal + sagittal multi-slice view.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--volume",      help="Single CBCT volume .nii.gz")
    grp.add_argument("--volumes-dir", help="Directory of volumes (batch)")
    ap.add_argument("--mask",      help="Mask .nii.gz (required with --volume)")
    ap.add_argument("--masks-dir", help="Mask directory (required with --volumes-dir)")
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--case-id",   default=None)
    ap.add_argument("--n-slices",  type=int, default=3)
    ap.add_argument("--out-w",     type=int, default=1344)
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--window-center", type=float, default=None, metavar="HU",
                    help="Grey-level window centre in HU. Requires "
                         "--window-width. Omit both to keep the per-volume "
                         "0.5/99.5 percentile window (default). Try "
                         "--window-center 400 --window-width 2000 for a "
                         "soft-tissue window.")
    ap.add_argument("--window-width",  type=float, default=None, metavar="HU",
                    help="Grey-level window width in HU. Requires "
                         "--window-center.")
    args = ap.parse_args()

    if (args.window_center is None) != (args.window_width is None):
        ap.error("--window-center and --window-width must be given together")

    if args.volume:
        if not args.mask:
            ap.error("--mask is required with --volume")
        cid = args.case_id or _case_id_from_path(args.mask)
        process_case(args.volume, args.mask, args.out_dir, cid,
                     args.n_slices, args.out_w,
                     window_center=args.window_center,
                     window_width=args.window_width)
    else:
        if not args.masks_dir:
            ap.error("--masks-dir is required with --volumes-dir")
        masks = sorted(glob.glob(os.path.join(args.masks_dir, "*.nii.gz")))
        if args.limit:
            masks = masks[:args.limit]
        for mp in masks:
            cid = _case_id_from_path(mp)
            vp  = _find_volume(mp, args.volumes_dir)
            if vp is None:
                print(f"  [skip] {cid}: no volume found", file=sys.stderr)
                continue
            process_case(vp, mp, args.out_dir, cid, args.n_slices, args.out_w,
                         window_center=args.window_center,
                         window_width=args.window_width)
    print("Done.", file=sys.stderr)