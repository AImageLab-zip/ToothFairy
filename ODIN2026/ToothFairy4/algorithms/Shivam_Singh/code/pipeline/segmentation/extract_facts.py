#!/usr/bin/env python3
"""
Extract structured clinical facts from ToothFairy FDI masks -- and reconcile a
facts file against the mask it came with.  (v4)

Two modes, one file:

  EXTRACT   -i <masks-dir> -o <out-dir>
            mask -> facts.json + phrases, for a case with no upstream facts.

  AUDIT     --facts-dir <dir> --masks-dir <dir> [--execute]
            an EXISTING facts file, corrected against its mask. Dry run by
            default. This was audit_facts.py until 2026-08-23; the merge is
            the one code/pipeline/README.md asked for ("run the two
            corrections at fact-extraction time"), and both modes now apply
            the same rules through the same code.

v4 changes:
  - audit_facts.py folded in. EXTRACT now emits `fov.maxilla: "excluded"` and
    `bridge_arches`, the two fields the renderer and the source rules need and
    that no earlier version of this script wrote -- so extracted facts are
    audited facts, and nothing downstream has to run a second pass to make
    them usable. `--no-audit` restores the v3 output exactly (that is what the
    upstream 622-case facts pool was built with, so use it to reproduce it).
  - The mask is read ONCE per case in both modes. The audit used to read it
    twice -- once canonical for the volumes, once raw for the bridge label.

v3 changes:
  - Orientation is derived from ANATOMY, not the header: the maxilla sits above
    the mandible, so their centroids define the superior-inferior axis and sign.
    This is robust to identity/flipped/honest headers alike (no DICOMOrient).
  - FOV (maxilla partial / condyles excluded) tests the anatomically-correct
    superior boundary.
  - When the maxilla is partially imaged, per-tooth UPPER absences are suppressed
    (a tooth outside the FOV is not "absent", just un-imaged) -> big precision win.
    Toggle with --keep-upper-absences.

Label scheme: teeth 11-48, mandible 1, maxilla 2, IAC 3/4, sinus 5/6,
bridge 8, crown 9, implant 10, pulp 50, incisive/lingual 51/52/53.

WHY THE AUDIT EXISTS
--------------------
A facts file extracted from the radiology REPORT describes the patient; the
mask describes the acquisition, and the two disagree in one systematic,
damaging way: when the maxilla was never in the scan volume, the report still
enumerates the upper teeth (usually as absent, because they are absent from
the picture the radiologist had), and the extraction records that verbatim.
Nothing downstream can tell that apart from a genuinely edentulous maxilla.

S0002 is the worked example. Its mask holds 252 mm^3 of maxilla -- 0.5% of the
segmentation, and nothing at all in the render -- and no maxillary tooth label
whatsoever. Its facts said:

    "fov": {"maxilla": "partial"}
    "teeth_absent": [11..28, 35, 37, 38, 46, 47, 48]

which produced the caption "maxilla (tan) ... Acquisition FOV: maxilla
partially included ... No maxillary tooth is present -- the maxilla is fully
edentulous." Every clause about the maxilla is false, and the last one is a
clinical claim about anatomy nobody imaged.

BONE and TEETH are judged separately, because the mask routinely contains one
without the other -- P045 has zero maxillary bone and all sixteen upper tooth
labels, and its report says exactly that: "Mandibular CT including within the
acquisition volume the mandibular bodies ... and partially the crowns of the
maxillary arch teeth."

  A. Maxillary BONE below MIN_ARCH_BONE_MM3 (the renderer's own threshold)
     -> fov.maxilla := "excluded"   (FOV_PHRASES renders that as the sentence
        "maxilla not included", and it is what schema v6.6's two-value
        maxilla_included answers from: no tan bone, whatever teeth are there)

  B. Bone below threshold AND no maxillary tooth label in the mask either
     -> every maxillary FDI (11-28, plus the 51-65 primary codes) removed from
        teeth_present / teeth_absent / crowns / implants / ian_close_teeth,
        so only lower teeth are mentioned

Why B is gated on the teeth and not on the bone alone, measured over all 622
cases -- "does the reference report describe any upper tooth?":

    maxillary bone   upper teeth      n     reports describing upper teeth
      in mask          in mask
    ---------------------------------------------------------------------
        yes              no            9        0   ( 0.0%)
        yes              yes         259      143   (55.2%)
        NO               no           98        1   ( 1.0%)
        NO               yes         256       33   (12.9%)

With no upper teeth in the mask the radiologists say nothing about upper
teeth 99 times out of 100 -- that is the group where stripping the facts
matches the reference. With upper teeth in the mask but no bone, one report
in eight still describes them, always hedged ("as far as can be visualized",
"only the dental elements are partially visualized"), so stripping those
would delete content the reference actually contains.

In AUDIT mode anything changed is reported line by line, and the original
value is kept in an `audit` block inside the file so a correction is never
silent and can always be undone. AUDIT IS DRY RUN BY DEFAULT -- pass
--execute to write.

Usage:
    # derive facts from masks (audited, unless --no-audit)
    python code/pipeline/segmentation/extract_facts.py -i dataset/validate/masks \
                               -o outputs/facts_build

    # see what the audit would change across a split
    python code/pipeline/segmentation/extract_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks

    # apply it
    python code/pipeline/segmentation/extract_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks --execute

    # one case, verbose
    python code/pipeline/segmentation/extract_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks \
                               --cases S0002 --verbose
"""
import os, json, argparse, glob, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

# SimpleITK preferred, nibabel accepted -- and the fallback is not a nicety.
# This is the ONLY module in the codebase that imports SimpleITK; every other
# generator reads masks with nibabel, so a container built for the pipeline is
# not guaranteed to carry it, and infer.py now calls this script
# on the critical path of a 15-minute budget. A missing import there is a
# failed submission, not a slow one.
#
# THE TWO READERS MUST AGREE ON AXIS ORDER, and they do not by default:
# SimpleITK's GetArrayFromImage is (z, y, x) while GetSpacing() is (x, y, z),
# and extract() below relies on exactly that pairing (see `samp`). nibabel
# gives (i, j, k) for both, so the array is transposed to match rather than the
# spacing being re-ordered -- one convention, chosen by the caller that already
# depends on it.
try:
    import SimpleITK as sitk
except ImportError:                                          # pragma: no cover
    sitk = None
    import nibabel as nib


def read_mask(path):
    """(array in z,y,x order, spacing in x,y,z order) -- either backend."""
    if sitk is not None:
        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img).astype(np.int16), img.GetSpacing()
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.int16).T      # (i,j,k) -> (k,j,i)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, zooms

MANDIBLE, MAXILLA = 1, 2
IAC = {"left": 3, "right": 4}
BRIDGE, CROWN, IMPLANT = 8, 9, 10
FDI_TEETH = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]
UPPER = [t for t in FDI_TEETH if t < 30]
LOWER = [t for t in FDI_TEETH if t >= 30]
POSTERIOR_LOWER = [t for t in LOWER if t % 10 >= 6]

# spatial order of teeth along each arch (right molars -> across midline -> left molars)
ARCH_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
ARCH_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

TOOTH_MIN_MM3 = 50.0
RESTO_MIN_MM3 = 20.0
IAN_NEAR_MM = 2.0
SUP_MARGIN = 2
EMIT_NEGATIVES = True
KEEP_UPPER_ABSENCES = True     # emitting true absences helps RadFact; suppression hurt


# ── The renderer's visibility threshold ──────────────────────────────────────
#
# MIN_ARCH_BONE_MM3 is IMPORTED rather than restated, so an audited facts file
# and the caption drawn from the same mask can never disagree: they are the
# same number by construction. The import is guarded because this module is the
# segmentation side's -- it needs only numpy/scipy and one mask reader, while
# create_3d_renders.py hard-requires skimage and PIL, which a segmentation
# environment has no reason to carry. The fallback below says so on stderr the
# first time it is used, because a threshold that has quietly drifted from the
# renderer's is exactly the failure the import exists to prevent.
_FALLBACK_MIN_ARCH_BONE_MM3 = 500.0     # keep in step with create_3d_renders.py
try:
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(next(
        p for p in _pathlib.Path(__file__).resolve().parents
        if (p / "_repo.py").is_file())))
    from _repo import add_code_paths      # noqa: E402
    add_code_paths()
    from create_3d_renders import MIN_ARCH_BONE_MM3   # noqa: E402
except (Exception, SystemExit):                              # pragma: no cover
    MIN_ARCH_BONE_MM3 = None

_warned_threshold = False


def arch_bone_min_mm3():
    """The bone-visibility threshold, from the renderer where it can be read."""
    global _warned_threshold
    if MIN_ARCH_BONE_MM3 is not None:
        return MIN_ARCH_BONE_MM3
    if not _warned_threshold:                                # pragma: no cover
        _warned_threshold = True
        print(f"[WARN] create_3d_renders is not importable here; using the "
              f"mirrored MIN_ARCH_BONE_MM3={_FALLBACK_MIN_ARCH_BONE_MM3:.0f} "
              f"mm^3. If the renderer's value has changed, the FOV verdict "
              f"and the caption now disagree.", file=sys.stderr)
    return _FALLBACK_MIN_ARCH_BONE_MM3


# ── The maxilla-exclusion rule ───────────────────────────────────────────────
#
# TWO SETS, AND THEY ARE NOT INTERCHANGEABLE.
#
# MAXILLA_FDIS is FDI notation, for filtering the FACTS: permanent upper teeth
# plus the primary upper codes, so a mixed-dentition case cannot slip an upper
# deciduous tooth past the filter.
#
# MAXILLA_TOOTH_LABELS is what may be looked up in the MASK, and it stops at
# the permanent range on purpose. The segmentation's label map is only FDI for
# 11-48; above that the ids are other anatomy entirely, and 51/52/53 are
# present in a great many masks -- S0002's carries 1, 2, 3, 4, 7, 9, 10, the
# lower teeth, and 50, 51, 52, 53. Reading the mask with the FDI set counts
# those as "upper primary teeth" and every bone-excluded case then looks like
# it still has upper teeth (354 of 354 instead of the true 256), which
# silently disables the strip in rule B below.
MAXILLA_TOOTH_LABELS = frozenset(UPPER)
MAXILLA_FDIS = MAXILLA_TOOTH_LABELS \
             | frozenset(range(51, 56)) | frozenset(range(61, 66))

# Fields that are flat lists of FDI numbers.
_FDI_LIST_FIELDS = ("teeth_present", "teeth_absent", "crowns", "implants",
                    "ian_close_teeth")

# Fields that are lists of objects carrying an FDI under one of these keys.
_FDI_OBJECT_KEYS = ("fdi", "fdi_number", "tooth", "tooth_number")

# What fov.maxilla is set to. "excluded" is create_3d_renders.FOV_PHRASES'
# own key for the sentence "maxilla not included"; that dict also accepts
# "not included"/"not_included" as aliases, so a hand-edited file still works.
FOV_EXCLUDED = "excluded"

# ── Bridge spans, derived from the mask ─────────────────────────────────────
#
# THE ONE THING THE AUDIT ADDS RATHER THAN REMOVES, which is why in AUDIT mode
# it is behind its own flag. Everything else there deletes a claim the mask
# cannot support; this writes a claim the mask alone can make, and a facts file
# extracted from a report has no counterpart to check it against --
# `bridge_present` is a bare bool with no span in it.
#
# Label 8 is Bridge in the ToothFairy label map (schema/tf3_*_label_map.json),
# and it marks the PONTIC -- the false tooth spanning the gap -- not the
# abutments carrying it. So the component's own extent is the middle of the
# bridge and never its ends.
#
# WHY THIS REPORTS AN ARCH AND NOT A SPAN. Closing the span means finding the
# abutments by dilation, and measured against the reference reports on the six
# validate cases that carry the label, that lands within about one position and
# not reliably on it: F067's pontics touch 14 and 16 where the report says
# "from 1.4 to 1.7"; P014's touch 32 and 42 where the report says "from 3.3 to
# 4.3". Absorbing the abutment CROWN label first (9) resolves two more cases
# and then overshoots on F014. The error is +/-1 position with no consistent
# sign, so an exact span would be wrong more often than right, while the ARCH
# the bridge sits in is never in doubt. State what the mask settles.
#
# Jawbone labels, for a pontic that touches no tooth at all -- the
# implant-supported case, where every neighbour is label 10.
JAWBONE_ARCH = {MANDIBLE: "mandible", MAXILLA: "maxilla"}
ARCH_FDIS = {"maxilla": frozenset(UPPER), "mandible": frozenset(LOWER)}

# How far to grow a pontic before asking what it touches. A pontic sits against
# its abutments with at most a cement gap between them, so this only has to
# cross a voxel or two.
BRIDGE_DILATION = 3

# Components below this are noise -- stray voxels of bridge label, not a unit.
MIN_BRIDGE_VOXELS = 50


def largest_cc_size(mask):
    """Voxel count of the biggest connected component.

    np.bincount replaces ndimage.sum(np.ones_like(lab), ...): same counts, but
    without allocating a full-volume array per call. This runs once per tooth
    (32x per case), so the allocation dominated fact extraction on slow CPUs.
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0                      # background
    return int(counts.max())


def centroid(mask):
    idx = np.argwhere(mask)
    return idx.mean(0) if len(idx) else None


def nearest_tooth(c, tc):
    if c is None or not tc:
        return None
    return min(tc, key=lambda t: np.linalg.norm(tc[t] - c))


def _estimate_slot_centroids(tc, arch):
    """Estimate physical centroids of ABSENT slots in an arch by interpolating /
    extrapolating from present neighbours along the arch order."""
    present = [(i, t) for i, t in enumerate(arch) if t in tc]
    if len(present) < 2:
        return {}
    est = {}
    for i, t in enumerate(arch):
        if t in tc:
            continue
        left = [(pi, pt) for pi, pt in present if pi < i]
        right = [(pi, pt) for pi, pt in present if pi > i]
        if left and right:
            (li, lt), (ri, rt) = left[-1], right[0]
            f = (i - li) / (ri - li)
            est[t] = tc[lt] * (1 - f) + tc[rt] * f
        elif len(left) >= 2:
            (l2i, l2t), (l1i, l1t) = left[-2], left[-1]
            step = (tc[l1t] - tc[l2t]) / max(l1i - l2i, 1)
            est[t] = tc[l1t] + step * (i - l1i)
        elif len(right) >= 2:
            (r1i, r1t), (r2i, r2t) = right[0], right[1]
            step = (tc[r2t] - tc[r1t]) / max(r2i - r1i, 1)
            est[t] = tc[r1t] + step * (i - r1i)
        elif left:
            est[t] = tc[left[-1][1]]
        elif right:
            est[t] = tc[right[0][1]]
    return est


def nearest_absent_slot(c, tc, absent):
    """Assign an implant to the ABSENT tooth slot physically closest to its
    centroid, using slot positions interpolated from present teeth. Geometric —
    respects left/right and does not cross the midline via FDI arithmetic."""
    if c is None or not tc:
        return None
    est = {}
    est.update(_estimate_slot_centroids(tc, ARCH_UPPER))
    est.update(_estimate_slot_centroids(tc, ARCH_LOWER))
    cand = {t: p for t, p in est.items() if t in absent}
    if not cand:                              # fall back to nearest present tooth's quadrant
        ref = nearest_tooth(c, tc)
        return ref
    return min(cand, key=lambda t: np.linalg.norm(cand[t] - c))


def superior_axis_end(mask):
    """(axis, end): array axis along S-I, and +1 if higher index = superior.
    From maxilla-vs-mandible centroids -> header independent."""
    ci = centroid(mask == MANDIBLE)
    cs = centroid(mask == MAXILLA)
    if ci is None or cs is None:
        return 0, +1
    diff = cs - ci                    # mandible -> maxilla points superior
    axis = int(np.argmax(np.abs(diff)))
    return axis, (+1 if diff[axis] > 0 else -1)


def touches_superior(m, axis, end, margin=SUP_MARGIN):
    idx = [slice(None)] * m.ndim
    idx[axis] = slice(m.shape[axis] - margin, None) if end > 0 else slice(0, margin)
    return bool(m[tuple(idx)].any())


# ── Mask measurement, shared by both modes ───────────────────────────────────

def label_volumes(mask, spacing):
    """
    {label_id: volume in mm^3} for every non-zero label. The same arithmetic as
    create_3d_renders.mask_label_volumes -- a voxel count times the voxel
    volume, and orientation-independent, so it does not matter that the audit
    reads the mask in (z,y,x) and the renderer in (i,j,k).
    """
    ids, counts = np.unique(mask, return_counts=True)
    scale = float(np.prod(spacing[:3]))
    return {int(i): float(c) * scale for i, c in zip(ids, counts) if int(i) != 0}


def measure_maxilla(mask, spacing):
    """(maxillary bone volume in mm^3, number of maxillary tooth LABELS)."""
    volumes = label_volumes(mask, spacing)
    n_teeth = sum(1 for fdi in MAXILLA_TOOTH_LABELS if fdi in volumes)
    return volumes.get(MAXILLA, 0.0), n_teeth


def derive_bridge_arches(mask):
    """
    Which ARCH each bridge in the mask sits in.

    -> [{arch, voxels, adjacent_teeth, implant_supported}], one entry per
    connected component of the bridge label, `arch` None only when nothing
    around the pontic identifies one.

    The arch is read off whatever the dilated pontic touches: a tooth label
    settles it outright, and where an implant-supported bridge touches no tooth
    the jawbone label under it does (1 lower, 2 upper). No span is computed --
    see the note above BRIDGE_DILATION for the measurement behind that.
    """
    labelled, n = ndimage.label(mask == BRIDGE)
    out = []
    for i in range(1, n + 1):
        component = labelled == i
        n_vox = int(component.sum())
        if n_vox < MIN_BRIDGE_VOXELS:
            continue
        grown = ndimage.binary_dilation(component, iterations=BRIDGE_DILATION)
        touched = {int(v) for v in set(mask[grown].flat)}
        teeth = sorted(f for f in touched if 11 <= f <= 48)

        arches = {arch for arch, fdis in ARCH_FDIS.items()
                  if any(f in fdis for f in teeth)}
        if not arches:                    # implant-supported: fall back to bone
            arches = {JAWBONE_ARCH[b] for b in touched if b in JAWBONE_ARCH}
        if not arches:
            # Still nothing: an implant-borne bridge standing clear of both the
            # teeth and the bone, so no label is within reach at all (F014,
            # S0000). Fall back to geometry -- whichever jawbone's centroid the
            # pontic is nearer to. Crude, and it cannot be ambiguous: the two
            # jaws are the two things it could be sitting on.
            near = None
            best = None
            for bone_id, arch in JAWBONE_ARCH.items():
                where = np.argwhere(mask == bone_id)
                if not len(where):
                    continue
                d = np.linalg.norm(np.argwhere(component).mean(0)
                                   - where.mean(0))
                if best is None or d < best:
                    best, near = d, arch
            if near:
                arches = {near}
        out.append({
            "arch": arches.pop() if len(arches) == 1 else None,
            "voxels": n_vox,
            "adjacent_teeth": teeth,
            "implant_supported": IMPLANT in touched,
        })
    return out


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _fdi_of(item):
    """The FDI an object-shaped entry refers to, or None."""
    if _is_int(item):
        return item
    if isinstance(item, dict):
        for key in _FDI_OBJECT_KEYS:
            if _is_int(item.get(key)):
                return item[key]
    return None


def _strip_maxilla(value):
    """
    Drop every maxillary entry from a list-valued field. Returns
    (new_value, removed) with removed the FDI numbers taken out. Non-list
    values and entries with no readable FDI are left alone -- this is a
    filter, not a validator.
    """
    if not isinstance(value, list):
        return value, []
    kept, removed = [], []
    for item in value:
        fdi = _fdi_of(item)
        if fdi is not None and fdi in MAXILLA_FDIS:
            removed.append(fdi)
        else:
            kept.append(item)
    return kept, removed


def apply_maxilla_rules(structured, bone_mm3, n_upper_teeth=0):
    """
    Rules A and B, applied IN PLACE to one `structured` facts block.

    -> (changes, original): `changes` human-readable strings, empty when the
    mask supports what the block already says; `original` the pre-edit values,
    keyed by field, for the caller to record.

    This is the one implementation of the rules. extract() calls it on facts it
    has just derived, audit_case() calls it on facts someone else wrote, and
    neither can drift from the other.

    A case whose maxilla IS in the volume is left untouched, and these rules
    only ever REMOVE claims the mask cannot support, never add one -- the mask
    says nothing about whether a tooth the report never mentioned exists.
    """
    changes, original = [], {}
    if bone_mm3 >= arch_bone_min_mm3():
        return changes, original

    fov = structured.get("fov")
    fov = fov if isinstance(fov, dict) else {}
    before = fov.get("maxilla")
    if before != FOV_EXCLUDED:
        original["fov.maxilla"] = before
        fov["maxilla"] = FOV_EXCLUDED
        structured["fov"] = fov
        changes.append(f"fov.maxilla: {before!r} -> {FOV_EXCLUDED!r}")

    # THE TIGHTER RULE. Upper teeth that ARE in the mask are in the picture,
    # and one reference report in eight describes them even with no maxillary
    # bone in the volume ("as far as can be visualized..."). Stripping them
    # would delete content the reference contains, so the teeth only go when
    # the mask has none of them either.
    if n_upper_teeth == 0:
        for field in _FDI_LIST_FIELDS:
            if field not in structured:
                continue
            kept, removed = _strip_maxilla(structured[field])
            if removed:
                original[field] = structured[field]
                structured[field] = kept
                changes.append(f"{field}: dropped {len(removed)} maxillary "
                               f"FDI(s) {sorted(removed)}")

        # bridge_present is a bool about the whole mouth; with no maxilla in
        # the volume it can only be describing the mandible, so it is left
        # alone unless the facts scope it per arch.
        for field in ("bridge_present_maxilla", "crowns_maxilla",
                      "implants_maxilla"):
            if field in structured and structured[field] not in (None, False, [], {}):
                original[field] = structured[field]
                structured[field] = [] if isinstance(structured[field], list) else False
                changes.append(f"{field}: cleared (maxilla not in the volume)")

    return changes, original


def audit_case(facts, bone_mm3, n_upper_teeth=0):
    """
    Apply the maxilla-excluded corrections to one parsed facts RECORD
    (the {"case_id", "structured", "phrases"} shape on disk).

    Returns (new_facts, changes) where `changes` is a list of human-readable
    strings, empty when the mask supports what the facts already say. `facts`
    is not mutated.

    Two independent corrections, gated separately -- see this module's
    docstring for the measurement behind the split:

      bone below threshold           -> fov.maxilla := excluded
      ...AND no upper tooth label    -> maxillary FDIs stripped as well
    """
    threshold = arch_bone_min_mm3()
    if bone_mm3 >= threshold:
        return facts, []

    out = json.loads(json.dumps(facts))          # deep copy, JSON-only data
    structured = out.get("structured")
    if not isinstance(structured, dict):
        return facts, []

    changes, original = apply_maxilla_rules(structured, bone_mm3, n_upper_teeth)

    if changes:
        out.setdefault("audit", {})["maxilla_excluded"] = {
            "reason": (f"maxillary bone {bone_mm3:.0f} mm^3 in the mask, below "
                       f"the {threshold:.0f} mm^3 visibility threshold "
                       f"(create_3d_renders.MIN_ARCH_BONE_MM3)"),
            "maxilla_bone_mm3": round(bone_mm3, 1),
            "maxillary_tooth_labels_in_mask": n_upper_teeth,
            "teeth_stripped": n_upper_teeth == 0,
            "original": original,
        }
    return out, changes


def extract(mask, spacing_xyz, audit=True):
    """
    mask + spacing -> (structured facts, phrases).

    `audit` applies the maxilla rules and derives `bridge_arches` here, at
    extraction time, which is where code/pipeline/README.md asks for them: the
    renderer's captions and two of the source rules read exactly those two
    fields, and a v3 facts file has neither. Pass audit=False for byte-exact
    v3 output -- the upstream 622-case pool was built that way.
    """
    samp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    voxvol = float(np.prod(spacing_xyz))
    facts, phrases = {}, []

    # ---- dentition presence (phrases emitted later, after FOV is known) ----
    present, absent, tc = [], [], {}
    for t in FDI_TEETH:
        m = mask == t
        if m.any() and largest_cc_size(m) * voxvol >= TOOTH_MIN_MM3:
            present.append(t); tc[t] = centroid(m)
        else:
            absent.append(t)
    facts["teeth_present"], facts["teeth_absent"] = present, absent
    n_upper = sum(1 for t in UPPER if t in present)

    # ---- FOV from anatomy ----
    si_axis, si_end = superior_axis_end(mask)
    facts["fov"] = {}
    max_m, mand_m = mask == MAXILLA, mask == MANDIBLE
    maxilla_partial = bool(max_m.any() and
                           (touches_superior(max_m, si_axis, si_end) or n_upper < 4))
    condyles_excluded = bool(mand_m.any() and touches_superior(mand_m, si_axis, si_end))
    if maxilla_partial:
        facts["fov"]["maxilla"] = "partial"
    if condyles_excluded:
        facts["fov"]["condyles"] = "excluded"
    suppress_upper = maxilla_partial and not KEEP_UPPER_ABSENCES

    # ---- the maxilla-exclusion rules, before any phrase is written ----
    #
    # Rules A and B are statements about the ACQUISITION and they overrule the
    # FOV verdict above: "partial" is what the superior-boundary test says when
    # SOME maxilla is in the volume, and 252 mm^3 of it is not some. The
    # tooth-label count is read off the mask, not off `present` above, because
    # the 622-case measurement in this module's docstring counted labels --
    # a fragment under TOOTH_MIN_MM3 is still a tooth that was imaged.
    maxilla_excluded = False
    if audit:
        bone_mm3, n_upper_labels = measure_maxilla(mask, spacing_xyz)
        apply_maxilla_rules(facts, bone_mm3, n_upper_labels)
        present, absent = facts["teeth_present"], facts["teeth_absent"]
        maxilla_excluded = facts["fov"].get("maxilla") == FOV_EXCLUDED

    # ---- dentition phrases (gate upper absences when maxilla partial) ----
    for t in absent:
        if t < 30 and suppress_upper:
            continue                    # upper tooth may be un-imaged, not absent
        phrases.append(f"Absence of tooth {t}.")
    for q, name in [(1, "first"), (2, "second"), (3, "third"), (4, "fourth")]:
        if q in (1, 2) and suppress_upper:
            continue
        qt = [t for t in FDI_TEETH if t // 10 == q]
        miss = [t for t in qt if t in absent]
        if len(miss) == len(qt):
            phrases.append(f"Complete edentulism of the {name} quadrant.")
        elif len(miss) >= 5:
            phrases.append(f"Partial edentulism of the {name} quadrant.")

    # ---- canals ----
    facts["canals"] = {}
    for side, lid in IAC.items():
        if (mask == lid).any():
            facts["canals"][side] = "present"
            phrases.append(f"{side.capitalize()} mandibular canal with a regular course.")

    # ---- crowns (dedup) ----
    facts["crowns"] = []
    cm = mask == CROWN
    if cm.any():
        lab, n = ndimage.label(cm); seen = set()
        for i in range(1, n + 1):
            cc = lab == i
            if cc.sum() * voxvol < RESTO_MIN_MM3:
                continue
            tid = nearest_tooth(centroid(cc), tc)
            if tid and tid not in seen:
                seen.add(tid); facts["crowns"].append(tid)
                phrases.append(f"Prosthetic crown on tooth {tid}.")

    # ---- implants (merge fragments, dedup slots) ----
    facts["implants"] = []
    im = mask == IMPLANT
    if im.any():
        im = ndimage.binary_closing(im, iterations=2)
        lab, n = ndimage.label(im); seen = set()
        for i in range(1, n + 1):
            cc = lab == i
            if cc.sum() * voxvol < RESTO_MIN_MM3:
                continue
            pos = nearest_absent_slot(centroid(cc), tc, absent)
            key = pos if pos is not None else f"cc{i}"
            if key in seen:
                continue
            seen.add(key); facts["implants"].append(pos)
            phrases.append(f"Presence of an endosseous implant in position {pos}."
                           if pos else "Presence of an endosseous implant.")

    facts["bridge_present"] = bool((mask == BRIDGE).any()
                                   and (mask == BRIDGE).sum() * voxvol >= RESTO_MIN_MM3)
    if facts["bridge_present"]:
        phrases.append("A fixed prosthetic bridge is present.")

    # ALWAYS write bridge_arches, even empty. Absent means "never derived",
    # empty means "derived, no bridge label in this mask", and a consumer that
    # cannot tell those apart has to treat both as unknown -- which left
    # P397's false-positive bridge standing, because its facts file simply had
    # no key for the source rule to read. The per-component voxel floor is
    # deliberately not the mm^3 floor bridge_present uses above: one is asking
    # "is there a bridge", the other "is this component a unit or noise".
    if audit:
        bridges = derive_bridge_arches(mask)
        facts["bridge_arches"] = sorted({b["arch"] for b in bridges if b["arch"]})
        if bridges:
            facts["bridges_detail"] = bridges

    # ---- IAN proximity ----
    iac = np.isin(mask, list(IAC.values()))
    facts["ian_close_teeth"] = []
    if iac.any():
        # The EDT is only ever read at posterior lower teeth, so compute it on
        # the bounding box of (canals + those teeth) instead of the whole
        # volume. Distances inside the box are identical because the box
        # contains every source voxel (it is built from the canals).
        roi = iac.copy()
        for t in POSTERIOR_LOWER:
            if t in tc:
                roi |= (mask == t)
        idx = np.argwhere(roi)
        if idx.size:
            lo = idx.min(0)
            hi = idx.max(0) + 1
            pad = int(np.ceil(IAN_NEAR_MM / min(samp))) + 2
            lo = np.maximum(lo - pad, 0)
            hi = np.minimum(hi + pad, mask.shape)
            sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            sub_iac = iac[sl]
            sub_mask = mask[sl]
            edt = ndimage.distance_transform_edt(~sub_iac, sampling=samp)
            for t in POSTERIOR_LOWER:
                if t not in tc:
                    continue
                m = (sub_mask == t)
                if m.any() and edt[m].min() <= IAN_NEAR_MM:
                    facts["ian_close_teeth"].append(t)
                    phrases.append(f"Tooth {t} is in close proximity to the mandibular canal.")

    # ---- FOV phrases ----
    if maxilla_excluded:
        phrases.append("The maxilla is not included in the acquisition.")
    elif maxilla_partial:
        phrases.append("The maxilla is partially included in the acquisition.")
    if condyles_excluded:
        phrases.append("The mandibular condyles are excluded from the acquisition and not visible.")

    if EMIT_NEGATIVES:
        phrases.append("No definite osteolytic or osteosclerotic lesions.")

    facts["present_label_ids"] = sorted(np.unique(mask).tolist())
    facts["si_axis"], facts["si_end"] = si_axis, si_end     # for debugging
    return facts, phrases


# ── EXTRACT mode ─────────────────────────────────────────────────────────────

def run_extract(args):
    out = Path(args.output); (out / "facts").mkdir(parents=True, exist_ok=True)
    masks = sorted(glob.glob(os.path.join(args.input, "*.nii.gz")))
    audit = not args.no_audit
    print(f"extracting facts from {len(masks)} masks"
          f"{'' if audit else '  [--no-audit: v3 output]'}")
    with open(out / "facts_all.jsonl", "w") as jl:
        for mp in masks:
            arr, spacing = read_mask(mp)
            case_id = Path(mp).name.replace(".nii.gz", "")
            facts, phrases = extract(arr, spacing, audit=audit)
            rec = {"case_id": case_id, "structured": facts, "phrases": phrases}
            (out / "facts" / f"{case_id}.json").write_text(json.dumps(rec, indent=2))
            jl.write(json.dumps(rec) + "\n")
            print(f"  {case_id}: {len(facts['teeth_present'])} teeth, "
                  f"{len(phrases)} phrases, si_axis={facts['si_axis']}({facts['si_end']:+d}), "
                  f"fov={facts['fov']}"
                  + (f", bridge_arches={facts['bridge_arches']}" if audit else ""))
    print("=== facts done ===")


# ── AUDIT mode ───────────────────────────────────────────────────────────────

def run_audit(args):
    facts_dir, masks_dir = Path(args.facts_dir), Path(args.masks_dir)
    if not facts_dir.is_dir():
        sys.exit(f"[FAIL] facts dir not found: {facts_dir}")
    if not masks_dir.is_dir():
        sys.exit(f"[FAIL] masks dir not found: {masks_dir}")

    paths = sorted(facts_dir.glob("*.json"))
    if args.cases:
        wanted = set(args.cases)
        paths = [p for p in paths if p.stem in wanted]
        missing = sorted(wanted - {p.stem for p in paths})
        if missing:
            print(f"[WARN] no facts file for: {missing}")
    if not paths:
        sys.exit(f"[FAIL] no facts files found under {facts_dir}")

    threshold = arch_bone_min_mm3()
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"[INFO] Auditing {len(paths)} case(s) against {masks_dir}  [{mode}]")
    print(f"[INFO] Maxilla counts as imaged at >= {threshold:.0f} mm^3 "
          f"of bone in the mask\n")

    n_changed = n_clean = n_skipped = n_bone_only = n_maxilla = 0
    n_spanned = n_spans_written = n_spans_unresolved = 0
    for path in paths:
        case_id = path.stem
        mask_path = masks_dir / f"{case_id}.nii.gz"
        if not mask_path.exists():
            print(f"[{case_id}] [SKIP] no mask: {mask_path}")
            n_skipped += 1
            continue
        try:
            facts = json.loads(path.read_text())
            # ONE read, used for both the volumes and the bridge label. The
            # volumes are voxel counts, so no canonical reorientation is
            # needed, and neither is the arch derivation -- it reads label
            # adjacency, which no reorientation changes.
            arr, spacing = read_mask(mask_path)
            bone, n_teeth = measure_maxilla(arr, spacing)
        except Exception as exc:
            print(f"[{case_id}] [SKIP] {exc}")
            n_skipped += 1
            continue

        new_facts, changes = audit_case(facts, bone, n_teeth)
        # The maxilla audit's own verdict, kept separate from the bridge
        # derivation below: both write to `changes`, and counting them together
        # made every bridge-only edit report as a maxilla correction ("bone NOT
        # imaged") in the summary line.
        maxilla_corrected = bool(changes)

        if args.derive_bridge_arches:
            bridges = derive_bridge_arches(arr)
            # ALWAYS write the field, even empty -- see the note in extract().
            structured = new_facts.setdefault("structured", {})
            arches_found = sorted({b["arch"] for b in bridges if b["arch"]})
            unresolved = sum(1 for b in bridges if not b["arch"])
            if structured.get("bridge_arches") != arches_found:
                structured["bridge_arches"] = arches_found
                detail = (f" -- from {len(bridges)} bridge component(s)"
                          + (f", {unresolved} with no identifiable arch"
                             if unresolved else "")) if bridges else ""
                changes.append(f"bridge_arches: {arches_found or 'none'}{detail}")
            if bridges:
                n_spanned += 1
                structured["bridges_detail"] = bridges
                n_spans_written += len(arches_found)
                n_spans_unresolved += unresolved

        if not changes:
            n_clean += 1
            if args.verbose:
                verdict = "imaged" if bone >= threshold else "already audited"
                print(f"[{case_id}] maxilla {bone:8.0f} mm^3, {n_teeth:2} upper "
                      f"tooth label(s) -- {verdict}, no change")
            continue

        n_changed += 1
        if maxilla_corrected:
            n_maxilla += 1
            if n_teeth:
                n_bone_only += 1
            print(f"[{case_id}] maxilla {bone:8.0f} mm^3, {n_teeth:2} upper tooth "
                  f"label(s) -- bone NOT imaged"
                  + ("; teeth kept (they are in the mask)" if n_teeth else ""))
        else:
            print(f"[{case_id}] maxilla {bone:8.0f} mm^3 -- imaged; "
                  f"bridge derivation only")
        for c in changes:
            print(f"    {c}")
        if args.execute:
            path.write_text(json.dumps(new_facts, indent=2))
            print(f"    -> wrote {path}")

    print(f"\n[INFO] ========== Facts Audit ==========")
    print(f"[INFO] Cases changed{'' if args.execute else ' (would be)'}: {n_changed}")
    # Split by WHICH audit changed the file. The bridge derivation writes its
    # key on every case it sees, so counting it as a maxilla correction (the
    # arithmetic that was here) reported every clean case as "upper teeth
    # stripped -- not in the mask".
    print(f"[INFO]   ...maxilla corrected: {n_maxilla}")
    print(f"[INFO]      ...fov only, upper teeth kept (in the mask): {n_bone_only}")
    print(f"[INFO]      ...fov + upper teeth stripped (not in the mask): "
          f"{n_maxilla - n_bone_only}")
    print(f"[INFO]   ...bridge_arches only, maxilla imaged: {n_changed - n_maxilla}")
    print(f"[INFO] Cases already consistent: {n_clean}")
    if args.derive_bridge_arches:
        print(f"[INFO] Cases with a bridge label: {n_spanned}  "
              f"-> {n_spans_written} arch(es) identified, "
              f"{n_spans_unresolved} component(s) with no identifiable arch")
    if n_skipped:
        print(f"[INFO] Cases skipped: {n_skipped}")
    if n_changed and not args.execute:
        print("[INFO] DRY RUN -- nothing was written. Re-run with --execute.")
    if n_changed and args.execute:
        print("[INFO] Re-run the image generators (or rebuild_captions.py "
              "--masks-dir) so the captions pick this up.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # EXTRACT mode
    ap.add_argument("-i", "--input",
                    help="EXTRACT: dir of {case_id}.nii.gz masks")
    ap.add_argument("-o", "--output",
                    help="EXTRACT: dir to write facts/{case_id}.json + "
                         "facts_all.jsonl into")
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--suppress-upper-absences", action="store_true",
                    help="drop per-tooth upper absences when maxilla is partial "
                         "(off by default — suppression was found to hurt the score)")
    ap.add_argument("--no-audit", action="store_true",
                    help="EXTRACT: skip the maxilla rules and bridge_arches, "
                         "for byte-exact v3 output. The upstream 622-case facts "
                         "pool was built this way; anything the pipeline reads "
                         "wants the audited fields.")
    # AUDIT mode
    ap.add_argument("--facts-dir",
                    help="AUDIT: per-case facts JSONs ({case_id}.json). Giving "
                         "this selects audit mode.")
    ap.add_argument("--masks-dir",
                    help="AUDIT: per-case segmentation masks ({case_id}.nii.gz)")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="AUDIT: only audit these case IDs")
    ap.add_argument("--execute", action="store_true",
                    help="AUDIT: write the corrected facts files. Without this "
                         "the run is a DRY RUN and only reports.")
    ap.add_argument("--verbose", action="store_true",
                    help="AUDIT: print the measured maxilla volume for every "
                         "case, not just the ones that change")
    ap.add_argument("--derive-bridge-arches", action="store_true",
                    help="AUDIT: also DERIVE structured.bridge_arches from the "
                         "mask's bridge label (8) -- which arch each bridge "
                         "sits in. The only thing the audit adds rather than "
                         "removes, hence its own flag. EXTRACT always writes "
                         "it unless --no-audit.")
    args = ap.parse_args()

    global EMIT_NEGATIVES, KEEP_UPPER_ABSENCES
    EMIT_NEGATIVES = not args.no_negatives
    KEEP_UPPER_ABSENCES = not args.suppress_upper_absences

    auditing = bool(args.facts_dir or args.masks_dir)
    if auditing:
        if not (args.facts_dir and args.masks_dir):
            ap.error("audit mode needs both --facts-dir and --masks-dir")
        if args.input or args.output:
            ap.error("--facts-dir/--masks-dir (audit) and -i/-o (extract) are "
                     "different modes; pass one pair, not both")
        return run_audit(args)

    if not (args.input and args.output):
        ap.error("extract mode needs -i/--input and -o/--output "
                 "(or --facts-dir/--masks-dir to audit existing facts)")
    return run_extract(args)


if __name__ == "__main__":
    main()
