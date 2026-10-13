#!/usr/bin/env python3
"""
audit_facts.py

Check each case's facts JSON against its segmentation MASK, and correct the
claims the mask contradicts.

Why this exists
---------------
The facts files are extracted from the radiology REPORT. A report describes
the patient; the mask describes the acquisition, and the two disagree in one
systematic, damaging way: when the maxilla was never in the scan volume, the
report still enumerates the upper teeth (usually as absent, because they are
absent from the picture the radiologist had), and the extraction records that
verbatim. Nothing downstream can tell that apart from a genuinely edentulous
maxilla.

S0002 is the worked example. Its mask holds 252 mm^3 of maxilla -- 0.5% of the
segmentation, and nothing at all in the render -- and no maxillary tooth label
whatsoever. Its facts said:

    "fov": {"maxilla": "partial"}
    "teeth_absent": [11..28, 35, 37, 38, 46, 47, 48]

which produced the caption "maxilla (tan) ... Acquisition FOV: maxilla
partially included ... No maxillary tooth is present -- the maxilla is fully
edentulous." Every clause about the maxilla is false, and the last one is a
clinical claim about anatomy nobody imaged.

What it does
------------
For every case, measure the mask (create_3d_renders.mask_label_volumes, the
same function the renderer captions from, so this tool and the caption can
never disagree). BONE and TEETH are judged separately, because the mask
routinely contains one without the other -- P045 has zero maxillary bone and
all sixteen upper tooth labels, and its report says exactly that:
"Mandibular CT including within the acquisition volume the mandibular bodies
... and partially the crowns of the maxillary arch teeth."

  A. Maxillary BONE below create_3d_renders.MIN_ARCH_BONE_MM3
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

Anything it changes is reported line by line, and the original value is kept
in an `audit` block inside the file so a correction is never silent and can
always be undone.

DRY RUN BY DEFAULT -- pass --execute to write.

Usage:
    # see what would change across a split
    python code/pipeline/segmentation/audit_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks

    # apply it
    python code/pipeline/segmentation/audit_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks --execute

    # one case, verbose
    python code/pipeline/segmentation/audit_facts.py --facts-dir dataset/training/facts \
                               --masks-dir dataset/training/masks \
                               --cases S0002 --verbose
"""

import argparse
import json
import sys
from pathlib import Path


# The mask measurement and the threshold both come from the renderer, so an
# audited facts file and the caption drawn from the same mask always agree.

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

from create_3d_renders import (ARCH_BONE_ID, MIN_ARCH_BONE_MM3,  # noqa: E402
                               mask_label_volumes)

try:
    import nibabel as nib
except ImportError:                                          # pragma: no cover
    sys.exit("pip install nibabel")


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
MAXILLA_FDIS = frozenset(range(11, 19)) | frozenset(range(21, 29)) \
             | frozenset(range(51, 56)) | frozenset(range(61, 66))
MAXILLA_TOOTH_LABELS = frozenset(range(11, 19)) | frozenset(range(21, 29))

# Fields that are flat lists of FDI numbers.
_FDI_LIST_FIELDS = ("teeth_present", "teeth_absent", "crowns", "implants",
                    "ian_close_teeth")

# ── Bridge spans, derived from the mask ─────────────────────────────────────
#
# THE ONE THING THIS TOOL ADDS RATHER THAN REMOVES, which is why it is behind
# its own flag. Everything else here deletes a claim the mask cannot support;
# this writes a claim the mask alone can make, and the facts file has no
# counterpart to check it against -- `bridge_present` is a bare bool with no
# span in it.
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
BRIDGE_ID = 8
IMPLANT_ID = 10
# Jawbone labels, for a pontic that touches no tooth at all -- the
# implant-supported case, where every neighbour is label 10.
JAWBONE_ARCH = {1: "mandible", 2: "maxilla"}
ARCH_FDIS = {
    "maxilla":  frozenset(range(11, 19)) | frozenset(range(21, 29)),
    "mandible": frozenset(range(31, 39)) | frozenset(range(41, 49)),
}

# How far to grow a pontic before asking what it touches. A pontic sits against
# its abutments with at most a cement gap between them, so this only has to
# cross a voxel or two.
BRIDGE_DILATION = 3

# Components below this are noise -- stray voxels of bridge label, not a unit.
MIN_BRIDGE_VOXELS = 50

# Fields that are lists of objects carrying an FDI under one of these keys.
_FDI_OBJECT_KEYS = ("fdi", "fdi_number", "tooth", "tooth_number")

# What fov.maxilla is set to. "excluded" is create_3d_renders.FOV_PHRASES'
# own key for the sentence "maxilla not included"; that dict also accepts
# "not included"/"not_included" as aliases, so a hand-edited file still works.
FOV_EXCLUDED = "excluded"


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


def measure_maxilla(mask_path):
    """
    (bone volume in mm^3, number of maxillary tooth labels) for one mask,
    measured the way the renderer does.
    """
    nii = nib.as_closest_canonical(nib.load(str(mask_path)))
    volumes = mask_label_volumes(nii.get_fdata().astype("int32"),
                                 nii.header.get_zooms())
    n_teeth = sum(1 for fdi in MAXILLA_TOOTH_LABELS if fdi in volumes)
    return volumes.get(ARCH_BONE_ID["maxilla"], 0.0), n_teeth


def derive_bridge_arches(mask):
    """
    Which ARCH each bridge in the mask sits in.

    -> [{arch, voxels, adjacent_teeth, implant_supported}], one entry per
    connected component of the bridge label, `arch` None only when nothing
    around the pontic identifies one.

    The arch is read off whatever the dilated pontic touches: a tooth label
    settles it outright, and where an implant-supported bridge touches no tooth
    the jawbone label under it does (1 lower, 2 upper). No span is computed --
    see the note above BRIDGE_ID for the measurement behind that.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError:                                      # pragma: no cover
        return []

    labelled, n = ndimage.label(mask == BRIDGE_ID)
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
            "implant_supported": IMPLANT_ID in touched,
        })
    return out


def audit_case(facts, bone_mm3, n_upper_teeth=0):
    """
    Apply the maxilla-excluded corrections to one parsed facts dict.

    Returns (new_facts, changes) where `changes` is a list of human-readable
    strings, empty when the mask supports what the facts already say. `facts`
    is not mutated.

    Two independent corrections, gated separately -- see this module's
    docstring for the measurement behind the split:

      bone below threshold           -> fov.maxilla := excluded
      ...AND no upper tooth label    -> maxillary FDIs stripped as well

    A case whose maxilla IS in the volume is returned untouched, and this tool
    only ever REMOVES claims the mask cannot support, never adds one -- the
    mask says nothing about whether a tooth the report never mentioned exists.
    """
    if bone_mm3 >= MIN_ARCH_BONE_MM3:
        return facts, []

    out = json.loads(json.dumps(facts))          # deep copy, JSON-only data
    structured = out.get("structured")
    if not isinstance(structured, dict):
        return facts, []

    changes, original = [], {}

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

    if changes:
        out.setdefault("audit", {})["maxilla_excluded"] = {
            "reason": (f"maxillary bone {bone_mm3:.0f} mm^3 in the mask, below "
                       f"the {MIN_ARCH_BONE_MM3:.0f} mm^3 visibility threshold "
                       f"(create_3d_renders.MIN_ARCH_BONE_MM3)"),
            "maxilla_bone_mm3": round(bone_mm3, 1),
            "maxillary_tooth_labels_in_mask": n_upper_teeth,
            "teeth_stripped": n_upper_teeth == 0,
            "original": original,
        }
    return out, changes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts-dir", required=True,
                    help="Per-case facts JSONs ({case_id}.json)")
    ap.add_argument("--masks-dir", required=True,
                    help="Per-case segmentation masks ({case_id}.nii.gz)")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="Only audit these case IDs")
    ap.add_argument("--execute", action="store_true",
                    help="Write the corrected facts files. Without this the "
                         "run is a DRY RUN and only reports.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print the measured maxilla volume for every case, "
                         "not just the ones that change")
    ap.add_argument("--derive-bridge-arches", action="store_true",
                    help="Also DERIVE structured.bridge_arches from the mask's "
                         "bridge label (8) -- which arch each bridge sits in. "
                         "The only thing this tool adds rather than removes, "
                         "hence its own flag.")
    args = ap.parse_args()

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

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"[INFO] Auditing {len(paths)} case(s) against {masks_dir}  [{mode}]")
    print(f"[INFO] Maxilla counts as imaged at >= {MIN_ARCH_BONE_MM3:.0f} mm^3 "
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
            bone, n_teeth = measure_maxilla(mask_path)
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
            # Read the mask once more, unresampled: the arch is read off label
            # adjacency, so no canonical reorientation is needed and the int
            # array is what ndimage.label wants.
            bridges = derive_bridge_arches(
                nib.load(str(mask_path)).get_fdata().astype("int32"))
            # ALWAYS write the field, even empty, and count writing it as a
            # change so the file is saved. Absent means "never audited", empty
            # means "audited, no bridge label in this mask", and a consumer
            # that cannot tell those apart has to treat both as unknown --
            # which left P397's false-positive bridge standing, because its
            # facts file simply had no key for the rule to read.
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
                verdict = "imaged" if bone >= MIN_ARCH_BONE_MM3 else "already audited"
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


if __name__ == "__main__":
    main()
