"""Score mask-derived findings against report-parsed findings, slot by slot.

This is the decision procedure for what the renderer is allowed to say: a slot
whose mask derivation does not clearly beat the corpus prior gets dropped rather
than shipped, because a wrong assertion costs RadFact precision.

Uses the NATIVE ToothFairy3 labels, not TF4_Segmentation_Masks/ -- the latter
disagree with the native labels under every orientation transform tested.
"""

from __future__ import annotations

import collections
import glob
import os
import re
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf4.mask_features import derive_findings  # noqa: E402
from tf4.report_parse import parse_report  # noqa: E402
from tf4.trinity_infer import to_training_frame  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF3_ROOT = os.environ.get("TF3_ROOT", os.path.join(_REPO, "Dataset123_ToothFairy3"))
TF4_ROOT = os.environ.get("TF4_ROOT", os.path.join(_REPO, "toothfairy4_v03"))

SCALAR_SLOTS = [
    "mandible_included", "maxilla_coverage", "condyles_included",
    "edentulism", "canal_course", "has_bridge", "has_implant", "has_crown",
]


def tf3_label_path(case: str) -> str | None:
    """TF4 case id (P001) -> native TF3 label file, if we have one."""
    m = re.match(r"([APFS])(\d+)$", case)
    if not m:
        return None
    cid = f"ToothFairy3{m.group(1)}_{int(m.group(2)):03d}"
    for split in ("labelsTr", "labelsTe"):
        p = os.path.join(TF3_ROOT, split, f"{cid}.nii.gz")
        if os.path.exists(p):
            return p
    return None


def reference_findings(case: str):
    """Union of the case's reports: a slot asserted by any surgeon counts."""
    reports = sorted(glob.glob(f"{TF4_ROOT}/{case}/reports_en/*.txt"))
    if not reports:
        return None
    merged = None
    for path in reports:
        f = parse_report(open(path).read(), case)
        if merged is None:
            merged = f
            continue
        for slot in SCALAR_SLOTS + ["atrophy", "osteolytic_lesion"]:
            if getattr(merged, slot) is None:
                setattr(merged, slot, getattr(f, slot))
        for fdi, present in f.teeth_present.items():
            merged.teeth_present.setdefault(fdi, present)
        merged.impacted |= f.impacted
        merged.implants |= f.implants
        merged.crowns |= f.crowns
        merged.canal_contact.update(f.canal_contact)
    return merged


def main(limit: int = 400) -> None:
    cases = sorted(os.listdir(TF4_ROOT))
    agree = collections.defaultdict(lambda: [0, 0])   # slot -> [match, comparable]
    derived_dist = collections.defaultdict(collections.Counter)
    ref_dist = collections.defaultdict(collections.Counter)
    tooth = [0, 0]
    tooth_absent = [0, 0]
    n = 0

    for case in cases:
        lp = tf3_label_path(case)
        if lp is None:
            continue
        ref = reference_findings(case)
        if ref is None:
            continue
        gt = np.ascontiguousarray(np.asarray(nib.load(lp).dataobj)).astype(np.uint8)
        # Native TF3 arrays are in the training frame; bring them to canonical.
        got = derive_findings(to_training_frame(gt), spacing=0.3, case_id=case)
        n += 1

        for slot in SCALAR_SLOTS:
            a, b = getattr(got, slot), getattr(ref, slot)
            if a is not None:
                derived_dist[slot][a] += 1
            if b is not None:
                ref_dist[slot][b] += 1
            if a is None or b is None:
                continue
            agree[slot][1] += 1
            agree[slot][0] += int(a == b)

        for fdi, present in ref.teeth_present.items():
            if fdi not in got.teeth_present:
                continue
            bucket = tooth if present else tooth_absent
            bucket[1] += 1
            bucket[0] += int(got.teeth_present[fdi] == present)

        if n >= limit:
            break

    print(f"calibrated on {n} cases with native TF3 labels\n")
    print(f"{'slot':22}{'agree':>8}{'n':>7}   derived distribution")
    for slot in SCALAR_SLOTS:
        m, c = agree[slot]
        rate = f"{100*m/c:6.1f}%" if c else "     --"
        print(f"  {slot:20}{rate}{c:>7}   {dict(derived_dist[slot])}")
    print("\nreference distributions:")
    for slot in SCALAR_SLOTS:
        print(f"  {slot:20} {dict(ref_dist[slot])}")
    print(f"\ntooth PRESENT stated by surgeon : {100*tooth[0]/max(tooth[1],1):.1f}% agree (n={tooth[1]})")
    print(f"tooth ABSENT  stated by surgeon : {100*tooth_absent[0]/max(tooth_absent[1],1):.1f}% agree (n={tooth_absent[1]})")


if __name__ == "__main__":
    main()
