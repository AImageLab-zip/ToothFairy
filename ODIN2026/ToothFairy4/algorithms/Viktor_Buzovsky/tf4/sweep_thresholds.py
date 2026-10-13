"""Sweep mask-derivation thresholds against the report-parsed references.

Reads the cached geometry from dump_features.py, so each sweep is instant rather
than a full mask reload. Reports, for every candidate threshold, how often the
derived slot matches what a surgeon actually wrote -- and how that compares with
simply asserting the corpus prior. A derivation that cannot beat its prior does
not belong in the renderer.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf4.calibrate import reference_findings  # noqa: E402
from tf4.schema import LOWER, UPPER  # noqa: E402

FEATURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "features.json")


def main() -> None:
    feats = json.load(open(FEATURES))
    refs = {}
    for case in feats:
        r = reference_findings(case)
        if r is not None:
            refs[case] = r
    print(f"{len(feats)} cases with geometry, {len(refs)} with parsed reports\n")

    # ---------------- tooth presence / absence ----------------
    print("tooth voxel threshold sweep (agreement with surgeon statements)")
    print(f"{'thresh':>8}{'present':>10}{'n':>7}{'absent':>10}{'n':>7}{'balanced':>10}")
    best = None
    for thresh in (50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000):
        pres = [0, 0]
        absent = [0, 0]
        for case, ref in refs.items():
            teeth = feats[case]["teeth"]
            for fdi, stated in ref.teeth_present.items():
                n = teeth.get(str(fdi))
                if n is None:
                    continue
                derived = n >= thresh
                bucket = pres if stated else absent
                bucket[1] += 1
                bucket[0] += int(derived == stated)
        p = pres[0] / max(pres[1], 1)
        a = absent[0] / max(absent[1], 1)
        bal = 2 * p * a / (p + a) if (p + a) else 0.0
        print(f"{thresh:>8}{100*p:>9.1f}%{pres[1]:>7}{100*a:>9.1f}%{absent[1]:>7}{100*bal:>9.1f}%")
        if best is None or bal > best[0]:
            best = (bal, thresh, p, a)
    print(f"  -> best balanced threshold {best[1]} (present {100*best[2]:.1f}%, absent {100*best[3]:.1f}%)\n")

    # ---------------- canal course ----------------
    print("canal lingual cutoff sweep (signed position; < cutoff => lingual)")
    print(f"{'cutoff':>8}{'agree':>9}{'n':>7}   derived")
    vals = [c for c in refs if feats[c].get("canal_rel_mean") is not None
            and refs[c].canal_course is not None]
    prior = max(
        sum(1 for c in vals if refs[c].canal_course == label) for label in ("lingual", "regular")
    ) / max(len(vals), 1)
    for cutoff in (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20):
        match = 0
        derived_counts = {"lingual": 0, "regular": 0}
        for c in vals:
            d = "lingual" if feats[c]["canal_rel_mean"] < cutoff else "regular"
            derived_counts[d] += 1
            match += int(d == refs[c].canal_course)
        print(f"{cutoff:>8.2f}{100*match/max(len(vals),1):>8.1f}%{len(vals):>7}   {derived_counts}")
    print(f"  -> always-majority prior would score {100*prior:.1f}%\n")

    # ---------------- maxilla coverage ----------------
    print("upper-jaw voxel threshold for maxilla 'none' vs included")
    print(f"{'thresh':>8}{'agree':>9}{'n':>7}")
    mx = [c for c in refs if refs[c].maxilla_coverage is not None]
    for thresh in (1000, 5000, 20000, 50000, 100000, 200000):
        match = 0
        for c in mx:
            d = "none" if feats[c]["upper_jaw"] < thresh else "partial"
            match += int(d == refs[c].maxilla_coverage)
        print(f"{thresh:>8}{100*match/max(len(mx),1):>8.1f}%{len(mx):>7}")
    base = sum(1 for c in mx if refs[c].maxilla_coverage == "none") / max(len(mx), 1)
    print(f"  -> always-'none' prior scores {100*max(base,1-base):.1f}%")


if __name__ == "__main__":
    main()
