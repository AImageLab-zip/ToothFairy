"""Estimate a claim's entailment rate without spending a RadFact run.

RadFact needs an LLM, an hour, and a GPU that cannot simultaneously hold the
segmentation model. Most slot decisions do not need it. For a per-tooth claim,
the question the metric asks is simply "does the reference say this about this
tooth", and a regex over the reference answers it for 527 cases in seconds.

The number that matters is the BREAK-EVEN entailment rate. Adding one candidate
phrase entailed with probability q changes F1 by

    dF1 = dF1/dP * (q - P)/n  +  dF1/dR * q/m
        with dF1/dP = 2R^2/(P+R)^2 and dF1/dR = 2P^2/(P+R)^2

which is positive above q* = P / (1 + P^2*n / (R^2*m)) -- about 0.24 at the
current operating point, and it stays in 0.22-0.30 all the way to P = R = 0.55.
Slots are rejected here only when they fall under that bar. Entailment by a
single reference is not correctness: reports are selective, so a true finding
the reference leaves out still counts as not entailed.

Every rate below is a lower bound on the true entailment rate: the reference is
one annotator, and annotators leave out plenty that is really there. Where it
matters, `--union` scores against every annotator for the case instead.

    python tf4/slot_calibration.py
    python tf4/slot_calibration.py --slot endodontic --union
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "toothfairy4_v03")
FINDINGS = os.path.join(ROOT, "work", "pred_findings.json")
FDI = r"(?:1[1-8]|2[1-8]|3[1-8]|4[1-8])"

UPPER = {18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28}

# What counts as the reference stating this finding about a tooth. Deliberately
# generous: a narrow pattern understates the rate, and understating is how the
# per-site crown claims got rejected the first time on a "17.6% precision"
# figure that came from a regex matching only "crown on tooth NN".
PATTERNS = {
    "endodontic": r"endodont|root canal (?:filling|treat|material|therapy)|post[- ]and[- ]core"
                  r"|endocanal|intracanal|gutta|obturat|beyond the apex|apicect",
    "crowns": r"\bcrown|prosthe|abutment|pontic|capsul",
    "restorations": r"filling|restorat|black class|composite|conservative|amalgam",
    "impacted": r"impact|unerupt|semi[- ]?includ|disodont|bony inclusion",
    "absent": r"absen|missing|avuls|agenes|post-extract|extracted",
}

# The findings-record key each slot reads, and how to turn it into a tooth set.
SLOTS = {
    "endodontic": ("endodontic", "endodontic"),
    "crowns": ("crowns", "crowns"),
    "restorations": ("restorations", "restorations"),
    "impacted": ("impacted", "impacted"),
    "absent": ("teeth", "absent"),
}


def reports_for(case: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(DATA, case, "reports_en", "*.txt")),
        key=lambda p: int(re.search(r"cbct_(\d+)_", p).group(1)),
    )


def stated_teeth(case: str, pattern: str, union: bool) -> set[int] | None:
    """Teeth the reference says this about. `union` uses every annotator."""
    paths = reports_for(case)
    if not paths:
        return None
    out: set[int] = set()
    for path in paths if union else paths[:1]:
        for sentence in re.split(r"(?<=[.;])\s+|\n+", open(path).read()):
            if re.search(pattern, sentence, re.I):
                out |= {int(t) for t in re.findall(FDI, sentence)}
    return out


def claimed_teeth(record: dict, slot: str) -> set[int]:
    key, kind = SLOTS[slot]
    value = record.get(key)
    if kind == "absent":
        return {int(k) for k, present in (value or {}).items() if present is False}
    return set(value or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="", help="one slot; default is all of them")
    ap.add_argument("--union", action="store_true",
                    help="score against every annotator, not just the reference")
    ap.add_argument("--by-arch", action="store_true",
                    help="split upper/lower -- they behave very differently")
    args = ap.parse_args()

    findings = json.load(open(FINDINGS))
    slots = [args.slot] if args.slot else list(SLOTS)

    print(f"{len(findings)} cases from {os.path.relpath(FINDINGS, ROOT)}; "
          f"reference = {'union of all annotators' if args.union else 'lowest annotator id'}")
    print(f"break-even entailment rate is about 0.24\n")
    print(f"{'slot':16s}{'arch':7s}{'claims/case':>12s}{'entailed':>10s}{'recall':>9s}{'verdict':>12s}")

    for slot in slots:
        pattern = PATTERNS[slot]
        tally = defaultdict(lambda: [0, 0])          # arch -> [claims, entailed]
        recall = defaultdict(lambda: [0, 0])         # arch -> [stated, covered]
        n_cases = 0
        for case, record in findings.items():
            stated = stated_teeth(case, pattern, args.union)
            if stated is None:
                continue
            n_cases += 1
            claims = claimed_teeth(record, slot)
            for fdi in claims:
                arch = "upper" if fdi in UPPER else "lower"
                for key in (("all",) if not args.by_arch else ("all", arch)):
                    tally[key][0] += 1
                    tally[key][1] += fdi in stated
            for fdi in stated:
                arch = "upper" if fdi in UPPER else "lower"
                for key in (("all",) if not args.by_arch else ("all", arch)):
                    recall[key][0] += 1
                    recall[key][1] += fdi in claims

        for arch in (["all"] if not args.by_arch else ["all", "lower", "upper"]):
            claims, entailed = tally[arch]
            if not claims:
                continue
            rate = entailed / claims
            stated_n, covered = recall[arch]
            verdict = "ship" if rate >= 0.35 else ("marginal" if rate >= 0.24 else "DROP")
            print(f"{slot:16s}{arch:7s}{claims / n_cases:12.2f}{rate:10.3f}"
                  f"{covered / max(1, stated_n):9.3f}{verdict:>12s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
