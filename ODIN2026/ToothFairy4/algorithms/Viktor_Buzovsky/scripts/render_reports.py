"""Render reports from the derived findings in work/pred_findings.json.

This is the last stage of the pipeline outside the submission container:

    python tf4/build_findings.py segment     # CBCT -> cached masks (GPU)
    python tf4/build_findings.py derive      # masks -> findings records (CPU)
    python scripts/render_reports.py         # findings -> work/reports/<case>.txt

    python scripts/render_reports.py --case F001    # print a single report
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tf4.radfact_fast import render_cached  # noqa: E402

FINDINGS = os.path.join(ROOT, "work", "pred_findings.json")
OUT_DIR = os.path.join(ROOT, "work", "reports")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="", help="print one case instead of writing all of them")
    args = ap.parse_args()

    findings = json.load(open(FINDINGS))
    if args.case:
        print(render_cached(args.case, findings[args.case], set()))
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    for case, rec in sorted(findings.items()):
        with open(os.path.join(OUT_DIR, f"{case}.txt"), "w") as fh:
            fh.write(render_cached(case, rec, set()) + "\n")
    print(f"wrote {len(findings)} reports to {os.path.relpath(OUT_DIR, ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
