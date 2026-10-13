#!/usr/bin/env python3
"""
select_sft_pool.py -- the training pool, as three case lists.

The pool is the whole training split. This file writes `all_cases.txt`, holds
out a stratified gate into `heldout.txt`, and writes the complement --
`train_minus_heldout.txt`, which is what vision_sft.sh takes as --case-list.
It prints the composition of each, because composition is the part that goes
wrong quietly.

TWO THINGS ARE CHECKED, NEITHER OF THEM A THRESHOLD
───────────────────────────────────────────────────
1. SPLIT. Nothing from `dataset/validate` may appear, ever. The pool is read
   from dataset/training so this cannot fire today, but it is checked rather
   than assumed: the cost of the check is a directory listing and the cost of
   the assumption is an experiment whose headline number is invalid.

2. AUDIT-CLEAN, if asked. A case whose report_facts carries an ERROR from
   audit_report_facts.py is a case whose labels are known-wrong. Passed in via
   --audit-json, and optional -- arm 6 passed nothing, so no case was excluded.

WHY THE HELD-OUT SET IS STRATIFIED AND THE POOL IS NOT
──────────────────────────────────────────────────────
Case IDs carry their sub-dataset in the prefix letter, and the four differ in
scanner, field of view and reconstruction. Training is 69% P; validate is
10/10/10/10, exactly 25% each:

    prefix   all training   validate
    A            15%          25%
    F             9%          25%
    P            69%          25%
    S             7%          25%

Held-out is the decision gate -- the number an arm is judged on before it costs
a validate run -- and a gate that reads a 69%-P sample to predict a 25%-P
result is not a gate. So it is drawn --heldout-per per prefix, 6 by default,
giving validate's own 6/6/6/6 shape.

The TRAINING pool is deliberately not balanced to match. Balancing it means
throwing away most of P, and every arm measured here has wanted more rows
rather than better-shaped ones; the skew is a property of the challenge's own
training split, and the gate is where it is corrected for.

Arm 6's own 24 held-out cases are not the 24 this draws, so the two are not
byte-comparable: same 6/6/6/6 shape, different names. The gate is the eval-loss
curve rather than a scored artifact -- the 0.4658 is validate-40, which no draw
here can touch -- so what differs is which 558 cases a rebuild trains on, not
what it is measured against. Reproducing arm 6's split exactly needs arm 6's
own heldout.txt.

Usage:
    python code/train/select_sft_pool.py --out-dir outputs/training_results/sft_pool
    python code/train/select_sft_pool.py --out-dir <dir> --audit-json <audit.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path


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

ROOT = REPO_ROOT

HELDOUT_PER = 6      # per prefix -> 24, matching validate's 10/10/10/10 shape


def case_of(text: str) -> str | None:
    m = re.match(r"([A-Z]+\d+)", str(text))
    return m.group(1) if m else None


def audit_error_cases(path: Path | None) -> set[str]:
    """Case ids with at least one surviving ERROR.

    Tolerant about the JSON shape on purpose -- audit_report_facts.py's --json
    has changed shape once already, and a pool silently built from zero
    exclusions is worse than a crash.
    """
    if not path:
        return set()
    blob = json.loads(path.read_text(encoding="utf-8"))
    items = blob if isinstance(blob, list) else (
        blob.get("findings") or blob.get("errors") or [])
    if not items:
        sys.exit(f"[FAIL] {path}: no findings parsed -- shape changed? "
                 f"top-level keys: {list(blob)[:8] if isinstance(blob, dict) else 'list'}")
    bad = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        level = str(it.get("level") or it.get("severity") or "").upper()
        if level != "ERROR":
            continue
        c = case_of(it.get("case") or it.get("stem") or it.get("file") or "")
        if c:
            bad.add(c)
    return bad


def validate_case_ids() -> set[str]:
    """Every case id under dataset/validate. The hard boundary."""
    d = ROOT / "dataset/validate/images"
    return {c for c in (case_of(p.name) for p in d.glob("*_0000.nii.gz")) if c}


def training_case_ids() -> set[str]:
    """Every case id under dataset/training -- the pool. Read from the volumes
    rather than from a list file, so it cannot disagree with what the renderers
    will find."""
    d = ROOT / "dataset/training/images"
    ids = {c for c in (case_of(p.name) for p in d.glob("*_0000.nii.gz")) if c}
    if not ids:
        sys.exit(f"[FAIL] no cases under {d} -- is the training split present?")
    return ids


def composition(cases) -> dict:
    return dict(sorted(collections.Counter(c[0] for c in cases).items()))


def fmt(cases, label: str) -> str:
    comp = composition(cases)
    n = len(cases)
    parts = " ".join(f"{k}:{v:>3} ({100 * v / n:>3.0f}%)" for k, v in comp.items()) if n else "-"
    calls = f"~{round(n * 19.5):,} calls"
    return f"  {label:<22} {n:>4} cases  {calls:>14}   {parts}"


def stratify(cases, per_prefix: int, rng) -> list[str]:
    """Take `per_prefix` of each prefix, at random."""
    by = collections.defaultdict(list)
    for c in cases:
        by[c[0]].append(c)
    out = []
    for p in sorted(by):
        pool = sorted(by[p])
        rng.shuffle(pool)
        if len(pool) < per_prefix:
            print(f"  [WARN] {p}: only {len(pool)} case(s), wanted {per_prefix}")
        out.extend(pool[:per_prefix])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path,
                    help="audit_report_facts.py --json output; cases with a "
                         "surviving ERROR are excluded from the pool")
    ap.add_argument("--heldout-per", type=int, default=HELDOUT_PER,
                    help=f"held-out cases per prefix (default {HELDOUT_PER} -> "
                         f"24 total, matching validate's 10/10/10/10 shape)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bad = audit_error_cases(args.audit_json)
    forbidden = validate_case_ids()
    cases = sorted(training_case_ids() - bad - forbidden)

    held = stratify(cases, args.heldout_per, rng)
    held_set = set(held)
    train = [c for c in cases if c not in held_set]

    print(f"[INFO] pool: dataset/training in full")
    print(f"[INFO] audit ERRORs exclude {len(bad)} case(s); "
          f"{len(forbidden)} validate ids are forbidden and none appeared")
    print()
    print("SELECTED:")
    print(fmt(cases, "all training cases"))
    print(fmt(train, "train minus held-out"))
    print(fmt(held, "held-out (the gate)"))
    print()

    # The two invariants worth crashing over. Held-out leaking into the
    # training pool is the one that would invalidate the gate without changing
    # any number that looks wrong.
    assert not (set(train) & held_set), "held-out leaked into training"
    assert not ((set(train) | held_set) & forbidden), "validate case in a pool"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, group in (("all_cases", cases), ("train_minus_heldout", train),
                        ("heldout", held)):
        f = args.out_dir / f"{name}.txt"
        f.write_text("\n".join(group) + "\n", encoding="utf-8")
        written[name] = {"n": len(group), "composition": composition(group),
                         "cases": group}
        print(f"[INFO] {f} ({len(group)} cases)")

    meta = args.out_dir / "pool.json"
    meta.write_text(json.dumps({
        "seed": args.seed,
        "heldout_per": args.heldout_per,
        "audit_error_cases": sorted(bad),
        "sets": written,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
