#!/usr/bin/env python3
"""Model output (per_region.jsonl) -> input pairs for the QLoRA narrative rewriter.

`pipeline/rewrite_pairs.py` builds *training* pairs out of the **GT slots**. To actually run the
rewriter the input has to be the **slots the model emitted**, and that conversion was never
kept as a script (the pair file existed, the code that produced it did not). It lives here so
that any run can be rewritten by the same procedure — comparing runs only holds up if the
procedure is reproducible.

Warning: `target` is taken **verbatim from `predictions.csv`**. The GT is not reassembled — if
the reference differs from the one the existing scoring used by even one character, before and
after rewriting are no longer comparable.

Warning, splitting a slot into lines: `gen` runs the bullets together as `- a - b` with no
newlines (measured: 1,269 slots, none containing a newline). So the split is on the **bullet
marker**, not on newlines. To avoid cutting hyphens inside a word (`moderate-to-severe`), only
a `-` surrounded by whitespace and followed by a capital letter or a digit is treated as a
boundary.

Usage:
  python -m toothfairy.pipeline.rewrite_pairs_from_run \
      --run-dir models/<run> \
      --out experiments/analysis/narrative-rewrite/pairs_<run>.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402

# Bullet boundary: a "- " at the start of a line, or a " - " surrounded by whitespace whose
# next character is a capital or a digit. Word-internal hyphens (moderate-to-severe,
# post-and-core) are not surrounded by whitespace and so are never matched.
_BULLET = re.compile(r"(?:^|\s)-\s+(?=[A-Z0-9])")


def split_slot(text: str) -> list[str]:
    """Split the text the model emitted for one region into a list of bullet lines."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip(" -\t") for p in _BULLET.split(text)]
    return [p for p in parts if p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="models/<run> with per_region.jsonl + predictions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-cls", action="store_true",
                    help="drop the arch summary (CLS) slots from the input — **if they were\n"
                         "dropped in training they must be dropped at inference too**\n"
                         "(models/rewrite_qlora_nocls was trained on 36 tokens).")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    if not run.is_absolute():
        run = REPO / run
    per_region, preds = run / "per_region.jsonl", run / "predictions.csv"
    for p in (per_region, preds):
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr)
            return 1

    # 1) reference: exactly the one the existing scoring used
    targets: dict[str, str] = {}
    with open(preds, newline="") as f:
        for row in csv.DictReader(f):
            targets[row["example_id"]] = row["target"]

    # 2) model slots
    slots: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []
    n_rows = 0
    for line in open(per_region):
        r = json.loads(line)
        n_rows += 1
        pid = r["pid"]
        if pid not in slots:
            slots[pid] = {}
            order.append(pid)
        if args.no_cls and r["region_id"] in ("cls_upper", "cls_lower"):
            continue
        lines = split_slot(r.get("gen"))
        if lines:                                   # silent regions are skipped (as in GT pairs)
            slots[pid][r["region_id"]] = lines

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_missing = 0
    with open(out, "w") as f:
        for pid in order:
            if pid not in targets:
                n_missing += 1
                continue
            f.write(json.dumps({"pid": pid, "report_file": "", "split": args.split,
                                "slots": slots[pid], "target": targets[pid]},
                               ensure_ascii=False) + "\n")
            n_written += 1

    n_slots = sum(len(v) for v in slots.values())
    n_lines = sum(len(x) for v in slots.values() for x in v.values())
    print(f"[pairs] {per_region.name} {n_rows} rows -> {n_written} patients "
          f"(no reference {n_missing}) · regions {n_slots} · slot lines {n_lines} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
