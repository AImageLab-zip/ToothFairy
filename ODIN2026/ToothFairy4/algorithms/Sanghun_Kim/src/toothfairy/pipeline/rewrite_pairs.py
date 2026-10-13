#!/usr/bin/env python3
"""Build training pairs for the QLoRA narrative rewriter — (slots -> full report).

The supervision exists exactly: each structure-wise ground-truth file is paired with the report
it belongs to.

One output jsonl line:
  {"pid", "report_file", "split", "slots", "bullets", "target"}
    slots       : {region_id: [line, ...]}  <- the source of the training input
    bullets     : fixed-order bullet digest (the same form the rewriter is fed at inference)
    target      : the full original report

## Read this before training

1. **The train/inference distribution mismatch is the main risk here.** The training input is
   the *complete and correct* ground-truth slots, whereas the real input is the model's
   *incomplete and partly wrong* slots. A rewriter taught that "the input was always
   sufficient" learns to fill the gaps plausibly — exactly the hallucination that hurts most.
2. **Do not judge by captioning.** Having learned the style, the rewriter always wins on
   BLEU/METEOR; that is the trap of this experiment. Judge on RadFact (the official score is
   also weighted RadFact 0.8 / captioning 0.2).
3. **Train on the train split only.** Putting val reports into training invalidates every val
   number measured afterwards. `--split train` is the default, and the split is also written
   into each output row.

Usage:
  python -m toothfairy.pipeline.rewrite_pairs --split train --no-cls \
      --out experiments/analysis/rewrite/pairs_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402

from toothfairy.schema.claims import FDI_ALL                                # noqa: E402

GT_DIR = REPO / "experiments/extraction/per_structure_per_report_split"
SLOT_ORDER = (["maxilla", "mandible", "nerve_right", "nerve_left"]
              + [f"fdi:{f}" for f in FDI_ALL] + ["cls_upper", "cls_lower"])


def bullets_of(slots: dict) -> str:
    return "\n".join(f"- {l}" for k in SLOT_ORDER for l in slots.get(k, []) if l.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("train", "val", "all"), default="train",
                    help="default train — putting val into training invalidates every val "
                         "number measured afterwards")
    ap.add_argument("--no-cls", action="store_true",
                    help="drop the arch summary slots (cls_upper, cls_lower) from the input "
                         "and use only the 36 remaining anatomical tokens. Rationale: the arch "
                         "slots carry only 245 of the 26,194 training GT sentences (0.94 "
                         "percent), yet the model speaks them more often than it should (a "
                         "target exists for 8.9 percent of slots but 14.5 percent are spoken), "
                         "and 12 of the 18 spoken lines were of the `Complete edentulism of "
                         "the mandible` kind, directly contradicting the individual FDI slots. "
                         "Included by default.")
    args = ap.parse_args(argv)

    val = {x.strip() for x in (REPO / "data/splits/val.txt").read_text().split() if x.strip()}
    rows, missing = [], 0

    for f in sorted(GT_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        pid = f.name.split("__")[0]
        split = "val" if pid in val else "train"
        if args.split != "all" and split != args.split:
            continue
        d = json.loads(f.read_text())
        rf = d.get("report_file")
        hits = list((REPO / "data" / pid / "reports_en").glob(rf)) if rf else []
        if not hits:
            missing += 1
            continue
        slots = {k: [s.strip() for s in v if s and s.strip()]
                 for k, v in d.items() if k in SLOT_ORDER and isinstance(v, list)}
        slots = {k: v for k, v in slots.items() if v}
        if args.no_cls:
            slots = {k: v for k, v in slots.items()
                     if k not in ("cls_upper", "cls_lower")}
        if not slots:
            continue
        rows.append({"pid": pid, "report_file": rf, "split": split, "slots": slots,
                     "bullets": bullets_of(slots),
                     "target": hits[0].read_text().strip()})

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_pat = len({r["pid"] for r in rows})
    print(f"done: {len(rows)} rows · {n_pat} patients · split={args.split} -> {out}")
    if missing:
        print(f"  warning: skipped {missing} records whose original report was not found")
    print("  warning: judge on RadFact, not captioning — having learned the style, the "
          "rewriter always improves BLEU/METEOR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
