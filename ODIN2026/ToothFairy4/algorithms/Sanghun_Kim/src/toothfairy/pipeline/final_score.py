#!/usr/bin/env python3
"""Official score — `Final = 0.8 * RadFact + 0.2 * avg(BLEU-4, METEOR)`.

**Every model evaluation stores this value alongside its components**, because looking at only one
of the two leads to a wrong conclusion in either direction: in one measurement three
post-processing variants were indistinguishable under RadFact (within 0.3 sigma) while their
captioning score ranged from 0.1317 to 0.1728, and conversely the variant that wins on captioning
alone differs by only 0.0023 in Final (39% chance of flipping).

Two inputs are paired up:
  predictions.csv               example_id, prediction, target
  radfact_.../per_sample_scores.json   logical_f1 per study_id (RadFact output)

WARNING: **when comparing several models, use `--common-with` to keep only the studies scored by
all of them.** RadFact deterministically drops a study when the judge fails to quote its evidence
verbatim from the source, and that dropped set differs per model (measured: 48-52 of 62 studies,
depending on the variant). Subtracting overall means would compare different patients.

Usage:
  python -m toothfairy.pipeline.final_score --csv <run>/predictions.csv \
      --radfact <run>/radfact/per_sample_scores.json

  # compare several variants on the common set of studies
  python -m toothfairy.pipeline.final_score --csv a.csv --radfact a.json \
      --common-with b.json c.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402


def _load_radfact(p: Path) -> dict:
    rows = json.loads(p.read_text())
    return {r["study_id"]: r for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="predictions.csv (example_id, prediction, target)")
    ap.add_argument("--radfact", required=True, help="per_sample_scores.json (RadFact output)")
    ap.add_argument("--common-with", nargs="*", default=[],
                    help="per_sample_scores.json of other variants — keep commonly scored studies")
    ap.add_argument("--out", default=None, help="default = <directory of csv>/final_score.json")
    args = ap.parse_args(argv)

    import sacrebleu
    from nltk.translate.meteor_score import meteor_score

    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else REPO / args.csv
    rf_path = Path(args.radfact) if Path(args.radfact).is_absolute() else REPO / args.radfact
    out_path = Path(args.out) if args.out else csv_path.parent / "final_score.json"

    preds = {r["example_id"]: r for r in csv.DictReader(open(csv_path))}
    rf = _load_radfact(rf_path)

    ids = {i for i, r in rf.items() if r.get("logical_f1") is not None} & set(preds)
    n_scored, n_total = len(ids), len(preds)
    for other in args.common_with:
        o = Path(other) if Path(other).is_absolute() else REPO / other
        og = _load_radfact(o)
        ids &= {i for i, r in og.items() if r.get("logical_f1") is not None}
    ids = sorted(ids)
    if not ids:
        raise SystemExit("no study was scored by all inputs — check the inputs")

    radfact = statistics.mean(rf[i]["logical_f1"] for i in ids)
    p = [preds[i]["prediction"] for i in ids]
    t = [preds[i]["target"] for i in ids]
    bleu4 = sacrebleu.corpus_bleu(p, [t]).score / 100.0
    meteor = statistics.mean(meteor_score([r.split()], q.split()) for q, r in zip(p, t))
    captioning = (bleu4 + meteor) / 2.0
    final = 0.8 * radfact + 0.2 * captioning

    result = {
        "final": round(final, 4),
        "radfact_logical_f1": round(radfact, 4),
        "bleu4": round(bleu4, 4),
        "meteor": round(meteor, 4),
        "captioning_avg": round(captioning, 4),
        "formula": "0.8 * RadFact + 0.2 * avg(BLEU-4, METEOR)",
        "n_used": len(ids),
        "n_radfact_scored": n_scored,
        "n_predictions": n_total,
        "common_with": args.common_with,
        "csv": str(csv_path),
        "radfact": str(rf_path),
        # RadFact precision and recall are needed to read off what actually changed; the F1 alone
        # does not show it.
        "radfact_precision": round(statistics.mean(rf[i]["logical_precision"] for i in ids), 4),
        "radfact_recall": round(statistics.mean(rf[i]["logical_recall"] for i in ids), 4),
        "candidate_phrases": round(statistics.mean(rf[i]["num_candidate_phrases"] for i in ids), 1),
        "reference_phrases": round(statistics.mean(rf[i]["num_reference_phrases"] for i in ids), 1),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if len(ids) < n_scored:
        print(f"\nWARNING: the common set shrank from {n_scored} to {len(ids)} studies. "
              f"The more variants are compared at once, the weaker the discrimination.")
    print(f"[final] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
