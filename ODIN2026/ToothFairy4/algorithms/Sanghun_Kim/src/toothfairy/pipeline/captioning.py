#!/usr/bin/env python3
"""ODIN captioning metrics (BLEU-4, METEOR) from predictions.csv.

The official ToothFairy4 Phase 1 captioning score is avg(BLEU-4, METEOR). RadFact is the
primary metric for model selection here and BLEU/METEOR are secondary, but they are computed
anyway: they carry 0.2 of the official score and they catch text collapse.

Input = the predictions.csv written by `toothfairy.cli.eval generate` (example_id, prediction,
target). No GPU
needed (CPU only). BLEU-4 = sacrebleu corpus BLEU (built-in 13a tokenizer, 0-1 scale). METEOR =
the mean of nltk sentence METEOR over cases (needs wordnet). Greedy generation plus a fixed
tokenisation makes this deterministic: the same CSV always gives the same numbers.

Usage:
  python -m toothfairy.pipeline.captioning \
    --csv <run>/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="predictions.csv (example_id,prediction,target)")
    ap.add_argument("--out", default=None, help="default = <csv dir>/captioning_metrics.json")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "captioning_metrics.json"

    rows = list(csv.DictReader(open(csv_path)))
    preds = [(r["prediction"] or "").strip() for r in rows]
    refs = [(r["target"] or "").strip() for r in rows]
    n = len(rows)

    import sacrebleu
    from nltk.translate.meteor_score import meteor_score

    # BLEU-4 (corpus). sacrebleu takes references as [ [ref_1, ref_2, ...] ]; here a single ref.
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    bleu4 = bleu.score / 100.0                                  # 0-1

    # METEOR: mean of the per-case sentence METEOR. Tokenisation is a whitespace split, so it
    # does not depend on punkt and stays deterministic.
    meteors = [meteor_score([r.split()], p.split()) if (p and r) else 0.0
               for p, r in zip(preds, refs)]
    meteor = mean(meteors) if meteors else 0.0

    captioning = (bleu4 + meteor) / 2.0                        # official avg(BLEU-4, METEOR)
    result = {
        "n": n,
        "bleu4": round(bleu4, 4),
        "meteor": round(meteor, 4),
        "captioning_avg": round(captioning, 4),
        "bleu_detail": bleu.format(),
        "csv": str(csv_path),
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[captioning] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
