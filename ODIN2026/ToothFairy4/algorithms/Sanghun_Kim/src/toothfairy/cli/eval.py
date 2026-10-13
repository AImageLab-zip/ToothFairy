#!/usr/bin/env python3
"""Evaluate a trained run: generate the reports, then score them.

    generate    run the model over the validation split -> predictions.csv, per_region.jsonl
    captioning  BLEU-4 and METEOR from predictions.csv (no GPU)
    score       the official combination, 0.8 x RadFact + 0.2 x mean(BLEU-4, METEOR)

``score`` reads a RadFact per-sample file; it does not compute the entailment itself. The
challenge's own scorer (radfact_lite, with the organisers' hosted judge) is not vendored
here — EVAL.md says how the reported numbers were produced and why a locally served judge
gives numbers on a different scale that must not be compared with the official ones.
"""
from __future__ import annotations

from toothfairy.pipeline import _cli, captioning, final_score, generate

COMMANDS = {
    "generate":   (generate.main,    "generate reports for the validation split"),
    "captioning": (captioning.main,  "BLEU-4 / METEOR from predictions.csv"),
    "score":      (final_score.main, "0.8 x RadFact + 0.2 x captioning"),
}


def main(argv: list[str] | None = None) -> int:
    return _cli.dispatch(__doc__, COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
