#!/usr/bin/env python3
"""The narrative rewriter: 36 structure-wise bullets -> one running report.

A separate QLoRA adapter, trained ground truth -> ground truth on the training split only,
so there is no distribution shift in what it learns: it is taught reporting style, not
clinical content. It is applied to the model's slot outputs at the very end of the pipeline.

    pairs           build training pairs from the reference slots (train split)
    pairs-from-run  build the pairs to rewrite from a run's slot outputs
    train           train the adapter on the pairs
    apply           rewrite with the adapter -> predictions.csv ready for scoring

``--no-cls`` drops the two arch-summary slots. The submitted adapter was trained that way,
and the choice is recorded in ``train_meta.json`` next to the adapter so that whatever
applies it feeds the same 36 slots.

Judge the result with RadFact, not with captioning: having learned the style, the rewriter
will almost certainly win on BLEU and METEOR — that is the trap of this step. The question
is whether clinical content was lost or invented.
"""
from __future__ import annotations

from toothfairy.pipeline import (_cli, rewrite_apply, rewrite_pairs,
                                 rewrite_pairs_from_run, rewrite_train)

COMMANDS = {
    "pairs":          (rewrite_pairs.main,          "training pairs from the reference slots"),
    "pairs-from-run": (rewrite_pairs_from_run.main, "pairs to rewrite from a run's slot outputs"),
    "train":          (rewrite_train.main,          "train the QLoRA adapter on the pairs"),
    "apply":          (rewrite_apply.main,          "rewrite with the adapter -> predictions.csv"),
}


def main(argv: list[str] | None = None) -> int:
    return _cli.dispatch(__doc__, COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
