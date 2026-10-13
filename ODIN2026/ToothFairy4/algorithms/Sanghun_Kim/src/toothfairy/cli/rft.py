#!/usr/bin/env python3
"""Rejection-sampling fine-tuning: the last stage of the submitted model.

The LoRA of the LLM stage is refined on its own samples. For every anatomical slot the
policy is sampled several times, each sample is scored against that slot's reference by
an entailment judge, the samples that clear the threshold are kept, and the LoRA is
fine-tuned on them for one epoch. Only utterances the model itself produced are ever
trained on, which is what the "Utterance Selection" in the name refers to.

Run in this order (TRAIN.md has the full recipe with the arguments that were used):

    rollout       sample the policy and score with the fast reward
    filter-cache  cache the official finding_filter decisions for both sides
    rescore       re-score the dump with the entailment judge, through that filter
    select        pick the samples to train on (rollout --report-only, no generation)
    finetune      one epoch of LoRA fine-tuning on the selected samples
    merge         fold the fine-tuned LoRA onto the starting policy -> one checkpoint

``rollout`` and ``finetune`` need a GPU. ``filter-cache`` and ``rescore`` talk to an
OpenAI-compatible endpoint; the submitted model used a locally served Qwen3-32B-AWQ.
"""
from __future__ import annotations

from toothfairy.pipeline import (_cli, rft_filter_cache, rft_finetune, rft_merge,
                                 rft_rescore, rft_rollout)


def _select(argv: list[str]) -> int:
    """`select` is the rollout program in report-only mode: it reads an existing dump and
    writes the selection, generating nothing. The flag is added here so that forgetting it
    cannot silently start a six-hour regeneration."""
    return rft_rollout.main(["--report-only", *argv])


COMMANDS = {
    "rollout":      (rft_rollout.main,      "sample the policy and score each sample"),
    "filter-cache": (rft_filter_cache.main, "cache the official finding_filter decisions"),
    "rescore":      (rft_rescore.main,      "re-score the dump with the entailment judge"),
    "select":       (_select,               "pick samples to train on (reads an existing dump)"),
    "finetune":     (rft_finetune.main,     "one epoch of LoRA fine-tuning on the selection"),
    "merge":        (rft_merge.main,        "merge the fine-tuned LoRA into one checkpoint"),
}


def main(argv: list[str] | None = None) -> int:
    return _cli.dispatch(__doc__, COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
