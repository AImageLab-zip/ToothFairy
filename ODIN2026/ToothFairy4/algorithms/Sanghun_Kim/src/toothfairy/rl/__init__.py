"""Reward for reinforcement learning — rule-based claim extraction, clinical terms, weighted sum.

Nothing here imports torch (it is all regular expressions, on the CPU). This package is the
single canonical implementation, so that the training loop and any offline scoring use exactly
the same code.

    from toothfairy.rl import region_reward
    total, terms = region_reward(gen, gt, {"pid": "A005", "region_id": "fdi:36"})

The sampling module `toothfairy.rl.rollout` is deliberately *not* re-exported here — it pulls
in torch, transformers and peft, and keeping it separate lets the CPU-only reward code run
without that heavy stack. The rejection-sampling pipeline imports it directly; see
`toothfairy.cli.rft`.
"""

from .extract import claim_f1, extract_claims, set_f1  # noqa: F401
from .reward import DEFAULT_WEIGHTS, region_reward  # noqa: F401
from .terms import (  # noqa: F401
    extract_fdi, extract_polarity, extract_terms,
    fdi_report, polarity_contradictions, polarity_f1, term_f1,
)

__all__ = [
    "DEFAULT_WEIGHTS", "region_reward",
    "claim_f1", "extract_claims", "set_f1",
    "extract_fdi", "extract_polarity", "extract_terms",
    "fdi_report", "polarity_contradictions", "polarity_f1", "term_f1",
]
