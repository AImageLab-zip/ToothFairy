"""Region reward — weighted sum of claim F1, finding-term F1 and polarity F1.

Definition
----------
    r = 0.7·claim_F1 + 0.2·term_F1 + 0.1·polarity_F1
    Normal region (GT is the empty string): 1.0 for silence, 0.0 for saying anything

What the three terms do
-----------------------
- **claim_F1** (`extract.claim_f1`) — F1 over the set of `(target, axis, value)` claims. The
  tooth number is part of the target, so a wrong number invalidates the claim as a whole.
  This is the main clinical-accuracy signal (0.7).
- **term_F1** (`terms.term_f1`) — F1 over the set of finding **types** named. It still gives a
  non-zero partial signal for wordings the claim rules cannot parse yet (regional
  descriptions, rare findings), which keeps the signal dense early in training (0.2).
- **polarity_F1** (`terms.polarity_f1`) — F1 over (concept, sign) pairs. It penalises
  present<->absent and included<->excluded flips on both precision and recall (0.1).

Return
------
`(total, per-term breakdown)`. The `contrib` entries of the breakdown are built so that they
**sum exactly to the total**, which makes it directly visible which term is rising during
training (i.e. where reward hacking would show up).
"""

from __future__ import annotations

from .extract import claim_f1, extract_claims, filter_claims
from .presence import strip_presence
from .terms import (fdi_report, polarity_contradictions, polarity_f1,
                    term_f1)

# Fixed weights of the reward. Do not change them ad hoc.
DEFAULT_WEIGHTS: dict[str, float] = {"claim": 0.7, "term": 0.2, "polarity": 0.1}

TERM_KEYS = ("claim", "term", "polarity")

# FDI gate — an extra multiplier on the weighted sum of the three terms above:
#   gate = (1 − invalid-FDI rate) × (own tooth named ? 1 : 0.5)


def _weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    bad = set(weights) - set(TERM_KEYS)
    if bad:
        raise KeyError(f"unknown reward weight(s): {sorted(bad)}; expected {TERM_KEYS}")
    w = {k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in TERM_KEYS}
    if sum(w.values()) <= 0:
        raise ValueError(f"weights must sum > 0, got {w}")
    return w




def region_reward(gen: str, gt: str, ctx: dict | None = None,
                  weights: dict[str, float] | None = None,
                  neutralize_presence: bool = True,
                  excluded_axes=None) -> tuple[float, dict]:
    """Reward for one region. Returns (total in 0..1, per-term breakdown).

    ctx = {"pid": str, "region_id": str|None, ...}
      If region_id is `"fdi:XX"` the correct tooth of that region is fixed to XX, so
        (1) claim extraction keeps only XX claims (a claim on another number disappears), and
        (2) the FDI gate additionally penalises invalid numbers.
    """
    w = _weights(weights)
    wsum = sum(w.values())
    rid = (ctx or {}).get("region_id")
    # Presence statements are neutralised: a slot that only says a tooth exists counts as
    # having said nothing. The three F1 terms are untouched (`Tooth 31 present` is already the
    # empty set for claims, finding terms and polarity alike); what changes is only whether
    # the region is judged normal or as carrying a finding. See rl.presence.
    if neutralize_presence:
        gen_s, gt_s = strip_presence(gen), strip_presence(gt)
    else:
        gen_s, gt_s = (gen or ""), (gt or "")
    g_empty, t_empty = not gen_s.strip(), not gt_s.strip()
    # When the scored axes are narrowed, the normal/finding decision must follow the same
    # axes: a region that carried only a field-of-view statement is left with zero scorable
    # claims, so keeping it as a "finding region" would degenerately score silence as 0.
    if excluded_axes and not t_empty:
        if not filter_claims(extract_claims(gt_s, rid), excluded_axes):
            out_skip = {"region_id": rid, "pid": (ctx or {}).get("pid"), "skip": True}
            return float("nan"), out_skip

    out: dict = {
        "region_id": rid,
        "pid": (ctx or {}).get("pid"),
        "weights": w,
        "empty_gt": t_empty,
        "empty_gen": g_empty,
    }

    # --- Normal region (empty GT), and the cases where only one side is empty ------------
    # A normal region scores 1 for silence and 0 for speaking. The reverse case (GT non-empty
    # but generation empty) is also scored 0 — otherwise a GT sentence whose claims the rules
    # fail to parse would award silence a 1.0 and "always stay silent" would win
    # (the same convention the token-level F1 uses).
    if t_empty or g_empty:
        v = 1.0 if (t_empty and g_empty) else 0.0
        out.update({
            "silence_ok": bool(t_empty and g_empty),
            "claim_f1": v, "term_f1": v, "polarity_f1": v,
            "fdi_gate": 1.0,
            "fdi_invalid_rate": 0.0, "fdi_own_ok": None,
            "polarity_contradictions": 0,
            "n_claim_gen": 0 if g_empty else len(extract_claims(gen_s, rid)),
            "n_claim_gt": 0 if t_empty else len(extract_claims(gt_s, rid)),
            "contrib": {k: w[k] / wsum * v for k in TERM_KEYS},
        })
        out["total"] = sum(out["contrib"].values())
        return out["total"], out

    # --- General path ------------------------------------------------------
    # Score on the presence-stripped text. The three F1 terms would come out the same on the
    # raw text (presence statements are the empty set for every extractor), but the FDI gate
    # would not: on raw text the numbers named by a presence statement would count as "FDI
    # named by the generation" and perturb the gate. Keep one single scoring input.
    parts = {
        "claim": claim_f1(gen_s, gt_s, ctx, excluded_axes),
        "term": term_f1(gen_s, gt_s, ctx),
        "polarity": polarity_f1(gen_s, gt_s, ctx),
    }
    fr = fdi_report(gen_s, gt_s, ctx)
    gate = (1.0 - fr["invalid_rate"]) * (0.5 if fr["own_ok"] is False else 1.0)

    contrib = {k: w[k] / wsum * parts[k] * gate for k in TERM_KEYS}
    total = sum(contrib.values())
    out.update({
        "silence_ok": None,
        "claim_f1": parts["claim"], "term_f1": parts["term"],
        "polarity_f1": parts["polarity"],
        "fdi_gate": gate,
        "fdi_invalid_rate": fr["invalid_rate"],
        "fdi_own_ok": fr["own_ok"],
        "polarity_contradictions": polarity_contradictions(gen_s, gt_s),
        "n_claim_gen": len(extract_claims(gen_s, rid)),
        "n_claim_gt": len(extract_claims(gt_s, rid)),
        "contrib": contrib,
        "total": total,
    })
    return total, out






