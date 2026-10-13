"""Region-level rollouts: sample the policy, score the samples, pick the reference.

Rejection-sampling fine-tuning needs three things from the policy, and this module is all
three. One sample = **one region**: a patient is drawn, some of that patient's finding
regions and some normal regions are picked, and G completions are sampled per region.

Reward reference = the **union of all reports** of the patient
--------------------------------------------------------------
The final evaluation (RadFact) already scores against the union. If the reward instead looked
at the single report drawn for that epoch, **13.7%** of the normal slots in the training set
(multi-report patients are 59.6% of train) would in fact be finding slots of another report of
the same patient -> the rule "the target is empty, so speaking scores 0" would **suppress true
findings**, and the logs would only show "the reward rises but the clinical score does not".
`grpo.reward_gt=union` (the default) uses the union both as the scoring reference and for the
finding/normal split of the sample (`reward_targets`). `sampled` is kept for control runs only.
**The teacher-forcing targets are left alone.**

Notes
-----
- **Dropout is off during sampling** (projector 0.2, LoRA 0.1). With it on, the policy that
  produced the samples differs from the policy whose log-probabilities are computed. The
  `repetition_penalty` used at evaluation time is kept out of sampling for the same reason.
- Qwen3.5 has no `config.eos_token_id` -> `<|im_end|>` must be passed explicitly at generation
  time, otherwise generation runs away.
- The visual prefix is **constant** per (patient, region) because encoder and projector are
  frozen -> compute it once and reuse it for all G samples.
- Normal regions often have zero reward variance inside their group (56% measured); the
  selection step skips those and counts them.

The chat assembly and the generation path of `Realizer` are reused unchanged, so what is
sampled here is exactly what the supervised stage produced.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from ..config import GRPOConfig
from ..data import ReportDataset
from ..generation import region_type_id
from ..generation.llm_backend import _chat_segments
from ..generation.postprocess import clean_gen
from ..generation.realizer import region_instruction
from ..losses import axis_soft_probs
from ..models import ReportModel
from ..training.trainer import to_device
from .extract import extract_claims
from .presence import strip_presence
from .reward import region_reward


# `_chat_segments` renders the chat template and tokenises it; the result depends only on the
# instruction string, of which there are 38. Cache them.
_SEG_CACHE: dict[str, tuple] = {}


def _segments(realizer, instruction: str):
    """Compute `_chat_segments` (chat template rendering + tokenisation) once per instruction."""
    seg = _SEG_CACHE.get(instruction)
    if seg is None:
        seg = _chat_segments(realizer, instruction)
        _SEG_CACHE[instruction] = seg
    return seg

@torch.no_grad()
def build_prompt(realizer, tokens: torch.Tensor, region_id: str,
                 prob: torch.Tensor | None) -> torch.Tensor:
    """Build **the same prompt** as `generate_region`, but compute the prefix once and return it
    as a tensor, so that the G samples and the policy/reference forwards can share it.
    = assemble_chat_prompt(realizer, _region_prefix(...), region_instruction(rid))."""
    device = realizer.region_type_emb.weight.device
    dtype = realizer.llm.get_input_embeddings().weight.dtype
    vision = realizer._region_prefix(tokens, region_type_id(region_id), prob)   # (P,hidden)
    pre, mid, _end = (x.to(device) for x in _segments(realizer, region_instruction(region_id)))
    ep, em = (realizer._embed_text(x).to(dtype) for x in (pre, mid))
    return torch.cat([ep, vision.to(dtype), em], dim=0)                          # (L,hidden)

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def eos_token_ids(tok) -> tuple[list[int], int]:
    """(stop tokens, pad token). Qwen3.5 has no config.eos_token_id, so <|im_end|> must be
    named explicitly."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = {i for i in (im_end, tok.eos_token_id) if i is not None and i >= 0}
    if not ids:
        raise RuntimeError("no stop token found - generation would run away")
    pad = tok.pad_token_id if tok.pad_token_id is not None else im_end
    return sorted(ids), int(pad)

def _trim_completion(row: torch.Tensor, eos_ids: set[int], pad_id: int) -> torch.Tensor:
    """Generated row -> the completion **up to and including the first stop token**.

    Why the stop token has to be kept: on a normal region, silence *is* the act of emitting the
    turn terminator as the very first token. Trimming it away leaves a completion of length 0,
    so no gradient reaches that behaviour (it can neither be reinforced nor suppressed).
    """
    out: list[int] = []
    for t in row.tolist():
        out.append(int(t))
        if int(t) in eos_ids:
            break
    while len(out) > 1 and out[-1] == pad_id and pad_id not in eos_ids:
        out.pop()
    return torch.tensor(out, dtype=torch.long, device=row.device)

@torch.no_grad()
def sample_completions(model: ReportModel, prompt: torch.Tensor, n: int, gc: GRPOConfig,
                       eos_ids: list[int], pad_id: int, greedy: bool) -> list[torch.Tensor]:
    """One shared prompt -> n completions. `num_return_sequences` keeps the prompt prefill to a
    **single** pass."""
    llm = model.realizer.llm
    was_training = llm.training
    llm.eval()                                   # so gradient checkpointing keeps use_cache
    prev_cache = getattr(llm.config, "use_cache", None)
    try:
        llm.config.use_cache = True
        e = prompt.unsqueeze(0)
        attn = e.new_ones(1, e.shape[1], dtype=torch.long)
        kw = dict(max_new_tokens=gc.max_new_tokens, num_beams=1,
                  eos_token_id=list(eos_ids), pad_token_id=pad_id)
        if greedy:
            kw.update(do_sample=False, num_return_sequences=1)
        else:
            # repetition_penalty is **not** applied here: a sampling policy that differs from
            # the scoring policy breaks GRPO.
            kw.update(do_sample=True, temperature=gc.temperature, top_p=gc.top_p,
                      num_return_sequences=n)
        out = llm.generate(inputs_embeds=e, attention_mask=attn, **kw)
    finally:
        if prev_cache is not None:
            llm.config.use_cache = prev_cache
        llm.train(was_training)
    eset = set(eos_ids)
    return [_trim_completion(out[i], eset, pad_id) for i in range(out.shape[0])]

# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@dataclass
class RegionGroup:
    pid: str
    region_id: str
    is_finding: bool
    prompt: torch.Tensor
    comps: list[torch.Tensor]
    rewards: list[float]
    terms: list[dict]
    adv: torch.Tensor | None = None
    ref_logp: torch.Tensor | None = None
    ref_mask: torch.Tensor | None = None
    skipped: bool = False

def sample_region_ids(targets: dict[str, str], gc: GRPOConfig,
                      rng: random.Random) -> tuple[list[str], list[str]]:
    """Region list -> (finding regions, normal regions), at the fixed 8:4 ratio.

    If there are fewer than 8 finding regions, `preserve_ratio_when_short` shrinks the number of
    normal regions to keep the 2:1 ratio (feeding all 4 anyway returns to the regime where
    "always stay silent" wins).
    """
    find = sorted(r for r, t in targets.items() if (t or "").strip())
    norm = sorted(r for r, t in targets.items() if not (t or "").strip())
    rng.shuffle(find)
    rng.shuffle(norm)
    f = find[:gc.n_finding_regions]
    k = gc.n_normal_regions
    if gc.preserve_ratio_when_short and len(f) < gc.n_finding_regions and gc.n_finding_regions:
        ratio = gc.n_normal_regions / gc.n_finding_regions
        k = max(1, int(round(len(f) * ratio))) if f else gc.n_normal_regions
    return f, norm[:k]

REWARD_GT_MODES = ("union", "sampled")

def reward_targets(batch: dict, gc: GRPOConfig,
                   union_targets: dict[str, dict[str, str]] | None) -> dict[str, str]:
    """Reference dict used for scoring the reward **and** for the finding/normal region split.

    union (default) — the **union of all reports** of that patient
      (`ReportDataset.union_targets_by_pid`). The final evaluation (RadFact) already scores
      against the union, so reward and evaluation share one reference. A slot that carries a
      finding in only one report now counts as a finding region rather than a normal one, which
      changes the 8:4 sampling as well.
    sampled         — the older behaviour (the single report drawn for that epoch). For control
      runs.

    ⚠️ Only the **reward reference** is selected here. The teacher-forcing targets
    (`batch["region_targets"]`) are left untouched (GRPO does not use them anyway, but the
    dataset object is shared with the other training paths).
    """
    if gc.reward_gt not in REWARD_GT_MODES:
        raise ValueError(f"grpo.reward_gt must be one of {REWARD_GT_MODES}, got {gc.reward_gt!r}")
    if gc.reward_gt == "sampled":
        return _clean_targets(batch["region_targets_text"][0], gc)
    pid = batch["pid"][0]
    if not union_targets:
        raise RuntimeError("reward_gt=union but no union_targets was given - build "
                           "ReportDataset.union_targets_by_pid() and pass it in")
    if pid not in union_targets:
        raise KeyError(f"patient {pid} is missing from the union GT "
                       f"(dataset and union map come from different sources)")
    return _clean_targets(union_targets[pid], gc)

def _clean_targets(targets: dict[str, str], gc: GRPOConfig) -> dict[str, str]:
    """Pre-process the scoring reference. Two things happen.

    (1) **Neutralise tooth-presence statements** — a region whose text only said
        `Tooth 31 present` becomes an empty string and is treated as a normal region
        (silence = 1.0). The point is that stating presence or not must score the same.
        ⚠️ Arch-level presence statements are deliberately not affected.
    (2) **Drop regions with no claim** — if the rule-based extractor still produces no claim
        after the presence statements are removed, the region is **removed from the dict** and
        thus from the sampling pool: there the claim F1 would be empty vs empty = 1.0, i.e.
        "say anything at all, and as long as it yields no claim you score full marks".
        ⚠️ Side effect: on the non-teeth side, claim-less regions are mostly field-of-view and
        negated-finding statements (which the extractor does not cover), so those regions are
        never learned in RL. Keeping them intact relies on the KL penalty.
    """
    out: dict[str, str] = {}
    for rid, txt in targets.items():
        t = strip_presence(txt) if gc.reward_neutralize_presence else (txt or "")
        if not t.strip():
            out[rid] = ""                                  # normal region (silence is correct)
            continue
        if gc.reward_drop_claimless and not extract_claims(t, rid):
            continue                                       # not scored (dropped from the pool)
        out[rid] = t
    return out

def rollout(model: ReportModel, batch: dict, gc: GRPOConfig, rng: random.Random,
            eos_ids: list[int], pad_id: int, greedy: bool = False,
            union_targets: dict[str, dict[str, str]] | None = None) -> list[RegionGroup]:
    """One patient -> one group per selected region (G samples + their rewards)."""
    pid = batch["pid"][0]
    targets: dict[str, str] = reward_targets(batch, gc, union_targets)
    find, norm = sample_region_ids(targets, gc, rng)
    with torch.no_grad():
        ctx, aux = model._encode(batch)
    soft = model._soft_on()
    prob_i = (axis_soft_probs(aux[0], model._logit_slices)
              if (soft and aux is not None) else None)
    tok = model.realizer.tokenizer
    n_sample = 1 if greedy else gc.group_size

    groups: list[RegionGroup] = []
    for rid in find + norm:
        gt = (targets.get(rid) or "").strip()
        tokens, prob = model._slot_tokens(ctx[0], rid, prob_i, soft)
        prompt = build_prompt(model.realizer, tokens, rid, prob)
        comps = sample_completions(model, prompt, n_sample, gc, eos_ids, pad_id, greedy)
        comps = [c for c in comps if int(c.numel()) > 0]
        if not comps:
            continue
        rewards, terms = [], []
        for c in comps:
            txt = clean_gen(tok.decode(c, skip_special_tokens=True))
            r, d = region_reward(txt, gt, {"pid": pid, "region_id": rid},
                                 neutralize_presence=gc.reward_neutralize_presence)
            rewards.append(float(r))
            terms.append(d)
        groups.append(RegionGroup(pid=pid, region_id=rid, is_finding=bool(gt),
                                  prompt=prompt, comps=comps, rewards=rewards, terms=terms))
    return groups

# --------------------------------------------------------------------------- #
# Validation reward (stopping criterion) — greedy, G=1, so only policy quality is measured
# and sampling noise does not enter
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_reward(model: ReportModel, ds: ReportDataset, collate, gc: GRPOConfig,
                    device, eos_ids: list[int], pad_id: int, n_patients: int,
                    union_targets: dict[str, dict[str, str]] | None = None) -> dict:
    order = _epoch_order(len(ds), 0, gc.seed)[:max(1, min(n_patients, len(ds)))]
    rows: list[dict] = []
    for idx in order:
        batch = to_device(collate([ds[idx]]), device)
        rng = random.Random(f"val:{gc.seed}:{idx}")      # same regions for every model compared
        groups = rollout(model, batch, gc, rng, eos_ids, pad_id, greedy=True,
                         union_targets=union_targets)
        for g in groups:
            rows.append({"is_finding": g.is_finding, "r": g.rewards[0], "d": g.terms[0]})
    if not rows:
        return {"val_reward": 0.0}
    fin = [r["r"] for r in rows if r["is_finding"]]
    nor = [r["r"] for r in rows if not r["is_finding"]]
    # ⚠️ `val_claim_f1` (averaged over all regions) is **not a pure claim F1**: on normal
    # regions `region_reward` fills all three terms with a binary value (silence 1.0 /
    # speaking 0.0), so silence accuracy is mixed in. Interpreting it needs the per-term values
    # over **finding regions only**, which are therefore reported alongside.
    fr = [r["d"] for r in rows if r["is_finding"]]
    def _fin_mean(key: str) -> float:
        v = [float(d[key]) for d in fr if d.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")
    return {
        "val_reward": float(np.mean([r["r"] for r in rows])),
        "val_reward_finding": float(np.mean(fin)) if fin else float("nan"),
        "val_reward_normal": float(np.mean(nor)) if nor else float("nan"),
        "val_claim_f1": float(np.mean([float(r["d"]["claim_f1"]) for r in rows])),
        # --- finding regions only (per-term breakdown) --------------------- #
        "val_claim_f1_finding": _fin_mean("claim_f1"),
        "val_term_f1_finding": _fin_mean("term_f1"),
        "val_polarity_f1_finding": _fin_mean("polarity_f1"),
        "val_fdi_gate_finding": _fin_mean("fdi_gate"),
        "val_n_regions": float(len(rows)),
        "val_n_finding": float(len(fin)),
    }

# --------------------------------------------------------------------------- #
# Data order — the step -> patient mapping is **deterministic** (a resumed run sees the same
# order)
# --------------------------------------------------------------------------- #
def _epoch_order(n: int, epoch: int, seed: int) -> list[int]:
    r = random.Random(f"{seed}:{epoch}")
    idx = list(range(n))
    r.shuffle(idx)
    return idx
