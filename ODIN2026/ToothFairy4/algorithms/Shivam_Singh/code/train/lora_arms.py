#!/usr/bin/env python3
"""The LoRA arm table, and the assertion that a config actually hit its targets.

Single source of truth for `docs/vision_sft_plan.md` §2. `trainer.py`
imports ARMS from here and calls assert_arm() before it touches a GPU, so the
declared parameter count is checked in exactly one place.

WHY A COUNT IS ASSERTED AT ALL. The Qwen3.5 checkpoint names its submodules
`model.visual.*` and `model.language_model.*`. Qwen2-VL -- which every LoRA
recipe on the internet was written against -- names them `visual.*` and
`model.layers.*`. A copied `target_modules` therefore matches NOTHING, and
`get_peft_model()` reports 0 trainable parameters *without raising*: the job
starts, burns its wall-clock, writes an adapter, and the adapter is empty.
The count is the only cheap check that catches it.

The reverse trap is just as quiet. The checkpoint also carries an `mtp`
(multi-token-prediction) head whose submodules are ALSO called
`self_attn.{q,k,v,o}_proj`, so the obvious language regex
`.*self_attn\\.[qkvo]_proj` silently trains 9 blocks where arm 3 declares 8,
and the 2-vs-3 parameter match the whole experiment rests on is gone. Every
pattern here is anchored at `^` and matched with re.fullmatch by peft.

Counts are arithmetic from the safetensors shapes, not measurements:

  vision block x27  qkv [3456,1152] proj [1152,1152]
                    fc1 [4304,1152] fc2 [1152,4304]      -> 481,248 * r
  merger x1         fc1 [4304,4304]? -- see below         ->  17,920 * r
  language x8       q,k,v,o over hidden 2048 / kv 512     -> 245,760 * r
  gdn x24           in_proj_qkv [8192,4096] in_proj_z and
                    out_proj [4096,4096] each             -> 688,128 * r

A LoRA pair on [out,in] adds r*(in+out) parameters, so each figure above is
sum(in+out) over that group's matrices.

WHY NO ARM TOUCHES mlp.{down,up,gate}_proj, AND WHY THAT IS THE WHOLE FENCE.
Those 93 modules (layers 1-31) are the ONLY ones AWQ quantizes; its skip list
is ["visual", "linear_attn", "self_attn", "model.layers.0.", "mtp"], so every
group in the table above is bf16 and bit-identical between models/Qwen3.5-9B
and models/Qwen3.5-9B-AWQ. That identity is what lets an adapter trained here
merge into the deployed checkpoint exactly (merge_meta.json records it as
`bases_bit_identical`). A LoRA on the MLPs would be a delta fitted to
pre-quantisation weights with nothing to merge it into, so the fence is not a
matter of taste: train anything on the bf16 side of that list, nothing on the
other.

Run it standalone to settle the peft-x-transformers-5.9.0 question with no GPU:

    python code/train/lora_arms.py --model models/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import sys

# ── The arms ──────────────────────────────────────────────────────────────
#
# `expect` is what print_trainable_parameters() must report. `--rank` is NOT
# free: changing r changes `expect` linearly, so the CLI recomputes it from
# `per_r` rather than letting a rank override silently disable the check.

ARMS = {
    # Arm 1 -- projector only. The two-matrix interface between the vision
    # tower and the LM. If this alone moves the result, the features were
    # already extracted and only the hand-off was wrong.
    "merger": {
        "pattern": r"^model\.visual\.merger\.linear_fc[12]$",
        "r": 16,
        "per_r": 17_920,
        "n_modules": 2,
    },
    # Arm 2 -- the primary arm. Whole vision tower + projector.
    "vision+merger": {
        "pattern": (r"^model\.visual\."
                    r"(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                    r"|merger\.linear_fc[12])$"),
        "r": 16,
        "per_r": 481_248 + 17_920,
        "n_modules": 27 * 4 + 2,
    },
    # Arm 3 -- the control. Same recipe, same step count, ~1.5% fewer
    # trainable parameters, different LOCATION. Only the 8 full_attention
    # layers own a self_attn; the other 24 are Gated DeltaNet. `mtp` is
    # excluded by the `model.language_model.` prefix.
    "language": {
        "pattern": r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$",
        "r": 32,
        "per_r": 245_760,
        "n_modules": 8 * 4,
    },
    # Arm 5 -- everything AWQ leaves in bf16. Added 2026-08-14 because arms 2
    # and 3 between them cover 8 of the language model's 32 layers: only the
    # full_attention layers own a `self_attn`, and the 24 Gated DeltaNet layers
    # were in no arm at all. This adds their in_proj_qkv / in_proj_z / out_proj.
    #
    # in_proj_a and in_proj_b are DELIBERATELY LEFT OUT. They are [32, 4096] --
    # GDN's gating projections -- and a rank-16 adapter on a 32-row output is
    # very nearly a full reparametrisation of the matrix rather than a low-rank
    # correction to it. They would add 8,256 * r for that.
    #
    # This is not a replacement for arm 2, it is the arm to SHIP: it costs the
    # same wall-clock (step time is dominated by the frozen base's forward and
    # backward, not by 23M vs 8M adapter parameters) and it puts trainable
    # weight next to the loss instead of 32 frozen layers away from it. What it
    # gives up is the arm-2-vs-arm-3 contrast -- where the gain LIVES -- which
    # is an ablation, not a submission.
    #
    # `mtp` carries its own `layers.N.self_attn.{q,k,v,o}_proj` and no
    # linear_attn; the `^model\.language_model\.layers\.` anchor excludes it,
    # exactly as it does for arm 3.
    "vision+language": {
        "pattern": (r"^(model\.visual\."
                    r"(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                    r"|merger\.linear_fc[12])"
                    r"|model\.language_model\.layers\.\d+\."
                    r"(self_attn\.[qkvo]_proj"
                    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)))$"),
        "r": 16,
        "per_r": 481_248 + 17_920 + 245_760 + 688_128,
        "n_modules": 27 * 4 + 2 + 8 * 4 + 24 * 3,
    },
    # Arm 4 -- capacity check, contingent on arm 2.
    "vision+merger-r64": {
        "pattern": (r"^model\.visual\."
                    r"(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                    r"|merger\.linear_fc[12])$"),
        "r": 64,
        "per_r": 481_248 + 17_920,
        "n_modules": 27 * 4 + 2,
    },
}


def expected_params(arm: str, r: int | None = None) -> int:
    """Trainable parameters for `arm` at rank `r` (its own rank by default)."""
    spec = ARMS[arm]
    return spec["per_r"] * (spec["r"] if r is None else r)


def lora_config(arm: str, r: int | None = None, alpha_mult: int = 2,
                dropout: float = 0.05):
    """The arm's LoraConfig. `lora_alpha = 2r` per §4, for every arm."""
    from peft import LoraConfig

    spec = ARMS[arm]
    rank = spec["r"] if r is None else r
    return LoraConfig(
        r=rank,
        lora_alpha=alpha_mult * rank,
        lora_dropout=dropout,
        bias="none",
        target_modules=spec["pattern"],
        task_type=None,          # not CAUSAL_LM: we drive the loss ourselves
    )


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_arm(model, arm: str, r: int | None = None) -> int:
    """Raise unless the applied adapter has exactly the declared parameters.

    Called by trainer.py after get_peft_model() and before the
    first forward pass. A mismatch here is never a rounding difference -- it
    means the regex hit a different set of modules than the plan costed, so
    the arm is not the arm.
    """
    got = count_trainable(model)
    want = expected_params(arm, r)
    if got != want:
        raise SystemExit(
            f"[FAIL] arm {arm!r}: trainable parameters {got:,} != declared "
            f"{want:,}.\n"
            f"       0 means the target pattern matched nothing (the "
            f"Qwen2-VL naming trap).\n"
            f"       A larger number usually means `mtp` or the Gated "
            f"DeltaNet layers were swept in.\n"
            f"       pattern: {ARMS[arm]['pattern']}")
    return got


# ── Standalone check: meta device, no GPU, no weights read ────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen3.5-9B",
                    help="checkpoint dir -- only config.json is read")
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    help="which arms to check (default: all)")
    args = ap.parse_args()

    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(args.model)
    print(f"[INFO] {args.model}: {cfg.model_type} "
          f"{getattr(cfg, 'architectures', None)}")

    # Meta device: the module TREE is built, no tensor is allocated, no
    # safetensors shard is read. Seconds, and no GPU.
    with init_empty_weights():
        base = AutoModelForImageTextToText.from_config(cfg)

    total = sum(p.numel() for p in base.parameters())
    print(f"[INFO] base parameters: {total:,}")

    ok = True
    for arm in args.arms:
        with init_empty_weights():
            model = get_peft_model(AutoModelForImageTextToText.from_config(cfg),
                                   lora_config(arm))
        got, want = count_trainable(model), expected_params(arm)
        hit = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
        mark = "PASS" if got == want else "FAIL"
        ok &= got == want
        print(f"[{mark}] {arm:<18} r={ARMS[arm]['r']:<3} "
              f"trainable {got:>12,}  declared {want:>12,}  "
              f"modules hit {hit}/{ARMS[arm]['n_modules']}  "
              f"({100 * got / total:.3f}% of base)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
