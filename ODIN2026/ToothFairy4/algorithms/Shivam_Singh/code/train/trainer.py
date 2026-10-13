#!/usr/bin/env python3
"""
trainer.py -- the arms of docs/vision_sft_plan.md §2, trained.

    python code/train/trainer.py --arm vision+merger --rows sft_targets.jsonl \\
        --out outputs/vsft_arm2 --epochs 2

Four things this file is responsible for, and three of them are refusals:

  1. **It refuses to start unless the adapter is the arm.** `lora_arms.assert_arm()`
     runs after get_peft_model() and before the first forward. A copied
     Qwen2-VL `target_modules` matches nothing here and peft reports 0 trainable
     parameters WITHOUT raising -- the job would run its full wall-clock and
     write an empty adapter. Arms 2 and 3 differ by 1.5% in trainable
     parameters and that contrast IS the experiment, so an unnoticed extra
     block is as damaging as an empty adapter.
  2. **It refuses to truncate.** A call over --max-length is dropped and
     counted (§4). Truncation removes the END of the target, and the target
     is emitted in schema order, so it would train a systematically incomplete
     answer.
  3. **It refuses to train a base whose gradients do not reach the arm.** With
     an AWQ checkpoint that is open item 3, and --probe-backward answers it in
     ~2 minutes rather than at the end of an 8-hour job: one forward, one
     backward, then look for a non-zero gradient on the first LoRA A matrix
     the arm targets.
  4. It drives the loss itself. See loss_on_batch().

WHY THE LOSS IS NOT THE MODEL'S OWN
───────────────────────────────────
Passing `labels=` to forward() computes cross-entropy over EVERY position
against a 248,320-token vocabulary: 6,700 x 248,320 logits, ~9 GB with the
fp32 upcast and its gradient, which is what OOMs a 40 GB card (the plan's memory-budget log entry) -- the
weights are not the problem, the logits are.

Every supervised position is in the target span at the END of the sequence, so
`logits_to_keep=n_target+1` slices the hidden states before `lm_head` and the
logits tensor becomes ~600 x 248,320, about 0.9 GB. That is the difference
between fitting one A100 and not.

It also lets the per-token weights of sft_collator.py apply at all: HF's
built-in loss is a plain mean over non-ignored positions, and the evidence README's 0.04
evidence weight cannot be expressed that way.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path


# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

ROOT = REPO_ROOT

import lora_arms                          # noqa: E402
import sft_collator as SC                 # noqa: E402
import sft_prompt as SP                   # noqa: E402
# RowDataset and load_rows live in dataset.py: which rows exist, and when
# they are encoded, is a separate question from how a step is taken.
from dataset import RowDataset, load_rows  # noqa: E402,F401


def loss_on_batch(model, inputs: dict):
    """Weighted cross-entropy over the target span only.

    Returns (loss, n_supervised). The normalisation is by WEIGHT, not by token
    count, so a row whose evidence survived the screen does not get a larger
    step than one whose evidence did not -- the weights decide loss mass, and
    the row count decides how often each row is seen.
    """
    import torch
    import torch.nn.functional as F

    inputs = dict(inputs)
    labels = inputs.pop("labels")
    weights = inputs.pop("weights")
    n_target = int(inputs.pop("n_target"))
    inputs.pop("n_prompt", None)

    out = model(**inputs, logits_to_keep=n_target + 1)
    # logits_to_keep=K returns the last K positions, L-K .. L-1. With
    # K = n_target+1 the first of those is the position BEFORE the target, so
    # logits[:-1] are exactly the predictions of the n_target target tokens.
    logits = out.logits[:, :-1, :]
    if logits.shape[1] != labels.shape[1]:
        raise RuntimeError(
            f"alignment: {logits.shape[1]} logits against {labels.shape[1]} "
            f"labels -- logits_to_keep did not slice what this assumes")

    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1), reduction="none", ignore_index=SC.IGNORE)
    w = weights.reshape(-1).to(per_token.dtype)
    total = w.sum().clamp(min=1e-6)
    return (per_token * w).sum() / total, int((labels != SC.IGNORE).sum())


def build_trainer_class():
    """Trainer with loss_on_batch(), built late so --help needs no torch."""
    from transformers import Trainer

    class LoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            loss, _ = loss_on_batch(model, inputs)
            return (loss, None) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only,
                            ignore_keys=None):
            """Eval loss through the SAME masked path as training.

            Overridden rather than inherited because the base implementation
            reads `labels` out of the batch and scores whatever the model
            returns against them. That is not the loss this arm trains on:
            loss_on_batch applies the §3.5 masking -- unstated, refused, gated
            and capped-negative positions are excluded -- so an inherited eval
            would report a different quantity under the same name, and the
            train/eval comparison the split exists for would be meaningless.
            """
            import torch as _t

            with _t.no_grad():
                loss, _ = loss_on_batch(model, self._prepare_inputs(inputs))
            return (loss.detach(), None, None)

    return LoRATrainer



def quantized_modules_in_checkpoint(model_dir: Path) -> int:
    """How many modules the checkpoint ACTUALLY stores 4-bit, from its headers."""
    from safetensors import safe_open

    n = 0
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:              # headers only, not weights
            n += sum(1 for k in h.keys() if k.endswith(".qweight"))
    return n


def awq_config_for_transformers(model_dir: Path, backend: str = "auto_trainable"):
    """Translate the checkpoint's `modules_to_not_convert` into transformers'
    matching semantics, and pin a kernel that can actually train. Returns
    (config, n_patterns_rewritten).

    THE KERNEL IS A CORRECTNESS SETTING, NOT A SPEED ONE.
    ─────────────────────────────────────────────────────
    Backend selection is per-device, so it differs between the login node and
    the GPU node. On the A100 the default `auto` picked `AwqMarlinLinear`,
    whose class attribute reads `SUPPORTS_TRAINING = False` -- an inference-only
    fused kernel with no backward. It happened to fail loudly, at post_init,
    because Marlin JIT-compiles CUDA C++ and there is no nvcc here. That crash
    was lucky: the quiet version of this is a kernel that runs and returns no
    gradient, which is indistinguishable from "AWQ cannot be trained" and would
    have been read as the answer to open item 3.

    `auto_trainable` restricts the candidates to kernels with SUPPORTS_TRAINING
    True. `torch_awq` (AwqTorchLinear: dequantize, then matmul, pure PyTorch)
    is the deterministic fallback -- slower, but nothing to compile, so it
    cannot fail this way. Whatever is chosen is printed after the load, because
    it changes the step time stage B exists to measure.

    THE TWO LIBRARIES READ THE SAME FIELD DIFFERENTLY, AND ONLY ONE SAYS SO.
    ────────────────────────────────────────────────────────────────────────
    vLLM matches each pattern as a SUBSTRING of the module name. transformers
    matches prefix-or-exact-or-suffix:

        re.match(f"{key}\\.", name) or re.match(key, name) or name.endswith(key)

    Qwen3.5-9B-AWQ ships `["visual", "linear_attn", "self_attn",
    "model.layers.0.", "mtp"]`, written for vLLM -- which serves this model
    every day, so the field is not wrong, it is just not portable. Under
    transformers' rule NONE of the five fire: "visual" is neither a prefix of
    `model.visual.blocks.0.mlp.linear_fc1` (that starts "model.") nor its
    suffix (that ends "linear_fc1"); "self_attn" likewise loses to "q_proj".

    The failure is not a quiet one but it is a MISLEADING one: transformers
    tries to 4-bit all 287 linear modules, and the first thing to complain is
    Marlin's `out_features 4304 must be divisible by 64` -- 4304 being the
    VISION tower's intermediate_size, a module that was never meant to be
    touched. Read at face value that error suggests an incompatible kernel and
    invites a backend override, which would fix nothing: the 194 extra modules
    store plain `.weight`, and converting them to a quant linear that expects
    `.qweight` breaks the load however it is packed.

    Wrapping each pattern as `.*p.*` reproduces vLLM's substring rule exactly
    (transformers feeds the pattern to `re.match`, so a regex is legal here).
    Verified against the real module names: as shipped, 287 converted, 194 of
    them wrong; wrapped, exactly the 93 the checkpoint stores 4-bit, with no
    error in either direction. Those 93 are all `mlp.{down,up,gate}_proj` in
    layers 1-31 -- and NO arm targets one of them, which is why the arms train
    unquantized tensors and merge back bit-exactly (plan log, 'AWQ quantizes almost nothing').
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    qc = getattr(cfg, "quantization_config", None)
    if not qc:
        return cfg, 0
    is_dict = isinstance(qc, dict)
    raw = list((qc.get("modules_to_not_convert") if is_dict
                else getattr(qc, "modules_to_not_convert", None)) or [])
    if not raw:
        return cfg, 0
    # Idempotent: a pattern already wrapped is left alone, so re-running against
    # a config we rewrote once does not nest `.*.*p.*.*`.
    wrapped = [p if p.startswith(".*") else f".*{re.escape(p)}.*" for p in raw]
    if is_dict:
        qc["modules_to_not_convert"] = wrapped
        qc["backend"] = backend
    else:
        qc.modules_to_not_convert = wrapped
        qc.backend = backend
    return cfg, len(wrapped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(lora_arms.ARMS))
    ap.add_argument("--rank", type=int, default=None,
                    help="override the arm's rank (the expected parameter "
                         "count scales with it; the assertion still holds)")
    ap.add_argument("--rows", type=Path,
                    default=ROOT / "outputs/training_results/vsft_pool_training/sft_targets.jsonl")
    ap.add_argument("--case-list", type=Path,
                    help="restrict to these cases -- "
                         "outputs/training_results/sft_pool/narrow.txt for "
                         "arm 1, omit for the wide pool")
    ap.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-9B-AWQ")
    ap.add_argument("--awq-backend", default="auto_trainable",
                    help="AWQ kernel selection. NOT a tuning knob -- the "
                         "default 'auto' picks per device and chose "
                         "AwqMarlinLinear on the A100, whose SUPPORTS_TRAINING "
                         "is False: an inference-only fused kernel with no "
                         "backward. 'auto_trainable' restricts the choice to "
                         "kernels that can train. Deterministic fallback if a "
                         "kernel fails to build: 'torch_awq' (dequantize + "
                         "matmul, pure PyTorch, nothing to compile).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--evidence-weight", type=float, default=0.04)
    ap.add_argument("--eval-case-list", type=Path,
                    help="watch eval loss on THESE cases -- normally "
                         "sft_pool/heldout.txt, the same 24 the arms are "
                         "compared on afterwards. Their rows are read straight "
                         "from --rows, bypassing --case-list, so the 30 cases "
                         "--eval-cases would otherwise reserve stay in "
                         "training. Mutually exclusive with --eval-cases.")
    ap.add_argument("--eval-cases", type=int, default=0,
                    help="hold N cases out of the ROWS as an eval split and "
                         "report eval loss twice an epoch. 0 (default) trains "
                         "blind, which is what every arm so far has done -- and "
                         "why the epoch count is still a guess. Arm 2's second "
                         "epoch looked flat, but its LR had annealed to 4e-08 "
                         "by then, so 'the loss stopped moving' and 'the "
                         "schedule stopped letting it move' were the same fact "
                         "and the run could not tell them apart. This is the "
                         "cheapest way to separate them. Split is BY CASE, "
                         "never by row: rows from one case share the same "
                         "images, so a row-wise split leaks.")
    ap.add_argument("--limit", type=int, help="train on N rows (stage B)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe-backward", action="store_true",
                    help="one forward+backward, report the arm's gradient, and "
                         "STOP. Open item 3 / R3 for the AWQ checkpoint.")
    args = ap.parse_args()

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_rows(args.rows, args.case_list, args.limit, args.seed)
    if not rows:
        sys.exit("[FAIL] no rows")
    cases = {r["case_id"] for r in rows}
    print(f"[INFO] {len(rows)} row(s), {len(cases)} case(s), arm {args.arm!r}")

    # §0, again and at the last possible moment. build_sft_targets.py refuses a
    # validate case at build time; this refuses one that reached a rows file by
    # any other route, because the cost of finding out later is the experiment.
    forbidden = {p.name.split("_0000")[0]
                 for p in (ROOT / "dataset/validate/images").glob("*_0000.nii.gz")}
    leaked = cases & forbidden
    if leaked:
        sys.exit(f"[FAIL] §0: {len(leaked)} validate case(s) in the training "
                 f"rows: {sorted(leaked)[:8]}")
    print(f"[PASS] §0: no validate case in the pool ({len(forbidden)} checked)")

    # §0's SECOND bullet -- the held-out scoring cases, which the check above
    # does not cover and by construction cannot: they are training-split cases,
    # so nothing about them looks forbidden. outputs/training_results/sft_pool/heldout.txt is the set
    # every arm is measured on (arm 1's 171-claims number, arm 2's, and whatever
    # arm 5 returns), and all 24 of them are inside all_582.txt. Train on the full
    # pool and 18 of them arrive with drafted evidence -- the arm is then
    # scored on cases it memorised, and the comparison it exists to support
    # quietly stops meaning anything. Nothing downstream would report it.
    #
    # ALLOW_HELDOUT=1 for the deliberate case (a final model trained on
    # everything once the arms have been chosen), because that IS a thing one
    # eventually wants -- but it should cost a decision, not a default.
    # A MISSING heldout.txt IS A FAILURE, NOT A SKIP. It used to be `if
    # exists()`, and on 2026-08-16 the pool moved under
    # outputs/training_results/ -- so the guard stopped running, silently, with
    # no line in the log either way. The rows file is not itself safe: 22 of the
    # 24 held-out cases have rows in it, and only --case-list keeps them out.
    # A check that disappears when its input moves is worse than no check,
    # because the [PASS] line's absence is the only signal and nobody reads for
    # a line that is not there.
    ho_file = Path(os.environ.get("HELDOUT_FILE")
                   or ROOT / "outputs/training_results/sft_pool/heldout.txt")
    if not ho_file.exists():
        if not os.environ.get("ALLOW_HELDOUT"):
            sys.exit(f"[FAIL] §0 held-out: {ho_file} not found -- cannot prove the "
                     f"held-out scoring cases stayed out of training. Point "
                     f"HELDOUT_FILE at it, or set ALLOW_HELDOUT=1 to train "
                     f"without the guard.")
        print(f"[WARN] §0 held-out: {ho_file} not found and ALLOW_HELDOUT set -- "
              f"held-out containment is UNVERIFIED for this run")
        heldout = set()
    else:
        heldout = {l.strip() for l in ho_file.read_text().splitlines()
                   if l.strip() and not l.startswith("#")}
    bleed = cases & heldout
    if bleed and not os.environ.get("ALLOW_HELDOUT"):
        sys.exit(
            f"[FAIL] §0 held-out: {len(bleed)} held-out scoring case(s) in the "
            f"training rows: {sorted(bleed)[:8]}\n"
            f"       Those are the cases every arm is compared on. Pass a "
            f"--case-list that excludes {ho_file}, or set ALLOW_HELDOUT=1 "
            f"if this run is deliberately trained on everything.")
    if bleed:
        print(f"[WARN] §0 held-out: training on {len(bleed)} held-out case(s) "
              f"(ALLOW_HELDOUT set) -- held-out scores are now invalid")
    else:
        print(f"[PASS] §0 held-out: no held-out scoring case in the pool "
              f"({len(heldout)} checked)")

    # The eval split, carved from the ROWS rather than from the case list, so
    # it can only ever contain cases that actually produced targets. It is a
    # SECOND reserved set, disjoint from heldout.txt above: that one exists to
    # compare arms after training, this one to watch generalisation during it.
    eval_rows: list[dict] = []
    if args.eval_case_list and args.eval_cases:
        sys.exit("[FAIL] --eval-case-list and --eval-cases are mutually "
                 "exclusive: one names the eval cases, the other samples them.")
    if args.eval_case_list:
        # Read from --rows directly. `rows` here is already filtered by
        # --case-list, which EXCLUDES these cases by construction -- that is
        # the whole point of the held-out file -- so they have to be loaded a
        # second time rather than partitioned out of what is in hand.
        #
        # This runs AFTER the §0 held-out guard above, and must: that guard
        # refuses a run whose TRAINING rows contain a held-out case, and these
        # rows never enter `rows`. Watching a case's loss is not training on it.
        want = {l.strip() for l in args.eval_case_list.read_text().splitlines()
                if l.strip() and not l.startswith("#")}
        eval_rows = [r for r in load_rows(args.rows, None, None, args.seed)
                     if r["case_id"] in want]
        if not eval_rows:
            sys.exit(f"[FAIL] --eval-case-list {args.eval_case_list} matched no "
                     f"row in {args.rows}")
        got = {r["case_id"] for r in eval_rows}
        print(f"[INFO] eval split: {len(got)} case(s) / {len(eval_rows)} row(s) "
              f"from {args.eval_case_list.name}; all {len(rows)} training "
              f"row(s) over {len(cases)} case(s) kept")
        if got & cases:
            sys.exit(f"[FAIL] {len(got & cases)} eval case(s) are also in the "
                     f"training rows: {sorted(got & cases)[:8]}")
    elif args.eval_cases:
        import random as _random
        pool = sorted({r["case_id"] for r in rows})
        n = min(args.eval_cases, max(0, len(pool) - 1))
        chosen = set(_random.Random(args.seed).sample(pool, n))
        eval_rows = [r for r in rows if r["case_id"] in chosen]
        rows = [r for r in rows if r["case_id"] not in chosen]
        cases = {r["case_id"] for r in rows}
        print(f"[INFO] eval split: {len(chosen)} case(s) / {len(eval_rows)} row(s) "
              f"held out of training; {len(rows)} row(s) over {len(cases)} case(s) left")
        if not rows:
            sys.exit("[FAIL] --eval-cases took every case")

    proc = AutoProcessor.from_pretrained(str(args.model))
    collator = SC.ToothCallCollator(proc, args.evidence_weight, args.max_length)
    dataset = RowDataset(rows, collator)
    eval_dataset = RowDataset(eval_rows, collator) if eval_rows else None

    print(f"[INFO] loading {args.model.name} ...")
    cfg, n_pat = awq_config_for_transformers(args.model, args.awq_backend)
    if n_pat:
        print(f"[INFO] AWQ: rewrote {n_pat} skip pattern(s), backend "
              f"{args.awq_backend!r}")
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), config=cfg, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True)

    # The post-condition for the rewrite above, and cheap. A skip list that
    # silently stops matching -- a renamed submodule, a different checkpoint,
    # a transformers release that changes the rule again -- shows up here as a
    # count mismatch instead of as a wrong number at the end of stage D.
    expected = quantized_modules_in_checkpoint(args.model)
    if expected:
        qmods = [m for m in model.modules() if hasattr(m, "qweight")]
        got = len(qmods)
        if got != expected:
            sys.exit(f"[FAIL] AWQ conversion: {got} quantized module(s) built "
                     f"but the checkpoint stores {expected}. The skip patterns "
                     f"are not matching what they did when this was written.")
        kern = type(qmods[0]).__name__
        trains = getattr(type(qmods[0]), "SUPPORTS_TRAINING", None)
        print(f"[PASS] AWQ: {got} quantized module(s) as {kern}, "
              f"SUPPORTS_TRAINING={trains}")
        # The kernel is chosen per device, so this is checked here rather than
        # trusted from the login node. A kernel with no backward returns no
        # gradient and would be misread as R3's answer.
        if trains is False:
            sys.exit(f"[FAIL] {kern} cannot train (SUPPORTS_TRAINING=False). "
                     f"Re-run with --awq-backend torch_awq.")

    quant = getattr(model, "hf_quantizer", None)
    if quant is not None:
        trainable = getattr(quant, "is_trainable", None)
        print(f"[INFO] quantizer {type(quant).__name__}: is_trainable={trainable}")
        if trainable is False:
            print("[WARN] transformers marks this checkpoint untrainable "
                  "(gptqmodel>=5.0.0 missing -- plan log, 'Why training never loads AWQ'). None of the arms touch a "
                  "quantized weight, so --probe-backward is what settles "
                  "whether it matters here.")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, lora_arms.lora_config(args.arm, args.rank))
    got = lora_arms.assert_arm(model, args.arm, args.rank)
    print(f"[PASS] arm {args.arm!r}: {got:,} trainable parameter(s)")

    if args.probe_backward:
        return probe_backward(model, dataset, collator, args)

    from transformers import TrainingArguments
    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,          # see ToothCallCollator.__call__
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        # Twice an epoch. Once is too coarse to separate "still improving" from
        # "the schedule ran out"; every 5 steps like training logging would pay
        # a full pass over the eval split 135 times.
        **({"eval_strategy": "steps",
            "eval_steps": max(1, int(len(rows) / args.grad_accum / 2)),
            "per_device_eval_batch_size": 1}
           if eval_dataset is not None else {}),
        report_to=[],
        remove_unused_columns=False,            # n_target is ours, not the model's
        label_names=["labels"],
        dataloader_num_workers=args.workers,
        seed=args.seed,
        gradient_checkpointing=False,           # already enabled on the base
    )
    trainer = build_trainer_class()(
        model=model, args=targs, train_dataset=dataset,
        eval_dataset=eval_dataset, data_collator=collator)

    print(f"[INFO] {len(rows)} row(s) x {args.epochs} epoch(s) / "
          f"{args.grad_accum} = ~{int(len(rows) * args.epochs / args.grad_accum)} "
          f"optimizer step(s)")
    trainer.train()

    # The training peak, not the probe's. the plan's memory-budget log entry budgeted 25-28 GB against a
    # 40 GB card; the probe measured 60.28 GiB on one row with checkpointing
    # already on, and rows vary in length, so the number that decides whether
    # an arm can have a plain a100 is this one.
    import torch as _torch
    if _torch.cuda.is_available():
        print(f"[INFO] peak GPU memory {_torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    (args.out / "train_meta.json").write_text(json.dumps({
        "arm": args.arm, "rank": args.rank or lora_arms.ARMS[args.arm]["r"],
        "trainable": got, "rows": len(rows), "cases": sorted(cases),
        "epochs": args.epochs, "lr": args.lr, "grad_accum": args.grad_accum,
        "evidence_weight": args.evidence_weight,
        "eval_cases": sorted({r["case_id"] for r in eval_rows}),
        "eval_rows": len(eval_rows),
        "content_format": collator.content_format,
        "dropped": dict(dataset.dropped),
        "collator_stats": dict(collator.stats),
    }, indent=2), encoding="utf-8")
    print(f"[INFO] adapter -> {args.out / 'adapter'}")
    print(f"[INFO] dropped rows: {dict(dataset.dropped) or 'none'}")
    return 0


def probe_backward(model, dataset, collator, args) -> int:
    """Open item 3: do gradients reach the arm on THIS checkpoint?

    READ lora_B, NOT lora_A. LoRA initialises A random and **B to zeros**, so
    the adapter starts as an exact no-op. That makes dL/dA proportional to B,
    which is zero -- so at step 0 EVERY lora_A gradient is exactly zero, on
    every model, quantized or not. Reproduced on a 16-dim nn.Linear with no
    vision tower and no quantization at all.

    The first version of this probe read lora_A and reported "0 non-zero" as
    R3 failing (job 555443, on a plain bf16 checkpoint that contains no
    quantized weight to blame). The gradients were arriving perfectly; the
    probe was measuring the one matrix guaranteed to be zero.

    So the test is: does some lora_B carry a non-zero gradient -- that is what
    "the backward reached the arm" means at step 0. Then take one optimizer
    step, which makes B non-zero, and check that lora_A comes alive on the
    second backward. Passing both means the loop closes, not just that one
    tensor was touched.
    """
    import torch

    batch = collator([dataset.rows[0]])
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def grads(tag):
        return [(n, p) for n, p in model.named_parameters()
                if p.requires_grad and tag in n]

    def report(tag):
        named = grads(tag)
        live = [(n, p.grad) for n, p in named if p.grad is not None]
        nz = [(n, g) for n, g in live if bool((g != 0).any())]
        print(f"[INFO] {len(named)} {tag} matrices, {len(live)} with a "
              f"gradient, {len(nz)} non-zero")
        for n, g in nz[:2]:
            print(f"   {n}: |grad| mean {g.abs().mean().item():.3e} "
                  f"max {g.abs().max().item():.3e}")
        return len(nz)

    if not grads("lora_B"):
        sys.exit("[FAIL] no trainable lora_B parameters")

    loss, n_sup = loss_on_batch(model, batch)
    print(f"[INFO] forward: loss {loss.item():.4f} over {n_sup} supervised token(s)")
    loss.backward()

    nz_b = report("lora_B")
    report("lora_A")           # expected 0 here; see the docstring
    print("[NOTE] lora_A is zero at step 0 BY CONSTRUCTION (B starts at zero) "
          "-- that is not evidence of anything.")

    # Step once so B != 0, then confirm A receives a gradient too. This is the
    # half that proves the path is live rather than merely connected.
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)

    loss2, _ = loss_on_batch(model, batch)
    loss2.backward()
    print("[INFO] after one optimizer step:")
    nz_a = report("lora_A")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[INFO] peak GPU memory {peak:.2f} GiB")

    if not nz_b:
        print(f"[FAIL] R3: no lora_B received a non-zero gradient on "
              f"{args.model.name}. The backward does not reach the arm at all "
              f"-- look at the graph (gradient checkpointing, a detached "
              f"vision path), not at the checkpoint's precision.")
        return 1
    if not nz_a:
        print("[FAIL] R3: lora_B trains but lora_A stays zero after a step. "
              "The adapter cannot learn a rank>0 update; suspect the optimizer "
              "or a frozen A.")
        return 1
    print(f"[PASS] R3: gradients reach the arm on {args.model.name} -- "
          f"{nz_b} lora_B at step 0, {nz_a} lora_A after one step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
