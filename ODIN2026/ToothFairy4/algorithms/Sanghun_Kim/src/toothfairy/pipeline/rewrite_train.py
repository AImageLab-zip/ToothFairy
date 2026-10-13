#!/usr/bin/env python3
"""Train a Qwen3.5-9B QLoRA that turns decomposed slots back into a narrative report.

## What this experiment measures

The mapping is **GT -> GT**, so there is no distribution shift: this measures the ceiling
reachable from formatting alone. Applying the rewriter to model output is a separate question —
there the input is incomplete, which opens the door to hallucination.

## Settings

- **Input format = slot labels** (`[fdi:36] 36 likely restored with post-and-core`).
- **Short system prompt.** Style, ordering and tooth grouping are taught by the data, since the
  target is the original report (91% of the training targets start with the mandible). The two
  constraints that remain describe failure modes actually observed.
- **LoRA identical to the LLM LoRA stage** — r 16, alpha 32, dropout 0.1, lr 2.0e-4.
- **Trained on the 560 training patients (903 rows) only.** The 62 validation patients are left
  untouched for the final comparison. Early stopping uses a dev set carved out of train by
  patient at 9:1 (using val for early stopping would contaminate the final numbers).

Warning: judge this with **RadFact**, not with captioning. Having learned the style, the model
will almost certainly win on BLEU/METEOR — that is the trap of this experiment. The question to
ask is whether clinical content was lost or invented.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402

SYSTEM_PROMPT = (
    "You are a dental and maxillofacial radiologist. You are given the findings of a\n"
    "dental CBCT, already extracted and grouped by anatomical structure. Write the\n"
    "findings section of the report.\n\n"
    "Use ONLY the findings given. Do not add, invent, infer, or drop any finding.\n"
    "Every tooth number in the input must be recoverable from your report — written\n"
    "explicitly, or covered by an explicit range you write. Do not introduce tooth\n"
    "numbers that are not in the input.\n\n"
    "Output ONLY the report text — no preamble, no bullet points, no headings."
)

# Slot presentation order = report order (mandible -> lower teeth -> mandibular canals ->
# maxilla -> upper teeth). The targets are written in that order (91.2% of val), so presenting
# the input the same way makes it easier to learn.
from toothfairy.schema.claims import FDI_ALL                                    # noqa: E402

SLOT_ORDER = (["mandible", "cls_lower"]
              + [f"fdi:{f}" for f in FDI_ALL if f // 10 in (3, 4)]
              + ["nerve_right", "nerve_left", "maxilla", "cls_upper"]
              + [f"fdi:{f}" for f in FDI_ALL if f // 10 in (1, 2)])


def user_prompt(slots: dict) -> str:
    lines = [f"[{k}] {l}" for k in SLOT_ORDER for l in slots.get(k, []) if l.strip()]
    return "Findings by structure:\n" + "\n".join(lines) + "\n\nWrite the report."


def build_examples(pairs_path: Path) -> list[dict]:
    rows = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    return [{"pid": r["pid"], "user": user_prompt(r["slots"]), "target": r["target"]}
            for r in rows if r.get("slots") and r.get("target")]


def split_by_patient(rows: list[dict], dev_frac: float, seed: int):
    """Split **by patient**. A patient can have several reports, so splitting by row would put
    the same patient in both train and dev and bias early stopping optimistically."""
    pids = sorted({r["pid"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_dev = max(1, int(round(len(pids) * dev_frac)))
    dev_pids = set(pids[:n_dev])
    tr = [r for r in rows if r["pid"] not in dev_pids]
    dv = [r for r in rows if r["pid"] in dev_pids]
    return tr, dv, dev_pids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="jsonl from `toothfairy.cli.rewrite pairs --split train`")
    ap.add_argument("--no-cls", action="store_true",
                    help="record that the pairs were built without the two arch-summary "
                         "slots. This does not change training — the pairs already are "
                         "what they are — it writes the fact into the adapter's "
                         "train_meta.json so that whatever applies the adapter feeds the "
                         "same 36 slots. Must match the flag used to build the pairs; "
                         "without it inference falls back to 38 and the adapter is used "
                         "on an input shape it was not trained on.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--lora-r", type=int, default=16)          # identical to the LLM LoRA stage
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 = dry-run (writes no checkpoint)")
    args = ap.parse_args(argv)

    import os
    os.environ.setdefault("TF4_TORCH_GDN_FALLBACK", "1")
    # Warning: setting the environment variable alone does nothing — it is our own code that
    # reads it and disables the transformers global. Omit this call and the weights still load,
    # but training and generation segfault.
    from toothfairy.generation.llm_backend import _maybe_torch_gdn_fallback
    if _maybe_torch_gdn_fallback():
        print("[qlora] Qwen3.5 GDN fast kernels off — using the pure torch path", flush=True)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO / args.out_dir
    if (out_dir / "done.flag").exists():
        print(f"[qlora] {out_dir}/done.flag exists — skipping (existing checkpoint protected).")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_examples(Path(args.pairs) if Path(args.pairs).is_absolute() else REPO / args.pairs)
    tr, dv, dev_pids = split_by_patient(rows, args.dev_frac, args.seed)
    print(f"[qlora] train {len(tr)} rows / dev {len(dv)} rows "
          f"({len(dev_pids)} patients) · {len(rows)} rows total")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def encode(rec):
        """Mask the prompt tokens with -100 so the loss covers **only the target report**."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rec["user"]}]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        h = tok(head, add_special_tokens=False)["input_ids"]
        t = tok(rec["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (h + t)[: args.max_len]
        labels = ([-100] * len(h) + t)[: args.max_len]
        return {"input_ids": ids, "labels": labels}

    class DS(Dataset):
        def __init__(self, recs): self.r = [encode(x) for x in recs]
        def __len__(self): return len(self.r)
        def __getitem__(self, i): return self.r[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        }

    tr_ld = DataLoader(DS(tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dv_ld = DataLoader(DS(dv), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg,
                                                 dtype=torch.bfloat16, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.config.use_cache = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-2)
    dev = torch.device("cuda")

    def evaluate():
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for b in dv_ld:
                b = {k: v.to(dev) for k, v in b.items()}
                ntok = int((b["labels"] != -100).sum())
                tot += float(model(**b).loss) * ntok; n += ntok
        model.train()
        return tot / max(n, 1)

    best, bad, step = math.inf, 0, 0
    import time
    for ep in range(1, args.epochs + 1):
        t0, run, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr_ld):
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} step {i}")
            (loss / args.grad_accum).backward()
            run += float(loss.detach()); nb += 1      # float() without detach() makes torch warn
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if nb % args.grad_accum:
            opt.step(); opt.zero_grad(set_to_none=True)
        vl = evaluate()
        mark = ""
        if vl < best - 1e-4:
            best, bad, mark = vl, 0, " *best"
            if not args.max_steps:
                model.save_pretrained(out_dir / "adapter_best")
        else:
            bad += 1
        print(f"[epoch {ep}/{args.epochs}] train={run/max(nb,1):.4f} dev={vl:.4f} "
              f"{time.time()-t0:.0f}s{mark}", flush=True)
        if args.max_steps:
            print("[qlora] dry-run finished — no checkpoint or done.flag written"); return 0
        if bad >= args.patience:
            print(f"[qlora] no improvement for {args.patience} epochs — early stop"); break

    # Travels with the adapter: the applying side reads it to check the prompt and to learn
    # whether the arch-summary slots were left out. Written next to adapter_best because the
    # adapter directory is what gets copied around.
    adapter_dir = out_dir / "adapter_best"
    if adapter_dir.is_dir():
        (adapter_dir / "train_meta.json").write_text(json.dumps({
            "system_prompt": SYSTEM_PROMPT, "no_cls": bool(args.no_cls),
            "model": args.model, "pairs": str(args.pairs),
        }, ensure_ascii=False, indent=2) + "\n")

    (out_dir / "meta.json").write_text(json.dumps({
        "model": args.model, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "lr": args.lr, "seed": args.seed,
        "grad_accum": args.grad_accum, "batch_size": args.batch_size,
        "best_dev_loss": best, "n_train_rows": len(tr), "n_dev_rows": len(dv),
        "dev_pids": sorted(dev_pids), "system_prompt": SYSTEM_PROMPT,
        "no_cls": bool(args.no_cls),
    }, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "done.flag").write_text(f"best dev loss {best:.4f}\n")
    print(f"[qlora] done — best dev {best:.4f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
