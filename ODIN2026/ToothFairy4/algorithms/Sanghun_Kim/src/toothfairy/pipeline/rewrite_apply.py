#!/usr/bin/env python3
"""Generate validation reports with the QLoRA narrative rewriter.

Uses the adapter trained by `pipeline/rewrite_train.py` to turn **per-slot text into a running
report**. It builds the **same prompts** as training (SYSTEM_PROMPT, user_prompt); a different
prompt would waste what was learned.

Output = <out-dir>/predictions_qlora.csv (example_id, prediction, target), which can be scored
directly with `pipeline/captioning.py` and fed to RadFact.

⚠️ The verdict is RadFact, not captioning. Having learned the style, the rewriter will almost
certainly win on BLEU/METEOR; the question to ask is whether clinical content was lost or
invented.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402



def _load_prompt_builders():
    """Take the prompts from the training module **as they are** (a copy would drift)."""
    from toothfairy.pipeline import rewrite_train
    return rewrite_train.SYSTEM_PROMPT, rewrite_train.user_prompt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="output of `toothfairy.cli.rewrite pairs --split val`")
    ap.add_argument("--adapter", required=True, help="models/rewrite_qlora_slotlabel/adapter_best")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import os
    os.environ.setdefault("TF4_TORCH_GDN_FALLBACK", "1")
    from toothfairy.generation.llm_backend import _maybe_torch_gdn_fallback
    if _maybe_torch_gdn_fallback():
        print("[eval] GDN fast-path kernels disabled - using the pure torch implementation",
              flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    SYSTEM_PROMPT, user_prompt = _load_prompt_builders()
    pairs_p = Path(args.pairs) if Path(args.pairs).is_absolute() else REPO / args.pairs
    rows = [json.loads(l) for l in pairs_p.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(rows)} validation rows")

    def write(arm, preds):
        p = out_dir / f"predictions_{arm}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["example_id", "prediction", "target"])
            w.writeheader()
            for r, pr in zip(rows, preds):
                w.writerow({"example_id": r["pid"], "prediction": pr, "target": r["target"]})
        print(f"[eval] {arm}: {len(preds)} rows -> {p}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg,
                                                 dtype=torch.bfloat16, device_map="cuda").eval()
    adapter = Path(args.adapter) if Path(args.adapter).is_absolute() else REPO / args.adapter
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    def generate(rec) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(rec["slots"])}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        import re
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = re.sub(r"<think>.*?</think>", " ", gen, flags=re.DOTALL)
        return " ".join(gen.replace("\n", " ").split()).strip()

    preds = []
    for i, rec in enumerate(rows, 1):
        preds.append(generate(rec))
        if i % 10 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}]", flush=True)
    write("qlora", preds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
