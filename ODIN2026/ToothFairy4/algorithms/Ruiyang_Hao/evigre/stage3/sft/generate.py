#!/usr/bin/env python3
"""Rewrite deterministic report templates with the released Qwen2.5 LoRA adapter."""

from __future__ import annotations

import argparse
import re

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "You are a dental radiologist. You are given a DRAFT CBCT report produced by a rigid template. "
    "Rewrite it into fluent, natural radiologist prose as used in real reports, while keeping the "
    "clinical content identical: preserve EXACTLY the findings and FDI tooth numbers in the DRAFT, "
    "add nothing that is not in the DRAFT, and drop nothing. Output the rewritten report only."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--id-col", default="case_id")
    parser.add_argument("--text-col", default="generated_report")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def normalized_teeth(text: str) -> set[str]:
    normalized = re.sub(r"([1-4])\.([1-8])", r"\1\2", str(text))
    return set(re.findall(r"\b[1-4][1-8]\b", normalized))


def main() -> int:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    frame = pd.read_csv(args.in_csv)
    drafts = frame[args.text_col].astype(str).tolist()
    outputs = []
    for start in range(0, len(drafts), args.batch_size):
        batch = drafts[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"DRAFT:\n{draft.strip()}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for draft in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
            for row in generated
        )

    added = 0
    dropped = 0
    affected = 0
    for draft, output in zip(drafts, outputs):
        added_teeth = normalized_teeth(output) - normalized_teeth(draft)
        dropped_teeth = normalized_teeth(draft) - normalized_teeth(output)
        added += len(added_teeth)
        dropped += len(dropped_teeth)
        affected += int(bool(added_teeth or dropped_teeth))

    frame[args.text_col] = outputs
    frame.to_csv(args.out_csv, index=False)
    print(f"wrote {len(frame)} reports to {args.out_csv}")
    print(f"tooth-number QC: added={added}, dropped={dropped}, affected={affected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
