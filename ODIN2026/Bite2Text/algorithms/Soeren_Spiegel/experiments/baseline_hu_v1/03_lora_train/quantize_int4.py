#!/usr/bin/env python3
"""Quantize a merged model dir to int4 (NF4) and save it pre-quantized.

Same NF4 recipe as the HEADLINE reference and as QLoRA train/eval there — so an int4 deploy matches an
int4 eval:

    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16

Saving the model post-quantization bakes ``quantization_config`` into config.json, so
``from_pretrained`` reloads it at int4 with no flags and no fp16->int4 conversion at load.

Usage:
    python quantize_int4.py --src /path/to/cache/model_export/merged_v03_ep4 \
                            --out /path/to/cache/model_export/model_v03_ep4_int4
"""
from __future__ import annotations

import _env  # noqa: F401  (blocks broken transformer_engine — MUST be first)

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="merged (unquantized) model dir")
    ap.add_argument("--out", required=True, help="output int4 model dir")
    ap.add_argument("--max-shard-size", default="2GB")
    # keep named modules in bf16 rather than NF4. Default skips the vision tower and the
    # LM head: the tower is now LoRA-adapted (attention + merger) and every score was measured
    # bf16, so NF4-quantizing weights that were never measured quantized adds unmeasured error
    # in a low-redundancy 1280-dim tower. Cost ~+1 GB tarball (~5 -> ~6 GB), still ~6 GB
    # inference VRAM against the 24 GB budget. Pass --skip-modules "" to quantize everything.
    ap.add_argument("--skip-modules", default="visual,lm_head",
                    help="comma-separated module-name substrings kept in bf16 (BitsAndBytes "
                         "llm_int8_skip_modules)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    skip = [m for m in (args.skip_modules or "").split(",") if m.strip()]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=skip or None,
    )
    print(f"Quantizing {src} -> NF4 (double-quant, bf16 compute); bf16-kept modules: {skip}")
    model = AutoModelForImageTextToText.from_pretrained(
        str(src), quantization_config=quant, device_map={"": 0})

    print("Saving int4 model …")
    model.save_pretrained(str(out), max_shard_size=args.max_shard_size)
    print("Saving processor …")
    AutoProcessor.from_pretrained(str(src)).save_pretrained(str(out))

    src_sz = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    out_sz = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n{src.name}: {src_sz/1e9:.1f} GB  ->  {out.name}: {out_sz/1e9:.1f} GB "
          f"({out_sz/src_sz:.0%})")
    for f in sorted(p.name for p in out.iterdir()):
        print("  ", f)


if __name__ == "__main__":
    main()
