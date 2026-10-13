#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and export a self-contained model dir.

Port of HEADLINE's `merge_export.py` (the HEADLINE Grand Challenge "Model
tarball" payload step), adapted for this repo's checkpoints.

Runs on CPU by default (`CUDA_VISIBLE_DEVICES=""`) so it doesn't disturb GPU work.

Deviation from the reference: dtype defaults to **bf16**, not fp16. Training ran in bf16
([train.py](train.py)), and NF4 quantization discards the source dtype anyway, so bf16 keeps
the merge numerically identical to training instead of round-tripping through fp16.

Usage:
    CUDA_VISIBLE_DEVICES="" python merge_export.py \
        --checkpoint runs/v03_ep6/adapter_ep4 --out-dir /path/to/cache/merged_v03_ep4
"""
from __future__ import annotations

import _env  # noqa: F401  (blocks broken transformer_engine — MUST be first)

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LoRA adapter checkpoint dir")
    ap.add_argument("--out-dir", required=True, help="output merged-model dir (fast local disk)")
    ap.add_argument("--base", default=None,
                    help="override base model (else read from adapter_config.json)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--max-shard-size", default="4GB",
                    help="shard safetensors so no single file exceeds this; a single ~16.5 GB "
                         "file fails to mmap on a 16 GB-RAM host")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = Path(__file__).parent / ckpt
    acfg = json.loads((ckpt / "adapter_config.json").read_text())
    base = args.base or acfg["base_model_name_or_path"]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"Base model : {base}")
    print(f"Adapter    : {ckpt}  (r={acfg.get('r')}, alpha={acfg.get('lora_alpha')})")
    print(f"Out dir    : {out}   dtype={args.dtype}")

    print("Loading base (CPU, low_cpu_mem_usage) …")
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None)

    print("Attaching adapter + merge_and_unload …")
    model = PeftModel.from_pretrained(model, str(ckpt), torch_dtype=dtype)
    model = model.merge_and_unload()
    model = model.to(dtype)

    print("Saving merged model …")
    model.save_pretrained(str(out), safe_serialization=True,
                          max_shard_size=args.max_shard_size)

    # Processor from the BASE so the merged dir is self-contained for
    # AutoProcessor.from_pretrained(MODEL_PATH).
    print("Saving processor …")
    AutoProcessor.from_pretrained(base).save_pretrained(str(out))

    files = sorted(p.name for p in out.iterdir())
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"\nDone. {len(files)} files, {total/1e9:.1f} GB in {out}:")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
