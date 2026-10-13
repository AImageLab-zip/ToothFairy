#!/usr/bin/env python3
"""Verify a pre-quantized int4 Bite2Text export end-to-end, under a VRAM budget.

Sibling of submission/verify_model.py (Task 1 / fiveviews). Reloads the int4 model dir with
NO quantization flags (the saved quantization_config must drive it), asserts the layers really
came back 4-bit, then renders a real case through the SAME dataset path the container uses
(``dataset.render_case_messages`` + ``train.processor_inputs``) and generates a report.

``--vram-cap-gib`` caps this process's allocator to a fraction of the device so a big card can
answer "does it fit on a 16 GiB GPU?". The cap binds the torch caching allocator only — a real
card also spends ~0.4-0.8 GB on the CUDA context outside it, so the reported headroom is
optimistic by about that much. Pass the cap you actually have to budget for.

Usage:
    python bite2text_pipeline/verify_model.py --model /path/to/cache/model_export/model_..._int4 \
        --split val --max-cases 2 --vram-cap-gib 16
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from bite2text_pipeline.dataset import (  # noqa: E402
    ZIP_PATH, load_split_records, render_case_messages,
)
from bite2text_pipeline.train import processor_inputs  # noqa: E402

GIB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="int4 model dir (tarball payload)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-cases", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="entrypoint.py default (chart + report needs 768, not 512)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                    help="entrypoint.py uses sdpa; flash_attention_2 avoids the ViT's dense "
                         "NxN mask, which is the biggest single spike at ~30k patches")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0,
                    help="cap the torch allocator to this many GiB (0 = uncapped)")
    args = ap.parse_args()

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    if args.vram_cap_gib:
        frac = (args.vram_cap_gib * GIB) / total
        if frac >= 1.0:
            print(f"cap {args.vram_cap_gib} GiB >= device {total/GIB:.1f} GiB — no cap applied")
        else:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print(f"VRAM cap: {args.vram_cap_gib:.1f} GiB "
                  f"({frac:.1%} of the {total/GIB:.1f} GiB device)")

    print(f"Reloading {args.model} with NO quant flags (saved config must drive it) …")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, device_map={"": 0}, attn_implementation=args.attn)
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 256, max_pixels=1280 * 1280)
    model.eval()

    qc = getattr(model.config, "quantization_config", None)
    print(f"quantization_config: {qc}")
    n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    print(f"Linear4bit modules: {n4}")
    assert n4 > 0, "no Linear4bit modules — the reload did NOT come back quantized"
    w = torch.cuda.memory_allocated()
    print(f"weights on GPU: {w/GIB:.2f} GiB")

    records = load_split_records(args.split)[: args.max_cases]
    zf = zipfile.ZipFile(ZIP_PATH)
    for rec in records:
        cid = rec["case_id"]
        msgs, n_r, n_p = render_case_messages(cid, report=None, zf=zf)
        if msgs is None:
            print(f"  render failed: {cid}")
            continue
        inputs = processor_inputs(processor, [msgs], add_generation_prompt=True).to(device)
        n_tok = inputs["input_ids"].shape[1]
        n_patch = inputs["pixel_values"].shape[0]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None)
        text = processor.decode(out[0][n_tok:], skip_special_tokens=True).strip()
        peak_a = torch.cuda.max_memory_allocated() / GIB
        peak_r = torch.cuda.max_memory_reserved() / GIB
        print(f"\n--- {cid}  renders={n_r} photos={n_p} prompt_tok={n_tok} "
              f"vit_patches={n_patch} gen_tok={out.shape[1]-n_tok} ---")
        print(f"    peak allocated {peak_a:.2f} GiB   peak reserved {peak_r:.2f} GiB "
              f"(activations {peak_a - w/GIB:.2f} GiB over weights)")
        print(text[:600] + ("…" if len(text) > 600 else ""))

    print(f"\npeak VRAM incl. generation: "
          f"allocated {torch.cuda.max_memory_allocated()/GIB:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved()/GIB:.2f} GiB")
    print("Add ~0.4-0.8 GiB for the CUDA context when comparing against a real card.")
    print("VERIFY OK")


if __name__ == "__main__":
    main()
