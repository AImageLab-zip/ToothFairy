#!/usr/bin/env python3
"""One CBCT volume -> one English report.

This is the whole inference path in one call: preprocessing, both frozen segmentation
networks, the aggregator, the 38 per-slot decodes and the narrative rewrite. There is no
feature cache and no ground truth involved — the perception features are extracted from
the volume on the fly. Peak VRAM is 19.8GB and one case takes about 170s on an A10G.

    python -m toothfairy.cli.report --model-dir /path/to/weights --volume case.nii.gz
    python -m toothfairy.cli.report --model-dir /path/to/weights --volume case.mha --out report.txt

The weights are not in this repository. The model directory is expected to hold:

    best.pt                the trained checkpoint (aggregator + projector + LoRA)
    config.resolved.yaml   the exact resolved config that produced it
    dental_segmentator/    nnU-Net model folder, fold 0, checkpoint_final.pth
    tf2_segmentator/       nnU-Net model folder, fold 5, checkpoint_final.pth
    llm/                   the frozen language model (config.realizer.llm_name points here)
    rewrite_adapter/       the narrative rewriter, with train_meta.json alongside it

The 38 slot outputs become one report through the rewrite adapter. ``train_meta.json`` must
travel with the adapter: it records whether the rewriter was trained without the two
arch-summary slots, so that 36 or 38 slots are fed accordingly, and it carries the training
prompt, which is compared against the one about to be used.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="directory holding the weights")
    ap.add_argument("--volume", required=True, help="CBCT volume (.nii.gz or .mha)")
    ap.add_argument("--out", default=None, help="write the report here (default: stdout)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-name", default="best.pt")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--offline", action="store_true",
                    help="never reach the Hugging Face hub; fail loudly instead")
    args = ap.parse_args(argv)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Avoids the fragmentation OOM of the 33-channel softmax patch computation, seen on large
    # volumes during feature extraction.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # imported after the environment is set

    from toothfairy.inference import ReportGenerator

    volume = Path(args.volume)
    if not volume.is_file():
        print(f"no such volume: {volume}", file=sys.stderr)
        return 2

    device = args.device if torch.cuda.is_available() else "cpu"
    generator = ReportGenerator.from_pretrained(args.model_dir, device=device,
                                                ckpt_name=args.ckpt_name)
    report = generator.generate(str(volume), max_new_tokens=args.max_new_tokens)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[report] {args.out}", flush=True)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
