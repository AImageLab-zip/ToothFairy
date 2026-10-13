#!/usr/bin/env python3
"""Build the two frozen-perception feature caches that every training stage reads.

    experiments/feature_cache/region_features_sc     dental segmentation features
    experiments/feature_cache/tf2_tooth_density_sc   per-tooth anchors (ToothFairy2)

Both use the input normalisation of the submitted model: the segmentation network's own CT
normalisation with the upper hard clip replaced by a soft clip (s = 1000). That setting is what
makes the cache the one the configs expect, so it is fixed here rather than exposed as a flag.

This runs the two frozen networks over every volume once and takes hours. Both extractors
are resumable — a volume whose .npz already exists is skipped — so an interrupted run can
simply be started again.

    python -m toothfairy.cli.cache                      # both caches
    python -m toothfairy.cli.cache --target teeth       # only the per-tooth anchors
    python -m toothfairy.cli.cache --device cuda:1
"""
from __future__ import annotations

import argparse

from toothfairy.pipeline import cache_dental, cache_teeth

# The normalisation the submitted model was trained with. Changing it produces a cache that no
# longer matches the checkpoints, which is why it is not a command-line flag.
SOFT_CLIP_S = "1000"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("dental", "teeth", "both"), default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="experiments/feature_cache",
                    help="the two caches are written to <out-root>/region_features_sc and "
                         "<out-root>/tf2_tooth_density_sc")
    args = ap.parse_args(argv)

    common = ["--device", args.device, "--soft-clip-s", SOFT_CLIP_S]

    if args.target in ("dental", "both"):
        print("[cache] dental features", flush=True)
        rc = cache_dental.main(common + ["--out-dir", f"{args.out_root}/region_features_sc"])
        if rc:
            return rc

    if args.target in ("teeth", "both"):
        print("[cache] per-tooth anchors", flush=True)
        rc = cache_teeth.main(common + ["--pooling", "argmax",
                                        "--out-dir", f"{args.out_root}/tf2_tooth_density_sc"])
        if rc:
            return rc

    print("[cache] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
