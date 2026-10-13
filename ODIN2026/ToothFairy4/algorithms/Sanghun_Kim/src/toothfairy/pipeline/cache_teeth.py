#!/usr/bin/env python
"""627-volume tf2 per-tooth density-split feature extraction.

Extracts, in a single sliding-window pass, the per-tooth bag **shared** by the whole-tooth pool
(`hybrid-dental-tf2`) and the density sub-pools (`hybrid-dental-tf2-density`, +hi/lo).

  - feature = tf2's own [encoder-stem; decoder-penultimate] (not the dental network), so
    feature, argmax and raw are all aligned in the same tf2 plans space.
  - **Only RAS originals are L-R flipped** (`prepare_tf2_volume`; LPS is left uncorrected). The
    purpose is not anatomical correctness but **agreement with the ground-truth convention**
    (feature and label must use the same handedness) — for the evidence and the measurements
    see the `prepare_tf2_volume` docstring.
  - Density thresholds are applied on the resampled **raw** volume: per-tooth p50 (dentin)
    anchor, `> k_high*p50` = hi (metal/endo), `< k_low*p50` = lo (caries). An empty subset gives
    a zero vector with mass 0.
  - row i = tf2 label i+1 = FDI_ALL[i] = categorical ground-truth row (identity).

Usage:
  python -m toothfairy.pipeline.cache_teeth --device cuda
  (Heavy inference: run it in a background session. Existing npz files are skipped = resumable.)

The driver mirrors `pipeline/cache_dental.py` (resume, _meta.json, flip counts).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from toothfairy.paths import REPO  # noqa: E402

MODEL_DIR = (REPO / "models/seg/ToothSeg/Dataset121_ToothFairy2_Teeth/"
             "nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm")
FOLD = 5
CHECKPOINT = "checkpoint_final.pth"
OUT_DIR = REPO / "experiments/feature_cache/tf2_tooth_density"


def _patch_trainer_lookup():
    """Bypass the nnU-Net trainer class lookup (the custom trainer name is not installed)."""
    import nnunetv2.inference.predict_from_raw_data as pf
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    pf.recursive_find_trainer_class_by_name = lambda name: nnUNetTrainer


def list_patients() -> list[str]:
    data = REPO / "data"
    return [d.name for d in sorted(data.iterdir())
            if d.is_dir() and (d / "cbct" / "volume.nii.gz").exists()]


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N volumes")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--k-hi", type=float, default=1.5,
                    help="hi band onset = k_hi * per-patient dentin mode")
    ap.add_argument("--k-lo", type=float, default=0.6,
                    help="lo (caries) cut = k_lo * per-patient dentin mode")
    ap.add_argument("--sat-lo", type=float, default=0.99,
                    help="ceiling(sat) pile = raw >= sat_lo * per-patient vmax")
    ap.add_argument("--pooling", default="soft", choices=["soft", "argmax"],
                    help="weighting of the whole-tooth pool: soft (default, per-label softmax "
                         "P[L,x]) or argmax (voxel assigned to its argmax tooth, weight 1.0). "
                         "The lo/hi/sat bands and the centers are produced in both modes.")
    ap.add_argument("--soft-clip-s", type=float, default=1000.0,
                    help="soft-clip tanh scale for feature normalization (0 = standard "
                         "CTNormalization). The density band thresholds run on the resampled "
                         "RAW volume and are unaffected by this value.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    _patch_trainer_lookup()
    from toothfairy.perception.tf2_tooth_density import (
        build_tf2_predictor, prepare_tf2_volume, extract_tooth_density_bag, N_TEETH)
    from toothfairy.perception.softclip_norm import install_softclip_ctnorm
    from toothfairy.perception.cache import check_cache_provenance

    if args.soft_clip_s > 0:                       # swap tf2's CTNormalization.run too (as dental)
        install_softclip_ctnorm(args.soft_clip_s)
    norm_desc = (f"soft-clip CTNormalization (s={args.soft_clip_s}) — feature only; bands use RAW"
                 if args.soft_clip_s > 0 else "standard CTNormalization")
    cache_params = {"normalization": norm_desc, "soft_clip_s": args.soft_clip_s,
                    "pooling": args.pooling, "k_hi": args.k_hi, "k_lo": args.k_lo,
                    "sat_lo": args.sat_lo}
    # Refuse to reuse a stale cache (different normalization, pooling or thresholds): a mismatch
    # with an existing finished cache aborts the run.
    check_cache_provenance(out_dir, cache_params)

    pids = list_patients()
    if args.limit:
        pids = pids[: args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    print(f"[info] {len(pids)} patient volumes; out={out_dir}; git={commit}", flush=True)

    pred = build_tf2_predictor(str(MODEL_DIR), fold=FOLD,
                               checkpoint_name=CHECKPOINT, device=args.device)
    n_heads = pred.label_manager.num_segmentation_heads
    print(f"[backbone] tf2 n_seg_heads={n_heads} "
          f"patch_size={pred.configuration_manager.patch_size} tile_step=1.0", flush=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="ras_prep_tf2_"))
    errors: dict[str, str] = {}
    n_flipped = 0
    present_counts: list[int] = []
    hidense_teeth_counts: list[int] = []   # hi_mass>0 OR sat_mass>0 (high-density coverage)
    t_start = time.time()
    done = 0
    for i, pid in enumerate(pids):
        out_f = out_dir / f"{pid}.npz"
        if out_f.exists():                           # resumable
            done += 1
            continue
        vol = REPO / "data" / pid / "cbct" / "volume.nii.gz"
        t0 = time.time()
        use_path = None
        flipped = False
        try:
            use_path, flipped = prepare_tf2_volume(str(vol), tmpdir)   # flip RAS only (GT rule)
            n_flipped += int(flipped)
            bag = extract_tooth_density_bag(pred, use_path,
                                            k_hi=args.k_hi, k_lo=args.k_lo, sat_lo=args.sat_lo,
                                            pooling=args.pooling)
            np.savez(out_f, **bag.to_npz_dict())
            stat = (int(bag.present.sum()),
                    int(((bag.hi_mass > 0) | (bag.sat_mass > 0)).sum()),
                    bag.dentin, bag.vmax)
            done += 1
            present_counts.append(stat[0])
            hidense_teeth_counts.append(stat[1])
            dt = time.time() - t0
            if i % 20 == 0 or dt > 120:
                print(f"  [{i+1}/{len(pids)}] {pid} {dt:5.1f}s flip={flipped} "
                      f"present={stat[0]}/{N_TEETH} hidense_teeth={stat[1]} "
                      f"dentin={stat[2]:.0f} vmax={stat[3]:.0f}", flush=True)
        except Exception as e:  # noqa: BLE001 — record it and carry on
            errors[pid] = f"{type(e).__name__}: {e}"
            print(f"  [{i+1}/{len(pids)}] {pid} FAILED {errors[pid]}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if flipped and use_path:
                Path(use_path).unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[info] L-R flip applied: {n_flipped} volumes (RAS originals only)", flush=True)

    meta = {
        "backbone": "tf2 (ToothFairy2_Teeth) — per-tooth density-split",
        "feature": "[encoder-stem; decoder-penultimate] 64ch, full per-label soft pooled per tooth (whole/lo/hi/sat)",
        "pooling": (f"whole={args.pooling}. soft = per-label softmax P[L,x] weighting "
                    "(pooled[L]=sum P[L,x]*feat / sum P[L,x], so boundary voxels contribute to "
                    "several teeth) / argmax = voxel assigned to its argmax tooth, weight 1.0. "
                    "Bands (lo/hi/sat) and centers are produced in both modes."),
        "row_index": "row i = tf2 label i+1 = FDI_ALL[i] = categorical GT row (identity)",
        "normalization": norm_desc,
        "cache_params": cache_params,           # what the provenance guard compares
        "anchor": "per-patient dentin mode D (mode of the histogram over all tooth voxels, "
                  "robust) — removes the self-reference defect of the per-tooth p50 anchor. "
                  "The threshold constants k_hi/k_lo/sat_lo are provisional defaults",
        "k_hi": args.k_hi, "k_lo": args.k_lo, "sat_lo": args.sat_lo,
        "subpools": "lo=raw<k_lo*D (caries) / hi=k_hi*D<raw<sat_lo*vmax (high density, "
                    "unsaturated) / sat=raw>=sat_lo*vmax (ceiling pile: recovers clipped metal)",
        "threshold_space": "resampled-RAW (avoids the CTNorm p99.5 clip)",
        "array_handedness": "prepare_tf2_volume: L-R flip on RAS originals only, LPS "
                            "uncorrected. Purpose = agreement with the GT convention, not "
                            "anatomical correctness; see the prepare_tf2_volume docstring",
        "model_dir": str(MODEL_DIR.relative_to(REPO)),
        "fold": FOLD, "checkpoint": CHECKPOINT, "n_seg_heads": int(n_heads),
        "patch_size": list(pred.configuration_manager.patch_size),
        "tile_step_size": 1.0,
        "n_patients": len(pids), "n_done": done, "n_errors": len(errors),
        "n_lps_flipped": n_flipped,
        "git_commit": commit,
        "errors": errors,
        "present_teeth_mean": (round(float(np.mean(present_counts)), 1)
                               if present_counts else None),
        "hidense_teeth_mean": (round(float(np.mean(hidense_teeth_counts)), 1)
                               if hidense_teeth_counts else None),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] done={done}/{len(pids)} errors={len(errors)} "
          f"flip={n_flipped} elapsed={meta['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
