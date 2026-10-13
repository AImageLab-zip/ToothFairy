#!/usr/bin/env python3
"""Offline re-cache of per-patch [encoder-stem; decoder-penultimate] region features.

The frozen dental segmentation network is deterministic, so we run it ONCE per CBCT
volume and persist the per-patch (pooled, mass, centre) bags for all 5 non-background
classes; the training loop then reads the small cache instead of re-running the
segmentation network.

Notes:
  - the perception compartment (`toothfairy.perception.{backbone,cache}`) does the work;
  - **there is no mass threshold** — every patch enters every class bag;
  - the LPS->RAS orientation correction is applied inside the extraction body itself, so a
    caller cannot forget it.
Usage:
  python -m toothfairy.pipeline.cache_dental --device cuda
  (Heavy inference: run it in a background session. Existing npz files are skipped = resumable.)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from toothfairy.paths import REPO  # noqa: E402

_LPS = ("L", "P", "S")            # nnU-Net ignores the affine and works in array order, so in an
#   LPS volume left/right in plans space is exactly reversed relative to RAS. The orientation fix
#   is built into the extraction body so it cannot be skipped: an LPS volume is extracted from a
#   temporary NIfTI whose L-R array axis has been flipped to RAS. A RAS volume is used as it is.

MODEL_DIR = (REPO / "models/seg/Dataset112_DentalSegmentator_v100/"
             "nnUNetTrainer__nnUNetPlans__3d_fullres")
FOLD = 0
CHECKPOINT = "checkpoint_final.pth"
# Canonical cache dir (config.PerceptionConfig.cache_dir), kept separate from older caches.
OUT_DIR = REPO / "experiments/feature_cache/region_features"


def _prepare_ras_volume(vol_path: str, tmpdir: Path):
    """For an LPS volume write a temporary NIfTI with the L-R axis flipped (so everything is RAS)
    and return its path; for RAS return the original path. Returns (path, flipped_bool)."""
    ax = nib.aff2axcodes(nib.load(vol_path).affine)
    if tuple(ax) != _LPS:
        return vol_path, False
    lr = next(i for i, c in enumerate(ax) if c in ("L", "R"))
    img = nib.load(vol_path)
    data = np.flip(np.asanyarray(img.dataobj), axis=lr).copy()
    fp = tmpdir / (Path(vol_path).parts[-3] + ".nii.gz")     # pid.nii.gz
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(fp))
    return str(fp), True


def _patch_trainer_lookup():
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
    ap.add_argument("--soft-clip-s", type=float, default=1000.0,
                    help="soft-clip tanh scale (relaxes the upper clip; 0 = standard "
                         "CTNormalization)")
    # Most of the cost is CPU preprocessing (~20s per volume) while the GPU idles, so parallelise
    # by sharding processes. Shards only ever touch disjoint volumes, so sharing one out-dir is
    # safe (the resume logic is unaffected).
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    _patch_trainer_lookup()
    from toothfairy.perception.backbone import build_predictor
    from toothfairy.perception.cache import extract_region_bag, check_cache_provenance
    from toothfairy.perception.region_pooling import POOL_CLASSES
    from toothfairy.perception.softclip_norm import install_softclip_ctnorm

    if args.soft_clip_s > 0:                       # swap nnU-Net's CTNormalization.run for ours
        install_softclip_ctnorm(args.soft_clip_s)
    norm_desc = (f"soft-clip CTNormalization (s={args.soft_clip_s})"
                 if args.soft_clip_s > 0 else "standard CTNormalization")
    cache_params = {"normalization": norm_desc, "soft_clip_s": args.soft_clip_s}
    # Refuse to reuse a stale cache (different normalisation): abort when the
    # parameters do not match those of an existing finished cache.
    check_cache_provenance(out_dir, cache_params)

    pids = list_patients()
    if args.limit:
        pids = pids[: args.limit]
    n_total = len(pids)
    if args.num_shards > 1:
        pids = pids[args.shard_idx::args.num_shards]
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    print(f"[info] {len(pids)}/{n_total} patient volumes "
          f"(shard {args.shard_idx}/{args.num_shards}); out={out_dir}; git={commit}", flush=True)

    pred = build_predictor(str(MODEL_DIR), fold=FOLD,
                           checkpoint_name=CHECKPOINT, device=args.device)
    n_heads = pred.label_manager.num_segmentation_heads
    print(f"[backbone] dental n_seg_heads={n_heads} "
          f"patch_size={pred.configuration_manager.patch_size}", flush=True)

    import tempfile
    import shutil
    tmpdir = Path(tempfile.mkdtemp(prefix="ras_prep_"))
    errors: dict[str, str] = {}
    per_class_counts = {c: [] for c in POOL_CLASSES}
    n_flipped = 0
    t_start = time.time()
    done = 0
    for i, pid in enumerate(pids):
        out_f = out_dir / f"{pid}.npz"
        if out_f.exists():
            done += 1
            continue
        vol = REPO / "data" / pid / "cbct" / "volume.nii.gz"
        t0 = time.time()
        use_path = None
        flipped = False
        try:
            use_path, flipped = _prepare_ras_volume(str(vol), tmpdir)  # unify orientation to RAS
            n_flipped += int(flipped)
            bag = extract_region_bag(pred, use_path)                   # no mass threshold
            # Write to a temp file and rename: an interrupted run must not leave a truncated
            # npz that the next run treats as finished and skips. The process id in the name
            # keeps concurrent shards from stepping on each other.
            tmp_f = out_f.with_suffix(f".part{os.getpid()}.npz")
            np.savez(tmp_f, **bag.to_npz_dict())
            tmp_f.replace(out_f)
            counts = {c: len(bag.mass[c]) for c in POOL_CLASSES}
            done += 1
            for c in POOL_CLASSES:
                per_class_counts[c].append(counts[c])
            dt = time.time() - t0
            if i % 20 == 0 or dt > 60:
                print(f"  [{i+1}/{len(pids)}] {pid} {dt:5.1f}s flip={flipped} "
                      f"patches={counts}", flush=True)
        except Exception as e:  # noqa: BLE001 — record and continue
            errors[pid] = f"{type(e).__name__}: {e}"
            print(f"  [{i+1}/{len(pids)}] {pid} FAILED {errors[pid]}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if flipped and use_path:                          # remove only the flipped temp file
                Path(use_path).unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[info] LPS→RAS flip applied: {n_flipped} volumes", flush=True)

    meta = {
        "backbone": "dental",
        "feature": "[encoder-stem; decoder-penultimate] 64ch dual",
        "classes": POOL_CLASSES,
        "model_dir": str(MODEL_DIR.relative_to(REPO)),
        "fold": FOLD, "checkpoint": CHECKPOINT, "n_seg_heads": int(n_heads),
        "patch_size": list(pred.configuration_manager.patch_size),
        "normalization": norm_desc,
        "cache_params": cache_params,           # what the provenance guard compares
        "mass_thr": None,                       # no mass threshold: every patch -> every class bag
        "n_patients": len(pids), "n_done": done, "n_errors": len(errors),
        "n_lps_flipped": n_flipped,
        "shard": [args.shard_idx, args.num_shards],
        "git_commit": commit,
        "errors": errors,
        # The statistics below cover only the v0 bags computed in this run (volumes skipped on
        # resume are excluded), which is what n_stats_from states.
        "n_stats_from": len(next(iter(per_class_counts.values()), [])),
        "patches_per_vol_mean_fresh": {
            c: (round(float(np.mean(v)), 1) if v else None)
            for c, v in per_class_counts.items()},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    # A sharded run writes one meta file per shard so they cannot overwrite each other. The merged
    # `_meta.json` — the file the provenance guard reads — is written once every shard has finished.
    meta_name = ("_meta.json" if args.num_shards == 1
                 else f"_meta.shard{args.shard_idx}of{args.num_shards}.json")
    (out_dir / meta_name).write_text(json.dumps(meta, indent=2))
    print(f"[done] done={done}/{len(pids)} errors={len(errors)} "
          f"elapsed={meta['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
