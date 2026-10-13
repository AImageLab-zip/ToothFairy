"""Rebuild work/pred_findings.json from predicted masks.

Split into two stages on purpose, because they have very different costs:

    segment   GPU, ~4 h for the corpus, and it caches the predicted mask to
              work/pred_masks/<case>.nii.gz (uint8, gzip -- a label map
              compresses to a couple of MB).
    derive    CPU, a few minutes, reading those cached masks.

Every earlier feature change re-ran the segmentation, so a one-line threshold
edit cost four hours of GPU. With the masks cached, only the derivation reruns,
and the GPU stays free for RadFact -- which matters because the judge needs
20.5 GB and segmentation ~9.5 GB, and the card is 32 GB, so they cannot both be
resident.

    python tf4/build_findings.py segment          # once, or after a retrain
    python tf4/build_findings.py derive           # after any feature change
    python tf4/build_findings.py segment --cases P001,P002
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "toothfairy4_v03")
MASKS = os.path.join(ROOT, "work", "pred_masks")
OUT = os.path.join(ROOT, "work", "pred_findings.json")


def all_cases() -> list[str]:
    return [
        c for c in sorted(os.listdir(DATA))
        if os.path.exists(os.path.join(DATA, c, "cbct", "volume.nii.gz"))
    ]


def volume_path(case: str) -> str:
    return os.path.join(DATA, case, "cbct", "volume.nii.gz")


def mask_path(case: str) -> str:
    return os.path.join(MASKS, f"{case}.nii.gz")


def prepared_volume(case: str):
    """Canonical-frame, 0.3 mm-resampled volume -- the exact array the mask is
    predicted from, so mask and image share a frame and a shape."""
    import nibabel as nib

    from tf4.trinity_infer import resample_to_target, to_canonical

    data, spacing = to_canonical(nib.load(volume_path(case)))
    return resample_to_target(data, spacing)


def segment(cases: list[str], overwrite: bool) -> int:
    import nibabel as nib
    import torch

    from tf4.trinity_infer import (
        from_training_frame, load_model, predict_volume, to_training_frame,
    )

    os.makedirs(MASKS, exist_ok=True)
    todo = [c for c in cases if overwrite or not os.path.exists(mask_path(c))]
    print(f"segmenting {len(todo)} of {len(cases)} cases "
          f"({len(cases) - len(todo)} already cached)", flush=True)
    if not todo:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(os.path.join(ROOT, "algorithm/model/trinity_weights_fp16.pth"), device=device)
    norm = json.load(open(os.path.join(ROOT, "algorithm/model/norm_stats.json")))

    started = time.time()
    for i, case in enumerate(todo, 1):
        data, spacing = prepared_volume(case)
        mask = from_training_frame(
            predict_volume(model, to_training_frame(data), norm, device=device)
        )
        # The affine only has to round-trip the voxel grid; downstream geometry
        # works in array space and recovers anatomical axes from the labels.
        img = nib.Nifti1Image(mask.astype(np.uint8), np.diag(list(spacing) + [1.0]))
        img.header.set_zooms(tuple(float(s) for s in spacing))
        nib.save(img, mask_path(case))
        rate = (time.time() - started) / i
        print(f"  [{i}/{len(todo)}] {case}  {rate:.1f}s/case  "
              f"eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)
    return 0


def _derive_one(case: str):
    """Worker: cached mask + resampled volume -> the serialisable findings dict."""
    import nibabel as nib

    from tf4.mask_features import derive_findings

    try:
        mask = np.asarray(nib.load(mask_path(case)).dataobj).astype(np.uint8)
        image, spacing = prepared_volume(case)
        if image.shape != mask.shape:
            return case, None, f"mask {mask.shape} vs image {image.shape}"
        f = derive_findings(mask, spacing=float(np.mean(spacing)), case_id=case, image=image)
        return case, {
            "teeth": {str(k): v for k, v in f.teeth_present.items()},
            "canal_contact": {k: int(v) for k, v in f.canal_contact.items()},
            "impacted": sorted(f.impacted),
            "maxilla": f.maxilla_coverage,
            "canal_course": f.canal_course,
            "edentulism": f.edentulism,
            "has_implant": f.has_implant,
            "has_bridge": f.has_bridge,
            "has_crown": f.has_crown,
            "sinuses": f.sinuses_included,
            "crowns": sorted(f.crowns),
            "endodontic": sorted(f.endodontic),
            "restorations": sorted(f.restorations),
        }, None
    except Exception as exc:  # noqa: BLE001
        return case, None, repr(exc)


def derive(cases: list[str], workers: int) -> int:
    from concurrent.futures import ProcessPoolExecutor

    have = [c for c in cases if os.path.exists(mask_path(c))]
    missing = len(cases) - len(have)
    if missing:
        print(f"note: {missing} cases have no cached mask; run `segment` first", flush=True)

    # Merge rather than replace: `derive --cases P001` must not throw away the
    # other 526 entries, which is exactly what it did the first time it ran.
    out: dict[str, dict] = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    kept = len(out)

    # Torch is only used for the resample here, and it grabs every core by
    # default, which fights the process pool.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    print(f"deriving {len(have)} cases on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for i, (case, rec, err) in enumerate(pool.map(_derive_one, have), 1):
            if err:
                print(f"  !! {case}: {err}", flush=True)
                continue
            out[case] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(have)}", flush=True)

    json.dump(out, open(OUT, "w"), indent=0)
    print(f"\nwrote {OUT}: {len(have)} derived, {len(out)} total "
          f"({kept} were already there)")
    for slot in ("crowns", "endodontic", "impacted"):
        vals = [v.get(slot) or [] for v in out.values()]
        per_case = sum(len(v) for v in vals) / max(1, len(vals))
        print(f"  {slot:11s} {per_case:.2f}/case, present in "
              f"{sum(1 for v in vals if v)}/{len(vals)} cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["segment", "derive"])
    ap.add_argument("--cases", default="", help="comma-separated case ids; default is all")
    ap.add_argument("--overwrite", action="store_true", help="re-segment cases that are cached")
    ap.add_argument("--workers", type=int, default=6, help="derive only")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or all_cases()
    if args.stage == "segment":
        return segment(cases, args.overwrite)
    return derive(cases, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
