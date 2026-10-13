#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_evaluate_curve_multimodal.py

Strict OOF multimodal screening for rebuilt Curve tasks.

Tasks
-----
1) curve_spee
2) curve_wilson
3) curve_wilson_location

Modalities compared on the SAME patient cohort within each task
---------------------------------------------------------------
- majority
- 3D only
- render only
- photo only
- 3D + render
- 3D + photo
- 3D + render + photo

Image representation
--------------------
This 04 is deliberately a FAST / STRICT screening stage:

    image -> frozen ImageNet ResNet18 -> 512-D embedding

No labels are used to train the image encoder, so embeddings may be safely
cached once for all CV folds.

If photo/render embeddings clearly improve OOF Accuracy, the next stage will
fine-tune only the winning image configuration with fold-specific CNN training.

Why not fine-tune every image combination immediately?
------------------------------------------------------
There are only ~870 photo_en cases. Running 5-fold CNN training for every
photo/render/fusion combination is expensive and makes model selection noisy.
This script first answers the high-value question:

    Does photo and/or STL render add predictive information beyond the strong
    3D curve geometry?

Fair comparison policy
----------------------
Within each task, ALL modality variants are evaluated on one strict COMMON
cohort so Accuracy is directly comparable.

Spee common cohort requires:
    photo_en GT
    geometry_any_valid
    Spee render ready
    EXACT intraoral_2 + intraoral_3 available

Wilson common cohort requires:
    photo_en GT
    geometry_any_valid
    Wilson render ready
    EXACT intraoral_1 available

IMPORTANT:
    No "nearest available photo" fallback is allowed.
    If exact intraoral_1/2/3 is missing, that patient is not in the relevant
    photo-common cohort.

Photo usage
-----------
Spee:
    intraoral_2 + intraoral_3
    shared frozen encoder
    pair feature = [mean(emb2, emb3), abs(emb2 - emb3)]
    => order-invariant 1024-D representation

Wilson:
    intraoral_1
    => 512-D representation

Render usage
------------
Spee:
    spee_left.png + spee_right.png
    shared frozen encoder
    pair feature = [mean(left,right), abs(left-right)]
    => order-invariant 1024-D

Wilson:
    wilson_frontal.png + wilson_oblique.png
    fixed semantic views
    feature = [frontal, oblique]
    => 1024-D

3D features
-----------
Reuses the EXACT feature construction from 02_evaluate_curve_geometry.py:
    Spee extended   ~101 dims
    Wilson extended ~220 dims

Models
------
3D baseline:
    Spee            direct ExtraTrees (selected in 02)
    Wilson          hierarchical Logistic (selected in 02)
    Wilson location direct ExtraTrees (selected in 02)

Image/fusion:
    Logistic Regression with strict TRAIN-ONLY inner CV.

For Spee/Wilson, both:
    direct
    hierarchical
are tested for every image/fusion family.

Primary selection metric
------------------------
Exact original-label field Accuracy.
Macro-F1 then Balanced Accuracy are tie-breakers.

Image encoder
-------------
Default:
    torchvision ResNet18 ImageNet weights, frozen, avgpool output (512-D)

Preprocessing:
    preserve aspect ratio
    letterbox to 224x224 using white background
    ImageNet normalization

Outputs
-------
evaluation_multimodal_v1/
    curve_multimodal_model_comparison_v1.csv
    curve_multimodal_fold_metrics_v1.csv
    curve_multimodal_oof_predictions_v1.csv
    curve_multimodal_confusions_v1.json
    curve_multimodal_summary_v1.json

Embedding cache
---------------
multimodal_v3/
    resnet18_frozen_embeddings_v1.npz

Run
---
conda activate odin_baseline
cd algorithm/runtime/frozen/curves/code
python 04_evaluate_curve_multimodal.py

Useful quick smoke test:
    python 04_evaluate_curve_multimodal.py --embedding-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageOps
except ImportError as e:
    raise SystemExit("Pillow is required.") from e

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import functional as TF
except ImportError as e:
    raise SystemExit(
        "torch + torchvision are required. Activate the environment used for "
        "your previous ResNet18 experiments."
    ) from e

try:
    from sklearn.metrics import confusion_matrix
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Activate odin_baseline."
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_RAW = Path("data/raw_data")
DEFAULT_GEOM = ROOT / "results" / "curve_geometry_features_v1.csv"
DEFAULT_MM_MANIFEST = ROOT / "multimodal_v3" / "manifest_v1.csv"
DEFAULT_EVAL02 = ROOT / "02_evaluate_curve_geometry.py"
DEFAULT_OUTPUT = ROOT / "evaluation_multimodal_v3"
DEFAULT_CACHE = ROOT / "multimodal_v3" / "resnet18_frozen_embeddings_v2.npz"

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SPEE_FULL_RAW = (
    "Increased",
    "Normal",
    "Reversed",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)

WILSON_FULL_RAW = (
    "Increased",
    "Normal",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)

LOCATION_RAW = (
    "Not_Mentioned",
    "Lower_Arch",
    "Not_Assessable",
)

SPEE_VISUAL_MODEL = (
    "Increased",
    "Normal",
    "Reversed",
)

WILSON_VISUAL_MODEL = (
    "Increased",
    "Normal",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--geometry-features", type=Path, default=DEFAULT_GEOM)
    p.add_argument("--multimodal-manifest", type=Path, default=DEFAULT_MM_MANIFEST)
    p.add_argument("--eval02-script", type=Path, default=DEFAULT_EVAL02)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
    )
    p.add_argument(
        "--embedding-only",
        action="store_true",
        help="Only build/update the frozen image embedding cache, then exit.",
    )
    p.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Ignore any existing cache and re-encode every image.",
    )

    return p.parse_args()


def load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_eval02(mod):
    required = (
        "load_gt",
        "bool_value",
        "spee_compact_columns",
        "add_spee_relative_features",
        "wilson_compact_columns",
        "wilson_extended_columns",
        "safe_outer_splits",
        "metrics",
        "collapse_spee",
        "collapse_wilson",
        "run_direct_model",
        "run_hierarchical",
    )
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise ImportError(
            f"02_evaluate_curve_geometry.py missing required functions: {missing}"
        )



def image_is_decodable(path):
    """
    Return True only if Pillow can actually decode the image.

    A path existing on disk is NOT enough: some dataset entries have a file
    with an image extension that is empty/corrupt/not a real image.

    We intentionally do not substitute another intraoral view when the exact
    required view is corrupt.
    """
    if path is None:
        return False

    p = Path(path)
    if not p.exists() or not p.is_file():
        return False

    try:
        with Image.open(p) as im:
            im.verify()

        # verify() checks structure but invalidates the decoder state, so reopen
        # and force a tiny real decode as an additional safety check.
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im.convert("RGB").load()

        return True
    except Exception:
        return False


def path_pair_decodable(path_a, path_b):
    return (
        bool(path_a)
        and bool(path_b)
        and str(path_a).lower() != "nan"
        and str(path_b).lower() != "nan"
        and image_is_decodable(path_a)
        and image_is_decodable(path_b)
    )


def exact_photo_by_index(
    patient_dir: Path,
    index: int,
):
    """
    STRICT exact view lookup.

    Accept:
        intraoral_1.jpg
        intraoral_1.jpeg
        intraoral_1.png
        ...

    Reject fallback:
        if intraoral_1 is absent, DO NOT substitute intraoral_2.
    """
    d = patient_dir / "intraoral-photo"
    if not d.exists():
        return None

    pattern = re.compile(
        rf"^intraoral[_-]?{int(index)}$",
        flags=re.I,
    )

    hits = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in PHOTO_EXTS:
            continue
        if pattern.match(p.stem):
            hits.append(p)

    if not hits:
        return None

    # If several files match the same exact view number, use the first one
    # that is genuinely decodable. Never fall back to another view number.
    for p in sorted(
        hits,
        key=lambda x: x.name.lower(),
    ):
        if image_is_decodable(p):
            return p

    return None


def add_strict_photo_paths(
    df: pd.DataFrame,
    raw_root: Path,
):
    photo1 = []
    photo2 = []
    photo3 = []

    for pid in df["patient_id"].astype(str):
        d = raw_root / pid
        p1 = exact_photo_by_index(d, 1)
        p2 = exact_photo_by_index(d, 2)
        p3 = exact_photo_by_index(d, 3)

        photo1.append(
            "" if p1 is None else str(p1.resolve())
        )
        photo2.append(
            "" if p2 is None else str(p2.resolve())
        )
        photo3.append(
            "" if p3 is None else str(p3.resolve())
        )

    df = df.copy()
    df["strict_photo_1_path"] = photo1
    df["strict_photo_2_path"] = photo2
    df["strict_photo_3_path"] = photo3

    df["strict_photo_1_ready"] = df[
        "strict_photo_1_path"
    ].ne("")
    df["strict_photo_2_ready"] = df[
        "strict_photo_2_path"
    ].ne("")
    df["strict_photo_3_ready"] = df[
        "strict_photo_3_path"
    ].ne("")
    df["strict_spee_side_pair_ready"] = (
        df["strict_photo_2_ready"]
        & df["strict_photo_3_ready"]
    )

    return df


def resolve_device(name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is not available"
            )
        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def letterbox_tensor(
    path: str,
    size: int = 224,
):
    """
    Preserve the complete image and its aspect ratio.

    White padding is deliberate:
      - STL renders already use white background
      - oral photographs are not cropped at the lateral edges
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid image dimensions: {path}"
            )

        scale = min(
            size / float(w),
            size / float(h),
        )
        nw = max(
            1,
            int(round(w * scale)),
        )
        nh = max(
            1,
            int(round(h * scale)),
        )

        im = im.resize(
            (nw, nh),
            resample=Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (size, size),
            color=(255, 255, 255),
        )
        left = (size - nw) // 2
        top = (size - nh) // 2
        canvas.paste(
            im,
            (left, top),
        )

        x = TF.to_tensor(canvas)
        x = TF.normalize(
            x,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return x


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            x = letterbox_tensor(p)
        except Exception as e:
            raise RuntimeError(
                f"Image decode failed during embedding: {p} | {type(e).__name__}: {e}"
            ) from e
        return p, x


class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        try:
            model = resnet18(
                weights=weights,
            )
        except Exception as e:
            raise RuntimeError(
                "Could not load torchvision ResNet18 ImageNet weights. "
                "If this machine is offline, make sure the weights are already "
                "cached from your previous ResNet18 experiments."
            ) from e

        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    def forward(self, x):
        y = self.features(x)
        return y.flatten(1)


def load_embedding_cache(path: Path):
    if not path.exists():
        return {}

    data = np.load(
        path,
        allow_pickle=False,
    )

    paths = data["paths"].astype(str)
    embeddings = data[
        "embeddings"
    ].astype(np.float32)

    if len(paths) != len(embeddings):
        raise ValueError(
            f"Corrupt embedding cache: {path}"
        )

    return {
        p: embeddings[i]
        for i, p in enumerate(paths)
    }


def save_embedding_cache(
    path: Path,
    mapping: Dict[str, np.ndarray],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        mapping.keys()
    )
    embeddings = np.vstack(
        [
            mapping[p]
            for p in paths
        ]
    ).astype(np.float32)

    np.savez_compressed(
        path,
        paths=np.asarray(
            paths,
            dtype=str,
        ),
        embeddings=embeddings,
    )


def encode_missing_images(
    needed_paths: Sequence[str],
    cache_path: Path,
    device,
    batch_size,
    num_workers,
    rebuild,
):
    all_paths = sorted(
        set(
            str(Path(p).expanduser().resolve())
            for p in needed_paths
            if p
        )
    )

    missing_files = [
        p
        for p in all_paths
        if not Path(p).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} requested images do not exist. "
            f"First: {missing_files[:3]}"
        )

    mapping = (
        {}
        if rebuild
        else load_embedding_cache(
            cache_path
        )
    )

    todo = [
        p
        for p in all_paths
        if p not in mapping
    ]

    print()
    print("=" * 104)
    print("IMAGE EMBEDDING CACHE")
    print("=" * 104)
    print(f"unique images needed : {len(all_paths)}")
    print(f"already cached       : {len(all_paths) - len(todo)}")
    print(f"need encoding        : {len(todo)}")
    print(f"device               : {device}")
    print("=" * 104)

    if not todo:
        return mapping

    model = FrozenResNet18().to(
        device
    )

    ds = PathDataset(todo)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            int(num_workers) > 0
        ),
    )

    done = 0
    t0 = time.time()

    with torch.inference_mode():
        for bi, (paths, x) in enumerate(
            loader,
            1,
        ):
            x = x.to(
                device,
                non_blocking=True,
            )

            emb = model(x).detach().cpu().numpy().astype(
                np.float32
            )

            for p, e in zip(
                paths,
                emb,
            ):
                mapping[
                    str(
                        Path(p)
                        .expanduser()
                        .resolve()
                    )
                ] = e

            done += len(paths)

            if (
                bi == 1
                or bi % 10 == 0
                or done == len(todo)
            ):
                elapsed = max(
                    time.time() - t0,
                    1e-6,
                )
                speed = done / elapsed
                print(
                    f"[Embedding] "
                    f"{done:5d}/{len(todo):5d} "
                    f"({100*done/len(todo):5.1f}%) | "
                    f"{speed:6.1f} img/s"
                )

                # Persist partial progress. If one unexpected file changes or
                # becomes unreadable later, rerunning will continue from cache.
                save_embedding_cache(
                    cache_path,
                    mapping,
                )

    save_embedding_cache(
        cache_path,
        mapping,
    )

    return mapping


def emb(
    mapping,
    path,
):
    key = str(
        Path(path)
        .expanduser()
        .resolve()
    )
    if key not in mapping:
        raise KeyError(
            f"Embedding not found: {key}"
        )
    return mapping[key]


def order_invariant_pair_feature(
    e1,
    e2,
):
    """
    [mean, absolute difference] is invariant to swapping the two views.
    """
    e1 = np.asarray(
        e1,
        dtype=np.float32,
    )
    e2 = np.asarray(
        e2,
        dtype=np.float32,
    )

    return np.concatenate(
        [
            0.5 * (e1 + e2),
            np.abs(e1 - e2),
        ],
        axis=0,
    )


def make_image_feature_matrices(
    df,
    mapping,
    task,
):
    """
    Build ONLY the image modalities required by the current task.

    This is intentionally task-specific.

    Spee common cohort guarantees:
        exact intraoral_2
        exact intraoral_3
        spee_left render
        spee_right render

    Wilson / Wilson-location common cohort guarantees:
        exact intraoral_1
        wilson_frontal render
        wilson_oblique render

    Do NOT try to access unrelated modalities. For example, a valid Spee case
    may legitimately have no intraoral_1; that must not break Spee evaluation.
    """
    if task == "spee":
        spee_photo = []
        spee_render = []

        for _, r in df.iterrows():
            e2 = emb(
                mapping,
                r["strict_photo_2_path"],
            )
            e3 = emb(
                mapping,
                r["strict_photo_3_path"],
            )
            spee_photo.append(
                order_invariant_pair_feature(
                    e2,
                    e3,
                )
            )

            el = emb(
                mapping,
                r["spee_left_render_path"],
            )
            er = emb(
                mapping,
                r["spee_right_render_path"],
            )
            spee_render.append(
                order_invariant_pair_feature(
                    el,
                    er,
                )
            )

        return {
            "spee_photo": np.vstack(
                spee_photo
            ).astype(np.float32),
            "spee_render": np.vstack(
                spee_render
            ).astype(np.float32),
        }

    if task in (
        "wilson",
        "wilson_location",
    ):
        wilson_photo = []
        wilson_render = []

        for _, r in df.iterrows():
            wilson_photo.append(
                emb(
                    mapping,
                    r["strict_photo_1_path"],
                )
            )

            ef = emb(
                mapping,
                r["wilson_frontal_render_path"],
            )
            eo = emb(
                mapping,
                r["wilson_oblique_render_path"],
            )
            wilson_render.append(
                np.concatenate(
                    [ef, eo],
                    axis=0,
                )
            )

        return {
            "wilson_photo": np.vstack(
                wilson_photo
            ).astype(np.float32),
            "wilson_render": np.vstack(
                wilson_render
            ).astype(np.float32),
        }

    raise ValueError(
        f"Unknown image-feature task: {task}"
    )

def dataframe_from_matrix(
    arr,
    prefix,
):
    arr = np.asarray(
        arr,
        dtype=np.float32,
    )
    return pd.DataFrame(
        arr,
        columns=[
            f"{prefix}_{i:04d}"
            for i in range(
                arr.shape[1]
            )
        ],
    )


def prepare_3d_feature_frames(
    merged,
    eval02,
):
    # Exact reconstruction of 02 Spee extended.
    spee_compact = (
        eval02.spee_compact_columns(
            merged
        )
    )
    spee_derived_df, spee_derived_cols = (
        eval02.add_spee_relative_features(
            merged
        )
    )

    spee_3d = pd.concat(
        [
            merged[
                spee_compact
            ].reset_index(drop=True),
            spee_derived_df[
                spee_derived_cols
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    wil_compact = (
        eval02.wilson_compact_columns(
            merged
        )
    )
    wil_extended = (
        eval02.wilson_extended_columns(
            merged,
            wil_compact,
        )
    )
    wilson_3d = merged[
        wil_extended
    ].reset_index(drop=True)

    return (
        spee_3d,
        wilson_3d,
        {
            "spee_extended": int(
                spee_3d.shape[1]
            ),
            "wilson_extended": int(
                wilson_3d.shape[1]
            ),
        },
    )


def normalized_render_scale_features(
    df,
    task,
):
    """
    v3 render is object-centered. Preserve its original metric scale explicitly.

    These are NOT necessary to reconstruct geometry; they simply give the
    linear fusion model a direct signal for the camera's physical scale.
    """
    if task == "spee":
        cols = [
            "spee_left_actual_bbox_width_mm",
            "spee_left_actual_bbox_height_mm",
            "spee_left_used_fov_width_mm",
            "spee_left_used_fov_height_mm",
            "spee_left_mm_per_pixel_mean",
            "spee_right_actual_bbox_width_mm",
            "spee_right_actual_bbox_height_mm",
            "spee_right_used_fov_width_mm",
            "spee_right_used_fov_height_mm",
            "spee_right_mm_per_pixel_mean",
        ]
    elif task == "wilson":
        cols = [
            "wilson_frontal_actual_bbox_width_mm",
            "wilson_frontal_actual_bbox_height_mm",
            "wilson_frontal_used_fov_width_mm",
            "wilson_frontal_used_fov_height_mm",
            "wilson_frontal_mm_per_pixel_mean",
            "wilson_oblique_actual_bbox_width_mm",
            "wilson_oblique_actual_bbox_height_mm",
            "wilson_oblique_used_fov_width_mm",
            "wilson_oblique_used_fov_height_mm",
            "wilson_oblique_mm_per_pixel_mean",
        ]
    else:
        raise ValueError(task)

    available = [
        c
        for c in cols
        if c in df.columns
    ]

    out = df[
        available
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).reset_index(drop=True)

    return out


def add_family(
    families,
    name,
    parts,
):
    frames = [
        p.reset_index(drop=True)
        for p in parts
        if p is not None
        and p.shape[1] > 0
    ]
    families[name] = pd.concat(
        frames,
        axis=1,
    )


def confusion_dict(
    y,
    pred,
):
    labels = sorted(
        set(y.tolist())
        | set(pred.tolist())
    )
    cm = confusion_matrix(
        y,
        pred,
        labels=labels,
    )
    return {
        "labels": labels,
        "matrix": cm.astype(int).tolist(),
    }


def annotate_fold_rows(
    rows,
    task,
    feature_family,
    model_short,
):
    out = []
    for row in rows:
        r = dict(row)
        r["task"] = task
        r["feature_family"] = feature_family
        r["model_short"] = model_short
        out.append(r)
    return out


def run_task(
    *,
    task,
    patient_ids,
    X3d,
    Xrender,
    Xphoto,
    y_raw,
    strat_y,
    eval02,
    args,
    baseline_kind,
    visual_classes=None,
    collapse_fn=None,
    hierarchical_allowed=True,
):
    """
    Returns:
        comparison rows
        fold rows
        oof DataFrame
        confusion dict
    """
    outer_splits = (
        eval02.safe_outer_splits(
            strat_y,
            args.outer_folds,
            args.seed
            + {
                "spee": 0,
                "wilson": 100,
                "wilson_location": 200,
            }[task],
        )
    )

    comparison = []
    fold_rows = []
    outputs = {}

    # --------------------------------------------------------------
    # Majority
    # --------------------------------------------------------------
    majority = pd.Series(
        y_raw
    ).value_counts().index[0]
    pred_majority = np.full(
        len(y_raw),
        majority,
        dtype=object,
    )
    comparison.append(
        {
            "task": task,
            "model": "majority",
            "feature_family": "none",
            **eval02.metrics(
                y_raw,
                pred_majority,
            ),
        }
    )
    outputs["majority"] = pred_majority

    # --------------------------------------------------------------
    # Selected 02-style 3D baseline on THIS exact common cohort
    # --------------------------------------------------------------
    if baseline_kind == "spee_3d_selected":
        pred, fr = eval02.run_direct_model(
            f"{task}_3d_selected_extratrees",
            "extratrees",
            X3d,
            y_raw,
            collapse_fn,
            outer_splits,
            args.inner_folds,
            args.seed + 1000,
            args.n_jobs,
        )
        model_short = "selected_3d"
        comparison.append(
            {
                "task": task,
                "model": model_short,
                "feature_family": "3d",
                **eval02.metrics(
                    y_raw,
                    pred,
                ),
            }
        )
        fold_rows += annotate_fold_rows(
            fr,
            task,
            "3d",
            model_short,
        )
        outputs[model_short] = pred

    elif baseline_kind == "wilson_3d_selected":
        pred, state, vis, fr = (
            eval02.run_hierarchical(
                f"{task}_3d_selected_hierarchical_logistic",
                "logistic",
                X3d,
                y_raw,
                visual_classes,
                "Not_Assessable",
                outer_splits,
                args.inner_folds,
                args.seed + 1100,
                args.n_jobs,
            )
        )
        model_short = "selected_3d"
        comparison.append(
            {
                "task": task,
                "model": model_short,
                "feature_family": "3d",
                **eval02.metrics(
                    y_raw,
                    pred,
                ),
            }
        )
        fold_rows += annotate_fold_rows(
            fr,
            task,
            "3d",
            model_short,
        )
        outputs[model_short] = pred

    elif baseline_kind == "location_3d_selected":
        pred, fr = eval02.run_direct_model(
            f"{task}_3d_selected_extratrees",
            "extratrees",
            X3d,
            y_raw,
            lambda x: x,
            outer_splits,
            args.inner_folds,
            args.seed + 1200,
            args.n_jobs,
        )
        model_short = "selected_3d"
        comparison.append(
            {
                "task": task,
                "model": model_short,
                "feature_family": "3d",
                **eval02.metrics(
                    y_raw,
                    pred,
                ),
            }
        )
        fold_rows += annotate_fold_rows(
            fr,
            task,
            "3d",
            model_short,
        )
        outputs[model_short] = pred

    else:
        raise ValueError(
            baseline_kind
        )

    # --------------------------------------------------------------
    # Frozen image / fusion families
    # --------------------------------------------------------------
    families = {}

    add_family(
        families,
        "render",
        [Xrender],
    )
    add_family(
        families,
        "photo",
        [Xphoto],
    )
    add_family(
        families,
        "3d_render",
        [X3d, Xrender],
    )
    add_family(
        families,
        "3d_photo",
        [X3d, Xphoto],
    )
    add_family(
        families,
        "3d_render_photo",
        [X3d, Xrender, Xphoto],
    )

    for fi, (
        family_name,
        X,
    ) in enumerate(
        families.items(),
        1,
    ):
        print()
        print(
            f"[{task}] "
            f"{fi}/{len(families)} "
            f"family={family_name} "
            f"dims={X.shape[1]}"
        )

        # Direct Logistic
        if collapse_fn is None:
            direct_collapse = (
                lambda x: x
            )
        else:
            direct_collapse = (
                collapse_fn
            )

        pred, fr = eval02.run_direct_model(
            f"{task}_{family_name}_direct_logistic",
            "logistic",
            X,
            y_raw,
            direct_collapse,
            outer_splits,
            args.inner_folds,
            args.seed
            + 2000
            + 100 * fi,
            args.n_jobs,
        )

        short = (
            f"{family_name}__direct_logistic"
        )
        comparison.append(
            {
                "task": task,
                "model": short,
                "feature_family": family_name,
                **eval02.metrics(
                    y_raw,
                    pred,
                ),
            }
        )
        fold_rows += annotate_fold_rows(
            fr,
            task,
            family_name,
            short,
        )
        outputs[short] = pred

        # Hierarchical Logistic for Spee / Wilson only.
        if hierarchical_allowed:
            hp, sp, vp, hfr = (
                eval02.run_hierarchical(
                    f"{task}_{family_name}_hierarchical_logistic",
                    "logistic",
                    X,
                    y_raw,
                    visual_classes,
                    "Not_Assessable",
                    outer_splits,
                    args.inner_folds,
                    args.seed
                    + 4000
                    + 100 * fi,
                    args.n_jobs,
                )
            )

            hshort = (
                f"{family_name}__hierarchical_logistic"
            )
            comparison.append(
                {
                    "task": task,
                    "model": hshort,
                    "feature_family": family_name,
                    **eval02.metrics(
                        y_raw,
                        hp,
                    ),
                }
            )
            fold_rows += annotate_fold_rows(
                hfr,
                task,
                family_name,
                hshort,
            )
            outputs[hshort] = hp

    # OOF table
    oof = pd.DataFrame(
        {
            "patient_id": np.asarray(
                patient_ids,
                dtype=object,
            ),
            "task": task,
            "gt": y_raw,
        }
    )
    for name, pred in outputs.items():
        oof[
            f"pred__{name}"
        ] = pred

    confusions = {
        name: confusion_dict(
            y_raw,
            pred,
        )
        for name, pred in outputs.items()
    }

    return (
        comparison,
        fold_rows,
        oof,
        confusions,
    )


def main():
    args = parse_args()

    args.gt = args.gt.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.geometry_features = args.geometry_features.expanduser().resolve()
    args.multimodal_manifest = args.multimodal_manifest.expanduser().resolve()
    args.eval02_script = args.eval02_script.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.embedding_cache = args.embedding_cache.expanduser().resolve()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval02 = load_module(
        args.eval02_script,
        "_curve_eval02_for_multimodal",
    )
    require_eval02(
        eval02
    )

    gt = eval02.load_gt(
        args.gt
    )
    geom = pd.read_csv(
        args.geometry_features,
        low_memory=False,
    )
    manifest = pd.read_csv(
        args.multimodal_manifest,
        low_memory=False,
    )

    # Avoid duplicate columns from geometry and multimodal QC.
    merged = (
        gt.loc[
            gt["gt_source"].eq(
                "photo_en"
            )
        ]
        .merge(
            geom,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            manifest,
            on="patient_id",
            how="left",
            suffixes=("", "__mm"),
            validate="one_to_one",
        )
        .sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )

    merged = add_strict_photo_paths(
        merged,
        args.raw_root,
    )

    # --------------------------------------------------------------
    # Decode-aware render preflight.
    #
    # Manifest boolean alone is insufficient if a previous run was interrupted
    # or a PNG later became corrupt. Require BOTH the manifest flag and actual
    # Pillow decode success.
    # --------------------------------------------------------------
    spee_render_manifest_ready = (
        merged["spee_render_ready"]
        .map(eval02.bool_value)
        .to_numpy(bool)
    )
    wilson_render_manifest_ready = (
        merged["wilson_render_ready"]
        .map(eval02.bool_value)
        .to_numpy(bool)
    )

    spee_render_decode_ready = np.asarray(
        [
            path_pair_decodable(a, b)
            for a, b in zip(
                merged["spee_left_render_path"].astype(str),
                merged["spee_right_render_path"].astype(str),
            )
        ],
        dtype=bool,
    )
    wilson_render_decode_ready = np.asarray(
        [
            path_pair_decodable(a, b)
            for a, b in zip(
                merged["wilson_frontal_render_path"].astype(str),
                merged["wilson_oblique_render_path"].astype(str),
            )
        ],
        dtype=bool,
    )

    spee_render_ready = (
        spee_render_manifest_ready
        & spee_render_decode_ready
    )
    wilson_render_ready = (
        wilson_render_manifest_ready
        & wilson_render_decode_ready
    )

    spee_geom_ready = (
        merged[
            "spee_geometry_any_valid"
        ]
        .map(eval02.bool_value)
        .to_numpy(bool)
    )
    wilson_geom_ready = (
        merged[
            "wilson_geometry_any_valid"
        ]
        .map(eval02.bool_value)
        .to_numpy(bool)
    )

    strict_spee_photo = (
        merged[
            "strict_spee_side_pair_ready"
        ].to_numpy(bool)
    )
    strict_wilson_photo = (
        merged[
            "strict_photo_1_ready"
        ].to_numpy(bool)
    )

    spee_label_ready = (
        merged[
            "gt_curve_spee"
        ]
        .isin(
            SPEE_FULL_RAW
        )
        .to_numpy(bool)
    )
    wilson_label_ready = (
        merged[
            "gt_curve_wilson"
        ]
        .isin(
            WILSON_FULL_RAW
        )
        .to_numpy(bool)
    )
    loc_label_ready = (
        merged[
            "gt_curve_wilson_location"
        ]
        .isin(
            LOCATION_RAW
        )
        .to_numpy(bool)
    )

    spee_common_mask = (
        spee_label_ready
        & spee_geom_ready
        & spee_render_ready
        & strict_spee_photo
    )
    wilson_common_mask = (
        wilson_label_ready
        & wilson_geom_ready
        & wilson_render_ready
        & strict_wilson_photo
    )
    location_common_mask = (
        loc_label_ready
        & wilson_geom_ready
        & wilson_render_ready
        & strict_wilson_photo
    )

    print("=" * 104)
    print("Curve multimodal strict OOF screening v3 (task-specific image features)")
    print("=" * 104)
    print(
        f"photo_en rows             : {len(merged)}"
    )
    print(
        f"strict exact intraoral_1  : "
        f"{int(merged['strict_photo_1_ready'].sum())}"
    )
    print(
        f"strict exact intraoral_2  : "
        f"{int(merged['strict_photo_2_ready'].sum())}"
    )
    print(
        f"strict exact intraoral_3  : "
        f"{int(merged['strict_photo_3_ready'].sum())}"
    )
    print(
        f"strict side 2+3 pair      : "
        f"{int(merged['strict_spee_side_pair_ready'].sum())}"
    )
    print(
        f"decodable Spee render pair: "
        f"{int(spee_render_ready.sum())}"
    )
    print(
        f"decodable Wilson pair     : "
        f"{int(wilson_render_ready.sum())}"
    )
    print(
        f"Spee common cohort        : "
        f"{int(spee_common_mask.sum())}"
    )
    print(
        f"Wilson common cohort      : "
        f"{int(wilson_common_mask.sum())}"
    )
    print(
        f"Wilson-location cohort    : "
        f"{int(location_common_mask.sum())}"
    )
    print("=" * 104)

    # ------------------------------------------------------------------
    # Determine images needed across ALL strict task cohorts.
    # ------------------------------------------------------------------
    needed = []

    for mask in (
        spee_common_mask,
        wilson_common_mask,
        location_common_mask,
    ):
        sub = merged.loc[mask]

        for c in (
            "strict_photo_1_path",
            "strict_photo_2_path",
            "strict_photo_3_path",
            "spee_left_render_path",
            "spee_right_render_path",
            "wilson_frontal_render_path",
            "wilson_oblique_render_path",
        ):
            if c in sub.columns:
                needed.extend(
                    [
                        x
                        for x in sub[c].astype(str)
                        if x
                        and x.lower() != "nan"
                    ]
                )

    device = resolve_device(
        args.device
    )

    embedding_map = encode_missing_images(
        needed,
        args.embedding_cache,
        device,
        args.batch_size,
        args.num_workers,
        args.rebuild_embeddings,
    )

    if args.embedding_only:
        print()
        print(
            f"Embedding-only mode complete: "
            f"{args.embedding_cache}"
        )
        return

    # ------------------------------------------------------------------
    # 3D extended feature frames, same construction as 02.
    # ------------------------------------------------------------------
    spee_3d_all, wilson_3d_all, feature_counts = (
        prepare_3d_feature_frames(
            merged,
            eval02,
        )
    )

    all_comparison = []
    all_folds = []
    all_oof = []
    all_confusions = {}

    # ==================================================================
    # SPEE
    # ==================================================================
    spee_df = (
        merged.loc[
            spee_common_mask
        ]
        .reset_index(drop=True)
    )
    spee_idx = np.where(
        spee_common_mask
    )[0]

    spee_images = (
        make_image_feature_matrices(
            spee_df,
            embedding_map,
            task="spee",
        )
    )

    spee_render = dataframe_from_matrix(
        spee_images["spee_render"],
        "spee_render_emb",
    )
    spee_photo = dataframe_from_matrix(
        spee_images["spee_photo"],
        "spee_photo_emb",
    )

    # Preserve render physical scale metadata.
    spee_render_scale = (
        normalized_render_scale_features(
            spee_df,
            "spee",
        )
    )
    spee_render = pd.concat(
        [
            spee_render,
            spee_render_scale,
        ],
        axis=1,
    )

    spee_3d = (
        spee_3d_all.iloc[
            spee_idx
        ]
        .reset_index(drop=True)
    )

    y_spee = (
        spee_df[
            "gt_curve_spee"
        ]
        .astype(str)
        .to_numpy(object)
    )
    strat_spee = np.asarray(
        [
            eval02.collapse_spee(x)
            for x in y_spee
        ],
        dtype=object,
    )

    (
        comp,
        folds,
        oof,
        conf,
    ) = run_task(
        task="spee",
        patient_ids=spee_df["patient_id"].astype(str).to_numpy(object),
        X3d=spee_3d,
        Xrender=spee_render,
        Xphoto=spee_photo,
        y_raw=y_spee,
        strat_y=strat_spee,
        eval02=eval02,
        args=args,
        baseline_kind="spee_3d_selected",
        visual_classes=SPEE_VISUAL_MODEL,
        collapse_fn=eval02.collapse_spee,
        hierarchical_allowed=True,
    )
    all_comparison += comp
    all_folds += folds
    all_oof.append(oof)
    all_confusions["spee"] = conf

    # ==================================================================
    # WILSON
    # ==================================================================
    wil_df = (
        merged.loc[
            wilson_common_mask
        ]
        .reset_index(drop=True)
    )
    wil_idx = np.where(
        wilson_common_mask
    )[0]

    wil_images = (
        make_image_feature_matrices(
            wil_df,
            embedding_map,
            task="wilson",
        )
    )

    wil_render = dataframe_from_matrix(
        wil_images["wilson_render"],
        "wilson_render_emb",
    )
    wil_photo = dataframe_from_matrix(
        wil_images["wilson_photo"],
        "wilson_photo_emb",
    )

    wil_render_scale = (
        normalized_render_scale_features(
            wil_df,
            "wilson",
        )
    )
    wil_render = pd.concat(
        [
            wil_render,
            wil_render_scale,
        ],
        axis=1,
    )

    wil_3d = (
        wilson_3d_all.iloc[
            wil_idx
        ]
        .reset_index(drop=True)
    )

    y_wil = (
        wil_df[
            "gt_curve_wilson"
        ]
        .astype(str)
        .to_numpy(object)
    )
    strat_wil = np.asarray(
        [
            eval02.collapse_wilson(x)
            for x in y_wil
        ],
        dtype=object,
    )

    (
        comp,
        folds,
        oof,
        conf,
    ) = run_task(
        task="wilson",
        patient_ids=wil_df["patient_id"].astype(str).to_numpy(object),
        X3d=wil_3d,
        Xrender=wil_render,
        Xphoto=wil_photo,
        y_raw=y_wil,
        strat_y=strat_wil,
        eval02=eval02,
        args=args,
        baseline_kind="wilson_3d_selected",
        visual_classes=WILSON_VISUAL_MODEL,
        collapse_fn=eval02.collapse_wilson,
        hierarchical_allowed=True,
    )
    all_comparison += comp
    all_folds += folds
    all_oof.append(oof)
    all_confusions["wilson"] = conf

    # ==================================================================
    # WILSON LOCATION
    # ==================================================================
    loc_df = (
        merged.loc[
            location_common_mask
        ]
        .reset_index(drop=True)
    )
    loc_idx = np.where(
        location_common_mask
    )[0]

    loc_images = (
        make_image_feature_matrices(
            loc_df,
            embedding_map,
            task="wilson_location",
        )
    )

    loc_render = dataframe_from_matrix(
        loc_images["wilson_render"],
        "wilson_render_emb",
    )
    loc_photo = dataframe_from_matrix(
        loc_images["wilson_photo"],
        "wilson_photo_emb",
    )

    loc_render_scale = (
        normalized_render_scale_features(
            loc_df,
            "wilson",
        )
    )
    loc_render = pd.concat(
        [
            loc_render,
            loc_render_scale,
        ],
        axis=1,
    )

    loc_3d = (
        wilson_3d_all.iloc[
            loc_idx
        ]
        .reset_index(drop=True)
    )

    y_loc = (
        loc_df[
            "gt_curve_wilson_location"
        ]
        .astype(str)
        .to_numpy(object)
    )

    (
        comp,
        folds,
        oof,
        conf,
    ) = run_task(
        task="wilson_location",
        patient_ids=loc_df["patient_id"].astype(str).to_numpy(object),
        X3d=loc_3d,
        Xrender=loc_render,
        Xphoto=loc_photo,
        y_raw=y_loc,
        strat_y=y_loc,
        eval02=eval02,
        args=args,
        baseline_kind="location_3d_selected",
        visual_classes=None,
        collapse_fn=None,
        hierarchical_allowed=False,
    )
    all_comparison += comp
    all_folds += folds
    all_oof.append(oof)
    all_confusions["wilson_location"] = conf

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    comparison_df = (
        pd.DataFrame(
            all_comparison
        )
        .sort_values(
            [
                "task",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    fold_df = pd.DataFrame(
        all_folds
    )
    oof_df = pd.concat(
        all_oof,
        ignore_index=True,
        sort=False,
    )

    best_by_task = {}
    for task in (
        "spee",
        "wilson",
        "wilson_location",
    ):
        sub = comparison_df.loc[
            comparison_df[
                "task"
            ].eq(task)
        ]
        if not sub.empty:
            best_by_task[
                task
            ] = sub.iloc[
                0
            ].to_dict()

    comparison_path = (
        args.output_dir
        / "curve_multimodal_model_comparison_v1.csv"
    )
    fold_path = (
        args.output_dir
        / "curve_multimodal_fold_metrics_v1.csv"
    )
    oof_path = (
        args.output_dir
        / "curve_multimodal_oof_predictions_v1.csv"
    )
    confusion_path = (
        args.output_dir
        / "curve_multimodal_confusions_v1.json"
    )
    summary_path = (
        args.output_dir
        / "curve_multimodal_summary_v1.json"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )
    oof_df.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_path.write_text(
        json.dumps(
            all_confusions,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": "curve-multimodal-v3-task-specific-image-features",
        "gt_source": "photo_en_only",
        "image_encoder": {
            "architecture": "ResNet18",
            "weights": "ImageNet DEFAULT",
            "frozen": True,
            "embedding_dim_per_image": 512,
            "preprocessing": "aspect-preserving white letterbox 224x224 + ImageNet normalization",
        },
        "strict_photo_policy": {
            "fallback_allowed": False,
            "decode_success_required": True,
            "corrupt_or_unreadable_exact_view": "treated as modality unavailable; patient excluded only from that strict common cohort",
            "spee_photo": "exact decodable intraoral_2 + exact decodable intraoral_3",
            "wilson_photo": "exact decodable intraoral_1",
        },
        "cohorts": {
            "photo_en_rows": int(
                len(merged)
            ),
            "strict_intraoral_1": int(
                merged[
                    "strict_photo_1_ready"
                ].sum()
            ),
            "strict_intraoral_2": int(
                merged[
                    "strict_photo_2_ready"
                ].sum()
            ),
            "strict_intraoral_3": int(
                merged[
                    "strict_photo_3_ready"
                ].sum()
            ),
            "strict_side_2_3_pair": int(
                merged[
                    "strict_spee_side_pair_ready"
                ].sum()
            ),
            "spee_common_n": int(
                spee_common_mask.sum()
            ),
            "wilson_common_n": int(
                wilson_common_mask.sum()
            ),
            "wilson_location_common_n": int(
                location_common_mask.sum()
            ),
        },
        "feature_counts": {
            **feature_counts,
            "spee_photo_embedding": int(
                spee_photo.shape[1]
            ),
            "spee_render_embedding_plus_scale": int(
                spee_render.shape[1]
            ),
            "wilson_photo_embedding": int(
                wil_photo.shape[1]
            ),
            "wilson_render_embedding_plus_scale": int(
                wil_render.shape[1]
            ),
        },
        "best_by_task": best_by_task,
        "selection_rule": (
            "Highest exact original-label OOF Accuracy on each task's strict "
            "common cohort. Macro-F1 then Balanced Accuracy are tie-breakers."
        ),
        "important_interpretation": (
            "This is a frozen-encoder screening stage. If a photo/render family "
            "beats the recomputed selected 3D baseline, fine-tune only that "
            "winning image configuration in the next stage."
        ),
        "bugfix_v3": (
            "Image feature construction is task-specific: Spee no longer "
            "requires intraoral_1/Wilson images, and Wilson no longer requires "
            "intraoral_2/3/Spee images."
        ),
        "outputs": {
            "comparison": str(
                comparison_path
            ),
            "fold_metrics": str(
                fold_path
            ),
            "oof": str(
                oof_path
            ),
            "confusions": str(
                confusion_path
            ),
            "embedding_cache": str(
                args.embedding_cache
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("MULTIMODAL RESULTS")
    print("=" * 104)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("BEST BY TASK")
    print("-" * 104)
    for task, row in best_by_task.items():
        print(
            f"{task:18s} | "
            f"{row['model']:38s} | "
            f"Acc={row['accuracy']:.4f} | "
            f"MacroF1={row['macro_f1']:.4f}"
        )

    print()
    print("Outputs:")
    print(f"  {comparison_path}")
    print(f"  {fold_path}")
    print(f"  {oof_path}")
    print(f"  {confusion_path}")
    print(f"  {summary_path}")
    print(f"  {args.embedding_cache}")
    print("=" * 104)


if __name__ == "__main__":
    main()
