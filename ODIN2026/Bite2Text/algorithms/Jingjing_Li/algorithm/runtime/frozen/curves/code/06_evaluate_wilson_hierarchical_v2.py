#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_evaluate_wilson_hierarchical_v2.py

LAST structured Wilson attempt before freeze.

Why this exists
---------------
The previous Wilson model reached about 60% exact field Accuracy, but its
confusion matrix showed two very different problems were being forced into one
multiclass classifier:

    A) report/assessability state:
           Not_Assessable  vs  assessable

    B) visual Wilson state, conditional on assessable:
           Increased  vs  Normal

This script explicitly separates those two decisions.

It also uses information that the previous targeted CNN largely ignored:
    - exact intraoral_1 ... intraoral_5 when decodable
    - photo quality / visibility features
    - frozen ResNet18 embeddings from all available exact views
    - Wilson frontal + oblique STL render embeddings
    - the existing 220-D Wilson geometry
    - lower posterior pair-focused derived geometry for:
          34/44, 35/45, 36/46, 37/47

The evaluation is STRICT NESTED OOF:
    outer 5-fold = final unbiased OOF evaluation
    inner 3-fold = choose:
        gate feature/model family
        visual feature/model family
        P(Not_Assessable) threshold
        P(Increased) threshold
    then selected models are refit on FULL outer-train
    outer validation is predicted exactly once

Primary selection metric:
    exact original-label field Accuracy

Rare original labels:
    Decreased     -> trained as Normal for the visual branch
    Not_Mentioned -> trained as Not_Assessable for the gate branch

BUT final scoring remains against the exact original GT string, so these rare
labels are still counted wrong unless the exact raw label is emitted.
This script intentionally emits only:
    Increased / Normal / Not_Assessable
because the two other labels have ~1 case each and are not worth sacrificing
plain Accuracy for.

Expected run
------------
conda activate odin_baseline
cd algorithm/runtime/frozen/curves/code

# Recommended: build/update cache first. Existing 6058-image cache is copied
# and only missing intraoral_4/5 (etc.) are encoded.
python 06_evaluate_wilson_hierarchical_v2.py --embedding-only

# Then strict nested OOF evaluation.
python 06_evaluate_wilson_hierarchical_v2.py

Outputs
-------
evaluation_wilson_hierarchical_v2/
    wilson_hierarchical_model_comparison_v2.csv
    wilson_hierarchical_fold_metrics_v2.csv
    wilson_hierarchical_oof_predictions_v2.csv
    wilson_hierarchical_confusion_v2.json
    wilson_hierarchical_inner_selection_v2.csv
    wilson_hierarchical_summary_v2.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageOps
except ImportError as e:
    raise SystemExit("Pillow is required.") from e

try:
    import torch
except ImportError as e:
    raise SystemExit("torch is required.") from e

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit("scikit-learn is required.") from e


ROOT = Path(__file__).resolve().parent

DEFAULT_04 = ROOT / "04_evaluate_curve_multimodal_v3.py"
DEFAULT_05 = ROOT / "05_finetune_curve_targeted.py"
DEFAULT_EVAL02 = ROOT / "02_evaluate_curve_geometry.py"

DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_RAW = Path("data/raw_data")
DEFAULT_GEOM = ROOT / "results" / "curve_geometry_features_v1.csv"
DEFAULT_MANIFEST = ROOT / "multimodal_v3" / "manifest_v1.csv"

# Preserve the successful old cache and extend a COPY of it.
DEFAULT_BASE_CACHE = (
    ROOT
    / "multimodal_v3"
    / "resnet18_frozen_embeddings_v2.npz"
)
DEFAULT_CACHE = (
    ROOT
    / "multimodal_v3"
    / "resnet18_frozen_embeddings_wilson_v2.npz"
)

DEFAULT_OUTPUT = ROOT / "evaluation_wilson_hierarchical_v2"

DEFAULT_05_OOF = (
    ROOT
    / "evaluation_finetune_v1"
    / "curve_finetune_oof_predictions_v1.csv"
)
DEFAULT_05_COMPARISON = (
    ROOT
    / "evaluation_finetune_v1"
    / "curve_finetune_model_comparison_v1.csv"
)
DEFAULT_04_COMPARISON = (
    ROOT
    / "evaluation_multimodal_v3"
    / "curve_multimodal_model_comparison_v1.csv"
)

RAW_LABELS = (
    "Increased",
    "Normal",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)

MODEL_LABELS = (
    "Increased",
    "Normal",
    "Not_Assessable",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--script04", type=Path, default=DEFAULT_04)
    p.add_argument("--script05", type=Path, default=DEFAULT_05)
    p.add_argument("--eval02-script", type=Path, default=DEFAULT_EVAL02)

    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--geometry-features", type=Path, default=DEFAULT_GEOM)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    p.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--baseline05-oof", type=Path, default=DEFAULT_05_OOF)
    p.add_argument(
        "--baseline05-comparison",
        type=Path,
        default=DEFAULT_05_COMPARISON,
    )
    p.add_argument(
        "--baseline04-comparison",
        type=Path,
        default=DEFAULT_04_COMPARISON,
    )

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    p.add_argument(
        "--embedding-only",
        action="store_true",
        help="Only update all-5-view frozen embedding cache, then exit.",
    )
    p.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Ignore both cache files and re-encode all needed images.",
    )

    return p.parse_args()


def load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is unavailable"
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def collapse_wilson(x):
    if x == "Decreased":
        return "Normal"
    if x == "Not_Mentioned":
        return "Not_Assessable"
    return x


def exact_metrics(y, pred):
    y = np.asarray(y, dtype=object)
    pred = np.asarray(pred, dtype=object)

    labels = sorted(
        set(y.tolist()) | set(pred.tolist())
    )

    return {
        "n": int(len(y)),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
    }


def confusion_dict(y, pred):
    labels = sorted(
        set(np.asarray(y, dtype=object).tolist())
        | set(np.asarray(pred, dtype=object).tolist())
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


def image_quality_features(path):
    """
    Cheap report/visibility features.

    Output names are fixed so final deployment can reproduce them.
    Missing/unreadable input -> all zeros except ready=0.
    """
    names = [
        "ready",
        "log_width",
        "log_height",
        "aspect",
        "gray_mean",
        "gray_std",
        "dark_frac",
        "bright_frac",
        "sharpness_dxdy_var",
        "edge_energy",
        "entropy_32bin",
        "nonwhite_frac",
    ]

    if not path:
        return np.zeros(len(names), np.float32), names

    p = Path(path)

    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size

            # Small fixed analysis resolution; quality features do not need
            # full image resolution.
            im.thumbnail((384, 384), Image.Resampling.BILINEAR)
            a = np.asarray(im, dtype=np.float32) / 255.0

        gray = (
            0.299 * a[..., 0]
            + 0.587 * a[..., 1]
            + 0.114 * a[..., 2]
        )

        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)

        edge_energy = float(
            0.5 * (
                np.mean(np.abs(dx))
                + np.mean(np.abs(dy))
            )
        )
        sharp = float(
            0.5 * (
                np.var(dx)
                + np.var(dy)
            )
        )

        hist, _ = np.histogram(
            gray,
            bins=32,
            range=(0.0, 1.0),
        )
        prob = hist.astype(np.float64)
        prob /= max(prob.sum(), 1.0)
        nz = prob > 0
        entropy = float(
            -np.sum(prob[nz] * np.log2(prob[nz]))
        )

        feat = np.asarray(
            [
                1.0,
                math.log1p(float(w)),
                math.log1p(float(h)),
                float(w) / max(float(h), 1.0),
                float(gray.mean()),
                float(gray.std()),
                float(np.mean(gray < 0.08)),
                float(np.mean(gray > 0.92)),
                sharp,
                edge_energy,
                entropy,
                float(np.mean(gray < 0.985)),
            ],
            dtype=np.float32,
        )

        return feat, names

    except Exception:
        return np.zeros(len(names), np.float32), names


def add_exact_photo_paths(df, raw_root, base05):
    out = df.copy()

    for idx in range(1, 6):
        paths = []

        for pid in out["patient_id"].astype(str):
            p = base05.exact_photo_by_index(
                raw_root / pid,
                idx,
            )
            paths.append(
                "" if p is None else str(p)
            )

        out[f"strict_photo_{idx}_path"] = paths
        out[f"strict_photo_{idx}_ready"] = (
            out[f"strict_photo_{idx}_path"].ne("")
        )

    return out


def build_quality_matrix(df):
    rows = []
    columns = None

    for _, r in df.iterrows():
        f = []

        for idx in range(1, 6):
            q, names = image_quality_features(
                r[f"strict_photo_{idx}_path"]
            )
            f.append(q)

            if columns is None:
                pass

        rows.append(
            np.concatenate(f).astype(np.float32)
        )

    columns = [
        f"photo{idx}_{name}"
        for idx in range(1, 6)
        for name in image_quality_features("")[1]
    ]

    return (
        np.vstack(rows).astype(np.float32),
        columns,
    )


def safe_numeric_array(df):
    return (
        df.apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(np.float32)
    )


def get_pair_focused_geometry(
    merged,
):
    """
    Explicitly emphasize the lower posterior pairs defining Curve of Wilson.

    Most raw pair measurements already exist somewhere in the old 220-D
    extended feature set. We preserve them explicitly AND add progression /
    cross-pair summary features so a linear model does not need to discover
    these relations by itself.
    """
    use_cols = []

    # Tooth-level local geometry.
    for tooth in (
        34, 35, 36, 37,
        44, 45, 46, 47,
    ):
        prefix = f"wilson_lower_tooth{tooth}_"
        suffixes = (
            "valid",
            "buccal_z_mm",
            "lingual_z_mm",
            "buccal_lingual_span_mm",
            "buccal_minus_lingual_z_mm",
            "transverse_slope_dz_ds",
            "transverse_inclination_deg",
            "buccal_landmark_points",
            "lingual_landmark_points",
            "buccal_vertical_std",
            "lingual_vertical_std",
        )

        for s in suffixes:
            c = prefix + s
            if c in merged.columns:
                use_cols.append(c)

    # Pair-level geometry.
    pair_metric_names = (
        "valid",
        "pair_width_mm",
        "quadratic_coef2_normalized",
        "quadratic_coef1_normalized",
        "fit_rmse_mm",
        "max_chord_depression_mm",
        "max_chord_elevation_mm",
        "mean_chord_depression_mm",
        "left_inclination_deg",
        "right_inclination_deg",
        "mean_inclination_deg",
        "inclination_asymmetry_deg",
        "mean_abs_inclination_deg",
    )

    pairs = (
        "34_44",
        "35_45",
        "36_46",
        "37_47",
    )

    for pair in pairs:
        for metric in pair_metric_names:
            c = f"wilson_lower_pair{pair}_{metric}"
            if c in merged.columns:
                use_cols.append(c)

    base = (
        merged[use_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    derived = {}
    progression_metrics = (
        "max_chord_depression_mm",
        "mean_chord_depression_mm",
        "mean_abs_inclination_deg",
        "mean_inclination_deg",
        "quadratic_coef2_normalized",
        "fit_rmse_mm",
    )

    xcoord = np.arange(
        len(pairs),
        dtype=np.float32,
    )

    for metric in progression_metrics:
        vals = []

        for pair in pairs:
            c = f"wilson_lower_pair{pair}_{metric}"
            if c in merged.columns:
                vals.append(
                    pd.to_numeric(
                        merged[c],
                        errors="coerce",
                    ).to_numpy(np.float32)
                )
            else:
                vals.append(
                    np.full(
                        len(merged),
                        np.nan,
                        dtype=np.float32,
                    )
                )

        a = np.stack(vals, axis=1)

        derived[
            f"pairprog_{metric}_mean"
        ] = np.nanmean(a, axis=1)
        derived[
            f"pairprog_{metric}_std"
        ] = np.nanstd(a, axis=1)
        derived[
            f"pairprog_{metric}_min"
        ] = np.nanmin(a, axis=1)
        derived[
            f"pairprog_{metric}_max"
        ] = np.nanmax(a, axis=1)
        derived[
            f"pairprog_{metric}_first_last"
        ] = a[:, -1] - a[:, 0]

        slope = np.full(
            len(merged),
            np.nan,
            dtype=np.float32,
        )

        for i in range(len(merged)):
            ok = np.isfinite(a[i])
            if ok.sum() >= 2:
                xx = xcoord[ok]
                yy = a[i, ok]
                x0 = xx - xx.mean()
                den = float(
                    np.sum(x0 * x0)
                )
                if den > 0:
                    slope[i] = float(
                        np.sum(
                            x0
                            * (
                                yy - yy.mean()
                            )
                        )
                        / den
                    )

        derived[
            f"pairprog_{metric}_slope"
        ] = slope

    # Validity / missingness itself may predict report assessability.
    valid_cols = [
        f"wilson_lower_pair{p}_valid"
        for p in pairs
        if f"wilson_lower_pair{p}_valid"
        in merged.columns
    ]

    if valid_cols:
        va = (
            merged[valid_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(np.float32)
        )
        derived[
            "lower_valid_pair_count_explicit"
        ] = np.nansum(
            va > 0.5,
            axis=1,
        ).astype(np.float32)

    derived_df = pd.DataFrame(
        derived
    )

    out = pd.concat(
        [
            base,
            derived_df,
        ],
        axis=1,
    )

    return out


def make_photo_embedding_features(
    df,
    embedding_map,
    mod04,
):
    concat_rows = []
    stats_rows = []

    for _, r in df.iterrows():
        embs = []
        masks = []

        for idx in range(1, 6):
            p = r[
                f"strict_photo_{idx}_path"
            ]

            if p:
                e = mod04.emb(
                    embedding_map,
                    p,
                ).astype(np.float32)
                mask = 1.0
            else:
                e = np.zeros(
                    512,
                    dtype=np.float32,
                )
                mask = 0.0

            embs.append(e)
            masks.append(mask)

        e5 = np.stack(
            embs,
            axis=0,
        )
        mask = np.asarray(
            masks,
            dtype=np.float32,
        )

        concat_rows.append(
            np.concatenate(
                [
                    e5.reshape(-1),
                    mask,
                ]
            )
        )

        ok = mask > 0.5
        if ok.any():
            mean = e5[ok].mean(axis=0)
            std = e5[ok].std(axis=0)
        else:
            mean = np.zeros(
                512,
                dtype=np.float32,
            )
            std = np.zeros(
                512,
                dtype=np.float32,
            )

        stats_rows.append(
            np.concatenate(
                [
                    mean,
                    std,
                    mask,
                ]
            )
        )

    return (
        np.vstack(concat_rows).astype(
            np.float32
        ),
        np.vstack(stats_rows).astype(
            np.float32
        ),
    )


def make_render_embedding_features(
    df,
    embedding_map,
    mod04,
):
    rows = []

    for _, r in df.iterrows():
        a = mod04.emb(
            embedding_map,
            r["wilson_frontal_render_path"],
        )
        b = mod04.emb(
            embedding_map,
            r["wilson_oblique_render_path"],
        )
        rows.append(
            np.concatenate(
                [a, b]
            )
        )

    return np.vstack(rows).astype(
        np.float32
    )


def row_geometry_meta(x):
    """
    Very cheap missingness/stability meta-features useful mainly for the
    assessability gate.
    """
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    finite = np.isfinite(x)
    count = finite.sum(axis=1).astype(
        np.float32
    )
    frac = count / max(
        x.shape[1],
        1,
    )

    absx = np.abs(x)

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
        abs_mean = np.nanmean(absx, axis=1)
        abs_max = np.nanmax(absx, axis=1)

    return np.stack(
        [
            count,
            frac,
            mean,
            std,
            abs_mean,
            abs_max,
        ],
        axis=1,
    ).astype(np.float32)


def hstack(*arrs):
    return np.concatenate(
        [
            np.asarray(a, dtype=np.float32)
            for a in arrs
        ],
        axis=1,
    ).astype(np.float32)


def build_feature_families(
    x_geom,
    x_pair,
    x_quality,
    x_photo_concat,
    x_photo_stats,
    x_render,
):
    gmeta = row_geometry_meta(
        hstack(
            x_geom,
            x_pair,
        )
    )

    # Keep gate and visual families intentionally small. Nested CV chooses
    # among these families; outer validation never chooses features.
    gate = {
        "quality_pair": hstack(
            x_quality,
            x_pair,
            gmeta,
        ),
        "geom_quality_pair": hstack(
            x_geom,
            x_pair,
            x_quality,
            gmeta,
        ),
        "photo_stats_geom": hstack(
            x_geom,
            x_pair,
            x_quality,
            x_photo_stats,
            gmeta,
        ),
        "photo_concat_geom": hstack(
            x_geom,
            x_pair,
            x_quality,
            x_photo_concat,
            gmeta,
        ),
        "all_modalities": hstack(
            x_geom,
            x_pair,
            x_quality,
            x_photo_concat,
            x_render,
            gmeta,
        ),
    }

    visual = {
        "geom_pair": hstack(
            x_geom,
            x_pair,
        ),
        "geom_render_pair": hstack(
            x_geom,
            x_pair,
            x_render,
        ),
        "geom_photo_stats_pair": hstack(
            x_geom,
            x_pair,
            x_photo_stats,
        ),
        "geom_photo_concat_pair": hstack(
            x_geom,
            x_pair,
            x_photo_concat,
        ),
        "all_modalities": hstack(
            x_geom,
            x_pair,
            x_photo_concat,
            x_render,
        ),
    }

    return gate, visual


def make_logit(
    *,
    C,
    class_weight=None,
    seed=0,
):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def make_extra(
    *,
    seed,
    n_jobs,
    class_weight=None,
):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    class_weight=class_weight,
                    random_state=int(seed),
                    n_jobs=int(n_jobs),
                ),
            ),
        ]
    )


def candidate_specs(
    gate_features,
    visual_features,
):
    """
    Candidate grids are deliberately finite and fixed before seeing outer OOF.
    """
    gate_specs = {
        "gate_none": {
            "family": None,
            "model": "none",
        },
        "gate_quality_pair_logit": {
            "family": "quality_pair",
            "model": "logit",
            "C": 0.10,
            "class_weight": None,
        },
        "gate_geom_quality_logit": {
            "family": "geom_quality_pair",
            "model": "logit",
            "C": 0.05,
            "class_weight": None,
        },
        "gate_photo_stats_geom_logit": {
            "family": "photo_stats_geom",
            "model": "logit",
            "C": 0.02,
            "class_weight": None,
        },
        "gate_photo_concat_geom_logit": {
            "family": "photo_concat_geom",
            "model": "logit",
            "C": 0.005,
            "class_weight": None,
        },
        "gate_all_logit": {
            "family": "all_modalities",
            "model": "logit",
            "C": 0.005,
            "class_weight": None,
        },
        "gate_all_balanced_logit": {
            "family": "all_modalities",
            "model": "logit",
            "C": 0.005,
            "class_weight": "balanced",
        },
        "gate_geom_quality_extra": {
            "family": "geom_quality_pair",
            "model": "extra",
            "class_weight": "balanced",
        },
    }

    visual_specs = {
        "visual_geom_pair_logit": {
            "family": "geom_pair",
            "model": "logit",
            "C": 0.10,
            "class_weight": None,
        },
        "visual_geom_render_logit": {
            "family": "geom_render_pair",
            "model": "logit",
            "C": 0.03,
            "class_weight": None,
        },
        "visual_geom_photo_stats_logit": {
            "family": "geom_photo_stats_pair",
            "model": "logit",
            "C": 0.03,
            "class_weight": None,
        },
        "visual_geom_photo_concat_logit": {
            "family": "geom_photo_concat_pair",
            "model": "logit",
            "C": 0.005,
            "class_weight": None,
        },
        "visual_all_logit": {
            "family": "all_modalities",
            "model": "logit",
            "C": 0.005,
            "class_weight": None,
        },
        "visual_geom_pair_extra": {
            "family": "geom_pair",
            "model": "extra",
            "class_weight": None,
        },
        "visual_all_extra": {
            "family": "all_modalities",
            "model": "extra",
            "class_weight": None,
        },
    }

    # Fail early if a spelling mismatch is introduced later.
    for spec in gate_specs.values():
        fam = spec.get("family")
        if fam is not None and fam not in gate_features:
            raise KeyError(f"Unknown gate feature family: {fam}")

    for spec in visual_specs.values():
        fam = spec.get("family")
        if fam not in visual_features:
            raise KeyError(f"Unknown visual feature family: {fam}")

    return gate_specs, visual_specs


def build_model_from_spec(
    spec,
    *,
    seed,
    n_jobs,
):
    if spec["model"] == "none":
        return None

    if spec["model"] == "logit":
        return make_logit(
            C=spec["C"],
            class_weight=spec.get(
                "class_weight"
            ),
            seed=seed,
        )

    if spec["model"] == "extra":
        return make_extra(
            seed=seed,
            n_jobs=n_jobs,
            class_weight=spec.get(
                "class_weight"
            ),
        )

    raise ValueError(
        spec["model"]
    )


def positive_probability(
    model,
    x,
    positive_class,
):
    prob = model.predict_proba(x)
    classes = list(
        model.classes_
    )

    if positive_class not in classes:
        return np.zeros(
            len(x),
            dtype=np.float32,
        )

    j = classes.index(
        positive_class
    )

    return prob[:, j].astype(
        np.float32
    )


def inner_oof_probabilities(
    *,
    outer_train_indices,
    y_model,
    feature_families,
    specs,
    eval02,
    inner_folds,
    seed,
    n_jobs,
    branch,
):
    """
    Generate inner OOF probabilities for every candidate.

    branch == gate:
        target 1 = Not_Assessable, 0 = assessable

    branch == visual:
        train only assessable rows
        target 1 = Increased, 0 = Normal
        predict probability for every inner validation row
    """
    outer_train_indices = np.asarray(
        outer_train_indices,
        dtype=int,
    )

    y_outer = y_model[
        outer_train_indices
    ]

    splits = eval02.safe_outer_splits(
        y_outer,
        inner_folds,
        seed,
    )

    out = {
        name: np.full(
            len(outer_train_indices),
            np.nan,
            dtype=np.float32,
        )
        for name in specs
    }

    # Special no-gate candidate.
    if branch == "gate" and "gate_none" in out:
        out["gate_none"][:] = 0.0

    for inner_fold, (
        tr_local,
        va_local,
    ) in enumerate(splits):
        tr_global = outer_train_indices[
            tr_local
        ]
        va_global = outer_train_indices[
            va_local
        ]

        for name, spec in specs.items():
            if spec["model"] == "none":
                continue

            x_all = feature_families[
                spec["family"]
            ]

            if branch == "gate":
                ytr = (
                    y_model[
                        tr_global
                    ]
                    == "Not_Assessable"
                ).astype(int)

                model = build_model_from_spec(
                    spec,
                    seed=(
                        seed
                        + 1000 * inner_fold
                        + abs(hash(name)) % 997
                    ),
                    n_jobs=n_jobs,
                )
                model.fit(
                    x_all[
                        tr_global
                    ],
                    ytr,
                )
                out[name][
                    va_local
                ] = positive_probability(
                    model,
                    x_all[
                        va_global
                    ],
                    1,
                )

            elif branch == "visual":
                assessable_train = (
                    y_model[
                        tr_global
                    ]
                    != "Not_Assessable"
                )
                train_use = tr_global[
                    assessable_train
                ]

                ytr = (
                    y_model[
                        train_use
                    ]
                    == "Increased"
                ).astype(int)

                model = build_model_from_spec(
                    spec,
                    seed=(
                        seed
                        + 2000 * inner_fold
                        + abs(hash(name)) % 997
                    ),
                    n_jobs=n_jobs,
                )
                model.fit(
                    x_all[
                        train_use
                    ],
                    ytr,
                )
                out[name][
                    va_local
                ] = positive_probability(
                    model,
                    x_all[
                        va_global
                    ],
                    1,
                )

            else:
                raise ValueError(branch)

    for name, a in out.items():
        if not np.all(
            np.isfinite(a)
        ):
            raise RuntimeError(
                f"Incomplete inner OOF probabilities: {branch} / {name}"
            )

    return out


def decode_hierarchical(
    p_na,
    p_inc,
    gate_threshold,
    visual_threshold,
):
    p_na = np.asarray(
        p_na,
        dtype=float,
    )
    p_inc = np.asarray(
        p_inc,
        dtype=float,
    )

    pred = np.where(
        p_inc >= float(
            visual_threshold
        ),
        "Increased",
        "Normal",
    ).astype(object)

    pred[
        p_na >= float(
            gate_threshold
        )
    ] = "Not_Assessable"

    return pred


def select_inner_configuration(
    *,
    y_raw_outer_train,
    gate_probs,
    visual_probs,
):
    """
    Jointly select branch families + both thresholds using ONLY inner OOF.

    Tie-breaks:
      1 exact raw Accuracy
      2 weighted F1
      3 macro F1
      4 thresholds closer to 0.5
    """
    gate_thresholds = np.round(
        np.arange(
            0.08,
            0.621,
            0.02,
        ),
        3,
    )
    visual_thresholds = np.round(
        np.arange(
            0.30,
            0.701,
            0.02,
        ),
        3,
    )

    best = None
    rows = []

    for gname, pna in gate_probs.items():
        if gname == "gate_none":
            gts = np.asarray(
                [1.10],
                dtype=float,
            )
        else:
            gts = gate_thresholds

        for vname, pinc in visual_probs.items():
            for tg in gts:
                # Vectorization over visual thresholds is possible, but this
                # loop is tiny compared with fitting the models.
                for tv in visual_thresholds:
                    pred = decode_hierarchical(
                        pna,
                        pinc,
                        tg,
                        tv,
                    )
                    m = exact_metrics(
                        y_raw_outer_train,
                        pred,
                    )

                    na_mask = (
                        np.asarray(
                            y_raw_outer_train,
                            dtype=object,
                        )
                        == "Not_Assessable"
                    )
                    na_recall = (
                        float(
                            np.mean(
                                pred[
                                    na_mask
                                ]
                                == "Not_Assessable"
                            )
                        )
                        if na_mask.any()
                        else 0.0
                    )

                    row = {
                        "gate_model": gname,
                        "visual_model": vname,
                        "gate_threshold": float(
                            tg
                        ),
                        "visual_threshold": float(
                            tv
                        ),
                        **m,
                        "na_recall": na_recall,
                    }
                    rows.append(row)

                    key = (
                        m["accuracy"],
                        m["weighted_f1"],
                        m["macro_f1"],
                        -abs(
                            float(tg) - 0.5
                        ),
                        -abs(
                            float(tv) - 0.5
                        ),
                    )

                    if (
                        best is None
                        or key > best[0]
                    ):
                        best = (
                            key,
                            row,
                        )

    if best is None:
        raise RuntimeError(
            "No inner configuration selected"
        )

    return best[1], pd.DataFrame(rows)


def fit_full_outer_models(
    *,
    outer_train,
    y_model,
    gate_features,
    visual_features,
    gate_specs,
    visual_specs,
    selected,
    seed,
    n_jobs,
):
    gname = selected[
        "gate_model"
    ]
    vname = selected[
        "visual_model"
    ]

    gate_model = None

    if gname != "gate_none":
        gspec = gate_specs[gname]
        gx = gate_features[
            gspec["family"]
        ]
        gy = (
            y_model[
                outer_train
            ]
            == "Not_Assessable"
        ).astype(int)

        gate_model = build_model_from_spec(
            gspec,
            seed=seed + 701,
            n_jobs=n_jobs,
        )
        gate_model.fit(
            gx[
                outer_train
            ],
            gy,
        )

    vspec = visual_specs[
        vname
    ]
    vx = visual_features[
        vspec["family"]
    ]

    assessable = (
        y_model[
            outer_train
        ]
        != "Not_Assessable"
    )
    vtrain = outer_train[
        assessable
    ]

    vy = (
        y_model[
            vtrain
        ]
        == "Increased"
    ).astype(int)

    visual_model = build_model_from_spec(
        vspec,
        seed=seed + 907,
        n_jobs=n_jobs,
    )
    visual_model.fit(
        vx[
            vtrain
        ],
        vy,
    )

    return (
        gate_model,
        visual_model,
    )


def predict_full_outer_models(
    *,
    indices,
    gate_model,
    visual_model,
    gate_features,
    visual_features,
    gate_specs,
    visual_specs,
    selected,
):
    gname = selected[
        "gate_model"
    ]
    vname = selected[
        "visual_model"
    ]

    if gname == "gate_none":
        pna = np.zeros(
            len(indices),
            dtype=np.float32,
        )
    else:
        gspec = gate_specs[gname]
        pna = positive_probability(
            gate_model,
            gate_features[
                gspec["family"]
            ][indices],
            1,
        )

    vspec = visual_specs[
        vname
    ]
    pinc = positive_probability(
        visual_model,
        visual_features[
            vspec["family"]
        ][indices],
        1,
    )

    pred = decode_hierarchical(
        pna,
        pinc,
        selected[
            "gate_threshold"
        ],
        selected[
            "visual_threshold"
        ],
    )

    return pred, pna, pinc


def pairwise_against_baseline(
    oof_df,
    baseline_path,
):
    if not baseline_path.exists():
        return None

    b = pd.read_csv(
        baseline_path
    )

    if "task" in b.columns:
        b = b.loc[
            b["task"].eq("wilson")
        ].copy()

    needed = {
        "patient_id",
        "gt",
        "pred_targeted_finetune",
    }

    if not needed.issubset(
        b.columns
    ):
        return None

    b = b[
        [
            "patient_id",
            "gt",
            "pred_targeted_finetune",
        ]
    ].rename(
        columns={
            "gt": "gt_baseline",
            "pred_targeted_finetune": (
                "pred_baseline"
            ),
        }
    )

    j = oof_df.merge(
        b,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(j) != len(oof_df):
        return {
            "warning": (
                f"baseline overlap {len(j)} != new OOF {len(oof_df)}"
            )
        }

    gt = j["gt"].astype(str).to_numpy()
    newp = j[
        "pred_hierarchical_v2"
    ].astype(str).to_numpy()
    oldp = j[
        "pred_baseline"
    ].astype(str).to_numpy()

    new_ok = newp == gt
    old_ok = oldp == gt

    return {
        "n": int(len(j)),
        "new_correct": int(
            new_ok.sum()
        ),
        "baseline_05_correct": int(
            old_ok.sum()
        ),
        "new_wins_old_wrong": int(
            np.sum(
                new_ok & ~old_ok
            )
        ),
        "old_wins_new_wrong": int(
            np.sum(
                old_ok & ~new_ok
            )
        ),
        "both_correct": int(
            np.sum(
                new_ok & old_ok
            )
        ),
        "both_wrong": int(
            np.sum(
                ~new_ok & ~old_ok
            )
        ),
        "net_correct_gain": int(
            new_ok.sum()
            - old_ok.sum()
        ),
    }


def add_reference_comparison_rows(
    rows,
    path,
    source,
):
    if not path.exists():
        return

    d = pd.read_csv(path)
    d = d.loc[
        d["task"].eq("wilson")
    ].copy()

    for _, r in d.iterrows():
        model = str(r["model"])

        keep = (
            "targeted_finetune"
            in model
            or model == "selected_3d"
            or model
            == "3d_render_photo__hierarchical_logistic"
        )
        if not keep:
            continue

        rows.append(
            {
                "task": "wilson",
                "model": (
                    f"reference__{model}"
                ),
                "source": source,
                "n": int(r["n"]),
                "accuracy": float(
                    r["accuracy"]
                ),
                "macro_f1": float(
                    r["macro_f1"]
                ),
                "weighted_f1": float(
                    r["weighted_f1"]
                ),
                "balanced_accuracy": float(
                    r["balanced_accuracy"]
                ),
                "correct_count": int(
                    round(
                        float(
                            r["accuracy"]
                        )
                        * int(
                            r["n"]
                        )
                    )
                ),
            }
        )


def main():
    args = parse_args()

    for attr in (
        "script04",
        "script05",
        "eval02_script",
        "gt",
        "raw_root",
        "geometry_features",
        "manifest",
        "base_cache",
        "embedding_cache",
        "output_dir",
        "baseline05_oof",
        "baseline05_comparison",
        "baseline04_comparison",
    ):
        setattr(
            args,
            attr,
            getattr(
                args,
                attr,
            ).expanduser().resolve(),
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        args.seed
    )
    device = resolve_device(
        args.device
    )

    mod04 = load_module(
        args.script04,
        "_curve04_for_wilson_v2",
    )
    base05 = load_module(
        args.script05,
        "_curve05_for_wilson_v2",
    )
    eval02 = load_module(
        args.eval02_script,
        "_curve02_for_wilson_v2",
    )

    base05.require_eval02(
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
        args.manifest,
        low_memory=False,
    )

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
        .reset_index(drop=True)
    )

    merged = add_exact_photo_paths(
        merged,
        args.raw_root,
        base05,
    )

    render_ready = np.asarray(
        [
            base05.path_pair_decodable(
                a,
                b,
            )
            for a, b in zip(
                merged[
                    "wilson_frontal_render_path"
                ].astype(str),
                merged[
                    "wilson_oblique_render_path"
                ].astype(str),
            )
        ],
        dtype=bool,
    )

    geom_ready = (
        merged[
            "wilson_geometry_any_valid"
        ]
        .map(
            eval02.bool_value
        )
        .to_numpy(bool)
    )

    # EXACTLY preserve the old Wilson common cohort for fair comparison:
    # valid raw GT + any valid geometry + valid Wilson render pair +
    # exact decodable intraoral_1.
    cohort_mask = (
        merged[
            "gt_curve_wilson"
        ]
        .isin(RAW_LABELS)
        .to_numpy(bool)
        & geom_ready
        & render_ready
        & merged[
            "strict_photo_1_ready"
        ].to_numpy(bool)
    )

    idx = np.where(
        cohort_mask
    )[0]

    cohort = (
        merged.iloc[idx]
        .reset_index(drop=True)
    )

    y_raw = (
        cohort[
            "gt_curve_wilson"
        ]
        .astype(str)
        .to_numpy(object)
    )
    y_model = np.asarray(
        [
            collapse_wilson(x)
            for x in y_raw
        ],
        dtype=object,
    )

    # 220-D exact Wilson extended geometry used in 02/04/05.
    _, wilson_3d_all = (
        base05.prepare_3d_feature_frames(
            merged,
            eval02,
        )
    )
    x_geom = safe_numeric_array(
        wilson_3d_all.iloc[
            idx
        ].reset_index(drop=True)
    )

    # New explicit lower-pair-focused geometry.
    pair_all = get_pair_focused_geometry(
        merged
    )
    x_pair = safe_numeric_array(
        pair_all.iloc[
            idx
        ].reset_index(drop=True)
    )

    # Exact 1..5 photo availability stats.
    exact_counts = {
        str(i): int(
            cohort[
                f"strict_photo_{i}_ready"
            ].sum()
        )
        for i in range(1, 6)
    }

    all5 = int(
        np.sum(
            np.all(
                np.stack(
                    [
                        cohort[
                            f"strict_photo_{i}_ready"
                        ].to_numpy(bool)
                        for i in range(
                            1,
                            6,
                        )
                    ],
                    axis=1,
                ),
                axis=1,
            )
        )
    )

    print("=" * 108)
    print("Wilson hierarchical v2")
    print("=" * 108)
    print(
        f"photo_en rows          : {len(merged)}"
    )
    print(
        f"strict common cohort   : {len(cohort)}"
    )
    print(
        f"GT counts              : {dict(pd.Series(y_raw).value_counts())}"
    )
    print(
        f"exact photo ready 1..5 : {exact_counts}"
    )
    print(
        f"all exact 1..5 ready   : {all5}"
    )
    print(
        f"Wilson 3D dims         : {x_geom.shape[1]}"
    )
    print(
        f"pair-focused dims      : {x_pair.shape[1]}"
    )
    print(
        f"device                  : {device}"
    )
    print("=" * 108)

    # ------------------------------------------------------------------
    # Frozen embedding cache: initialize from prior successful cache.
    # ------------------------------------------------------------------
    if (
        not args.rebuild_embeddings
        and not args.embedding_cache.exists()
        and args.base_cache.exists()
    ):
        args.embedding_cache.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            args.base_cache,
            args.embedding_cache,
        )
        print(
            f"Initialized Wilson-v2 cache from:\n"
            f"  {args.base_cache}\n"
            f"to:\n"
            f"  {args.embedding_cache}"
        )

    needed_paths = []

    for _, r in cohort.iterrows():
        for i in range(1, 6):
            p = r[
                f"strict_photo_{i}_path"
            ]
            if p:
                needed_paths.append(p)

        needed_paths.extend(
            [
                r[
                    "wilson_frontal_render_path"
                ],
                r[
                    "wilson_oblique_render_path"
                ],
            ]
        )

    embedding_map = (
        mod04.encode_missing_images(
            needed_paths=needed_paths,
            cache_path=args.embedding_cache,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rebuild=args.rebuild_embeddings,
        )
    )

    if args.embedding_only:
        print()
        print(
            "Embedding-only complete:"
        )
        print(
            f"  {args.embedding_cache}"
        )
        return

    # ------------------------------------------------------------------
    # Feature matrices
    # ------------------------------------------------------------------
    t_feat = time.time()

    x_quality, quality_columns = (
        build_quality_matrix(
            cohort
        )
    )
    (
        x_photo_concat,
        x_photo_stats,
    ) = make_photo_embedding_features(
        cohort,
        embedding_map,
        mod04,
    )
    x_render = (
        make_render_embedding_features(
            cohort,
            embedding_map,
            mod04,
        )
    )

    gate_features, visual_features = (
        build_feature_families(
            x_geom=x_geom,
            x_pair=x_pair,
            x_quality=x_quality,
            x_photo_concat=x_photo_concat,
            x_photo_stats=x_photo_stats,
            x_render=x_render,
        )
    )

    gate_specs, visual_specs = (
        candidate_specs(
            gate_features,
            visual_features,
        )
    )

    print()
    print("Feature families:")
    for name, x in gate_features.items():
        print(
            f"  gate   {name:28s} {x.shape[1]:5d} dims"
        )
    for name, x in visual_features.items():
        print(
            f"  visual {name:28s} {x.shape[1]:5d} dims"
        )
    print(
        f"Feature preparation time: "
        f"{time.time() - t_feat:.1f}s"
    )

    # ------------------------------------------------------------------
    # Strict outer OOF
    # ------------------------------------------------------------------
    outer_splits = (
        eval02.safe_outer_splits(
            y_model,
            args.outer_folds,
            args.seed + 100,
        )
    )

    oof_pred = np.empty(
        len(cohort),
        dtype=object,
    )
    oof_p_na = np.full(
        len(cohort),
        np.nan,
        dtype=np.float32,
    )
    oof_p_inc = np.full(
        len(cohort),
        np.nan,
        dtype=np.float32,
    )
    oof_fold = np.full(
        len(cohort),
        -1,
        dtype=int,
    )

    fold_rows = []
    selection_rows = []

    for fold, (
        outer_tr,
        outer_va,
    ) in enumerate(
        outer_splits
    ):
        print()
        print("=" * 108)
        print(
            f"WILSON HIERARCHICAL | Fold {fold+1}/{len(outer_splits)}"
        )
        print("=" * 108)
        print(
            f"outer train / val : "
            f"{len(outer_tr)} / {len(outer_va)}"
        )

        t0 = time.time()
        inner_seed = (
            args.seed
            + 10000
            + 1000 * fold
        )

        gate_probs = (
            inner_oof_probabilities(
                outer_train_indices=outer_tr,
                y_model=y_model,
                feature_families=gate_features,
                specs=gate_specs,
                eval02=eval02,
                inner_folds=args.inner_folds,
                seed=inner_seed,
                n_jobs=args.n_jobs,
                branch="gate",
            )
        )

        visual_probs = (
            inner_oof_probabilities(
                outer_train_indices=outer_tr,
                y_model=y_model,
                feature_families=visual_features,
                specs=visual_specs,
                eval02=eval02,
                inner_folds=args.inner_folds,
                seed=inner_seed + 317,
                n_jobs=args.n_jobs,
                branch="visual",
            )
        )

        selected, inner_grid = (
            select_inner_configuration(
                y_raw_outer_train=y_raw[
                    outer_tr
                ],
                gate_probs=gate_probs,
                visual_probs=visual_probs,
            )
        )

        selected_row = {
            "fold": int(fold),
            **selected,
        }
        selection_rows.append(
            selected_row
        )

        print(
            "Selected inner configuration:"
        )
        print(
            f"  gate            : {selected['gate_model']}"
        )
        print(
            f"  visual          : {selected['visual_model']}"
        )
        print(
            f"  P(NA) threshold : {selected['gate_threshold']:.3f}"
        )
        print(
            f"  P(Inc) threshold: {selected['visual_threshold']:.3f}"
        )
        print(
            f"  inner Acc       : {selected['accuracy']:.4f}"
        )
        print(
            f"  inner NA recall : {selected['na_recall']:.4f}"
        )

        gate_model, visual_model = (
            fit_full_outer_models(
                outer_train=np.asarray(
                    outer_tr,
                    dtype=int,
                ),
                y_model=y_model,
                gate_features=gate_features,
                visual_features=visual_features,
                gate_specs=gate_specs,
                visual_specs=visual_specs,
                selected=selected,
                seed=inner_seed + 991,
                n_jobs=args.n_jobs,
            )
        )

        (
            pred,
            pna,
            pinc,
        ) = predict_full_outer_models(
            indices=np.asarray(
                outer_va,
                dtype=int,
            ),
            gate_model=gate_model,
            visual_model=visual_model,
            gate_features=gate_features,
            visual_features=visual_features,
            gate_specs=gate_specs,
            visual_specs=visual_specs,
            selected=selected,
        )

        oof_pred[
            outer_va
        ] = pred
        oof_p_na[
            outer_va
        ] = pna
        oof_p_inc[
            outer_va
        ] = pinc
        oof_fold[
            outer_va
        ] = int(fold)

        fm = exact_metrics(
            y_raw[
                outer_va
            ],
            pred,
        )

        raw_val = y_raw[
            outer_va
        ]

        na_mask = (
            raw_val
            == "Not_Assessable"
        )
        visual_mask = np.isin(
            raw_val,
            [
                "Increased",
                "Normal",
            ],
        )

        na_recall = (
            float(
                np.mean(
                    pred[
                        na_mask
                    ]
                    == "Not_Assessable"
                )
            )
            if na_mask.any()
            else 0.0
        )

        visual_acc = (
            float(
                np.mean(
                    pred[
                        visual_mask
                    ]
                    == raw_val[
                        visual_mask
                    ]
                )
            )
            if visual_mask.any()
            else 0.0
        )

        elapsed = (
            time.time() - t0
        )

        fold_rows.append(
            {
                "task": "wilson",
                "model": (
                    "hierarchical_v2"
                ),
                "fold": int(fold),
                "outer_train_n": int(
                    len(outer_tr)
                ),
                "outer_val_n": int(
                    len(outer_va)
                ),
                "gate_model": (
                    selected[
                        "gate_model"
                    ]
                ),
                "visual_model": (
                    selected[
                        "visual_model"
                    ]
                ),
                "gate_threshold": float(
                    selected[
                        "gate_threshold"
                    ]
                ),
                "visual_threshold": float(
                    selected[
                        "visual_threshold"
                    ]
                ),
                "inner_selected_accuracy": float(
                    selected[
                        "accuracy"
                    ]
                ),
                "outer_accuracy": float(
                    fm[
                        "accuracy"
                    ]
                ),
                "outer_macro_f1": float(
                    fm[
                        "macro_f1"
                    ]
                ),
                "outer_weighted_f1": float(
                    fm[
                        "weighted_f1"
                    ]
                ),
                "outer_balanced_accuracy": float(
                    fm[
                        "balanced_accuracy"
                    ]
                ),
                "outer_na_recall": (
                    na_recall
                ),
                "outer_visual_increased_normal_accuracy": (
                    visual_acc
                ),
                "elapsed_seconds": float(
                    elapsed
                ),
            }
        )

        print(
            f"OUTER Fold {fold+1}: "
            f"Acc={fm['accuracy']:.4f} | "
            f"visual(I/N)={visual_acc:.4f} | "
            f"NA recall={na_recall:.4f} | "
            f"MacroF1={fm['macro_f1']:.4f} | "
            f"time={elapsed/60:.1f} min"
        )

    if (
        np.any(
            oof_fold < 0
        )
        or not np.all(
            np.isfinite(
                oof_p_na
            )
        )
        or not np.all(
            np.isfinite(
                oof_p_inc
            )
        )
    ):
        raise RuntimeError(
            "Incomplete outer OOF"
        )

    overall = exact_metrics(
        y_raw,
        oof_pred,
    )

    na_mask = (
        y_raw
        == "Not_Assessable"
    )
    visual_mask = np.isin(
        y_raw,
        [
            "Increased",
            "Normal",
        ],
    )

    overall_na_recall = float(
        np.mean(
            oof_pred[
                na_mask
            ]
            == "Not_Assessable"
        )
    )
    overall_visual_acc = float(
        np.mean(
            oof_pred[
                visual_mask
            ]
            == y_raw[
                visual_mask
            ]
        )
    )

    oof_df = pd.DataFrame(
        {
            "patient_id": (
                cohort[
                    "patient_id"
                ]
                .astype(str)
                .to_numpy()
            ),
            "task": "wilson",
            "outer_fold": oof_fold,
            "gt": y_raw,
            "pred_hierarchical_v2": (
                oof_pred
            ),
            "p_not_assessable": (
                oof_p_na
            ),
            "p_increased_if_assessable": (
                oof_p_inc
            ),
            "correct": (
                y_raw
                == oof_pred
            ),
        }
    )

    pairwise = pairwise_against_baseline(
        oof_df,
        args.baseline05_oof,
    )

    comparison_rows = [
        {
            "task": "wilson",
            "model": (
                "hierarchical_v2"
            ),
            "source": "06",
            **overall,
            "correct_count": int(
                np.sum(
                    y_raw
                    == oof_pred
                )
            ),
        }
    ]

    add_reference_comparison_rows(
        comparison_rows,
        args.baseline05_comparison,
        "05_targeted",
    )
    add_reference_comparison_rows(
        comparison_rows,
        args.baseline04_comparison,
        "04_multimodal",
    )

    # Deduplicate identical reference names/results.
    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
        .drop_duplicates(
            subset=[
                "model",
                "source",
                "n",
                "accuracy",
            ]
        )
        .sort_values(
            [
                "accuracy",
                "weighted_f1",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    fold_df = pd.DataFrame(
        fold_rows
    )
    selection_df = pd.DataFrame(
        selection_rows
    )

    confusion = confusion_dict(
        y_raw,
        oof_pred,
    )

    comparison_path = (
        args.output_dir
        / "wilson_hierarchical_model_comparison_v2.csv"
    )
    fold_path = (
        args.output_dir
        / "wilson_hierarchical_fold_metrics_v2.csv"
    )
    oof_path = (
        args.output_dir
        / "wilson_hierarchical_oof_predictions_v2.csv"
    )
    confusion_path = (
        args.output_dir
        / "wilson_hierarchical_confusion_v2.json"
    )
    selection_path = (
        args.output_dir
        / "wilson_hierarchical_inner_selection_v2.csv"
    )
    summary_path = (
        args.output_dir
        / "wilson_hierarchical_summary_v2.json"
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
    selection_df.to_csv(
        selection_path,
        index=False,
        encoding="utf-8-sig",
    )
    confusion_path.write_text(
        json.dumps(
            confusion,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    acc = overall[
        "accuracy"
    ]

    if acc >= 0.70:
        stop_rule = (
            "SUCCESS_70_PLUS_FREEZE"
        )
    elif acc >= 0.69:
        stop_rule = (
            "VERY_CLOSE_ALLOW_ONE_CALIBRATION_ONLY"
        )
    elif acc >= 0.67:
        stop_rule = (
            "PROMISING_ALLOW_ONE_SMALL_CALIBRATION_ONLY"
        )
    elif acc >= 0.64:
        stop_rule = (
            "IMPROVED_BUT_BELOW_TARGET_STOP_RECOMMENDED"
        )
    else:
        stop_rule = (
            "STOP_AND_FREEZE_BEST_PREVIOUS_MODEL"
        )

    summary = {
        "version": (
            "wilson-hierarchical-v2-all5-pair-focused"
        ),
        "gt_source": (
            "photo_en_only"
        ),
        "common_n": int(
            len(cohort)
        ),
        "gt_counts": {
            str(k): int(v)
            for k, v in (
                pd.Series(
                    y_raw
                )
                .value_counts()
                .items()
            )
        },
        "exact_photo_ready_counts": (
            exact_counts
        ),
        "all_exact_1_to_5_ready": int(
            all5
        ),
        "feature_dimensions": {
            "wilson_extended_3d": int(
                x_geom.shape[1]
            ),
            "pair_focused_geometry": int(
                x_pair.shape[1]
            ),
            "photo_quality": int(
                x_quality.shape[1]
            ),
            "photo_concat_embeddings": int(
                x_photo_concat.shape[1]
            ),
            "photo_mean_std_embeddings": int(
                x_photo_stats.shape[1]
            ),
            "render_embeddings": int(
                x_render.shape[1]
            ),
        },
        "cv_policy": {
            "outer_folds": int(
                args.outer_folds
            ),
            "inner_folds": int(
                args.inner_folds
            ),
            "outer_validation_used_for_model_or_threshold_selection": False,
            "selected_models_refit_on_full_outer_train": True,
            "primary_metric": (
                "exact original-label Accuracy"
            ),
        },
        "result": {
            **overall,
            "correct_count": int(
                np.sum(
                    y_raw
                    == oof_pred
                )
            ),
            "not_assessable_recall": (
                overall_na_recall
            ),
            "increased_normal_only_accuracy": (
                overall_visual_acc
            ),
        },
        "pairwise_vs_05_targeted": (
            pairwise
        ),
        "stop_rule": stop_rule,
        "target_interpretation": {
            "70_percent_requires_correct": int(
                math.ceil(
                    0.70
                    * len(cohort)
                )
            ),
            "current_correct": int(
                np.sum(
                    y_raw
                    == oof_pred
                )
            ),
        },
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
            "confusion": str(
                confusion_path
            ),
            "inner_selection": str(
                selection_path
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
    print("=" * 108)
    print("WILSON HIERARCHICAL v2 FINAL")
    print("=" * 108)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print(
        f"New exact OOF         : "
        f"{summary['result']['correct_count']}/{len(cohort)} "
        f"= {100*overall['accuracy']:.2f}%"
    )
    print(
        f"Increased/Normal Acc  : "
        f"{100*overall_visual_acc:.2f}%"
    )
    print(
        f"Not_Assessable recall : "
        f"{100*overall_na_recall:.2f}%"
    )
    print(
        f"STOP RULE             : "
        f"{stop_rule}"
    )

    if pairwise is not None:
        print()
        print("Pairwise vs 05 targeted:")
        for k, v in pairwise.items():
            print(
                f"  {k:30s}: {v}"
            )

    print()
    print("Outputs:")
    print(f"  {comparison_path}")
    print(f"  {fold_path}")
    print(f"  {oof_path}")
    print(f"  {confusion_path}")
    print(f"  {selection_path}")
    print(f"  {summary_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
