#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_evaluate_curve_geometry.py

Strict nested-OOF evaluation for the rebuilt 3D Curve tasks:

    curve_spee
    curve_wilson
    curve_wilson_location

GT policy
---------
Primary GT:
    cleaned_orthodontic_data_v3.json
    -> photo_report_v3
    -> photo_report_v3_input_source == "photo_en"

Old ios_report labels are NOT used for training.

Metrics
-------
Primary:
    exact field Accuracy

Also:
    Macro-F1
    Weighted-F1
    Balanced Accuracy
    confusion matrix

Why hierarchical models?
------------------------
The audit showed that the final report labels contain both anatomy and
report-state labels:

Spee:
    Increased / Normal / Reversed / Decreased
    Not_Assessable / Not_Mentioned

Wilson:
    Increased / Normal / Decreased
    Not_Assessable / Not_Mentioned

Therefore we compare:

1) majority
2) direct_collapsed_logistic
3) direct_collapsed_extratrees
4) hierarchical_logistic
5) hierarchical_extratrees

"collapsed" sacrifices ultra-rare labels intentionally for exact accuracy:
    Spee:
        Decreased      -> Reversed
        Not_Mentioned  -> Not_Assessable
    Wilson:
        Decreased      -> Normal
        Not_Mentioned  -> Not_Assessable

The reported metric is ALWAYS against the original raw GT label, so those
collapsed cases are still counted wrong unless the original string matches.

For Spee/Wilson we also evaluate visual-only classification separately.

Feature families
----------------
Spee:
    compact  : curated curve-level features
    extended : compact + per-tooth relative-height/stability features

Wilson:
    compact  : curated aggregate features
    extended : compact + per-tooth cusp/inclination + pair-curve features

No absolute raw STL X/Y/Z features are used.

CV
--
- 5 outer folds
- inner folds tune hyperparameters using TRAIN ONLY
- outer validation is untouched
- all models within one task use the exact same outer folds

Outputs
-------
evaluation_geometry_v1/
    curve_geometry_model_comparison_v1.csv
    curve_geometry_fold_metrics_v1.csv
    curve_geometry_oof_predictions_v1.csv
    curve_geometry_confusions_v1.json
    curve_geometry_evaluation_summary_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Activate odin_baseline."
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "results" / "curve_geometry_features_v1.csv"
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_OUTPUT = ROOT / "evaluation_geometry_v1"

ALIGNMENT_KEYS = (
    "4_alignment_and_arch",
    "4_alignment",
    "alignment_and_arch",
)

SPEE_VISUAL_RAW = (
    "Increased",
    "Normal",
    "Reversed",
    "Decreased",
)
SPEE_VISUAL_MODEL = (
    "Increased",
    "Normal",
    "Reversed",
)
SPEE_FULL_RAW = (
    "Increased",
    "Normal",
    "Reversed",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)

WILSON_VISUAL_RAW = (
    "Increased",
    "Normal",
    "Decreased",
)
WILSON_VISUAL_MODEL = (
    "Increased",
    "Normal",
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def get_alignment_block(report):
    if not isinstance(report, dict):
        return {}
    for key in ALIGNMENT_KEYS:
        block = report.get(key)
        if isinstance(block, dict):
            return block
    return {}


def normalize_label(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def load_gt(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for rec in data:
        if not isinstance(rec, dict):
            continue

        report = rec.get("photo_report_v3") or {}
        block = get_alignment_block(report)

        rows.append(
            {
                "patient_id": rec.get("patient_id"),
                "gt_source": rec.get("photo_report_v3_input_source"),
                "gt_curve_spee": normalize_label(block.get("curve_spee")),
                "gt_curve_wilson": normalize_label(block.get("curve_wilson")),
                "gt_curve_wilson_location": normalize_label(
                    block.get("curve_wilson_location")
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient_id in GT.")
    return out


def bool_value(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def numeric_usable(df, cols):
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() >= 20 and df[c].nunique(dropna=True) >= 2:
            out.append(c)
    return out


def spee_compact_columns(df):
    exact = []

    for side in ("left", "right"):
        for suffix in (
            "n_landmarks",
            "ap_span_mm",
            "coef2_normalized",
            "coef1_normalized",
            "fit_rmse_mm",
            "fit_r2",
            "max_chord_depth_mm",
            "max_reverse_elevation_mm",
            "signed_dominant_depth_mm",
            "mean_chord_depth_mm",
            "positive_depth_area_mm",
            "reverse_area_mm",
            "observed_max_depth_mm",
            "observed_max_reverse_mm",
            "observed_mean_depth_mm",
            "posterior_slope_dz_dx",
            "anterior_slope_dz_dx",
        ):
            exact.append(f"spee_{side}_{suffix}")

    exact += [
        c
        for c in df.columns
        if c.startswith("spee_bilateral_")
    ]

    return numeric_usable(df, list(dict.fromkeys(exact)))


def add_spee_relative_features(df):
    out = pd.DataFrame(index=df.index)

    side_teeth = {
        "left": (31,33,34,35,36,37),
        "right": (41,43,44,45,46,47),
    }

    cols = []

    for side, teeth in side_teeth.items():
        zcols = [
            f"spee_{side}_tooth{t}_z_mm"
            for t in teeth
            if f"spee_{side}_tooth{t}_z_mm" in df.columns
        ]

        if not zcols:
            continue

        z = df[zcols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        row_mean = z.mean(axis=1)

        for c in zcols:
            tooth = c.split("_tooth", 1)[1].split("_", 1)[0]
            name = f"derived_spee_{side}_tooth{tooth}_z_centered"
            out[name] = z[c] - row_mean
            cols.append(name)

        # Adjacent height changes along anatomical sequence.
        for a, b in zip(teeth[:-1], teeth[1:]):
            ca = f"spee_{side}_tooth{a}_z_mm"
            cb = f"spee_{side}_tooth{b}_z_mm"
            if ca in df.columns and cb in df.columns:
                name = f"derived_spee_{side}_delta_z_{a}_{b}"
                out[name] = (
                    pd.to_numeric(df[cb], errors="coerce")
                    - pd.to_numeric(df[ca], errors="coerce")
                )
                cols.append(name)

        for t in teeth:
            for suffix in (
                "vertical_std",
                "landmark_points",
            ):
                c = f"spee_{side}_tooth{t}_{suffix}"
                if c in df.columns:
                    name = f"derived_{c}"
                    out[name] = pd.to_numeric(
                        df[c],
                        errors="coerce",
                    )
                    cols.append(name)

    cols = numeric_usable(out, cols)
    return out, cols


def wilson_compact_columns(df):
    cols = []

    for arch in ("lower", "upper"):
        for c in df.columns:
            if not c.startswith(f"wilson_{arch}_"):
                continue

            # aggregates only; per-tooth/pair are added in extended
            if "_tooth" in c or "_pair" in c:
                continue

            allowed_tokens = (
                "valid_teeth",
                "valid_pairs",
                "tooth_inclination_",
                "buccal_minus_lingual_z_",
                "pair_chord_depression_",
                "pair_chord_elevation_",
                "pair_abs_inclination_",
                "pair_inclination_asymmetry_",
                "pair_quadratic_coef2_",
                "posterior_progression_",
            )
            if any(tok in c for tok in allowed_tokens):
                cols.append(c)

    return numeric_usable(df, cols)


def wilson_extended_columns(df, compact):
    cols = list(compact)

    allowed_suffixes = (
        "_buccal_lingual_span_mm",
        "_buccal_minus_lingual_z_mm",
        "_transverse_slope_dz_ds",
        "_transverse_inclination_deg",
        "_buccal_landmark_points",
        "_lingual_landmark_points",
        "_buccal_vertical_std",
        "_lingual_vertical_std",
        "_pair_width_mm",
        "_quadratic_coef2_normalized",
        "_quadratic_coef1_normalized",
        "_fit_rmse_mm",
        "_max_chord_depression_mm",
        "_max_chord_elevation_mm",
        "_mean_chord_depression_mm",
        "_mean_inclination_deg",
        "_inclination_asymmetry_deg",
        "_mean_abs_inclination_deg",
    )

    for c in df.columns:
        if not c.startswith("wilson_"):
            continue
        if any(c.endswith(s) for s in allowed_suffixes):
            cols.append(c)

    return numeric_usable(df, list(dict.fromkeys(cols)))


def safe_outer_splits(y, n_splits, seed):
    vc = pd.Series(y).value_counts()
    k = min(int(n_splits), int(vc.min()))
    if k < 2:
        raise RuntimeError(
            f"Cannot stratify labels: {vc.to_dict()}"
        )
    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed,
    )
    return list(skf.split(np.zeros((len(y),1)), y))


def metrics(y, pred):
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
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


def make_logistic(C, class_weight, seed):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_extratrees(
    max_depth,
    min_samples_leaf,
    class_weight,
    seed,
    n_jobs,
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
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def inner_splits(y, requested, seed):
    vc = pd.Series(y).value_counts()
    k = min(int(requested), int(vc.min()))
    if k < 2:
        return None
    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed,
    )
    return list(skf.split(np.zeros((len(y),1)), y))


def tune_model(
    family,
    X,
    y,
    inner_folds,
    seed,
    n_jobs,
):
    splits = inner_splits(
        y,
        inner_folds,
        seed,
    )

    if splits is None:
        return None

    candidates = []

    if family == "logistic":
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            for cw in (None, "balanced"):
                candidates.append(
                    {
                        "family": family,
                        "C": C,
                        "class_weight": cw,
                    }
                )

    elif family == "extratrees":
        for depth, leaf in (
            (4, 5),
            (8, 5),
            (None, 5),
            (None, 10),
        ):
            for cw in (None, "balanced"):
                candidates.append(
                    {
                        "family": family,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "class_weight": cw,
                    }
                )
    else:
        raise ValueError(family)

    best = None

    for ci, cfg in enumerate(candidates):
        oof = np.empty(len(y), dtype=object)

        for fi, (tr, va) in enumerate(splits):
            if family == "logistic":
                model = make_logistic(
                    cfg["C"],
                    cfg["class_weight"],
                    seed + 100*ci + fi,
                )
            else:
                model = make_extratrees(
                    cfg["max_depth"],
                    cfg["min_samples_leaf"],
                    cfg["class_weight"],
                    seed + 100*ci + fi,
                    n_jobs,
                )

            model.fit(
                X.iloc[tr],
                y[tr],
            )
            oof[va] = model.predict(
                X.iloc[va]
            )

        m = metrics(y, oof)

        key = (
            m["accuracy"],
            m["macro_f1"],
            m["balanced_accuracy"],
            m["weighted_f1"],
        )

        if best is None or key > best[0]:
            best = (
                key,
                cfg,
                m,
            )

    return {
        "config": best[1],
        "inner_metrics": best[2],
    }


def fit_from_config(cfg, X, y, seed, n_jobs):
    if cfg["family"] == "logistic":
        model = make_logistic(
            cfg["C"],
            cfg["class_weight"],
            seed,
        )
    else:
        model = make_extratrees(
            cfg["max_depth"],
            cfg["min_samples_leaf"],
            cfg["class_weight"],
            seed,
            n_jobs,
        )

    model.fit(X, y)
    return model


def collapse_spee(label):
    if label == "Decreased":
        return "Reversed"
    if label == "Not_Mentioned":
        return "Not_Assessable"
    return label


def collapse_wilson(label):
    if label == "Decreased":
        return "Normal"
    if label == "Not_Mentioned":
        return "Not_Assessable"
    return label


def run_direct_model(
    name,
    family,
    X,
    y_raw,
    collapse_fn,
    outer_splits,
    inner_folds,
    seed,
    n_jobs,
):
    n = len(y_raw)
    pred = np.empty(n, dtype=object)
    fold_rows = []

    y_model = np.asarray(
        [collapse_fn(x) for x in y_raw],
        dtype=object,
    )

    for fold, (tr, va) in enumerate(outer_splits):
        tuned = tune_model(
            family,
            X.iloc[tr].reset_index(drop=True),
            y_model[tr],
            inner_folds,
            seed + 1000*fold,
            n_jobs,
        )
        if tuned is None:
            majority = pd.Series(
                y_model[tr]
            ).value_counts().index[0]
            pv = np.full(len(va), majority, dtype=object)
            cfg = {"family": family, "fallback": "majority"}
        else:
            cfg = tuned["config"]
            model = fit_from_config(
                cfg,
                X.iloc[tr],
                y_model[tr],
                seed + 2000 + fold,
                n_jobs,
            )
            pv = model.predict(
                X.iloc[va]
            ).astype(object)

        pred[va] = pv

        fold_rows.append(
            {
                "model": name,
                "fold": int(fold),
                "config": json.dumps(
                    cfg,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **metrics(
                    y_raw[va],
                    pv,
                ),
            }
        )

    return pred, fold_rows


def run_hierarchical(
    name,
    family,
    X,
    y_raw,
    visual_train_classes,
    nonvisual_decode,
    outer_splits,
    inner_folds,
    seed,
    n_jobs,
):
    """
    state:
        VISUAL vs NONVISUAL
    visual:
        train only labels in visual_train_classes
    """
    n = len(y_raw)
    pred = np.empty(n, dtype=object)
    state_pred_all = np.empty(n, dtype=object)
    visual_pred_all = np.empty(n, dtype=object)
    fold_rows = []

    state_y = np.asarray(
        [
            "VISUAL"
            if x in set(visual_train_classes) or x in ("Decreased",)
            else "NONVISUAL"
            for x in y_raw
        ],
        dtype=object,
    )

    for fold, (tr, va) in enumerate(outer_splits):
        # State model
        tuned_state = tune_model(
            family,
            X.iloc[tr].reset_index(drop=True),
            state_y[tr],
            inner_folds,
            seed + 3000*fold,
            n_jobs,
        )

        if tuned_state is None:
            maj = pd.Series(
                state_y[tr]
            ).value_counts().index[0]
            state_va = np.full(
                len(va),
                maj,
                dtype=object,
            )
            state_cfg = {
                "family": family,
                "fallback": "majority",
            }
        else:
            state_cfg = tuned_state["config"]
            state_model = fit_from_config(
                state_cfg,
                X.iloc[tr],
                state_y[tr],
                seed + 4000 + fold,
                n_jobs,
            )
            state_va = state_model.predict(
                X.iloc[va]
            ).astype(object)

        # Visual model excludes ultra-rare raw classes from training.
        visual_mask_tr = np.isin(
            y_raw[tr],
            visual_train_classes,
        )
        vtr = tr[visual_mask_tr]
        vy = y_raw[vtr]

        tuned_visual = tune_model(
            family,
            X.iloc[vtr].reset_index(drop=True),
            vy,
            inner_folds,
            seed + 5000*fold,
            n_jobs,
        )

        if tuned_visual is None:
            maj = pd.Series(vy).value_counts().index[0]
            visual_va = np.full(
                len(va),
                maj,
                dtype=object,
            )
            visual_cfg = {
                "family": family,
                "fallback": "majority",
            }
        else:
            visual_cfg = tuned_visual["config"]
            visual_model = fit_from_config(
                visual_cfg,
                X.iloc[vtr],
                vy,
                seed + 6000 + fold,
                n_jobs,
            )
            visual_va = visual_model.predict(
                X.iloc[va]
            ).astype(object)

        final_va = np.where(
            state_va == "VISUAL",
            visual_va,
            nonvisual_decode,
        ).astype(object)

        pred[va] = final_va
        state_pred_all[va] = state_va
        visual_pred_all[va] = visual_va

        fold_rows.append(
            {
                "model": name,
                "fold": int(fold),
                "state_config": json.dumps(
                    state_cfg,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "visual_config": json.dumps(
                    visual_cfg,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **metrics(
                    y_raw[va],
                    final_va,
                ),
            }
        )

    return pred, state_pred_all, visual_pred_all, fold_rows


def threshold_rule_spee(
    Xdf,
    y_raw,
    outer_splits,
):
    """
    Simple interpretable VISUAL-only rule using signed bilateral depth.

    Search on each outer TRAIN:
        signed_depth < reverse_threshold -> Reversed
        max_depth    > increased_threshold -> Increased
        else -> Normal

    This is intentionally only a baseline.
    """
    signed_col = "spee_bilateral_signed_dominant_depth_mm_mean"
    depth_col = "spee_bilateral_max_chord_depth_mm_mean"

    signed = pd.to_numeric(
        Xdf[signed_col],
        errors="coerce",
    ).to_numpy(float)
    depth = pd.to_numeric(
        Xdf[depth_col],
        errors="coerce",
    ).to_numpy(float)

    pred = np.empty(len(y_raw), dtype=object)

    for fold, (tr, va) in enumerate(outer_splits):
        # train on model-supported visual labels
        tr2 = tr[
            np.isin(
                y_raw[tr],
                SPEE_VISUAL_MODEL,
            )
        ]

        best = None

        for rt in np.arange(-1.5, 0.81, 0.2):
            for it in np.arange(0.4, 3.01, 0.1):
                p = np.where(
                    signed[tr2] < rt,
                    "Reversed",
                    np.where(
                        depth[tr2] > it,
                        "Increased",
                        "Normal",
                    ),
                )

                m = metrics(
                    y_raw[tr2],
                    p.astype(object),
                )
                key = (
                    m["accuracy"],
                    m["macro_f1"],
                    m["balanced_accuracy"],
                )
                if best is None or key > best[0]:
                    best = (key, float(rt), float(it))

        rt = best[1]
        it = best[2]

        pred[va] = np.where(
            signed[va] < rt,
            "Reversed",
            np.where(
                depth[va] > it,
                "Increased",
                "Normal",
            ),
        ).astype(object)

    return pred


def threshold_rule_wilson(
    Xdf,
    y_raw,
    outer_splits,
):
    """
    Simple visual-only one-threshold baseline.
    Tune both threshold and orientation.
    """
    c = "wilson_lower_pair_abs_inclination_mean"
    x = pd.to_numeric(
        Xdf[c],
        errors="coerce",
    ).to_numpy(float)

    pred = np.empty(len(y_raw), dtype=object)

    for fold, (tr, va) in enumerate(outer_splits):
        tr2 = tr[
            np.isin(
                y_raw[tr],
                WILSON_VISUAL_MODEL,
            )
        ]

        qs = np.nanquantile(
            x[tr2],
            np.linspace(0.05, 0.95, 37),
        )

        best = None

        for threshold in np.unique(qs):
            for high_label in (
                "Increased",
                "Normal",
            ):
                low_label = (
                    "Normal"
                    if high_label == "Increased"
                    else "Increased"
                )

                p = np.where(
                    x[tr2] >= threshold,
                    high_label,
                    low_label,
                ).astype(object)

                m = metrics(
                    y_raw[tr2],
                    p,
                )
                key = (
                    m["accuracy"],
                    m["macro_f1"],
                    m["balanced_accuracy"],
                )

                if best is None or key > best[0]:
                    best = (
                        key,
                        float(threshold),
                        high_label,
                        low_label,
                    )

        threshold, high_label, low_label = (
            best[1],
            best[2],
            best[3],
        )

        pred[va] = np.where(
            x[va] >= threshold,
            high_label,
            low_label,
        ).astype(object)

    return pred


def evaluate_visual_only(
    task_name,
    X,
    y_raw,
    visual_raw,
    visual_model,
    outer_folds,
    seed,
    inner_folds,
    n_jobs,
):
    mask = np.isin(
        y_raw,
        visual_raw,
    )
    idx = np.where(mask)[0]

    y = y_raw[idx]
    Xv = X.iloc[idx].reset_index(drop=True)

    # For robust fold creation, rare Decreased is kept in metric but
    # mapped only for stratification to nearest supported rare class.
    if task_name == "spee":
        strat = np.asarray(
            [
                "Reversed" if x == "Decreased" else x
                for x in y
            ],
            dtype=object,
        )
        collapse_fn = collapse_spee
    else:
        strat = np.asarray(
            [
                "Normal" if x == "Decreased" else x
                for x in y
            ],
            dtype=object,
        )
        collapse_fn = collapse_wilson

    splits = safe_outer_splits(
        strat,
        outer_folds,
        seed,
    )

    pred_log, folds_log = run_direct_model(
        f"{task_name}_visual_logistic",
        "logistic",
        Xv,
        y,
        collapse_fn,
        splits,
        inner_folds,
        seed + 10,
        n_jobs,
    )
    pred_tree, folds_tree = run_direct_model(
        f"{task_name}_visual_extratrees",
        "extratrees",
        Xv,
        y,
        collapse_fn,
        splits,
        inner_folds,
        seed + 20,
        n_jobs,
    )

    majority = pd.Series(
        y
    ).value_counts().index[0]
    pred_maj = np.full(
        len(y),
        majority,
        dtype=object,
    )

    rows = [
        {
            "task": task_name,
            "scope": "visual_only",
            "model": "majority",
            "feature_family": "none",
            **metrics(y, pred_maj),
        },
        {
            "task": task_name,
            "scope": "visual_only",
            "model": "logistic",
            "feature_family": X.attrs.get(
                "feature_family",
                "unknown",
            ),
            **metrics(y, pred_log),
        },
        {
            "task": task_name,
            "scope": "visual_only",
            "model": "extratrees",
            "feature_family": X.attrs.get(
                "feature_family",
                "unknown",
            ),
            **metrics(y, pred_tree),
        },
    ]

    return {
        "idx": idx,
        "y": y,
        "pred_majority": pred_maj,
        "pred_logistic": pred_log,
        "pred_extratrees": pred_tree,
        "comparison": rows,
        "fold_rows": folds_log + folds_tree,
    }


def confusion_dict(y, pred):
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


def main():
    args = parse_args()

    feat_path = args.features.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    feat = pd.read_csv(
        feat_path,
        low_memory=False,
    )
    gt = load_gt(gt_path)

    merged = (
        gt.loc[
            gt["gt_source"].eq("photo_en")
        ]
        .merge(
            feat,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    # ---------------- feature families ----------------
    spee_compact = spee_compact_columns(
        merged
    )
    spee_derived_df, spee_derived_cols = (
        add_spee_relative_features(
            merged
        )
    )
    spee_ext_df = pd.concat(
        [
            merged[spee_compact].reset_index(drop=True),
            spee_derived_df[spee_derived_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    spee_ext_cols = list(
        spee_ext_df.columns
    )

    X_spee_compact = merged[
        spee_compact
    ].copy()
    X_spee_compact.attrs[
        "feature_family"
    ] = "spee_compact"

    X_spee_extended = spee_ext_df.copy()
    X_spee_extended.attrs[
        "feature_family"
    ] = "spee_extended"

    wil_compact = wilson_compact_columns(
        merged
    )
    wil_extended = wilson_extended_columns(
        merged,
        wil_compact,
    )

    X_wil_compact = merged[
        wil_compact
    ].copy()
    X_wil_compact.attrs[
        "feature_family"
    ] = "wilson_compact"

    X_wil_extended = merged[
        wil_extended
    ].copy()
    X_wil_extended.attrs[
        "feature_family"
    ] = "wilson_extended"

    print("=" * 104)
    print("Curve geometry strict OOF v1")
    print("=" * 104)
    print(f"photo_en rows        : {len(merged)}")
    print(f"Spee compact         : {len(spee_compact)} features")
    print(f"Spee extended        : {len(spee_ext_cols)} features")
    print(f"Wilson compact       : {len(wil_compact)} features")
    print(f"Wilson extended      : {len(wil_extended)} features")
    print("=" * 104)

    comparison_rows = []
    fold_rows = []
    confusions = {}
    oof_rows = []

    # ==============================================================
    # SPEE FULL FIELD
    # ==============================================================
    spee_mask = (
        merged["gt_curve_spee"].isin(
            SPEE_FULL_RAW
        ).to_numpy(bool)
        & merged["spee_geometry_any_valid"].map(bool_value).to_numpy(bool)
    )
    spee_idx = np.where(
        spee_mask
    )[0]
    y_spee = (
        merged.loc[
            spee_idx,
            "gt_curve_spee",
        ]
        .astype(str)
        .to_numpy(object)
    )

    # Stratification collapses ultra-rare labels ONLY for fold assignment.
    spee_strat = np.asarray(
        [collapse_spee(x) for x in y_spee],
        dtype=object,
    )
    spee_splits = safe_outer_splits(
        spee_strat,
        args.outer_folds,
        args.seed,
    )

    majority = pd.Series(
        y_spee
    ).value_counts().index[0]
    pred_majority = np.full(
        len(y_spee),
        majority,
        dtype=object,
    )
    comparison_rows.append(
        {
            "task": "spee",
            "scope": "full_field",
            "model": "majority",
            "feature_family": "none",
            **metrics(
                y_spee,
                pred_majority,
            ),
        }
    )

    spee_outputs = {
        "majority": pred_majority,
    }

    for family_name, Xfull in (
        ("compact", X_spee_compact),
        ("extended", X_spee_extended),
    ):
        X = Xfull.iloc[
            spee_idx
        ].reset_index(drop=True)

        for model_family in (
            "logistic",
            "extratrees",
        ):
            name = (
                f"spee_full_direct_{family_name}_{model_family}"
            )
            pred, fr = run_direct_model(
                name,
                model_family,
                X,
                y_spee,
                collapse_spee,
                spee_splits,
                args.inner_folds,
                args.seed + 100,
                args.n_jobs,
            )
            fold_rows += fr
            comparison_rows.append(
                {
                    "task": "spee",
                    "scope": "full_field",
                    "model": f"direct_{model_family}",
                    "feature_family": f"spee_{family_name}",
                    **metrics(
                        y_spee,
                        pred,
                    ),
                }
            )
            spee_outputs[
                f"direct_{family_name}_{model_family}"
            ] = pred

            hname = (
                f"spee_full_hierarchical_{family_name}_{model_family}"
            )
            hp, sp, vp, hfr = run_hierarchical(
                hname,
                model_family,
                X,
                y_spee,
                SPEE_VISUAL_MODEL,
                "Not_Assessable",
                spee_splits,
                args.inner_folds,
                args.seed + 200,
                args.n_jobs,
            )
            fold_rows += hfr
            comparison_rows.append(
                {
                    "task": "spee",
                    "scope": "full_field",
                    "model": f"hierarchical_{model_family}",
                    "feature_family": f"spee_{family_name}",
                    **metrics(
                        y_spee,
                        hp,
                    ),
                }
            )
            spee_outputs[
                f"hierarchical_{family_name}_{model_family}"
            ] = hp

    # interpretable visual threshold baseline, reported on full field too
    Xtmp = merged.iloc[
        spee_idx
    ].reset_index(drop=True)
    threshold_pred = threshold_rule_spee(
        Xtmp,
        y_spee,
        spee_splits,
    )
    comparison_rows.append(
        {
            "task": "spee",
            "scope": "full_field",
            "model": "geometry_threshold_visual_only",
            "feature_family": "2_scalar",
            **metrics(
                y_spee,
                threshold_pred,
            ),
        }
    )
    spee_outputs[
        "geometry_threshold_visual_only"
    ] = threshold_pred

    # Visual-only on both feature families
    for fam, Xfull in (
        ("compact", X_spee_compact),
        ("extended", X_spee_extended),
    ):
        X = Xfull.iloc[
            spee_idx
        ].reset_index(drop=True)
        X.attrs["feature_family"] = f"spee_{fam}"

        vis = evaluate_visual_only(
            "spee",
            X,
            y_spee,
            SPEE_VISUAL_RAW,
            SPEE_VISUAL_MODEL,
            args.outer_folds,
            args.seed + 300,
            args.inner_folds,
            args.n_jobs,
        )
        comparison_rows += vis["comparison"]
        fold_rows += vis["fold_rows"]

    # ==============================================================
    # WILSON FULL FIELD
    # ==============================================================
    wil_mask = (
        merged["gt_curve_wilson"].isin(
            WILSON_FULL_RAW
        ).to_numpy(bool)
        & merged["wilson_geometry_any_valid"].map(bool_value).to_numpy(bool)
    )
    wil_idx = np.where(
        wil_mask
    )[0]
    y_wil = (
        merged.loc[
            wil_idx,
            "gt_curve_wilson",
        ]
        .astype(str)
        .to_numpy(object)
    )
    wil_strat = np.asarray(
        [collapse_wilson(x) for x in y_wil],
        dtype=object,
    )
    wil_splits = safe_outer_splits(
        wil_strat,
        args.outer_folds,
        args.seed + 1,
    )

    majority = pd.Series(
        y_wil
    ).value_counts().index[0]
    pred_majority = np.full(
        len(y_wil),
        majority,
        dtype=object,
    )
    comparison_rows.append(
        {
            "task": "wilson",
            "scope": "full_field",
            "model": "majority",
            "feature_family": "none",
            **metrics(
                y_wil,
                pred_majority,
            ),
        }
    )

    wil_outputs = {
        "majority": pred_majority,
    }

    for family_name, Xfull in (
        ("compact", X_wil_compact),
        ("extended", X_wil_extended),
    ):
        X = Xfull.iloc[
            wil_idx
        ].reset_index(drop=True)

        for model_family in (
            "logistic",
            "extratrees",
        ):
            name = (
                f"wilson_full_direct_{family_name}_{model_family}"
            )
            pred, fr = run_direct_model(
                name,
                model_family,
                X,
                y_wil,
                collapse_wilson,
                wil_splits,
                args.inner_folds,
                args.seed + 400,
                args.n_jobs,
            )
            fold_rows += fr
            comparison_rows.append(
                {
                    "task": "wilson",
                    "scope": "full_field",
                    "model": f"direct_{model_family}",
                    "feature_family": f"wilson_{family_name}",
                    **metrics(
                        y_wil,
                        pred,
                    ),
                }
            )
            wil_outputs[
                f"direct_{family_name}_{model_family}"
            ] = pred

            hname = (
                f"wilson_full_hierarchical_{family_name}_{model_family}"
            )
            hp, sp, vp, hfr = run_hierarchical(
                hname,
                model_family,
                X,
                y_wil,
                WILSON_VISUAL_MODEL,
                "Not_Assessable",
                wil_splits,
                args.inner_folds,
                args.seed + 500,
                args.n_jobs,
            )
            fold_rows += hfr
            comparison_rows.append(
                {
                    "task": "wilson",
                    "scope": "full_field",
                    "model": f"hierarchical_{model_family}",
                    "feature_family": f"wilson_{family_name}",
                    **metrics(
                        y_wil,
                        hp,
                    ),
                }
            )
            wil_outputs[
                f"hierarchical_{family_name}_{model_family}"
            ] = hp

    Xtmp = merged.iloc[
        wil_idx
    ].reset_index(drop=True)
    threshold_pred = threshold_rule_wilson(
        Xtmp,
        y_wil,
        wil_splits,
    )
    comparison_rows.append(
        {
            "task": "wilson",
            "scope": "full_field",
            "model": "geometry_threshold_visual_only",
            "feature_family": "1_scalar",
            **metrics(
                y_wil,
                threshold_pred,
            ),
        }
    )
    wil_outputs[
        "geometry_threshold_visual_only"
    ] = threshold_pred

    for fam, Xfull in (
        ("compact", X_wil_compact),
        ("extended", X_wil_extended),
    ):
        X = Xfull.iloc[
            wil_idx
        ].reset_index(drop=True)
        X.attrs["feature_family"] = f"wilson_{fam}"

        vis = evaluate_visual_only(
            "wilson",
            X,
            y_wil,
            WILSON_VISUAL_RAW,
            WILSON_VISUAL_MODEL,
            args.outer_folds,
            args.seed + 600,
            args.inner_folds,
            args.n_jobs,
        )
        comparison_rows += vis["comparison"]
        fold_rows += vis["fold_rows"]

    # ==============================================================
    # WILSON LOCATION
    # ==============================================================
    loc_mask = (
        merged["gt_curve_wilson_location"].isin(
            LOCATION_RAW
        ).to_numpy(bool)
        & merged["wilson_geometry_any_valid"].map(bool_value).to_numpy(bool)
    )
    loc_idx = np.where(
        loc_mask
    )[0]
    y_loc = (
        merged.loc[
            loc_idx,
            "gt_curve_wilson_location",
        ]
        .astype(str)
        .to_numpy(object)
    )
    loc_splits = safe_outer_splits(
        y_loc,
        args.outer_folds,
        args.seed + 2,
    )

    loc_majority = pd.Series(
        y_loc
    ).value_counts().index[0]
    pred_loc_majority = np.full(
        len(y_loc),
        loc_majority,
        dtype=object,
    )
    comparison_rows.append(
        {
            "task": "wilson_location",
            "scope": "full_field",
            "model": "majority",
            "feature_family": "none",
            **metrics(
                y_loc,
                pred_loc_majority,
            ),
        }
    )

    loc_outputs = {
        "majority": pred_loc_majority,
    }

    for family_name, Xfull in (
        ("compact", X_wil_compact),
        ("extended", X_wil_extended),
    ):
        X = Xfull.iloc[
            loc_idx
        ].reset_index(drop=True)

        # No label collapse needed.
        for model_family in (
            "logistic",
            "extratrees",
        ):
            name = (
                f"location_{family_name}_{model_family}"
            )
            pred, fr = run_direct_model(
                name,
                model_family,
                X,
                y_loc,
                lambda x: x,
                loc_splits,
                args.inner_folds,
                args.seed + 700,
                args.n_jobs,
            )
            fold_rows += fr
            comparison_rows.append(
                {
                    "task": "wilson_location",
                    "scope": "full_field",
                    "model": model_family,
                    "feature_family": f"wilson_{family_name}",
                    **metrics(
                        y_loc,
                        pred,
                    ),
                }
            )
            loc_outputs[
                f"{family_name}_{model_family}"
            ] = pred

    # ------------------------- build comparison --------------------
    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        [
            "task",
            "scope",
            "accuracy",
            "macro_f1",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )
    fold_df = pd.DataFrame(
        fold_rows
    )

    # best full-field per task
    best_by_task = {}
    for task in (
        "spee",
        "wilson",
        "wilson_location",
    ):
        sub = comparison_df.loc[
            (comparison_df["task"] == task)
            & (comparison_df["scope"] == "full_field")
        ]
        if not sub.empty:
            best_by_task[task] = (
                sub.iloc[0].to_dict()
            )

    # confusion matrices for every full-output variant
    for task, y, outputs in (
        ("spee", y_spee, spee_outputs),
        ("wilson", y_wil, wil_outputs),
        ("wilson_location", y_loc, loc_outputs),
    ):
        confusions[task] = {
            model: confusion_dict(
                y,
                pred,
            )
            for model, pred in outputs.items()
        }

    # OOF: compact table, all evaluated cohort patient ids + best candidates
    oof_parts = []

    for task, idx, y, outputs in (
        ("spee", spee_idx, y_spee, spee_outputs),
        ("wilson", wil_idx, y_wil, wil_outputs),
        ("wilson_location", loc_idx, y_loc, loc_outputs),
    ):
        part = pd.DataFrame(
            {
                "patient_id": merged.iloc[idx]["patient_id"].to_numpy(),
                "task": task,
                "gt": y,
            }
        )
        for name, pred in outputs.items():
            part[f"pred__{name}"] = pred
        oof_parts.append(part)

    oof_df = pd.concat(
        oof_parts,
        ignore_index=True,
        sort=False,
    )

    comparison_path = (
        out_dir
        / "curve_geometry_model_comparison_v1.csv"
    )
    fold_path = (
        out_dir
        / "curve_geometry_fold_metrics_v1.csv"
    )
    oof_path = (
        out_dir
        / "curve_geometry_oof_predictions_v1.csv"
    )
    confusion_path = (
        out_dir
        / "curve_geometry_confusions_v1.json"
    )
    summary_path = (
        out_dir
        / "curve_geometry_evaluation_summary_v1.json"
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
            confusions,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": "curve-geometry-evaluation-v1-strict-nested-oof",
        "gt_source": "photo_en_only",
        "cohorts": {
            "photo_en_rows": int(len(merged)),
            "spee_full_n": int(len(y_spee)),
            "wilson_full_n": int(len(y_wil)),
            "wilson_location_n": int(len(y_loc)),
        },
        "feature_counts": {
            "spee_compact": int(len(spee_compact)),
            "spee_extended": int(len(spee_ext_cols)),
            "wilson_compact": int(len(wil_compact)),
            "wilson_extended": int(len(wil_extended)),
        },
        "best_full_field_by_task": best_by_task,
        "selection_rule": (
            "Primary = highest original-label exact field Accuracy. "
            "Macro-F1 then Balanced Accuracy are tie-breakers."
        ),
        "rare_label_policy": {
            "spee": {
                "Decreased": "collapsed to Reversed for model fitting only",
                "Not_Mentioned": "collapsed to Not_Assessable for model fitting only",
            },
            "wilson": {
                "Decreased": "collapsed to Normal for model fitting only",
                "Not_Mentioned": "collapsed to Not_Assessable for model fitting only",
            },
            "important": (
                "Metrics are always computed against original GT strings, "
                "so collapsed rare cases are not credited as exact matches."
            ),
        },
        "outputs": {
            "comparison": str(comparison_path),
            "fold_metrics": str(fold_path),
            "oof": str(oof_path),
            "confusions": str(confusion_path),
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
    print("FULL-FIELD RESULTS")
    print("=" * 104)

    full = comparison_df.loc[
        comparison_df["scope"].eq(
            "full_field"
        )
    ]
    print(
        full.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 104)
    print("VISUAL-ONLY RESULTS")
    print("=" * 104)

    vis = comparison_df.loc[
        comparison_df["scope"].eq(
            "visual_only"
        )
    ]
    print(
        vis.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Outputs:")
    print(f"  {comparison_path}")
    print(f"  {fold_path}")
    print(f"  {oof_path}")
    print(f"  {confusion_path}")
    print(f"  {summary_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
