#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_evaluate_crossbite_fusion_v3.py

Branch-specific strict OOF evaluation for Crossbite with dual-threshold OR.

Why v2?
-------
The previous 04 trained BOTH posterior and anterior geometry against the global
label:

    crossbite_present

That is not the correct supervision for a branch-specific geometry model.

Example:
    GT crossbite teeth = [16, 17]

This patient is globally crossbite-positive, but should be NEGATIVE for the
anterior branch.

v2 therefore derives two branch-specific labels from crossbite_teeth:

    anterior_involvement
    posterior_involvement

and trains the branches independently.

Branch label policy
-------------------
Global negative:
    anterior = 0
    posterior = 0

Anterior-only positive:
    anterior = 1
    posterior = 0

Posterior-only positive:
    anterior = 0
    posterior = 1

Mixed anterior+posterior positive:
    anterior = 1
    posterior = 1

Positive but crossbite_teeth empty / unusable:
    anterior = unknown
    posterior = unknown

Those unknown positive cases are EXCLUDED from branch classifier training, but
they remain in:
    - meta-fusion training
    - final global crossbite_present OOF evaluation

This is intentional: they can teach the meta-fusion how global crossbite looks
from the two branch probabilities without forcing an unsupported anatomical
branch label.

Strict nesting
--------------
One fixed OUTER 5-fold split is used for final OOF evaluation.

Inside every outer-training fold:

    posterior branch:
        train only on known posterior_involvement labels

    anterior branch:
        train only on known anterior_involvement labels

    inner OOF branch probabilities are generated for ALL outer-training
    patients, including teeth-empty positives.

Those inner OOF branch probabilities are then used to:

    1) tune simple OR-type fusion thresholds
    2) train a 2-D Logistic meta-fusion
    3) train a 4-D Logistic meta-fusion with anterior reliability/QC features

The untouched outer-validation fold is never used for branch fitting, meta
fitting, or threshold tuning.

Models compared
---------------
Reference models:
    posterior_global
        Old 52-feature posterior-whole classifier trained on global present.

    anterior_global_raw
        Old low-dimensional anterior classifier trained on global present.

    concat_raw_reference
        52 posterior + 45 anterior raw features trained on global present.

New branch-specific models:
    branch_dual_threshold_or
        Jointly tuned Tposterior and Tanterior on inner OOF, then:
        (Pposterior >= Tposterior) OR (Panterior >= Tanterior)

    branch_max
        max(P_posterior_involvement, P_anterior_involvement)

    branch_noisy_or
        1 - (1-Pposterior)*(1-Panterior)

    branch_meta_logistic
        Logistic meta model on:
            Pposterior, Panterior

    branch_meta_logistic_qc
        Logistic meta model on:
            Pposterior
            Panterior
            anterior_bad_tooth_count
            anterior_any_bad_tooth

Anterior branch geometry
------------------------
Uses the RADIAL INCISAL geometry from 03.

Base features:
    per tooth 13/12/11/21/22/23:
        q20
        median
        negative_ratio

Derived:
    all/right/left robust aggregates

For the branch-specific anterior model, two QC/reliability features are also
included:
    bad_tooth_count
    any_bad_tooth

The raw geometry values are NOT deleted.

Positive subgroup reporting
---------------------------
    anterior_only
    posterior_only
    mixed_anterior_posterior
    teeth_empty
    other
    ALL_POSITIVE

Output
------
evaluation_fusion_v3/

    crossbite_branch_label_coverage_v3.csv
    crossbite_branch_model_comparison_v3.csv
    crossbite_branch_fold_metrics_v3.csv
    crossbite_branch_oof_predictions_v3.csv
    crossbite_branch_subgroup_recall_v3.csv
    crossbite_branch_involvement_metrics_v3.csv
    crossbite_branch_thresholds_v3.csv
    crossbite_branch_summary_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Activate the odin_baseline environment."
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_POSTERIOR = ROOT / "results" / "crossbite_geometry_features_v3.csv"
DEFAULT_ANTERIOR = ROOT / "results" / "anterior_crossbite_geometry_features_v2.csv"
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_OUTPUT = ROOT / "evaluation_fusion_v3"

ANTERIOR_TEETH = (13, 12, 11, 21, 22, 23)
RIGHT_ANTERIOR = (13, 12, 11)
LEFT_ANTERIOR = (21, 22, 23)
SIDES = ("left", "right")

POST_WHOLE_SUFFIXES = (
    "mean_margin_mm",
    "median_margin_mm",
    "q10_margin_mm",
    "q20_margin_mm",
    "min_margin_mm",
    "std_margin_mm",
    "negative_ratio",
    "negative_05_ratio",
    "negative_10_ratio",
    "longest_negative_run_mm",
    "longest_negative_05_run_mm",
    "negative_area_mm2",
    "valid_slice_ratio",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--posterior-features",
        type=Path,
        default=DEFAULT_POSTERIOR,
    )
    p.add_argument(
        "--anterior-features",
        type=Path,
        default=DEFAULT_ANTERIOR,
    )
    p.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=1)

    p.add_argument(
        "--threshold-objective",
        choices=("macro_f1", "balanced_accuracy"),
        default="macro_f1",
    )
    p.add_argument(
        "--dual-grid-min",
        type=float,
        default=0.20,
        help="Minimum threshold for dual-threshold OR grid search.",
    )
    p.add_argument(
        "--dual-grid-max",
        type=float,
        default=0.95,
        help="Maximum threshold for dual-threshold OR grid search.",
    )
    p.add_argument(
        "--dual-grid-step",
        type=float,
        default=0.025,
        help="Grid step for posterior/anterior threshold joint search.",
    )
    p.add_argument(
        "--include-ios-fallback",
        action="store_true",
        help="Diagnostic only; primary evaluation remains photo_en only.",
    )

    return p.parse_args()


def resolve_posterior_features(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.exists():
        return p

    fallback = ROOT / "results" / "crossbite_geometry_features.csv"
    if fallback.exists():
        print(
            "WARNING: requested posterior feature CSV not found; "
            f"using fallback: {fallback}"
        )
        return fallback.resolve()

    raise FileNotFoundError(
        f"Posterior feature CSV not found: {p}"
    )


def bool_value(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in (
        "true",
        "1",
        "yes",
        "y",
    )


def bool_series(s: pd.Series) -> pd.Series:
    return s.map(bool_value).astype(bool)


def require_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    context: str,
) -> None:
    missing = [
        c for c in cols
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing columns for {context}: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )


def load_gt(path: Path) -> pd.DataFrame:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON list: {path}"
        )

    rows = []

    for rec in data:
        if not isinstance(rec, dict):
            continue

        pid = rec.get("patient_id")
        source = rec.get(
            "photo_report_v3_input_source"
        )
        report = rec.get("photo_report_v3") or {}
        trans = (
            report.get("2_transverse_occlusion")
            or {}
        )

        present = trans.get("crossbite_present")
        teeth = trans.get("crossbite_teeth")

        if not isinstance(teeth, list):
            teeth = []

        rows.append(
            {
                "patient_id": pid,
                "gt_source": source,
                "gt_crossbite_present": (
                    int(present)
                    if isinstance(present, bool)
                    else np.nan
                ),
                "gt_crossbite_location": trans.get(
                    "crossbite_location"
                ),
                "gt_crossbite_type": trans.get(
                    "crossbite_type"
                ),
                "gt_crossbite_teeth_list": teeth,
                "gt_crossbite_teeth": (
                    json.dumps(
                        teeth,
                        ensure_ascii=False,
                    )
                ),
            }
        )

    out = pd.DataFrame(rows)

    if out["patient_id"].duplicated().any():
        dups = out.loc[
            out["patient_id"].duplicated(),
            "patient_id",
        ].tolist()[:10]
        raise ValueError(
            f"Duplicate patient_id in GT JSON, e.g. {dups}"
        )

    return out


def prefix_dataframe(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    rename = {
        c: f"{prefix}{c}"
        for c in df.columns
        if c != "patient_id"
    }
    return df.rename(columns=rename)


def safe_numeric(
    df: pd.DataFrame,
    cols: Iterable[str],
) -> List[str]:
    valid = []

    for c in cols:
        if c not in df.columns:
            continue

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        if (
            df[c].notna().sum() >= 5
            and df[c].nunique(dropna=True) >= 2
        ):
            valid.append(c)

    return valid


def posterior_valid_mask(
    df: pd.DataFrame,
) -> pd.Series:
    cols = [
        "post__left_full_valid",
        "post__right_full_valid",
        "post__left_occlusal_valid",
        "post__right_occlusal_valid",
    ]
    require_columns(
        df,
        cols,
        "posterior whole validity",
    )

    mask = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    for c in cols:
        mask &= bool_series(df[c])

    return mask


def anterior_valid_mask(
    df: pd.DataFrame,
) -> pd.Series:
    candidates = (
        "ant__anterior_radial_valid",
        "ant__anterior_geometry_valid",
    )

    for c in candidates:
        if c in df.columns:
            return bool_series(df[c])

    raise ValueError(
        "Could not find anterior radial validity column."
    )


def build_posterior_whole_features(
    df: pd.DataFrame,
) -> List[str]:
    cols = []

    for side in SIDES:
        for variant in ("full", "occlusal"):
            for suffix in POST_WHOLE_SUFFIXES:
                cols.append(
                    f"post__{side}_{variant}_{suffix}"
                )

    cols = safe_numeric(df, cols)

    if not cols:
        raise ValueError(
            "No posterior whole features found."
        )

    return cols


def tooth_group(teeth) -> str:
    """
    Anatomical broad group using the LAST FDI digit.

    1/2/3 = anterior (incisors/canine)
    4/5/6/7/8 = posterior

    This also naturally treats primary canine 53/63 as anterior.
    """
    if (
        not isinstance(teeth, list)
        or len(teeth) == 0
    ):
        return "teeth_empty"

    has_ant = False
    has_post = False
    usable = 0

    for t in teeth:
        try:
            n = int(str(t).strip())
        except Exception:
            continue

        digit = abs(n) % 10

        if digit in (1, 2, 3):
            has_ant = True
            usable += 1
        elif digit in (4, 5, 6, 7, 8):
            has_post = True
            usable += 1

    if usable == 0:
        return "other"

    if has_ant and has_post:
        return "mixed_anterior_posterior"
    if has_ant:
        return "anterior_only"
    if has_post:
        return "posterior_only"

    return "other"


def derive_branch_labels(
    global_y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns float arrays with {0,1,NaN}.

    Positive teeth-empty/other:
        branch label unknown -> NaN
    """
    ant = np.full(
        len(global_y),
        np.nan,
        dtype=float,
    )
    post = np.full(
        len(global_y),
        np.nan,
        dtype=float,
    )

    for i, (y, g) in enumerate(
        zip(global_y, groups)
    ):
        if int(y) == 0:
            ant[i] = 0.0
            post[i] = 0.0
            continue

        if g == "anterior_only":
            ant[i] = 1.0
            post[i] = 0.0

        elif g == "posterior_only":
            ant[i] = 0.0
            post[i] = 1.0

        elif g == "mixed_anterior_posterior":
            ant[i] = 1.0
            post[i] = 1.0

        # teeth_empty / other stay NaN

    return ant, post


def _row_smallest(
    arr: np.ndarray,
) -> np.ndarray:
    out = np.full(
        arr.shape[0],
        np.nan,
        dtype=float,
    )

    for i in range(arr.shape[0]):
        vals = arr[i][np.isfinite(arr[i])]
        if len(vals):
            out[i] = float(np.min(vals))

    return out


def _row_second_smallest(
    arr: np.ndarray,
) -> np.ndarray:
    out = np.full(
        arr.shape[0],
        np.nan,
        dtype=float,
    )

    for i in range(arr.shape[0]):
        vals = arr[i][np.isfinite(arr[i])]

        if len(vals) >= 2:
            out[i] = float(
                np.partition(vals, 1)[1]
            )
        elif len(vals) == 1:
            out[i] = float(vals[0])

    return out


def _row_two_smallest_mean(
    arr: np.ndarray,
) -> np.ndarray:
    out = np.full(
        arr.shape[0],
        np.nan,
        dtype=float,
    )

    for i in range(arr.shape[0]):
        vals = arr[i][np.isfinite(arr[i])]

        if len(vals) >= 2:
            two = np.partition(vals, 1)[:2]
            out[i] = float(np.mean(two))
        elif len(vals) == 1:
            out[i] = float(vals[0])

    return out


def _row_nanmax(
    arr: np.ndarray,
) -> np.ndarray:
    out = np.full(
        arr.shape[0],
        np.nan,
        dtype=float,
    )

    for i in range(arr.shape[0]):
        vals = arr[i][np.isfinite(arr[i])]
        if len(vals):
            out[i] = float(np.max(vals))

    return out


def derive_scope_features(
    out: pd.DataFrame,
    q20: np.ndarray,
    median: np.ndarray,
    neg: np.ndarray,
    teeth: Tuple[int, ...],
    all_teeth_index: Dict[int, int],
    prefix: str,
) -> List[str]:
    idx = [
        all_teeth_index[t]
        for t in teeth
    ]

    q = q20[:, idx]
    m = median[:, idx]
    n = neg[:, idx]

    names = []

    def add(name: str, values):
        c = f"{prefix}{name}"
        out[c] = values
        names.append(c)

    valid = (
        np.isfinite(q)
        & np.isfinite(m)
        & np.isfinite(n)
    )

    add(
        "valid_teeth",
        np.sum(valid, axis=1).astype(float),
    )
    add(
        "worst_q20",
        _row_smallest(q),
    )
    add(
        "second_worst_q20",
        _row_second_smallest(q),
    )
    add(
        "worst_two_q20_mean",
        _row_two_smallest_mean(q),
    )
    add(
        "worst_median",
        _row_smallest(m),
    )
    add(
        "max_negative_ratio",
        _row_nanmax(n),
    )
    add(
        "q20_below0_count",
        np.sum(
            np.isfinite(q)
            & (q < 0.0),
            axis=1,
        ).astype(float),
    )
    add(
        "q20_below05_count",
        np.sum(
            np.isfinite(q)
            & (q < -0.5),
            axis=1,
        ).astype(float),
    )
    add(
        "median_below0_count",
        np.sum(
            np.isfinite(m)
            & (m < 0.0),
            axis=1,
        ).astype(float),
    )

    return names


def build_anterior_raw_features(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    List[str],
    pd.DataFrame,
]:
    """
    45 raw geometry features + separate reliability/QC table.

    Geometry values are NOT masked.
    """
    required = []

    for tooth in ANTERIOR_TEETH:
        base = (
            f"ant__anterior_radial_"
            f"tooth{tooth}_incisal"
        )
        required += [
            f"{base}_q20_margin_mm",
            f"{base}_median_margin_mm",
            f"{base}_negative_ratio",
            f"{base}_implausible_margin_flag",
        ]

    require_columns(
        df,
        required,
        "anterior radial incisal",
    )

    n_rows = len(df)
    n_teeth = len(ANTERIOR_TEETH)

    q20 = np.full(
        (n_rows, n_teeth),
        np.nan,
        dtype=float,
    )
    med = np.full_like(q20, np.nan)
    neg = np.full_like(q20, np.nan)
    bad = np.zeros_like(
        q20,
        dtype=bool,
    )

    for j, tooth in enumerate(
        ANTERIOR_TEETH
    ):
        base = (
            f"ant__anterior_radial_"
            f"tooth{tooth}_incisal"
        )

        q20[:, j] = pd.to_numeric(
            df[f"{base}_q20_margin_mm"],
            errors="coerce",
        ).to_numpy(float)

        med[:, j] = pd.to_numeric(
            df[f"{base}_median_margin_mm"],
            errors="coerce",
        ).to_numpy(float)

        neg[:, j] = pd.to_numeric(
            df[f"{base}_negative_ratio"],
            errors="coerce",
        ).to_numpy(float)

        bad[:, j] = bool_series(
            df[
                f"{base}_implausible_margin_flag"
            ]
        ).to_numpy(bool)

    out = pd.DataFrame(
        index=df.index
    )
    cols = []

    # 18 tooth-level features.
    for j, tooth in enumerate(
        ANTERIOR_TEETH
    ):
        for suffix, arr in (
            ("q20", q20),
            ("median", med),
            ("negative_ratio", neg),
        ):
            c = (
                f"ant_raw__tooth"
                f"{tooth}_{suffix}"
            )
            out[c] = arr[:, j]
            cols.append(c)

    idx_map = {
        t: i
        for i, t in enumerate(
            ANTERIOR_TEETH
        )
    }

    # 27 aggregate features.
    cols += derive_scope_features(
        out,
        q20,
        med,
        neg,
        ANTERIOR_TEETH,
        idx_map,
        "ant_raw__all_",
    )
    cols += derive_scope_features(
        out,
        q20,
        med,
        neg,
        RIGHT_ANTERIOR,
        idx_map,
        "ant_raw__right_",
    )
    cols += derive_scope_features(
        out,
        q20,
        med,
        neg,
        LEFT_ANTERIOR,
        idx_map,
        "ant_raw__left_",
    )

    cols = safe_numeric(
        out,
        cols,
    )

    qc = pd.DataFrame(
        {
            "anterior_bad_tooth_count": (
                np.sum(
                    bad,
                    axis=1,
                ).astype(float)
            ),
            "anterior_any_bad_tooth": (
                np.any(
                    bad,
                    axis=1,
                ).astype(float)
            ),
        },
        index=df.index,
    )

    return out, cols, qc


def make_pipeline(
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
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
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_meta_pipeline(
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
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
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def binary_metrics(
    y_true,
    y_pred,
    prob=None,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    out = {
        "n": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "positive_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "positive_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": (
            float(tn / (tn + fp))
            if (tn + fp)
            else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if (
        prob is not None
        and len(np.unique(y_true)) == 2
    ):
        prob = np.asarray(
            prob,
            dtype=float,
        )
        out["roc_auc"] = float(
            roc_auc_score(
                y_true,
                prob,
            )
        )
        out["pr_auc"] = float(
            average_precision_score(
                y_true,
                prob,
            )
        )
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def tune_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    objective: str,
) -> Tuple[float, dict]:
    best = None

    for t in np.linspace(
        0.05,
        0.95,
        181,
    ):
        pred = (
            prob >= t
        ).astype(int)

        m = binary_metrics(
            y_true,
            pred,
            prob,
        )

        key = (
            float(m[objective]),
            float(
                m["balanced_accuracy"]
            ),
            float(
                m["positive_recall"]
            ),
            -abs(float(t) - 0.5),
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(t),
                m,
            )

    return best[1], best[2]


def tune_dual_threshold_or(
    y_true: np.ndarray,
    posterior_prob: np.ndarray,
    anterior_prob: np.ndarray,
    objective: str,
    grid_min: float,
    grid_max: float,
    grid_step: float,
) -> Tuple[float, float, dict]:
    """
    Jointly tune TWO independent thresholds on inner-OOF probabilities:

        global_crossbite =
            (Pposterior >= Tposterior)
            OR
            (Panterior >= Tanterior)

    This is intentionally different from branch_max:
        max(Pposterior, Panterior) >= one shared threshold

    because the two branch probabilities are not on the same calibrated scale.

    The search is performed ONLY on the outer-training fold's inner-OOF branch
    probabilities. The selected pair is then applied unchanged to the untouched
    outer-validation fold.
    """
    if grid_step <= 0:
        raise ValueError("--dual-grid-step must be > 0")
    if not (0.0 <= grid_min < grid_max <= 1.0):
        raise ValueError(
            "Require 0 <= --dual-grid-min < --dual-grid-max <= 1."
        )

    # Include the upper endpoint robustly.
    grid = np.arange(
        grid_min,
        grid_max + grid_step * 0.5,
        grid_step,
        dtype=float,
    )
    grid = np.clip(grid, 0.0, 1.0)
    grid = np.unique(np.round(grid, 10))

    best = None

    for t_post in grid:
        post_positive = posterior_prob >= t_post

        for t_ant in grid:
            pred = (
                post_positive
                | (anterior_prob >= t_ant)
            ).astype(int)

            m = binary_metrics(
                y_true,
                pred,
                prob=None,
            )

            # Primary objective is Macro-F1 by default.
            # Tie-breaking favors balanced performance, then recall, then
            # specificity, then a less-extreme threshold pair.
            key = (
                float(m[objective]),
                float(m["balanced_accuracy"]),
                float(m["positive_recall"]),
                float(m["specificity"]),
                -(
                    abs(float(t_post) - 0.5)
                    + abs(float(t_ant) - 0.5)
                ),
            )

            if best is None or key > best[0]:
                best = (
                    key,
                    float(t_post),
                    float(t_ant),
                    m,
                )

    return best[1], best[2], best[3]


def assign_master_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    counts = np.bincount(
        y,
        minlength=2,
    )

    if counts.min() < n_splits:
        raise ValueError(
            f"Not enough cases for {n_splits}-fold CV: "
            f"{counts.tolist()}"
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dummy = np.zeros(
        (len(y), 1),
        dtype=float,
    )

    fold_ids = np.full(
        len(y),
        -1,
        dtype=int,
    )

    for fold, (_, va) in enumerate(
        skf.split(
            dummy,
            y,
        )
    ):
        fold_ids[va] = fold

    return fold_ids


def make_inner_splits(
    y_global: np.ndarray,
    subgroup: np.ndarray,
    n_splits: int,
    seed: int,
):
    """
    Prefer composite stratification so anterior/posterior/mixed/empty positives
    are distributed more evenly across inner folds.

    Fallback to global y when a stratum is too small.
    """
    strata = np.asarray(
        [
            (
                "negative"
                if int(y) == 0
                else str(g)
            )
            for y, g in zip(
                y_global,
                subgroup,
            )
        ],
        dtype=object,
    )

    counts = pd.Series(
        strata
    ).value_counts()

    use_composite = bool(
        len(counts)
        and counts.min() >= n_splits
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dummy = np.zeros(
        (len(y_global), 1),
        dtype=float,
    )

    target = (
        strata
        if use_composite
        else y_global
    )

    return list(
        splitter.split(
            dummy,
            target,
        )
    )


def evaluate_global_logistic_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    subgroup: np.ndarray,
    fold_ids: np.ndarray,
    eligible: np.ndarray,
    model_name: str,
    args,
):
    """
    Reference global-label model with strict nested threshold tuning.
    """
    prob = np.full(
        len(y),
        np.nan,
        dtype=float,
    )
    pred = np.full(
        len(y),
        np.nan,
        dtype=float,
    )
    threshold_used = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    for fold in sorted(
        np.unique(fold_ids)
    ):
        train_idx = np.where(
            eligible
            & (fold_ids != fold)
        )[0]
        valid_idx = np.where(
            eligible
            & (fold_ids == fold)
        )[0]

        y_train = y[train_idx]

        min_class = int(
            np.bincount(
                y_train,
                minlength=2,
            ).min()
        )
        inner_n = min(
            args.inner_folds,
            min_class,
        )

        inner_splits = make_inner_splits(
            y_train,
            subgroup[train_idx],
            inner_n,
            args.seed + 1000 + int(fold),
        )

        model = make_pipeline(
            args.seed + int(fold)
        )

        train_prob = cross_val_predict(
            model,
            X.iloc[train_idx],
            y_train,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=args.n_jobs,
        )[:, 1]

        threshold, train_m = (
            tune_threshold(
                y_train,
                train_prob,
                args.threshold_objective,
            )
        )

        fitted = clone(model)
        fitted.fit(
            X.iloc[train_idx],
            y_train,
        )

        vp = fitted.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]
        vd = (
            vp >= threshold
        ).astype(int)

        prob[valid_idx] = vp
        pred[valid_idx] = vd
        threshold_used[valid_idx] = threshold

        m = binary_metrics(
            y[valid_idx],
            vd,
            vp,
        )

        fold_rows.append(
            {
                "model": model_name,
                "fold": int(fold),
                "threshold": float(threshold),
                "threshold_train_objective": float(
                    train_m[
                        args.threshold_objective
                    ]
                ),
                "n_features": int(X.shape[1]),
                **m,
            }
        )

    idx = np.where(eligible)[0]

    overall = {
        "model": model_name,
        "n_features": int(X.shape[1]),
        **binary_metrics(
            y[idx],
            pred[idx].astype(int),
            prob[idx],
        ),
    }

    out = pd.DataFrame(
        {
            f"{model_name}_prob": prob,
            f"{model_name}_pred": pred,
            f"{model_name}_threshold": threshold_used,
        },
        index=X.index,
    )

    return out, fold_rows, overall


def generate_inner_branch_probs(
    X: pd.DataFrame,
    branch_label: np.ndarray,
    train_idx: np.ndarray,
    y_global: np.ndarray,
    subgroup: np.ndarray,
    inner_n: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    """
    Generates OOF branch probabilities for EVERY outer-training patient.

    Branch model fitting in each inner fold uses only rows with KNOWN branch
    labels. The inner-validation fold receives predictions regardless of whether
    its branch label is known.
    """
    local_y_global = y_global[train_idx]
    local_subgroup = subgroup[train_idx]

    splits = make_inner_splits(
        local_y_global,
        local_subgroup,
        inner_n,
        seed,
    )

    local_prob = np.full(
        len(train_idx),
        np.nan,
        dtype=float,
    )

    for inner_fold, (
        local_tr,
        local_va,
    ) in enumerate(splits):
        global_tr = train_idx[local_tr]
        global_va = train_idx[local_va]

        known = np.isfinite(
            branch_label[global_tr]
        )
        fit_idx = global_tr[known]

        y_fit = (
            branch_label[fit_idx]
            .astype(int)
        )

        if len(np.unique(y_fit)) != 2:
            raise RuntimeError(
                "Inner branch training fold lacks one class. "
                f"seed={seed}, inner_fold={inner_fold}"
            )

        model = make_pipeline(
            seed + inner_fold
        )

        model.fit(
            X.iloc[fit_idx],
            y_fit,
        )

        local_prob[local_va] = (
            model.predict_proba(
                X.iloc[global_va]
            )[:, 1]
        )

    if not np.all(
        np.isfinite(local_prob)
    ):
        raise RuntimeError(
            "Missing inner OOF branch probabilities."
        )

    return local_prob


def fit_branch_outer(
    X: pd.DataFrame,
    branch_label: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
):
    known = np.isfinite(
        branch_label[train_idx]
    )
    fit_idx = train_idx[known]
    y_fit = (
        branch_label[fit_idx]
        .astype(int)
    )

    if len(np.unique(y_fit)) != 2:
        raise RuntimeError(
            "Outer branch training set lacks one class."
        )

    model = make_pipeline(seed)
    model.fit(
        X.iloc[fit_idx],
        y_fit,
    )

    valid_prob = model.predict_proba(
        X.iloc[valid_idx]
    )[:, 1]

    return valid_prob, len(fit_idx)


def fusion_score(
    p_post: np.ndarray,
    p_ant: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "max":
        return np.maximum(
            p_post,
            p_ant,
        )

    if mode == "noisy_or":
        return (
            1.0
            - (1.0 - p_post)
            * (1.0 - p_ant)
        )

    raise ValueError(
        f"Unknown fusion mode: {mode}"
    )


def tune_meta_threshold_with_cv(
    X_meta: pd.DataFrame,
    y: np.ndarray,
    subgroup: np.ndarray,
    args,
    seed: int,
):
    """
    Meta-level CV is separate from branch OOF generation.

    The inputs to the meta model are already branch-OOF probabilities, but we
    still use meta-level CV to tune the final classification threshold rather
    than tuning on in-sample fitted meta probabilities.
    """
    min_class = int(
        np.bincount(
            y,
            minlength=2,
        ).min()
    )
    inner_n = min(
        args.inner_folds,
        min_class,
    )

    splits = make_inner_splits(
        y,
        subgroup,
        inner_n,
        seed,
    )

    model = make_meta_pipeline(seed)

    meta_oof_prob = cross_val_predict(
        model,
        X_meta,
        y,
        cv=splits,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]

    threshold, train_m = tune_threshold(
        y,
        meta_oof_prob,
        args.threshold_objective,
    )

    fitted = clone(model)
    fitted.fit(
        X_meta,
        y,
    )

    return (
        fitted,
        threshold,
        train_m,
    )


def evaluate_branch_fusions_oof(
    X_post: pd.DataFrame,
    X_ant_branch: pd.DataFrame,
    qc: pd.DataFrame,
    y_global: np.ndarray,
    ant_label: np.ndarray,
    post_label: np.ndarray,
    subgroup: np.ndarray,
    fold_ids: np.ndarray,
    eligible: np.ndarray,
    args,
):
    n = len(y_global)

    post_prob = np.full(
        n,
        np.nan,
        dtype=float,
    )
    ant_prob = np.full(
        n,
        np.nan,
        dtype=float,
    )

    post_branch_pred = np.full(
        n,
        np.nan,
        dtype=float,
    )
    ant_branch_pred = np.full(
        n,
        np.nan,
        dtype=float,
    )

    post_branch_threshold = np.full(
        n,
        np.nan,
        dtype=float,
    )
    ant_branch_threshold = np.full(
        n,
        np.nan,
        dtype=float,
    )

    model_names = (
        "branch_dual_threshold_or",
        "branch_max",
        "branch_noisy_or",
        "branch_meta_logistic",
        "branch_meta_logistic_qc",
    )

    model_prob = {
        m: np.full(
            n,
            np.nan,
            dtype=float,
        )
        for m in model_names
    }
    model_pred = {
        m: np.full(
            n,
            np.nan,
            dtype=float,
        )
        for m in model_names
    }
    model_threshold = {
        m: np.full(
            n,
            np.nan,
            dtype=float,
        )
        for m in model_names
    }

    fold_rows = []
    threshold_rows = []

    for fold in sorted(
        np.unique(fold_ids)
    ):
        train_idx = np.where(
            eligible
            & (fold_ids != fold)
        )[0]
        valid_idx = np.where(
            eligible
            & (fold_ids == fold)
        )[0]

        y_train_global = y_global[train_idx]

        min_class = int(
            np.bincount(
                y_train_global,
                minlength=2,
            ).min()
        )
        inner_n = min(
            args.inner_folds,
            min_class,
        )

        # -------------------------------------------------------------
        # 1) Branch OOF probabilities on OUTER TRAIN
        # -------------------------------------------------------------
        inner_post_prob = (
            generate_inner_branch_probs(
                X=X_post,
                branch_label=post_label,
                train_idx=train_idx,
                y_global=y_global,
                subgroup=subgroup,
                inner_n=inner_n,
                seed=args.seed + 3000 + int(fold),
                n_jobs=args.n_jobs,
            )
        )

        inner_ant_prob = (
            generate_inner_branch_probs(
                X=X_ant_branch,
                branch_label=ant_label,
                train_idx=train_idx,
                y_global=y_global,
                subgroup=subgroup,
                inner_n=inner_n,
                seed=args.seed + 4000 + int(fold),
                n_jobs=args.n_jobs,
            )
        )

        # -------------------------------------------------------------
        # 2) Branch thresholds for branch-involvement diagnostics
        # -------------------------------------------------------------
        local_post_label = post_label[train_idx]
        known_post = np.isfinite(
            local_post_label
        )

        post_t, post_tm = tune_threshold(
            local_post_label[
                known_post
            ].astype(int),
            inner_post_prob[
                known_post
            ],
            args.threshold_objective,
        )

        local_ant_label = ant_label[train_idx]
        known_ant = np.isfinite(
            local_ant_label
        )

        ant_t, ant_tm = tune_threshold(
            local_ant_label[
                known_ant
            ].astype(int),
            inner_ant_prob[
                known_ant
            ],
            args.threshold_objective,
        )

        # -------------------------------------------------------------
        # 3) Fit branches on ALL known OUTER TRAIN branch labels
        # -------------------------------------------------------------
        valid_post_prob, n_post_fit = (
            fit_branch_outer(
                X_post,
                post_label,
                train_idx,
                valid_idx,
                args.seed + 5000 + int(fold),
            )
        )

        valid_ant_prob, n_ant_fit = (
            fit_branch_outer(
                X_ant_branch,
                ant_label,
                train_idx,
                valid_idx,
                args.seed + 6000 + int(fold),
            )
        )

        post_prob[valid_idx] = valid_post_prob
        ant_prob[valid_idx] = valid_ant_prob

        # Branch diagnostic predictions only where branch GT is known.
        known_valid_post = np.isfinite(
            post_label[valid_idx]
        )
        known_valid_ant = np.isfinite(
            ant_label[valid_idx]
        )

        post_branch_threshold[valid_idx] = post_t
        ant_branch_threshold[valid_idx] = ant_t

        post_branch_pred[
            valid_idx[known_valid_post]
        ] = (
            valid_post_prob[
                known_valid_post
            ]
            >= post_t
        ).astype(int)

        ant_branch_pred[
            valid_idx[known_valid_ant]
        ] = (
            valid_ant_prob[
                known_valid_ant
            ]
            >= ant_t
        ).astype(int)

        # -------------------------------------------------------------
        # 4) Dual-threshold OR (main v3 experiment)
        # -------------------------------------------------------------
        dual_post_t, dual_ant_t, dual_tm = tune_dual_threshold_or(
            y_true=y_train_global,
            posterior_prob=inner_post_prob,
            anterior_prob=inner_ant_prob,
            objective=args.threshold_objective,
            grid_min=args.dual_grid_min,
            grid_max=args.dual_grid_max,
            grid_step=args.dual_grid_step,
        )

        dual_valid_pred = (
            (valid_post_prob >= dual_post_t)
            | (valid_ant_prob >= dual_ant_t)
        ).astype(int)

        # There is no single natural probability for a two-threshold OR rule.
        # We keep max(raw branch probabilities) only as a diagnostic score in
        # the OOF file, but do NOT use it to report ROC-AUC / PR-AUC.
        dual_valid_score = np.maximum(
            valid_post_prob,
            valid_ant_prob,
        )

        model_prob[
            "branch_dual_threshold_or"
        ][valid_idx] = dual_valid_score
        model_pred[
            "branch_dual_threshold_or"
        ][valid_idx] = dual_valid_pred

        # model_threshold is intentionally NaN for this model because it has
        # TWO thresholds. They are stored explicitly in threshold_rows below.
        model_threshold[
            "branch_dual_threshold_or"
        ][valid_idx] = np.nan

        dual_m = binary_metrics(
            y_global[valid_idx],
            dual_valid_pred,
            prob=None,
        )

        fold_rows.append(
            {
                "model": "branch_dual_threshold_or",
                "fold": int(fold),
                "threshold": np.nan,
                "posterior_threshold": float(dual_post_t),
                "anterior_threshold": float(dual_ant_t),
                "threshold_train_objective": float(
                    dual_tm[
                        args.threshold_objective
                    ]
                ),
                "n_features": 2,
                "posterior_branch_train_n": int(
                    n_post_fit
                ),
                "anterior_branch_train_n": int(
                    n_ant_fit
                ),
                **dual_m,
            }
        )

        threshold_rows.append(
            {
                "model": "branch_dual_threshold_or",
                "fold": int(fold),
                "threshold": np.nan,
                "dual_posterior_threshold": float(
                    dual_post_t
                ),
                "dual_anterior_threshold": float(
                    dual_ant_t
                ),
                "posterior_branch_threshold": float(
                    post_t
                ),
                "anterior_branch_threshold": float(
                    ant_t
                ),
                "dual_grid_min": float(
                    args.dual_grid_min
                ),
                "dual_grid_max": float(
                    args.dual_grid_max
                ),
                "dual_grid_step": float(
                    args.dual_grid_step
                ),
            }
        )

        # -------------------------------------------------------------
        # 5) Simple shared-threshold branch OR fusions (references)
        # -------------------------------------------------------------
        for model_name, mode in (
            ("branch_max", "max"),
            (
                "branch_noisy_or",
                "noisy_or",
            ),
        ):
            train_score = fusion_score(
                inner_post_prob,
                inner_ant_prob,
                mode,
            )

            t, tm = tune_threshold(
                y_train_global,
                train_score,
                args.threshold_objective,
            )

            valid_score = fusion_score(
                valid_post_prob,
                valid_ant_prob,
                mode,
            )
            valid_pred = (
                valid_score >= t
            ).astype(int)

            model_prob[
                model_name
            ][valid_idx] = valid_score
            model_pred[
                model_name
            ][valid_idx] = valid_pred
            model_threshold[
                model_name
            ][valid_idx] = t

            m = binary_metrics(
                y_global[valid_idx],
                valid_pred,
                valid_score,
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "threshold": float(t),
                    "threshold_train_objective": float(
                        tm[
                            args.threshold_objective
                        ]
                    ),
                    "n_features": 2,
                    "posterior_branch_train_n": int(
                        n_post_fit
                    ),
                    "anterior_branch_train_n": int(
                        n_ant_fit
                    ),
                    **m,
                }
            )

            threshold_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "threshold": float(t),
                    "posterior_branch_threshold": float(
                        post_t
                    ),
                    "anterior_branch_threshold": float(
                        ant_t
                    ),
                }
            )

        # -------------------------------------------------------------
        # 6) Learned 2-D meta fusion
        # -------------------------------------------------------------
        meta_train_2 = pd.DataFrame(
            {
                "posterior_prob": inner_post_prob,
                "anterior_prob": inner_ant_prob,
            }
        )

        meta_valid_2 = pd.DataFrame(
            {
                "posterior_prob": valid_post_prob,
                "anterior_prob": valid_ant_prob,
            }
        )

        (
            fitted_meta_2,
            meta_t_2,
            meta_tm_2,
        ) = tune_meta_threshold_with_cv(
            meta_train_2,
            y_train_global,
            subgroup[train_idx],
            args,
            args.seed + 7000 + int(fold),
        )

        valid_meta_prob_2 = (
            fitted_meta_2.predict_proba(
                meta_valid_2
            )[:, 1]
        )
        valid_meta_pred_2 = (
            valid_meta_prob_2
            >= meta_t_2
        ).astype(int)

        model_prob[
            "branch_meta_logistic"
        ][valid_idx] = valid_meta_prob_2
        model_pred[
            "branch_meta_logistic"
        ][valid_idx] = valid_meta_pred_2
        model_threshold[
            "branch_meta_logistic"
        ][valid_idx] = meta_t_2

        m2 = binary_metrics(
            y_global[valid_idx],
            valid_meta_pred_2,
            valid_meta_prob_2,
        )

        fold_rows.append(
            {
                "model": "branch_meta_logistic",
                "fold": int(fold),
                "threshold": float(meta_t_2),
                "threshold_train_objective": float(
                    meta_tm_2[
                        args.threshold_objective
                    ]
                ),
                "n_features": 2,
                "posterior_branch_train_n": int(
                    n_post_fit
                ),
                "anterior_branch_train_n": int(
                    n_ant_fit
                ),
                **m2,
            }
        )

        threshold_rows.append(
            {
                "model": "branch_meta_logistic",
                "fold": int(fold),
                "threshold": float(meta_t_2),
                "posterior_branch_threshold": float(
                    post_t
                ),
                "anterior_branch_threshold": float(
                    ant_t
                ),
            }
        )

        # -------------------------------------------------------------
        # 7) Learned meta fusion + anterior reliability indicators
        # -------------------------------------------------------------
        qc_train = qc.iloc[
            train_idx
        ].reset_index(drop=True)
        qc_valid = qc.iloc[
            valid_idx
        ].reset_index(drop=True)

        meta_train_qc = pd.DataFrame(
            {
                "posterior_prob": inner_post_prob,
                "anterior_prob": inner_ant_prob,
                "anterior_bad_tooth_count": (
                    qc_train[
                        "anterior_bad_tooth_count"
                    ].to_numpy(float)
                ),
                "anterior_any_bad_tooth": (
                    qc_train[
                        "anterior_any_bad_tooth"
                    ].to_numpy(float)
                ),
            }
        )

        meta_valid_qc = pd.DataFrame(
            {
                "posterior_prob": valid_post_prob,
                "anterior_prob": valid_ant_prob,
                "anterior_bad_tooth_count": (
                    qc_valid[
                        "anterior_bad_tooth_count"
                    ].to_numpy(float)
                ),
                "anterior_any_bad_tooth": (
                    qc_valid[
                        "anterior_any_bad_tooth"
                    ].to_numpy(float)
                ),
            }
        )

        (
            fitted_meta_qc,
            meta_t_qc,
            meta_tm_qc,
        ) = tune_meta_threshold_with_cv(
            meta_train_qc,
            y_train_global,
            subgroup[train_idx],
            args,
            args.seed + 8000 + int(fold),
        )

        valid_meta_prob_qc = (
            fitted_meta_qc.predict_proba(
                meta_valid_qc
            )[:, 1]
        )
        valid_meta_pred_qc = (
            valid_meta_prob_qc
            >= meta_t_qc
        ).astype(int)

        model_prob[
            "branch_meta_logistic_qc"
        ][valid_idx] = valid_meta_prob_qc
        model_pred[
            "branch_meta_logistic_qc"
        ][valid_idx] = valid_meta_pred_qc
        model_threshold[
            "branch_meta_logistic_qc"
        ][valid_idx] = meta_t_qc

        mq = binary_metrics(
            y_global[valid_idx],
            valid_meta_pred_qc,
            valid_meta_prob_qc,
        )

        fold_rows.append(
            {
                "model": "branch_meta_logistic_qc",
                "fold": int(fold),
                "threshold": float(meta_t_qc),
                "threshold_train_objective": float(
                    meta_tm_qc[
                        args.threshold_objective
                    ]
                ),
                "n_features": 4,
                "posterior_branch_train_n": int(
                    n_post_fit
                ),
                "anterior_branch_train_n": int(
                    n_ant_fit
                ),
                **mq,
            }
        )

        threshold_rows.append(
            {
                "model": "branch_meta_logistic_qc",
                "fold": int(fold),
                "threshold": float(meta_t_qc),
                "posterior_branch_threshold": float(
                    post_t
                ),
                "anterior_branch_threshold": float(
                    ant_t
                ),
            }
        )

    # -------------------------------------------------------------
    # Overall global model metrics
    # -------------------------------------------------------------
    idx = np.where(eligible)[0]
    model_rows = []

    for model_name in model_names:
        pp = model_prob[
            model_name
        ][idx]
        dd = model_pred[
            model_name
        ][idx]

        if not np.all(
            np.isfinite(pp)
        ):
            raise RuntimeError(
                f"{model_name}: missing global OOF probabilities"
            )

        model_rows.append(
            {
                "model": model_name,
                "n_features": (
                    4
                    if model_name
                    == "branch_meta_logistic_qc"
                    else 2
                ),
                **binary_metrics(
                    y_global[idx],
                    dd.astype(int),
                    (
                        None
                        if model_name
                        == "branch_dual_threshold_or"
                        else pp
                    ),
                ),
            }
        )

    # -------------------------------------------------------------
    # Branch involvement diagnostic metrics
    # -------------------------------------------------------------
    branch_metric_rows = []

    for branch_name, (
        label,
        prob_arr,
        pred_arr,
    ) in {
        "posterior_involvement": (
            post_label,
            post_prob,
            post_branch_pred,
        ),
        "anterior_involvement": (
            ant_label,
            ant_prob,
            ant_branch_pred,
        ),
    }.items():
        known = (
            eligible
            & np.isfinite(label)
            & np.isfinite(pred_arr)
        )
        ii = np.where(known)[0]

        branch_metric_rows.append(
            {
                "branch": branch_name,
                "known_n": int(len(ii)),
                "positive_n": int(
                    np.sum(
                        label[ii] == 1
                    )
                ),
                "negative_n": int(
                    np.sum(
                        label[ii] == 0
                    )
                ),
                **binary_metrics(
                    label[ii].astype(int),
                    pred_arr[ii].astype(int),
                    prob_arr[ii],
                ),
            }
        )

    # OOF dataframe.
    oof = pd.DataFrame(
        {
            "posterior_involvement_prob": post_prob,
            "posterior_involvement_pred": post_branch_pred,
            "posterior_involvement_threshold": post_branch_threshold,
            "anterior_involvement_prob": ant_prob,
            "anterior_involvement_pred": ant_branch_pred,
            "anterior_involvement_threshold": ant_branch_threshold,
        }
    )

    for m in model_names:
        if m == "branch_dual_threshold_or":
            oof[f"{m}_diagnostic_max_prob"] = model_prob[m]
        else:
            oof[f"{m}_prob"] = model_prob[m]

        oof[f"{m}_pred"] = model_pred[m]
        oof[f"{m}_threshold"] = (
            model_threshold[m]
        )

    return (
        oof,
        pd.DataFrame(fold_rows),
        pd.DataFrame(model_rows),
        pd.DataFrame(branch_metric_rows),
        pd.DataFrame(threshold_rows),
    )


def subgroup_recall_table(
    base_df: pd.DataFrame,
    y: np.ndarray,
    eligible: np.ndarray,
    oof: pd.DataFrame,
    model_names: List[str],
) -> pd.DataFrame:
    groups = (
        base_df[
            "positive_subgroup"
        ]
        .astype(str)
        .to_numpy()
    )

    rows = []

    positive_idx = np.where(
        eligible
        & (y == 1)
    )[0]

    ordered_groups = (
        "anterior_only",
        "posterior_only",
        "mixed_anterior_posterior",
        "teeth_empty",
        "other",
        "ALL_POSITIVE",
    )

    for model in model_names:
        c = f"{model}_pred"

        pred = pd.to_numeric(
            oof[c],
            errors="coerce",
        ).to_numpy(float)

        for group in ordered_groups:
            if group == "ALL_POSITIVE":
                ii = positive_idx
            else:
                ii = positive_idx[
                    groups[
                        positive_idx
                    ]
                    == group
                ]

            finite = np.isfinite(
                pred[ii]
            )
            jj = ii[finite]

            if len(jj) == 0:
                tp = 0
                fn = 0
                recall = np.nan
            else:
                d = pred[jj].astype(int)
                tp = int(
                    np.sum(d == 1)
                )
                fn = int(
                    np.sum(d == 0)
                )
                recall = float(
                    tp / len(jj)
                )

            rows.append(
                {
                    "model": model,
                    "positive_subgroup": group,
                    "n_positive": int(
                        len(ii)
                    ),
                    "tp": tp,
                    "fn": fn,
                    "recall": recall,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    post_path = resolve_posterior_features(
        args.posterior_features
    )
    ant_path = (
        args.anterior_features
        .expanduser()
        .resolve()
    )
    gt_path = (
        args.gt
        .expanduser()
        .resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not ant_path.exists():
        raise FileNotFoundError(
            f"Anterior feature CSV not found: {ant_path}"
        )
    if not gt_path.exists():
        raise FileNotFoundError(
            f"GT JSON not found: {gt_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Crossbite branch-specific fusion v3 + dual-threshold OR")
    print("=" * 100)
    print(f"posterior : {post_path}")
    print(f"anterior  : {ant_path}")
    print(f"GT        : {gt_path}")
    print(f"output    : {output_dir}")
    print("=" * 100)

    post = pd.read_csv(
        post_path,
        low_memory=False,
    )
    ant = pd.read_csv(
        ant_path,
        low_memory=False,
    )

    require_columns(
        post,
        ["patient_id"],
        "posterior CSV",
    )
    require_columns(
        ant,
        ["patient_id"],
        "anterior CSV",
    )

    if post[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate patient_id in posterior CSV"
        )

    if ant[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate patient_id in anterior CSV"
        )

    post = prefix_dataframe(
        post,
        "post__",
    )
    ant = prefix_dataframe(
        ant,
        "ant__",
    )

    gt = load_gt(gt_path)

    if args.include_ios_fallback:
        source_mask = gt[
            "gt_source"
        ].isin(
            (
                "photo_en",
                "ios_en_fallback",
            )
        )
        source_mode = (
            "photo_en+ios_en_fallback"
        )
    else:
        source_mask = (
            gt["gt_source"]
            .eq("photo_en")
        )
        source_mode = "photo_en_only"

    gt = gt.loc[
        source_mask
        & gt[
            "gt_crossbite_present"
        ].notna()
    ].copy()

    merged = (
        gt.merge(
            post,
            on="patient_id",
            how="left",
            validate="one_to_one",
            indicator="post_merge",
        )
        .merge(
            ant,
            on="patient_id",
            how="left",
            validate="one_to_one",
            indicator="ant_merge",
        )
    )

    matched = (
        merged["post_merge"].eq("both")
        & merged["ant_merge"].eq("both")
    )

    merged = (
        merged.loc[matched]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    y = (
        merged[
            "gt_crossbite_present"
        ]
        .astype(int)
        .to_numpy()
    )

    merged[
        "positive_subgroup"
    ] = merged[
        "gt_crossbite_teeth_list"
    ].map(tooth_group)

    subgroup = (
        merged[
            "positive_subgroup"
        ]
        .astype(str)
        .to_numpy()
    )

    ant_label, post_label = (
        derive_branch_labels(
            y,
            subgroup,
        )
    )

    merged[
        "gt_anterior_involvement"
    ] = ant_label
    merged[
        "gt_posterior_involvement"
    ] = post_label

    fold_ids = assign_master_folds(
        y,
        args.outer_folds,
        args.seed,
    )
    merged["outer_fold"] = fold_ids

    post_valid = (
        posterior_valid_mask(
            merged
        )
        .to_numpy(bool)
    )
    ant_valid = (
        anterior_valid_mask(
            merged
        )
        .to_numpy(bool)
    )

    # Strict apples-to-apples final evaluation cohort.
    common = (
        post_valid
        & ant_valid
    )

    post_cols = (
        build_posterior_whole_features(
            merged
        )
    )
    X_post = (
        merged[
            post_cols
        ].copy()
    )

    (
        ant_raw_df,
        ant_raw_cols,
        ant_qc,
    ) = build_anterior_raw_features(
        merged
    )

    X_ant_global = (
        ant_raw_df[
            ant_raw_cols
        ].copy()
    )

    # Branch-specific anterior gets QC as reliability FEATURES, not masking.
    X_ant_branch = pd.concat(
        [
            X_ant_global.reset_index(
                drop=True
            ),
            ant_qc.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    X_ant_branch.index = merged.index

    X_concat = pd.concat(
        [
            X_post.reset_index(
                drop=True
            ),
            X_ant_global.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    X_concat.index = merged.index

    # -------------------------------------------------------------
    # Coverage + branch label counts
    # -------------------------------------------------------------
    coverage_rows = []

    def add_cov(name, mask):
        coverage_rows.append(
            {
                "cohort": name,
                "n": int(
                    np.sum(mask)
                ),
                "negative": int(
                    np.sum(
                        mask
                        & (y == 0)
                    )
                ),
                "positive": int(
                    np.sum(
                        mask
                        & (y == 1)
                    )
                ),
                "coverage_ratio": float(
                    np.mean(mask)
                ),
            }
        )

    add_cov(
        "posterior_valid",
        post_valid,
    )
    add_cov(
        "anterior_radial_valid",
        ant_valid,
    )
    add_cov(
        "strict_common",
        common,
    )

    branch_label_rows = []

    for name, label in (
        (
            "anterior_involvement",
            ant_label,
        ),
        (
            "posterior_involvement",
            post_label,
        ),
    ):
        known = (
            common
            & np.isfinite(label)
        )

        branch_label_rows.append(
            {
                "cohort": (
                    f"{name}_known_on_common"
                ),
                "n": int(
                    np.sum(known)
                ),
                "negative": int(
                    np.sum(
                        known
                        & (label == 0)
                    )
                ),
                "positive": int(
                    np.sum(
                        known
                        & (label == 1)
                    )
                ),
                "coverage_ratio": float(
                    np.mean(known)
                ),
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
        + branch_label_rows
    )

    print()
    print("COHORT")
    print("-" * 100)
    print(
        f"matched rows                     : {len(merged)}"
    )
    print(
        f"global negative                  : {int(np.sum(y == 0))}"
    )
    print(
        f"global positive                  : {int(np.sum(y == 1))}"
    )
    print(
        f"posterior valid                  : {int(post_valid.sum())}"
    )
    print(
        f"anterior radial valid            : {int(ant_valid.sum())}"
    )
    print(
        f"strict common                    : {int(common.sum())}"
    )
    print(
        "anterior branch known on common : "
        f"{int(np.sum(common & np.isfinite(ant_label)))}"
    )
    print(
        "  anterior positives             : "
        f"{int(np.sum(common & (ant_label == 1)))}"
    )
    print(
        "posterior branch known on common: "
        f"{int(np.sum(common & np.isfinite(post_label)))}"
    )
    print(
        "  posterior positives            : "
        f"{int(np.sum(common & (post_label == 1)))}"
    )

    subgroup_counts = (
        merged.loc[
            common & (y == 1),
            "positive_subgroup",
        ]
        .value_counts()
    )

    print()
    print("Positive subgroups on common:")
    print(
        subgroup_counts.to_string()
    )

    # -------------------------------------------------------------
    # Reference global-label models
    # -------------------------------------------------------------
    model_rows = []
    fold_rows = []

    oof_base = pd.DataFrame(
        {
            "patient_id": merged[
                "patient_id"
            ],
            "gt_crossbite_present": y,
            "gt_crossbite_location": merged[
                "gt_crossbite_location"
            ],
            "gt_crossbite_teeth": merged[
                "gt_crossbite_teeth"
            ],
            "positive_subgroup": subgroup,
            "gt_anterior_involvement": ant_label,
            "gt_posterior_involvement": post_label,
            "outer_fold": fold_ids,
            "posterior_valid": post_valid.astype(
                int
            ),
            "anterior_radial_valid": ant_valid.astype(
                int
            ),
            "common_eligible": common.astype(
                int
            ),
            "anterior_bad_tooth_count": ant_qc[
                "anterior_bad_tooth_count"
            ].to_numpy(float),
            "anterior_any_bad_tooth": ant_qc[
                "anterior_any_bad_tooth"
            ].to_numpy(float),
        }
    )

    # Baseline.
    ci = np.where(common)[0]
    baseline_pred = np.zeros(
        len(ci),
        dtype=int,
    )
    model_rows.append(
        {
            "model": "baseline_all_false",
            "n_features": 0,
            **binary_metrics(
                y[ci],
                baseline_pred,
            ),
        }
    )

    baseline_full = np.full(
        len(y),
        np.nan,
    )
    baseline_full[ci] = 0.0
    oof_base[
        "baseline_all_false_pred"
    ] = baseline_full

    reference_specs = (
        (
            "posterior_global",
            X_post,
        ),
        (
            "anterior_global_raw",
            X_ant_global,
        ),
        (
            "concat_raw_reference",
            X_concat,
        ),
    )

    for model_name, X in reference_specs:
        print(
            f"[REFERENCE OOF] {model_name}: "
            f"n={int(common.sum())}, "
            f"features={X.shape[1]}"
        )

        (
            pred_df,
            fr,
            overall,
        ) = evaluate_global_logistic_oof(
            X=X,
            y=y,
            subgroup=subgroup,
            fold_ids=fold_ids,
            eligible=common,
            model_name=model_name,
            args=args,
        )

        oof_base = pd.concat(
            [
                oof_base,
                pred_df.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        fold_rows.extend(fr)
        model_rows.append(overall)

    # -------------------------------------------------------------
    # NEW branch-specific fusion
    # -------------------------------------------------------------
    print()
    print("[BRANCH OOF] posterior involvement + anterior involvement")

    (
        branch_oof,
        branch_fold_df,
        branch_model_df,
        branch_metric_df,
        threshold_df,
    ) = evaluate_branch_fusions_oof(
        X_post=X_post,
        X_ant_branch=X_ant_branch,
        qc=ant_qc,
        y_global=y,
        ant_label=ant_label,
        post_label=post_label,
        subgroup=subgroup,
        fold_ids=fold_ids,
        eligible=common,
        args=args,
    )

    oof = pd.concat(
        [
            oof_base,
            branch_oof.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    fold_rows.extend(
        branch_fold_df.to_dict(
            orient="records"
        )
    )
    model_rows.extend(
        branch_model_df.to_dict(
            orient="records"
        )
    )

    model_df = pd.DataFrame(
        model_rows
    ).sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
        ],
        ascending=False,
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    # -------------------------------------------------------------
    # Positive subgroup recall
    # -------------------------------------------------------------
    model_names = [
        "baseline_all_false",
        "posterior_global",
        "anterior_global_raw",
        "concat_raw_reference",
        "branch_dual_threshold_or",
        "branch_max",
        "branch_noisy_or",
        "branch_meta_logistic",
        "branch_meta_logistic_qc",
    ]

    subgroup_df = (
        subgroup_recall_table(
            base_df=merged,
            y=y,
            eligible=common,
            oof=oof,
            model_names=model_names,
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    coverage_path = (
        output_dir
        / "crossbite_branch_label_coverage_v3.csv"
    )
    model_path = (
        output_dir
        / "crossbite_branch_model_comparison_v3.csv"
    )
    fold_path = (
        output_dir
        / "crossbite_branch_fold_metrics_v3.csv"
    )
    oof_path = (
        output_dir
        / "crossbite_branch_oof_predictions_v3.csv"
    )
    subgroup_path = (
        output_dir
        / "crossbite_branch_subgroup_recall_v3.csv"
    )
    branch_metric_path = (
        output_dir
        / "crossbite_branch_involvement_metrics_v3.csv"
    )
    threshold_path = (
        output_dir
        / "crossbite_branch_thresholds_v3.csv"
    )
    summary_path = (
        output_dir
        / "crossbite_branch_summary_v3.json"
    )

    coverage_df.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )
    model_df.to_csv(
        model_path,
        index=False,
        encoding="utf-8-sig",
    )
    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )
    oof.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )
    subgroup_df.to_csv(
        subgroup_path,
        index=False,
        encoding="utf-8-sig",
    )
    branch_metric_df.to_csv(
        branch_metric_path,
        index=False,
        encoding="utf-8-sig",
    )
    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Summary.
    subgroup_summary = {}

    for model in model_names:
        part = subgroup_df.loc[
            subgroup_df[
                "model"
            ].eq(model)
        ]

        subgroup_summary[model] = {
            str(
                r["positive_subgroup"]
            ): (
                None
                if pd.isna(
                    r["recall"]
                )
                else float(
                    r["recall"]
                )
            )
            for _, r in part.iterrows()
        }

    best = (
        model_df
        .iloc[0]
        .to_dict()
    )

    def clean_json_value(v):
        if isinstance(
            v,
            np.integer,
        ):
            return int(v)
        if isinstance(
            v,
            np.floating,
        ):
            if np.isnan(v):
                return None
            return float(v)
        if pd.isna(v):
            return None
        return v

    summary = {
        "version": (
            "crossbite-branch-fusion-v3-dual-threshold-strict-nested-oof"
        ),
        "posterior_features": str(
            post_path
        ),
        "anterior_features": str(
            ant_path
        ),
        "gt": str(gt_path),
        "gt_source_mode": source_mode,
        "outer_folds": int(
            args.outer_folds
        ),
        "inner_folds": int(
            args.inner_folds
        ),
        "seed": int(args.seed),
        "threshold_objective": (
            args.threshold_objective
        ),
        "dual_threshold_grid": {
            "minimum": float(args.dual_grid_min),
            "maximum": float(args.dual_grid_max),
            "step": float(args.dual_grid_step),
            "rule": (
                "(Pposterior >= Tposterior) OR "
                "(Panterior >= Tanterior)"
            ),
            "selection": (
                "joint search on outer-train inner-OOF probabilities only"
            ),
        },
        "branch_label_policy": {
            "global_negative": {
                "anterior": 0,
                "posterior": 0,
            },
            "anterior_only": {
                "anterior": 1,
                "posterior": 0,
            },
            "posterior_only": {
                "anterior": 0,
                "posterior": 1,
            },
            "mixed_anterior_posterior": {
                "anterior": 1,
                "posterior": 1,
            },
            "teeth_empty_positive": {
                "anterior": None,
                "posterior": None,
                "note": (
                    "excluded from branch training, "
                    "retained for global meta-fusion"
                ),
            },
        },
        "cohort": {
            "matched_rows": int(
                len(merged)
            ),
            "global_negative": int(
                np.sum(y == 0)
            ),
            "global_positive": int(
                np.sum(y == 1)
            ),
            "posterior_valid": int(
                post_valid.sum()
            ),
            "anterior_radial_valid": int(
                ant_valid.sum()
            ),
            "strict_common": int(
                common.sum()
            ),
            "strict_common_negative": int(
                np.sum(
                    common
                    & (y == 0)
                )
            ),
            "strict_common_positive": int(
                np.sum(
                    common
                    & (y == 1)
                )
            ),
            "anterior_branch_known": int(
                np.sum(
                    common
                    & np.isfinite(
                        ant_label
                    )
                )
            ),
            "anterior_branch_positive": int(
                np.sum(
                    common
                    & (ant_label == 1)
                )
            ),
            "posterior_branch_known": int(
                np.sum(
                    common
                    & np.isfinite(
                        post_label
                    )
                )
            ),
            "posterior_branch_positive": int(
                np.sum(
                    common
                    & (post_label == 1)
                )
            ),
        },
        "feature_counts": {
            "posterior_whole": int(
                X_post.shape[1]
            ),
            "anterior_global_raw": int(
                X_ant_global.shape[1]
            ),
            "anterior_branch_with_reliability": int(
                X_ant_branch.shape[1]
            ),
            "concat_raw_reference": int(
                X_concat.shape[1]
            ),
            "branch_dual_threshold_or": 2,
            "branch_meta_logistic": 2,
            "branch_meta_logistic_qc": 4,
        },
        "best_model": {
            str(k): clean_json_value(v)
            for k, v in best.items()
        },
        "branch_involvement_metrics": (
            branch_metric_df
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "positive_subgroup_recall": (
            subgroup_summary
        ),
        "outputs": {
            "coverage": str(
                coverage_path
            ),
            "model_comparison": str(
                model_path
            ),
            "fold_metrics": str(
                fold_path
            ),
            "oof_predictions": str(
                oof_path
            ),
            "subgroup_recall": str(
                subgroup_path
            ),
            "branch_involvement_metrics": str(
                branch_metric_path
            ),
            "thresholds": str(
                threshold_path
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

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print()
    print("=" * 100)
    print("GLOBAL CROSSBITE_PRESENT — STRICT COMMON OOF")
    print("=" * 100)

    show_cols = [
        "model",
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "positive_precision",
        "positive_recall",
        "specificity",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print(
        model_df[
            [
                c
                for c in show_cols
                if c in model_df.columns
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.4f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print("BRANCH INVOLVEMENT OOF")
    print("=" * 100)

    print(
        branch_metric_df.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.4f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print("POSITIVE SUBGROUP RECALL")
    print("=" * 100)

    pivot = subgroup_df.pivot(
        index="model",
        columns="positive_subgroup",
        values="recall",
    )

    desired = [
        c
        for c in (
            "anterior_only",
            "posterior_only",
            "mixed_anterior_posterior",
            "teeth_empty",
            "other",
            "ALL_POSITIVE",
        )
        if c in pivot.columns
    ]

    print(
        pivot[
            desired
        ].to_string(
            float_format=(
                lambda x: f"{x:.4f}"
            )
        )
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)
    print(f"coverage       : {coverage_path}")
    print(f"model compare  : {model_path}")
    print(f"fold metrics   : {fold_path}")
    print(f"OOF predictions: {oof_path}")
    print(f"subgroup recall: {subgroup_path}")
    print(f"branch metrics : {branch_metric_path}")
    print(f"thresholds     : {threshold_path}")
    print(f"summary        : {summary_path}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
