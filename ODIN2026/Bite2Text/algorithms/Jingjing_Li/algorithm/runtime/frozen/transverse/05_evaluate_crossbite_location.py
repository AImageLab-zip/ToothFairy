#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_evaluate_crossbite_location.py

Goal
----
Finish the next Transverse layer as quickly as possible:

    crossbite_present
        -> crossbite_location
           None / Left_Side / Right_Side / Bilateral

Competition-oriented design
----------------------------
The FINAL present classifier has already been selected:

    concat_raw_reference
    posterior 52 + anterior 45 = 97 features

This script does NOT change that classifier. It reuses its strict OOF prediction
from evaluation_fusion_v3 and evaluates location on the SAME outer folds.

We compare three practical location strategies:

1) direct4
   One 4-class Logistic model:
       None / Left_Side / Right_Side / Bilateral
   using the same 97 geometry features.

2) positive3 + present gate
   Train only on true crossbite positives whose report location is explicitly:
       Left_Side / Right_Side / Bilateral
   The model only learns "where".
   At inference:
       selected crossbite_present == False -> None
       selected crossbite_present == True  -> 3-class location prediction

3) shared-side binary + present gate
   Convert every patient into a LEFT row and RIGHT row and train ONE shared
   side-involvement Logistic model. This enforces left/right symmetry and helps
   the smaller Right_Side class.
   It predicts P(left involved) and P(right involved), then decodes:
       Left / Right / Bilateral
   A small Bilateral scale is tuned on inner OOF only.

Primary competition metric
--------------------------
END-TO-END location field accuracy on report GT:
    None / Left_Side / Right_Side / Bilateral

We also report POSITIVE-ONLY location accuracy:
    Left_Side / Right_Side / Bilateral

Important
---------
- Not_Assessable / Not_Mentioned are NOT treated as visual classes.
- No GT crossbite_teeth is used as the location label.
  We train directly against report crossbite_location because that is what the
  final report generator needs to reproduce.
- Same master outer folds as crossbite_present are reused from the v3 OOF file.
- Hyperparameter selection is performed only inside outer-training data.

Inputs
------
Posterior:
  results/crossbite_geometry_features_v3.csv
  fallback: results/crossbite_geometry_features.csv

Anterior:
  results/anterior_crossbite_geometry_features_v2.csv

Present OOF:
  evaluation_fusion_v3/crossbite_branch_oof_predictions_v3.csv

GT:
  data/cleaned_orthodontic_data_v3.json

Outputs
-------
evaluation_location_v1/
  crossbite_location_model_comparison_v1.csv
  crossbite_location_fold_metrics_v1.csv
  crossbite_location_oof_predictions_v1.csv
  crossbite_location_summary_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
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
        "scikit-learn is required. Activate the odin_baseline environment."
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_POSTERIOR = ROOT / "results" / "crossbite_geometry_features_v3.csv"
DEFAULT_ANTERIOR = ROOT / "results" / "anterior_crossbite_geometry_features_v2.csv"
DEFAULT_PRESENT_OOF = (
    ROOT
    / "evaluation_fusion_v3"
    / "crossbite_branch_oof_predictions_v3.csv"
)
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_OUTPUT = ROOT / "evaluation_location_v1"

LOCATION_CLASSES = (
    "None",
    "Left_Side",
    "Right_Side",
    "Bilateral",
)
POSITIVE_LOCATION_CLASSES = (
    "Left_Side",
    "Right_Side",
    "Bilateral",
)

ANTERIOR_TEETH = (13, 12, 11, 21, 22, 23)
RIGHT_ANTERIOR = (13, 12, 11)
LEFT_ANTERIOR = (21, 22, 23)

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
    p.add_argument("--posterior-features", type=Path, default=DEFAULT_POSTERIOR)
    p.add_argument("--anterior-features", type=Path, default=DEFAULT_ANTERIOR)
    p.add_argument("--present-oof", type=Path, default=DEFAULT_PRESENT_OOF)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def bool_value(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def bool_series(s: pd.Series) -> pd.Series:
    return s.map(bool_value).astype(bool)


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {context}: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )


def resolve_posterior(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.exists():
        return p

    fallback = ROOT / "results" / "crossbite_geometry_features.csv"
    if fallback.exists():
        print(
            "WARNING: crossbite_geometry_features_v3.csv not found; "
            f"using {fallback}"
        )
        return fallback.resolve()

    raise FileNotFoundError(f"Posterior feature CSV not found: {p}")


def prefix_dataframe(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(
        columns={
            c: f"{prefix}{c}"
            for c in df.columns
            if c != "patient_id"
        }
    )


def load_gt(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("GT JSON must be a top-level list.")

    rows = []
    for rec in data:
        if not isinstance(rec, dict):
            continue

        report = rec.get("photo_report_v3") or {}
        trans = report.get("2_transverse_occlusion") or {}

        present = trans.get("crossbite_present")
        location = trans.get("crossbite_location")

        rows.append(
            {
                "patient_id": rec.get("patient_id"),
                "gt_source": rec.get("photo_report_v3_input_source"),
                "gt_crossbite_present": (
                    int(present) if isinstance(present, bool) else np.nan
                ),
                "gt_crossbite_location": location,
            }
        )

    out = pd.DataFrame(rows)

    if out["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient_id in GT JSON.")

    return out


def safe_numeric(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    valid = []
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() >= 5 and df[c].nunique(dropna=True) >= 2:
            valid.append(c)
    return valid


def posterior_valid_mask(df: pd.DataFrame) -> pd.Series:
    cols = [
        "post__left_full_valid",
        "post__right_full_valid",
        "post__left_occlusal_valid",
        "post__right_occlusal_valid",
    ]
    require_columns(df, cols, "posterior validity")

    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= bool_series(df[c])
    return mask


def anterior_valid_mask(df: pd.DataFrame) -> pd.Series:
    for c in (
        "ant__anterior_radial_valid",
        "ant__anterior_geometry_valid",
    ):
        if c in df.columns:
            return bool_series(df[c])
    raise ValueError("Anterior radial validity column not found.")


def build_posterior_52(df: pd.DataFrame) -> List[str]:
    cols = []
    for side in ("left", "right"):
        for variant in ("full", "occlusal"):
            for suffix in POST_WHOLE_SUFFIXES:
                cols.append(f"post__{side}_{variant}_{suffix}")

    cols = safe_numeric(df, cols)
    if not cols:
        raise ValueError("Could not build posterior 52-feature set.")
    return cols


def _row_smallest(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v):
            out[i] = float(np.min(v))
    return out


def _row_second_smallest(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v) >= 2:
            out[i] = float(np.partition(v, 1)[1])
        elif len(v) == 1:
            out[i] = float(v[0])
    return out


def _row_two_smallest_mean(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v) >= 2:
            out[i] = float(np.mean(np.partition(v, 1)[:2]))
        elif len(v) == 1:
            out[i] = float(v[0])
    return out


def _row_nanmax(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v):
            out[i] = float(np.max(v))
    return out


def derive_scope(
    out: pd.DataFrame,
    q20: np.ndarray,
    med: np.ndarray,
    neg: np.ndarray,
    teeth: Tuple[int, ...],
    idx_map: Dict[int, int],
    prefix: str,
) -> List[str]:
    idx = [idx_map[t] for t in teeth]
    q = q20[:, idx]
    m = med[:, idx]
    n = neg[:, idx]

    cols = []

    def add(name, values):
        c = f"{prefix}{name}"
        out[c] = values
        cols.append(c)

    valid = np.isfinite(q) & np.isfinite(m) & np.isfinite(n)

    add("valid_teeth", np.sum(valid, axis=1).astype(float))
    add("worst_q20", _row_smallest(q))
    add("second_worst_q20", _row_second_smallest(q))
    add("worst_two_q20_mean", _row_two_smallest_mean(q))
    add("worst_median", _row_smallest(m))
    add("max_negative_ratio", _row_nanmax(n))
    add("q20_below0_count", np.sum(np.isfinite(q) & (q < 0), axis=1).astype(float))
    add("q20_below05_count", np.sum(np.isfinite(q) & (q < -0.5), axis=1).astype(float))
    add("median_below0_count", np.sum(np.isfinite(m) & (m < 0), axis=1).astype(float))

    return cols


def build_anterior_45(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Dict[str, np.ndarray]]:
    """
    Rebuild exactly the same 45 raw anterior features used by the selected
    concat_raw_reference model.

    Also returns the raw per-tooth arrays for side-symmetric location modeling.
    """
    n = len(df)
    q20 = np.full((n, 6), np.nan)
    med = np.full((n, 6), np.nan)
    neg = np.full((n, 6), np.nan)

    required = []
    for tooth in ANTERIOR_TEETH:
        base = f"ant__anterior_radial_tooth{tooth}_incisal"
        required.extend(
            [
                f"{base}_q20_margin_mm",
                f"{base}_median_margin_mm",
                f"{base}_negative_ratio",
            ]
        )
    require_columns(df, required, "anterior radial tooth features")

    for j, tooth in enumerate(ANTERIOR_TEETH):
        base = f"ant__anterior_radial_tooth{tooth}_incisal"
        q20[:, j] = pd.to_numeric(
            df[f"{base}_q20_margin_mm"], errors="coerce"
        ).to_numpy(float)
        med[:, j] = pd.to_numeric(
            df[f"{base}_median_margin_mm"], errors="coerce"
        ).to_numpy(float)
        neg[:, j] = pd.to_numeric(
            df[f"{base}_negative_ratio"], errors="coerce"
        ).to_numpy(float)

    out = pd.DataFrame(index=df.index)
    cols = []

    for j, tooth in enumerate(ANTERIOR_TEETH):
        for suffix, arr in (
            ("q20", q20),
            ("median", med),
            ("negative_ratio", neg),
        ):
            c = f"ant_raw__tooth{tooth}_{suffix}"
            out[c] = arr[:, j]
            cols.append(c)

    idx_map = {t: i for i, t in enumerate(ANTERIOR_TEETH)}

    cols += derive_scope(
        out,
        q20,
        med,
        neg,
        ANTERIOR_TEETH,
        idx_map,
        "ant_raw__all_",
    )
    cols += derive_scope(
        out,
        q20,
        med,
        neg,
        RIGHT_ANTERIOR,
        idx_map,
        "ant_raw__right_",
    )
    cols += derive_scope(
        out,
        q20,
        med,
        neg,
        LEFT_ANTERIOR,
        idx_map,
        "ant_raw__left_",
    )

    cols = safe_numeric(out, cols)

    arrays = {
        "q20": q20,
        "median": med,
        "negative_ratio": neg,
    }
    return out, cols, arrays


def make_multiclass_pipeline(C: float, class_weight, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
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


def make_binary_pipeline(C: float, class_weight, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def location_metrics(y_true, y_pred, labels=LOCATION_CLASSES) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
    }


def safe_stratified_splits(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
):
    counts = pd.Series(labels).value_counts()
    k = min(int(n_splits), int(counts.min()))
    if k < 2:
        raise RuntimeError(
            f"Cannot stratify: class counts={counts.to_dict()}"
        )

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(labels), 1))
    return list(skf.split(dummy, labels))


def tune_multiclass(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    inner_folds: int,
):
    """
    Tune only small regularization/class-weight choices on inner OOF accuracy.
    Competition target here is exact class correctness.
    """
    candidates = []
    splits = safe_stratified_splits(
        y,
        inner_folds,
        seed,
    )

    for C in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        for class_weight in (None, "balanced"):
            oof = np.empty(len(y), dtype=object)

            for inner_fold, (tr, va) in enumerate(splits):
                model = make_multiclass_pipeline(
                    C=C,
                    class_weight=class_weight,
                    seed=seed + inner_fold,
                )
                model.fit(X.iloc[tr], y[tr])
                oof[va] = model.predict(X.iloc[va])

            m = location_metrics(
                y,
                oof,
                labels=tuple(sorted(set(y.tolist()))),
            )
            candidates.append(
                (
                    (
                        m["accuracy"],
                        m["macro_f1"],
                        m["balanced_accuracy"],
                        -abs(np.log10(C)),
                        1 if class_weight is None else 0,
                    ),
                    C,
                    class_weight,
                    m,
                )
            )

    best = max(candidates, key=lambda z: z[0])
    return best[1], best[2], best[3]


def positive_argmax_from_model(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(X)
    classes = model.named_steps["classifier"].classes_.astype(str)

    available = [
        c for c in POSITIVE_LOCATION_CLASSES
        if c in classes
    ]
    if len(available) != 3:
        raise RuntimeError(
            f"Expected all 3 positive location classes, got {classes.tolist()}"
        )

    idx = [int(np.where(classes == c)[0][0]) for c in available]
    local = prob[:, idx]
    winner = np.argmax(local, axis=1)
    return np.asarray([available[i] for i in winner], dtype=object)


# -------------------------------------------------------------------------
# Shared side model
# -------------------------------------------------------------------------

ROLE_TEETH = {
    "right": {
        "canine": 13,
        "lateral": 12,
        "central": 11,
    },
    "left": {
        "canine": 23,
        "lateral": 22,
        "central": 21,
    },
}


def side_target(location: str, side: str) -> int:
    if location == "Bilateral":
        return 1
    if side == "left" and location == "Left_Side":
        return 1
    if side == "right" and location == "Right_Side":
        return 1
    return 0


def build_side_rows(
    merged: pd.DataFrame,
    ant_arrays: Dict[str, np.ndarray],
    patient_indices: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    One shared model for both sides.

    For each patient we create:
       row(self=left, other=right)
       row(self=right, other=left)

    Tooth identities are normalized to roles:
       canine / lateral / central
    so left and right use the same feature coordinates.
    """
    rows = []
    targets = []
    owner = []

    idx_map = {t: i for i, t in enumerate(ANTERIOR_TEETH)}

    for patient_idx in patient_indices:
        location = str(
            merged.loc[patient_idx, "gt_crossbite_location"]
        )

        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            r = {}

            # Posterior: self and contralateral side.
            for role, src_side in (("self", side), ("other", other)):
                for variant in ("full", "occlusal"):
                    for suffix in POST_WHOLE_SUFFIXES:
                        src = f"post__{src_side}_{variant}_{suffix}"
                        if src in merged.columns:
                            r[
                                f"post_{role}_{variant}_{suffix}"
                            ] = merged.loc[patient_idx, src]

            # Anterior: normalize FDI to canine/lateral/central roles.
            for role, src_side in (("self", side), ("other", other)):
                teeth = ROLE_TEETH[src_side]
                for tooth_role, tooth in teeth.items():
                    j = idx_map[tooth]
                    for value_name, arr in ant_arrays.items():
                        r[
                            f"ant_{role}_{tooth_role}_{value_name}"
                        ] = arr[patient_idx, j]

                # compact side summaries from its three normalized teeth
                js = [idx_map[t] for t in teeth.values()]
                q = ant_arrays["q20"][patient_idx, js]
                m = ant_arrays["median"][patient_idx, js]
                n = ant_arrays["negative_ratio"][patient_idx, js]

                qv = q[np.isfinite(q)]
                mv = m[np.isfinite(m)]
                nv = n[np.isfinite(n)]

                r[f"ant_{role}_worst_q20"] = (
                    float(np.min(qv)) if len(qv) else np.nan
                )
                r[f"ant_{role}_worst_median"] = (
                    float(np.min(mv)) if len(mv) else np.nan
                )
                r[f"ant_{role}_max_negative_ratio"] = (
                    float(np.max(nv)) if len(nv) else np.nan
                )
                r[f"ant_{role}_q20_below0_count"] = float(
                    np.sum(qv < 0.0)
                )
                r[f"ant_{role}_q20_below05_count"] = float(
                    np.sum(qv < -0.5)
                )

            rows.append(r)
            targets.append(side_target(location, side))
            owner.append(patient_idx)

    X = pd.DataFrame(rows)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove unusable / constant columns inside this whole constructed table.
    keep = [
        c for c in X.columns
        if X[c].notna().sum() >= 5
        and X[c].nunique(dropna=True) >= 2
    ]
    X = X[keep]

    return (
        X.reset_index(drop=True),
        np.asarray(targets, dtype=int),
        np.asarray(owner, dtype=int),
    )


def side_inner_oof(
    merged: pd.DataFrame,
    ant_arrays: Dict[str, np.ndarray],
    train_patient_idx: np.ndarray,
    seed: int,
    inner_folds: int,
):
    """
    Patient-group-safe inner OOF for the shared side model.
    """
    labels4 = (
        merged.loc[
            train_patient_idx,
            "gt_crossbite_location",
        ]
        .astype(str)
        .to_numpy()
    )

    splits = safe_stratified_splits(
        labels4,
        inner_folds,
        seed,
    )

    p_left = np.full(len(train_patient_idx), np.nan)
    p_right = np.full(len(train_patient_idx), np.nan)

    # Map global patient index -> local train position.
    pos_map = {
        int(g): i
        for i, g in enumerate(train_patient_idx)
    }

    # Tune binary model hyperparameters by SIDE-ROW accuracy.
    best_cfg = None

    for C in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        for cw in (None, "balanced"):
            pred_all = []
            true_all = []

            for inner_fold, (tr_local, va_local) in enumerate(splits):
                tr_pat = train_patient_idx[tr_local]
                va_pat = train_patient_idx[va_local]

                Xtr, ytr, _ = build_side_rows(
                    merged,
                    ant_arrays,
                    tr_pat,
                )
                Xva, yva, _ = build_side_rows(
                    merged,
                    ant_arrays,
                    va_pat,
                )

                # Align columns exactly.
                Xva = Xva.reindex(columns=Xtr.columns)

                model = make_binary_pipeline(
                    C=C,
                    class_weight=cw,
                    seed=seed + inner_fold,
                )
                model.fit(Xtr, ytr)
                pred_all.extend(model.predict(Xva).tolist())
                true_all.extend(yva.tolist())

            acc = accuracy_score(true_all, pred_all)
            macro = f1_score(
                true_all,
                pred_all,
                average="macro",
                zero_division=0,
            )
            key = (
                float(acc),
                float(macro),
                -abs(np.log10(C)),
                1 if cw is None else 0,
            )

            if best_cfg is None or key > best_cfg[0]:
                best_cfg = (key, C, cw)

    _, best_C, best_cw = best_cfg

    # Now generate inner OOF probabilities using the selected config.
    for inner_fold, (tr_local, va_local) in enumerate(splits):
        tr_pat = train_patient_idx[tr_local]
        va_pat = train_patient_idx[va_local]

        Xtr, ytr, _ = build_side_rows(
            merged,
            ant_arrays,
            tr_pat,
        )
        Xva, _, owner_va = build_side_rows(
            merged,
            ant_arrays,
            va_pat,
        )
        Xva = Xva.reindex(columns=Xtr.columns)

        model = make_binary_pipeline(
            C=best_C,
            class_weight=best_cw,
            seed=seed + 100 + inner_fold,
        )
        model.fit(Xtr, ytr)
        pp = model.predict_proba(Xva)[:, 1]

        # build_side_rows outputs left then right for each patient.
        for prob, owner_idx, side_ord in zip(
            pp,
            owner_va,
            np.tile(np.asarray([0, 1]), len(va_pat)),
        ):
            local_pos = pos_map[int(owner_idx)]
            if side_ord == 0:
                p_left[local_pos] = prob
            else:
                p_right[local_pos] = prob

    if not np.all(np.isfinite(p_left)) or not np.all(np.isfinite(p_right)):
        raise RuntimeError("Missing inner shared-side OOF probability.")

    return p_left, p_right, best_C, best_cw


def decode_positive_side(
    p_left: np.ndarray,
    p_right: np.ndarray,
    bilateral_scale: float,
) -> np.ndarray:
    score_left = p_left * (1.0 - p_right)
    score_right = p_right * (1.0 - p_left)
    score_bilat = bilateral_scale * p_left * p_right

    scores = np.column_stack(
        [score_left, score_right, score_bilat]
    )
    names = np.asarray(
        ["Left_Side", "Right_Side", "Bilateral"],
        dtype=object,
    )
    return names[np.argmax(scores, axis=1)]


def decode_raw4_side(
    p_left: np.ndarray,
    p_right: np.ndarray,
    bilateral_scale: float,
) -> np.ndarray:
    score_none = (1.0 - p_left) * (1.0 - p_right)
    score_left = p_left * (1.0 - p_right)
    score_right = p_right * (1.0 - p_left)
    score_bilat = bilateral_scale * p_left * p_right

    scores = np.column_stack(
        [score_none, score_left, score_right, score_bilat]
    )
    names = np.asarray(LOCATION_CLASSES, dtype=object)
    return names[np.argmax(scores, axis=1)]


def tune_bilateral_scale(
    y_location: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
) -> Tuple[float, float]:
    positive = np.isin(
        y_location,
        POSITIVE_LOCATION_CLASSES,
    )

    best = None
    for scale in (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
    ):
        pred = decode_positive_side(
            p_left[positive],
            p_right[positive],
            scale,
        )
        acc = accuracy_score(
            y_location[positive],
            pred,
        )
        macro = f1_score(
            y_location[positive],
            pred,
            labels=list(POSITIVE_LOCATION_CLASSES),
            average="macro",
            zero_division=0,
        )
        key = (
            float(acc),
            float(macro),
            -abs(np.log(scale)),
        )
        if best is None or key > best[0]:
            best = (key, float(scale), float(acc))
    return best[1], best[2]


def fit_shared_side_outer(
    merged: pd.DataFrame,
    ant_arrays: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    C: float,
    class_weight,
    seed: int,
):
    Xtr, ytr, _ = build_side_rows(
        merged,
        ant_arrays,
        train_idx,
    )
    Xva, _, owner_va = build_side_rows(
        merged,
        ant_arrays,
        valid_idx,
    )
    Xva = Xva.reindex(columns=Xtr.columns)

    model = make_binary_pipeline(
        C=C,
        class_weight=class_weight,
        seed=seed,
    )
    model.fit(Xtr, ytr)
    pp = model.predict_proba(Xva)[:, 1]

    p_left = np.full(len(valid_idx), np.nan)
    p_right = np.full(len(valid_idx), np.nan)

    pos_map = {
        int(g): i
        for i, g in enumerate(valid_idx)
    }

    for prob, owner_idx, side_ord in zip(
        pp,
        owner_va,
        np.tile(np.asarray([0, 1]), len(valid_idx)),
    ):
        pos = pos_map[int(owner_idx)]
        if side_ord == 0:
            p_left[pos] = prob
        else:
            p_right[pos] = prob

    return p_left, p_right


def main():
    args = parse_args()

    post_path = resolve_posterior(args.posterior_features)
    ant_path = args.anterior_features.expanduser().resolve()
    present_oof_path = args.present_oof.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (ant_path, present_oof_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    post = pd.read_csv(post_path, low_memory=False)
    ant = pd.read_csv(ant_path, low_memory=False)
    present_oof = pd.read_csv(present_oof_path, low_memory=False)
    gt = load_gt(gt_path)

    require_columns(post, ["patient_id"], "posterior")
    require_columns(ant, ["patient_id"], "anterior")
    require_columns(
        present_oof,
        [
            "patient_id",
            "outer_fold",
            "common_eligible",
            "concat_raw_reference_pred",
        ],
        "selected present OOF",
    )

    post = prefix_dataframe(post, "post__")
    ant = prefix_dataframe(ant, "ant__")

    gt = gt.loc[
        gt["gt_source"].eq("photo_en")
        & gt["gt_crossbite_present"].notna()
    ].copy()

    merged = (
        gt.merge(post, on="patient_id", how="left", validate="one_to_one")
        .merge(ant, on="patient_id", how="left", validate="one_to_one")
        .merge(
            present_oof[
                [
                    "patient_id",
                    "outer_fold",
                    "common_eligible",
                    "concat_raw_reference_pred",
                ]
            ],
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    merged["gt_crossbite_present"] = (
        merged["gt_crossbite_present"].astype(int)
    )

    explicit_location = merged[
        "gt_crossbite_location"
    ].isin(LOCATION_CLASSES)

    post_valid = posterior_valid_mask(merged).to_numpy(bool)
    ant_valid = anterior_valid_mask(merged).to_numpy(bool)
    common = (
        post_valid
        & ant_valid
        & bool_series(merged["common_eligible"]).to_numpy(bool)
    )

    eligible = common & explicit_location.to_numpy(bool)

    y_location = (
        merged["gt_crossbite_location"]
        .astype(str)
        .to_numpy()
    )
    y_present = (
        merged["gt_crossbite_present"]
        .astype(int)
        .to_numpy()
    )
    folds = (
        pd.to_numeric(merged["outer_fold"], errors="raise")
        .astype(int)
        .to_numpy()
    )
    present_pred = (
        pd.to_numeric(
            merged["concat_raw_reference_pred"],
            errors="coerce",
        )
        .to_numpy(float)
    )

    post_cols = build_posterior_52(merged)
    ant_df, ant_cols, ant_arrays = build_anterior_45(merged)

    X_concat = pd.concat(
        [
            merged[post_cols].reset_index(drop=True),
            ant_df[ant_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    X_concat.index = merged.index

    print("=" * 100)
    print("Crossbite location v1")
    print("=" * 100)
    print(f"matched bool-GT rows       : {len(merged)}")
    print(f"strict common              : {int(common.sum())}")
    print(f"explicit location eligible : {int(eligible.sum())}")
    print(f"97 concat features         : {X_concat.shape[1]}")
    print()
    print("Explicit location counts on eligible:")
    print(
        merged.loc[
            eligible,
            "gt_crossbite_location",
        ].value_counts().to_string()
    )
    print("=" * 100)

    n = len(merged)
    pred_store = {}
    prob_left_store = np.full(n, np.nan)
    prob_right_store = np.full(n, np.nan)

    fold_rows = []

    model_names = (
        "direct4_raw",
        "direct4_present_gated",
        "positive3_present_gated",
        "positive3_oracle_present",
        "side_shared_raw4",
        "side_shared_present_gated",
        "side_shared_oracle_present",
    )

    for name in model_names:
        pred_store[name] = np.full(n, None, dtype=object)

    for fold in sorted(np.unique(folds)):
        train_idx = np.where(
            eligible & (folds != fold)
        )[0]
        valid_idx = np.where(
            eligible & (folds == fold)
        )[0]

        ytr4 = y_location[train_idx]

        # -------------------------------------------------------------
        # A) Direct 4-class
        # -------------------------------------------------------------
        C4, cw4, tune4 = tune_multiclass(
            X_concat.iloc[train_idx],
            ytr4,
            seed=args.seed + 1000 + int(fold),
            inner_folds=args.inner_folds,
        )

        direct4 = make_multiclass_pipeline(
            C=C4,
            class_weight=cw4,
            seed=args.seed + 1100 + int(fold),
        )
        direct4.fit(
            X_concat.iloc[train_idx],
            ytr4,
        )

        raw4 = direct4.predict(
            X_concat.iloc[valid_idx]
        ).astype(object)

        forced_positive4 = positive_argmax_from_model(
            direct4,
            X_concat.iloc[valid_idx],
        )

        gated4 = np.where(
            present_pred[valid_idx] >= 0.5,
            forced_positive4,
            "None",
        ).astype(object)

        pred_store["direct4_raw"][valid_idx] = raw4
        pred_store["direct4_present_gated"][valid_idx] = gated4

        # -------------------------------------------------------------
        # B) Positive-only 3-class
        # -------------------------------------------------------------
        train_pos = train_idx[
            np.isin(
                y_location[train_idx],
                POSITIVE_LOCATION_CLASSES,
            )
            & (y_present[train_idx] == 1)
        ]

        C3, cw3, tune3 = tune_multiclass(
            X_concat.iloc[train_pos],
            y_location[train_pos],
            seed=args.seed + 2000 + int(fold),
            inner_folds=args.inner_folds,
        )

        positive3 = make_multiclass_pipeline(
            C=C3,
            class_weight=cw3,
            seed=args.seed + 2100 + int(fold),
        )
        positive3.fit(
            X_concat.iloc[train_pos],
            y_location[train_pos],
        )

        positive3_pred = positive3.predict(
            X_concat.iloc[valid_idx]
        ).astype(object)

        gated3 = np.where(
            present_pred[valid_idx] >= 0.5,
            positive3_pred,
            "None",
        ).astype(object)

        oracle3 = np.where(
            y_present[valid_idx] == 1,
            positive3_pred,
            "None",
        ).astype(object)

        pred_store["positive3_present_gated"][valid_idx] = gated3
        pred_store["positive3_oracle_present"][valid_idx] = oracle3

        # -------------------------------------------------------------
        # C) Shared left/right side model
        # -------------------------------------------------------------
        pL_inner, pR_inner, side_C, side_cw = side_inner_oof(
            merged=merged,
            ant_arrays=ant_arrays,
            train_patient_idx=train_idx,
            seed=args.seed + 3000 + int(fold),
            inner_folds=args.inner_folds,
        )

        bilat_scale, inner_pos_acc = tune_bilateral_scale(
            y_location[train_idx],
            pL_inner,
            pR_inner,
        )

        pL_valid, pR_valid = fit_shared_side_outer(
            merged=merged,
            ant_arrays=ant_arrays,
            train_idx=train_idx,
            valid_idx=valid_idx,
            C=side_C,
            class_weight=side_cw,
            seed=args.seed + 3100 + int(fold),
        )

        prob_left_store[valid_idx] = pL_valid
        prob_right_store[valid_idx] = pR_valid

        side_pos = decode_positive_side(
            pL_valid,
            pR_valid,
            bilat_scale,
        )
        side_raw4 = decode_raw4_side(
            pL_valid,
            pR_valid,
            bilat_scale,
        )
        side_gated = np.where(
            present_pred[valid_idx] >= 0.5,
            side_pos,
            "None",
        ).astype(object)
        side_oracle = np.where(
            y_present[valid_idx] == 1,
            side_pos,
            "None",
        ).astype(object)

        pred_store["side_shared_raw4"][valid_idx] = side_raw4
        pred_store["side_shared_present_gated"][valid_idx] = side_gated
        pred_store["side_shared_oracle_present"][valid_idx] = side_oracle

        # Fold metrics for competition-facing models.
        for model_name in (
            "direct4_raw",
            "direct4_present_gated",
            "positive3_present_gated",
            "side_shared_raw4",
            "side_shared_present_gated",
        ):
            pred = pred_store[model_name][valid_idx]
            m = location_metrics(
                y_location[valid_idx],
                pred,
            )

            positive_valid = np.isin(
                y_location[valid_idx],
                POSITIVE_LOCATION_CLASSES,
            )
            pos_acc = (
                float(
                    accuracy_score(
                        y_location[valid_idx][positive_valid],
                        pred[positive_valid],
                    )
                )
                if positive_valid.any()
                else np.nan
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    **m,
                    "positive_location_accuracy": pos_acc,
                    "direct4_C": C4,
                    "direct4_class_weight": str(cw4),
                    "positive3_C": C3,
                    "positive3_class_weight": str(cw3),
                    "side_C": side_C,
                    "side_class_weight": str(side_cw),
                    "bilateral_scale": bilat_scale,
                    "inner_side_positive_accuracy": inner_pos_acc,
                }
            )

    # -----------------------------------------------------------------
    # Overall
    # -----------------------------------------------------------------
    eval_idx = np.where(eligible)[0]
    positive_eval = eval_idx[
        np.isin(
            y_location[eval_idx],
            POSITIVE_LOCATION_CLASSES,
        )
        & (y_present[eval_idx] == 1)
    ]

    comparison_rows = []
    for model_name in model_names:
        pred = pred_store[model_name][eval_idx]

        if any(v is None for v in pred):
            raise RuntimeError(
                f"{model_name}: missing OOF prediction on eligible cohort."
            )

        overall = location_metrics(
            y_location[eval_idx],
            pred,
        )

        pos_pred = pred_store[model_name][positive_eval]
        pos_true = y_location[positive_eval]

        pos_acc = float(
            accuracy_score(pos_true, pos_pred)
        )
        pos_macro = float(
            f1_score(
                pos_true,
                pos_pred,
                labels=list(POSITIVE_LOCATION_CLASSES),
                average="macro",
                zero_division=0,
            )
        )

        comparison_rows.append(
            {
                "model": model_name,
                **overall,
                "positive_n": int(len(positive_eval)),
                "positive_location_accuracy": pos_acc,
                "positive_location_macro_f1": pos_macro,
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        ["accuracy", "positive_location_accuracy"],
        ascending=False,
    )

    fold_df = pd.DataFrame(fold_rows)

    oof = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "gt_crossbite_present": y_present,
            "gt_crossbite_location": y_location,
            "outer_fold": folds,
            "common_eligible": common.astype(int),
            "location_eligible": eligible.astype(int),
            "selected_present_pred": present_pred,
            "side_shared_left_prob": prob_left_store,
            "side_shared_right_prob": prob_right_store,
        }
    )

    for model_name in model_names:
        oof[f"{model_name}_pred"] = pred_store[model_name]

    # Confusion matrices, flattened for easy inspection.
    confusion = {}
    for model_name in model_names:
        pred = pred_store[model_name][eval_idx]
        cm = confusion_matrix(
            y_location[eval_idx],
            pred,
            labels=list(LOCATION_CLASSES),
        )
        confusion[model_name] = {
            true_label: {
                pred_label: int(cm[i, j])
                for j, pred_label in enumerate(LOCATION_CLASSES)
            }
            for i, true_label in enumerate(LOCATION_CLASSES)
        }

    summary = {
        "version": "crossbite-location-v1-competition-oof",
        "selected_present_model": "concat_raw_reference",
        "gt_source": "photo_en_only",
        "location_classes": list(LOCATION_CLASSES),
        "excluded_visual_training_labels": [
            "Not_Assessable",
            "Not_Mentioned",
        ],
        "cohort": {
            "matched_bool_gt": int(len(merged)),
            "strict_common": int(common.sum()),
            "explicit_location_eligible": int(eligible.sum()),
            "positive_location_n": int(len(positive_eval)),
            "location_counts": {
                str(k): int(v)
                for k, v in (
                    merged.loc[
                        eligible,
                        "gt_crossbite_location",
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
        },
        "feature_counts": {
            "posterior": int(len(post_cols)),
            "anterior": int(len(ant_cols)),
            "concat": int(X_concat.shape[1]),
        },
        "models": comparison_df.to_dict(orient="records"),
        "confusion_matrices": confusion,
        "selection_rule": (
            "Primary: highest end-to-end 4-class location accuracy. "
            "Positive-only location accuracy is the main tie-breaker."
        ),
    }

    comparison_path = (
        out_dir
        / "crossbite_location_model_comparison_v1.csv"
    )
    fold_path = (
        out_dir
        / "crossbite_location_fold_metrics_v1.csv"
    )
    oof_path = (
        out_dir
        / "crossbite_location_oof_predictions_v1.csv"
    )
    summary_path = (
        out_dir
        / "crossbite_location_summary_v1.json"
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
    oof.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )
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
    print("=" * 100)
    print("LOCATION OOF MODEL COMPARISON")
    print("=" * 100)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Outputs:")
    print(f"  {comparison_path}")
    print(f"  {fold_path}")
    print(f"  {oof_path}")
    print(f"  {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
