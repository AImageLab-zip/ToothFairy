#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_evaluate_transverse_severity_type.py

Final quick OOF test for the last two Transverse fields.

Key finding from 10
-------------------
Visual labels are almost degenerate:

severity explicit visual:
    Normal 474 / 491 = 96.54%

type explicit visual:
    Maxillary_Constriction 302 / 304 = 99.34%

So the hard problem is NOT visual subclassification. It is REPORT STATE:
    should the field be a visual label,
    Not_Mentioned,
    or Not_Assessable?

Competition-oriented strategy
-----------------------------
severity:
    state target:
        VISUAL  -> output Normal
        NM      -> output Not_Mentioned
        NA      -> output Not_Assessable

    Mild/Moderate/Severe are intentionally collapsed into VISUAL for state
    training and decoded as Normal. They are too rare to justify a stable
    standalone classifier.

type:
    state target:
        VISUAL  -> output Maxillary_Constriction
        NM      -> output Not_Mentioned
        NA      -> output Not_Assessable

    Contraction is intentionally decoded as Maxillary_Constriction because only
    two explicit Contraction cases exist.

Models compared
---------------
1) cheap baseline from 10:
   severity:
       selected present False -> Normal
       selected present True  -> Not_Mentioned

   type:
       selected present False -> Not_Mentioned
       selected present True  -> Maxillary_Constriction

2) state_logistic_97:
   posterior 52 + anterior 45 geometry -> report state

3) state_logistic_97_no_na:
   For the ultra-rare NA issue, collapse NA to the majority non-visual state:
       severity: NA -> Not_Mentioned for training/decoding
       type:     NA -> Not_Mentioned
   This directly optimizes exact accuracy and avoids unstable tiny classes.

Strict OOF
----------
Same outer fold ids as crossbite-present OOF.
Hyperparameters are selected only on outer-training data.

Primary metric
--------------
Exact field accuracy on the strict common cohort.

Outputs
-------
evaluation_transverse_final_v1/
    transverse_final_model_comparison_v1.csv
    transverse_final_fold_metrics_v1.csv
    transverse_final_oof_predictions_v1.csv
    transverse_final_summary_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
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
DEFAULT_POSTERIOR = ROOT / "results" / "crossbite_geometry_features_v3.csv"
DEFAULT_ANTERIOR = ROOT / "results" / "anterior_crossbite_geometry_features_v2.csv"
DEFAULT_PRESENT_OOF = (
    ROOT / "evaluation_fusion_v3" / "crossbite_branch_oof_predictions_v3.csv"
)
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_OUTPUT = ROOT / "evaluation_transverse_final_v1"

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
    return p.parse_args()


def require_columns(df, cols, context):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def bool_value(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def prefix_dataframe(df, prefix):
    return df.rename(
        columns={c: f"{prefix}{c}" for c in df.columns if c != "patient_id"}
    )


def resolve_posterior(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.exists():
        return p
    fallback = ROOT / "results" / "crossbite_geometry_features.csv"
    if fallback.exists():
        print(f"WARNING: using fallback posterior file: {fallback}")
        return fallback.resolve()
    raise FileNotFoundError(p)


def load_gt(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for rec in data:
        report = rec.get("photo_report_v3") or {}
        trans = report.get("2_transverse_occlusion") or {}
        rows.append(
            {
                "patient_id": rec.get("patient_id"),
                "gt_source": rec.get("photo_report_v3_input_source"),
                "gt_severity": trans.get("severity"),
                "gt_transverse_type": trans.get("type"),
            }
        )
    out = pd.DataFrame(rows)
    if out["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient_id in GT.")
    return out


def safe_numeric(df, cols):
    valid = []
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() >= 5 and df[c].nunique(dropna=True) >= 2:
            valid.append(c)
    return valid


def build_posterior_52(df):
    cols = []
    for side in ("left", "right"):
        for variant in ("full", "occlusal"):
            for suffix in POST_WHOLE_SUFFIXES:
                cols.append(f"post__{side}_{variant}_{suffix}")
    return safe_numeric(df, cols)


def _row_smallest(arr):
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v):
            out[i] = float(np.min(v))
    return out


def _row_second_smallest(arr):
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v) >= 2:
            out[i] = float(np.partition(v, 1)[1])
        elif len(v) == 1:
            out[i] = float(v[0])
    return out


def _row_two_smallest_mean(arr):
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v) >= 2:
            out[i] = float(np.mean(np.partition(v, 1)[:2]))
        elif len(v) == 1:
            out[i] = float(v[0])
    return out


def _row_nanmax(arr):
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        v = arr[i][np.isfinite(arr[i])]
        if len(v):
            out[i] = float(np.max(v))
    return out


def derive_scope(out, q20, med, neg, teeth, idx_map, prefix):
    idx = [idx_map[t] for t in teeth]
    q = q20[:, idx]
    m = med[:, idx]
    n = neg[:, idx]
    cols = []

    def add(name, values):
        c = prefix + name
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


def build_anterior_45(df):
    n = len(df)
    q20 = np.full((n, 6), np.nan)
    med = np.full((n, 6), np.nan)
    neg = np.full((n, 6), np.nan)

    required = []
    for tooth in ANTERIOR_TEETH:
        b = f"ant__anterior_radial_tooth{tooth}_incisal"
        required += [
            f"{b}_q20_margin_mm",
            f"{b}_median_margin_mm",
            f"{b}_negative_ratio",
        ]
    require_columns(df, required, "anterior radial")

    for j, tooth in enumerate(ANTERIOR_TEETH):
        b = f"ant__anterior_radial_tooth{tooth}_incisal"
        q20[:, j] = pd.to_numeric(df[f"{b}_q20_margin_mm"], errors="coerce")
        med[:, j] = pd.to_numeric(df[f"{b}_median_margin_mm"], errors="coerce")
        neg[:, j] = pd.to_numeric(df[f"{b}_negative_ratio"], errors="coerce")

    out = pd.DataFrame(index=df.index)
    cols = []

    for j, tooth in enumerate(ANTERIOR_TEETH):
        for suffix, arr in (
            ("q20", q20),
            ("median", med),
            ("negative_ratio", neg),
        ):
            c = f"ant97__tooth{tooth}_{suffix}"
            out[c] = arr[:, j]
            cols.append(c)

    idx_map = {t: i for i, t in enumerate(ANTERIOR_TEETH)}
    cols += derive_scope(out, q20, med, neg, ANTERIOR_TEETH, idx_map, "ant97__all_")
    cols += derive_scope(out, q20, med, neg, RIGHT_ANTERIOR, idx_map, "ant97__right_")
    cols += derive_scope(out, q20, med, neg, LEFT_ANTERIOR, idx_map, "ant97__left_")
    cols = safe_numeric(out, cols)
    return out, cols


def make_pipeline(C, class_weight, seed):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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


def safe_splits(y, n_splits, seed):
    counts = pd.Series(y).value_counts()
    k = min(int(n_splits), int(counts.min()))
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(y), 1)), y))


def tune_multiclass(X, y, inner_folds, seed):
    splits = safe_splits(y, inner_folds, seed)
    if splits is None:
        return None

    best = None
    labels = sorted(set(y.tolist()))

    for C in (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        for cw in (None, "balanced"):
            oof = np.empty(len(y), dtype=object)

            for inner_fold, (tr, va) in enumerate(splits):
                model = make_pipeline(C, cw, seed + inner_fold)
                model.fit(X.iloc[tr], y[tr])
                oof[va] = model.predict(X.iloc[va])

            acc = accuracy_score(y, oof)
            macro = f1_score(
                y, oof, labels=labels, average="macro", zero_division=0
            )
            bal = balanced_accuracy_score(y, oof)
            key = (
                float(acc),
                float(macro),
                float(bal),
                -abs(np.log10(C)),
                1 if cw is None else 0,
            )

            if best is None or key > best[0]:
                best = (key, float(C), cw)

    return {"C": best[1], "class_weight": best[2]}


def raw_to_state(field, value, no_na=False):
    if field == "severity":
        if value == "Not_Mentioned":
            return "NM"
        if value == "Not_Assessable":
            return "NM" if no_na else "NA"
        if value in ("Normal", "Mild", "Moderate", "Severe"):
            return "VISUAL"
        return None

    if field == "type":
        if value == "Not_Mentioned":
            return "NM"
        if value == "Not_Assessable":
            return "NM" if no_na else "NA"
        if value in ("Contraction", "Maxillary_Constriction"):
            return "VISUAL"
        return None

    raise ValueError(field)


def state_to_raw(field, state):
    if state == "NM":
        return "Not_Mentioned"
    if state == "NA":
        return "Not_Assessable"
    if state == "VISUAL":
        if field == "severity":
            return "Normal"
        if field == "type":
            return "Maxillary_Constriction"
    raise ValueError((field, state))


def eval_metrics(y, pred):
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main():
    args = parse_args()

    post_path = resolve_posterior(args.posterior_features)
    ant_path = args.anterior_features.expanduser().resolve()
    present_path = args.present_oof.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    post = prefix_dataframe(pd.read_csv(post_path, low_memory=False), "post__")
    ant = prefix_dataframe(pd.read_csv(ant_path, low_memory=False), "ant__")
    present = pd.read_csv(present_path, low_memory=False)
    gt = load_gt(gt_path)

    require_columns(
        present,
        [
            "patient_id",
            "outer_fold",
            "common_eligible",
            "concat_raw_reference_pred",
        ],
        "present OOF",
    )

    gt = gt.loc[gt["gt_source"].eq("photo_en")].copy()

    merged = (
        gt.merge(post, on="patient_id", how="left", validate="one_to_one")
        .merge(ant, on="patient_id", how="left", validate="one_to_one")
        .merge(
            present[
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

    common = merged["common_eligible"].map(bool_value).to_numpy(bool)
    folds = pd.to_numeric(merged["outer_fold"], errors="raise").astype(int).to_numpy()
    present_pred = (
        pd.to_numeric(merged["concat_raw_reference_pred"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )

    post_cols = build_posterior_52(merged)
    ant_df, ant_cols = build_anterior_45(merged)

    X = pd.concat(
        [
            merged[post_cols].reset_index(drop=True),
            ant_df[ant_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    X.index = merged.index

    print("=" * 100)
    print("Final Transverse severity/type OOF v1")
    print("=" * 100)
    print(f"matched rows  : {len(merged)}")
    print(f"strict common : {int(common.sum())}")
    print(f"features      : {X.shape[1]}")
    print("=" * 100)

    comparison_rows = []
    fold_rows = []
    oof = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "outer_fold": folds,
            "common_eligible": common.astype(int),
            "selected_present_pred": present_pred,
            "gt_severity": merged["gt_severity"],
            "gt_transverse_type": merged["gt_transverse_type"],
        }
    )

    for field, gt_col in (
        ("severity", "gt_severity"),
        ("type", "gt_transverse_type"),
    ):
        eligible = common & merged[gt_col].notna().to_numpy(bool)
        eval_idx = np.where(eligible)[0]
        y_raw = merged.loc[eval_idx, gt_col].astype(str).to_numpy(object)

        # Cheap rule from 10.
        if field == "severity":
            cheap = np.where(
                present_pred[eval_idx] >= 0.5,
                "Not_Mentioned",
                "Normal",
            ).astype(object)
        else:
            cheap = np.where(
                present_pred[eval_idx] >= 0.5,
                "Maxillary_Constriction",
                "Not_Mentioned",
            ).astype(object)

        oof.loc[eval_idx, f"{field}__cheap_present_rule_pred"] = cheap
        comparison_rows.append(
            {
                "field": field,
                "model": "cheap_present_rule",
                **eval_metrics(y_raw, cheap),
            }
        )

        for no_na, model_name in (
            (False, "state_logistic_97"),
            (True, "state_logistic_97_no_na"),
        ):
            pred_raw = np.full(len(merged), None, dtype=object)

            for fold in sorted(np.unique(folds)):
                train_idx = np.where(
                    eligible & (folds != fold)
                )[0]
                valid_idx = np.where(
                    eligible & (folds == fold)
                )[0]

                y_state_train = np.asarray(
                    [
                        raw_to_state(field, v, no_na=no_na)
                        for v in merged.loc[train_idx, gt_col].astype(str)
                    ],
                    dtype=object,
                )

                keep_train = np.asarray([v is not None for v in y_state_train])
                tr = train_idx[keep_train]
                ys = y_state_train[keep_train]

                tuned = tune_multiclass(
                    X.loc[tr].reset_index(drop=True),
                    ys,
                    inner_folds=args.inner_folds,
                    seed=args.seed + int(fold) * 100 + (1 if no_na else 0),
                )

                if tuned is None:
                    majority = pd.Series(ys).value_counts().index[0]
                    state_pred = np.full(len(valid_idx), majority, dtype=object)
                    C = np.nan
                    cw = None
                else:
                    model = make_pipeline(
                        tuned["C"],
                        tuned["class_weight"],
                        args.seed + 10000 + int(fold),
                    )
                    model.fit(X.loc[tr], ys)
                    state_pred = model.predict(X.loc[valid_idx]).astype(object)
                    C = tuned["C"]
                    cw = tuned["class_weight"]

                raw_pred = np.asarray(
                    [state_to_raw(field, s) for s in state_pred],
                    dtype=object,
                )
                pred_raw[valid_idx] = raw_pred

                fold_y = merged.loc[valid_idx, gt_col].astype(str).to_numpy(object)
                fold_rows.append(
                    {
                        "field": field,
                        "model": model_name,
                        "fold": int(fold),
                        "C": C,
                        "class_weight": str(cw),
                        **eval_metrics(fold_y, raw_pred),
                    }
                )

            pred_eval = pred_raw[eval_idx]
            oof.loc[eval_idx, f"{field}__{model_name}_pred"] = pred_eval

            comparison_rows.append(
                {
                    "field": field,
                    "model": model_name,
                    **eval_metrics(y_raw, pred_eval),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["field", "accuracy"],
        ascending=[True, False],
    )
    fold_df = pd.DataFrame(fold_rows)

    comparison_path = out_dir / "transverse_final_model_comparison_v1.csv"
    fold_path = out_dir / "transverse_final_fold_metrics_v1.csv"
    oof_path = out_dir / "transverse_final_oof_predictions_v1.csv"
    summary_path = out_dir / "transverse_final_summary_v1.json"

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")

    best = {}
    for field in ("severity", "type"):
        sub = comparison_df.loc[comparison_df["field"].eq(field)]
        best[field] = sub.iloc[0].to_dict()

    summary = {
        "version": "transverse-final-v1-strict-oof",
        "feature_count": int(X.shape[1]),
        "strict_common": int(common.sum()),
        "models": comparison_df.to_dict(orient="records"),
        "best_by_field": best,
        "design_note": {
            "severity_visual_decode": "Normal",
            "type_visual_decode": "Maxillary_Constriction",
            "rare_visual_labels_sacrificed_for_accuracy": True,
        },
        "selection_rule": "Highest exact field accuracy wins.",
        "outputs": {
            "comparison": str(comparison_path),
            "fold_metrics": str(fold_path),
            "oof": str(oof_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Outputs:")
    print(" ", comparison_path)
    print(" ", fold_path)
    print(" ", oof_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
