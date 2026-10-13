"""Landmarks -> the occlusal slot dict the report is rendered from.

Geometry only; no text is read at inference. Per field, the strategy was chosen
by measured accuracy on a hand-labelled dev set:

  gradient-boosted classifier on 25 geometric features
      constr, xbite_present, mr_class, ml_class, cr_class, cl_class
  shallow decision tree on the raw millimetre measurement
      overjet, overbite, wilson
  training-set majority (no model beat its own majority baseline)
      spee, cru, crl, mid, mid_mag, deep

The classifiers are fitted on first use from ``data/occlusal_train_*.csv``:
features computed from landmarks of the 791 training subjects, labels extracted
from their reports. The frozen 198-subject test split is not in those files.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .config import DATA_DIR
from .features import compute_features

FEATURE_COLS = [
    "overjet_mm", "overbite_mm", "molar_ap_right_mm", "molar_ap_left_mm",
    "canine_ap_right_mm", "canine_ap_left_mm", "transverse_right_mm", "transverse_left_mm",
    "transverse_premolar_right_mm", "transverse_premolar_left_mm", "midline_dev_mm",
    "intermolar_width_upper_mm", "intermolar_width_lower_mm",
    "interpremolar_width_upper_mm", "interpremolar_width_lower_mm",
    "little_index_upper_mm", "little_index_lower_mm",
    "arch_crowding_upper_mm", "arch_crowding_lower_mm",
    "curve_of_spee_upper_mm", "curve_of_wilson_upper_mm",
    "curve_of_spee_lower_mm", "curve_of_wilson_lower_mm",
    "n_teeth_upper", "n_teeth_lower",
]

GBM_FIELDS = ["constr", "xbite_present", "mr_class", "ml_class", "cr_class", "cl_class"]
TREE_FIELDS = {
    "overjet": ["overjet_mm"],
    "overbite": ["overbite_mm"],
    "wilson": ["curve_of_wilson_upper_mm", "curve_of_wilson_lower_mm"],
}
MAJORITY_FIELDS = ["spee", "cru", "crl", "mid", "mid_mag", "deep"]
BOOL_FIELDS = {"deep", "xbite_present"}

# Fine hedge words collapsed to the buckets that are actually separable in mm.
COLLAPSE = {
    "normal": "normal", "normal_or_slightly_increased": "normal",
    "normal_or_slightly_reduced": "normal",
    "increased": "increased", "slightly_increased": "increased",
    "markedly_increased": "markedly_increased", "accentuated": "markedly_increased",
    "reduced": "reduced", "slightly_reduced": "reduced",
    "markedly_reduced": "markedly_reduced",
    "reverse": "reverse", "reversed": "reverse", "zero": "reverse",
}

_MODELS: dict = {}
_MAJORITY: dict = {}


def _train() -> None:
    if _MODELS:
        return
    # Once torch has initialised its thread pool on every core, an unbounded
    # OpenMP fit here oversubscribes the CPU and turns seconds into many minutes.
    with threadpool_limits(limits=4):
        _fit()


def _fit() -> None:
    table = pd.concat([pd.read_csv(DATA_DIR / "occlusal_train_fit.csv"),
                       pd.read_csv(DATA_DIR / "occlusal_train_dev.csv")], ignore_index=True)
    features = table[FEATURE_COLS]

    for field in GBM_FIELDS:
        mask = table[field].notna()
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0)
        clf.fit(features[mask], table.loc[mask, field].astype(str))
        _MODELS[field] = clf

    for field, cols in TREE_FIELDS.items():
        mask = table[field].notna() & table[cols].notna().all(axis=1)
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0)
        clf.fit(table.loc[mask, cols], table.loc[mask, field].astype(str).map(COLLAPSE))
        _MODELS[field] = clf

    for field in MAJORITY_FIELDS:
        values = table[field].dropna().astype(str)
        if field == "spee":
            values = values.map(COLLAPSE)
        majority = values.value_counts().idxmax()
        _MAJORITY[field] = (majority == "True") if field in BOOL_FIELDS else majority


def predict_slots(landmarks: dict) -> Optional[dict]:
    """``{fdi_str: {name: [x, y, z]}}`` for both jaws -> slot dict.

    Returns ``None`` when too few teeth were found to build an anatomical frame.
    """
    _train()
    feats = compute_features(landmarks)
    if feats is None:
        return None
    row = pd.DataFrame([feats]).reindex(columns=FEATURE_COLS)

    slots: dict = {}
    for field in GBM_FIELDS:
        pred = _MODELS[field].predict(row[FEATURE_COLS])[0]
        slots[field] = (pred == "True") if field in BOOL_FIELDS else pred

    for field, cols in TREE_FIELDS.items():
        sub = row[cols]
        if sub.notna().all(axis=1).iloc[0]:
            slots[field] = _MODELS[field].predict(sub)[0]
        else:
            slots[field] = _MAJORITY.get(field, "normal")

    for field in MAJORITY_FIELDS:
        slots[field] = _MAJORITY[field]

    return slots
