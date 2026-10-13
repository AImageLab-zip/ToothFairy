"""Adapter for the frozen Scheme 5--6 predictor using a shared CaseBundle."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
import time
from pathlib import Path
from typing import Any

import numpy as np

from .case_bundle import CaseBundle


@lru_cache(maxsize=2)
def _load_predictor(code_text: str, asset_text: str, device: str) -> tuple[Any, Any]:
    code = Path(code_text).resolve()
    assets = Path(asset_text).resolve()
    code_dir = code / "code"
    entrypoint = code_dir / "inference.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Split Dental/Hygiene entrypoint is missing: {entrypoint}")
    sys.path.insert(0, str(code_dir))
    try:
        spec = importlib.util.spec_from_file_location("bite2text_frozen_dental_hygiene", entrypoint)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load frozen Dental/Hygiene entrypoint: {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        load_started = time.perf_counter()
        predictor = module.DentalHygienePredictor(
            device=device,
            asset_root=assets,
            load_segmenter=False,
        )
        print(
            f"B2T_MODEL_LOAD dental_hygiene count=4 seconds={time.perf_counter() - load_started:.6f}",
            flush=True,
        )
        return module, predictor
    finally:
        sys.path.remove(str(code_dir))


def predict_dental_hygiene(
    bundle: CaseBundle,
    *,
    code_root: Path | str | None = None,
    asset_root: Path | str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Return fields 5--6 without rerunning DADSN or decoding TIFF again."""

    bundle.require_segmentation()
    code = Path(code_root or os.environ.get("B2T_DENTAL_HYGIENE_CODE_ROOT", "/opt/app/runtime/frozen/dental_hygiene")).resolve()
    assets = Path(asset_root or os.environ.get("B2T_DENTAL_HYGIENE_ASSET_ROOT", "/opt/ml/model/frozen/dental_hygiene")).resolve()
    module, predictor = _load_predictor(str(code), str(assets), device)
    upper_counts = np.bincount(bundle.upper_c17, minlength=17)[1:17].astype(np.int64)
    lower_counts = np.bincount(bundle.lower_c17, minlength=17)[1:17].astype(np.int64)
    missing_features = module.segmentation_features(upper_counts, lower_counts)
    frozen_views = {
        "lower": bundle.photos["lower_occlusal"],
        "upper": bundle.photos["upper_occlusal"],
    }
    photo_features = predictor._photo_features(frozen_views)
    missing_teeth = module.normalize_fdi_set(predictor.missing_model.predict(missing_features)[0])
    restorations = module.normalize_fdi_set(predictor.restoration_model.predict(photo_features)[0])
    probability = float(predictor.pigmentation_model.predict_proba(photo_features)[0, 1])
    threshold = float(predictor.config["fields"]["pigmentation"]["probability_threshold"])
    return {
        "5_dental_status": {
            "missing_teeth": missing_teeth,
            "restorations": restorations,
            "caries": [],
        },
        "6_oral_hygiene": {
            "hygiene_status": "Not_Mentioned",
            "pigmentation": True if probability >= threshold else None,
        },
    }
