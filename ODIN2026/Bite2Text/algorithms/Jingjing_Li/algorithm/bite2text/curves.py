"""Shared CaseBundle adapter for the frozen Curves runtime."""
from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .case_bundle import CaseBundle


@lru_cache(maxsize=1)
def _runtime(root_text: str):
    root = Path(root_text).resolve()
    path = root / "predict_curves.py"
    spec = importlib.util.spec_from_file_location("bite2text_frozen_curves", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predict_curves(
    bundle: CaseBundle,
    *,
    code_root: Path | str | None = None,
    asset_root: Path | str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Predict frozen Curves fields using original DADSN FDI and shared semantic TIFF views."""
    bundle.require_segmentation()
    if bundle.upper_fdi_original is None or bundle.lower_fdi_original is None:
        raise RuntimeError("Curves requires original DADSN FDI, not Crowding-selected segmentation")
    code = Path(code_root or os.environ.get("B2T_CURVES_CODE_ROOT", "/opt/app/runtime/frozen/curves")).resolve()
    assets = Path(asset_root or os.environ.get("B2T_CURVES_ASSET_ROOT", "/opt/ml/model/frozen/curves")).resolve()
    result = _runtime(str(code)).predict(
        bundle.upper_stl_path,
        bundle.lower_stl_path,
        bundle.upper_fdi_original,
        bundle.lower_fdi_original,
        bundle.photos,
        bundle.case_id,
        asset_root=assets,
        device=device,
    )
    result.pop("_debug", None)
    return result
