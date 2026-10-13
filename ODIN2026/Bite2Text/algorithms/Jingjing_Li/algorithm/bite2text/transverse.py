"""CaseBundle adapter for the frozen Transverse single-case predictor."""
from __future__ import annotations
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from .case_bundle import CaseBundle

@lru_cache(maxsize=1)
def _module(root: str):
    path = Path(root) / 'predict_transverse.py'
    spec = importlib.util.spec_from_file_location('bite2text_frozen_transverse', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def predict_transverse(
    bundle: CaseBundle,
    *,
    code_root: Path | str | None = None,
    asset_root: Path | str | None = None,
) -> dict[str, Any]:
    bundle.require_segmentation()
    code = Path(code_root or os.environ.get('B2T_TRANSVERSE_CODE_ROOT', '/opt/app/runtime/frozen/transverse')).resolve()
    assets = Path(asset_root or os.environ.get('B2T_TRANSVERSE_ASSET_ROOT', '/opt/ml/model/frozen/transverse')).resolve()
    upper_fdi = bundle.upper_fdi_original if bundle.upper_fdi_original is not None else bundle.upper_fdi
    lower_fdi = bundle.lower_fdi_original if bundle.lower_fdi_original is not None else bundle.lower_fdi
    result = _module(str(code)).predict(
        bundle.upper_stl_path,
        bundle.lower_stl_path,
        upper_fdi,
        lower_fdi,
        bundle.case_id,
        asset_root=assets,
    )
    result.pop('_debug', None)
    return result
