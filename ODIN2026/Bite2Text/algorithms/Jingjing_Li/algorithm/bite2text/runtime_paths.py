"""Container code roots and Grand Challenge model-asset roots."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("B2T_MODEL_ROOT", "/opt/ml/model")).expanduser().resolve()
RUNTIME_ROOT = Path(os.environ.get("B2T_RUNTIME_ROOT", "/opt/app/runtime")).expanduser().resolve()

REPORT_ASSET_ROOT = MODEL_ROOT / "report_runtime"
FROZEN_ASSET_ROOT = MODEL_ROOT / "frozen"
DADSN_ASSET_ROOT = MODEL_ROOT / "dadsn"
FROZEN_CODE_ROOT = RUNTIME_ROOT / "frozen"
DADSN_CODE_ROOT = RUNTIME_ROOT / "dadsn"


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"Missing {description}: {path}")
    return path
