"""Raw-only adapter for the selected clinical-frontal midline algorithm."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_bundle import CaseBundle


MIDLINE_THRESHOLD_MM = 0.5


@dataclass(frozen=True)
class MidlinePrediction:
    label: str
    method: str
    absolute_lateral_offset_mm: float | None
    diagnostics: dict[str, Any]


def predict_midline(bundle: CaseBundle, runtime_root: Path | str | None = None) -> MidlinePrediction:
    """Predict the sole v3 ``midline`` field from live mesh/FDI inputs.

    The clinical-frontal method is the selected primary method. The geometric
    method is used only when the clinical computation cannot produce an offset.
    """

    bundle.require_segmentation()
    root = Path(runtime_root or os.environ.get("B2T_MIDLINE_CODE_ROOT", "/opt/app/runtime/dadsn/midline")).resolve()
    clinical_midline, geometric_midline = _load_runtime(root)

    with tempfile.TemporaryDirectory(prefix="bite2text_midline_") as directory:
        directory_path = Path(directory)
        upper_labels = directory_path / "upper_fdi.json"
        lower_labels = directory_path / "lower_fdi.json"
        upper_labels.write_text(json.dumps(bundle.upper_fdi.tolist()), encoding="utf-8")
        lower_labels.write_text(json.dumps(bundle.lower_fdi.tolist()), encoding="utf-8")

        failures: dict[str, str] = {}
        for method, predictor in (
            ("clinical_frontal", clinical_midline.calculate_clinical_midlines),
            ("geometric_3d", geometric_midline.calculate_midlines),
        ):
            try:
                result, _ = predictor(
                    bundle.upper_stl_path,
                    upper_labels,
                    bundle.lower_stl_path,
                    lower_labels,
                )
                offset = float(result["comparison"]["absolute_lateral_offset_mm"])
                label = "Deviated" if offset > MIDLINE_THRESHOLD_MM else "Aligned"
                return MidlinePrediction(
                    label=label,
                    method=method,
                    absolute_lateral_offset_mm=offset,
                    diagnostics={"result": result, "failures": failures},
                )
            except Exception as error:
                failures[method] = f"{type(error).__name__}: {error}"

    return MidlinePrediction(
        label="Not_Assessable",
        method="unavailable",
        absolute_lateral_offset_mm=None,
        diagnostics={"failures": failures},
    )


def _load_runtime(root: Path) -> tuple[Any, Any]:
    required = (root / "crowding.py", root / "midline.py", root / "clinical_midline.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete bundled midline runtime: {missing}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    clinical = importlib.import_module("clinical_midline")
    geometric = importlib.import_module("midline")
    if Path(clinical.__file__).resolve().parent != root or Path(geometric.__file__).resolve().parent != root:
        raise RuntimeError("Conflicting midline runtime module is already loaded")
    return clinical, geometric
