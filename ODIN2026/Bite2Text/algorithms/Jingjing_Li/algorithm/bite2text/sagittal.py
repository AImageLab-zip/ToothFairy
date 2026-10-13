"""Raw-STL adapter for the frozen sagittal five-fold ensemble."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .case_bundle import CaseBundle


RUNTIME_PATIENT_ID = "generic_case"


def predict_sagittal(
    bundle: CaseBundle,
    *,
    code_root: Path | str | None = None,
    asset_root: Path | str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Render registered arches and invoke the frozen calibrated ensemble."""

    code = Path(code_root or os.environ.get("B2T_SAGITTAL_CODE_ROOT", "/opt/app/runtime/frozen/sagittal")).resolve()
    assets = Path(asset_root or os.environ.get("B2T_SAGITTAL_ASSET_ROOT", "/opt/ml/model/frozen/sagittal")).resolve()
    renderer = code / "pipeline" / "render_relation.py"
    predictor = code / "pipeline" / "05_infer_ensemble_v2.py"
    required = (renderer, predictor, code / "pipeline" / "common.py", assets / "runs", assets / "calibration")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete split sagittal runtime: {missing}")

    with tempfile.TemporaryDirectory(prefix="bite2text_sagittal_") as directory:
        temporary = Path(directory)
        scans = temporary / "scans"
        renders = temporary / "renders"
        output = temporary / "output"
        scans.mkdir()
        shutil.copy2(bundle.upper_stl_path, scans / f"{RUNTIME_PATIENT_ID}_ios_upper.stl")
        shutil.copy2(bundle.lower_stl_path, scans / f"{RUNTIME_PATIENT_ID}_ios_lower.stl")
        environment = os.environ.copy()
        environment.setdefault("PYVISTA_OFF_SCREEN", "true")
        _run(
            [
                sys.executable,
                str(renderer),
                "--stl-dir", str(scans),
                "--output-dir", str(renders),
                "--patients", RUNTIME_PATIENT_ID,
                "--overwrite",
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                str(predictor),
                "--root", str(assets),
                "--render-dir", str(renders),
                "--output-dir", str(output),
                "--patient-id", RUNTIME_PATIENT_ID,
                "--device", device,
            ],
            environment,
        )
        result_path = output / f"{RUNTIME_PATIENT_ID}_sagittal_v3.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Frozen sagittal inference produced no result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        section = result.get("1_sagittal_occlusion")
        if not isinstance(section, dict):
            raise ValueError("Frozen sagittal output lacks 1_sagittal_occlusion")
        return section


def _run(command: list[str], environment: dict[str, str]) -> None:
    try:
        subprocess.run(command, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Frozen sagittal command failed with exit code {error.returncode}: {command}") from error
