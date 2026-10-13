"""CaseBundle adapter for the frozen Scheme 3 Vertical runtime.

The Vertical freeze has its own deterministic two-stage DADSN contract. This
adapter intentionally invokes that frozen runtime rather than using shared
CaseBundle labels, whose QC-selection and seed semantics differ.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .case_bundle import CaseBundle


def predict_vertical(
    bundle: CaseBundle,
    *,
    code_root: Path | str | None = None,
    asset_root: Path | str | None = None,
    device: str = "auto",
    include_debug: bool = False,
) -> dict[str, Any]:
    """Predict frozen v3 Scheme 3 from the raw modalities in ``bundle``."""

    code = Path(
        code_root
        or os.environ.get("B2T_VERTICAL_CODE_ROOT", "/opt/app/runtime/frozen/vertical")
    ).resolve()
    assets = Path(
        asset_root
        or os.environ.get("B2T_VERTICAL_ASSET_ROOT", "/opt/ml/model/frozen/vertical")
    ).resolve()
    entrypoint = code / "code" / "inference.py"
    _validate_runtime(code, assets, entrypoint)

    with tempfile.TemporaryDirectory(prefix="bite2text_vertical_") as directory:
        output_path = Path(directory) / "prediction.json"
        command = [
            sys.executable,
            str(entrypoint),
            "--upper-stl",
            str(bundle.upper_stl_path),
            "--lower-stl",
            str(bundle.lower_stl_path),
            "--photo-tiff",
            str(bundle.photo_tiff_path),
            "--device",
            device,
            "--output",
            str(output_path),
        ]
        if include_debug:
            command.append("--debug")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["VERTICAL_PYTHON"] = sys.executable
        environment["VERTICAL_ASSET_ROOT"] = str(assets)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=code / "code",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        seconds = time.perf_counter() - started
        print(f"B2T_VERTICAL_SUBPROCESS seconds={seconds:.6f}", flush=True)
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Frozen Vertical runtime failed with exit code "
                f"{completed.returncode}:\n{completed.stdout[-4000:]}"
            )
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Frozen Vertical runtime produced no result: {output_path}"
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))

    _validate_result(result, asset_root=assets, include_debug=include_debug)
    if not include_debug:
        result.pop("_debug", None)
    return result


def _validate_runtime(code: Path, assets: Path, entrypoint: Path) -> None:
    required = (
        entrypoint,
        code / "code" / "segmentation_runner.py",
        code / "code" / "geometry.py",
        code / "code" / "features.py",
        code / "code" / "task_features.py",
        code / "code" / "dadsn_model.py",
        code / "code" / "dadsn_iops.py",
        code / "code" / "predict1_retry_configurable.py",
        assets / "models" / "dadsn_all_align_best_model.pth",
        assets / "models" / "overjet_extratrees.joblib",
        assets / "models" / "overjet_catboost.cbm",
        assets / "models" / "overbite_extratrees.joblib",
        assets / "models" / "overbite_catboost.cbm",
        assets / "config" / "model_config_v1.json",
        assets / "config" / "ensemble_contract_v1.json",
        assets / "label_c17" / "color_dict.json",
        assets / "label_c17" / "label_decoder_upper.json",
        assets / "label_c17" / "label_decoder_lower.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete split Vertical runtime: {missing}")


def _validate_result(
    result: Any,
    *,
    asset_root: Path,
    include_debug: bool,
) -> None:
    if not isinstance(result, dict):
        raise ValueError("Frozen Vertical output must be an object")
    section = result.get("3_vertical_occlusion")
    if not isinstance(section, dict):
        raise ValueError("Frozen Vertical output lacks 3_vertical_occlusion")
    if set(section) != {"overjet", "overbite"}:
        raise ValueError(f"Unexpected Vertical fields: {sorted(section)}")
    config = json.loads(
        (asset_root / "config" / "model_config_v1.json").read_text(encoding="utf-8")
    )
    for field in ("overjet", "overbite"):
        allowed = set(config["fields"][field]["labels"])
        if section[field] not in allowed:
            raise ValueError(f"Unexpected {field} label: {section[field]!r}")
    if include_debug and not isinstance(result.get("_debug"), dict):
        raise ValueError("Frozen Vertical debug output is missing")
