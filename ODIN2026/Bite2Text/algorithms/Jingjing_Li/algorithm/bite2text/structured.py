"""Structured v3 aggregation over frozen Bite2Text field predictors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from .case_bundle import CaseBundle
from .crowding import predict_crowding
from .curves import predict_curves
from .dental_hygiene import predict_dental_hygiene
from .midline import predict_midline
from .sagittal import predict_sagittal
from .transverse import predict_transverse
from .vertical import predict_vertical


SCHEMA_TEMPLATE: dict[str, dict[str, Any]] = {
    "1_sagittal_occlusion": {
        "right_molar": "Not_Assessable",
        "right_canine": "Not_Assessable",
        "left_molar": "Not_Assessable",
        "left_canine": "Not_Assessable",
    },
    "2_transverse_occlusion": {
        "severity": "Not_Assessable",
        "type": "Not_Assessable",
        "crossbite_present": None,
        "crossbite_type": "Not_Assessable",
        "crossbite_location": "Not_Assessable",
        "crossbite_teeth": [],
    },
    "3_vertical_occlusion": {
        "overjet": "Not_Assessable",
        "overbite": "Not_Assessable",
    },
    "4_alignment_and_arch": {
        "midline": "Not_Assessable",
        "upper_crowding": "Not_Assessable",
        "lower_crowding": "Not_Assessable",
        "curve_spee": "Not_Assessable",
        "curve_wilson": "Not_Assessable",
        "curve_wilson_location": "Not_Assessable",
    },
    "5_dental_status": {
        "missing_teeth": [],
        "restorations": [],
        "caries": [],
    },
    "6_oral_hygiene": {
        "hygiene_status": "Not_Assessable",
        "pigmentation": None,
    },
}

FIELD_COUNT = sum(len(section) for section in SCHEMA_TEMPLATE.values())


def predict_structured(
    bundle: CaseBundle,
    *,
    runtime_roots: dict[str, Path | str] | None = None,
    device: str = "auto",
    include_debug: bool = False,
) -> dict[str, Any]:
    """Return the complete 23-field v3 structure from one raw ``CaseBundle``.

    Predictors own their explicit frozen fallbacks. A module failure is not
    silently converted to a schema fallback because that would hide a broken
    submission bundle.
    """

    roots = runtime_roots or {}
    field_device = _resolve_device(device)
    report = deepcopy(SCHEMA_TEMPLATE)
    total_started = time.perf_counter()

    def run_stage(name: str, function):
        elapsed = time.perf_counter() - total_started
        print(f"B2T_STAGE_START {name} elapsed_total={elapsed:.6f}", flush=True)
        started = time.perf_counter()
        value = function()
        seconds = time.perf_counter() - started
        print(
            f"B2T_STAGE_DONE {name} seconds={seconds:.6f} "
            f"elapsed_total={time.perf_counter() - total_started:.6f}",
            flush=True,
        )
        return value

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bite2text-vertical") as executor:
        vertical_future = executor.submit(
            run_stage,
            "vertical",
            lambda: predict_vertical(
                bundle,
                code_root=roots.get("vertical_code"),
                asset_root=roots.get("vertical"),
                device=field_device,
                include_debug=include_debug,
            ),
        )

        _update_fields(
            report,
            "1_sagittal_occlusion",
            run_stage("sagittal", lambda: predict_sagittal(
                bundle,
                code_root=roots.get("sagittal_code"),
                asset_root=roots.get("sagittal"),
                device=field_device,
            )),
        )
        _update_fields(
            report,
            "2_transverse_occlusion",
            run_stage("transverse", lambda: predict_transverse(
                bundle,
                code_root=roots.get("transverse_code"),
                asset_root=roots.get("transverse"),
            )),
        )

        midline = run_stage("midline", lambda: predict_midline(bundle, runtime_root=roots.get("midline")))
        _update_fields(report, "4_alignment_and_arch", {"midline": midline.label})
        _update_fields(
            report,
            "4_alignment_and_arch",
            run_stage("crowding", lambda: predict_crowding(
                bundle,
                runtime_root=roots.get("crowding"),
                dadsn_root=roots.get("dadsn"),
                dadsn_code_root=roots.get("dadsn_code"),
                device=field_device,
            )),
        )
        _update_fields(
            report,
            "4_alignment_and_arch",
            run_stage("curves", lambda: predict_curves(
                bundle,
                code_root=roots.get("curves_code"),
                asset_root=roots.get("curves"),
                device=field_device,
            )),
        )
        _merge(
            report,
            run_stage("dental_hygiene", lambda: predict_dental_hygiene(
                bundle,
                code_root=roots.get("dental_hygiene_code"),
                asset_root=roots.get("dental_hygiene"),
                device=field_device,
            )),
        )
        _merge(report, vertical_future.result())

    print(f"B2T_ELAPSED_TOTAL seconds={time.perf_counter() - total_started:.6f}", flush=True)

    _validate_shape(report)
    if include_debug:
        report["_debug"] = {"field_count": FIELD_COUNT}
    return report


def _merge(report: dict[str, dict[str, Any]], prediction: dict[str, Any]) -> None:
    for section, fields in prediction.items():
        if section == "_debug":
            continue
        if section not in report or not isinstance(fields, dict):
            raise ValueError(f"Unexpected structured predictor section: {section!r}")
        unexpected = set(fields) - set(report[section])
        if unexpected:
            raise ValueError(f"Unexpected fields in {section}: {sorted(unexpected)}")
        report[section].update(fields)


def _update_fields(
    report: dict[str, dict[str, Any]],
    section: str,
    fields: dict[str, Any],
) -> None:
    if section not in report or not isinstance(fields, dict):
        raise ValueError(f"Invalid structured predictor output for {section}")
    unknown = set(fields) - set(report[section])
    if unknown:
        raise ValueError(f"Unexpected fields in {section}: {sorted(unknown)}")
    report[section].update(fields)


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cuda", "cuda:0"}:
        raise RuntimeError(f"Production structured runtime must use cuda:0, got {device!r}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Grand Challenge structured runtime requires CUDA; CPU fallback is disabled")
    return "cuda:0"


def _validate_shape(report: dict[str, Any]) -> None:
    if set(report) - {"_debug"} != set(SCHEMA_TEMPLATE):
        raise ValueError("Structured report sections do not match the v3 contract")
    count = 0
    for section, expected in SCHEMA_TEMPLATE.items():
        actual = report.get(section)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Structured report fields do not match {section}")
        count += len(actual)
    if count != FIELD_COUNT or FIELD_COUNT != 23:
        raise RuntimeError(f"Unexpected structured field count: {count}")
