"""Raw-only adapter for the frozen existing Crowding routing pipeline."""

from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .case_bundle import CaseBundle


CLASS_NAMES = ("Spacing", "None", "Mild", "Moderate", "Severe")


def predict_crowding(
    bundle: CaseBundle,
    *,
    runtime_root: Path | str | None = None,
    dadsn_root: Path | str | None = None,
    dadsn_code_root: Path | str | None = None,
    device: str = "cuda",
) -> dict[str, str]:
    """Run the existing fixed route without training, reselection, or retuning."""

    bundle.require_segmentation()
    if bundle.upper_crowding_mask is None or bundle.lower_crowding_mask is None:
        raise RuntimeError("CaseBundle lacks selected compat Crowding masks")
    if not bundle.crowding_qc_summary or not bundle.crowding_qc_tooth_metrics:
        raise RuntimeError("CaseBundle lacks selected segmentation QC required by Crowding routing")

    frozen = Path(runtime_root or os.environ.get("B2T_CROWDING_ASSET_ROOT", "/opt/ml/model/frozen/crowding")).resolve()
    dad_assets = Path(dadsn_root or os.environ.get("DADSN_ASSET_ROOT", "/opt/ml/model/dadsn")).resolve()
    dad_code = Path(dadsn_code_root or os.environ.get("DADSN_CODE_ROOT", "/opt/app/runtime/dadsn")).resolve()
    _validate_runtime(frozen, dad_code, dad_assets)

    with tempfile.TemporaryDirectory(prefix="bite2text_crowding_") as temporary:
        root = Path(temporary)
        case_results = root / "case_results" / bundle.case_id
        masks = root / "masks"
        qc_summary = root / "selected_qc.csv"
        qc_teeth = root / "selected_tooth_metrics.csv"
        case_results.mkdir(parents=True)
        masks.mkdir()
        _write_csv(qc_summary, bundle.crowding_qc_summary)
        _write_csv(qc_teeth, bundle.crowding_qc_tooth_metrics)
        _write_geometry(bundle, case_results, dad_code)
        _write_mask(masks / f"{bundle.case_id}_ios_upper_label.png", bundle.upper_crowding_mask)
        _write_mask(masks / f"{bundle.case_id}_ios_lower_label.png", bundle.lower_crowding_mask)

        route_dir = root / "route"
        _run([
            sys.executable, str(dad_code / "geometry_router.py"), "route",
            "--profile", str(frozen / "router" / "geometry_profile.json"),
            "--selected-qc", str(qc_summary),
            "--case-results-dir", str(root / "case_results"),
            "--output-dir", str(route_dir),
        ])
        routes = _read_by_arch(route_dir / "routing_table.csv")
        predictions: dict[str, str] = {}
        arches_by_route: dict[str, list[str]] = {}
        for arch in ("upper", "lower"):
            arches_by_route.setdefault(routes[arch]["route"], []).append(arch)
        for route, arches in arches_by_route.items():
            if route == "full_catboost":
                route_predictions = _predict_primary(root, frozen, dad_code, arches)
            elif route == "segmentation_fallback":
                route_predictions = _predict_segmentation_fallback(
                    root, frozen, dad_code, qc_summary, qc_teeth, arches
                )
            elif route == "visual_fallback":
                route_predictions = _predict_visual_fallback(
                    bundle, root, frozen, dad_code, qc_summary, qc_teeth, arches, device
                )
            else:
                raise RuntimeError(f"Unexpected frozen Crowding route: {route}")
            for arch in arches:
                predictions[arch] = route_predictions[arch]
        return {"upper_crowding": predictions["upper"], "lower_crowding": predictions["lower"]}


def _validate_runtime(frozen: Path, dad_code: Path, dad_assets: Path) -> None:
    required = (
        frozen / "models" / "primary" / "upper_model.cbm",
        frozen / "models" / "primary" / "lower_model.cbm",
        frozen / "models" / "segmentation_fallback" / "upper_model.cbm",
        frozen / "models" / "segmentation_fallback" / "lower_model.cbm",
        frozen / "router" / "geometry_profile.json",
        frozen / "config.json",
        dad_assets / "outputs" / "All_align" / "models" / "best_model.pth",
        dad_code / "crowding_catboost_v2.py",
        dad_code / "segmentation_fallback_catboost.py",
        dad_code / "geometry_router.py",
        dad_code / "render_stl_multiview.py",
        dad_code / "visual_fallback_model.py",
        dad_code / "midline" / "crowding.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete split Crowding runtime: {missing}")


def _write_geometry(bundle: CaseBundle, output: Path, dad: Path) -> None:
    midline_root = str(dad / "midline")
    if midline_root not in sys.path:
        sys.path.insert(0, midline_root)
    module = importlib.import_module("crowding")
    with tempfile.TemporaryDirectory(prefix="bite2text_crowding_labels_") as directory:
        labels_root = Path(directory)
        for arch, mesh_path, labels in (
            ("upper", bundle.upper_stl_path, bundle.upper_fdi),
            ("lower", bundle.lower_stl_path, bundle.lower_fdi),
        ):
            assert labels is not None
            labels_path = labels_root / f"{arch}_fdi.json"
            labels_path.write_text(json.dumps(labels.tolist()), encoding="utf-8")
            try:
                geometry = module.calculate_crowding(mesh_path, labels_path, arch)
            except ValueError as error:
                print(
                    f"B2T_CROWDING_GEOMETRY_UNAVAILABLE arch={arch} error={error}",
                    flush=True,
                )
                continue
            (output / f"{arch}_crowding.json").write_text(
                json.dumps(geometry, ensure_ascii=False), encoding="utf-8"
            )


def _write_mask(path: Path, values: np.ndarray) -> None:
    image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L")
    image.save(path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty Crowding route input: {path.name}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        subprocess.run(command, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Frozen Crowding command failed with exit code {error.returncode}: {command}") from error


def _read_by_arch(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {str(row["arch"]).lower(): row for row in csv.DictReader(handle)}
    if set(rows) != {"upper", "lower"}:
        raise RuntimeError(f"Crowding route did not produce both arches: {rows.keys()}")
    return rows


def _predict_primary(root: Path, frozen: Path, dad: Path, arches: list[str]) -> dict[str, str]:
    output = root / "primary_predictions.csv"
    _run([
        sys.executable, str(dad / "crowding_catboost_v2.py"), "predict",
        "--case-results-dir", str(root / "case_results"), "--mask-dir", str(root / "masks"),
        "--model-dir", str(frozen / "models" / "primary"), "--output-csv", str(output),
    ])
    return {arch: _prediction_for_arch(output, arch) for arch in arches}


def _predict_segmentation_fallback(
    root: Path,
    frozen: Path,
    dad: Path,
    qc_summary: Path,
    qc_teeth: Path,
    arches: list[str],
) -> dict[str, str]:
    output = root / "segmentation_predictions.csv"
    _run([
        sys.executable, str(dad / "segmentation_fallback_catboost.py"), "predict",
        "--selected-qc", str(qc_summary), "--selected-tooth-metrics", str(qc_teeth),
        "--model-dir", str(frozen / "models" / "segmentation_fallback"), "--output-csv", str(output),
    ])
    return {arch: _prediction_for_arch(output, arch) for arch in arches}


def _predict_visual_fallback(
    bundle: CaseBundle,
    root: Path,
    frozen: Path,
    dad: Path,
    qc_summary: Path,
    qc_teeth: Path,
    arches: list[str],
    device: str,
) -> dict[str, str]:
    segmentation_output = root / "visual_route_segmentation_predictions.csv"
    _run([
        sys.executable, str(dad / "segmentation_fallback_catboost.py"), "predict",
        "--selected-qc", str(qc_summary), "--selected-tooth-metrics", str(qc_teeth),
        "--model-dir", str(frozen / "models" / "segmentation_fallback"), "--output-csv", str(segmentation_output),
    ])
    segmentation_probabilities = {
        arch: np.asarray(_probabilities_for_arch(segmentation_output, arch), dtype=np.float64)
        for arch in arches
    }
    scans = root / "visual_input"
    renders = root / "visual_renders"
    scans.mkdir()
    sources = {"upper": bundle.upper_stl_path, "lower": bundle.lower_stl_path}
    for arch in arches:
        shutil.copy2(sources[arch], scans / f"{bundle.case_id}_ios_{arch}.stl")
    _run([
        sys.executable, str(dad / "render_stl_multiview.py"), "--input-dir", str(scans),
        "--output-dir", str(renders), "--overwrite",
    ])
    fold_probabilities: dict[str, list[list[float]]] = {arch: [] for arch in arches}
    for fold in range(5):
        output = root / f"visual_fold_{fold}.csv"
        _run([
            sys.executable, str(dad / "visual_fallback_model.py"), "predict",
            "--tensor-dir", str(renders), "--model-dir", str(frozen / "models" / "visual_fallback" / f"fold_{fold}"),
            "--output-csv", str(output), "--device", device,
        ])
        for arch in arches:
            fold_probabilities[arch].append(_probabilities_for_arch(output, arch))
    config = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    predictions: dict[str, str] = {}
    for arch in arches:
        visual_probabilities = np.mean(
            np.asarray(fold_probabilities[arch], dtype=np.float64), axis=0
        )
        alpha = float(config["visual_fallback"]["alpha_visual_by_arch"][arch])
        blended = (
            (1.0 - alpha) * segmentation_probabilities[arch]
            + alpha * visual_probabilities
        )
        predictions[arch] = CLASS_NAMES[int(np.argmax(blended))]
    return predictions


def _prediction_for_arch(path: Path, arch: str) -> str:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["arch"]).lower() == arch:
                prediction = str(row["prediction"])
                if prediction not in CLASS_NAMES:
                    raise ValueError(f"Unexpected Crowding class: {prediction}")
                return prediction
    raise RuntimeError(f"No {arch} Crowding prediction in {path}")


def _probabilities_for_arch(path: Path, arch: str) -> list[float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["arch"]).lower() == arch:
                return [float(row[f"prob_{name.lower()}"]) for name in CLASS_NAMES]
    raise RuntimeError(f"No {arch} visual Crowding prediction in {path}")
