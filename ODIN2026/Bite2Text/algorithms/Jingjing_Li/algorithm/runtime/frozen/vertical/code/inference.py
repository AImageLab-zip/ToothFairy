#!/usr/bin/env python3
"""Frozen Scheme 3 inference from two registered STL scans."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np
import trimesh
from catboost import CatBoostClassifier

from features import geometry_features
from geometry import measure_arrays
from segmentation_runner import FrozenTwoPassSegmenter
from task_features import overbite_geometry_features, segmentation_features

CODE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ["VERTICAL_ASSET_ROOT"]).expanduser().resolve()
MODELS_DIR = ASSET_ROOT / "models"
CONFIG_PATH = ASSET_ROOT / "config" / "model_config_v1.json"
SEED = 20260812


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda:0"
    if name == "cpu" or name.startswith("cuda"):
        return name
    raise ValueError(f"Unsupported device: {name}")


def load_segmented_arch(payload: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(payload["raw_ply"], force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 100:
        raise ValueError(f"Invalid segmented mesh: {payload['raw_ply']}")
    points = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(
        json.loads(Path(payload["labels_json"]).read_text(encoding="utf-8")),
        dtype=np.int32,
    ).reshape(-1)
    if len(points) != len(labels):
        raise RuntimeError(
            f"Segmented mesh/label mismatch: {len(points)} != {len(labels)}"
        )
    return points, labels


class VerticalOcclusionPredictor:
    def __init__(self, device: str = "auto", seed: int = SEED):
        self.device = resolve_device(device)
        load_started = time.perf_counter()
        self.seed = int(seed)
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.ensemble = json.loads((ASSET_ROOT / "config" / "ensemble_contract_v1.json").read_text(encoding="utf-8"))
        self.overjet_model = joblib.load(MODELS_DIR / "overjet_extratrees.joblib")
        self.overbite_model = joblib.load(MODELS_DIR / "overbite_extratrees.joblib")
        self.overjet_catboost = CatBoostClassifier().load_model(MODELS_DIR / "overjet_catboost.cbm")
        self.overbite_catboost = CatBoostClassifier().load_model(MODELS_DIR / "overbite_catboost.cbm")
        print(
            f"B2T_MODEL_LOAD vertical count=4 seconds={time.perf_counter() - load_started:.6f}",
            flush=True,
        )
        self.segmenter = FrozenTwoPassSegmenter(CODE_ROOT, ASSET_ROOT, device=self.device)

    def _ensemble_predict(
        self,
        task: str,
        extra_model: Any,
        catboost_model: Any,
        extra_features: np.ndarray,
        catboost_features: np.ndarray,
    ) -> str:
        contract = self.ensemble[task]
        labels = list(contract["labels"])
        probability = np.zeros(len(labels), dtype=np.float64)
        extra_values = np.nan_to_num(extra_model.predict_proba(extra_features)[0], nan=0.0)
        for source, label in enumerate(extra_model.classes_):
            probability[labels.index(str(label))] += float(contract["extra_weight"]) * extra_values[source]
        cat_values = np.nan_to_num(catboost_model.predict_proba(catboost_features)[0], nan=0.0)
        for source, label in enumerate(catboost_model.classes_):
            probability[labels.index(str(label))] += float(contract["catboost_weight"]) * cat_values[source]
        return str(labels[int(np.argmax(probability))])

    def predict(
        self,
        upper_stl: str | Path,
        lower_stl: str | Path,
        photo_tiff: str | Path | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Predict v3 Scheme 3; TIFF is accepted for the shared API but unused."""
        if photo_tiff is not None:
            photo_path = Path(photo_tiff).expanduser().resolve()
            if not photo_path.is_file():
                raise FileNotFoundError(photo_path)

        failure: str | None = None
        geometry: dict[str, Any] | None = None
        segmentation_debug: dict[str, Any] = {}
        segmented: dict[str, Any] | None = None
        try:
            segmented = self.segmenter.segment_pair(upper_stl, lower_stl)
            upper_points, upper_labels = load_segmented_arch(segmented["upper"])
            lower_points, lower_labels = load_segmented_arch(segmented["lower"])
            segmentation_debug = segmented["debug"]
            geometry, _ = measure_arrays(
                case_id="raw_case",
                upper_points=upper_points,
                upper_labels=upper_labels,
                lower_points=lower_points,
                lower_labels=lower_labels,
                source={
                    "upper_mesh": str(Path(upper_stl).expanduser().resolve()),
                    "lower_mesh": str(Path(lower_stl).expanduser().resolve()),
                    "segmentation": (
                        "frozen deterministic two-pass DADSN C17 labels mapped "
                        "to original registered STL vertices"
                    ),
                    "registered_common_coordinates_assumed": True,
                },
            )
            rich = geometry_features(geometry).reshape(1, -1)
            internal_upper = np.load(segmented["upper"]["internal_npy"], allow_pickle=False)
            internal_lower = np.load(segmented["lower"]["internal_npy"], allow_pickle=False)
            seg = segmentation_features(internal_upper, internal_lower).reshape(1, -1)
            overjet = self._ensemble_predict(
                "overjet",
                self.overjet_model,
                self.overjet_catboost,
                rich,
                np.concatenate([rich, seg], axis=1),
            )
            overbite = self._ensemble_predict(
                "overbite",
                self.overbite_model,
                self.overbite_catboost,
                rich,
                np.concatenate([overbite_geometry_features(geometry).reshape(1, -1), seg], axis=1),
            )
        except RuntimeError as error:
            if "Grand Challenge" in str(error) and "CUDA" in str(error):
                raise
            failure = f"{type(error).__name__}: {error}"
            overjet = str(self.config["fallback"]["overjet"])
            overbite = str(self.config["fallback"]["overbite"])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            overjet = str(self.config["fallback"]["overjet"])
            overbite = str(self.config["fallback"]["overbite"])
        finally:
            if segmented is not None and segmented.get("_temporary") is not None:
                segmented["_temporary"].cleanup()

        result: dict[str, Any] = {
            "3_vertical_occlusion": {
                "overjet": overjet,
                "overbite": overbite,
            }
        }
        if include_debug:
            result["_debug"] = {
                "device": self.device,
                "photo_tiff_used": False,
                "segmentation": segmentation_debug,
                "geometry": geometry,
                "fallback_reason": failure,
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-stl", type=Path, required=True)
    parser.add_argument("--lower-stl", type=Path, required=True)
    parser.add_argument("--photo-tiff", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = VerticalOcclusionPredictor(device=args.device)
    result = predictor.predict(
        args.upper_stl,
        args.lower_stl,
        photo_tiff=args.photo_tiff,
        include_debug=args.debug,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
