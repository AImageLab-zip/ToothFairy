from pathlib import Path
from collections import Counter, defaultdict
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import trimesh
import torch.nn as nn
import torch.nn.functional as F
import yaml
from IPython.display import display
from PIL import Image
from torchvision.models import (
    convnext_tiny,
    efficientnet_b0,
    efficientnet_b3,
    efficientnet_v2_s,
    mobilenet_v3_small,
)
from torchvision.transforms import v2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from ultralytics import YOLO
from ultralytics.utils import ops as yolo_ops


Image.MAX_IMAGE_PIXELS = None

# Change only CASE_NUMBER. Every execution treats the selected case files as new input.
CASE_NUMBER = "F0000"
BITE2TEXT_ROOT = Path("/data/MICCAI/Bite2Text")
AREA_REFERENCE_PRED_ROOT = BITE2TEXT_ROOT / "bite2text_pointcept/landmarks/predictions"
DATASET_ROOT = BITE2TEXT_ROOT / "dataset"
SUBMISSION_ROOT = BITE2TEXT_ROOT / "submission"
MODELS_ROOT = SUBMISSION_ROOT / "models"
RUNTIME_ROOT = SUBMISSION_ROOT / "runtime"
CONFIG_ROOT = SUBMISSION_ROOT / "config"

CASE_ID = str(CASE_NUMBER).upper()
CASE_ID = CASE_ID if CASE_ID.startswith("F") else f"F{int(CASE_ID)}"
CASE_ROOT = DATASET_ROOT / CASE_ID
INPUT_DIR = CASE_ROOT / "intraoral-photo"
UPPER_STL = CASE_ROOT / "ios" / "ios_upper.stl"
LOWER_STL = CASE_ROOT / "ios" / "ios_lower.stl"

VIEW_WEIGHT = MODELS_ROOT / "view/view_efficientnet_b3_5class.pt"
UPPER_CROWDING_WEIGHT = MODELS_ROOT / "crowding/upper_efficientnet_v2_s.pt"
LOWER_CROWDING_WEIGHT = MODELS_ROOT / "crowding/lower_efficientnet_v2_s.pt"
TOOTHSEG_CHECKPOINT = (
    BITE2TEXT_ROOT
    / "complexity_aihub_dental/yolo11m_seg_fdi24_20260816/outputs/"
    "yolo11m_seg_fdi24_seed42/weights/best.pt"
)
TEETHLAND_INSTSEG_WEIGHT = MODELS_ROOT / "3dteethland/instseg_full.ckpt"
TEETHLAND_LANDMARK_WEIGHT = MODELS_ROOT / "3dteethland/landmarks_full.ckpt"
CANINE_CALIBRATION_PATH = (
    BITE2TEXT_ROOT / "experiments/canine_occlusion_bite2text_calibration/calibration_result.json"
)
CANINE_ERUPTION_GATE_PATH = (
    BITE2TEXT_ROOT / "experiments/canine_occlusion_bite2text_calibration/eruption_gate_calibration.json"
)
CANINE_MIN_LANDMARK_SCORE = 0.50
CANINE_UPPER_OCCLUSAL_CAP = 0.10

TOOTHSEG_RUNTIME = RUNTIME_ROOT / "toothseg2d"
TOOTHSEG_ROI_MODULE = TOOTHSEG_RUNTIME / "toothseg_primary_arch_roi.py"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIEW_INFERENCE_BATCH_SIZE = 8
TARGET_FDIS = (11, 21, 31, 41)
EXPECTED_PERMANENT_FDIS = tuple(
    10 * quadrant + position
    for quadrant in (1, 2, 3, 4)
    for position in range(1, 8)
)
SEGMENTATION_SCORE_THRESHOLD = 0.01
MIDLINE_SCORE_THRESHOLD = 0.20
TOOTHSEG_IMAGE_SIZE = 768
TOOTHSEG_MINIMUM_TEETH = 2
CENTERED_THRESHOLD_TOOTH_WIDTH = 0.063
SLIGHTLY_DEVIATED_THRESHOLD_TOOTH_WIDTH = 0.160
CROWDING_NAMES = ["Mild", "Moderate", "Severe"]
CROWDING_DISPLAY_NAMES = {
    "Mild": "None to mild",
    "Moderate": "Mild to moderate",
    "Severe": "Moderate to severe",
}
REQUIRED_INTRAORAL_VIEWS = ("left", "right", "upper", "front", "lower")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BITS_REPO = Path("/data/MICCAI/Bits2Bites/bits2bites_pointcept")
POINTCEPT_RUNTIME = RUNTIME_ROOT / "pointcept"
TEETHLAND_ROOT = Path("/data/MICCAI/Bits2Bites/bits2bites_input_representation/external/3dteethland")
TEETHLAND_PYTHON = Path("/data/MICCAI/Bits2Bites/bits2bites_input_representation/.venv_3dteethland/bin/python")
BASE_CONFIG = CONFIG_ROOT / "3dteethland_config_full.yaml"
LEGACY_INFERENCE_SCRIPT = POINTCEPT_RUNTIME / "infer_resilient.py"

RAW_STL_OVERJET_SPEC = {
    "display_name": "Overjet (raw-STL Official Base balanced)",
    "task": "overjet",
    "run_id": "369367248013",
    "selection": "best_accuracy",
    "weight": MODELS_ROOT / "pointcept/overjet/overjet_raw_stl_balanced_run369367248013_best_accuracy.pth",
}
RAW_STL_POINTS_PER_ARCH = 4096
RAW_STL_INFERENCE_SEED = 42

TOOTHMASK_CURVE_SPECS = (
    {
        "display_name": "Curve of Spee (tooth-mask 2-class)",
        "task": "spee",
        "run_id": "369367253595",
        "selection": "best_macro_f1",
        "weight": MODELS_ROOT / "pointcept/curve/curve_spee_toothmask_2class_run369367253595_best_macro_f1.pth",
    },
    {
        "display_name": "Curve of Wilson (tooth-mask best accuracy)",
        "task": "wilson",
        "run_id": "369367253578",
        "selection": "best_accuracy",
        "weight": MODELS_ROOT / "pointcept/curve/curve_wilson_toothmask_run369367253578_best_accuracy.pth",
    },
)
ANTERIOR_BITE_TO_OVERBITE = {
    "Normal": "Normal",
    "Deep Bite": "Increased",
    "Open Bite": "Open bite",
    "Inverted Bite": "Decreased",
}
POINT_CLASSES = ("Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint")
POINT_CLASS_TO_INDEX = {name: index for index, name in enumerate(POINT_CLASSES)}

# ---- Legacy Utonia / official curve full-landmark inference helpers ----

def resolve_ios_pair(case_id):
    expected = {"lower": LOWER_STL, "upper": UPPER_STL}
    missing = [str(path) for path in expected.values() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            f"New STL input is incomplete for {case_id}; expected non-empty files: {missing}"
        )
    return expected

def prediction_paths(directory, case_id, arch):
    return directory / f"{case_id}_{arch}.json", directory / f"{case_id}_{arch}__kpt.json"


def valid_prediction_pair(directory, case_id):
    try:
        for arch in ("lower", "upper"):
            segmentation_path, landmark_path = prediction_paths(directory, case_id, arch)
            segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
            landmarks = json.loads(landmark_path.read_text(encoding="utf-8"))
            instances, labels = segmentation.get("instances"), segmentation.get("labels")
            objects = landmarks.get("objects")
            if not isinstance(instances, list) or not instances:
                return False
            if not isinstance(labels, list) or len(labels) != len(instances):
                return False
            if not isinstance(objects, list) or not objects:
                return False
        return True
    except (OSError, json.JSONDecodeError):
        return False


def stable_case_seed(case_id, seed):
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


# ---- Final selected FDI surface-area occlusion rules ----
# Molar: Bite2Text train weak-label optimized.
# Transverse: dataset-balanced Bite2Text train + Bits2Bites GT optimized.
# Anterior/overbite remains the original all-200 Bits2Bites calibration.
AREA_RULE_FDIS = {
    "upper": np.asarray([11, 12, 14, 15, 16, 17, 21, 22, 24, 25, 26, 27]),
    "lower": np.asarray([31, 32, 34, 35, 36, 37, 41, 42, 44, 45, 46, 47]),
}
AREA_SIGMAS = (0.25, 0.5, 1.0, 2.0, 3.0)
AREA_MOLAR_ORDER = ("Class II Full", "Class II Edge to Edge", "Class I", "Class III")
AREA_MOLAR_CUTS = (0.00028324080575471624, 0.2039917686280751, 0.9691732640782045)
AREA_ANTERIOR_RULE = (-0.07453999298570295, 0.3191343115257904, -1.8204783276291432)
AREA_TRANSVERSE_RULE = {
    "cross_cut": -0.046029583612910735,
    "overlap_cut": 0.23539054323117095,
    "outer_cut": 0.11263625755982448,
}

def area_weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))


def area_weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else np.nan


def pack_area_arch(points, labels, weights):
    by_fdi = {}
    for fdi in np.unique(labels):
        selected = labels == fdi
        by_fdi[int(fdi)] = (points[selected], weights[selected])
    return {"pts": points, "labels": labels, "weights": weights, "by_fdi": by_fdi}


def build_area_arch_from_surface(coord, faces, labels, arch):
    faces = np.asarray(faces, dtype=np.int64)
    triangles = np.asarray(coord, dtype=np.float64)[faces]
    face_labels = np.asarray(labels, dtype=np.int64)[faces]
    homogeneous = np.all(face_labels == face_labels[:, :1], axis=1)
    triangles = triangles[homogeneous]
    triangle_labels = face_labels[homogeneous, 0]
    selected = np.isin(triangle_labels, AREA_RULE_FDIS[arch])
    triangles, triangle_labels = triangles[selected], triangle_labels[selected]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    valid = np.isfinite(triangle_area) & (triangle_area > 1e-8)
    triangles, triangle_labels, triangle_area = (
        triangles[valid], triangle_labels[valid], triangle_area[valid]
    )
    # Exact symmetric three-point barycentric quadrature used by the calibrated rule.
    barycentric = np.asarray(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        dtype=np.float64,
    )
    points = np.einsum("qa,fad->fqd", barycentric, triangles).reshape(-1, 3)
    repeated_labels = np.repeat(triangle_labels, 3)
    weights = np.repeat(triangle_area / 3.0, 3)
    return pack_area_arch(points, repeated_labels, weights)



def load_area_arch_from_prediction(stl_path, prediction_path, arch):
    # This is intentionally identical to bits2bites_area_occlusion_rulebase_inference.ipynb.
    mesh = trimesh.load_mesh(stl_path, process=True)
    labels = np.asarray(
        json.loads(Path(prediction_path).read_text(encoding="utf-8"))["labels"], dtype=np.int64
    )
    if len(labels) != len(mesh.vertices):
        raise ValueError(
            f"Area-rule mesh/FDI mismatch for {arch}: vertices={len(mesh.vertices)}, labels={len(labels)}"
        )
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    face_labels = labels[np.asarray(mesh.faces)]
    homogeneous = np.all(face_labels == face_labels[:, :1], axis=1)
    triangles, triangle_labels = triangles[homogeneous], face_labels[homogeneous, 0]
    selected = np.isin(triangle_labels, AREA_RULE_FDIS[arch])
    triangles, triangle_labels = triangles[selected], triangle_labels[selected]
    triangle_area = trimesh.triangles.area(triangles)
    valid = np.isfinite(triangle_area) & (triangle_area > 1e-8)
    triangles, triangle_labels, triangle_area = (
        triangles[valid], triangle_labels[valid], triangle_area[valid]
    )
    barycentric = np.asarray(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        dtype=np.float64,
    )
    points = np.einsum("qa,fad->fqd", barycentric, triangles).reshape(-1, 3)
    return pack_area_arch(
        points, np.repeat(triangle_labels, 3), np.repeat(triangle_area / 3.0, 3)
    )


def load_area_arches(case_id, ios_pair, fresh_prediction_dir):
    cached_paths = {
        arch: AREA_REFERENCE_PRED_ROOT / f"{case_id}_{arch}.json" for arch in ("upper", "lower")
    }
    if all(path.is_file() for path in cached_paths.values()):
        prediction_paths_by_arch = cached_paths
        source = "validated Bite2Text FDI predictions used by standalone area-rule notebook"
    else:
        prediction_paths_by_arch = {
            arch: Path(fresh_prediction_dir) / f"{case_id}_{arch}.json" for arch in ("upper", "lower")
        }
        source = "fresh 3DTeethLand FDI predictions (reference cache unavailable)"
    arches = {
        arch: load_area_arch_from_prediction(ios_pair[arch], prediction_paths_by_arch[arch], arch)
        for arch in ("upper", "lower")
    }
    return arches, source


def area_tooth(arch, fdi):
    return arch["by_fdi"].get(int(fdi), (np.empty((0, 3)), np.empty(0)))


def area_concat_teeth(arch, fdis):
    point_parts, weight_parts = [], []
    for fdi in fdis:
        points, weights = area_tooth(arch, fdi)
        if len(points):
            point_parts.append(points)
            weight_parts.append(weights)
    if not point_parts:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(point_parts), np.concatenate(weight_parts)


def area_surface_band(points, weights, axis, quantile, upper):
    cut = area_weighted_quantile(points[:, axis], weights, quantile)
    selected = points[:, axis] >= cut if upper else points[:, axis] <= cut
    return points[selected], weights[selected]


def area_nearest_profiles(query_points, query_weights, candidates, prefix, output):
    trees, valid_ids = [], []
    for index, (points, _weights) in enumerate(candidates):
        if len(points):
            trees.append(cKDTree(points))
            valid_ids.append(index)
    if not trees or not len(query_points):
        return
    distances = np.column_stack(
        [tree.query(query_points, k=1, workers=-1)[0] for tree in trees]
    )
    nearest = np.argmin(distances, axis=1)
    candidate_index = np.asarray(valid_ids)[nearest].astype(float)
    minimum_distance = distances[np.arange(len(distances)), nearest]
    for sigma in AREA_SIGMAS:
        contact_weight = query_weights * np.exp(-0.5 * (minimum_distance / sigma) ** 2)
        denominator = contact_weight.sum()
        output[f"{prefix}_s{sigma:g}_area"] = float(denominator)
        output[f"{prefix}_s{sigma:g}_frac"] = float(
            denominator / max(query_weights.sum(), 1e-8)
        )
        output[f"{prefix}_s{sigma:g}_idx"] = area_weighted_mean(
            candidate_index, contact_weight
        )
        for candidate in range(len(candidates)):
            output[f"{prefix}_s{sigma:g}_p{candidate}"] = float(
                contact_weight[candidate_index == candidate].sum() / max(denominator, 1e-8)
            )


def area_molar_features(upper, lower, side, output):
    if side == "right":
        upper_fdi, lower_fdi = 16, 46
        lower_candidates = (45, 46, 47)
        upper_candidates = (15, 16, 17)
    else:
        upper_fdi, lower_fdi = 26, 36
        lower_candidates = (35, 36, 37)
        upper_candidates = (25, 26, 27)
    upper_points, upper_weights = area_tooth(upper, upper_fdi)
    lower_points, lower_weights = area_tooth(lower, lower_fdi)
    if not len(upper_points) or not len(lower_points):
        output[f"molar_{side}_available"] = 0
        return
    output[f"molar_{side}_available"] = 1
    upper_band, upper_band_weights = area_surface_band(
        upper_points, upper_weights, 2, 0.65, upper=False
    )
    lower_band, lower_band_weights = area_surface_band(
        lower_points, lower_weights, 2, 0.35, upper=True
    )
    area_nearest_profiles(
        upper_band,
        upper_band_weights,
        [area_tooth(lower, fdi) for fdi in lower_candidates],
        f"mf_{side}",
        output,
    )
    area_nearest_profiles(
        lower_band,
        lower_band_weights,
        [area_tooth(upper, fdi) for fdi in upper_candidates],
        f"mr_{side}",
        output,
    )
    scores = []
    for sigma in (1.0, 2.0, 3.0):
        forward = output.get(f"mf_{side}_s{sigma:g}_idx")
        reverse = output.get(f"mr_{side}_s{sigma:g}_idx")
        if np.isfinite(forward) and np.isfinite(reverse):
            scores.append(forward - reverse)
    output[f"molar_{side}_score"] = float(np.median(scores)) if scores else np.nan


def area_anterior_features(upper, lower, output):
    upper_points, upper_weights = area_concat_teeth(upper, (11, 12, 21, 22))
    lower_points, lower_weights = area_concat_teeth(lower, (31, 32, 41, 42))
    if not len(upper_points) or not len(lower_points):
        output["anterior_available"] = 0
        return
    output["anterior_available"] = 1
    upper_edge, upper_edge_weights = area_surface_band(
        upper_points, upper_weights, 2, 0.20, upper=False
    )
    lower_edge, lower_edge_weights = area_surface_band(
        lower_points, lower_weights, 2, 0.80, upper=True
    )
    crown_height = np.nanmedian(
        [
            area_weighted_quantile(upper_points[:, 2], upper_weights, 0.95)
            - area_weighted_quantile(upper_points[:, 2], upper_weights, 0.05),
            area_weighted_quantile(lower_points[:, 2], lower_weights, 0.95)
            - area_weighted_quantile(lower_points[:, 2], lower_weights, 0.05),
        ]
    )
    upper_z = area_weighted_quantile(upper_edge[:, 2], upper_edge_weights, 0.50)
    lower_z = area_weighted_quantile(lower_edge[:, 2], lower_edge_weights, 0.50)
    output["ant_overlap_mm"] = lower_z - upper_z
    output["ant_overlap_norm"] = (lower_z - upper_z) / max(crown_height, 1e-6)

    posterior_upper, posterior_upper_weights = area_concat_teeth(upper, (16, 26))
    posterior_lower, posterior_lower_weights = area_concat_teeth(lower, (36, 46))
    posterior = np.concatenate([posterior_upper, posterior_lower])
    posterior_weights = np.concatenate([posterior_upper_weights, posterior_lower_weights])
    incisor_xy = np.asarray(
        [
            area_weighted_quantile(
                np.r_[upper_points[:, axis], lower_points[:, axis]],
                np.r_[upper_weights, lower_weights],
                0.5,
            )
            for axis in (0, 1)
        ]
    )
    posterior_xy = np.asarray(
        [area_weighted_quantile(posterior[:, axis], posterior_weights, 0.5) for axis in (0, 1)]
    )
    direction = incisor_xy - posterior_xy
    direction /= max(np.linalg.norm(direction), 1e-8)
    upper_front = area_weighted_quantile(upper_points[:, :2] @ direction, upper_weights, 0.90)
    lower_front = area_weighted_quantile(lower_points[:, :2] @ direction, lower_weights, 0.90)
    output["ant_overjet_mm"] = upper_front - lower_front


def area_transverse_features(upper, lower, output):
    upper_incisors, upper_incisor_weights = area_concat_teeth(upper, (11, 21))
    lower_incisors, lower_incisor_weights = area_concat_teeth(lower, (31, 41))
    if not len(upper_incisors) or not len(lower_incisors):
        output["transverse_available"] = 0
        return
    middle_x = area_weighted_quantile(
        np.r_[upper_incisors[:, 0], lower_incisors[:, 0]],
        np.r_[upper_incisor_weights, lower_incisor_weights],
        0.5,
    )
    pair_rows = []
    for side, pairs in {
        "right": ((14, 44), (15, 45), (16, 46), (17, 47)),
        "left": ((24, 34), (25, 35), (26, 36), (27, 37)),
    }.items():
        for upper_fdi, lower_fdi in pairs:
            upper_points, upper_weights = area_tooth(upper, upper_fdi)
            lower_points, lower_weights = area_tooth(lower, lower_fdi)
            if not len(upper_points) or not len(lower_points):
                continue
            side_sign = np.sign(
                area_weighted_quantile(
                    np.r_[upper_points[:, 0], lower_points[:, 0]],
                    np.r_[upper_weights, lower_weights],
                    0.5,
                )
                - middle_x
            )
            if side_sign == 0:
                side_sign = 1.0
            upper_lateral = side_sign * (upper_points[:, 0] - middle_x)
            lower_lateral = side_sign * (lower_points[:, 0] - middle_x)
            upper_q10 = area_weighted_quantile(upper_lateral, upper_weights, 0.10)
            upper_q90 = area_weighted_quantile(upper_lateral, upper_weights, 0.90)
            lower_q10 = area_weighted_quantile(lower_lateral, lower_weights, 0.10)
            lower_q90 = area_weighted_quantile(lower_lateral, lower_weights, 0.90)
            width = np.nanmedian([upper_q90 - upper_q10, lower_q90 - lower_q10])
            offset = (upper_q90 - lower_q90) / max(width, 1e-6)
            overlap = max(0.0, min(upper_q90, lower_q90) - max(upper_q10, lower_q10)) / max(
                max(upper_q90, lower_q90) - min(upper_q10, lower_q10), 1e-6
            )
            pair_rows.append((side, upper_fdi, lower_fdi, offset, overlap))
            output[f"trans_{side}_{upper_fdi}_{lower_fdi}_offset"] = offset
            output[f"trans_{side}_{upper_fdi}_{lower_fdi}_overlap"] = overlap
    if not pair_rows:
        output["transverse_available"] = 0
        return
    output["transverse_available"] = 1
    values = np.asarray([[row[3], row[4]] for row in pair_rows], dtype=float)
    output["trans_offset_min"] = float(np.nanmin(values[:, 0]))
    output["trans_overlap_min"] = float(np.nanmin(values[:, 1]))


def area_predict_ordered(value, cuts, order):
    return np.asarray(order, dtype=object)[np.digitize(np.asarray([value], dtype=float), cuts)][0]


def area_threshold_margin(value, cuts):
    return float(min(abs(value - cut) for cut in cuts)) if np.isfinite(value) else np.nan


def infer_area_occlusion(area_arches):
    upper, lower = area_arches["upper"], area_arches["lower"]
    features = {}
    area_molar_features(upper, lower, "right", features)
    area_molar_features(upper, lower, "left", features)
    area_anterior_features(upper, lower, features)
    area_transverse_features(upper, lower, features)

    result = {"calibration": "final task-specific area-occlusion rules", "molar": {}}
    for side in ("right", "left"):
        score = features.get(f"molar_{side}_score", np.nan)
        result["molar"][side] = {
            "classification": (
                area_predict_ordered(score, AREA_MOLAR_CUTS, AREA_MOLAR_ORDER)
                if np.isfinite(score)
                else "Unavailable"
            ),
            "area_score": float(score) if np.isfinite(score) else None,
            "threshold_margin": area_threshold_margin(score, AREA_MOLAR_CUTS) if np.isfinite(score) else None,
        }

    overlap = features.get("ant_overlap_norm", np.nan)
    overjet = features.get("ant_overjet_mm", np.nan)
    if np.isfinite(overlap) and np.isfinite(overjet):
        open_cut, deep_cut, inverted_overjet_cut = AREA_ANTERIOR_RULE
        raw_anterior = (
            "Open Bite"
            if overlap < open_cut
            else "Inverted Bite"
            if overjet < inverted_overjet_cut
            else "Normal"
            if overlap < deep_cut
            else "Deep Bite"
        )
        overbite = ANTERIOR_BITE_TO_OVERBITE[raw_anterior]
    else:
        raw_anterior, overbite = "Unavailable", "Unavailable"
    result["overbite"] = {
        "classification": overbite,
        "raw_anterior_bite": raw_anterior,
        "overlap_to_crown_height": float(overlap) if np.isfinite(overlap) else None,
        "overjet_mm": float(overjet) if np.isfinite(overjet) else None,
        "threshold_margin": (
            area_threshold_margin(overlap, AREA_ANTERIOR_RULE[:2]) if np.isfinite(overlap) else None
        ),
    }

    pair_values = []
    for key, value in features.items():
        match = re.match(r"trans_(right|left)_(\d+)_(\d+)_offset$", key)
        if match and np.isfinite(value):
            pair_values.append((float(value), match.group(1), int(match.group(2)), int(match.group(3))))
    minimum_offset = features.get("trans_offset_min", np.nan)
    minimum_overlap = features.get("trans_overlap_min", np.nan)
    complete = len(pair_values) == 8
    location = ""
    if not np.isfinite(minimum_offset):
        transverse = "Unavailable"
    elif minimum_offset < AREA_TRANSVERSE_RULE["cross_cut"]:
        transverse = "Cross Bite"
        worst = min(pair_values)
        location = f"{worst[1]} (FDI {worst[2]}/{worst[3]})"
    elif (
        complete
        and minimum_overlap < AREA_TRANSVERSE_RULE["overlap_cut"]
        and minimum_offset > AREA_TRANSVERSE_RULE["outer_cut"]
    ):
        transverse = "Scissor Bite"
        worst = min(pair_values)
        location = f"{worst[1]} (FDI {worst[2]}/{worst[3]})"
    else:
        transverse = "Normal"
    result["transverse_bite"] = {
        "classification": transverse,
        "location": location,
        "minimum_outer_surface_offset": float(minimum_offset) if np.isfinite(minimum_offset) else None,
        "minimum_overlap": float(minimum_overlap) if np.isfinite(minimum_overlap) else None,
        "posterior_fdi_completeness": f"{len(pair_values)}/8 pairs",
        "scissor_subtype_assessable": complete,
    }

    return result


def area_occlusion_table_rows(result, source):
    rows = []
    for side in ("right", "left"):
        item = result["molar"][side]
        score = item["area_score"]
        margin = item["threshold_margin"]
        rows.append(
            {
                "item": "Molar relationship",
                "view": side,
                "image": None,
                "result": item["classification"],
                "confidence": None,
                "detail": (
                    f"area-occlusion rule; score={score:.3f}; threshold margin={margin:.3f}; "
                    f"calibration=Bite2Text train weak labels; source={source}"
                    if score is not None
                    else f"area-occlusion rule unavailable; source={source}"
                ),
            }
        )
    overbite = result["overbite"]
    rows.append(
        {
            "item": "Overbite (area-occlusion rule)",
            "view": "3D FDI tooth surfaces",
            "image": None,
            "result": overbite["classification"],
            "confidence": None,
            "detail": (
                f"raw={overbite['raw_anterior_bite']}; "
                f"overlap/crown={overbite['overlap_to_crown_height']}; "
                f"overjet_mm={overbite['overjet_mm']}; calibration=200/200 GT cases; source={source}"
            ),
        }
    )
    transverse = result["transverse_bite"]
    rows.append(
        {
            "item": "Transverse Bite (area-occlusion rule)",
            "view": "3D FDI tooth surfaces",
            "image": None,
            "result": transverse["classification"],
            "confidence": None,
            "detail": (
                f"location={transverse['location'] or '-'}; "
                f"outer_offset={transverse['minimum_outer_surface_offset']}; "
                f"FDI completeness={transverse['posterior_fdi_completeness']}; "
                f"calibration=dataset-balanced Bite2Text train + Bits2Bites GT; source={source}"
            ),
        }
    )

    return rows



def compacted_mesh_surface(stl_path):
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(stl_path))
    mesh_set.meshing_remove_duplicate_vertices()
    mesh = mesh_set.current_mesh()
    mesh.compact()
    coord = np.asarray(mesh.vertex_matrix(), dtype=np.float32)
    normal = np.asarray(mesh.vertex_normal_matrix(), dtype=np.float32)
    faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
    if coord.ndim != 2 or coord.shape[1] != 3 or normal.shape != coord.shape:
        raise ValueError(f"Invalid compacted mesh geometry: {stl_path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid compacted mesh faces: {stl_path}")
    return coord, normal, faces



def build_toothmask_pointset(case_id, ios_pair, prediction_dir, transform, sample_points=12288, seed=42):
    original_to_ras = np.asarray(transform["homogeneous_original_to_ras"], dtype=np.float64)
    all_teeth = {}
    for arch in ("upper", "lower"):
        segmentation_path, _ = prediction_paths(prediction_dir, case_id, arch)
        segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
        labels = np.asarray(segmentation.get("labels", []), dtype=np.int64).reshape(-1)
        instances = np.asarray(segmentation.get("instances", []), dtype=np.int64).reshape(-1)
        coord, normal, faces = compacted_mesh_surface(ios_pair[arch])
        if len(labels) != len(coord) or len(instances) != len(coord):
            raise ValueError(
                f"3DTeethLand/mesh vertex mismatch for {arch}: "
                f"mesh={len(coord)}, labels={len(labels)}, instances={len(instances)}"
            )
        homogeneous = np.concatenate(
            [coord.astype(np.float64), np.ones((len(coord), 1), dtype=np.float64)], axis=1
        )
        coord = (homogeneous @ original_to_ras.T)[:, :3].astype(np.float32)
        normal = (normal.astype(np.float64) @ original_to_ras[:3, :3].T).astype(np.float32)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        quadrants = (1, 2) if arch == "upper" else (3, 4)
        valid = (
            (instances >= 0)
            & np.isin(labels // 10, quadrants)
            & ((labels % 10) >= 1)
            & ((labels % 10) <= 8)
        )
        for fdi in sorted(int(value) for value in np.unique(labels[valid])):
            index = np.flatnonzero(valid & (labels == fdi))
            if len(index) >= 8:
                all_teeth[fdi] = (coord[index], normal[index])
    if len(all_teeth) < 8:
        raise RuntimeError(f"Too few valid FDI tooth masks for {case_id}: {sorted(all_teeth)}")

    rng = np.random.default_rng(stable_case_seed(case_id, seed))
    fdis = sorted(all_teeth)
    base, remainder = divmod(sample_points, len(fdis))
    sampled_coord, sampled_normal, sampled_fdi = [], [], []
    for order, fdi in enumerate(fdis):
        tooth_coord, tooth_normal = all_teeth[fdi]
        count = base + int(order < remainder)
        choice = rng.choice(len(tooth_coord), size=count, replace=len(tooth_coord) < count)
        sampled_coord.append(tooth_coord[choice])
        sampled_normal.append(tooth_normal[choice])
        sampled_fdi.append(np.full(count, fdi, dtype=np.int16))
    coord = np.concatenate(sampled_coord).astype(np.float32)
    normal = np.concatenate(sampled_normal).astype(np.float32)
    fdi = np.concatenate(sampled_fdi)
    permutation = rng.permutation(len(coord))
    coord, normal, fdi = coord[permutation], normal[permutation], fdi[permutation]
    coord -= np.median(coord, axis=0, keepdims=True).astype(np.float32)
    quadrant, position = fdi // 10, fdi % 10
    arch_onehot = np.zeros((sample_points, 2), dtype=np.float32)
    arch_onehot[:, 0] = np.isin(quadrant, (1, 2))
    arch_onehot[:, 1] = np.isin(quadrant, (3, 4))
    toothmask = {
        "coord": coord,
        "normal": normal,
        "arch_onehot": arch_onehot,
        "position_onehot": np.eye(8, dtype=np.float32)[position - 1],
        "quadrant_onehot": np.eye(4, dtype=np.float32)[quadrant - 1],
        "fdi": fdi,
        "detected_fdis": [int(value) for value in fdis],
        "missing_fdis": sorted(set(EXPECTED_PERMANENT_FDIS) - set(fdis)),
    }
    return toothmask




def canine_weighted_point_median(points, weights):
    points = np.asarray(points, dtype=np.float64)
    return np.asarray(
        [area_weighted_quantile(points[:, axis], weights, 0.5) for axis in range(3)],
        dtype=np.float64,
    )


def canine_surface_samples(vertices, faces, labels):
    face_labels = labels[faces]
    homogeneous = np.all(face_labels == face_labels[:, :1], axis=1)
    triangles = vertices[faces[homogeneous]]
    face_fdis = face_labels[homogeneous, 0]
    areas = trimesh.triangles.area(triangles)
    valid = np.isfinite(areas) & (areas > 1e-8) & (face_fdis > 0)
    triangles, areas, face_fdis = triangles[valid], areas[valid], face_fdis[valid]
    barycentric = np.asarray(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        dtype=np.float64,
    )
    points = np.einsum("qa,fad->fqd", barycentric, triangles).reshape(-1, 3)
    return points, np.repeat(areas / 3.0, 3), np.repeat(face_fdis, 3)


def load_canine_arch_prediction(stl_path, segmentation_path, landmark_path):
    mesh = trimesh.load_mesh(stl_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    segmentation = json.loads(Path(segmentation_path).read_text(encoding="utf-8"))
    landmark_json = json.loads(Path(landmark_path).read_text(encoding="utf-8"))
    labels = np.asarray(segmentation.get("labels", []), dtype=np.int64).reshape(-1)
    instances = np.asarray(segmentation.get("instances", []), dtype=np.int64).reshape(-1)
    if len(vertices) != len(labels) or len(labels) != len(instances):
        raise ValueError(
            f"Canine STL/3DTeethLand mismatch: vertices={len(vertices)}, "
            f"labels={len(labels)}, instances={len(instances)}"
        )

    votes = defaultdict(Counter)
    for instance, fdi in zip(instances, labels):
        if int(instance) >= 0 and int(fdi) > 0:
            votes[int(instance)][int(fdi)] += 1
    instance_to_fdi = {
        instance: counts.most_common(1)[0][0] for instance, counts in votes.items()
    }
    landmarks = {}
    for item in landmark_json.get("objects", []):
        fdi = instance_to_fdi.get(int(item.get("instance_id", -999)))
        landmark_class = str(item.get("class", ""))
        score = float(item.get("score", 0.0))
        coord = np.asarray(item.get("coord", []), dtype=np.float64)
        if fdi is None or coord.shape != (3,) or not np.all(np.isfinite(coord)):
            continue
        key = (int(fdi), landmark_class)
        if key not in landmarks or score > landmarks[key]["score"]:
            landmarks[key] = {"coord": coord, "score": score}

    points, weights, fdis = canine_surface_samples(vertices, faces, labels)
    tooth_surfaces = {
        int(fdi): (points[fdis == int(fdi)], weights[fdis == int(fdi)])
        for fdi in np.unique(fdis)
        if int(fdi) > 0
    }
    positive = vertices[labels > 0]
    if not len(positive):
        raise RuntimeError(f"No FDI-labeled tooth points in {segmentation_path}")
    return {
        "tooth_surfaces": tooth_surfaces,
        "landmarks": landmarks,
        "arch_points": positive,
        "present_fdis": sorted(tooth_surfaces),
        "point_counts": {
            int(fdi): int(np.sum(labels == int(fdi)))
            for fdi in np.unique(labels)
            if int(fdi) > 0
        },
    }


def build_canine_arches(case_id, ios_pair, prediction_dir):
    arches = {}
    for arch in ("upper", "lower"):
        segmentation_path, landmark_path = prediction_paths(prediction_dir, case_id, arch)
        arches[arch] = load_canine_arch_prediction(
            ios_pair[arch], segmentation_path, landmark_path
        )
    return arches


def pointset_from_predictions(case_id, ios_pair, prediction_dir, score_threshold=0.5):
    objects = []
    for arch in ("lower", "upper"):
        segmentation_path, landmark_path = prediction_paths(prediction_dir, case_id, arch)
        objects.extend(select_scan(landmark_path, segmentation_path, arch, score_threshold))
    transformed, transform, quality = canonicalize(case_id, objects, max_arch_axis_angle=55.0)
    toothmask = build_toothmask_pointset(case_id, ios_pair, prediction_dir, transform)
    raw_stl = build_raw_stl_pointset(case_id, ios_pair, prediction_dir)
    area_arches, area_source = load_area_arches(case_id, ios_pair, prediction_dir)
    canine_arches = build_canine_arches(case_id, ios_pair, prediction_dir)
    return {
        "version": "3.1",
        "patient": case_id,
        "coordinate_system": "RAS",
        "units": "mm",
        "objects": transformed,
        "toothmask": toothmask,
        "raw_stl": raw_stl,
        "area_arches": area_arches,
        "area_source": area_source,
        "canine_arches": canine_arches,
        "transform": transform,
        "quality": quality,
    }


def load_full_pointset(case_id):
    """Always run 3DTeethLand from the selected case's two STL inputs.

    Prepared pointsets and shared prediction caches are deliberately ignored.
    All intermediate segmentation and landmark files live in a temporary folder.
    """
    ios_pair = resolve_ios_pair(case_id)
    with tempfile.TemporaryDirectory(prefix=f"{case_id}_fresh_3dteethland_") as temporary:
        temporary = Path(temporary)
        stage_dir = temporary / "staging"
        prediction_dir = temporary / "predictions"
        stage_dir.mkdir()
        prediction_dir.mkdir()
        for arch in ("lower", "upper"):
            staged_path = stage_dir / f"{case_id}_{arch}.stl"
            try:
                staged_path.symlink_to(ios_pair[arch].resolve())
            except OSError:
                shutil.copy2(ios_pair[arch], staged_path)

        config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        config["seed"] = 42
        config["version"] = f"fresh_submission_{case_id}"
        config["out_dir"] = str(prediction_dir)
        config["work_dir"] = str(temporary / "logs")
        config["model"]["instseg"]["checkpoint_path"] = str(TEETHLAND_INSTSEG_WEIGHT)
        config["model"]["landmarks"]["checkpoint_path"] = str(TEETHLAND_LANDMARK_WEIGHT)
        config["datamodule"]["root"] = str(stage_dir)
        config["datamodule"]["regex_filter"] = ""
        config["datamodule"]["batch_size"] = 1
        config["datamodule"]["num_workers"] = 2
        config["datamodule"]["persistent_workers"] = False
        config_path = temporary / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"Running fresh 3DTeethLand inference from {LOWER_STL.name} and {UPPER_STL.name}...")
        subprocess.run(
            [
                str(TEETHLAND_PYTHON),
                str(LEGACY_INFERENCE_SCRIPT),
                "landmarks",
                "--devices",
                "1",
                "--config",
                str(config_path),
            ],
            cwd=TEETHLAND_ROOT,
            check=True,
        )
        if not valid_prediction_pair(prediction_dir, case_id):
            raise RuntimeError("Fresh 3DTeethLand inference did not produce valid segmentation/landmark pairs")
        pointset = pointset_from_predictions(case_id, ios_pair, prediction_dir)
        return pointset, "fresh 3DTeethLand inference from input STL files"

RAW_STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def raw_stl_normalize_vector(vector, name):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-6:
        raise RuntimeError(f"Cannot determine raw-STL RAS {name} axis")
    return vector / length


def raw_stl_read_binary_triangles(path):
    import struct

    with Path(path).open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    expected_size = 84 + triangle_count * RAW_STL_DTYPE.itemsize
    if Path(path).stat().st_size != expected_size:
        raise RuntimeError(f"The raw-STL overjet model requires a valid binary STL: {path}")
    records = np.fromfile(path, dtype=RAW_STL_DTYPE, count=triangle_count, offset=84)
    if len(records) != triangle_count:
        raise RuntimeError(f"Truncated binary STL: {path}")
    return records["vertices"].astype(np.float64, copy=False)


def raw_stl_sample_surface(path, point_count, rng):
    triangles = raw_stl_read_binary_triangles(path)
    edge_1, edge_2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_1, edge_2)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    triangles, cross, double_area = triangles[valid], cross[valid], double_area[valid]
    if not len(triangles):
        raise RuntimeError(f"No valid STL triangles: {path}")
    face_index = rng.choice(len(triangles), size=point_count, replace=True, p=double_area / double_area.sum())
    selected = triangles[face_index]
    root_u, v = np.sqrt(rng.random(point_count)), rng.random(point_count)
    coord = (
        (1.0 - root_u)[:, None] * selected[:, 0]
        + (root_u * (1.0 - v))[:, None] * selected[:, 1]
        + (root_u * v)[:, None] * selected[:, 2]
    )
    normal = cross[face_index]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-12)
    return coord.astype(np.float32), normal.astype(np.float32)


def raw_stl_predicted_joint_ras(case_id, ios_pair, prediction_dir):
    """Recreate the training RAS convention from inferred FDI crown centroids."""
    centroids = {}
    for arch in ("upper", "lower"):
        segmentation_path, _ = prediction_paths(prediction_dir, case_id, arch)
        labels = np.asarray(json.loads(segmentation_path.read_text(encoding="utf-8")).get("labels", []), dtype=np.int64)
        coord, _, _ = compacted_mesh_surface(ios_pair[arch])
        if len(labels) != len(coord):
            raise RuntimeError(f"FDI/mesh vertex mismatch for raw-STL RAS: {arch}")
        quadrants = (1, 2) if arch == "upper" else (3, 4)
        valid = np.isin(labels // 10, quadrants) & np.isin(labels % 10, range(1, 9))
        for fdi in sorted(int(value) for value in np.unique(labels[valid])):
            points = coord[valid & (labels == fdi)]
            if len(points) >= 8:
                centroids[fdi] = points.mean(axis=0).astype(np.float64)
    upper = [value for fdi, value in centroids.items() if fdi // 10 in (1, 2)]
    lower = [value for fdi, value in centroids.items() if fdi // 10 in (3, 4)]
    right = [value for fdi, value in centroids.items() if fdi // 10 in (1, 4)]
    left = [value for fdi, value in centroids.items() if fdi // 10 in (2, 3)]
    anterior = [value for fdi, value in centroids.items() if fdi % 10 in (1, 2, 3)]
    posterior = [value for fdi, value in centroids.items() if fdi % 10 in (5, 6, 7, 8)]
    if min(len(upper), len(lower), len(right), len(left), len(anterior), len(posterior)) < 2:
        raise RuntimeError(f"Insufficient inferred FDI teeth for raw-STL RAS: {case_id}")
    right_hint = np.mean(right, axis=0) - np.mean(left, axis=0)
    anterior_hint = np.mean(anterior, axis=0) - np.mean(posterior, axis=0)
    superior_hint = np.mean(upper, axis=0) - np.mean(lower, axis=0)
    superior_axis = raw_stl_normalize_vector(superior_hint, "superior")
    anterior_axis = raw_stl_normalize_vector(
        anterior_hint - np.dot(anterior_hint, superior_axis) * superior_axis, "anterior"
    )
    right_axis = raw_stl_normalize_vector(np.cross(anterior_axis, superior_axis), "right")
    if np.dot(right_axis, right_hint) < 0:
        raise RuntimeError(f"Inferred FDI axes are not right-handed for {case_id}")
    anterior_axis = raw_stl_normalize_vector(np.cross(superior_axis, right_axis), "anterior")
    basis = np.stack([right_axis, anterior_axis, superior_axis], axis=1)
    if np.linalg.det(basis) < 0.999:
        raise RuntimeError(f"Invalid raw-STL RAS basis for {case_id}")
    return np.mean(list(centroids.values()), axis=0).astype(np.float32), basis.astype(np.float32), sorted(centroids)


def build_raw_stl_pointset(case_id, ios_pair, prediction_dir):
    origin, basis, detected_fdis = raw_stl_predicted_joint_ras(case_id, ios_pair, prediction_dir)
    rng = np.random.default_rng(RAW_STL_INFERENCE_SEED + int(str(case_id).lstrip("F")))
    arrays = []
    for arch_index, arch in enumerate(("upper", "lower")):
        coord, normal = raw_stl_sample_surface(ios_pair[arch], RAW_STL_POINTS_PER_ARCH, rng)
        coord = (coord - origin) @ basis
        normal = normal @ basis
        arch_onehot = np.zeros((RAW_STL_POINTS_PER_ARCH, 2), dtype=np.float32)
        arch_onehot[:, arch_index] = 1.0
        arrays.append((coord, normal, arch_onehot))
    coord = np.concatenate([item[0] for item in arrays], axis=0).astype(np.float32)
    normal = np.concatenate([item[1] for item in arrays], axis=0).astype(np.float32)
    arch_onehot = np.concatenate([item[2] for item in arrays], axis=0)
    order = rng.permutation(len(coord))
    coord, normal, arch_onehot = coord[order], normal[order], arch_onehot[order]
    coord = coord - coord.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(coord, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Invalid raw-STL point-cloud radius")
    return {
        "coord": (coord / radius).astype(np.float32),
        "normal": normal,
        "arch_onehot": arch_onehot,
        "detected_fdis": detected_fdis,
    }


def raw_stl_official_base_model_config(num_classes, class_weights):
    return dict(
        type="SingleTaskClassifier",
        backbone_embed_dim=512,
        task_index=4,
        num_classes=num_classes,
        class_weight=class_weights,
        label_smoothing=0.0,
        backbone=dict(
            type="PT-v3m1", in_channels=8, order=("z", "z-trans"), stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48, 48, 48, 48, 48),
            mlp_ratio=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            pre_norm=True, shuffle_orders=True, enable_rpe=False, enable_flash=False, cls_mode=True,
        ),
    )


def make_raw_stl_inference_batch(raw_stl, grid_size=0.01):
    transform = Compose([
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "label_4"), feat_keys=["coord", "normal", "arch_onehot"]),
    ])
    sample = transform({
        "coord": raw_stl["coord"].copy(), "normal": raw_stl["normal"].copy(),
        "arch_onehot": raw_stl["arch_onehot"].copy(), "label_4": np.asarray([0], dtype=np.int64),
        "index_valid_keys": ["coord", "normal", "arch_onehot"],
    })
    return collate_fn([sample])


def toothmask_curve_model_config(num_classes, class_weights):
    return dict(
        type="MultiTaskClassifier",
        backbone_embed_dim=512,
        num_classes_list=[num_classes],
        stl_task=0,
        class_weights=[class_weights],
        loss_type="ce",
        label_smoothing=0.03,
        backbone=dict(
            type="PT-v3m1",
            in_channels=20,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=True,
        ),
    )


def make_toothmask_curve_inference_batch(pointset, deterministic_seed, grid_size=0.4):
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    toothmask = pointset["toothmask"]
    transform = Compose([
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "label_0"),
            feat_keys=["coord", "normal", "arch_onehot", "position_onehot", "quadrant_onehot"],
        ),
    ])
    sample = transform({
        "coord": toothmask["coord"].copy(),
        "normal": toothmask["normal"].copy(),
        "arch_onehot": toothmask["arch_onehot"].copy(),
        "position_onehot": toothmask["position_onehot"].copy(),
        "quadrant_onehot": toothmask["quadrant_onehot"].copy(),
        "label_0": np.asarray([0], dtype=np.int64),
        "index_valid_keys": ["coord", "normal", "arch_onehot", "position_onehot", "quadrant_onehot"],
    })
    return collate_fn([sample])


def move_to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_cuda(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_cuda(item) for item in value)
    return value


def ensure_legacy_pointcept_runtime():
    sys.path.insert(0, str(BITS_REPO))
    sys.path.insert(0, str(POINTCEPT_RUNTIME))
    try:
        import addict  # noqa: F401
        import spconv.pytorch  # noqa: F401
        import timm  # noqa: F401
        import torch_cluster  # noqa: F401
        import torch_geometric  # noqa: F401
        import torch_scatter  # noqa: F401
        import torch_sparse  # noqa: F401
        import wandb  # noqa: F401
        import sklearn  # noqa: F401
        import pymeshlab  # noqa: F401
    except ModuleNotFoundError:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "addict", "einops", "timm", "tensorboard", "tensorboardX", "termcolor",
                "matplotlib", "plyfile", "yapf", "wandb", "scipy", "pandas", "ftfy", "regex",
                "scikit-learn", "pymeshlab",
            ],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "spconv-cu126"], check=True)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-index",
                "torch-scatter", "torch-cluster", "torch-sparse", "-f",
                "https://data.pyg.org/whl/torch-2.7.0+cu128.html",
            ],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch-geometric"], check=True)
    for extension_name in ("pointops", "pointgroup_ops"):
        try:
            importlib.import_module(extension_name)
        except ModuleNotFoundError:
            build_env = os.environ.copy()
            build_env.update(TORCH_CUDA_ARCH_LIST="8.6", MAX_JOBS="4")
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--no-build-isolation",
                    "--no-deps", "--force-reinstall", str(BITS_REPO / "libs" / extension_name),
                ],
                env=build_env,
                check=True,
            )
            importlib.invalidate_caches()
            importlib.import_module(extension_name)
    globals()["canonicalize"] = importlib.import_module("build_ras_landmark_pointsets").canonicalize
    globals()["select_scan"] = importlib.import_module("build_ras_landmark_pointsets").select_scan
    globals()["collate_fn"] = importlib.import_module("pointcept.datasets").collate_fn
    globals()["Compose"] = importlib.import_module("pointcept.datasets.transform").Compose
    globals()["build_model"] = importlib.import_module("pointcept.models").build_model




def load_canine_rule_assets():
    calibration_payload = json.loads(CANINE_CALIBRATION_PATH.read_text(encoding="utf-8"))
    gate = json.loads(CANINE_ERUPTION_GATE_PATH.read_text(encoding="utf-8"))
    return calibration_payload["calibration"]["hybrid"], gate


def canine_side_fdis(side):
    if side == "right":
        return 13, 43, 44
    if side == "left":
        return 23, 33, 34
    raise ValueError(side)


def canine_bounded_ratio(numerator, denominator):
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 2.0))


def canine_gate_features(side, upper, lower):
    upper_canine, lower_canine, lower_premolar = canine_side_fdis(side)
    contra_upper = 23 if side == "right" else 13
    contra_lower = 33 if side == "right" else 43
    upper_counts, lower_counts = upper["point_counts"], lower["point_counts"]
    uc_n = upper_counts.get(upper_canine, 0)
    lc_n = lower_counts.get(lower_canine, 0)
    lp_n = lower_counts.get(lower_premolar, 0)
    upper_values = [value for fdi, value in upper_counts.items() if 11 <= fdi <= 28]
    lower_values = [value for fdi, value in lower_counts.items() if 31 <= fdi <= 48]
    upper_median = float(np.median(upper_values)) if upper_values else 0.0
    lower_median = float(np.median(lower_values)) if lower_values else 0.0
    lower_scores = [
        lower["landmarks"].get((fdi, landmark_class), {"score": 0.0})["score"]
        for fdi in (lower_canine, lower_premolar)
        for landmark_class in ("Mesial", "Distal")
    ]
    upper_scores = [
        upper["landmarks"].get((upper_canine, landmark_class), {"score": 0.0})["score"]
        for landmark_class in ("Cusp", "FacialPoint", "InnerPoint", "OuterPoint", "Mesial", "Distal")
    ]
    return {
        "required_missing_count": int(sum(value <= 0 for value in (uc_n, lc_n, lp_n))),
        "upper_canine_arch_ratio": canine_bounded_ratio(uc_n, upper_median),
        "lower_canine_arch_ratio": canine_bounded_ratio(lc_n, lower_median),
        "lower_premolar_arch_ratio": canine_bounded_ratio(lp_n, lower_median),
        "upper_canine_contra_ratio": canine_bounded_ratio(uc_n, upper_counts.get(contra_upper, 0)),
        "lower_canine_contra_ratio": canine_bounded_ratio(lc_n, lower_counts.get(contra_lower, 0)),
        "lower_md_completeness": float(
            np.mean(np.asarray(lower_scores) >= CANINE_MIN_LANDMARK_SCORE)
        ),
        "lower_md_min_score": float(min(lower_scores)),
        "upper_landmark_completeness": float(
            np.mean(np.asarray(upper_scores) >= CANINE_MIN_LANDMARK_SCORE)
        ),
        "upper_landmark_min_score": float(min(upper_scores)),
    }


def evaluate_canine_gate(side, upper, lower, gate):
    features = canine_gate_features(side, upper, lower)
    values = np.asarray([features[name] for name in gate["features"]], dtype=np.float64)
    mean = np.asarray(gate["mean"], dtype=np.float64)
    scale = np.asarray(gate["scale"], dtype=np.float64)
    coefficient = np.asarray(gate["coefficient"], dtype=np.float64)
    logit = float(np.dot((values - mean) / scale, coefficient) + gate["intercept"])
    probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
    reasons = []
    if features["required_missing_count"] > 0:
        reasons.append(f"{features['required_missing_count']} required FDI mask(s) missing")
    if features["lower_md_completeness"] < 1.0:
        reasons.append(
            f"lower canine/premolar M/D completeness={features['lower_md_completeness']:.2f}"
        )
    threshold = float(gate["probability_threshold"])
    if probability >= threshold:
        reasons.append(f"quality probability={probability:.3f} >= {threshold:.3f}")
    return {
        "cannot_assess": bool(reasons),
        "probability": probability,
        "reason": "; ".join(reasons) if reasons else f"quality probability={probability:.3f} < {threshold:.3f}",
        "features": features,
    }


def canine_surface_center(points, weights):
    if len(points) < 10 or float(np.sum(weights)) <= 0:
        raise ValueError(f"Insufficient canine crown surface: {len(points)} samples")
    return canine_weighted_point_median(points, weights)


def canine_mask_tip(upper, lower, fdi):
    points, weights = upper["tooth_surfaces"].get(
        int(fdi), (np.empty((0, 3)), np.empty(0))
    )
    if len(points) < 30:
        raise ValueError(f"Insufficient upper FDI {fdi} surface: {len(points)} samples")
    direction = np.median(lower["arch_points"], axis=0) - np.median(upper["arch_points"], axis=0)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    projection = points @ direction
    cutoff = area_weighted_quantile(
        projection, weights, 1.0 - CANINE_UPPER_OCCLUSAL_CAP
    )
    selected = projection >= cutoff
    return canine_surface_center(points[selected], weights[selected]), int(np.sum(selected))


def canine_landmark(prediction, fdi, landmark_class):
    item = prediction["landmarks"].get((int(fdi), landmark_class))
    if item is None:
        raise KeyError(f"Missing {landmark_class} landmark for FDI {fdi}")
    if item["score"] < CANINE_MIN_LANDMARK_SCORE:
        raise ValueError(
            f"Low-confidence {landmark_class} landmark for FDI {fdi}: {item['score']:.3f}"
        )
    return item["coord"], item["score"]


def classify_canine_score(score, calibration):
    value = float(calibration["sign"]) * float(score)
    if value < float(calibration["low"]):
        return "Class II"
    if value > float(calibration["high"]):
        return "Class III"
    return "Class I"


def infer_canine_side(side, upper, lower, calibration, gate):
    upper_canine, lower_canine, lower_premolar = canine_side_fdis(side)
    gate_result = evaluate_canine_gate(side, upper, lower, gate)
    base = {
        "side": side,
        "upper_canine": upper_canine,
        "lower_canine": lower_canine,
        "lower_first_premolar": lower_premolar,
        "gate_probability": gate_result["probability"],
        "gate_reason": gate_result["reason"],
    }
    if gate_result["cannot_assess"]:
        return {**base, "classification": "Cannot be assessed", "r": None, "status": "gate"}
    try:
        canine_tip, cap_count = canine_mask_tip(upper, lower, upper_canine)
        mesial_canine, score_mc = canine_landmark(lower, lower_canine, "Mesial")
        distal_canine, score_dc = canine_landmark(lower, lower_canine, "Distal")
        mesial_premolar, score_mp = canine_landmark(lower, lower_premolar, "Mesial")
        distal_premolar, score_dp = canine_landmark(lower, lower_premolar, "Distal")
        center_canine = (mesial_canine + distal_canine) / 2.0
        center_premolar = (mesial_premolar + distal_premolar) / 2.0
        axis = center_premolar - center_canine
        axis_length = float(np.linalg.norm(axis))
        canine_width = float(np.linalg.norm(mesial_canine - distal_canine))
        if axis_length < 1e-6 or canine_width < 1e-6:
            raise ValueError(
                f"Degenerate canine geometry: center={axis_length:.6f}, width={canine_width:.6f}"
            )
        axis /= axis_length
        embrasure = (distal_canine + mesial_premolar) / 2.0
        raw_projection = float(np.dot(canine_tip - embrasure, axis))
        score = raw_projection / canine_width
        return {
            **base,
            "classification": classify_canine_score(score, calibration),
            "r": float(score),
            "raw_projection": raw_projection,
            "canine_width": canine_width,
            "cap_points": cap_count,
            "minimum_md_confidence": float(min(score_mc, score_dc, score_mp, score_dp)),
            "status": "ok",
        }
    except (KeyError, ValueError) as error:
        return {
            **base,
            "classification": "Cannot be assessed",
            "r": None,
            "status": str(error),
        }


def infer_canine_occlusion(canine_arches):
    calibration, gate = load_canine_rule_assets()
    upper, lower = canine_arches["upper"], canine_arches["lower"]
    sides = {
        side: infer_canine_side(side, upper, lower, calibration, gate)
        for side in ("right", "left")
    }
    rows = []
    for side, result in sides.items():
        rows.append(
            {
                "item": "Canine relationship",
                "view": side,
                "image": None,
                "result": result["classification"],
                "confidence": None,
                "detail": (
                    f"mask-tip + mandibular M/D rule; r={result['r']}; "
                    f"gate_probability={result['gate_probability']:.3f}; status={result['status']}; "
                    "calibration=Bite2Text validation-selected hybrid rule; "
                    "source=shared fresh 3DTeethLand FDI masks and landmarks"
                ),
            }
        )
    return rows, {
        "method": "FDI-mask canine tip + mandibular M/D landmarks",
        "thresholds": {
            "sign": float(calibration["sign"]),
            "class_ii_below": float(calibration["low"]),
            "class_iii_above": float(calibration["high"]),
        },
        "sides": sides,
    }


def infer_3d_table_rows(case_id):
    required = [
        BITS_REPO,
        POINTCEPT_RUNTIME,
        TEETHLAND_ROOT,
        TEETHLAND_PYTHON,
        BASE_CONFIG,
        LEGACY_INFERENCE_SCRIPT,
        TEETHLAND_INSTSEG_WEIGHT,
        TEETHLAND_LANDMARK_WEIGHT,
        *[spec["weight"] for spec in TOOTHMASK_CURVE_SPECS],
        RAW_STL_OVERJET_SPEC["weight"],
        CANINE_CALIBRATION_PATH,
        CANINE_ERUPTION_GATE_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required 3D paths: " + "; ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 3DTeethLand and tooth-mask PTv3 inference")
    ensure_legacy_pointcept_runtime()
    pointset, source = load_full_pointset(case_id)
    if (
        not pointset.get("area_arches")
        or not pointset.get("toothmask")
        or not pointset.get("raw_stl")
        or not pointset.get("canine_arches")
    ):
        raise RuntimeError(f"Fresh FDI surface representations were not created for {case_id}")

    area_result = infer_area_occlusion(pointset["area_arches"])
    rows = area_occlusion_table_rows(area_result, pointset['area_source'])
    canine_rows, canine_result = infer_canine_occlusion(pointset["canine_arches"])
    rows.extend(canine_rows)
    toothmask = pointset["toothmask"]
    tooth_presence = {
        "reference": "permanent dentition excluding third molars",
        "expected_fdis": [int(value) for value in EXPECTED_PERMANENT_FDIS],
        "detected_fdis": [int(value) for value in toothmask["detected_fdis"]],
        "missing_fdis": [int(value) for value in toothmask["missing_fdis"]],
    }
    missing_fdis = tooth_presence["missing_fdis"]
    rows.append(
        {
            "item": "Missing teeth",
            "view": "3D tooth masks",
            "image": None,
            "result": (
                "FDI " + ", ".join(str(value) for value in missing_fdis)
                if missing_fdis
                else "No missing teeth detected"
            ),
            "confidence": None,
            "detail": (
                f"detected FDI={tooth_presence['detected_fdis']}; "
                "reference=28 permanent teeth excluding third molars; "
                "source=fresh 3DTeethLand FDI masks"
            ),
        }
    )
    for task_offset, spec in enumerate(TOOTHMASK_CURVE_SPECS):
        checkpoint_path = spec["weight"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        classes = tuple(payload["classes"])
        expected_task = spec["task"]
        if payload.get("task") != expected_task:
            raise ValueError(
                f"Checkpoint task mismatch: expected {expected_task}, got {payload.get('task')}"
            )
        model = build_model(toothmask_curve_model_config(len(classes), payload["class_weights"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.cuda().eval()
        if hasattr(model.backbone, "shuffle_orders"):
            model.backbone.shuffle_orders = False
        batch = move_to_cuda(
            make_toothmask_curve_inference_batch(
                pointset,
                deterministic_seed=42 + task_offset,
                grid_size=float(payload.get("grid_size_mm", 0.4)),
            )
        )
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            probabilities = F.softmax(model(batch)["logits"][0], dim=1)[0].float().cpu().numpy()
        predicted_index = int(probabilities.argmax())
        sorted_probabilities = np.sort(probabilities)[::-1]
        rows.append(
            {
                "item": spec["display_name"],
                "view": "3D tooth masks",
                "image": None,
                "result": classes[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "detail": (
                    f"official_base_toothmask; run={spec['run_id']}; weight={spec['selection']}; "
                    f"margin={float(sorted_probabilities[0] - sorted_probabilities[1]):.3f}; "
                    f"sample_points={int(payload.get('sample_points', 12288))}; "
                    f"source=fresh 3DTeethLand FDI masks"
                ),
            }
        )
        del model, batch, payload
        gc.collect()
        torch.cuda.empty_cache()

    # The official-base raw-STL classifier is evaluated after the curve models.
    checkpoint_path = RAW_STL_OVERJET_SPEC["weight"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = tuple(payload["classes"])
    if payload.get("task") != "overjet" or payload.get("input_mode") != "raw_stl":
        raise ValueError("Raw-STL overjet checkpoint metadata does not match the expected model")
    model = build_model(raw_stl_official_base_model_config(len(classes), payload["class_weight"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.cuda().eval()
    if hasattr(model.backbone, "shuffle_orders"):
        model.backbone.shuffle_orders = False
    batch = move_to_cuda(make_raw_stl_inference_batch(pointset["raw_stl"], grid_size=0.01))
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        probabilities = F.softmax(model(batch)["logits"], dim=1)[0].float().cpu().numpy()
    predicted_index = int(probabilities.argmax())
    sorted_probabilities = np.sort(probabilities)[::-1]
    rows.append({
        "item": RAW_STL_OVERJET_SPEC["display_name"],
        "view": "3D upper/lower STL",
        "image": None,
        "result": classes[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "detail": (
            f"official_base_raw_stl; run={RAW_STL_OVERJET_SPEC['run_id']}; "
            f"weight={RAW_STL_OVERJET_SPEC['selection']}; margin={float(sorted_probabilities[0] - sorted_probabilities[1]):.3f}; "
            f"sample_points={RAW_STL_POINTS_PER_ARCH * 2}; grid=0.01; "
            "source=fresh input upper/lower STL with inferred-FDI RAS"
        ),
    })
    del model, batch, payload
    gc.collect()
    torch.cuda.empty_cache()
    return rows, area_result, canine_result, tooth_presence


# Corpus-derived deterministic report rules.
# The 997 combined case files contain 2,404 English source reports. Their most
# stable finding order is sagittal -> transverse -> vertical -> midline ->
# Spee/Wilson -> crowding. Only values already present in summary_rows are used.
REPORT_UNAVAILABLE_VALUES = {"", "unknown", "unavailable", "none", "nan", "not available"}


def summary_result(summary_rows, item_prefix, view=None):
    item_prefix = str(item_prefix).casefold()
    target_view = None if view is None else str(view).casefold()
    for row in summary_rows:
        row_item = str(row.get("item", "")).casefold()
        row_view = str(row.get("view", "")).casefold()
        if not row_item.startswith(item_prefix):
            continue
        if target_view is not None and row_view != target_view:
            continue
        value = row.get("result")
        if value is None or str(value).strip().casefold() in REPORT_UNAVAILABLE_VALUES:
            return None
        return str(value).strip()
    return None


def normalized_report_value(value):
    return " ".join(str(value).replace("_", " ").replace("-", " ").split()).casefold()


def curve_report_phrase(value):
    if value is None:
        return None
    normalized = normalized_report_value(value)
    if "normal" in normalized:
        return "within normal limits"
    if "increase" in normalized or "accentuat" in normalized or "deep" in normalized:
        return "increased"
    if "decrease" in normalized or "reduc" in normalized or "flat" in normalized:
        return "decreased"
    if "revers" in normalized:
        return "reversed"
    return str(value).strip().lower()


def crowding_report_phrase(value):
    if value is None:
        return None
    normalized = normalized_report_value(value)
    if ("none" in normalized or normalized.startswith("no ")) and "mild" in normalized:
        return "none-to-mild"
    if "mild" in normalized and "moderate" in normalized:
        return "mild-to-moderate"
    if "moderate" in normalized and "severe" in normalized:
        return "moderate-to-severe"
    if normalized in {"none", "no", "absent"}:
        return "no"
    if "severe" in normalized:
        return "severe"
    if "moderate" in normalized:
        return "moderate"
    if "mild" in normalized or "slight" in normalized:
        return "mild"
    return str(value).strip().lower()


def generate_rule_based_report_legacy(case_id, summary_rows):
    sentences = []
    unavailable = []

    # Sagittal molar relationship: the corpus usually reports the right side first.
    right_molar = summary_result(summary_rows, "Molar relationship", "right")
    left_molar = summary_result(summary_rows, "Molar relationship", "left")
    if right_molar and left_molar:
        sentences.append(
            f"Sagittally, there is a {right_molar} molar relationship on the right "
            f"and a {left_molar} molar relationship on the left."
        )
    elif right_molar or left_molar:
        side = "right" if right_molar else "left"
        value = right_molar or left_molar
        sentences.append(f"Sagittally, there is a {value} molar relationship on the {side}.")
        unavailable.append(f"{('left' if right_molar else 'right')} molar relationship")
    else:
        unavailable.append("bilateral molar relationship")

    # Transverse bite wording follows the most frequent normal/crossbite/scissor-bite forms.
    transverse = summary_result(summary_rows, "Transverse Bite")
    if transverse:
        normalized = normalized_report_value(transverse)
        if "normal" in normalized:
            sentences.append("The patient presents a transverse constriction of the maxilla in the absence of crossbite.")
        elif "scissor" in normalized:
            sentences.append("Transversely, a scissor bite is present.")
        elif "cross" in normalized:
            sentences.append("Transversely, a crossbite is present.")
        else:
            sentences.append(f"Transversely, the predicted relationship is {transverse.lower()}.")
    else:
        unavailable.append("transverse bite")

    # Vertical relationship mapping matches the recurrent corpus terminology.
    overbite = summary_result(summary_rows, "Overbite")
    if overbite:
        normalized = normalized_report_value(overbite)
        if "open" in normalized:
            sentences.append("From a vertical standpoint, an anterior open bite is present.")
        elif "increase" in normalized or "deep" in normalized:
            sentences.append("From a vertical standpoint, there is a deep bite.")
        elif "decrease" in normalized or "reduc" in normalized or "inverted" in normalized:
            sentences.append("From a vertical standpoint, the overbite is reduced.")
        elif "normal" in normalized:
            sentences.append("From a vertical standpoint, the overbite is within normal limits.")
        else:
            sentences.append(f"From a vertical standpoint, the overbite is {overbite.lower()}.")
    else:
        unavailable.append("overbite")

    midline = summary_result(summary_rows, "Midline", "front")
    if midline:
        normalized = normalized_report_value(midline)
        if "center" in normalized or "coincident" in normalized or "align" in normalized:
            sentences.append("The midlines are centered with each other.")
        elif "slight" in normalized or "mild" in normalized:
            sentences.append("The midlines are slightly deviated.")
        elif "deviat" in normalized or "discrep" in normalized:
            sentences.append("The dental midlines are deviated relative to each other.")
        else:
            sentences.append(f"The dental midline relationship is {midline.lower()}.")
    else:
        unavailable.append("dental midlines")

    spee = curve_report_phrase(summary_result(summary_rows, "Curve of Spee"))
    wilson = curve_report_phrase(summary_result(summary_rows, "Curve of Wilson"))
    if spee and wilson:
        if spee == wilson:
            verb = "are"
            sentences.append(f"The Curve of Spee and the Curve of Wilson {verb} {spee}.")
        else:
            sentences.append(
                f"The Curve of Spee is {spee}, while the Curve of Wilson is {wilson}."
            )
    elif spee:
        sentences.append(f"The Curve of Spee is {spee}.")
        unavailable.append("Curve of Wilson")
    elif wilson:
        sentences.append(f"The Curve of Wilson is {wilson}.")
        unavailable.append("Curve of Spee")
    else:
        unavailable.extend(["Curve of Spee", "Curve of Wilson"])

    missing_teeth = summary_result(summary_rows, "Missing teeth", "3D tooth masks")
    if missing_teeth:
        if normalized_report_value(missing_teeth).startswith("no missing"):
            sentences.append(
                "No missing teeth were identified among the 28 permanent teeth excluding third molars."
            )
        else:
            sentences.append(
                "The following permanent teeth were not identified in the 3D tooth-mask output: "
                f"{missing_teeth}."
            )
    else:
        unavailable.append("missing teeth")

    upper_crowding = crowding_report_phrase(summary_result(summary_rows, "Crowding", "upper"))
    lower_crowding = crowding_report_phrase(summary_result(summary_rows, "Crowding", "lower"))
    if upper_crowding and lower_crowding:
        if upper_crowding == lower_crowding:
            if upper_crowding == "no":
                sentences.append("No crowding is present in the upper and lower arches.")
            else:
                sentences.append(
                    f"{upper_crowding.capitalize()} crowding is present in both the upper and lower arches."
                )
        else:
            upper_text = "no" if upper_crowding == "no" else upper_crowding
            lower_text = "no" if lower_crowding == "no" else lower_crowding
            sentences.append(
                f"There is {upper_text} crowding in the upper arch and {lower_text} crowding "
                "in the lower arch."
            )
    elif upper_crowding or lower_crowding:
        arch = "upper" if upper_crowding else "lower"
        value = upper_crowding or lower_crowding
        if value == "no":
            sentences.append(f"No crowding is present in the {arch} arch.")
        else:
            sentences.append(f"{value.capitalize()} crowding is present in the {arch} arch.")
        unavailable.append(f"{('lower' if upper_crowding else 'upper')} arch crowding")
    else:
        unavailable.append("upper and lower arch crowding")

    if unavailable:
        sentences.append(
            "The following findings could not be assessed from the available inference output: "
            + ", ".join(unavailable)
            + "."
        )
    body = " ".join(sentences) if sentences else "No reportable inference findings were available."
    return f"CASE_ID: {case_id}\n\n{body}"


# Keep the original rule available for an in-notebook, same-prediction comparison.


def canonical_relationship_phrase(value):
    """Map classifier labels to the word order most frequent in the reference corpus."""
    normalized = normalized_report_value(value)
    if "class ii" in normalized and ("edge to edge" in normalized or "end to end" in normalized):
        return "end-to-end Class II"
    if "class ii" in normalized and "full" in normalized:
        return "full Class II"
    if "class iii" in normalized:
        return "Class III"
    if "class ii" in normalized:
        return "Class II"
    if "class i" in normalized:
        return "Class I"
    return str(value).strip()


def optimized_crowding_phrase(value):
    """Use corpus-native severity wording while retaining conservative interval labels."""
    if value is None:
        return None
    normalized = normalized_report_value(value)
    if ("none" in normalized or normalized.startswith("no ")) and "mild" in normalized:
        # Preserve the classifier's intentionally broad None/Mild output.
        return "none-to-mild"
    if "mild" in normalized and "moderate" in normalized:
        return "mild/moderate"
    if "moderate" in normalized and "severe" in normalized:
        return "moderate/severe"
    if normalized in {"none", "no", "absent"}:
        return "no"
    if "severe" in normalized:
        return "severe"
    if "moderate" in normalized:
        return "moderate"
    if "mild" in normalized or "slight" in normalized:
        return "mild"
    return str(value).strip().lower()



def relationship_article(phrase):
    return "an" if str(phrase).lower().startswith(("end-to-end", "edge-to-edge")) else "a"


def canine_report_value(value):
    if value is None:
        return None, False
    normalized = normalized_report_value(value)
    cannot_assess = "cannot be assessed" in normalized or "not assessable" in normalized
    return (None, True) if cannot_assess else (canonical_relationship_phrase(value), False)


def sagittal_side_clause(side, molar, canine):
    molar_phrase = canonical_relationship_phrase(molar) if molar else None
    canine_phrase, canine_cannot = canine_report_value(canine)
    if molar_phrase and canine_phrase and molar_phrase == canine_phrase:
        clause = (
            f"{relationship_article(molar_phrase)} {molar_phrase} molar and canine relationship "
            f"on the {side}"
        )
    elif molar_phrase and canine_phrase:
        clause = (
            f"{relationship_article(molar_phrase)} {molar_phrase} molar relationship and "
            f"{relationship_article(canine_phrase)} {canine_phrase} canine relationship on the {side}"
        )
    elif molar_phrase and canine_cannot:
        clause = (
            f"{relationship_article(molar_phrase)} {molar_phrase} molar relationship, "
            f"canine not assessable, on the {side}"
        )
    elif molar_phrase:
        clause = f"{relationship_article(molar_phrase)} {molar_phrase} molar relationship on the {side}"
    elif canine_phrase:
        clause = f"{relationship_article(canine_phrase)} {canine_phrase} canine relationship on the {side}"
    elif canine_cannot:
        clause = f"a canine relationship that is not assessable on the {side}"
    else:
        clause = None
    return clause, canine_cannot


def missing_teeth_report_sentence(value):
    if value is None or normalized_report_value(value).startswith("no missing"):
        return None
    fdis = []
    for token in __import__("re").findall(r"\b[1-4][1-8]\b", str(value)):
        if token not in fdis:
            fdis.append(token)
    if not fdis:
        return None
    return f"Teeth {'-'.join(fdis)} are not present in the arch."


def generate_rule_based_report(case_id, summary_rows):
    """Generate a deterministic report using corpus-matched order and phrasing only."""
    sentences = []
    unavailable = []
    missing_teeth_value = summary_result(summary_rows, "Missing teeth", "3D tooth masks")
    # 1) Transverse and 2) vertical findings usually open the combined reports.
    transverse = summary_result(summary_rows, "Transverse Bite")
    if transverse:
        normalized = normalized_report_value(transverse)
        if "normal" in normalized:
            sentences.append(
                "The patient presents a transverse constriction of the maxilla in the absence of "
                "crossbite; the inter-arch relationships are within normal limits."
            )
        elif "scissor" in normalized:
            sentences.append("Transversely, a scissor bite is present.")
        elif "cross" in normalized:
            sentences.append(
                "The patient presents a transverse constriction of the maxilla with crossbite."
            )
        else:
            sentences.append(f"Transversely, the inter-arch relationship is {transverse.lower()}.")
    else:
        unavailable.append("transverse relationship")

    overbite = summary_result(summary_rows, "Overbite")
    if overbite:
        normalized = normalized_report_value(overbite)
        if "open" in normalized:
            sentences.append("From a vertical standpoint, an anterior open bite is present.")
        elif "increase" in normalized or "deep" in normalized:
            sentences.append("From a vertical standpoint, there is a deep bite.")
        elif "decrease" in normalized or "reduc" in normalized or "inverted" in normalized:
            sentences.append("From a vertical standpoint, the overbite is reduced.")
        elif "normal" in normalized:
            sentences.append("From a vertical standpoint, the overbite is within normal limits.")
        else:
            sentences.append(f"From a vertical standpoint, the overbite is {overbite.lower()}.")
    else:
        unavailable.append("vertical relationship")

    # Missing teeth are mentioned before sagittal findings in the common report order.
    missing_sentence = missing_teeth_report_sentence(missing_teeth_value)
    if missing_sentence:
        sentences.append(missing_sentence)

    # 3) Sagittal molar and canine relationships are usually reported in one sentence.
    right_molar = summary_result(summary_rows, "Molar relationship", "right")
    left_molar = summary_result(summary_rows, "Molar relationship", "left")
    right_canine = summary_result(summary_rows, "Canine relationship", "right")
    left_canine = summary_result(summary_rows, "Canine relationship", "left")
    right_clause, right_canine_cannot = sagittal_side_clause("right", right_molar, right_canine)
    left_clause, left_canine_cannot = sagittal_side_clause("left", left_molar, left_canine)
    if right_clause and left_clause:
        sentences.append(f"Sagittally, there is {right_clause}, and {left_clause}.")
    elif right_clause or left_clause:
        sentences.append(f"Sagittally, there is {right_clause or left_clause}.")
        unavailable.append(f"{('left' if right_clause else 'right')} sagittal relationship")
    else:
        unavailable.append("bilateral sagittal relationship")
    # 3b) Raw-STL overjet classifier; wording follows the most recurrent combined-report forms.
    overjet = summary_result(summary_rows, "Overjet")
    if overjet:
        normalized = normalized_report_value(overjet)
        if "normal" in normalized:
            sentences.append("The overjet is within normal limits.")
        elif "deep" in normalized or "increase" in normalized:
            sentences.append("The overjet is increased.")
        elif "negative" in normalized or "reduc" in normalized:
            sentences.append("The overjet is negative, consistent with an anterior crossbite.")
        else:
            sentences.append(f"The overjet is {overjet.lower()}.")
    else:
        unavailable.append("overjet")

    # 4) Dental midlines.
    midline = summary_result(summary_rows, "Midline", "front")
    if midline:
        normalized = normalized_report_value(midline)
        if "center" in normalized or "coincident" in normalized or "align" in normalized:
            sentences.append("The dental midlines are centered with each other.")
        elif "slight" in normalized or "mild" in normalized:
            sentences.append("The dental midlines are slightly deviated relative to each other.")
        elif "deviat" in normalized or "discrep" in normalized:
            sentences.append("The dental midlines are deviated relative to each other.")
        else:
            sentences.append(f"The dental midline relationship is {midline.lower()}.")
    else:
        unavailable.append("dental midlines")

    # 5) Curves, expressed as lower-arch curves to match common IOS report wording.
    spee = curve_report_phrase(summary_result(summary_rows, "Curve of Spee"))
    wilson = curve_report_phrase(summary_result(summary_rows, "Curve of Wilson"))
    if spee and wilson:
        if spee == wilson:
            sentences.append(f"The lower Curve of Spee and the lower Curve of Wilson are {spee}.")
        else:
            sentences.append(
                f"The lower Curve of Spee is {spee}, while the lower Curve of Wilson is {wilson}."
            )
    elif spee:
        sentences.append(f"The lower Curve of Spee is {spee}.")
        unavailable.append("Curve of Wilson")
    elif wilson:
        sentences.append(f"The lower Curve of Wilson is {wilson}.")
        unavailable.append("Curve of Spee")
    else:
        unavailable.extend(["Curve of Spee", "Curve of Wilson"])

    # 6) Crowding is the usual closing orthodontic finding.
    upper = optimized_crowding_phrase(summary_result(summary_rows, "Crowding", "upper"))
    lower = optimized_crowding_phrase(summary_result(summary_rows, "Crowding", "lower"))
    if upper and lower:
        if upper == lower:
            if upper == "no":
                sentences.append("No crowding is present in the upper and lower arches.")
            else:
                sentences.append(f"{upper.capitalize()} crowding is present in the upper and lower arches.")
        elif upper == "no":
            sentences.append(f"No crowding is present in the upper arch and {lower} crowding in the lower arch.")
        elif lower == "no":
            sentences.append(f"{upper.capitalize()} crowding is present in the upper arch and absent in the lower arch.")
        else:
            sentences.append(f"There is {upper} crowding in the upper arch and {lower} crowding in the lower arch.")
    elif upper or lower:
        arch = "upper" if upper else "lower"
        value = upper or lower
        if value == "no":
            sentences.append(f"No crowding is present in the {arch} arch.")
        else:
            sentences.append(f"{value.capitalize()} crowding is present in the {arch} arch.")
        unavailable.append(f"{('lower' if upper else 'upper')} arch crowding")
    else:
        unavailable.append("upper and lower arch crowding")


    # Only genuinely unavailable model findings receive the short transparency statement.
    if unavailable:
        sentences.append("The remaining unavailable findings could not be assessed reliably.")
    body = " ".join(sentences) if sentences else "No reportable inference findings were available."
    # Grand Challenge evaluates the report value verbatim. Keep identifiers outside it.
    return body


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


for required in [
    CASE_ROOT,
    INPUT_DIR,
    UPPER_STL,
    LOWER_STL,
    VIEW_WEIGHT,
    UPPER_CROWDING_WEIGHT,
    LOWER_CROWDING_WEIGHT,
    TOOTHSEG_CHECKPOINT,
    TOOTHSEG_ROI_MODULE,
    TEETHLAND_INSTSEG_WEIGHT,
    TEETHLAND_LANDMARK_WEIGHT,
    BASE_CONFIG,
]:
    require(required)

sys.path.insert(0, str(TOOTHSEG_RUNTIME))
from toothseg_primary_arch_roi import select_primary_arch_roi


def imagenet_transform(image_size):
    return v2.Compose(
        [
            v2.Resize((int(image_size), int(image_size))),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_view_model(model_name, num_classes):
    if model_name in {"mobilenet_v3_small", "mobilenet_v3_small_bite2text_adapted"}:
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.35, inplace=True), nn.Linear(features, num_classes))
    elif model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.45, inplace=True), nn.Linear(features, num_classes))
    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported view model: {model_name}")
    return model



def infer_views(weight_path, image_paths, images, use_autocast):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    model_name = checkpoint.get("architecture", checkpoint.get("model_name", "mobilenet_v3_small"))
    transform = imagenet_transform(int(checkpoint.get("image_size", 224)))
    model = build_view_model(model_name, len(classes))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(DEVICE).eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), VIEW_INFERENCE_BATCH_SIZE):
            batch_paths = image_paths[start : start + VIEW_INFERENCE_BATCH_SIZE]
            batch = torch.stack([transform(images[path]) for path in batch_paths]).to(DEVICE)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_autocast and DEVICE.type == "cuda",
            ):
                probabilities = model(batch).softmax(dim=1).float().cpu()
            rows.extend(probabilities.unbind(0))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return classes, torch.stack(rows), model_name, checkpoint



def assign_required_intraoral_views(classes, probability_matrix, image_paths, required_views=REQUIRED_INTRAORAL_VIEWS):
    """Select one unique image per required view from any folder containing >=5 IOPs.

    A rectangular Hungarian assignment maximizes the joint log probability over
    all images without factorial enumeration. Exactly five images are retained;
    additional images remain auditable but are excluded downstream.
    """
    class_to_index = {name: index for index, name in enumerate(classes)}
    n_images = len(image_paths)
    n_views = len(required_views)
    if n_images < n_views:
        raise RuntimeError(
            f"Expected at least {n_views} intraoral images for forced 5-view assignment, found {n_images}"
        )

    missing_classes = [view for view in required_views if view not in class_to_index]
    if missing_classes:
        fallback = {
            view: {
                "confidence": 0.0,
                "index": index,
                "path": image_paths[index],
                "mode": "fallback first-five sorted order; missing classifier classes: "
                + ", ".join(missing_classes),
            }
            for index, view in enumerate(required_views)
        }
        return fallback, f"fallback first-five sorted order ({n_images} candidates)"

    try:
        if probability_matrix.ndim != 2 or probability_matrix.shape[0] != n_images:
            raise ValueError(
                f"Malformed view probability matrix: {tuple(probability_matrix.shape)} for {n_images} images"
            )
        class_indices = [class_to_index[view] for view in required_views]
        # score shape: required views x candidate images. linear_sum_assignment
        # chooses every view row and five distinct image columns.
        score = probability_matrix[:, class_indices].clamp_min(1e-12).log().T.cpu().numpy()
        view_rows, image_columns = linear_sum_assignment(-score)
        if len(view_rows) != n_views or set(view_rows.tolist()) != set(range(n_views)):
            raise RuntimeError("Hungarian assignment did not cover all five required views")
        image_for_view = {int(view_row): int(image_index) for view_row, image_index in zip(view_rows, image_columns)}
        assignment = {}
        for view_row, view in enumerate(required_views):
            image_index = image_for_view[view_row]
            class_index = class_to_index[view]
            assignment[view] = {
                "confidence": float(probability_matrix[image_index, class_index]),
                "index": image_index,
                "path": image_paths[image_index],
                "mode": f"global Hungarian one-to-one assignment (selected 5 of {n_images})",
            }
        return assignment, f"global Hungarian one-to-one assignment (selected 5 of {n_images})"
    except Exception as error:
        fallback = {
            view: {
                "confidence": 0.0,
                "index": index,
                "path": image_paths[index],
                "mode": f"fallback first-five sorted order after {type(error).__name__}: {error}",
            }
            for index, view in enumerate(required_views)
        }
        return fallback, f"fallback first-five sorted order ({n_images} candidates)"


def mask_center(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    return np.asarray([xs.mean(), ys.mean()], dtype=np.float64)


def load_boundary_instances(npz_path, minimum_confidence=0.0):
    instances = []
    with np.load(npz_path, allow_pickle=False) as data:
        height, width = map(int, data["image_shape"])
        for index, (fdi, confidence, crop_box) in enumerate(
            zip(data["fdis"], data["confidences"], data["crop_boxes"])
        ):
            confidence = float(confidence)
            if confidence < minimum_confidence:
                continue
            x1, y1, x2, y2 = map(int, crop_box)
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = data[f"mask_{index}"].astype(bool)
            if mask.any():
                instances.append(
                    {
                        "fdi": int(fdi),
                        "confidence": confidence,
                        "mask": mask,
                        "center": mask_center(mask),
                    }
                )
    return instances


def save_yolo_boundary_npz(result, output_path):
    height, width = map(int, result.orig_shape)
    records = []
    if result.boxes is not None and result.masks is not None:
        class_indices = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        for class_index, confidence, mask_tensor in zip(
            class_indices, confidences, result.masks.data
        ):
            mask = (
                yolo_ops.scale_masks(
                    mask_tensor[None, None], (height, width), padding=True, mode="nearest"
                )[0, 0]
                .gt(0.5)
                .cpu()
                .numpy()
            )
            if mask.any():
                ys, xs = np.nonzero(mask)
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                records.append(
                    {
                        "class_index": int(class_index),
                        "fdi": int(result.names[int(class_index)]),
                        "confidence": float(confidence),
                        "crop_box": [x1, y1, x2, y2],
                        "mask_crop": mask[y1:y2, x1:x2].astype(np.uint8),
                    }
                )
    arrays = {
        "class_indices": np.asarray([item["class_index"] for item in records], dtype=np.int16),
        "fdis": np.asarray([item["fdi"] for item in records], dtype=np.int16),
        "confidences": np.asarray([item["confidence"] for item in records], dtype=np.float32),
        "bbox_scores": np.asarray([item["confidence"] for item in records], dtype=np.float32),
        "crop_boxes": np.asarray([item["crop_box"] for item in records], dtype=np.int32).reshape(-1, 4),
        "image_shape": np.asarray([height, width], dtype=np.int32),
    }
    for index, item in enumerate(records):
        arrays[f"mask_{index}"] = item["mask_crop"]
    np.savez_compressed(output_path, **arrays)


def run_yolo_boundary_inference(image_paths, output_dir):
    model = YOLO(str(TOOTHSEG_CHECKPOINT))
    output_paths = {}
    for index, image_path in enumerate(image_paths):
        result = model.predict(
            source=str(image_path),
            imgsz=TOOTHSEG_IMAGE_SIZE,
            conf=SEGMENTATION_SCORE_THRESHOLD,
            iou=0.7,
            max_det=64,
            device=0 if DEVICE.type == "cuda" else "cpu",
            retina_masks=False,
            verbose=False,
        )[0]
        output_path = output_dir / f"{index:03d}_{image_path.name}.npz"
        save_yolo_boundary_npz(result, output_path)
        output_paths[image_path] = output_path
    del model, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_paths


def mask_iou(first, second):
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return intersection / union if union else 0.0


def select_distinct_pair(first_candidates, second_candidates, reference_axis=None):
    candidates = []
    for first in first_candidates:
        for second in second_candidates:
            vector = second["center"] - first["center"]
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6:
                continue
            overlap = mask_iou(first["mask"], second["mask"])
            radii = [
                math.sqrt(max(1, int(item["mask"].sum())) / math.pi)
                for item in (first, second)
            ]
            if overlap >= 0.35 or distance <= 0.40 * float(np.mean(radii)):
                continue
            direction = vector / distance
            alignment = 1.0
            if reference_axis is not None:
                alignment = float(np.dot(direction, reference_axis))
                if alignment <= 0.20:
                    continue
            score = (
                math.log(max(first["confidence"], 1e-6))
                + math.log(max(second["confidence"], 1e-6))
                - 3.0 * overlap
                + 0.25 * alignment
            )
            candidates.append((score, first, second))
    return max(candidates, key=lambda row: row[0])[1:] if candidates else None


def projected_width(mask, axis):
    ys, xs = np.nonzero(mask)
    projection = np.column_stack([xs, ys]).astype(np.float64) @ axis
    return float(np.percentile(projection, 95) - np.percentile(projection, 5))


def calculate_midline(front_path, front_probability, npz_path):
    all_instances = load_boundary_instances(npz_path, MIDLINE_SCORE_THRESHOLD)
    target_candidates = {
        fdi: [item for item in all_instances if item["fdi"] == fdi] for fdi in TARGET_FDIS
    }
    upper_pair = select_distinct_pair(target_candidates[11], target_candidates[21])
    lower_pair = None
    if upper_pair is not None:
        reference_axis = upper_pair[1]["center"] - upper_pair[0]["center"]
        reference_axis /= np.linalg.norm(reference_axis)
        lower_pair = select_distinct_pair(
            target_candidates[41], target_candidates[31], reference_axis=reference_axis
        )

    incisors = {}
    if upper_pair is not None:
        incisors.update({11: upper_pair[0], 21: upper_pair[1]})
    if lower_pair is not None:
        incisors.update({41: lower_pair[0], 31: lower_pair[1]})
    missing_fdis = [fdi for fdi in TARGET_FDIS if fdi not in incisors]
    result = {
        "classification": "Unknown",
        "front_image": front_path.name,
        "front_view_confidence": float(front_probability),
        "missing_fdis": missing_fdis,
        "segmentation_score_threshold": MIDLINE_SCORE_THRESHOLD,
        "centered_threshold_tooth_width": CENTERED_THRESHOLD_TOOTH_WIDTH,
        "slightly_deviated_threshold_tooth_width": SLIGHTLY_DEVIATED_THRESHOLD_TOOTH_WIDTH,
        "incisors": {
            str(fdi): {
                "confidence": float(item["confidence"]),
                "candidate_count": len(target_candidates[fdi]),
                "mask_area_px": int(item["mask"].sum()),
            }
            for fdi, item in sorted(incisors.items())
        },
    }
    if missing_fdis:
        return result

    c11, c21 = incisors[11]["center"], incisors[21]["center"]
    c41, c31 = incisors[41]["center"], incisors[31]["center"]
    upper_axis = c21 - c11
    lower_axis = c31 - c41
    upper_axis /= np.linalg.norm(upper_axis)
    lower_axis /= np.linalg.norm(lower_axis)
    if np.dot(upper_axis, lower_axis) < 0:
        lower_axis *= -1
    horizontal_axis = upper_axis + lower_axis
    horizontal_axis /= np.linalg.norm(horizontal_axis)
    upper_midpoint = (c11 + c21) / 2
    lower_midpoint = (c41 + c31) / 2
    signed_offset_px = float(np.dot(lower_midpoint - upper_midpoint, horizontal_axis))
    reference_width_px = float(
        np.mean([projected_width(incisors[fdi]["mask"], horizontal_axis) for fdi in TARGET_FDIS])
    )
    signed_ratio = signed_offset_px / max(reference_width_px, 1e-6)
    absolute_ratio = abs(signed_ratio)
    if absolute_ratio <= CENTERED_THRESHOLD_TOOTH_WIDTH:
        classification = "Centered"
    elif absolute_ratio <= SLIGHTLY_DEVIATED_THRESHOLD_TOOTH_WIDTH:
        classification = "Slightly deviated"
    else:
        classification = "Deviated"
    direction = (
        "patient left"
        if signed_ratio > 0
        else "patient right"
        if signed_ratio < 0
        else "none"
    )
    pair_angle = float(
        np.degrees(np.arccos(np.clip(np.dot(upper_axis, lower_axis), -1.0, 1.0)))
    )
    result.update(
        {
            "classification": classification,
            "signed_offset_px": signed_offset_px,
            "signed_offset_tooth_width": float(signed_ratio),
            "absolute_offset_tooth_width": float(absolute_ratio),
            "lower_midline_shift_direction": direction,
            "reference_incisor_width_px": reference_width_px,
            "upper_lower_pair_angle_deg": pair_angle,
        }
    )
    return result


def build_crowding_model(model_name):
    if model_name == "efficientnet_v2_s":
        model, dropout = efficientnet_v2_s(weights=None), 0.25
    elif model_name == "efficientnet_b3":
        model, dropout = efficientnet_b3(weights=None), 0.45
    elif model_name == "efficientnet_b0":
        model, dropout = efficientnet_b0(weights=None), 0.35
    else:
        raise ValueError(f"Unsupported crowding model: {model_name}")
    feature_dim = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(feature_dim, 3))
    return model




def infer_crowding(arch, image_path, image, view_probability, npz_path, weight_path):
    roi_status = "tooth ROI"
    try:
        roi = select_primary_arch_roi(npz_path, arch, minimum_teeth=TOOTHSEG_MINIMUM_TEETH)
        input_image = image.crop(roi["crop_box"])
        roi_payload = {
            "status": roi_status,
            "crop_box": [int(value) for value in roi["crop_box"]],
            "crop_fraction": float(roi["crop_fraction"]),
            "selected_teeth": int(roi["selected_detection_count"]),
            "selected_fdis": [int(value) for value in roi["selected_fdis"]],
        }
    except Exception as error:
        input_image = image
        roi_payload = {
            "status": f"original fallback: {type(error).__name__}: {error}",
            "crop_box": [0, 0, image.width, image.height],
            "crop_fraction": 1.0,
            "selected_teeth": 0,
            "selected_fdis": [],
        }

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    image_size = int(checkpoint.get("image_size", 320))
    model = build_crowding_model(model_name)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError(f"No model/state_dict in {weight_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    tensor = imagenet_transform(image_size)(input_image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"
    ):
        probabilities = model(tensor).softmax(dim=1)[0].float().cpu()
    predicted_index = int(probabilities.argmax())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raw_classification = CROWDING_NAMES[predicted_index]
    display_classification = CROWDING_DISPLAY_NAMES.get(raw_classification, raw_classification)
    return {
        "classification": display_classification,
        "raw_classification": raw_classification,
        "complexity_level": predicted_index + 1,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            CROWDING_DISPLAY_NAMES.get(name, name): float(probabilities[index])
            for index, name in enumerate(CROWDING_NAMES)
        },
        "raw_probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(CROWDING_NAMES)
        },
        "image": image_path.name,
        "view_confidence": float(view_probability),
        "model_name": model_name,
        "model_image_size": image_size,
        "model_epoch": checkpoint.get("epoch", checkpoint.get("best_epoch", "unknown")),
        "roi": roi_payload,
    }


image_paths = sorted(
    path
    for path in INPUT_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
    and not any(part.startswith(".") for part in path.relative_to(INPUT_DIR).parts)
)
if len(image_paths) < len(REQUIRED_INTRAORAL_VIEWS):
    raise RuntimeError(
        f"Expected at least {len(REQUIRED_INTRAORAL_VIEWS)} intraoral photographs under {INPUT_DIR}, "
        f"found {len(image_paths)}: {[path.name for path in image_paths]}"
    )
images = {path: Image.open(path).convert("RGB") for path in image_paths}

# View selection: accept five or more photographs. Classify every candidate,
# then select exactly one unique left/right/upper/front/lower image by a global
# rectangular Hungarian assignment. Extra images are audited and skipped.
crowding_classes, crowding_matrix, crowding_view_model, crowding_view_checkpoint = infer_views(
    VIEW_WEIGHT, image_paths, images, use_autocast=True
)
forced_views, forced_view_assignment_mode = assign_required_intraoral_views(
    crowding_classes, crowding_matrix, image_paths
)
selected_view_paths = {item["path"] for item in forced_views.values()}
unselected_image_paths = [path for path in image_paths if path not in selected_view_paths]
view_assignment_rows = [
    {
        "view": view,
        "selected_image": item["path"].name,
        "confidence": item["confidence"],
        "assignment_mode": item["mode"],
    }
    for view, item in forced_views.items()
]
# Release decoded extras after view inference; downstream branches see only the
# selected five images, and only front/upper/lower require 2D tooth segmentation.
images = {path: images[path] for path in selected_view_paths}
front_path = forced_views["front"]["path"]
front_index = forced_views["front"]["index"]
front_class_index = crowding_classes.index("front") if "front" in crowding_classes else None
midline_assignment_mode = forced_view_assignment_mode
midline_view_model = crowding_view_model
midline_view_checkpoint = crowding_view_checkpoint

best_by_arch = {
    arch: (forced_views[arch]["confidence"], forced_views[arch]["index"], forced_views[arch]["path"])
    for arch in ("upper", "lower")
}
selected_paths = [front_path] + [best_by_arch[arch][2] for arch in ("upper", "lower")]
unique_paths = list(dict.fromkeys(selected_paths))
path_to_npz = {}

with tempfile.TemporaryDirectory(prefix=f"{CASE_ID}_midline_crowding_") as temporary_text:
    temporary = Path(temporary_text)
    output_dir = temporary / "masks"
    output_dir.mkdir()
    path_to_npz = run_yolo_boundary_inference(unique_paths, output_dir)
    midline = calculate_midline(
        front_path,
        float(forced_views["front"]["confidence"]),
        path_to_npz[front_path],
    )
    midline.update(
        {
            "view_assignment_mode": midline_assignment_mode,
            "view_model": midline_view_model,
            "view_model_epoch": midline_view_checkpoint.get("best_epoch", "unknown"),
        }
    )
    if midline["classification"] == "Unknown":
        midline["raw_classification"] = "Unknown"
        midline["classification"] = "Slightly deviated"
        midline["fallback_from_unknown"] = True

    crowding = {"upper": None, "lower": None}
    for arch, weight_path in (("upper", UPPER_CROWDING_WEIGHT), ("lower", LOWER_CROWDING_WEIGHT)):
        if arch not in best_by_arch:
            continue
        confidence, index, path = best_by_arch[arch]
        crowding[arch] = infer_crowding(
            arch,
            path,
            images[path],
            confidence,
            path_to_npz[path],
            weight_path,
        )

warnings_list = []
if midline.get("fallback_from_unknown", False):
    warnings_list.append(
        f"Midline fallback applied because required FDI masks were missing: {midline['missing_fdis']}"
    )
for arch in ("upper", "lower"):
    if crowding[arch] is None:
        warnings_list.append(f"No {arch} image was selected by the crowding view classifier")
three_d_rows, area_occlusion, canine_occlusion, tooth_presence = infer_3d_table_rows(CASE_ID)

output = {
    "case_id": CASE_ID,
    "input_dir": str(INPUT_DIR),
    "upper_stl": str(UPPER_STL),
    "lower_stl": str(LOWER_STL),
    "fresh_3dteethland": True,
    "intraoral_image_count": len(image_paths),
    "selected_intraoral_image_count": len(selected_view_paths),
    "unselected_intraoral_images": [path.name for path in unselected_image_paths],
    "midline": midline,
    "crowding": crowding,
    "area_occlusion": area_occlusion,
    "canine_occlusion": canine_occlusion,
    "tooth_presence": tooth_presence,
    "curve_models": "FDI tooth-mask PTv3 only",
    "forced_view_assignment": {
        view: {
            "image": item["path"].name,
            "confidence": item["confidence"],
            "mode": item["mode"],
        }
        for view, item in forced_views.items()
    },
    "shared_tooth_segmentation": {
        "checkpoint": str(TOOTHSEG_CHECKPOINT),
        "score_threshold": SEGMENTATION_SCORE_THRESHOLD,
        "temporary_outputs_deleted": True,
    },
    "warnings": warnings_list,
}
summary_rows = [
    {
        "item": "Midline",
        "view": "front",
        "image": midline["front_image"],
        "result": midline["classification"],
        "confidence": midline["front_view_confidence"],
        "detail": (
            f"offset={midline['signed_offset_tooth_width']:+.3f} tooth widths; "
            f"shift={midline['lower_midline_shift_direction']}"
            if "signed_offset_tooth_width" in midline
            else f"fallback from Unknown; missing FDI={midline['missing_fdis']}"
        ),
    }
]
for arch in ("upper", "lower"):
    item = crowding[arch]
    summary_rows.append(
        {
            "item": "Crowding",
            "view": arch,
            "image": item["image"] if item else None,
            "result": item["classification"] if item else "Unavailable",
            "confidence": item["confidence"] if item else None,
            "detail": (
                f"raw={item.get('raw_classification', '-')}; level={item['complexity_level']}; "
                f"ROI teeth={item['roi']['selected_teeth']}; ROI fraction={item['roi']['crop_fraction']:.1%}"
                if item
                else "No image selected"
            ),
        }
    )

summary_rows.extend(three_d_rows)

print(f"\n===== VIEW ASSIGNMENT: selected 5 of {len(image_paths)} IOPs =====")
display(pd.DataFrame(view_assignment_rows).style.format({"confidence": "{:.2%}"}, na_rep="-"))
if unselected_image_paths:
    print("Skipped extra IOPs:", ", ".join(path.name for path in unselected_image_paths))

print("\n===== SUMMARY TABLE =====")
summary_table = pd.DataFrame(summary_rows)
display(summary_table.style.format({"confidence": "{:.2%}"}, na_rep="-"))

optimized_report = generate_rule_based_report(CASE_ID, summary_rows)

print("\n===== CORPUS-OPTIMIZED RULE-BASED REPORT =====")
print(f"CASE_ID: {CASE_ID}")
print(optimized_report)

combined_report_path = BITE2TEXT_ROOT / "combined_reports_en_20260715" / f"{CASE_ID}.txt"
print("\n===== COMBINED REFERENCE REPORT =====")
print(f"CASE_ID: {CASE_ID}")
if combined_report_path.is_file():
    print(combined_report_path.read_text(encoding="utf-8", errors="replace").strip())
else:
    print(f"[Combined report not found: {combined_report_path}]")
