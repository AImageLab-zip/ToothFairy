#!/usr/bin/env python3
"""Official Grand Challenge inference entrypoint for Bite2Text PTv3.

The program consumes the official lower/upper IOS sockets, applies the same
paired IOS normalization and conservative post-correction used to create the
training data, samples 32,768 surface points per jaw, and runs the fixed PTv3
seven-head checkpoint.  It writes exactly one supported output field:

    /output/diagnostic-imaging-report.json -> {"report": "..."}

All paths can be overridden with environment variables for server-side smoke
tests.  The intraoral photograph socket is accepted but is not used by this
geometry-only checkpoint.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

# In development the entrypoint lives next to the Bits2Bites checkout; a
# container can instead set BITE2TEXT_POINTCEPT_ROOT explicitly.
_default_pointcept_root = Path(__file__).resolve().parents[1] / "Bits2Bites"
_pointcept_root = Path(
    os.getenv("BITE2TEXT_POINTCEPT_ROOT", str(_default_pointcept_root))
).expanduser()
if _pointcept_root.is_dir():
    sys.path.insert(0, str(_pointcept_root.resolve()))

from pointcept.datasets.transform import Compose
from pointcept.models import build_model
from pointcept.utils.config import Config

from prepare_ptv3_dataset import load_stl_triangles, sample_surface
from report_renderer import render_report as render_structured_report


HEAD_NAMES = (
    "right_molar_relation",
    "right_canine_relation",
    "left_molar_relation",
    "left_canine_relation",
    "overjet",
    "vertical_relation",
    "midline_relation",
)

RELATION_TEXT = {
    "class_i": "Class I",
    "class_ii_edge_to_edge": "edge-to-edge Class II",
    "class_ii_full": "full Class II",
    "class_ii_unspecified": "Class II",
    "class_iii": "Class III",
}
OVERJET_TEXT = {
    "normal": "within normal limits",
    "increased": "increased",
    "reduced": "reduced",
    "negative": "negative",
    "edge_to_edge": "edge-to-edge",
}
VERTICAL_SENTENCES = {
    "normal": "From a vertical standpoint, the overbite is within normal limits.",
    "increased": "From a vertical standpoint, the overbite is increased.",
    "reduced": "From a vertical standpoint, the overbite is reduced.",
    "deep_bite": "From a vertical standpoint, there is a deep bite.",
    "open_bite": "From a vertical standpoint, there is an open bite.",
}
MIDLINE_SENTENCES = {
    "coincident": "The dental midlines are coincident.",
    "slightly_deviated": "The dental midlines are slightly deviated.",
    "deviated": "The dental midlines are deviated.",
}

ROTATE_Y_180 = np.diag([-1.0, 1.0, -1.0])
ROTATE_Z_180 = np.diag([-1.0, -1.0, 1.0])
ROTATE_XZ_ARCH_TO_XY = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
)


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_single_file(*, preferred: Iterable[Path], locations: Iterable[Path]) -> Path:
    for path in preferred:
        if path.is_file():
            return path.resolve()
    matches: list[Path] = []
    for location in locations:
        for pattern in ("*.obj", "*.OBJ", "*.stl", "*.STL"):
            matches.extend(Path(path).resolve() for path in glob.glob(str(location / pattern)))
    unique = sorted(set(matches))
    if not unique:
        raise RuntimeError(f"No STL mesh found below {[str(path) for path in locations]}")
    if len(unique) > 1:
        print(f"Multiple IOS meshes found; using {unique[0].name}", flush=True)
    return unique[0]


def input_files(input_path: Path) -> tuple[Path, Path]:
    """Resolve the two official input sockets and return (upper, lower)."""
    manifest_path = input_path / "inputs.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        if not isinstance(payload, list):
            raise RuntimeError(f"{manifest_path} must contain a JSON list")
        slugs = {
            str((item.get("socket") or {}).get("slug") or "")
            for item in payload
            if isinstance(item, dict)
        }
        required = {"3d-lower-teeth-scan", "3d-upper-teeth-scan"}
        if not required.issubset(slugs):
            raise RuntimeError(f"Missing required IOS socket(s); found {sorted(slugs)}")

    lower = find_single_file(
        preferred=(
            input_path / "3d-lower-teeth-scan.obj",
            input_path / "3d-lower-teeth-scan.stl",
        ),
        locations=(
            input_path / "files" / "ios-lower",
            input_path / "ios-lower",
            input_path / "files" / "3d-lower-teeth-scan",
            input_path / "3d-lower-teeth-scan",
        ),
    )
    upper = find_single_file(
        preferred=(
            input_path / "3d-upper-teeth-scan.obj",
            input_path / "3d-upper-teeth-scan.stl",
        ),
        locations=(
            input_path / "files" / "ios-upper",
            input_path / "ios-upper",
            input_path / "files" / "3d-upper-teeth-scan",
            input_path / "3d-upper-teeth-scan",
        ),
    )
    return upper, lower


def load_triangle_mesh(source: Path) -> tuple[trimesh.Trimesh, str]:
    """Load an OBJ/STL mesh by validated content, not only by its suffix.

    Grand Challenge uses the interface's configured ``.obj`` path even when
    the uploaded file is a binary STL.  In that case an OBJ parser can return
    an empty mesh without raising, so every candidate must be validated.
    """
    suffix_type = source.suffix.lower().lstrip(".")
    candidates = tuple(dict.fromkeys((suffix_type, "obj", "stl")))
    failures: list[str] = []
    for file_type in candidates:
        if file_type not in {"obj", "stl"}:
            continue
        try:
            mesh = trimesh.load_mesh(
                str(source), file_type=file_type, force="mesh", process=False
            )
        except Exception as error:  # pragma: no cover - parser details vary
            failures.append(f"{file_type}: {type(error).__name__}: {error}")
            continue
        if not isinstance(mesh, trimesh.Trimesh):
            failures.append(f"{file_type}: returned {type(mesh).__name__}")
            continue
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            failures.append(f"{file_type}: no valid vertices")
            continue
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            failures.append(f"{file_type}: no triangular faces")
            continue
        return mesh, file_type
    raise RuntimeError(f"Could not parse triangle mesh {source}; attempts: {failures}")


def stage_mesh_as_stl(source: Path, destination: Path) -> None:
    """Stage an official mesh as a genuine STL file for IOS-Normalizer."""
    mesh, detected_type = load_triangle_mesh(source)
    if detected_type == "stl":
        shutil.copy2(source, destination)
    else:
        mesh.export(str(destination), file_type="stl")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Failed to convert {source} to STL")


def run_ios_normalizer(upper: Path, lower: Path, temporary_root: Path) -> tuple[Path, Path, str]:
    """Run paired-jaw IOS-Normalizer without changing the mounted input."""
    if os.getenv("BITE2TEXT_SKIP_NORMALIZATION", "0") == "1":
        return upper, lower, "normalization_skipped"

    normalizer_root = env_path("BITE2TEXT_NORMALIZER_ROOT", "/opt/ml/ios-normalizer")
    # Do not resolve this path: a venv's ``bin/python`` is commonly a symlink,
    # and executing its resolved system target would lose the venv packages.
    normalizer_python = Path(
        os.getenv("BITE2TEXT_NORMALIZER_PYTHON", sys.executable)
    ).expanduser().absolute()
    normalizer_checkpoint = env_path(
        "BITE2TEXT_NORMALIZER_CHECKPOINT", "/opt/ml/model/ios_normalizer_best.pt"
    )
    normalize_pair = Path(__file__).with_name("normalize_pair.py").resolve()
    for required in (normalizer_python, normalizer_checkpoint, normalize_pair):
        if not required.is_file():
            raise RuntimeError(f"Missing IOS-Normalizer dependency: {required}")

    patient_root = temporary_root / "input_pair"
    oriented_root = temporary_root / "oriented_pair"
    patient_root.mkdir()
    oriented_root.mkdir()
    staged_lower = patient_root / "ios_lower.stl"
    staged_upper = patient_root / "ios_upper.stl"
    stage_mesh_as_stl(lower, staged_lower)
    stage_mesh_as_stl(upper, staged_upper)
    command = [
        str(normalizer_python),
        str(normalize_pair),
        "--normalizer-root",
        str(normalizer_root),
        "--checkpoint",
        str(normalizer_checkpoint),
        "--lower",
        str(staged_lower),
        "--upper",
        str(staged_upper),
        "--output-dir",
        str(oriented_root),
        "--seed",
        str(int(os.getenv("BITE2TEXT_NORMALIZER_SEED", "20260809"))),
        "--device",
        os.getenv("BITE2TEXT_NORMALIZER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise RuntimeError(f"IOS-Normalizer failed with code {result.returncode}: {detail}")
    normalized_lower = oriented_root / "ios_lower_oriented.stl"
    normalized_upper = oriented_root / "ios_upper_oriented.stl"
    if not normalized_lower.is_file() or not normalized_upper.is_file():
        raise RuntimeError("IOS-Normalizer did not create both paired output meshes")
    summary = " | ".join(line.strip() for line in result.stdout.splitlines()[-4:] if line.strip())
    return normalized_upper, normalized_lower, summary


def load_mesh(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise RuntimeError(f"Invalid mesh: {path}")
    return mesh, vertices


def lower_front_direction(vertices: np.ndarray) -> tuple[float, float]:
    xy = vertices[:, :2]
    centered = xy - xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    width_vector = eigenvectors[:, -1]
    depth_vector = np.array([-width_vector[1], width_vector[0]])
    width = centered @ width_vector
    depth = centered @ depth_vector
    low_cut, high_cut = np.quantile(depth, [0.15, 0.85])

    def robust_width(values: np.ndarray) -> float:
        low, high = np.quantile(values, [0.05, 0.95])
        return float(high - low)

    low_width = robust_width(width[depth <= low_cut])
    high_width = robust_width(width[depth >= high_cut])
    if low_width <= 1e-8 or high_width <= 1e-8:
        raise RuntimeError("Could not infer lower-arch anterior direction")
    log_ratio = math.log(low_width / high_width)
    front = depth_vector if log_ratio > 0 else -depth_vector
    return math.degrees(math.atan2(front[0], front[1])), abs(log_ratio)


def infer_axis_layout(lower: np.ndarray, upper: np.ndarray) -> str:
    spans = []
    for vertices in (lower, upper):
        q05, q95 = np.quantile(vertices, [0.05, 0.95], axis=0)
        spans.append(q95 - q05)
    mean_span = np.mean(spans, axis=0)
    return "XY_arch_Z_vertical" if mean_span[1] >= mean_span[2] else "XZ_arch_Y_vertical"


def postcorrect_triangles(upper_path: Path, lower_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply the same conservative paired correction used by the training set."""
    upper_mesh, upper = load_mesh(upper_path)
    lower_mesh, lower = load_mesh(lower_path)
    actions: list[str] = []
    axis_layout = infer_axis_layout(lower, upper)
    if axis_layout == "XZ_arch_Y_vertical":
        lower = lower @ ROTATE_XZ_ARCH_TO_XY
        upper = upper @ ROTATE_XZ_ARCH_TO_XY
        actions.append("rotate_xz_arch_to_xy")

    vertical_before = float(np.median(upper[:, 2]) - np.median(lower[:, 2]))
    if vertical_before <= 0:
        lower = lower @ ROTATE_Y_180
        upper = upper @ ROTATE_Y_180
        actions.append("rotate_y_180_for_upper_positive_z")

    front_before, front_confidence = lower_front_direction(lower)
    if abs(front_before) >= 150.0:
        lower = lower @ ROTATE_Z_180
        upper = upper @ ROTATE_Z_180
        actions.append("rotate_z_180_for_anterior_positive_y")

    lower_mesh.vertices = lower
    upper_mesh.vertices = upper
    lower_triangles = np.asarray(lower_mesh.triangles, dtype=np.float32)
    upper_triangles = np.asarray(upper_mesh.triangles, dtype=np.float32)
    diagnostics = {
        "source_axis_layout": axis_layout,
        "post_correction_actions": actions,
        "vertical_gap_before": vertical_before,
        "vertical_gap_after": float(np.median(upper[:, 2]) - np.median(lower[:, 2])),
        "lower_front_angle_before": front_before,
        "lower_front_angle_after": lower_front_direction(lower)[0],
        "lower_front_shape_confidence": front_confidence,
    }
    return upper_triangles, lower_triangles, diagnostics


def sampled_coordinates(upper_triangles: np.ndarray, lower_triangles: np.ndarray) -> np.ndarray:
    points_per_jaw = int(os.getenv("BITE2TEXT_POINTS_PER_JAW", "32768"))
    if points_per_jaw < 1024:
        raise ValueError("BITE2TEXT_POINTS_PER_JAW must be at least 1024")
    seed = int(os.getenv("BITE2TEXT_SEED", "2026"))
    rng = np.random.default_rng(seed)
    upper = sample_surface(upper_triangles, points_per_jaw, rng)
    lower = sample_surface(lower_triangles, points_per_jaw, rng)
    coord = np.concatenate([upper, lower], axis=0).astype(np.float32, copy=False)
    if not np.isfinite(coord).all():
        raise RuntimeError("Surface sampling produced non-finite coordinates")
    return coord


def inference_sample(coord: np.ndarray, seed: int) -> dict[str, torch.Tensor]:
    """Reproduce the validation transform exactly, excluding unavailable labels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transform = Compose(
        [
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=["coord", "point_label_onehot"],
            ),
        ]
    )
    return transform(
        {
            "coord": coord,
            "point_label_onehot": np.zeros((coord.shape[0], 6), dtype=np.float32),
            "index_valid_keys": ["coord", "point_label_onehot"],
        }
    )


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"Missing PTv3 config/checkpoint: {config_path}, {checkpoint_path}")
    cfg = Config.fromfile(str(config_path))
    model = build_model(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    state_dict = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value) for key, value in source.items()
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def predict(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    head_vocabs: dict[str, list[str]],
    head_names: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, str], dict[str, float], int]:
    batch = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
    use_amp = os.getenv("BITE2TEXT_AMP", "0") == "1"
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            # The installed spconv build has no valid FP16 inference kernel for
            # some single-case sparse shapes. FP32 is the reliable default;
            # AMP remains opt-in for container builds that support that path.
            enabled=device.type == "cuda" and use_amp,
        ):
            logits = model(batch)["logits"]
    labels: dict[str, str] = {}
    confidences: dict[str, float] = {}
    if len(logits) != len(head_names):
        raise RuntimeError(f"Model emitted {len(logits)} heads for {len(head_names)} vocabularies")
    for head, values in zip(head_names, logits, strict=True):
        probabilities = values.float().softmax(dim=1)
        index = int(probabilities.argmax(dim=1).item())
        labels[head] = head_vocabs[head][index]
        confidences[head] = float(probabilities[0, index].item())
    return labels, confidences, int(sample["coord"].shape[0])


def render_report(labels: dict[str, str]) -> str:
    right = (
        f"{RELATION_TEXT[labels['right_molar_relation']]} molar relationship and "
        f"{RELATION_TEXT[labels['right_canine_relation']]} canine relationship on the right"
    )
    left = (
        f"{RELATION_TEXT[labels['left_molar_relation']]} molar relationship and "
        f"{RELATION_TEXT[labels['left_canine_relation']]} canine relationship on the left"
    )
    return " ".join(
        [
            f"From a sagittal standpoint, there is a {right}, and a {left}.",
            f"The overjet is {OVERJET_TEXT[labels['overjet']] }.",
            VERTICAL_SENTENCES[labels["vertical_relation"]],
            MIDLINE_SENTENCES[labels["midline_relation"]],
        ]
    )


def run() -> int:
    input_path = env_path("BITE2TEXT_INPUT_PATH", "/input")
    output_path = env_path("BITE2TEXT_OUTPUT_PATH", "/output")
    config_path = env_path("BITE2TEXT_CONFIG", "/opt/ml/model/config.py")
    checkpoint_path = env_path("BITE2TEXT_CHECKPOINT", "/opt/ml/model/model_best.pth")
    vocab_path = env_path("BITE2TEXT_HEAD_VOCABS", "/opt/ml/model/head_vocabs.json")
    seed = int(os.getenv("BITE2TEXT_SEED", "2026"))
    requested_device = os.getenv("BITE2TEXT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    upper_path, lower_path = input_files(input_path)
    head_vocabs = read_json(vocab_path)
    if not isinstance(head_vocabs, dict) or not head_vocabs:
        raise RuntimeError(f"Invalid head vocabularies in {vocab_path}")
    head_names = tuple(head_vocabs)

    with tempfile.TemporaryDirectory(prefix="bite2text_ptv3_") as temporary:
        normalized_upper, normalized_lower, normalizer_summary = run_ios_normalizer(
            upper_path, lower_path, Path(temporary)
        )
        upper_triangles, lower_triangles, orientation = postcorrect_triangles(
            normalized_upper, normalized_lower
        )
        coord = sampled_coordinates(upper_triangles, lower_triangles)
        sample = inference_sample(coord, seed)
        model = load_model(config_path, checkpoint_path, device)
        labels, confidences, voxel_points = predict(
            model, sample, head_vocabs, head_names, device
        )

    report = render_structured_report(labels)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "diagnostic-imaging-report.json"
    output_file.write_text(json.dumps({"report": report}, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "device": str(device),
                "input": {"upper": str(upper_path), "lower": str(lower_path)},
                "normalizer": normalizer_summary,
                "orientation": orientation,
                "sampled_points": int(coord.shape[0]),
                "voxel_points": voxel_points,
                "labels": labels,
                "confidence": {key: round(value, 4) for key, value in confidences.items()},
                "output": str(output_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
