#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-case Dental Status and Hygiene inference from 2 STL + one five-frame TIFF."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
import trimesh

from dadsn_model import DeformableTeethNet
from dadsn_alignment import align_scan
from photo_loader import load_five_view_tiff


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSET_ROOT = Path("/opt/ml/model/frozen/dental_hygiene")
IMAGE_SIZE = 224
SEGMENTATION_POINTS = 16000
SEED = 20260810
COLOR_FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
    "gray_mean",
    "gray_std",
    "gray_q10",
    "gray_q50",
    "gray_q90",
    "gray_dark_fraction",
    "gray_bright_fraction",
    "saturated_fraction",
)
PERMANENT_FDI = tuple(range(11, 19)) + tuple(range(21, 29)) + tuple(range(31, 39)) + tuple(range(41, 49))
PRIMARY_FDI = tuple(range(51, 56)) + tuple(range(61, 66)) + tuple(range(71, 76)) + tuple(range(81, 86))
VALID_FDI = {str(tooth) for tooth in (*PERMANENT_FDI, *PRIMARY_FDI)}


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid photograph dimensions")
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def color_features(image: Image.Image) -> np.ndarray:
    resized = image.copy().convert("RGB")
    resized.thumbnail((384, 384), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    saturation = hsv[..., 1]
    values = np.concatenate(
        [
            rgb.reshape(-1, 3).mean(axis=0),
            rgb.reshape(-1, 3).std(axis=0),
            hsv.reshape(-1, 3).mean(axis=0),
            hsv.reshape(-1, 3).std(axis=0),
            np.asarray(
                [
                    gray.mean(),
                    gray.std(),
                    *np.quantile(gray, (0.1, 0.5, 0.9)).tolist(),
                    np.mean(gray < 0.2),
                    np.mean(gray > 0.9),
                    np.mean(saturation > 0.5),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    if values.shape != (len(COLOR_FEATURE_NAMES),):
        raise RuntimeError(f"Unexpected color feature shape: {values.shape}")
    return values


class FrozenResNet18(nn.Module):
    def __init__(self, state_path: Path):
        super().__init__()
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs).flatten(1)


class DadsnSegmenter:
    def __init__(self, checkpoint_path: Path, device: torch.device, seed: int):
        self.device = device
        self.seed = seed
        self.model = DeformableTeethNet(num_classes=17, k=32, emb_dims=1024).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @staticmethod
    def _patch_indices(vertex_count: int) -> list[np.ndarray]:
        if vertex_count <= SEGMENTATION_POINTS:
            extra = SEGMENTATION_POINTS - vertex_count
            padding = np.random.choice(vertex_count, extra, replace=True)
            return [np.concatenate([np.arange(vertex_count), padding])]
        indices = np.arange(vertex_count)
        np.random.shuffle(indices)
        patches: list[np.ndarray] = []
        while len(indices) >= SEGMENTATION_POINTS:
            patches.append(indices[:SEGMENTATION_POINTS])
            indices = indices[SEGMENTATION_POINTS:]
        if len(indices):
            extra = SEGMENTATION_POINTS - len(indices)
            remaining = np.setdiff1d(np.arange(vertex_count), indices, assume_unique=True)
            padding = np.random.choice(remaining, extra, replace=False)
            patches.append(np.concatenate([indices, padding]))
        return patches

    def predict_counts(self, path: str | Path, arch: str) -> np.ndarray:
        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        if arch not in {"upper", "lower"}:
            raise ValueError(f"Unsupported arch: {arch}")

        numpy_state = np.random.get_state()
        np.random.seed(self.seed + (0 if arch == "upper" else 1))
        try:
            mesh = trimesh.load(source, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 100 or not len(mesh.faces):
                raise ValueError(f"Invalid STL mesh: {source}")
            try:
                mesh = align_scan(mesh)
            except Exception:
                # This matches the source DADSN fallback: segmentation proceeds on
                # the raw mesh only when alignment itself cannot be fitted.
                pass
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if vertices.shape != normals.shape:
                raise ValueError(f"Invalid mesh normals: {source}")
            logits_sum = np.zeros((len(vertices), 17), dtype=np.float32)
            prediction_count = np.zeros(len(vertices), dtype=np.float32)

            with torch.inference_mode():
                for selected in self._patch_indices(len(vertices)):
                    patch_vertices = vertices[selected]
                    patch_normals = normals[selected]
                    mean = patch_vertices.mean(axis=0)
                    standard_deviation = patch_vertices.std(axis=0)
                    standard_deviation[standard_deviation < 1e-8] = 1.0
                    points = np.concatenate(
                        [(patch_vertices - mean) / standard_deviation, patch_normals],
                        axis=1,
                    ).astype(np.float32)
                    tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1).to(self.device)
                    logits, _, _ = self.model(tensor)
                    values = logits.detach().cpu().numpy()[0]
                    if values.shape == (17, len(selected)):
                        values = values.T
                    if values.shape != (len(selected), 17):
                        raise RuntimeError(f"Unexpected DADSN output shape: {values.shape}")
                    logits_sum[selected] += values
                    prediction_count[selected] += 1.0
                    del tensor, logits

            valid = prediction_count > 0
            if not valid.all():
                raise RuntimeError(f"DADSN did not predict every STL vertex: {valid.sum()}/{len(valid)}")
            labels = np.argmax(logits_sum / prediction_count[:, None], axis=1)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return np.bincount(labels.clip(min=0), minlength=17)[1:17].astype(np.float32)
        finally:
            np.random.set_state(numpy_state)


def segmentation_features(upper_counts: np.ndarray, lower_counts: np.ndarray) -> np.ndarray:
    counts = np.concatenate([upper_counts, lower_counts]).astype(np.float32)
    if counts.shape != (32,):
        raise ValueError(f"Expected 32 segmentation counts, found {counts.shape}")
    totals = np.asarray([counts[:16].sum(), counts[16:].sum()], dtype=np.float32)
    fractions = counts.copy()
    fractions[:16] /= max(float(totals[0]), 1.0)
    fractions[16:] /= max(float(totals[1]), 1.0)
    present = (counts >= 50.0).astype(np.float32)
    return np.concatenate([np.log1p(counts), fractions * 100.0, present]).reshape(1, -1)


def normalize_fdi_set(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"Expected a list-valued FDI prediction, got {type(value).__name__}")
    normalized = sorted({str(item) for item in value}, key=int)
    invalid = [tooth for tooth in normalized if tooth not in VALID_FDI]
    if invalid:
        raise ValueError(f"Model returned invalid FDI values: {invalid}")
    return normalized


class DentalHygienePredictor:
    def __init__(
        self,
        device: str = "auto",
        seed: int = SEED,
        asset_root: str | Path = DEFAULT_ASSET_ROOT,
        load_segmenter: bool = True,
    ):
        seed_everything(seed)
        self.device = resolve_device(device)
        self.seed = seed
        self.asset_root = Path(asset_root).expanduser().resolve()
        models_dir = self.asset_root / "models"
        config_path = self.asset_root / "config" / "model_config_v1.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.missing_model = joblib.load(models_dir / "missing_teeth_randomforest.joblib")
        self.restoration_model = joblib.load(models_dir / "restorations_extratrees.joblib")
        self.pigmentation_model = joblib.load(models_dir / "pigmentation_logistic.joblib")
        self.photo_encoder = FrozenResNet18(models_dir / "resnet18_imagenet_state_dict.pt").to(self.device)
        self.photo_encoder.eval()
        self.segmenter = (
            DadsnSegmenter(models_dir / "dadsn_all_align_best_model.pth", self.device, seed)
            if load_segmenter
            else None
        )

    def _photo_features(self, views: dict[str, Image.Image]) -> np.ndarray:
        selected = [views["lower"], views["upper"]]
        tensors = []
        colors = []
        for image in selected:
            colors.append(color_features(image))
            tensors.append(
                TF.normalize(
                    TF.to_tensor(letterbox(image)),
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            embeddings = self.photo_encoder(batch).cpu().numpy().astype(np.float32)
        return np.concatenate(
            [
                np.concatenate([embeddings[index], colors[index]])
                for index in range(2)
            ]
        ).reshape(1, -1).astype(np.float32)

    def predict(
        self,
        upper_stl: str | Path,
        lower_stl: str | Path,
        photo_tiff: str | Path,
        single_frame_layout: str | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        views = load_five_view_tiff(photo_tiff, layout=single_frame_layout)
        upper_counts = self.segmenter.predict_counts(upper_stl, "upper")
        lower_counts = self.segmenter.predict_counts(lower_stl, "lower")
        missing_X = segmentation_features(upper_counts, lower_counts)
        photo_X = self._photo_features(views)

        missing_teeth = normalize_fdi_set(self.missing_model.predict(missing_X)[0])
        restorations = normalize_fdi_set(self.restoration_model.predict(photo_X)[0])
        probability = float(self.pigmentation_model.predict_proba(photo_X)[0, 1])
        threshold = float(self.config["fields"]["pigmentation"]["probability_threshold"])

        result: dict[str, Any] = {
            "5_dental_status": {
                "missing_teeth": missing_teeth,
                "restorations": restorations,
                "caries": [],
            },
            "6_oral_hygiene": {
                "hygiene_status": "Not_Mentioned",
                "pigmentation": True if probability >= threshold else None,
            },
        }
        if include_debug:
            result["_debug"] = {
                "pigmentation_probability": probability,
                "pigmentation_threshold": threshold,
                "upper_c17_counts": upper_counts.astype(int).tolist(),
                "lower_c17_counts": lower_counts.astype(int).tolist(),
                "device": str(self.device),
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-stl", required=True, type=Path)
    parser.add_argument("--lower-stl", required=True, type=Path)
    parser.add_argument("--photo-tiff", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--single-frame-layout",
        choices=("auto", "horizontal5", "vertical5", "grid_2x3", "grid_3x2"),
        default=None,
    )
    parser.add_argument("--include-debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = DentalHygienePredictor(device=args.device)
    result = predictor.predict(
        upper_stl=args.upper_stl,
        lower_stl=args.lower_stl,
        photo_tiff=args.photo_tiff,
        single_frame_layout=args.single_frame_layout,
        include_debug=args.include_debug,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
