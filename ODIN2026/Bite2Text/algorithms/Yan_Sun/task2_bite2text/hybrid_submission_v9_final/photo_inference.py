#!/usr/bin/env python3
"""Robust five-view intraoral-photo inference for Bite2Text v6."""

from __future__ import annotations

import itertools
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision import models, transforms


SLOT_NAMES = ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"]
SUPPORTED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".mha",
    ".dzi",
}


@dataclass
class DecodedPhoto:
    name: str
    image: Image.Image


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(value) if value.isdigit() else value.lower()
        for value in re.split(r"(\d+)", path.name)
    )


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        # JPEG/PNG remain supported. A failed HEIC case is handled by the
        # caller's v5 fallback instead of failing the complete submission.
        pass


def find_photo_files(input_path: Path) -> list[Path]:
    """Find the official photo socket while accepting local smoke-test layouts."""
    def is_supported_source(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not any(parent.name.endswith("_files") for parent in path.parents)
        )

    roots = (
        input_path / "images" / "intraoral-photo",
        input_path / "images" / "2d-intraoral-photographs",
        input_path / "intraoral-photo",
        input_path / "2d-intraoral-photographs",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path.resolve()
                for path in root.rglob("*")
                if is_supported_source(path)
            )
    if not files:
        files.extend(
            path.resolve()
            for path in input_path.rglob("*")
            if is_supported_source(path)
        )
    return sorted(set(files), key=_natural_key)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], [0.5, 99.5])
    if high <= low:
        low = float(values[finite].min())
        high = float(values[finite].max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def decode_mha(path: Path) -> list[DecodedPhoto]:
    """Decode common LOCAL/external MetaImage photo stacks without SimpleITK."""
    payload = path.read_bytes()
    match = re.search(br"(?mi)^ElementDataFile\s*=\s*(.+?)\r?\n", payload)
    if match is None:
        raise RuntimeError("MHA header has no ElementDataFile")
    header = payload[: match.end()].decode("ascii", errors="replace")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    dimensions = [int(value) for value in fields["dimsize"].split()]
    if len(dimensions) not in (2, 3):
        raise RuntimeError(f"Unsupported MHA dimensions: {dimensions}")
    channels = int(fields.get("elementnumberofchannels", "1"))
    dtype_by_name = {
        "MET_UCHAR": np.dtype("u1"),
        "MET_CHAR": np.dtype("i1"),
        "MET_USHORT": np.dtype("u2"),
        "MET_SHORT": np.dtype("i2"),
        "MET_UINT": np.dtype("u4"),
        "MET_INT": np.dtype("i4"),
        "MET_FLOAT": np.dtype("f4"),
        "MET_DOUBLE": np.dtype("f8"),
    }
    element_type = fields.get("elementtype", "")
    if element_type not in dtype_by_name:
        raise RuntimeError(f"Unsupported MHA element type: {element_type}")
    dtype = dtype_by_name[element_type]
    byte_order_msb = fields.get(
        "binarydatabyteordermsb", fields.get("elementbyteordermsb", "false")
    ).lower() in {"true", "1"}
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if byte_order_msb else "<")

    data_file = fields["elementdatafile"]
    raw = (
        payload[match.end() :]
        if data_file.upper() == "LOCAL"
        else (path.parent / data_file).read_bytes()
    )
    if fields.get("compresseddata", "false").lower() in {"true", "1"}:
        raw = zlib.decompress(raw)
    count = math.prod(dimensions) * channels
    values = np.frombuffer(raw, dtype=dtype, count=count)
    if values.size != count:
        raise RuntimeError(f"MHA expected {count} values, decoded {values.size}")
    shape = tuple(reversed(dimensions)) + ((channels,) if channels > 1 else ())
    values = values.reshape(shape)
    frames = values if len(dimensions) == 3 else values[np.newaxis, ...]
    output: list[DecodedPhoto] = []
    for frame_index, frame in enumerate(frames):
        array = _to_uint8(frame)
        if array.ndim == 3 and array.shape[-1] not in (3, 4):
            raise RuntimeError(f"Unsupported MHA channel count: {array.shape[-1]}")
        image = Image.fromarray(array).convert("RGB")
        output.append(DecodedPhoto(f"{path.name}#frame-{frame_index + 1}", image))
    return output


def decode_dzi(path: Path) -> list[DecodedPhoto]:
    """Reconstruct the highest Deep Zoom level using only Pillow."""
    root = ElementTree.parse(path).getroot()
    tile_size = int(root.attrib["TileSize"])
    overlap = int(root.attrib.get("Overlap", "0"))
    image_format = root.attrib["Format"]
    size = next(element for element in root if element.tag.endswith("Size"))
    width, height = int(size.attrib["Width"]), int(size.attrib["Height"])
    level = int(math.ceil(math.log2(max(width, height))))
    tile_root = path.with_name(f"{path.stem}_files") / str(level)
    canvas = Image.new("RGB", (width, height), "black")
    columns = int(math.ceil(width / tile_size))
    rows = int(math.ceil(height / tile_size))
    for row in range(rows):
        for column in range(columns):
            tile_path = tile_root / f"{column}_{row}.{image_format}"
            if not tile_path.is_file():
                raise RuntimeError(f"Missing DZI tile: {tile_path}")
            with Image.open(tile_path) as tile_source:
                tile = tile_source.convert("RGB")
            left_overlap = overlap if column > 0 else 0
            top_overlap = overlap if row > 0 else 0
            target_width = min(tile_size, width - column * tile_size)
            target_height = min(tile_size, height - row * tile_size)
            tile = tile.crop(
                (
                    left_overlap,
                    top_overlap,
                    left_overlap + target_width,
                    top_overlap + target_height,
                )
            )
            canvas.paste(tile, (column * tile_size, row * tile_size))
    return [DecodedPhoto(path.name, canvas)]


def decode_photo_file(path: Path) -> list[DecodedPhoto]:
    if path.suffix.lower() == ".mha":
        return decode_mha(path)
    if path.suffix.lower() == ".dzi":
        return decode_dzi(path)
    output: list[DecodedPhoto] = []
    with Image.open(path) as source:
        for frame_index, frame in enumerate(ImageSequence.Iterator(source)):
            image = ImageOps.exif_transpose(frame).convert("RGB").copy()
            name = (
                path.name
                if getattr(source, "n_frames", 1) == 1
                else f"{path.name}#frame-{frame_index + 1}"
            )
            output.append(DecodedPhoto(name, image))
    return output


def decode_photo_files(paths: list[Path]) -> tuple[list[DecodedPhoto], list[str]]:
    photos: list[DecodedPhoto] = []
    errors: list[str] = []
    for path in paths:
        try:
            photos.extend(decode_photo_file(path))
        except Exception as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    return photos, errors


def montage_candidates(photo: DecodedPhoto) -> list[tuple[str, list[DecodedPhoto]]]:
    """Generate plausible panels when a socket contains one flattened montage."""
    width, height = photo.image.size
    candidates: list[tuple[str, list[DecodedPhoto]]] = []
    for columns, rows in ((5, 1), (1, 5), (3, 2), (2, 3)):
        panel_width, panel_height = width // columns, height // rows
        if min(panel_width, panel_height) < 160:
            continue
        panels: list[DecodedPhoto] = []
        for row in range(rows):
            for column in range(columns):
                left, top = column * panel_width, row * panel_height
                right = width if column == columns - 1 else (column + 1) * panel_width
                bottom = height if row == rows - 1 else (row + 1) * panel_height
                panel = photo.image.crop((left, top, right, bottom))
                gray = np.asarray(panel.resize((64, 64)).convert("L"), dtype=np.float32)
                if float(gray.std()) < 5.0:
                    continue
                panels.append(
                    DecodedPhoto(
                        f"{photo.name}#tile-r{row + 1}c{column + 1}", panel
                    )
                )
        if len(panels) >= 5:
            candidates.append((f"flattened_montage_{columns}x{rows}", panels))
    return candidates


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def load_images(photos: list[DecodedPhoto], transform: transforms.Compose) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for photo in photos:
        tensors.append(transform(photo.image))
    if not tensors:
        raise RuntimeError("No decodable intraoral photographs")
    return torch.stack(tensors)


def load_view_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    if class_names != ["frontal", "lateral", "occlusal"]:
        raise RuntimeError(f"Unexpected view classes: {class_names}")
    image_size = int(checkpoint.get("image_size", 224))
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), class_names, image_size


@torch.inference_mode()
def classify_views(
    model: nn.Module, images: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    captured: list[torch.Tensor] = []

    def capture(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured.append(output.flatten(1).detach())

    hook = model.avgpool.register_forward_hook(capture)
    logits = model(images.to(device, non_blocking=True))
    hook.remove()
    probabilities = logits.float().softmax(dim=1).cpu().numpy()
    embeddings = F.normalize(captured.pop(), dim=1).float().cpu().numpy()
    return probabilities, embeddings


def choose_diverse_pair(
    candidates: list[int],
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    class_index: int,
    diversity_weight: float,
) -> list[int]:
    if len(candidates) <= 2:
        return candidates
    best: tuple[float, tuple[int, int]] | None = None
    for first, second in itertools.combinations(candidates, 2):
        score = float(
            np.log(max(probabilities[first, class_index], 1e-8))
            + np.log(max(probabilities[second, class_index], 1e-8))
            + diversity_weight * (1.0 - np.dot(embeddings[first], embeddings[second]))
        )
        item = (score, (first, second))
        if best is None or item[0] > best[0]:
            best = item
    return list(best[1]) if best is not None else candidates[:2]


def select_views(
    paths: list[DecodedPhoto],
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    class_names: list[str],
    diversity_weight: float = 1.0,
    incomplete_fill_threshold: float = 0.2,
) -> tuple[list[DecodedPhoto | None], list[dict[str, Any]]]:
    """Reproduce the final training-time structural view assignment."""
    slot_to_class = {
        "frontal": "frontal",
        "lateral_a": "lateral",
        "lateral_b": "lateral",
        "occlusal_a": "occlusal",
        "occlusal_b": "occlusal",
    }
    slot_class_indices = [class_names.index(slot_to_class[slot]) for slot in SLOT_NAMES]
    slot_to_image: dict[str, int] = {}

    if len(paths) >= len(SLOT_NAMES):
        cost = -np.log(np.clip(probabilities[:, slot_class_indices], 1e-8, 1.0))
        image_indices, slot_indices = linear_sum_assignment(cost)
        slot_to_image = {
            SLOT_NAMES[slot_index]: image_index
            for image_index, slot_index in zip(
                image_indices.tolist(), slot_indices.tolist(), strict=True
            )
        }
        predicted_indices = probabilities.argmax(axis=1)
        for class_name, pair_slots in {
            "lateral": ["lateral_a", "lateral_b"],
            "occlusal": ["occlusal_a", "occlusal_b"],
        }.items():
            class_index = class_names.index(class_name)
            protected = {
                image_index
                for slot, image_index in slot_to_image.items()
                if slot not in pair_slots
            }
            candidates = sorted(
                {slot_to_image[slot] for slot in pair_slots}
                | {
                    index
                    for index, predicted in enumerate(predicted_indices.tolist())
                    if predicted == class_index and index not in protected
                }
            )
            replacement = choose_diverse_pair(
                candidates,
                probabilities,
                embeddings,
                class_index,
                diversity_weight,
            )
            if len(replacement) == 2:
                replacement = sorted(replacement)
                slot_to_image.update(dict(zip(pair_slots, replacement, strict=True)))
    else:
        predicted_indices = probabilities.argmax(axis=1)
        used: set[int] = set()
        for class_name, slots in {
            "frontal": ["frontal"],
            "lateral": ["lateral_a", "lateral_b"],
            "occlusal": ["occlusal_a", "occlusal_b"],
        }.items():
            class_index = class_names.index(class_name)
            candidates = [
                index
                for index, predicted in enumerate(predicted_indices.tolist())
                if predicted == class_index and index not in used
            ]
            chosen = (
                [max(candidates, key=lambda index: probabilities[index, class_index])]
                if len(slots) == 1 and candidates
                else choose_diverse_pair(
                    candidates, probabilities, embeddings, class_index, 0.0
                )
            )
            for slot, image_index in zip(slots, sorted(chosen)):
                slot_to_image[slot] = image_index
                used.add(image_index)
        missing = [slot for slot in SLOT_NAMES if slot not in slot_to_image]
        remaining = [index for index in range(len(paths)) if index not in used]
        if missing and remaining:
            missing_classes = [class_names.index(slot_to_class[slot]) for slot in missing]
            cost = -np.log(
                np.clip(probabilities[np.ix_(remaining, missing_classes)], 1e-8, 1.0)
            )
            image_rows, slot_columns = linear_sum_assignment(cost)
            for image_row, slot_column in zip(
                image_rows.tolist(), slot_columns.tolist(), strict=True
            ):
                image_index = remaining[image_row]
                slot = missing[slot_column]
                class_index = missing_classes[slot_column]
                if probabilities[image_index, class_index] >= incomplete_fill_threshold:
                    slot_to_image[slot] = image_index

    selected: list[DecodedPhoto | None] = []
    trace: list[dict[str, Any]] = []
    for slot, class_index in zip(SLOT_NAMES, slot_class_indices, strict=True):
        image_index = slot_to_image.get(slot)
        selected.append(paths[image_index] if image_index is not None else None)
        trace.append(
            {
                "slot": slot,
                "filename": paths[image_index].name if image_index is not None else None,
                "probability": (
                    round(float(probabilities[image_index, class_index]), 6)
                    if image_index is not None
                    else None
                ),
            }
        )
    return selected, trace


class MultiViewClassifier(nn.Module):
    def __init__(self, class_counts: list[int]) -> None:
        super().__init__()
        backbone_model = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        projection_dim = 256
        hidden_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(512, projection_dim), nn.GELU(), nn.LayerNorm(projection_dim)
        )
        self.slot_embeddings = nn.Parameter(torch.zeros(len(SLOT_NAMES), projection_dim))
        fused_dim = len(SLOT_NAMES) * projection_dim + 2 * projection_dim + len(SLOT_NAMES)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, count) for count in class_counts)

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> list[torch.Tensor]:
        batch_size, views, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch_size * views, channels, height, width))
        features = features.flatten(1).reshape(batch_size, views, -1)
        projected = self.projection(features) + self.slot_embeddings.unsqueeze(0)
        mask = view_mask.unsqueeze(-1)
        masked = projected * mask
        mean_pool = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = projected.masked_fill(mask == 0, -1e4).max(dim=1).values
        fused = torch.cat([masked.flatten(1), mean_pool, max_pool, view_mask], dim=1)
        hidden = self.fusion(fused)
        return [head(hidden) for head in self.heads]


def load_photo_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, list[str]], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head_vocabs = dict(checkpoint["head_vocabs"])
    if list(checkpoint["slot_names"]) != SLOT_NAMES:
        raise RuntimeError("Photo checkpoint slot order is incompatible")
    model = MultiViewClassifier([len(values) for values in head_vocabs.values()])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), head_vocabs, int(checkpoint.get("image_size", 224))


@torch.inference_mode()
def run_photo_inference(
    input_path: Path,
    view_checkpoint_path: Path,
    photo_checkpoint_path: Path,
    expected_head_vocabs: dict[str, list[str]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    register_heif()
    paths = find_photo_files(input_path)
    if not paths:
        raise RuntimeError("No intraoral photographs found")
    decoded_photos, decode_errors = decode_photo_files(paths)
    if not decoded_photos:
        raise RuntimeError(f"No decodable intraoral photographs: {decode_errors}")

    view_model, class_names, view_image_size = load_view_model(
        view_checkpoint_path, device
    )
    candidate_sets: list[tuple[str, list[DecodedPhoto]]] = [
        ("individual_or_stack", decoded_photos)
    ]
    if len(decoded_photos) == 1:
        candidate_sets.extend(montage_candidates(decoded_photos[0]))
    best: tuple[float, str, list[DecodedPhoto | None], list[dict[str, Any]]] | None = None
    for source_mode, candidate_photos in candidate_sets:
        view_images = load_images(candidate_photos, eval_transform(view_image_size))
        view_probabilities, embeddings = classify_views(view_model, view_images, device)
        selected, trace = select_views(
            candidate_photos, view_probabilities, embeddings, class_names
        )
        assigned = [row["probability"] for row in trace if row["probability"] is not None]
        coverage = len(assigned)
        score = (
            float(np.mean(assigned)) * coverage / len(SLOT_NAMES)
            if assigned
            else 0.0
        )
        item = (score, source_mode, selected, trace)
        if best is None or item[0] > best[0]:
            best = item
    del view_model
    if best is None:
        raise RuntimeError("Photo view selection produced no candidates")
    _selection_score, source_mode, selected, trace = best

    photo_model, head_vocabs, image_size = load_photo_model(photo_checkpoint_path, device)
    if head_vocabs != expected_head_vocabs:
        raise RuntimeError("Photo and PTv3 head vocabularies differ")
    transform = eval_transform(image_size)
    tensors: list[torch.Tensor] = []
    mask: list[float] = []
    for path in selected:
        if path is None:
            tensors.append(torch.zeros(3, image_size, image_size))
            mask.append(0.0)
        else:
            tensors.append(load_images([path], transform)[0])
            mask.append(1.0)
    images = torch.stack(tensors).unsqueeze(0).to(device, non_blocking=True)
    view_mask = torch.tensor([mask], dtype=torch.float32, device=device)
    logits = photo_model(images, view_mask)
    probabilities = {
        head: values.float().softmax(dim=1)[0].cpu().numpy()
        for head, values in zip(head_vocabs, logits, strict=True)
    }
    return probabilities, {
        "files_found": len(paths),
        "frames_decoded": len(decoded_photos),
        "source_mode": source_mode,
        "decode_errors": decode_errors,
        "found": len(decoded_photos),
        "used": int(sum(mask)),
        "selection": trace,
    }
