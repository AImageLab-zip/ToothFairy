#!/usr/bin/env python3
"""Classify, select, and arrange five canonical intraoral photo views."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_CLASS_NAMES = [
    "frontal",
    "right_buccal",
    "left_buccal",
    "lower_occlusal",
    "upper_occlusal",
]
DISPLAY_NAMES = {
    "frontal": "FRONTAL",
    "right_buccal": "RIGHT BUCCAL",
    "left_buccal": "LEFT BUCCAL",
    "lower_occlusal": "LOWER OCCLUSAL",
    "upper_occlusal": "UPPER OCCLUSAL",
    "lateral_a": "LATERAL A",
    "lateral_b": "LATERAL B",
    "occlusal_a": "OCCLUSAL A",
    "occlusal_b": "OCCLUSAL B",
}


def register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


class InferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: transforms.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.rows[index]["source_path"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def load_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    all_rows: list[dict[str, str]] = []
    patient_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["valid"].lower() != "true":
                continue
            row["global_index"] = str(len(all_rows))
            all_rows.append(row)
            patient_rows[row["patient_id"]].append(row)
    return all_rows, patient_rows


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", 224))
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, image_size


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    rows: list[dict[str, str]],
    image_size: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    loader = DataLoader(
        InferenceDataset(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    probabilities = np.zeros((len(rows), model.fc.out_features), dtype=np.float32)
    embeddings = np.zeros((len(rows), model.fc.in_features), dtype=np.float32)
    captured_features: list[torch.Tensor] = []

    def capture_features(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured_features.append(output.flatten(1).detach())

    hook = model.avgpool.register_forward_hook(capture_features)
    for images, indices in loader:
        captured_features.clear()
        logits = model(images.to(device, non_blocking=True))
        batch_probabilities = logits.softmax(dim=1).cpu().numpy()
        batch_embeddings = F.normalize(captured_features.pop(), dim=1).cpu().numpy()
        probabilities[indices.numpy()] = batch_probabilities
        embeddings[indices.numpy()] = batch_embeddings
    hook.remove()
    return probabilities, embeddings


def assignment_payload(
    row: dict[str, str],
    probability_vector: np.ndarray,
    assigned_class_index: int,
    model_class_names: list[str],
) -> dict[str, Any]:
    sorted_probabilities = np.sort(probability_vector)
    assigned_probability = float(probability_vector[assigned_class_index])
    second_best = float(sorted_probabilities[-2]) if len(sorted_probabilities) > 1 else 0.0
    return {
        "row": row,
        "probability": assigned_probability,
        "margin": assigned_probability - second_best,
        "predicted_class": model_class_names[int(probability_vector.argmax())],
        "all_probabilities": {
            name: float(probability_vector[index])
            for index, name in enumerate(model_class_names)
        },
    }


def choose_diverse_pair(
    candidate_indices: list[int],
    local_probabilities: np.ndarray,
    local_embeddings: np.ndarray,
    class_index: int,
    diversity_weight: float,
) -> list[int]:
    if len(candidate_indices) <= 2:
        return candidate_indices
    best: tuple[float, tuple[int, int]] | None = None
    for first, second in itertools.combinations(candidate_indices, 2):
        probability_score = float(
            np.log(max(local_probabilities[first, class_index], 1e-8))
            + np.log(max(local_probabilities[second, class_index], 1e-8))
        )
        cosine_similarity = float(np.dot(local_embeddings[first], local_embeddings[second]))
        score = probability_score + diversity_weight * (1.0 - cosine_similarity)
        item = (score, (first, second))
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        return candidate_indices[:2]
    return list(best[1])


def assign_structural_views(
    rows: list[dict[str, str]],
    local_probabilities: np.ndarray,
    local_embeddings: np.ndarray,
    model_class_names: list[str],
    diversity_weight: float,
    fill_threshold: float | None,
) -> dict[str, dict[str, Any] | None]:
    slot_names = ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"]
    result: dict[str, dict[str, Any] | None] = {name: None for name in slot_names}
    predicted_indices = local_probabilities.argmax(axis=1)
    used: set[int] = set()

    category_slots = {
        "frontal": ["frontal"],
        "lateral": ["lateral_a", "lateral_b"],
        "occlusal": ["occlusal_a", "occlusal_b"],
    }
    for class_name, slots in category_slots.items():
        class_index = model_class_names.index(class_name)
        candidates = [
            index
            for index, predicted_index in enumerate(predicted_indices.tolist())
            if predicted_index == class_index and index not in used
        ]
        if len(slots) == 1:
            selected = (
                [max(candidates, key=lambda index: local_probabilities[index, class_index])]
                if candidates
                else []
            )
        else:
            selected = choose_diverse_pair(
                candidates,
                local_probabilities,
                local_embeddings,
                class_index,
                diversity_weight,
            )
        selected = sorted(selected, key=lambda index: int(rows[index]["source_position"]))
        for slot, image_index in zip(slots, selected):
            result[slot] = assignment_payload(
                rows[image_index],
                local_probabilities[image_index],
                class_index,
                model_class_names,
            )
            used.add(image_index)

    if fill_threshold is not None:
        missing_slots = [slot for slot, assignment in result.items() if assignment is None]
        remaining_indices = [index for index in range(len(rows)) if index not in used]
        if missing_slots and remaining_indices:
            slot_to_class = {
                "frontal": "frontal",
                "lateral_a": "lateral",
                "lateral_b": "lateral",
                "occlusal_a": "occlusal",
                "occlusal_b": "occlusal",
            }
            slot_class_indices = [
                model_class_names.index(slot_to_class[slot]) for slot in missing_slots
            ]
            cost = -np.log(
                np.clip(
                    local_probabilities[np.ix_(remaining_indices, slot_class_indices)],
                    1e-8,
                    1.0,
                )
            )
            image_rows, slot_columns = linear_sum_assignment(cost)
            for image_row, slot_column in zip(image_rows.tolist(), slot_columns.tolist()):
                image_index = remaining_indices[image_row]
                slot = missing_slots[slot_column]
                class_index = slot_class_indices[slot_column]
                if local_probabilities[image_index, class_index] < fill_threshold:
                    continue
                result[slot] = assignment_payload(
                    rows[image_index],
                    local_probabilities[image_index],
                    class_index,
                    model_class_names,
                )
                used.add(image_index)
    return result


def assign_structural_complete(
    rows: list[dict[str, str]],
    local_probabilities: np.ndarray,
    local_embeddings: np.ndarray,
    model_class_names: list[str],
    diversity_weight: float,
) -> dict[str, dict[str, Any] | None]:
    """Fill all five slots, then replace only same-category duplicate candidates."""
    slot_names = ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"]
    slot_to_class = {
        "frontal": "frontal",
        "lateral_a": "lateral",
        "lateral_b": "lateral",
        "occlusal_a": "occlusal",
        "occlusal_b": "occlusal",
    }
    slot_class_indices = [
        model_class_names.index(slot_to_class[slot]) for slot in slot_names
    ]
    cost = -np.log(
        np.clip(local_probabilities[:, slot_class_indices], 1e-8, 1.0)
    )
    image_indices, slot_indices = linear_sum_assignment(cost)
    slot_to_image: dict[str, int] = {
        slot_names[slot_index]: image_index
        for image_index, slot_index in zip(image_indices.tolist(), slot_indices.tolist())
    }

    predicted_indices = local_probabilities.argmax(axis=1)
    pair_definitions = {
        "lateral": ["lateral_a", "lateral_b"],
        "occlusal": ["occlusal_a", "occlusal_b"],
    }
    for class_name, pair_slots in pair_definitions.items():
        class_index = model_class_names.index(class_name)
        current_pair = [slot_to_image[slot] for slot in pair_slots]
        protected = {
            image_index
            for slot, image_index in slot_to_image.items()
            if slot not in pair_slots
        }
        candidates = sorted(
            set(current_pair)
            | {
                index
                for index, predicted_index in enumerate(predicted_indices.tolist())
                if predicted_index == class_index and index not in protected
            }
        )
        replacement = choose_diverse_pair(
            candidates,
            local_probabilities,
            local_embeddings,
            class_index,
            diversity_weight,
        )
        if len(replacement) == 2:
            replacement = sorted(
                replacement, key=lambda index: int(rows[index]["source_position"])
            )
            for slot, image_index in zip(pair_slots, replacement):
                slot_to_image[slot] = image_index

    result: dict[str, dict[str, Any] | None] = {name: None for name in slot_names}
    for slot in slot_names:
        image_index = slot_to_image[slot]
        class_index = model_class_names.index(slot_to_class[slot])
        result[slot] = assignment_payload(
            rows[image_index],
            local_probabilities[image_index],
            class_index,
            model_class_names,
        )
    return result


def assign_views(
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    model_class_names: list[str],
    slot_names: list[str],
    slot_to_class: dict[str, str],
    diversity_weight: float,
    incomplete_fill_threshold: float,
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {name: None for name in slot_names}
    if not rows:
        return result
    local_probabilities = np.stack(
        [probabilities[int(row["global_index"])] for row in rows], axis=0
    )
    local_embeddings = np.stack(
        [embeddings[int(row["global_index"])] for row in rows], axis=0
    )
    if model_class_names == ["frontal", "lateral", "occlusal"]:
        if len(rows) >= 5:
            return assign_structural_complete(
                rows,
                local_probabilities,
                local_embeddings,
                model_class_names,
                diversity_weight,
            )
        return assign_structural_views(
            rows,
            local_probabilities,
            local_embeddings,
            model_class_names,
            0.0,
            fill_threshold=incomplete_fill_threshold,
        )
    slot_class_indices = [model_class_names.index(slot_to_class[name]) for name in slot_names]
    slot_probabilities = local_probabilities[:, slot_class_indices]
    cost = -np.log(np.clip(slot_probabilities, 1e-8, 1.0))
    image_indices, class_indices = linear_sum_assignment(cost)
    for image_index, class_index in zip(image_indices.tolist(), class_indices.tolist()):
        probability_vector = local_probabilities[image_index]
        model_class_index = slot_class_indices[class_index]
        result[slot_names[class_index]] = assignment_payload(
            rows[image_index], probability_vector, model_class_index, model_class_names
        )
    return result


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image.draft("RGB", size)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "black")
        canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return canvas


def make_montage(
    patient_id: str,
    assignments: dict[str, dict[str, Any] | None],
    slot_names: list[str],
    output_path: Path,
    panel_size: tuple[int, int] = (384, 288),
) -> None:
    font = ImageFont.load_default()
    label_height = 32
    header_height = 34
    columns, rows_count = 3, 2
    canvas = Image.new(
        "RGB",
        (columns * panel_size[0], header_height + rows_count * (panel_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"Patient {patient_id}", fill="black", font=font)
    for index, class_name in enumerate(slot_names):
        grid_row, grid_col = divmod(index, columns)
        x = grid_col * panel_size[0]
        y = header_height + grid_row * (panel_size[1] + label_height)
        assignment = assignments[class_name]
        if assignment is None:
            panel = Image.new("RGB", panel_size, "#777777")
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.text((12, 12), "MISSING", fill="white", font=font)
            label = f"{DISPLAY_NAMES.get(class_name, class_name.upper())} | MISSING"
        else:
            panel = fit_image(Path(assignment["row"]["source_path"]), panel_size)
            label = (
                f"{DISPLAY_NAMES.get(class_name, class_name.upper())} | "
                f"p={assignment['probability']:.3f} | {assignment['row']['filename']}"
            )
        canvas.paste(panel, (x, y))
        draw.text((x + 6, y + panel_size[1] + 9), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=88)


def make_montage_job(job: tuple[str, dict[str, Any], list[str], Path]) -> str:
    patient_id, assignments, slot_names, output_path = job
    make_montage(patient_id, assignments, slot_names, output_path)
    return patient_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--montage-workers", type=int, default=8)
    parser.add_argument("--diversity-weight", type=float, default=2.0)
    parser.add_argument("--incomplete-fill-threshold", type=float, default=0.20)
    parser.add_argument("--skip-montages", action="store_true")
    args = parser.parse_args()

    register_heif()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_class_names, image_size = build_model(args.checkpoint, device)
    if model_class_names == ["frontal", "lateral", "occlusal"]:
        slot_names = ["frontal", "lateral_a", "lateral_b", "occlusal_a", "occlusal_b"]
        slot_to_class = {
            "frontal": "frontal",
            "lateral_a": "lateral",
            "lateral_b": "lateral",
            "occlusal_a": "occlusal",
            "occlusal_b": "occlusal",
        }
    else:
        slot_names = model_class_names
        slot_to_class = {name: name for name in slot_names}
    all_rows, patient_rows = load_manifest(args.manifest)
    probabilities, embeddings = predict_probabilities(
        model, all_rows, image_size, device, args.batch_size, args.workers
    )

    image_prediction_rows: list[dict[str, Any]] = []
    for row, probability_vector in zip(all_rows, probabilities):
        prediction_row: dict[str, Any] = {
            "patient_id": row["patient_id"],
            "filename": row["filename"],
            "source_path": row["source_path"],
            "predicted_class": model_class_names[int(probability_vector.argmax())],
            "max_probability": float(probability_vector.max()),
        }
        for class_index, class_name in enumerate(model_class_names):
            prediction_row[f"probability_{class_name}"] = float(probability_vector[class_index])
        image_prediction_rows.append(prediction_row)
    with (args.output_dir / "all_image_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_prediction_rows[0]))
        writer.writeheader()
        writer.writerows(image_prediction_rows)

    selection_rows: list[dict[str, Any]] = []
    selection_json: dict[str, Any] = {}
    montage_jobs: list[tuple[str, dict[str, Any], list[str], Path]] = []
    for patient_id in sorted(patient_rows):
        assignments = assign_views(
            patient_rows[patient_id],
            probabilities,
            embeddings,
            model_class_names,
            slot_names,
            slot_to_class,
            args.diversity_weight,
            args.incomplete_fill_threshold,
        )
        json_assignments: dict[str, Any] = {}
        for slot_index, class_name in enumerate(slot_names):
            assignment = assignments[class_name]
            if assignment is None:
                selection_rows.append(
                    {
                        "patient_id": patient_id,
                        "slot_index": slot_index,
                        "view": class_name,
                        "filename": "",
                        "source_path": "",
                        "probability": "",
                        "margin": "",
                        "predicted_class": "",
                        "missing": True,
                    }
                )
                json_assignments[class_name] = None
                continue
            row = assignment["row"]
            selection_rows.append(
                {
                    "patient_id": patient_id,
                    "slot_index": slot_index,
                    "view": class_name,
                    "filename": row["filename"],
                    "source_path": row["source_path"],
                    "probability": assignment["probability"],
                    "margin": assignment["margin"],
                    "predicted_class": assignment["predicted_class"],
                    "missing": False,
                }
            )
            json_assignments[class_name] = {
                "filename": row["filename"],
                "source_path": row["source_path"],
                "probability": assignment["probability"],
                "margin": assignment["margin"],
                "predicted_class": assignment["predicted_class"],
                "all_probabilities": assignment["all_probabilities"],
            }
        selection_json[patient_id] = json_assignments
        if not args.skip_montages:
            montage_jobs.append(
                (
                    patient_id,
                    assignments,
                    slot_names,
                    args.output_dir / "montages" / f"{patient_id}.jpg",
                )
            )

    with (args.output_dir / "five_view_selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    (args.output_dir / "five_view_selection.json").write_text(
        json.dumps(selection_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    probabilities_assigned = [
        float(row["probability"])
        for row in selection_rows
        if not row["missing"] and row["probability"] != ""
    ]
    summary = {
        "patients": len(patient_rows),
        "source_images": len(all_rows),
        "selected_images": sum(not row["missing"] for row in selection_rows),
        "missing_slots": sum(bool(row["missing"]) for row in selection_rows),
        "mean_assigned_probability": float(np.mean(probabilities_assigned)),
        "median_assigned_probability": float(np.median(probabilities_assigned)),
        "assignments_below_0_5": sum(value < 0.5 for value in probabilities_assigned),
        "assignments_below_0_8": sum(value < 0.8 for value in probabilities_assigned),
        "model_class_names": model_class_names,
        "slot_names": slot_names,
        "checkpoint": str(args.checkpoint),
        "diversity_weight": args.diversity_weight,
        "incomplete_fill_threshold": args.incomplete_fill_threshold,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if montage_jobs:
        with ProcessPoolExecutor(max_workers=args.montage_workers) as executor:
            for completed, _patient_id in enumerate(
                executor.map(make_montage_job, montage_jobs), start=1
            ):
                if completed % 100 == 0 or completed == len(montage_jobs):
                    print(
                        json.dumps(
                            {"montages_completed": completed, "montages_total": len(montage_jobs)}
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
