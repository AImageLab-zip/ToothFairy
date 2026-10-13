#!/usr/bin/env python3
"""PTv3-constrained report retrieval for the ODIN 2026 Bite2Text task."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from inference import (
    env_path,
    inference_sample,
    input_files,
    load_model,
    postcorrect_triangles,
    run_ios_normalizer,
    sampled_coordinates,
)
from retrieval_inference import descriptor_from_coord, load_assets


CV_F1 = {
    "right_molar_relation": 0.3829,
    "right_canine_relation": 0.3765,
    "left_molar_relation": 0.3867,
    "left_canine_relation": 0.3955,
    "overjet": 0.6366,
    "vertical_relation": 0.5437,
    "midline_relation": 0.3770,
    "crossbite": 0.4997,
    "upper_crowding": 0.2522,
    "lower_crowding": 0.2543,
    "curve_spee": 0.7466,
    "curve_wilson": 0.6514,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_labels(
    path: Path, reports_path: Path, patient_ids: np.ndarray
) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise RuntimeError(f"Missing retrieval labels: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "bite2text-hybrid-labels-v1":
        raise RuntimeError("Unsupported retrieval label asset version")
    if payload.get("patient_ids") != patient_ids.tolist():
        raise RuntimeError("Retrieval label order does not match descriptor index")
    if payload.get("retrieval_reports_sha256") != sha256_file(reports_path):
        raise RuntimeError("Retrieval labels were not built for the mounted reports file")
    values = payload.get("target_values")
    if not isinstance(values, list) or len(values) != len(patient_ids):
        raise RuntimeError("Invalid retrieval target label count")
    return values


def select_report(
    descriptor: np.ndarray,
    predicted_labels: dict[str, str],
    patient_ids: np.ndarray,
    database: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    reports: list[str],
    candidate_labels: list[dict[str, str | None]],
    top_k: int,
    blend_lambda: float,
) -> tuple[int, float, float, float]:
    query = (descriptor - mean) / scale
    norm = float(np.linalg.norm(query))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Degenerate standardized query descriptor")
    query /= norm
    similarities = database @ query
    if not np.isfinite(similarities).all():
        raise RuntimeError("Non-finite retrieval similarities")
    candidate_indices = np.argsort(-similarities)[: min(top_k, len(similarities))]
    total_weight = sum(CV_F1.values())
    best: tuple[float, int, float] | None = None
    for index_value in candidate_indices:
        index = int(index_value)
        numerator = 0.0
        for head, weight in CV_F1.items():
            candidate_value = candidate_labels[index].get(head)
            if candidate_value is not None and predicted_labels.get(head) == candidate_value:
                numerator += weight
        agreement = numerator / total_weight
        hybrid_score = float(similarities[index]) + blend_lambda * agreement
        item = (hybrid_score, index, agreement)
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        raise RuntimeError("No retrieval candidate was available")
    hybrid_score, index, agreement = best
    if not reports[index]:
        raise RuntimeError(f"Selected empty report for {patient_ids[index]}")
    return index, float(similarities[index]), agreement, hybrid_score


def predict_labels(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    head_vocabs: dict[str, list[str]],
    device: torch.device,
) -> tuple[dict[str, str], dict[str, float], int]:
    batch = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
    use_amp = os.getenv("BITE2TEXT_AMP", "0") == "1"
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and use_amp,
        ):
            logits = model(batch)["logits"]
    head_names = tuple(head_vocabs)
    if len(logits) != len(head_names):
        raise RuntimeError(f"Model emitted {len(logits)} heads; expected {len(head_names)}")
    labels: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for head, values in zip(head_names, logits, strict=True):
        probabilities = values.float().softmax(dim=1)
        index = int(probabilities.argmax(dim=1).item())
        labels[head] = head_vocabs[head][index]
        confidence[head] = float(probabilities[0, index].item())
    return labels, confidence, int(sample["coord"].shape[0])


def run() -> int:
    input_path = env_path("BITE2TEXT_INPUT_PATH", "/input")
    output_path = env_path("BITE2TEXT_OUTPUT_PATH", "/output")
    index_path = env_path("BITE2TEXT_RETRIEVAL_INDEX", "/opt/ml/model/retrieval_index.npz")
    reports_path = env_path("BITE2TEXT_RETRIEVAL_REPORTS", "/opt/ml/model/retrieval_reports.json")
    labels_path = env_path("BITE2TEXT_RETRIEVAL_LABELS", "/opt/ml/model/retrieval_labels.json")
    config_path = env_path("BITE2TEXT_CONFIG", "/opt/ml/model/config.py")
    checkpoint_path = env_path("BITE2TEXT_CHECKPOINT", "/opt/ml/model/model_final.pth")
    vocab_path = env_path("BITE2TEXT_HEAD_VOCABS", "/opt/ml/model/head_vocabs.json")
    seed = int(os.getenv("BITE2TEXT_SEED", "2026"))
    top_k = int(os.getenv("BITE2TEXT_HYBRID_TOP_K", "50"))
    blend_lambda = float(os.getenv("BITE2TEXT_HYBRID_LAMBDA", "0.5"))
    requested_device = os.getenv("BITE2TEXT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    patient_ids, database, mean, scale, reports = load_assets(index_path, reports_path)
    candidate_labels = load_candidate_labels(labels_path, reports_path, patient_ids)
    head_vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    head_names = tuple(head_vocabs)
    if set(head_names) != set(CV_F1):
        raise RuntimeError("Model head vocabularies do not match hybrid reliability weights")
    upper_path, lower_path = input_files(input_path)

    with tempfile.TemporaryDirectory(prefix="bite2text_hybrid_") as temporary:
        normalized_upper, normalized_lower, normalizer_summary = run_ios_normalizer(
            upper_path, lower_path, Path(temporary)
        )
        upper_triangles, lower_triangles, orientation = postcorrect_triangles(
            normalized_upper, normalized_lower
        )
        coord = sampled_coordinates(upper_triangles, lower_triangles)
        geometry_descriptor = descriptor_from_coord(coord)
        sample = inference_sample(coord, seed)
        model = load_model(config_path, checkpoint_path, device)
        predicted_labels, confidence, voxel_points = predict_labels(
            model, sample, head_vocabs, device
        )
        selected_index, similarity, agreement, hybrid_score = select_report(
            geometry_descriptor,
            predicted_labels,
            patient_ids,
            database,
            mean,
            scale,
            reports,
            candidate_labels,
            top_k,
            blend_lambda,
        )

    report = reports[selected_index]
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "diagnostic-imaging-report.json"
    output_file.write_text(
        json.dumps({"report": report}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "normalizer": normalizer_summary,
                "orientation": orientation,
                "sampled_points": int(coord.shape[0]),
                "voxel_points": voxel_points,
                "predicted_labels": predicted_labels,
                "mean_confidence": round(float(np.mean(list(confidence.values()))), 4),
                "retrieved_patient_id": str(patient_ids[selected_index]),
                "cosine_similarity": round(similarity, 6),
                "label_agreement": round(agreement, 6),
                "hybrid_score": round(hybrid_score, 6),
                "top_k": top_k,
                "lambda": blend_lambda,
                "output": str(output_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
