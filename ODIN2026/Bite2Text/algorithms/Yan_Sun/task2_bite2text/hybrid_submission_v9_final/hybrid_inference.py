#!/usr/bin/env python3
"""PTv3-constrained report retrieval for the ODIN 2026 Bite2Text task."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
from photo_inference import run_photo_inference
from report_sanitizer import rejection_reason, sanitize_report, split_sentences


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

PHOTO_F1 = {
    "right_canine_relation": 0.2794493467017999,
    "left_canine_relation": 0.33103239430043335,
    "overjet": 0.3989859426890845,
    "vertical_relation": 0.4345494822975876,
    "midline_relation": 0.4111193194504122,
    "upper_crowding": 0.2804054676068764,
    "lower_crowding": 0.25965581387691944,
    "curve_wilson": 0.6321341380293211,
}

MIDLINE_SENTENCES = {
    "coincident": "The dental midlines are coincident.",
    "slightly_deviated": "The dental midlines are slightly deviated.",
    "deviated": "The dental midlines are deviated relative to each other.",
}


def correct_midline_sentence(
    report: str,
    predicted_label: str,
    confidence: float,
    threshold: float,
) -> tuple[str, bool]:
    """Replace only a retrieved midline sentence when OOF-calibrated confidence allows."""
    replacement = MIDLINE_SENTENCES.get(predicted_label)
    if replacement is None or confidence < threshold:
        return report, False
    report_sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", report.strip())
        if value.strip()
    ]
    corrected: list[str] = []
    replaced = False
    for sentence in report_sentences:
        if "midline" in sentence.lower():
            if not replaced:
                corrected.append(replacement)
                replaced = True
            continue
        corrected.append(sentence)
    if not replaced:
        corrected.append(replacement)
    return " ".join(corrected), True


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


def unsupported_sentence_count(report: str) -> int:
    """Count sentences that assert facts unsupported by the deployed heads."""
    return sum(
        rejection_reason(sentence) is not None for sentence in split_sentences(report)
    )


def contradiction_risk(
    predicted_labels: dict[str, str],
    confidence: dict[str, float],
    candidate: dict[str, str | None],
    threshold: float,
) -> float:
    """Reliability-weighted high-confidence conflicts with a candidate report."""
    numerator = 0.0
    for head, weight in CV_F1.items():
        candidate_value = candidate.get(head)
        if candidate_value is None or candidate_value == predicted_labels[head]:
            continue
        numerator += weight * max(0.0, float(confidence[head]) - threshold)
    return numerator / sum(CV_F1.values())


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
    photo_probabilities: dict[str, np.ndarray] | None = None,
    photo_lambda: float = 0.0,
    head_vocabs: dict[str, list[str]] | None = None,
    confidence: dict[str, float] | None = None,
    risk_config: dict[str, float | int | bool] | None = None,
) -> tuple[int, float, float, float, float, dict[str, object]]:
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
    photo_total_weight = sum(PHOTO_F1.values())
    candidates: list[tuple[float, int, float, float]] = []
    for index_value in candidate_indices:
        index = int(index_value)
        numerator = 0.0
        for head, weight in CV_F1.items():
            candidate_value = candidate_labels[index].get(head)
            if candidate_value is not None and predicted_labels.get(head) == candidate_value:
                numerator += weight
        agreement = numerator / total_weight
        photo_agreement = 0.0
        if photo_probabilities is not None:
            if head_vocabs is None:
                raise RuntimeError("Head vocabularies are required for photo fusion")
            photo_numerator = 0.0
            for head, weight in PHOTO_F1.items():
                candidate_value = candidate_labels[index].get(head)
                if candidate_value is None:
                    continue
                try:
                    label_index = head_vocabs[head].index(candidate_value)
                except ValueError:
                    continue
                photo_numerator += weight * float(photo_probabilities[head][label_index])
            photo_agreement = photo_numerator / photo_total_weight
        hybrid_score = (
            float(similarities[index])
            + blend_lambda * agreement
            + photo_lambda * photo_agreement
        )
        candidates.append((hybrid_score, index, agreement, photo_agreement))
    if not candidates:
        raise RuntimeError("No retrieval candidate was available")
    baseline_position = max(range(len(candidates)), key=lambda position: candidates[position][0])
    selected_position = baseline_position
    baseline_score, baseline_index, _, _ = candidates[baseline_position]
    risk_summary: dict[str, object] = {
        "enabled": False,
        "reranked": False,
        "baseline_patient_id": str(patient_ids[baseline_index]),
    }
    if risk_config and bool(risk_config.get("enabled", True)) and confidence is not None:
        margin = float(risk_config["margin"])
        unsupported_penalty = float(risk_config["unsupported_penalty"])
        contradiction_threshold = float(risk_config["contradiction_threshold"])
        contradiction_gate = float(risk_config["contradiction_gate"])
        contradiction_penalty = float(risk_config["contradiction_penalty"])
        min_contradiction_improvement = float(
            risk_config.get("min_contradiction_improvement", 0.0)
        )
        no_new_unsupported = bool(risk_config.get("no_new_unsupported", False))
        unsupported_gate = int(risk_config["unsupported_gate"])
        unsupported = np.asarray(
            [unsupported_sentence_count(reports[item[1]]) for item in candidates],
            dtype=np.float64,
        )
        contradiction = np.asarray(
            [
                contradiction_risk(
                    predicted_labels,
                    confidence,
                    candidate_labels[item[1]],
                    contradiction_threshold,
                )
                for item in candidates
            ],
            dtype=np.float64,
        )
        base_unsupported = int(unsupported[baseline_position])
        base_contradiction = float(contradiction[baseline_position])
        unsupported_trigger = base_unsupported >= unsupported_gate
        contradiction_trigger = (
            contradiction_penalty > 0 and base_contradiction >= contradiction_gate
        )
        if unsupported_trigger or contradiction_trigger:
            penalized = np.asarray(
                [item[0] for item in candidates], dtype=np.float64
            ) - unsupported_penalty * unsupported - contradiction_penalty * contradiction
            base_scores = np.asarray([item[0] for item in candidates], dtype=np.float64)
            penalized[base_scores < baseline_score - margin] = -np.inf
            alternative_position = int(np.argmax(penalized))
            alternative_unsupported = int(unsupported[alternative_position])
            alternative_contradiction = float(contradiction[alternative_position])
            improves_unsupported = (
                unsupported_trigger and alternative_unsupported < base_unsupported
            )
            improves_contradiction = (
                contradiction_trigger
                and base_contradiction - alternative_contradiction
                >= min_contradiction_improvement - 1e-12
            )
            introduces_unsupported = (
                no_new_unsupported
                and alternative_unsupported > base_unsupported
                and not improves_unsupported
            )
            if alternative_position != baseline_position and (
                improves_unsupported or improves_contradiction
            ) and not introduces_unsupported:
                selected_position = alternative_position
            risk_summary.update(
                {
                    "enabled": True,
                    "reranked": selected_position != baseline_position,
                    "reason": "+".join(
                        value
                        for value, enabled in (
                            ("unsupported", improves_unsupported),
                            ("contradiction", improves_contradiction),
                        )
                        if enabled
                    ),
                    "baseline_unsupported": base_unsupported,
                    "selected_unsupported": int(unsupported[selected_position]),
                    "baseline_contradiction": round(base_contradiction, 6),
                    "selected_contradiction": round(
                        float(contradiction[selected_position]), 6
                    ),
                    "original_score_drop": round(
                        baseline_score - candidates[selected_position][0], 6
                    ),
                }
            )
    hybrid_score, index, agreement, photo_agreement = candidates[selected_position]
    if not reports[index]:
        raise RuntimeError(f"Selected empty report for {patient_ids[index]}")
    risk_summary["selected_patient_id"] = str(patient_ids[index])
    return (
        index,
        float(similarities[index]),
        agreement,
        photo_agreement,
        hybrid_score,
        risk_summary,
    )


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
    photo_checkpoint_path = env_path(
        "BITE2TEXT_PHOTO_CHECKPOINT", "/opt/ml/model/photo_model_final.pt"
    )
    view_checkpoint_path = env_path(
        "BITE2TEXT_VIEW_CHECKPOINT", "/opt/ml/model/photo_view_classifier.pt"
    )
    seed = int(os.getenv("BITE2TEXT_SEED", "2026"))
    top_k = int(os.getenv("BITE2TEXT_HYBRID_TOP_K", "50"))
    blend_lambda = float(os.getenv("BITE2TEXT_HYBRID_LAMBDA", "0.5"))
    photo_lambda = float(os.getenv("BITE2TEXT_PHOTO_LAMBDA", "0.2"))
    midline_threshold = float(os.getenv("BITE2TEXT_MIDLINE_THRESHOLD", "0.45"))
    risk_config: dict[str, float | int | bool] = {
        "enabled": os.getenv("BITE2TEXT_RISK_RERANK", "1") != "0",
        "margin": float(os.getenv("BITE2TEXT_RISK_MARGIN", "0.02")),
        "unsupported_penalty": float(os.getenv("BITE2TEXT_UNSUPPORTED_PENALTY", "0.005")),
        "unsupported_gate": int(os.getenv("BITE2TEXT_UNSUPPORTED_GATE", "5")),
        "contradiction_threshold": float(
            os.getenv("BITE2TEXT_CONTRADICTION_THRESHOLD", "0.65")
        ),
        "contradiction_gate": float(os.getenv("BITE2TEXT_CONTRADICTION_GATE", "0.01")),
        "contradiction_penalty": float(
            os.getenv("BITE2TEXT_CONTRADICTION_PENALTY", "0.5")
        ),
        "min_contradiction_improvement": float(
            os.getenv("BITE2TEXT_MIN_CONTRADICTION_IMPROVEMENT", "0.015")
        ),
        "no_new_unsupported": os.getenv("BITE2TEXT_NO_NEW_UNSUPPORTED", "1") != "0",
    }
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
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        photo_probabilities: dict[str, np.ndarray] | None = None
        photo_summary: dict[str, object] = {"status": "v5_fallback"}
        try:
            photo_probabilities, details = run_photo_inference(
                input_path,
                view_checkpoint_path,
                photo_checkpoint_path,
                head_vocabs,
                device,
            )
            photo_summary = {"status": "used", **details}
        except Exception as error:
            if os.getenv("BITE2TEXT_PHOTO_STRICT", "0") == "1":
                raise
            photo_summary = {
                "status": "v5_fallback",
                "reason": f"{type(error).__name__}: {error}",
            }

        (
            selected_index,
            similarity,
            agreement,
            photo_agreement,
            hybrid_score,
            risk_summary,
        ) = select_report(
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
            photo_probabilities,
            photo_lambda if photo_probabilities is not None else 0.0,
            head_vocabs,
            confidence,
            risk_config,
        )

    report, midline_corrected = correct_midline_sentence(
        reports[selected_index],
        predicted_labels["midline_relation"],
        confidence["midline_relation"],
        midline_threshold,
    )
    sanitizer_enabled = os.getenv("BITE2TEXT_PRECISION_SANITIZER", "1") != "0"
    if sanitizer_enabled:
        report, sanitizer_summary = sanitize_report(report)
    else:
        sanitizer_summary = {"version": "disabled"}
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
                "photo_agreement": round(photo_agreement, 6),
                "photo": photo_summary,
                "hybrid_score": round(hybrid_score, 6),
                "risk_rerank": risk_summary,
                "top_k": top_k,
                "lambda": blend_lambda,
                "photo_lambda": photo_lambda if photo_probabilities is not None else 0.0,
                "midline_threshold": midline_threshold,
                "midline_corrected": midline_corrected,
                "precision_sanitizer": sanitizer_summary,
                "output": str(output_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
