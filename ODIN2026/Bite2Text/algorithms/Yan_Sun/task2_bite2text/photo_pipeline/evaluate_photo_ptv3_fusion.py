#!/usr/bin/env python3
"""Summarize photo OOF predictions and cross-fit fusion with PTv3 OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def score_head(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    mask = targets >= 0
    predictions = probabilities.argmax(axis=1)
    if not mask.any():
        return {"macro_f1": None, "accuracy": None, "observed": 0}
    return {
        "macro_f1": float(
            f1_score(targets[mask], predictions[mask], average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(targets[mask], predictions[mask])),
        "observed": int(mask.sum()),
    }


def aggregate_metrics(
    targets: np.ndarray,
    probabilities: list[np.ndarray],
    head_names: list[str],
) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    f1_values: list[float] = []
    for head_index, head_name in enumerate(head_names):
        metrics = score_head(targets[:, head_index], probabilities[head_index])
        heads[head_name] = metrics
        if metrics["macro_f1"] is not None:
            f1_values.append(float(metrics["macro_f1"]))
    return {"mean_macro_f1": float(np.mean(f1_values)), "heads": heads}


def load_photo_oof(
    root: Path, head_names: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    patient_ids: list[str] = []
    fold_values: list[int] = []
    target_parts: list[np.ndarray] = []
    logit_parts: list[list[np.ndarray]] = [[] for _ in head_names]
    for fold in range(1, 6):
        path = root / f"fold{fold}" / "val_predictions_best.npz"
        payload = np.load(path)
        payload_head_names = payload["head_names"].tolist()
        if payload_head_names != head_names:
            raise RuntimeError(f"Head mismatch in {path}: {payload_head_names}")
        ids = payload["patient_ids"].astype(str).tolist()
        patient_ids.extend(ids)
        fold_values.extend([fold] * len(ids))
        target_parts.append(payload["targets"].astype(np.int64))
        for head_index, head_name in enumerate(head_names):
            logit_parts[head_index].append(payload[f"logits_{head_name}"].astype(np.float64))
    targets = np.concatenate(target_parts, axis=0)
    logits = [np.concatenate(parts, axis=0) for parts in logit_parts]
    probabilities = [softmax(values) for values in logits]
    return patient_ids, np.asarray(fold_values, dtype=np.int64), targets, probabilities


def load_ptv3_oof(
    path: Path,
    head_vocabs: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            patient_id = record["patient_id"]
            probabilities: dict[str, np.ndarray] = {}
            targets: dict[str, int] = {}
            for head_name, vocab in head_vocabs.items():
                probabilities[head_name] = np.asarray(
                    [record["probabilities"][head_name][value] for value in vocab],
                    dtype=np.float64,
                )
                target_value = record["target_values"].get(head_name)
                targets[head_name] = -1 if target_value is None else vocab.index(target_value)
            records[patient_id] = {
                "fold": int(record["fold"]),
                "probabilities": probabilities,
                "targets": targets,
            }
    return records


def choose_weight(
    targets: np.ndarray,
    ptv3_probabilities: np.ndarray,
    photo_probabilities: np.ndarray,
    mask: np.ndarray,
    grid: list[float],
) -> tuple[float, list[dict[str, float]]]:
    trace: list[dict[str, float]] = []
    best: tuple[float, float] | None = None
    for weight in grid:
        fused = (1.0 - weight) * ptv3_probabilities + weight * photo_probabilities
        score = score_head(targets[mask], fused[mask])["macro_f1"]
        value = float(score) if score is not None else -1.0
        trace.append({"photo_weight": weight, "macro_f1": value})
        candidate = (value, -weight)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return 0.0, trace
    return -best[1], trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-cv-root", type=Path, required=True)
    parser.add_argument("--ptv3-oof", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    head_names = list(head_vocabs)
    photo_ids, photo_folds, photo_targets, photo_probabilities = load_photo_oof(
        args.photo_cv_root, head_names
    )
    ptv3_records = load_ptv3_oof(args.ptv3_oof, head_vocabs)
    if len(set(photo_ids)) != len(photo_ids):
        raise RuntimeError("Duplicate patient IDs in photo OOF")
    if set(photo_ids) != set(ptv3_records):
        raise RuntimeError(
            f"OOF patient mismatch: photo={len(set(photo_ids))}, PTv3={len(ptv3_records)}"
        )

    order = np.argsort(np.asarray(photo_ids))
    patient_ids = [photo_ids[index] for index in order]
    folds = photo_folds[order]
    targets = photo_targets[order]
    photo_probabilities = [values[order] for values in photo_probabilities]
    ptv3_probabilities: list[np.ndarray] = []
    for head_name in head_names:
        ptv3_probabilities.append(
            np.stack([ptv3_records[patient_id]["probabilities"][head_name] for patient_id in patient_ids])
        )
    for row_index, patient_id in enumerate(patient_ids):
        if folds[row_index] != ptv3_records[patient_id]["fold"]:
            raise RuntimeError(f"Fold mismatch for {patient_id}")
        for head_index, head_name in enumerate(head_names):
            expected = ptv3_records[patient_id]["targets"][head_name]
            if targets[row_index, head_index] != expected:
                raise RuntimeError(
                    f"Target mismatch for {patient_id}/{head_name}: "
                    f"photo={targets[row_index, head_index]} PTv3={expected}"
                )

    grid = np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step).round(10).tolist()
    crossfit_probabilities = [np.zeros_like(values) for values in photo_probabilities]
    weight_selection: dict[str, Any] = {}
    recommended_weights: dict[str, float] = {}
    for head_index, head_name in enumerate(head_names):
        per_fold: dict[str, Any] = {}
        observed = targets[:, head_index] >= 0
        for outer_fold in range(1, 6):
            train_mask = observed & (folds != outer_fold)
            validation_mask = folds == outer_fold
            weight, trace = choose_weight(
                targets[:, head_index],
                ptv3_probabilities[head_index],
                photo_probabilities[head_index],
                train_mask,
                grid,
            )
            crossfit_probabilities[head_index][validation_mask] = (
                (1.0 - weight) * ptv3_probabilities[head_index][validation_mask]
                + weight * photo_probabilities[head_index][validation_mask]
            )
            per_fold[str(outer_fold)] = {
                "photo_weight": weight,
                "selection_cases": int(train_mask.sum()),
                "trace": trace,
            }
        full_weight, full_trace = choose_weight(
            targets[:, head_index],
            ptv3_probabilities[head_index],
            photo_probabilities[head_index],
            observed,
            grid,
        )
        recommended_weights[head_name] = full_weight
        weight_selection[head_name] = {
            "outer_folds": per_fold,
            "recommended_full_training_photo_weight": full_weight,
            "full_oof_trace_for_final_configuration_only": full_trace,
        }

    photo_metrics = aggregate_metrics(targets, photo_probabilities, head_names)
    ptv3_metrics = aggregate_metrics(targets, ptv3_probabilities, head_names)
    crossfit_metrics = aggregate_metrics(targets, crossfit_probabilities, head_names)
    deltas: dict[str, Any] = {}
    for head_name in head_names:
        ptv3_f1 = ptv3_metrics["heads"][head_name]["macro_f1"]
        photo_f1 = photo_metrics["heads"][head_name]["macro_f1"]
        fused_f1 = crossfit_metrics["heads"][head_name]["macro_f1"]
        deltas[head_name] = {
            "photo_minus_ptv3": float(photo_f1 - ptv3_f1),
            "crossfit_fusion_minus_ptv3": float(fused_f1 - ptv3_f1),
        }

    summary = {
        "cases": len(patient_ids),
        "folds": sorted(set(folds.tolist())),
        "weight_grid": grid,
        "photo_oof": photo_metrics,
        "ptv3_oof_recomputed": ptv3_metrics,
        "crossfit_fusion_oof": crossfit_metrics,
        "deltas": deltas,
        "weight_selection": weight_selection,
        "recommended_full_training_photo_weights": recommended_weights,
    }
    (args.output_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    arrays: dict[str, Any] = {
        "patient_ids": np.asarray(patient_ids),
        "folds": folds,
        "targets": targets,
        "head_names": np.asarray(head_names),
    }
    for head_index, head_name in enumerate(head_names):
        arrays[f"photo_probabilities_{head_name}"] = photo_probabilities[head_index]
        arrays[f"ptv3_probabilities_{head_name}"] = ptv3_probabilities[head_index]
        arrays[f"crossfit_probabilities_{head_name}"] = crossfit_probabilities[head_index]
    np.savez_compressed(args.output_dir / "oof_probabilities.npz", **arrays)
    print(
        json.dumps(
            {
                "cases": len(patient_ids),
                "photo_mean_macro_f1": photo_metrics["mean_macro_f1"],
                "ptv3_mean_macro_f1": ptv3_metrics["mean_macro_f1"],
                "crossfit_fusion_mean_macro_f1": crossfit_metrics["mean_macro_f1"],
                "recommended_photo_weights": recommended_weights,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
