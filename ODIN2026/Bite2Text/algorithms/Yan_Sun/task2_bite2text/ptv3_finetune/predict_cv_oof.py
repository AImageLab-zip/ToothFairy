#!/usr/bin/env python3
"""Generate out-of-fold PTv3 predictions for Bite2Text.

Each patient's prediction is produced only by the checkpoint whose validation
split contains that patient.  The resulting JSONL is therefore suitable for
honest offline report-generation and retrieval-reranking experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["patient_id"]] = record
    return records


def load_checkpoint(model: torch.nn.Module, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    state_dict = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in source.items()
    )
    model.load_state_dict(state_dict, strict=True)
    return int(checkpoint.get("epoch", -1))


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key != "name"
    }


def parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > 5 for fold in folds):
        raise argparse.ArgumentTypeError("folds must be a comma-separated subset of 1,2,3,4,5")
    return folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits2bites-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1,2,3,4,5"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--experiment-prefix", default="bite2text_ptv3_v3_official_12head")
    parser.add_argument("--base-seed", type=int, default=20260810)
    args = parser.parse_args()

    bits2bites_root = args.bits2bites_root.resolve()
    sys.path.insert(0, str(bits2bites_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Imports must follow the Bits2Bites sys.path insertion so the custom
    # Bite2Text dataset and MultiTaskClassifier registrations are available.
    from pointcept.datasets import build_dataset, collate_fn
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    from report_renderer import render_report

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = read_manifest(args.manifest.resolve())
    raw_data_root = args.raw_data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    reference_vocabs: dict[str, list[str]] | None = None

    for fold in args.folds:
        fold_root = args.data_root.resolve() / f"fold{fold}"
        vocabs: dict[str, list[str]] = read_json(fold_root / "head_vocabs.json")
        if reference_vocabs is None:
            reference_vocabs = vocabs
        elif vocabs != reference_vocabs:
            raise RuntimeError(f"Fold {fold} vocabularies differ from earlier folds")
        head_names = tuple(vocabs)
        seed = args.base_seed + fold - 1
        experiment = f"{args.experiment_prefix}_fold{fold}_stage2_joint_seed{seed}"
        config_path = bits2bites_root / "configs" / "dental" / f"{args.experiment_prefix}_fold{fold}_stage2_joint.py"
        checkpoint_path = bits2bites_root / "exp" / "dental" / experiment / "model" / "model_best.pth"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing fold {fold} config/checkpoint: {config_path}, {checkpoint_path}")

        cfg = Config.fromfile(str(config_path))
        dataset = build_dataset(cfg.data.val)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        model = build_model(cfg.model)
        checkpoint_epoch = load_checkpoint(model, checkpoint_path)
        model.to(device).eval()

        fold_rows = 0
        with torch.inference_mode():
            for batch in loader:
                names = list(batch["name"])
                result = model(to_device(batch, device))
                logits = result["logits"]
                if len(logits) != len(head_names):
                    raise RuntimeError(
                        f"Fold {fold}: model emitted {len(logits)} heads, expected {len(head_names)}"
                    )
                probabilities = [values.float().softmax(dim=1).cpu() for values in logits]
                for row_index, data_name in enumerate(names):
                    patient_id = data_name.removeprefix("dental_")
                    record = manifest.get(patient_id)
                    if record is None or not record.get("target_report_path"):
                        raise RuntimeError(f"No official report target for OOF patient {patient_id}")
                    predicted_labels: dict[str, str] = {}
                    confidence: dict[str, float] = {}
                    probability_map: dict[str, dict[str, float]] = {}
                    for head_index, head_name in enumerate(head_names):
                        values = probabilities[head_index][row_index]
                        predicted_index = int(values.argmax().item())
                        predicted_labels[head_name] = vocabs[head_name][predicted_index]
                        confidence[head_name] = float(values[predicted_index].item())
                        probability_map[head_name] = {
                            label: float(values[index].item())
                            for index, label in enumerate(vocabs[head_name])
                        }
                    reference = (raw_data_root / record["target_report_path"]).read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                    all_rows.append(
                        {
                            "patient_id": patient_id,
                            "fold": fold,
                            "checkpoint_epoch": checkpoint_epoch,
                            "predicted_labels": predicted_labels,
                            "confidence": confidence,
                            "probabilities": probability_map,
                            "structured_prediction": render_report(predicted_labels),
                            "reference": reference,
                            "target_values": record["target_values"],
                        }
                    )
                    fold_rows += 1

        fold_summaries.append(
            {
                "fold": fold,
                "cases": fold_rows,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
            }
        )
        del model, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(fold_summaries[-1]), flush=True)

    patient_ids = [row["patient_id"] for row in all_rows]
    if len(patient_ids) != len(set(patient_ids)):
        duplicates = sorted(patient_id for patient_id in set(patient_ids) if patient_ids.count(patient_id) > 1)
        raise RuntimeError(f"OOF patients occurred more than once: {duplicates[:10]}")
    all_rows.sort(key=lambda row: row["patient_id"])
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    summary_path = output.with_suffix(".summary.json")
    summary = {
        "folds": args.folds,
        "cases": len(all_rows),
        "unique_patients": len(set(patient_ids)),
        "fold_summaries": fold_summaries,
        "output": str(output),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
