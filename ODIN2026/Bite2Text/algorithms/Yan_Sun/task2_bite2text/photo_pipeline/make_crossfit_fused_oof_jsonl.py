#!/usr/bin/env python3
"""Inject strict cross-fit photo/PTv3 probabilities into the PTv3 OOF JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-oof", type=Path, required=True)
    parser.add_argument("--ptv3-oof", type=Path, required=True)
    parser.add_argument("--head-vocabs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head_vocabs = json.loads(args.head_vocabs.read_text(encoding="utf-8"))
    head_names = list(head_vocabs)
    payload = np.load(args.fusion_oof)
    patient_ids = payload["patient_ids"].astype(str).tolist()
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    folds = payload["folds"].astype(int)
    fused = {
        head_name: payload[f"crossfit_probabilities_{head_name}"]
        for head_name in head_names
    }

    output_rows: list[dict] = []
    for line in args.ptv3_oof.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        patient_id = row["patient_id"]
        index = index_by_id[patient_id]
        if int(row["fold"]) != int(folds[index]):
            raise RuntimeError(f"Fold mismatch for {patient_id}")
        row["fusion_source"] = "strict_crossfit_photo_ptv3"
        for head_name, vocab in head_vocabs.items():
            probabilities = fused[head_name][index]
            probability_by_label = {
                label: float(probabilities[label_index])
                for label_index, label in enumerate(vocab)
            }
            prediction_index = int(np.argmax(probabilities))
            row["probabilities"][head_name] = probability_by_label
            row["predicted_labels"][head_name] = vocab[prediction_index]
            row["confidence"][head_name] = float(probabilities[prediction_index])
        output_rows.append(row)
    if len(output_rows) != len(patient_ids):
        raise RuntimeError(f"Expected {len(patient_ids)} rows, wrote {len(output_rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
