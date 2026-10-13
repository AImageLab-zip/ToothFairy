#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_infer_ensemble_v2.py

使用 5-fold ensemble + calibration_v2 进行最终 sagittal v3 推理。

依赖同目录已有 common.py：
- MolarHierarchicalModel
- CanineHierarchicalModel
- SingleViewTransform
- PairViewTransform

关键变化：
- 每个 fold 的概率先做 temperature scaling，再平均；
- Class II 不再简单 argmax subtype；
- 使用 calibration_v2 中的 subtype_full_threshold；
- 不主动预测 Class II unspecified；
- Not_Mentioned 不由视觉模型预测。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import time

import numpy as np
import torch
from PIL import Image

from common import (
    MolarHierarchicalModel,
    CanineHierarchicalModel,
    SingleViewTransform,
    PairViewTransform,
)

DEFAULT_ROOT = Path(
    "work/sagittal/sagittal_v3"
)
DEFAULT_RENDER_DIR = (
    DEFAULT_ROOT / "render"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sagittal v3 hierarchical 5-fold ensemble inference v2"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=DEFAULT_RENDER_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "inference_v2"
        ),
    )
    parser.add_argument(
        "--patient-id",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--patient-list",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--scan-render-dir",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    return parser.parse_args()


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def temperature_scale_one(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )
    probability = np.clip(
        probability,
        1e-8,
        1.0,
    )
    probability = (
        probability
        / probability.sum()
    )

    logits = (
        np.log(probability)
        / float(temperature)
    )
    logits -= logits.max()
    exp_value = np.exp(logits)
    return (
        exp_value
        / exp_value.sum()
    )


def discover_patients(
    args: argparse.Namespace,
    render_dir: Path,
) -> list[str]:
    patients = set(
        args.patient_id
    )

    if args.patient_list is not None:
        for line in (
            args.patient_list
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):
            value = line.strip()
            if value:
                patients.add(value)

    if args.scan_render_dir:
        suffix = "_left_pure_side.png"

        for path in render_dir.glob(
            f"*{suffix}"
        ):
            patients.add(
                path.name[
                    :-len(suffix)
                ]
            )

    if not patients:
        raise RuntimeError(
            "没有 patient_id。使用 "
            "--patient-id / --patient-list / "
            "--scan-render-dir"
        )

    return sorted(patients)


def load_ensemble(
    root: Path,
    task: str,
    device: torch.device,
):
    task_dir = (
        root / "runs" / task
    )

    fold_dirs = sorted(
        [
            path
            for path in task_dir.glob(
                "fold_*"
            )
            if (
                path / "best_model.pt"
            ).exists()
        ],
        key=lambda path: int(
            path.name.split("_")[-1]
        ),
    )

    if len(fold_dirs) != 5:
        raise RuntimeError(
            f"{task}: 期望 5 个 fold，"
            f"实际找到 {len(fold_dirs)}"
        )

    models = []
    height = None
    width = None

    for fold_dir in fold_dirs:
        checkpoint = load_checkpoint(
            fold_dir
            / "best_model.pt",
            device,
        )

        if task == "molar":
            model = (
                MolarHierarchicalModel(
                    pretrained=False
                )
            )
        else:
            model = (
                CanineHierarchicalModel(
                    pretrained=False
                )
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )
        model.to(device)
        model.eval()
        models.append(model)

        this_height = int(
            checkpoint.get(
                "input_height",
                256,
            )
        )
        this_width = int(
            checkpoint.get(
                "input_width",
                416,
            )
        )

        if height is None:
            height = this_height
            width = this_width
        elif (
            height != this_height
            or width != this_width
        ):
            raise ValueError(
                f"{task}: fold 输入尺寸不一致"
            )

    return (
        models,
        int(height),
        int(width),
    )


def load_calibration(
    root: Path,
    task: str,
) -> dict[str, Any]:
    path = (
        root
        / "calibration"
        / f"{task}_calibration_v2.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"缺少 calibration v2: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def final_label(
    coarse_probability: np.ndarray,
    subtype_probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    na_threshold = config[
        "na_threshold"
    ]
    full_threshold = float(
        config[
            "subtype_full_threshold"
        ]
    )

    if na_threshold is None:
        coarse_id = int(
            np.argmax(
                coarse_probability
            )
        )
    else:
        if (
            coarse_probability[3]
            >= float(na_threshold)
        ):
            coarse_id = 3
        else:
            coarse_id = int(
                np.argmax(
                    coarse_probability[:3]
                )
            )

    if coarse_id == 0:
        label = "Class I"
    elif coarse_id == 2:
        label = "Class III"
    elif coarse_id == 3:
        label = "Not_Assessable"
    else:
        label = (
            "Class II full"
            if (
                subtype_probability[1]
                >= full_threshold
            )
            else "Class II head-to-head"
        )

    detail = {
        "label": label,
        "coarse_id": coarse_id,
        "coarse_probability": [
            float(value)
            for value in coarse_probability
        ],
        "subtype_probability": {
            "head_to_head": float(
                subtype_probability[0]
            ),
            "full": float(
                subtype_probability[1]
            ),
        },
        "na_threshold": na_threshold,
        "subtype_full_threshold": (
            full_threshold
        ),
    }

    return label, detail


@torch.no_grad()
def predict_molar(
    models,
    pure_path: Path,
    side: str,
    height: int,
    width: int,
    device: torch.device,
    calibration: dict[str, Any],
):
    transform = SingleViewTransform(
        height=height,
        width=width,
        train=False,
    )

    with Image.open(
        pure_path
    ) as source:
        image = source.convert("RGB")

    tensor = (
        transform(
            image,
            side,
        )
        .unsqueeze(0)
        .to(device)
    )

    coarse_list = []
    subtype_list = []

    for model in models:
        (
            coarse_logits,
            subtype_logits,
        ) = model(tensor)

        coarse_raw = (
            torch.softmax(
                coarse_logits,
                dim=1,
            )[0]
            .cpu()
            .numpy()
        )
        subtype_raw = (
            torch.softmax(
                subtype_logits,
                dim=1,
            )[0]
            .cpu()
            .numpy()
        )

        coarse_list.append(
            temperature_scale_one(
                coarse_raw,
                float(
                    calibration[
                        "coarse_temperature"
                    ]
                ),
            )
        )
        subtype_list.append(
            temperature_scale_one(
                subtype_raw,
                float(
                    calibration[
                        "subtype_temperature"
                    ]
                ),
            )
        )

    return (
        np.mean(
            coarse_list,
            axis=0,
        ),
        np.mean(
            subtype_list,
            axis=0,
        ),
    )


@torch.no_grad()
def predict_canine(
    models,
    pure_path: Path,
    oblique_path: Path,
    side: str,
    height: int,
    width: int,
    device: torch.device,
    calibration: dict[str, Any],
):
    transform = PairViewTransform(
        height=height,
        width=width,
        train=False,
    )

    with Image.open(
        pure_path
    ) as source:
        pure = source.convert("RGB")

    with Image.open(
        oblique_path
    ) as source:
        oblique = source.convert(
            "RGB"
        )

    (
        pure_tensor,
        oblique_tensor,
    ) = transform(
        pure,
        oblique,
        side,
    )

    pure_tensor = (
        pure_tensor
        .unsqueeze(0)
        .to(device)
    )
    oblique_tensor = (
        oblique_tensor
        .unsqueeze(0)
        .to(device)
    )

    coarse_list = []
    subtype_list = []

    for model in models:
        (
            coarse_logits,
            subtype_logits,
        ) = model(
            pure_tensor,
            oblique_tensor,
        )

        coarse_raw = (
            torch.softmax(
                coarse_logits,
                dim=1,
            )[0]
            .cpu()
            .numpy()
        )
        subtype_raw = (
            torch.softmax(
                subtype_logits,
                dim=1,
            )[0]
            .cpu()
            .numpy()
        )

        coarse_list.append(
            temperature_scale_one(
                coarse_raw,
                float(
                    calibration[
                        "coarse_temperature"
                    ]
                ),
            )
        )
        subtype_list.append(
            temperature_scale_one(
                subtype_raw,
                float(
                    calibration[
                        "subtype_temperature"
                    ]
                ),
            )
        )

    return (
        np.mean(
            coarse_list,
            axis=0,
        ),
        np.mean(
            subtype_list,
            axis=0,
        ),
    )


def main() -> int:
    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )
    render_dir = (
        args.render_dir
        .expanduser()
        .resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if (
            args.device != "cuda"
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    patients = discover_patients(
        args,
        render_dir,
    )

    model_load_started = time.perf_counter()
    (
        molar_models,
        molar_height,
        molar_width,
    ) = load_ensemble(
        root,
        "molar",
        device,
    )
    (
        canine_models,
        canine_height,
        canine_width,
    ) = load_ensemble(
        root,
        "canine",
        device,
    )
    print(
        f"B2T_MODEL_LOAD sagittal count=10 seconds={time.perf_counter() - model_load_started:.6f}",
        flush=True,
    )

    molar_calibration = (
        load_calibration(
            root,
            "molar",
        )
    )
    canine_calibration = (
        load_calibration(
            root,
            "canine",
        )
    )

    summary_rows = []

    for patient_id in patients:
        sagittal = {}
        details = {}

        for side in (
            "right",
            "left",
        ):
            pure_path = (
                render_dir
                / (
                    f"{patient_id}_{side}"
                    "_pure_side.png"
                )
            )
            oblique_path = (
                render_dir
                / (
                    f"{patient_id}_{side}"
                    "_canine_oblique_45.png"
                )
            )

            if not pure_path.exists():
                raise FileNotFoundError(
                    pure_path
                )
            if not oblique_path.exists():
                raise FileNotFoundError(
                    oblique_path
                )

            (
                molar_coarse,
                molar_subtype,
            ) = predict_molar(
                molar_models,
                pure_path,
                side,
                molar_height,
                molar_width,
                device,
                molar_calibration,
            )

            (
                molar_label,
                molar_detail,
            ) = final_label(
                molar_coarse,
                molar_subtype,
                molar_calibration,
            )

            (
                canine_coarse,
                canine_subtype,
            ) = predict_canine(
                canine_models,
                pure_path,
                oblique_path,
                side,
                canine_height,
                canine_width,
                device,
                canine_calibration,
            )

            (
                canine_label,
                canine_detail,
            ) = final_label(
                canine_coarse,
                canine_subtype,
                canine_calibration,
            )

            sagittal[
                f"{side}_molar"
            ] = molar_label
            sagittal[
                f"{side}_canine"
            ] = canine_label

            details[
                f"{side}_molar"
            ] = molar_detail
            details[
                f"{side}_canine"
            ] = canine_detail

        result = {
            "patient_id": patient_id,
            "1_sagittal_occlusion": (
                sagittal
            ),
            "details": details,
            "policy": {
                "Class II unspecified": (
                    "not actively predicted by "
                    "the visual model in v2"
                ),
                "Not_Mentioned": (
                    "handled downstream by "
                    "report-generation policy"
                ),
            },
        }

        json_path = (
            output_dir
            / f"{patient_id}_sagittal_v3.json"
        )
        json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "patient_id": patient_id,
                **sagittal,
                "json_path": str(
                    json_path
                ),
            }
        )

        print(
            patient_id,
            sagittal,
        )

    summary_path = (
        output_dir
        / "sagittal_v3_summary.csv"
    )
    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print(
        "\nsummary:",
        summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
