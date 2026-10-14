from __future__ import annotations

import json
import argparse
import os
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/outputs/yolo")
DATA_YAML = PROJECT_ROOT / "dataset_fdi24_seed42" / "data.yaml"
PRETRAINED = PROJECT_ROOT / "pretrained" / "yolo11m-seg.pt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUN_NAME = "yolo11m_seg_fdi24_seed42"
RUN_DIR = OUTPUT_ROOT / RUN_NAME


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    global PROJECT_ROOT, DATA_YAML, PRETRAINED, OUTPUT_ROOT, RUN_DIR
    parser = argparse.ArgumentParser(description="Final2 YOLO11m-seg training recipe")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    args = parser.parse_args()
    PROJECT_ROOT = args.project_root.resolve()
    DATA_YAML = PROJECT_ROOT / "dataset_fdi24_seed42/data.yaml"
    PRETRAINED = args.pretrained.resolve()
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_DIR = OUTPUT_ROOT / RUN_NAME
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(PROJECT_ROOT / ".ultralytics"))
    from ultralytics import YOLO, __version__ as ultralytics_version

    if not DATA_YAML.is_file() or not PRETRAINED.is_file():
        raise FileNotFoundError(f"Missing data config or pretrained model: {DATA_YAML}, {PRETRAINED}")
    completion = OUTPUT_ROOT / "completion.json"
    convenient_best = OUTPUT_ROOT / "best_yolo11m_seg_fdi24.pt"
    if completion.is_file() and convenient_best.is_file():
        print(completion.read_text(encoding="utf-8"))
        return

    last_checkpoint = RUN_DIR / "weights" / "last.pt"
    if last_checkpoint.is_file():
        print(f"Resuming partial run from {last_checkpoint}", flush=True)
        model = YOLO(str(last_checkpoint))
        train_result = model.train(resume=True)
    else:
        model = YOLO(str(PRETRAINED))
        train_result = model.train(
            data=str(DATA_YAML),
            task="segment",
            epochs=160,
            patience=30,
            imgsz=768,
            batch=32,
            device=0,
            workers=12,
            project=str(OUTPUT_ROOT),
            name=RUN_NAME,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            cos_lr=True,
            seed=42,
            deterministic=True,
            amp=True,
            cache="ram",
            plots=True,
            save=True,
            save_period=-1,
            max_det=32,
            overlap_mask=True,
            mask_ratio=4,
            degrees=3.0,
            translate=0.05,
            scale=0.25,
            shear=2.0,
            perspective=0.0002,
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.5,
            close_mosaic=15,
            mixup=0.0,
            copy_paste=0.0,
        )

    best_checkpoint = RUN_DIR / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not create {best_checkpoint}")
    best_model = YOLO(str(best_checkpoint))
    validation = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=768,
        batch=32,
        device=0,
        workers=12,
        max_det=32,
        plots=True,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, convenient_best)
    summary = {
        "status": "complete",
        "architecture": "YOLO11m-seg",
        "ultralytics_version": ultralytics_version,
        "classes": 24,
        "class_definition": "FDI 11-16, 21-26, 31-36, 41-46",
        "selection": "Ultralytics best.pt selected on validation fitness",
        "data": str(DATA_YAML),
        "best_checkpoint": str(best_checkpoint),
        "convenient_best_checkpoint": str(convenient_best),
        "validation": jsonable(validation.results_dict),
        "training_save_dir": str(getattr(train_result, "save_dir", RUN_DIR)),
    }
    completion.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
