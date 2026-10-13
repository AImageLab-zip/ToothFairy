#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_finetune_curve_targeted.py

Targeted CNN fine-tuning after 04 frozen-encoder screening.

We deliberately fine-tune ONLY the configurations justified by 04:

Spee
----
Winner in frozen screening:
    3D + Spee STL renders + intraoral side photos

This script fine-tunes:
    exact intraoral_2 + intraoral_3
    +
    spee_left.png + spee_right.png
    +
    101-D Spee geometry

Architecture:
    photo ResNet18 encoder shared by intraoral_2/3
    render ResNet18 encoder shared by spee_left/right
    each image-pair represented by:
        [mean(emb_a, emb_b), abs(emb_a - emb_b)]
    +
    3D MLP
    -> fusion head

Wilson
------
Frozen screening showed essentially no gain from generic image embeddings.
We run ONE controlled test only:

    wilson_frontal.png + wilson_oblique.png
    +
    220-D Wilson geometry

Architecture:
    one shared ResNet18 render encoder
    fixed semantic view concat [frontal, oblique]
    +
    3D MLP
    -> fusion head

Wilson location
---------------
NOT fine-tuned here. 04 showed no useful multimodal improvement; keep its
selected 3D baseline for now.

Leakage policy
--------------
- GT = photo_report_v3, photo_en only
- same strict common-cohort policy as 04
- 5 outer patient folds
- outer validation NEVER used for early stopping
- each outer-train fold gets an INNER stratified holdout used only for:
      epoch selection / early stopping
- 3D imputer + scaler fit on INNER-TRAIN only
- outer validation predicted exactly once with the best inner-selected model
- no label information is used by test preprocessing

Rare-label model policy
-----------------------
Spee training:
    Decreased     -> Reversed
    Not_Mentioned -> Not_Assessable

Wilson training:
    Decreased     -> Normal
    Not_Mentioned -> Not_Assessable

FINAL metrics are always calculated against the ORIGINAL raw GT strings, so
collapsed rare labels are still counted wrong unless the prediction exactly
matches the raw GT.

Fine-tuning policy
------------------
- ResNet18 ImageNet initialization
- layers conv1..layer3 frozen
- layer4 trainable with small LR
- projection/fusion head trainable with larger LR
- fixed hyperparameters, no outer-validation tuning
- AMP on CUDA
- early stopping on INNER exact-field Accuracy

Outputs
-------
evaluation_finetune_v1/
    curve_finetune_model_comparison_v1.csv
    curve_finetune_fold_metrics_v1.csv
    curve_finetune_oof_predictions_v1.csv
    curve_finetune_confusions_v1.json
    curve_finetune_summary_v1.json
    checkpoints/
        spee_fold0.pt ... spee_fold4.pt
        wilson_fold0.pt ... wilson_fold4.pt

Recommended first run
---------------------
# Spee only (priority):
python 05_finetune_curve_targeted.py --task spee

# Then Wilson controlled test:
python 05_finetune_curve_targeted.py --task wilson

# Or both:
python 05_finetune_curve_targeted.py --task all
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageOps
except ImportError as e:
    raise SystemExit("Pillow is required.") from e

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import functional as TF
except ImportError as e:
    raise SystemExit(
        "torch + torchvision are required. Activate odin_baseline."
    ) from e

try:
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Activate odin_baseline."
    ) from e


ROOT = Path(__file__).resolve().parent
DEFAULT_GT = Path("data/cleaned_orthodontic_data_v3.json")
DEFAULT_RAW = Path("data/raw_data")
DEFAULT_GEOM = ROOT / "results" / "curve_geometry_features_v1.csv"
DEFAULT_MANIFEST = ROOT / "multimodal_v3" / "manifest_v1.csv"
DEFAULT_EVAL02 = ROOT / "02_evaluate_curve_geometry.py"
DEFAULT_PREVIOUS = (
    ROOT
    / "evaluation_multimodal_v3"
    / "curve_multimodal_model_comparison_v1.csv"
)
DEFAULT_OUTPUT = ROOT / "evaluation_finetune_v1"

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SPEE_RAW = (
    "Increased",
    "Normal",
    "Reversed",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)
SPEE_MODEL_CLASSES = (
    "Increased",
    "Normal",
    "Reversed",
    "Not_Assessable",
)

WILSON_RAW = (
    "Increased",
    "Normal",
    "Decreased",
    "Not_Assessable",
    "Not_Mentioned",
)
WILSON_MODEL_CLASSES = (
    "Increased",
    "Normal",
    "Not_Assessable",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--task",
        choices=("spee", "wilson", "all"),
        default="all",
    )

    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--geometry-features", type=Path, default=DEFAULT_GEOM)
    p.add_argument("--multimodal-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--eval02-script", type=Path, default=DEFAULT_EVAL02)
    p.add_argument("--previous-comparison", type=Path, default=DEFAULT_PREVIOUS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument(
        "--inner-val-fraction",
        type=float,
        default=0.15,
    )
    p.add_argument("--seed", type=int, default=20260810)

    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)

    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable mild training augmentation.",
    )
    p.add_argument(
        "--no-save-checkpoints",
        action="store_true",
    )

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducible enough while retaining cuDNN performance.
    torch.backends.cudnn.benchmark = True


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_eval02(mod):
    required = (
        "load_gt",
        "bool_value",
        "spee_compact_columns",
        "add_spee_relative_features",
        "wilson_compact_columns",
        "wilson_extended_columns",
        "safe_outer_splits",
    )
    missing = [
        x
        for x in required
        if not hasattr(mod, x)
    ]
    if missing:
        raise ImportError(
            f"02_evaluate_curve_geometry.py missing: {missing}"
        )


def image_is_decodable(path):
    if path is None:
        return False

    p = Path(path)
    if not p.exists() or not p.is_file():
        return False

    try:
        with Image.open(p) as im:
            im.verify()

        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im.convert("RGB").load()

        return True
    except Exception:
        return False


def exact_photo_by_index(
    patient_dir: Path,
    index: int,
):
    d = patient_dir / "intraoral-photo"
    if not d.exists():
        return None

    pat = re.compile(
        rf"^intraoral[_-]?{int(index)}$",
        flags=re.I,
    )

    hits = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in PHOTO_EXTS:
            continue
        if pat.match(p.stem):
            hits.append(p)

    for p in sorted(
        hits,
        key=lambda x: x.name.lower(),
    ):
        if image_is_decodable(p):
            return p.resolve()

    return None


def add_strict_photo_paths(
    df,
    raw_root,
):
    p1 = []
    p2 = []
    p3 = []

    for pid in df["patient_id"].astype(str):
        d = raw_root / pid

        a = exact_photo_by_index(d, 1)
        b = exact_photo_by_index(d, 2)
        c = exact_photo_by_index(d, 3)

        p1.append(
            "" if a is None else str(a)
        )
        p2.append(
            "" if b is None else str(b)
        )
        p3.append(
            "" if c is None else str(c)
        )

    out = df.copy()

    out["strict_photo_1_path"] = p1
    out["strict_photo_2_path"] = p2
    out["strict_photo_3_path"] = p3

    out["strict_photo_1_ready"] = (
        out["strict_photo_1_path"].ne("")
    )
    out["strict_photo_2_ready"] = (
        out["strict_photo_2_path"].ne("")
    )
    out["strict_photo_3_ready"] = (
        out["strict_photo_3_path"].ne("")
    )

    out["strict_spee_side_pair_ready"] = (
        out["strict_photo_2_ready"]
        & out["strict_photo_3_ready"]
    )

    return out


def path_pair_decodable(a, b):
    if not a or not b:
        return False

    if str(a).lower() == "nan" or str(b).lower() == "nan":
        return False

    return (
        image_is_decodable(a)
        and image_is_decodable(b)
    )


def prepare_3d_feature_frames(
    merged,
    eval02,
):
    # Exact reconstruction of 02 Spee extended.
    spee_compact = (
        eval02.spee_compact_columns(
            merged
        )
    )
    spee_derived_df, spee_derived_cols = (
        eval02.add_spee_relative_features(
            merged
        )
    )

    spee_3d = pd.concat(
        [
            merged[
                spee_compact
            ].reset_index(drop=True),
            spee_derived_df[
                spee_derived_cols
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    wil_compact = (
        eval02.wilson_compact_columns(
            merged
        )
    )
    wil_extended = (
        eval02.wilson_extended_columns(
            merged,
            wil_compact,
        )
    )

    wilson_3d = (
        merged[
            wil_extended
        ]
        .reset_index(drop=True)
    )

    return spee_3d, wilson_3d


def collapse_spee(x):
    if x == "Decreased":
        return "Reversed"
    if x == "Not_Mentioned":
        return "Not_Assessable"
    return x


def collapse_wilson(x):
    if x == "Decreased":
        return "Normal"
    if x == "Not_Mentioned":
        return "Not_Assessable"
    return x


def exact_metrics(
    y_raw,
    pred,
):
    y_raw = np.asarray(
        y_raw,
        dtype=object,
    )
    pred = np.asarray(
        pred,
        dtype=object,
    )

    labels = sorted(
        set(y_raw.tolist())
        | set(pred.tolist())
    )

    return {
        "n": int(len(y_raw)),
        "accuracy": float(
            accuracy_score(
                y_raw,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_raw,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_raw,
                pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_raw,
                pred,
            )
        ),
    }


def confusion_dict(
    y,
    pred,
):
    labels = sorted(
        set(np.asarray(y, dtype=object).tolist())
        | set(np.asarray(pred, dtype=object).tolist())
    )
    cm = confusion_matrix(
        y,
        pred,
        labels=labels,
    )

    return {
        "labels": labels,
        "matrix": cm.astype(int).tolist(),
    }


def letterbox_pil(
    im: Image.Image,
    size=224,
):
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")

    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image size")

    scale = min(
        size / float(w),
        size / float(h),
    )
    nw = max(
        1,
        int(round(w * scale)),
    )
    nh = max(
        1,
        int(round(h * scale)),
    )

    im = im.resize(
        (nw, nh),
        resample=Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        color=(255, 255, 255),
    )
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(
        im,
        (left, top),
    )

    return canvas


class ImagePreprocessor:
    def __init__(
        self,
        *,
        train,
        domain,
        augment,
    ):
        self.train = bool(train)
        self.domain = domain
        self.augment = bool(augment)

        # Mild augmentation only. No horizontal flip: AP/LR orientation may
        # contain task information.
        if (
            self.train
            and self.augment
            and self.domain == "photo"
        ):
            self.photo_jitter = T.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            )
        else:
            self.photo_jitter = None

        if self.train and self.augment:
            self.affine = T.RandomAffine(
                degrees=3.0,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=255,
            )
        else:
            self.affine = None

    def __call__(
        self,
        path,
    ):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")

            if self.photo_jitter is not None:
                im = self.photo_jitter(im)

            im = letterbox_pil(
                im,
                size=224,
            )

            if self.affine is not None:
                im = self.affine(im)

            x = TF.to_tensor(im)
            x = TF.normalize(
                x,
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            )

            return x


class CurveFineTuneDataset(Dataset):
    def __init__(
        self,
        frame,
        x3d,
        y_model,
        y_raw,
        *,
        task,
        train,
        augment,
    ):
        self.frame = (
            frame
            .reset_index(drop=True)
        )
        self.x3d = np.asarray(
            x3d,
            dtype=np.float32,
        )
        self.y_model = np.asarray(
            y_model,
            dtype=np.int64,
        )
        self.y_raw = np.asarray(
            y_raw,
            dtype=object,
        )
        self.task = task

        self.photo_proc = (
            ImagePreprocessor(
                train=train,
                domain="photo",
                augment=augment,
            )
        )
        self.render_proc = (
            ImagePreprocessor(
                train=train,
                domain="render",
                augment=augment,
            )
        )

        if not (
            len(self.frame)
            == len(self.x3d)
            == len(self.y_model)
            == len(self.y_raw)
        ):
            raise ValueError(
                "Dataset lengths do not match"
            )

    def __len__(self):
        return len(self.frame)

    def __getitem__(
        self,
        i,
    ):
        r = self.frame.iloc[i]

        if self.task == "spee":
            photo_a = self.photo_proc(
                r["strict_photo_2_path"]
            )
            photo_b = self.photo_proc(
                r["strict_photo_3_path"]
            )
            render_a = self.render_proc(
                r["spee_left_render_path"]
            )
            render_b = self.render_proc(
                r["spee_right_render_path"]
            )

            return {
                "photo_a": photo_a,
                "photo_b": photo_b,
                "render_a": render_a,
                "render_b": render_b,
                "x3d": torch.from_numpy(
                    self.x3d[i]
                ),
                "y": int(
                    self.y_model[i]
                ),
                "raw": str(
                    self.y_raw[i]
                ),
                "patient_id": str(
                    r["patient_id"]
                ),
            }

        if self.task == "wilson":
            render_a = self.render_proc(
                r["wilson_frontal_render_path"]
            )
            render_b = self.render_proc(
                r["wilson_oblique_render_path"]
            )

            return {
                "render_a": render_a,
                "render_b": render_b,
                "x3d": torch.from_numpy(
                    self.x3d[i]
                ),
                "y": int(
                    self.y_model[i]
                ),
                "raw": str(
                    self.y_raw[i]
                ),
                "patient_id": str(
                    r["patient_id"]
                ),
            }

        raise ValueError(
            self.task
        )


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        *,
        train_layer4=True,
    ):
        super().__init__()

        backbone = resnet18(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.layer4 = backbone.layer4
        self.pool = backbone.avgpool

        for p in self.stem.parameters():
            p.requires_grad_(False)

        for p in self.layer4.parameters():
            p.requires_grad_(
                bool(train_layer4)
            )

    def forward(
        self,
        x,
    ):
        # Frozen stem stays eval-like through BatchNorm because the enclosing
        # model's train() will otherwise switch it. We enforce no-grad and
        # explicitly keep stem in eval() through train().
        with torch.no_grad():
            x = self.stem(x)

        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)

    def train(
        self,
        mode=True,
    ):
        super().train(mode)
        self.stem.eval()
        return self


def invariant_pair(
    a,
    b,
):
    return torch.cat(
        [
            0.5 * (a + b),
            torch.abs(a - b),
        ],
        dim=1,
    )


class SpeeFusionNet(nn.Module):
    def __init__(
        self,
        d3,
        n_classes,
        dropout,
    ):
        super().__init__()

        # Separate encoders because real photos and synthetic STL renders are
        # different domains.
        self.photo_encoder = (
            ResNet18Encoder(
                train_layer4=True
            )
        )
        self.render_encoder = (
            ResNet18Encoder(
                train_layer4=True
            )
        )

        self.photo_proj = nn.Sequential(
            nn.Linear(
                1024,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.render_proj = nn.Sequential(
            nn.Linear(
                1024,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.geom = nn.Sequential(
            nn.Linear(
                d3,
                128,
            ),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(
                dropout * 0.67
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                256 + 256 + 128,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                n_classes,
            ),
        )

    def forward(
        self,
        photo_a,
        photo_b,
        render_a,
        render_b,
        x3d,
    ):
        pa = self.photo_encoder(
            photo_a
        )
        pb = self.photo_encoder(
            photo_b
        )
        p = self.photo_proj(
            invariant_pair(
                pa,
                pb,
            )
        )

        ra = self.render_encoder(
            render_a
        )
        rb = self.render_encoder(
            render_b
        )
        r = self.render_proj(
            invariant_pair(
                ra,
                rb,
            )
        )

        g = self.geom(
            x3d
        )

        return self.classifier(
            torch.cat(
                [
                    p,
                    r,
                    g,
                ],
                dim=1,
            )
        )

    def backbone_parameters(
        self,
    ):
        return list(
            self.photo_encoder.layer4.parameters()
        ) + list(
            self.render_encoder.layer4.parameters()
        )

    def head_parameters(
        self,
    ):
        backbone_ids = {
            id(p)
            for p in self.backbone_parameters()
        }
        return [
            p
            for p in self.parameters()
            if p.requires_grad
            and id(p) not in backbone_ids
        ]


class WilsonFusionNet(nn.Module):
    def __init__(
        self,
        d3,
        n_classes,
        dropout,
    ):
        super().__init__()

        self.render_encoder = (
            ResNet18Encoder(
                train_layer4=True
            )
        )

        # Frontal / oblique have fixed semantic meaning, so concatenate.
        self.render_proj = nn.Sequential(
            nn.Linear(
                1024,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.geom = nn.Sequential(
            nn.Linear(
                d3,
                128,
            ),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(
                dropout * 0.67
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                256 + 128,
                192,
            ),
            nn.LayerNorm(192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                192,
                n_classes,
            ),
        )

    def forward(
        self,
        render_a,
        render_b,
        x3d,
    ):
        a = self.render_encoder(
            render_a
        )
        b = self.render_encoder(
            render_b
        )

        r = self.render_proj(
            torch.cat(
                [a, b],
                dim=1,
            )
        )
        g = self.geom(
            x3d
        )

        return self.classifier(
            torch.cat(
                [r, g],
                dim=1,
            )
        )

    def backbone_parameters(
        self,
    ):
        return list(
            self.render_encoder.layer4.parameters()
        )

    def head_parameters(
        self,
    ):
        backbone_ids = {
            id(p)
            for p in self.backbone_parameters()
        }
        return [
            p
            for p in self.parameters()
            if p.requires_grad
            and id(p) not in backbone_ids
        ]


def build_model(
    task,
    d3,
    n_classes,
    dropout,
):
    if task == "spee":
        return SpeeFusionNet(
            d3,
            n_classes,
            dropout,
        )

    if task == "wilson":
        return WilsonFusionNet(
            d3,
            n_classes,
            dropout,
        )

    raise ValueError(task)


def forward_batch(
    model,
    batch,
    task,
    device,
):
    x3d = batch["x3d"].to(
        device,
        non_blocking=True,
    )

    if task == "spee":
        return model(
            batch["photo_a"].to(
                device,
                non_blocking=True,
            ),
            batch["photo_b"].to(
                device,
                non_blocking=True,
            ),
            batch["render_a"].to(
                device,
                non_blocking=True,
            ),
            batch["render_b"].to(
                device,
                non_blocking=True,
            ),
            x3d,
        )

    if task == "wilson":
        return model(
            batch["render_a"].to(
                device,
                non_blocking=True,
            ),
            batch["render_b"].to(
                device,
                non_blocking=True,
            ),
            x3d,
        )

    raise ValueError(task)


def fit_3d_preprocessor(
    train_df,
):
    imp = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )
    x = imp.fit_transform(
        train_df
    )

    scaler = StandardScaler()
    x = scaler.fit_transform(
        x
    )

    return imp, scaler


def transform_3d(
    df,
    imp,
    scaler,
):
    return scaler.transform(
        imp.transform(df)
    ).astype(np.float32)


def make_inner_split(
    y_model_labels,
    val_fraction,
    seed,
):
    y = np.asarray(
        y_model_labels,
        dtype=object,
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(
            val_fraction
        ),
        random_state=seed,
    )

    tr, va = next(
        splitter.split(
            np.zeros(
                (len(y), 1)
            ),
            y,
        )
    )

    return tr, va


def make_optimizer(
    model,
    args,
):
    groups = []

    backbone = [
        p
        for p in model.backbone_parameters()
        if p.requires_grad
    ]
    head = [
        p
        for p in model.head_parameters()
        if p.requires_grad
    ]

    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": float(
                    args.backbone_lr
                ),
            }
        )
    if head:
        groups.append(
            {
                "params": head,
                "lr": float(
                    args.head_lr
                ),
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=float(
            args.weight_decay
        ),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler_amp,
    task,
    device,
):
    model.train()

    total_loss = 0.0
    total_n = 0

    use_amp = (
        device.type == "cuda"
    )

    for batch in loader:
        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            logits = forward_batch(
                model,
                batch,
                task,
                device,
            )
            loss = criterion(
                logits,
                y,
            )

        scaler_amp.scale(
            loss
        ).backward()

        scaler_amp.unscale_(
            optimizer
        )
        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=5.0,
        )

        scaler_amp.step(
            optimizer
        )
        scaler_amp.update()

        n = int(y.shape[0])
        total_loss += (
            float(loss.item()) * n
        )
        total_n += n

    return (
        total_loss
        / max(total_n, 1)
    )


@torch.inference_mode()
def predict_loader(
    model,
    loader,
    task,
    device,
    class_names,
):
    model.eval()

    raw_gt = []
    pred_labels = []
    patient_ids = []

    use_amp = (
        device.type == "cuda"
    )

    for batch in loader:
        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            logits = forward_batch(
                model,
                batch,
                task,
                device,
            )

        pred_idx = (
            logits.argmax(
                dim=1
            )
            .detach()
            .cpu()
            .numpy()
        )

        pred_labels.extend(
            [
                class_names[int(i)]
                for i in pred_idx
            ]
        )
        raw_gt.extend(
            list(batch["raw"])
        )
        patient_ids.extend(
            list(batch["patient_id"])
        )

    return (
        np.asarray(
            patient_ids,
            dtype=object,
        ),
        np.asarray(
            raw_gt,
            dtype=object,
        ),
        np.asarray(
            pred_labels,
            dtype=object,
        ),
    )


def save_checkpoint(
    path,
    *,
    model,
    task,
    fold,
    epoch,
    class_names,
    feature_columns,
    imputer,
    scaler,
    metrics_inner,
    args,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save sklearn preprocessing statistics explicitly instead of pickle so the
    # checkpoint remains transparent and easier to port.
    preprocessing = {
        "imputer_statistics": (
            imputer.statistics_.tolist()
        ),
        "imputer_indicator_features": (
            imputer.indicator_.features_.tolist()
            if hasattr(
                imputer,
                "indicator_",
            )
            and imputer.indicator_ is not None
            else []
        ),
        "scaler_mean": (
            scaler.mean_.tolist()
        ),
        "scaler_scale": (
            scaler.scale_.tolist()
        ),
    }

    torch.save(
        {
            "version": (
                "curve-targeted-finetune-v1"
            ),
            "task": task,
            "fold": int(fold),
            "best_epoch": int(epoch),
            "class_names": list(
                class_names
            ),
            "feature_columns": list(
                feature_columns
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "preprocessing": preprocessing,
            "inner_metrics": (
                metrics_inner
            ),
            "training_config": {
                "head_lr": float(
                    args.head_lr
                ),
                "backbone_lr": float(
                    args.backbone_lr
                ),
                "weight_decay": float(
                    args.weight_decay
                ),
                "dropout": float(
                    args.dropout
                ),
                "batch_size": int(
                    args.batch_size
                ),
                "epochs": int(
                    args.epochs
                ),
                "augmentation": bool(
                    not args.no_augment
                ),
            },
        },
        path,
    )


def run_task(
    *,
    task,
    frame,
    x3d_df,
    y_raw,
    eval02,
    args,
    device,
):
    if task == "spee":
        class_names = list(
            SPEE_MODEL_CLASSES
        )
        collapse_fn = (
            collapse_spee
        )
        outer_seed = int(
            args.seed
        )
    elif task == "wilson":
        class_names = list(
            WILSON_MODEL_CLASSES
        )
        collapse_fn = (
            collapse_wilson
        )
        # Match 04 Wilson outer-fold seed policy.
        outer_seed = int(
            args.seed + 100
        )
    else:
        raise ValueError(task)

    class_to_idx = {
        c: i
        for i, c in enumerate(
            class_names
        )
    }

    y_raw = np.asarray(
        y_raw,
        dtype=object,
    )
    y_collapsed = np.asarray(
        [
            collapse_fn(x)
            for x in y_raw
        ],
        dtype=object,
    )

    for x in np.unique(
        y_collapsed
    ):
        if x not in class_to_idx:
            raise ValueError(
                f"{task}: collapsed label not in class list: {x}"
            )

    outer_splits = (
        eval02.safe_outer_splits(
            y_collapsed,
            args.outer_folds,
            outer_seed,
        )
    )

    oof_pred = np.empty(
        len(frame),
        dtype=object,
    )
    oof_fold = np.full(
        len(frame),
        -1,
        dtype=int,
    )

    fold_rows = []

    for fold, (
        outer_tr,
        outer_va,
    ) in enumerate(
        outer_splits
    ):
        fold_seed = (
            int(args.seed)
            + 10000
            + fold
            + (
                0
                if task == "spee"
                else 1000
            )
        )
        set_seed(
            fold_seed
        )

        outer_train_frame = (
            frame.iloc[
                outer_tr
            ]
            .reset_index(drop=True)
        )
        outer_train_x3d = (
            x3d_df.iloc[
                outer_tr
            ]
            .reset_index(drop=True)
        )
        outer_train_raw = (
            y_raw[
                outer_tr
            ]
        )
        outer_train_collapsed = (
            y_collapsed[
                outer_tr
            ]
        )

        inner_tr_local, inner_va_local = (
            make_inner_split(
                outer_train_collapsed,
                args.inner_val_fraction,
                fold_seed,
            )
        )

        # Fit 3D preprocessing ONLY on inner training.
        imp, scaler = (
            fit_3d_preprocessor(
                outer_train_x3d.iloc[
                    inner_tr_local
                ]
            )
        )

        inner_train_x = (
            transform_3d(
                outer_train_x3d.iloc[
                    inner_tr_local
                ],
                imp,
                scaler,
            )
        )
        inner_val_x = (
            transform_3d(
                outer_train_x3d.iloc[
                    inner_va_local
                ],
                imp,
                scaler,
            )
        )
        outer_val_x = (
            transform_3d(
                x3d_df.iloc[
                    outer_va
                ],
                imp,
                scaler,
            )
        )

        inner_train_frame = (
            outer_train_frame.iloc[
                inner_tr_local
            ]
            .reset_index(drop=True)
        )
        inner_val_frame = (
            outer_train_frame.iloc[
                inner_va_local
            ]
            .reset_index(drop=True)
        )
        outer_val_frame = (
            frame.iloc[
                outer_va
            ]
            .reset_index(drop=True)
        )

        inner_train_raw = (
            outer_train_raw[
                inner_tr_local
            ]
        )
        inner_val_raw = (
            outer_train_raw[
                inner_va_local
            ]
        )
        outer_val_raw = (
            y_raw[
                outer_va
            ]
        )

        inner_train_labels = np.asarray(
            [
                collapse_fn(x)
                for x in inner_train_raw
            ],
            dtype=object,
        )
        inner_val_labels = np.asarray(
            [
                collapse_fn(x)
                for x in inner_val_raw
            ],
            dtype=object,
        )
        outer_val_labels = np.asarray(
            [
                collapse_fn(x)
                for x in outer_val_raw
            ],
            dtype=object,
        )

        inner_train_y = np.asarray(
            [
                class_to_idx[x]
                for x in inner_train_labels
            ],
            dtype=np.int64,
        )
        inner_val_y = np.asarray(
            [
                class_to_idx[x]
                for x in inner_val_labels
            ],
            dtype=np.int64,
        )
        outer_val_y = np.asarray(
            [
                class_to_idx[x]
                for x in outer_val_labels
            ],
            dtype=np.int64,
        )

        ds_tr = CurveFineTuneDataset(
            inner_train_frame,
            inner_train_x,
            inner_train_y,
            inner_train_raw,
            task=task,
            train=True,
            augment=(
                not args.no_augment
            ),
        )
        ds_iv = CurveFineTuneDataset(
            inner_val_frame,
            inner_val_x,
            inner_val_y,
            inner_val_raw,
            task=task,
            train=False,
            augment=False,
        )
        ds_ov = CurveFineTuneDataset(
            outer_val_frame,
            outer_val_x,
            outer_val_y,
            outer_val_raw,
            task=task,
            train=False,
            augment=False,
        )

        loader_tr = DataLoader(
            ds_tr,
            batch_size=int(
                args.batch_size
            ),
            shuffle=True,
            num_workers=int(
                args.num_workers
            ),
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                int(
                    args.num_workers
                ) > 0
            ),
            drop_last=False,
        )
        loader_iv = DataLoader(
            ds_iv,
            batch_size=int(
                args.batch_size
            ),
            shuffle=False,
            num_workers=int(
                args.num_workers
            ),
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                int(
                    args.num_workers
                ) > 0
            ),
        )
        loader_ov = DataLoader(
            ds_ov,
            batch_size=int(
                args.batch_size
            ),
            shuffle=False,
            num_workers=int(
                args.num_workers
            ),
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                int(
                    args.num_workers
                ) > 0
            ),
        )

        model = build_model(
            task,
            d3=int(
                inner_train_x.shape[1]
            ),
            n_classes=len(
                class_names
            ),
            dropout=float(
                args.dropout
            ),
        ).to(
            device
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = make_optimizer(
            model,
            args,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    int(args.epochs),
                    1,
                ),
            )
        )

        amp_scaler = (
            torch.cuda.amp.GradScaler(
                enabled=(
                    device.type == "cuda"
                )
            )
        )

        best_key = None
        best_state = None
        best_epoch = 0
        best_inner_metrics = None
        bad_epochs = 0

        print()
        print("=" * 104)
        print(
            f"{task.upper()} | Fold {fold+1}/{len(outer_splits)}"
        )
        print("=" * 104)
        print(
            f"outer train / val : "
            f"{len(outer_tr)} / {len(outer_va)}"
        )
        print(
            f"inner train / val : "
            f"{len(inner_tr_local)} / {len(inner_va_local)}"
        )
        print(
            f"3D input dim      : "
            f"{inner_train_x.shape[1]}"
        )
        print(
            f"train labels      : "
            f"{dict(Counter(inner_train_labels.tolist()))}"
        )
        print(
            f"inner-val labels  : "
            f"{dict(Counter(inner_val_labels.tolist()))}"
        )

        t_fold = time.time()

        for epoch in range(
            1,
            int(args.epochs) + 1,
        ):
            loss = train_one_epoch(
                model,
                loader_tr,
                optimizer,
                criterion,
                amp_scaler,
                task,
                device,
            )

            (
                _,
                inner_raw_gt,
                inner_pred,
            ) = predict_loader(
                model,
                loader_iv,
                task,
                device,
                class_names,
            )

            m = exact_metrics(
                inner_raw_gt,
                inner_pred,
            )

            key = (
                m["accuracy"],
                m["macro_f1"],
                m["balanced_accuracy"],
                m["weighted_f1"],
            )

            improved = (
                best_key is None
                or key > best_key
            )

            if improved:
                best_key = key
                best_state = copy.deepcopy(
                    {
                        k: v.detach().cpu()
                        for k, v
                        in model.state_dict().items()
                    }
                )
                best_epoch = int(
                    epoch
                )
                best_inner_metrics = dict(
                    m
                )
                bad_epochs = 0
            else:
                bad_epochs += 1

            scheduler.step()

            print(
                f"[{task}][Fold {fold+1}] "
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"loss={loss:.4f} | "
                f"inner Acc={m['accuracy']:.4f} | "
                f"MacroF1={m['macro_f1']:.4f} | "
                f"best={best_key[0]:.4f}@{best_epoch:02d}"
            )

            if (
                epoch >= int(
                    args.min_epochs
                )
                and bad_epochs
                >= int(
                    args.patience
                )
            ):
                print(
                    f"[{task}][Fold {fold+1}] "
                    f"Early stop at epoch {epoch}; "
                    f"best epoch={best_epoch}"
                )
                break

        if best_state is None:
            raise RuntimeError(
                f"{task} fold {fold}: no best model state"
            )

        model.load_state_dict(
            best_state
        )
        model.to(
            device
        )

        (
            patient_ids,
            outer_raw_gt_check,
            outer_pred,
        ) = predict_loader(
            model,
            loader_ov,
            task,
            device,
            class_names,
        )

        if not np.array_equal(
            outer_raw_gt_check,
            outer_val_raw.astype(object),
        ):
            raise RuntimeError(
                f"{task} fold {fold}: outer GT ordering mismatch"
            )

        oof_pred[
            outer_va
        ] = outer_pred
        oof_fold[
            outer_va
        ] = int(
            fold
        )

        outer_metrics = (
            exact_metrics(
                outer_val_raw,
                outer_pred,
            )
        )

        elapsed = (
            time.time() - t_fold
        )

        fold_rows.append(
            {
                "task": task,
                "model": (
                    "targeted_finetune"
                ),
                "fold": int(
                    fold
                ),
                "outer_n": int(
                    len(outer_va)
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "inner_best_accuracy": float(
                    best_inner_metrics[
                        "accuracy"
                    ]
                ),
                "inner_best_macro_f1": float(
                    best_inner_metrics[
                        "macro_f1"
                    ]
                ),
                "outer_accuracy": float(
                    outer_metrics[
                        "accuracy"
                    ]
                ),
                "outer_macro_f1": float(
                    outer_metrics[
                        "macro_f1"
                    ]
                ),
                "outer_weighted_f1": float(
                    outer_metrics[
                        "weighted_f1"
                    ]
                ),
                "outer_balanced_accuracy": float(
                    outer_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "elapsed_seconds": float(
                    elapsed
                ),
            }
        )

        print(
            f"[{task}][Fold {fold+1}] "
            f"OUTER: Acc={outer_metrics['accuracy']:.4f} | "
            f"MacroF1={outer_metrics['macro_f1']:.4f} | "
            f"WeightedF1={outer_metrics['weighted_f1']:.4f} | "
            f"time={elapsed/60:.1f} min"
        )

        if not args.no_save_checkpoints:
            ckpt = (
                args.output_dir
                / "checkpoints"
                / f"{task}_fold{fold}.pt"
            )
            save_checkpoint(
                ckpt,
                model=model,
                task=task,
                fold=fold,
                epoch=best_epoch,
                class_names=class_names,
                feature_columns=list(
                    x3d_df.columns
                ),
                imputer=imp,
                scaler=scaler,
                metrics_inner=best_inner_metrics,
                args=args,
            )

        # Release GPU memory before next fold.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(
        oof_fold < 0
    ):
        raise RuntimeError(
            f"{task}: incomplete OOF predictions"
        )

    overall = exact_metrics(
        y_raw,
        oof_pred,
    )

    oof = pd.DataFrame(
        {
            "patient_id": (
                frame[
                    "patient_id"
                ]
                .astype(str)
                .to_numpy()
            ),
            "task": task,
            "outer_fold": oof_fold,
            "gt": y_raw,
            "pred_targeted_finetune": (
                oof_pred
            ),
            "correct": (
                y_raw == oof_pred
            ),
        }
    )

    return (
        overall,
        pd.DataFrame(
            fold_rows
        ),
        oof,
        confusion_dict(
            y_raw,
            oof_pred,
        ),
    )


def get_previous_reference(
    path,
    task,
):
    if not path.exists():
        return []

    df = pd.read_csv(
        path
    )

    sub = df.loc[
        df["task"].eq(task)
    ].copy()

    keep_models = (
        ["selected_3d"]
        if task == "wilson"
        else [
            "selected_3d",
            "3d_render_photo__direct_logistic",
        ]
    )

    rows = []

    for model in keep_models:
        x = sub.loc[
            sub["model"].eq(
                model
            )
        ]
        if x.empty:
            continue

        r = x.iloc[0]

        rows.append(
            {
                "task": task,
                "model": (
                    f"reference_04__{model}"
                ),
                "source": "04_multimodal_v3",
                "n": int(
                    r["n"]
                ),
                "accuracy": float(
                    r["accuracy"]
                ),
                "macro_f1": float(
                    r["macro_f1"]
                ),
                "weighted_f1": float(
                    r["weighted_f1"]
                ),
                "balanced_accuracy": float(
                    r["balanced_accuracy"]
                ),
            }
        )

    return rows


def main():
    args = parse_args()

    args.gt = (
        args.gt.expanduser().resolve()
    )
    args.raw_root = (
        args.raw_root.expanduser().resolve()
    )
    args.geometry_features = (
        args.geometry_features
        .expanduser()
        .resolve()
    )
    args.multimodal_manifest = (
        args.multimodal_manifest
        .expanduser()
        .resolve()
    )
    args.eval02_script = (
        args.eval02_script
        .expanduser()
        .resolve()
    )
    args.previous_comparison = (
        args.previous_comparison
        .expanduser()
        .resolve()
    )
    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        int(args.seed)
    )
    device = resolve_device(
        args.device
    )

    eval02 = load_module(
        args.eval02_script,
        "_curve_eval02_for_finetune",
    )
    require_eval02(
        eval02
    )

    gt = eval02.load_gt(
        args.gt
    )
    geom = pd.read_csv(
        args.geometry_features,
        low_memory=False,
    )
    manifest = pd.read_csv(
        args.multimodal_manifest,
        low_memory=False,
    )

    merged = (
        gt.loc[
            gt["gt_source"].eq(
                "photo_en"
            )
        ]
        .merge(
            geom,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            manifest,
            on="patient_id",
            how="left",
            suffixes=("", "__mm"),
            validate="one_to_one",
        )
        .sort_values(
            "patient_id"
        )
        .reset_index(drop=True)
    )

    merged = add_strict_photo_paths(
        merged,
        args.raw_root,
    )

    # Decode-aware render readiness.
    spee_render_ready = np.asarray(
        [
            path_pair_decodable(
                a,
                b,
            )
            for a, b in zip(
                merged[
                    "spee_left_render_path"
                ].astype(str),
                merged[
                    "spee_right_render_path"
                ].astype(str),
            )
        ],
        dtype=bool,
    )
    wilson_render_ready = np.asarray(
        [
            path_pair_decodable(
                a,
                b,
            )
            for a, b in zip(
                merged[
                    "wilson_frontal_render_path"
                ].astype(str),
                merged[
                    "wilson_oblique_render_path"
                ].astype(str),
            )
        ],
        dtype=bool,
    )

    spee_geom_ready = (
        merged[
            "spee_geometry_any_valid"
        ]
        .map(
            eval02.bool_value
        )
        .to_numpy(bool)
    )
    wilson_geom_ready = (
        merged[
            "wilson_geometry_any_valid"
        ]
        .map(
            eval02.bool_value
        )
        .to_numpy(bool)
    )

    spee_mask = (
        merged[
            "gt_curve_spee"
        ]
        .isin(
            SPEE_RAW
        )
        .to_numpy(bool)
        & spee_geom_ready
        & spee_render_ready
        & merged[
            "strict_spee_side_pair_ready"
        ].to_numpy(bool)
    )

    # Intentionally keep the SAME 04 Wilson common cohort for direct comparison:
    # geometry + render + exact decodable intraoral_1, even though this targeted
    # Wilson model itself does not consume the photo.
    wilson_mask = (
        merged[
            "gt_curve_wilson"
        ]
        .isin(
            WILSON_RAW
        )
        .to_numpy(bool)
        & wilson_geom_ready
        & wilson_render_ready
        & merged[
            "strict_photo_1_ready"
        ].to_numpy(bool)
    )

    spee_3d_all, wilson_3d_all = (
        prepare_3d_feature_frames(
            merged,
            eval02,
        )
    )

    print("=" * 104)
    print("Curve targeted fine-tuning v1")
    print("=" * 104)
    print(
        f"device              : {device}"
    )
    print(
        f"photo_en rows       : {len(merged)}"
    )
    print(
        f"Spee strict cohort  : {int(spee_mask.sum())}"
    )
    print(
        f"Wilson strict cohort: {int(wilson_mask.sum())}"
    )
    print(
        f"epochs              : {args.epochs}"
    )
    print(
        f"batch size          : {args.batch_size}"
    )
    print(
        f"backbone/head LR    : "
        f"{args.backbone_lr:g} / {args.head_lr:g}"
    )
    print(
        f"augmentation        : {not args.no_augment}"
    )
    print("=" * 104)

    requested = (
        ["spee", "wilson"]
        if args.task == "all"
        else [args.task]
    )

    comparison_rows = []
    fold_parts = []
    oof_parts = []
    confusions = {}
    best_results = {}

    for task in requested:
        if task == "spee":
            idx = np.where(
                spee_mask
            )[0]
            frame = (
                merged.iloc[
                    idx
                ]
                .reset_index(drop=True)
            )
            x3d = (
                spee_3d_all.iloc[
                    idx
                ]
                .reset_index(drop=True)
            )
            y_raw = (
                frame[
                    "gt_curve_spee"
                ]
                .astype(str)
                .to_numpy(object)
            )
            model_description = (
                "fine-tuned photo2/3 + Spee render L/R + 3D"
            )
        else:
            idx = np.where(
                wilson_mask
            )[0]
            frame = (
                merged.iloc[
                    idx
                ]
                .reset_index(drop=True)
            )
            x3d = (
                wilson_3d_all.iloc[
                    idx
                ]
                .reset_index(drop=True)
            )
            y_raw = (
                frame[
                    "gt_curve_wilson"
                ]
                .astype(str)
                .to_numpy(object)
            )
            model_description = (
                "fine-tuned Wilson frontal/oblique render + 3D"
            )

        print()
        print(
            f"START TASK: {task.upper()} | "
            f"n={len(frame)} | 3D dims={x3d.shape[1]}"
        )

        (
            overall,
            fold_df,
            oof_df,
            confusion,
        ) = run_task(
            task=task,
            frame=frame,
            x3d_df=x3d,
            y_raw=y_raw,
            eval02=eval02,
            args=args,
            device=device,
        )

        row = {
            "task": task,
            "model": (
                "targeted_finetune"
            ),
            "source": "05",
            "description": (
                model_description
            ),
            **overall,
        }
        comparison_rows.append(
            row
        )
        best_results[
            task
        ] = row

        comparison_rows.extend(
            get_previous_reference(
                args.previous_comparison,
                task,
            )
        )

        fold_parts.append(
            fold_df
        )
        oof_parts.append(
            oof_df
        )
        confusions[
            task
        ] = confusion

        print()
        print(
            f"{task.upper()} FINAL OOF | "
            f"Acc={overall['accuracy']:.4f} | "
            f"MacroF1={overall['macro_f1']:.4f} | "
            f"WeightedF1={overall['weighted_f1']:.4f} | "
            f"BalAcc={overall['balanced_accuracy']:.4f}"
        )

    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
        .sort_values(
            [
                "task",
                "accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    fold_df = (
        pd.concat(
            fold_parts,
            ignore_index=True,
        )
        if fold_parts
        else pd.DataFrame()
    )
    oof_df = (
        pd.concat(
            oof_parts,
            ignore_index=True,
        )
        if oof_parts
        else pd.DataFrame()
    )

    comparison_path = (
        args.output_dir
        / "curve_finetune_model_comparison_v1.csv"
    )
    fold_path = (
        args.output_dir
        / "curve_finetune_fold_metrics_v1.csv"
    )
    oof_path = (
        args.output_dir
        / "curve_finetune_oof_predictions_v1.csv"
    )
    confusion_path = (
        args.output_dir
        / "curve_finetune_confusions_v1.json"
    )
    summary_path = (
        args.output_dir
        / "curve_finetune_summary_v1.json"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )
    oof_df.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_path.write_text(
        json.dumps(
            confusions,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": (
            "curve-targeted-finetune-v1"
        ),
        "gt_source": (
            "photo_en_only"
        ),
        "tasks_requested": (
            requested
        ),
        "cohorts": {
            "spee_common_n": int(
                spee_mask.sum()
            ),
            "wilson_common_n": int(
                wilson_mask.sum()
            ),
        },
        "model_policy": {
            "spee": (
                "separate photo/render ResNet18 encoders; "
                "conv1-layer3 frozen; layer4 fine-tuned; "
                "photo2/3 order-invariant pair + Spee render L/R "
                "order-invariant pair + 3D"
            ),
            "wilson": (
                "shared Wilson render ResNet18; conv1-layer3 frozen; "
                "layer4 fine-tuned; frontal/oblique fixed-view concat + 3D"
            ),
            "wilson_location": (
                "not fine-tuned; retain selected 3D baseline"
            ),
        },
        "cv_policy": {
            "outer_folds": int(
                args.outer_folds
            ),
            "inner_validation_fraction": float(
                args.inner_val_fraction
            ),
            "outer_validation_used_for_early_stopping": False,
            "three_d_preprocessing_fit_scope": (
                "inner-train only"
            ),
            "primary_metric": (
                "original-label exact Accuracy"
            ),
        },
        "rare_label_policy": {
            "spee": {
                "Decreased": (
                    "train as Reversed"
                ),
                "Not_Mentioned": (
                    "train as Not_Assessable"
                ),
            },
            "wilson": {
                "Decreased": (
                    "train as Normal"
                ),
                "Not_Mentioned": (
                    "train as Not_Assessable"
                ),
            },
            "evaluation": (
                "always against original raw GT label"
            ),
        },
        "training_config": {
            "epochs": int(
                args.epochs
            ),
            "min_epochs": int(
                args.min_epochs
            ),
            "patience": int(
                args.patience
            ),
            "batch_size": int(
                args.batch_size
            ),
            "head_lr": float(
                args.head_lr
            ),
            "backbone_lr": float(
                args.backbone_lr
            ),
            "weight_decay": float(
                args.weight_decay
            ),
            "dropout": float(
                args.dropout
            ),
            "augmentation": bool(
                not args.no_augment
            ),
            "device": str(
                device
            ),
        },
        "targeted_finetune_results": (
            best_results
        ),
        "selection_rule_after_05": (
            "Compare targeted-finetune OOF Accuracy against the matching "
            "04 strict-common reference. Keep fine-tune only if it actually "
            "adds correctly classified cases; otherwise retain the simpler "
            "04/3D model."
        ),
        "outputs": {
            "comparison": str(
                comparison_path
            ),
            "fold_metrics": str(
                fold_path
            ),
            "oof": str(
                oof_path
            ),
            "confusions": str(
                confusion_path
            ),
            "checkpoints": str(
                args.output_dir
                / "checkpoints"
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("TARGETED FINE-TUNE RESULTS")
    print("=" * 104)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print("Outputs:")
    print(f"  {comparison_path}")
    print(f"  {fold_path}")
    print(f"  {oof_path}")
    print(f"  {confusion_path}")
    print(f"  {summary_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
