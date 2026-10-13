#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode


COARSE_LABELS = [
    "Class I",
    "Class II",
    "Class III",
    "Not_Assessable",
]

SUBTYPE_LABELS = [
    "Class II head-to-head",
    "Class II full",
]

FINAL_LABELS = [
    "Class I",
    "Class II head-to-head",
    "Class II full",
    "Class II unspecified",
    "Class III",
    "Not_Assessable",
]

RAW_ALLOWED = set(FINAL_LABELS) | {
    "Class II",
    "Not_Mentioned",
}

COARSE_TO_ID = {
    label: index
    for index, label in enumerate(COARSE_LABELS)
}

SUBTYPE_TO_ID = {
    label: index
    for index, label in enumerate(SUBTYPE_LABELS)
}


def parse_v3_label(
    raw_label: str,
) -> dict[str, Any]:
    """
    将 v3 原始标签拆成：
      - mention
      - Level 1 coarse
      - Level 2 subtype

    Not_Mentioned:
      不作为视觉临床类别训练。

    Class II unspecified:
      Level 1 = Class II
      Level 2 target = ignore (-1)
    """
    if raw_label not in RAW_ALLOWED:
        raise ValueError(
            f"未知 sagittal label: {raw_label!r}"
        )

    if raw_label == "Not_Mentioned":
        return {
            "mention": 0,
            "coarse_label": "",
            "coarse_id": -1,
            "subtype_label": "",
            "subtype_id": -1,
        }

    if raw_label == "Class I":
        return {
            "mention": 1,
            "coarse_label": "Class I",
            "coarse_id": 0,
            "subtype_label": "",
            "subtype_id": -1,
        }

    if raw_label in {
        "Class II",
        "Class II unspecified",
    }:
        return {
            "mention": 1,
            "coarse_label": "Class II",
            "coarse_id": 1,
            "subtype_label": "Class II unspecified",
            "subtype_id": -1,
        }

    if raw_label == "Class II head-to-head":
        return {
            "mention": 1,
            "coarse_label": "Class II",
            "coarse_id": 1,
            "subtype_label": raw_label,
            "subtype_id": 0,
        }

    if raw_label == "Class II full":
        return {
            "mention": 1,
            "coarse_label": "Class II",
            "coarse_id": 1,
            "subtype_label": raw_label,
            "subtype_id": 1,
        }

    if raw_label == "Class III":
        return {
            "mention": 1,
            "coarse_label": "Class III",
            "coarse_id": 2,
            "subtype_label": "",
            "subtype_id": -1,
        }

    if raw_label == "Not_Assessable":
        return {
            "mention": 1,
            "coarse_label": "Not_Assessable",
            "coarse_id": 3,
            "subtype_label": "",
            "subtype_id": -1,
        }

    raise RuntimeError(
        f"未处理标签: {raw_label}"
    )


def coarse_collapse(
    raw_label: str,
) -> str:
    parsed = parse_v3_label(raw_label)
    if parsed["coarse_id"] < 0:
        return "Not_Mentioned"
    return str(parsed["coarse_label"])


def apply_right_flip(
    image: Image.Image,
    side: str,
) -> Image.Image:
    """
    右侧固定水平翻转，使左右两侧的 anterior/posterior
    视觉方向与之前 stage1_dual_view 一致。

    这不是随机增强。
    """
    if side == "right":
        return image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
    return image


class SingleViewTransform:
    def __init__(
        self,
        height: int = 256,
        width: int = 416,
        train: bool = False,
    ) -> None:
        self.height = height
        self.width = width
        self.train = train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _resize(
        self,
        image: Image.Image,
    ) -> Image.Image:
        return TF.resize(
            image,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def __call__(
        self,
        image: Image.Image,
        side: str,
    ) -> torch.Tensor:
        image = image.convert("RGB")
        image = apply_right_flip(
            image,
            side,
        )
        image = self._resize(image)

        if self.train:
            angle = random.uniform(
                -4.0,
                4.0,
            )
            translate = [
                int(
                    random.uniform(
                        -0.02,
                        0.02,
                    )
                    * self.width
                ),
                int(
                    random.uniform(
                        -0.02,
                        0.02,
                    )
                    * self.height
                ),
            ]
            scale = random.uniform(
                0.96,
                1.04,
            )

            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                fill=[255, 255, 255],
            )

            image = ImageEnhance.Brightness(
                image
            ).enhance(
                random.uniform(
                    0.92,
                    1.08,
                )
            )

            image = ImageEnhance.Contrast(
                image
            ).enhance(
                random.uniform(
                    0.92,
                    1.08,
                )
            )

        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=self.mean,
            std=self.std,
        )


class PairViewTransform:
    """
    canine pure + oblique 使用完全同步的增强参数。
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 416,
        train: bool = False,
    ) -> None:
        self.height = height
        self.width = width
        self.train = train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _prepare(
        self,
        image: Image.Image,
        side: str,
    ) -> Image.Image:
        image = image.convert("RGB")
        image = apply_right_flip(
            image,
            side,
        )
        return TF.resize(
            image,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def __call__(
        self,
        pure: Image.Image,
        oblique: Image.Image,
        side: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        pure = self._prepare(
            pure,
            side,
        )
        oblique = self._prepare(
            oblique,
            side,
        )

        if self.train:
            angle = random.uniform(
                -4.0,
                4.0,
            )
            translate = [
                int(
                    random.uniform(
                        -0.02,
                        0.02,
                    )
                    * self.width
                ),
                int(
                    random.uniform(
                        -0.02,
                        0.02,
                    )
                    * self.height
                ),
            ]
            scale = random.uniform(
                0.96,
                1.04,
            )
            brightness = random.uniform(
                0.92,
                1.08,
            )
            contrast = random.uniform(
                0.92,
                1.08,
            )

            transformed = []
            for image in (
                pure,
                oblique,
            ):
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=(
                        InterpolationMode.BILINEAR
                    ),
                    fill=[255, 255, 255],
                )
                image = ImageEnhance.Brightness(
                    image
                ).enhance(brightness)
                image = ImageEnhance.Contrast(
                    image
                ).enhance(contrast)
                transformed.append(image)

            pure, oblique = transformed

        pure_tensor = TF.normalize(
            TF.to_tensor(pure),
            mean=self.mean,
            std=self.std,
        )
        oblique_tensor = TF.normalize(
            TF.to_tensor(oblique),
            mean=self.mean,
            std=self.std,
        )

        return (
            pure_tensor,
            oblique_tensor,
        )


def build_resnet18(
    pretrained: bool,
) -> tuple[nn.Module, int]:
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )
    backbone = resnet18(
        weights=weights
    )
    feature_dim = (
        backbone.fc.in_features
    )
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class MolarHierarchicalModel(nn.Module):
    """
    pure side
       ↓
    ResNet18
       ├─ coarse: I / II / III / NA
       └─ subtype: head-to-head / full
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        (
            self.backbone,
            feature_dim,
        ) = build_resnet18(
            pretrained=pretrained
        )

        self.shared = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.coarse_head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(256, 4),
        )

        self.subtype_head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        feature = self.backbone(
            image
        )
        feature = self.shared(
            feature
        )
        return (
            self.coarse_head(feature),
            self.subtype_head(feature),
        )

    def set_backbone_trainable(
        self,
        trainable: bool,
    ) -> None:
        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = (
                trainable
            )


class CanineHierarchicalModel(nn.Module):
    """
    pure side + oblique 45°
             ↓
       shared ResNet18
             ↓
    [pure, oblique, |diff|, product]
             ↓
          fusion
        ├─ coarse
        └─ subtype
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        (
            self.backbone,
            feature_dim,
        ) = build_resnet18(
            pretrained=pretrained
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                feature_dim * 4,
                512,
            ),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.coarse_head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(256, 4),
        )

        self.subtype_head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        pure: torch.Tensor,
        oblique: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        pure_feature = self.backbone(
            pure
        )
        oblique_feature = self.backbone(
            oblique
        )

        fused = torch.cat(
            [
                pure_feature,
                oblique_feature,
                torch.abs(
                    pure_feature
                    - oblique_feature
                ),
                pure_feature
                * oblique_feature,
            ],
            dim=1,
        )

        feature = self.fusion(
            fused
        )

        return (
            self.coarse_head(feature),
            self.subtype_head(feature),
        )

    def set_backbone_trainable(
        self,
        trainable: bool,
    ) -> None:
        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = (
                trainable
            )


def temperature_scale_probability(
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
    logits = np.log(
        probability
    ) / float(temperature)
    logits -= logits.max()
    exp_value = np.exp(logits)
    return (
        exp_value
        / exp_value.sum()
    )


def final_label_from_probabilities(
    coarse_probability: np.ndarray,
    subtype_probability: np.ndarray,
    na_threshold: float | None = None,
    subtype_uncertain_threshold: float = 0.60,
) -> tuple[
    str,
    dict[str, Any],
]:
    """
    逻辑层级推理：

    Level 1:
      I / II / III / NA

    如果 II:
      Level 2:
        head-to-head / full

      若两者都不够确定：
        -> Class II unspecified
    """
    coarse_probability = np.asarray(
        coarse_probability,
        dtype=np.float64,
    )
    subtype_probability = np.asarray(
        subtype_probability,
        dtype=np.float64,
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
            >= na_threshold
        ):
            coarse_id = 3
        else:
            coarse_id = int(
                np.argmax(
                    coarse_probability[:3]
                )
            )

    detail: dict[str, Any] = {
        "coarse_id": coarse_id,
        "coarse_label": (
            COARSE_LABELS[coarse_id]
        ),
        "subtype_confidence": None,
    }

    if coarse_id == 0:
        return "Class I", detail

    if coarse_id == 2:
        return "Class III", detail

    if coarse_id == 3:
        return "Not_Assessable", detail

    subtype_id = int(
        np.argmax(
            subtype_probability
        )
    )
    subtype_confidence = float(
        subtype_probability[subtype_id]
    )
    detail["subtype_id"] = subtype_id
    detail["subtype_confidence"] = (
        subtype_confidence
    )

    if (
        subtype_confidence
        < subtype_uncertain_threshold
    ):
        return (
            "Class II unspecified",
            detail,
        )

    return (
        SUBTYPE_LABELS[subtype_id],
        detail,
    )
