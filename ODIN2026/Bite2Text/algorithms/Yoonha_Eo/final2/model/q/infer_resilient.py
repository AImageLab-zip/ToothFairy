#!/usr/bin/env python3
"""3DTeethLand inference with a fallback for scans producing <5 tooth centroids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


TEETHLAND_ROOT = Path(__file__).resolve().parents[1] / "external" / "3dteethland"
if not TEETHLAND_ROOT.is_dir():
    # In the Bite2Text project this wrapper is executed with the upstream repo as cwd.
    TEETHLAND_ROOT = Path.cwd()
sys.path.insert(0, str(TEETHLAND_ROOT))

from teethland.datamodules.teethinstseg import TeethInstSegDataModule  # noqa: E402
from infer import predict  # noqa: E402


ORIGINAL_DETERMINE_ROT_MATRIX = TeethInstSegDataModule.determine_rot_matrix


def pca_rot_matrix(centroids) -> torch.Tensor:
    xy = centroids.C[:, :2].detach().cpu().numpy()
    if xy.shape[0] < 2 or np.allclose(xy, xy[0]):
        angle = 0.0
    else:
        covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        degrees = float(np.degrees(np.arctan2(direction[1], direction[0])))
        while degrees > 90:
            degrees -= 180
        while degrees <= -90:
            degrees += 180
        angle = -degrees / 180.0 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    matrix = np.array(
        [
            [cosval, -sinval, 0, 0],
            [sinval, cosval, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return torch.from_numpy(matrix).to(centroids.C)


def resilient_determine_rot_matrix(self, centroids, verbose: bool = False):
    try:
        return ORIGINAL_DETERMINE_ROT_MATRIX(self, centroids, verbose=verbose)
    except cv2.error as error:
        count = int(centroids.C.shape[0])
        print(
            f"[Bite2Text fallback] fitEllipse failed for {count} tooth centroids; using PCA orientation: {error}",
            flush=True,
        )
        return pca_rot_matrix(centroids)


TeethInstSegDataModule.determine_rot_matrix = resilient_determine_rot_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["instances", "landmarks"])
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--config", default="teethland/config/config.yaml", type=str)
    args = parser.parse_args()
    predict(args.stage, args.mixed, args.devices, args.config)


if __name__ == "__main__":
    main()
