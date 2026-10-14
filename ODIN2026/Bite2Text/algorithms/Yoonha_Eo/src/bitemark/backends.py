"""Interfaces that a private inference package must implement.

The public repository intentionally contains no checkpoint loading code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class PrivateIOPBackend(Protocol):
    def classify_views(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return an N x 5 probability matrix ordered by required view names."""

    def classify_crowding(self, arch: str, image_path: Path) -> str:
        """Return a calibrated crowding label for upper or lower arch."""

    def measure_midline(self, frontal_image: Path) -> float:
        """Return displacement normalized by central-incisor width."""


class PrivateIOSBackend(Protocol):
    def extract_features(self, upper_stl: Path, lower_stl: Path) -> Mapping[str, object]:
        """Run private tooth/landmark and point-cloud models and return item features."""


class MissingPrivateBackend(RuntimeError):
    pass


class NoWeightsBackend:
    """Fail-closed backend included to make non-distribution explicit."""

    def _missing(self) -> None:
        raise MissingPrivateBackend(
            "No pretrained weights are distributed. Provide a private backend and finetuned models."
        )

    def classify_views(self, image_paths: Sequence[Path]) -> np.ndarray:
        self._missing()

    def classify_crowding(self, arch: str, image_path: Path) -> str:
        self._missing()

    def measure_midline(self, frontal_image: Path) -> float:
        self._missing()

    def extract_features(self, upper_stl: Path, lower_stl: Path) -> Mapping[str, object]:
        self._missing()
