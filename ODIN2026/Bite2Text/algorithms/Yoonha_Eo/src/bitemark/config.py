"""Configuration loading with a guard against unfitted public parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class UnfittedConfigurationError(RuntimeError):
    """Raised when placeholder parameters are used as if they were calibrated."""


@dataclass(frozen=True)
class PublicConfig:
    raw: dict[str, Any]

    @property
    def needs_to_be_finetuned(self) -> bool:
        return bool(self.raw["release"]["needs_to_be_finetuned"])

    def require_explicit_placeholder_override(self, allow_unfitted: bool) -> None:
        if self.needs_to_be_finetuned and not allow_unfitted:
            raise UnfittedConfigurationError(
                "Public parameters are random placeholders and NEED TO BE FINETUNED. "
                "Use --allow-unfitted-placeholder only for the synthetic demonstration."
            )

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing configuration section: {name}")
        return value


def load_config(path: str | Path) -> PublicConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or "release" not in raw:
        raise ValueError("Invalid Bitemark configuration")
    release = raw["release"]
    if release.get("contains_real_calibration") is not False:
        raise ValueError("Public configuration must explicitly exclude real calibration")
    if release.get("clinically_validated") is not False:
        raise ValueError("Public placeholder must not claim clinical validation")
    return PublicConfig(raw=raw)
