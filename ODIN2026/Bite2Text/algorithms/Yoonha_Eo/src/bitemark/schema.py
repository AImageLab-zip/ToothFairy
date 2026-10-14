"""Shared public schemas for structured item-level findings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    image_paths: tuple[Path, ...]
    upper_stl: Path
    lower_stl: Path


@dataclass(frozen=True)
class Finding:
    item: str
    value: str
    source: str
    side: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineOutput:
    case_id: str
    report: str
    findings: tuple[Finding, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "report": self.report,
            "findings": [finding.to_dict() for finding in self.findings],
            "warnings": list(self.warnings),
        }
