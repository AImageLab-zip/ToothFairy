"""Top-level public orchestration over precomputed or private-backend features."""

from __future__ import annotations

from collections.abc import Mapping

from .backends import PrivateIOPBackend, PrivateIOSBackend
from .config import PublicConfig
from .iop import assign_required_views, findings_from_synthetic_iop
from .ios import findings_from_synthetic_ios
from .report import generate_report
from .schema import CaseInput, PipelineOutput

PUBLIC_WARNING = (
    "UNFITTED PUBLIC PLACEHOLDER: random parameters, no pretrained weights, "
    "not for clinical use or challenge submission."
)


def run_from_features(
    payload: Mapping[str, object], config: PublicConfig, allow_unfitted: bool = False
) -> PipelineOutput:
    config.require_explicit_placeholder_override(allow_unfitted)
    findings = findings_from_synthetic_iop(payload["iop"], config.section("iop"))
    findings.extend(
        findings_from_synthetic_ios(payload["ios"], config.section("ios"), config.section("labels"))
    )
    return PipelineOutput(
        case_id=str(payload.get("case_id", "UNKNOWN")),
        report=generate_report(findings),
        findings=tuple(findings),
        warnings=(PUBLIC_WARNING,),
    )


def run_case(
    case: CaseInput,
    config: PublicConfig,
    iop_backend: PrivateIOPBackend,
    ios_backend: PrivateIOSBackend,
    allow_unfitted: bool = False,
) -> PipelineOutput:
    """Run the notebook-equivalent routing with user-supplied private backends."""
    probabilities = iop_backend.classify_views(case.image_paths)
    assigned = assign_required_views(case.image_paths, probabilities)
    payload = {
        "case_id": case.case_id,
        "views": {view: str(path) for view, path in assigned.items()},
        "iop": {
            "upper_crowding": iop_backend.classify_crowding("upper", assigned["upper"]),
            "lower_crowding": iop_backend.classify_crowding("lower", assigned["lower"]),
            "midline_displacement_tooth_width": iop_backend.measure_midline(assigned["front"]),
        },
        "ios": dict(ios_backend.extract_features(case.upper_stl, case.lower_stl)),
    }
    return run_from_features(payload, config, allow_unfitted=allow_unfitted)
