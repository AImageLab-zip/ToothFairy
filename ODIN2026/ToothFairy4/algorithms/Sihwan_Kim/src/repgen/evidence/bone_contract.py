"""Fail-closed R212 Bone Atrophy evidence and writer decision contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REGION_SENTENCES = {
    "upper_left_anterior": "Alveolar ridge atrophy is present in the left anterior maxillary ridge.",
    "upper_left_posterior": "Alveolar ridge atrophy is present in the left posterior maxillary ridge.",
    "upper_right_anterior": "Alveolar ridge atrophy is present in the right anterior maxillary ridge.",
    "upper_right_posterior": "Alveolar ridge atrophy is present in the right posterior maxillary ridge.",
    "lower_left_anterior": "Alveolar ridge atrophy is present in the left anterior mandibular ridge.",
    "lower_left_posterior": "Alveolar ridge atrophy is present in the left posterior mandibular ridge.",
    "lower_right_anterior": "Alveolar ridge atrophy is present in the right anterior mandibular ridge.",
    "lower_right_posterior": "Alveolar ridge atrophy is present in the right posterior mandibular ridge.",
}


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "r212s_bone_atrophy_runtime_contract_v1":
        raise ValueError("unsupported R212 runtime contract schema")
    if contract.get("severity_enabled") or contract.get("cawood_howell_enabled"):
        raise ValueError("unsupported R212 severity or Cawood-Howell surface")
    if len(contract.get("checkpoints", [])) != 5:
        raise ValueError("R212 runtime requires five checkpoints")
    references = contract.get("selection_score_references", {})
    if not isinstance(references, Mapping) or set(references) != set("01234"):
        raise ValueError("R212 runtime requires five fold ECDF references")
    if any(not references[key] for key in references):
        raise ValueError("R212 fold ECDF reference cannot be empty")
    return contract


def resolve_checkpoints(
    model_root: Path,
    contract: Mapping[str, Any],
    verify_hash: bool = True,
) -> list[Path]:
    """Resolve and verify all R212 checkpoints inside the read-only model root."""
    root = model_root.resolve()
    resolved: list[Path] = []
    for row in contract.get("checkpoints", []):
        path = (root / str(row["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("R212 checkpoint escapes model root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes = int(row.get("bytes", -1))
        if expected_bytes >= 0 and path.stat().st_size != expected_bytes:
            raise ValueError(f"R212 checkpoint size mismatch: {path.name}")
        if verify_hash:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != str(row.get("sha256", "")):
                raise ValueError(f"R212 checkpoint hash mismatch: {path.name}")
        resolved.append(path)
    if len(resolved) != 5:
        raise ValueError("R212 runtime requires five resolved checkpoints")
    return resolved


def decide(
    score_result: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    calibrated = score_result.get("calibrated_case_score")
    available = bool(score_result.get("available")) and calibrated is not None
    case_threshold = float(contract.get("case_threshold", 2.0))
    case_emit = bool(available and float(calibrated) >= case_threshold)

    ranked = list(score_result.get("regions") or [])
    top = ranked[0] if ranked else {}
    second = ranked[1] if len(ranked) > 1 else {}
    top_region = str(top.get("region", ""))
    top_score = float(top.get("score", 0.0))
    margin = top_score - float(second.get("score", 0.0)) if second else 0.0
    regional_emit = bool(
        case_emit
        and contract.get("regional_enabled")
        and top_region in REGION_SENTENCES
        and top_score >= float(contract.get("regional_top_score_threshold", 2.0))
        and margin >= float(contract.get("regional_margin_threshold", 2.0))
    )
    sentence = (
        REGION_SENTENCES[top_region]
        if regional_emit
        else str(contract.get("case_sentence", ""))
        if case_emit
        else ""
    )
    return {
        "selected": case_emit,
        "regional_selected": regional_emit,
        "sentence": sentence,
        "decision": (
            "emit_regional" if regional_emit else "emit_case_only" if case_emit else "abstain"
        ),
        "calibrated_case_score": calibrated,
        "case_threshold": case_threshold,
        "top_region": top_region,
        "top_region_score": top_score,
        "top1_top2_margin": margin,
        "severity_emitted": False,
        "cawood_howell_emitted": False,
    }
