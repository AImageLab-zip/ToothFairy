"""R238 T4-compatible evidence-grounded recall report runtime."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import anatomy as BASE
from .evidence import models as weak_runtime
from .evidence import bone_contract as R212_CONTRACT
from .evidence import bone_atrophy as R212_RUNTIME
from .evidence import impacted as R225_IMPACTED
from .evidence import implants as R230_IMPLANT
from .evidence import tooth_occupancy as R233_ABSENCE
from .reporting import writer as R238_WRITER


NUM_SEGMENTATION_HEADS = 47
FP16_BYTES = 2
ACTIVE_WEAK_LABELS = (
    "endodontic",
    "impacted",
    "periapical",
    "periodontal",
    "sinus_mucosal",
)
R212_CONTRACT_PATH = (
    weak_runtime.MODEL_ROOT / "r212s_bone_atrophy_runtime_contract.json"
)
R223_BONE_ATROPHY_THRESHOLD = 0.50
R223_ENDODONTIC_THRESHOLD = 0.10
R223_PERIAPICAL_THRESHOLD = 0.40
R223_PERIODONTAL_THRESHOLD = 0.94
R223_IMPLANT_MIN_COMPONENT_VOXELS = 1500.0
R223_SINUS_COVERAGE_POLICY = "direct_or_any_upper_tooth"
R223_FOV_NOT_INCLUDED_MAX_Z_MM = 100.0
R231_REGION_POLICY = {
    "minimum_calibrated_case_score": 0.95,
    "minimum_top_region_score": 0.65,
    "minimum_top1_top2_margin": 0.08,
    "maximum_top_region_std": 0.15,
    "minimum_observed_slots": 2,
}
R231_REGION_TEXT = {
    "upper_left_anterior": "upper left anterior alveolar ridge",
    "upper_left_posterior": "upper left posterior alveolar ridge",
    "upper_right_anterior": "upper right anterior alveolar ridge",
    "upper_right_posterior": "upper right posterior alveolar ridge",
    "lower_left_anterior": "lower left anterior alveolar ridge",
    "lower_left_posterior": "lower left posterior alveolar ridge",
    "lower_right_anterior": "lower right anterior alveolar ridge",
    "lower_right_posterior": "lower right posterior alveolar ridge",
}

weak_runtime.THRESHOLDS["endodontic"] = R223_ENDODONTIC_THRESHOLD
weak_runtime.THRESHOLDS["periapical"] = R223_PERIAPICAL_THRESHOLD
weak_runtime.THRESHOLDS["periodontal"] = R223_PERIODONTAL_THRESHOLD

REFERENCE_STYLE_REPLACEMENTS = {
    "Alveolar process atrophy is present.": (
        "Atrophy of the alveolar processes is observed."
    ),
    "Endodontic treatment is present.": (
        "Sequelae of endodontic treatment are noted."
    ),
    "Dental implants are present.": "Endosseous dental implants are present.",
    "Prosthetic crowns are present.": (
        "Prosthetic crowns are present as coronal restorations."
    ),
    "A fixed prosthetic bridge is present.": (
        "A fixed bridge-type prosthetic rehabilitation is present."
    ),
    "Partial edentulism is present.": (
        "Partial edentulism of the dental arches is present."
    ),
    "Endosseous dental implants are present.": (
        "Presence of endosseous dental implants is noted."
    ),
    "Periodontal bone loss is present.": (
        "Periodontal bone resorption is present."
    ),
    "The course and foraminal emergence of the inferior alveolar canals are regular.": (
        "The course and emergence of the inferior alveolar canals are regular bilaterally."
    ),
    "The examined volume provides assessable coverage for the reported dental and osseous findings.": (
        "The available CBCT volume is adequate for assessment of the reported dental and osseous findings."
    ),
    "The maxilla or mandibular condyles may be partially included or not included in the scan volume. Mandibular condyles are not included in the acquisition field when outside the scan volume.": (
        "The maxilla or mandibular condyles may be partially included or not included in the scan volume."
    ),
}

R223_HEADING_REPLACEMENTS = {
    "Additional evidence-gated findings": "Additional findings",
    "Prosthetic status": "Dental and prosthetic status",
}

PROSTHETIC_RELATION_POLICY = copy.deepcopy(BASE.PROSTHETIC_RELATION_POLICY)
PROSTHETIC_RELATION_POLICY["implant"] = {
    "promoted": True,
    "promotion_tier": "forced_experimental",
    "kind": "slot_hungarian",
    "observed_slot_penalty_mm": 0.0,
    "source": "R166_stable_implant_relation_forced_R172",
}
PROSTHETIC_RELATION_POLICY["crown"] = {
    **PROSTHETIC_RELATION_POLICY["crown"],
    "promoted": True,
    "promotion_tier": "forced_experimental",
    "source": "R165_surface_covers_forced_R172",
}
PROSTHETIC_RELATION_POLICY["bridge"] = {
    "promoted": True,
    "promotion_tier": "forced_experimental",
    "kind": "slot_surface_span_with_fragment_merge",
    "slot_surface_threshold_mm": 12.0,
    "max_slots": 4,
    "fragment_merge_distance_mm": 20.0,
    "source": "R166_modal_OOF_slot_surface_span_plus_fragment_merge_R172",
}

BASE.PROSTHETIC_RELATION_POLICY = PROSTHETIC_RELATION_POLICY
_ORIGINAL_BRIDGE_RELATION = BASE._assign_bridge_relations
_ORIGINAL_EXTRACT_DIRECT_OBJECTS = BASE.extract_direct_objects


def estimate_dense_logit_gib(size_xyz: Iterable[int]) -> float:
    voxel_count = int(np.prod(tuple(int(value) for value in size_xyz), dtype=np.int64))
    return voxel_count * NUM_SEGMENTATION_HEADS * FP16_BYTES / (1024**3)


def _fragment_clusters(
    components: list[dict[str, Any]],
    slots: dict[str, tuple[np.ndarray, str]],
    maximum_distance_mm: float,
) -> list[list[int]]:
    parents = list(range(len(components)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(components)):
        left_arch = BASE._component_arch(components[left], slots)
        left_center = np.asarray(components[left]["centroid"], dtype=np.float64)
        for right in range(left + 1, len(components)):
            if BASE._component_arch(components[right], slots) != left_arch:
                continue
            right_center = np.asarray(components[right]["centroid"], dtype=np.float64)
            distance_mm = float(
                np.linalg.norm(left_center - right_center) * BASE.TARGET_SPACING_XYZ[0]
            )
            if distance_mm <= maximum_distance_mm:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(components)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _assign_bridge_relations_r178(
    mask: np.ndarray,
    components: list[dict[str, Any]],
    centroids: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not components:
        return [], []
    if not policy.get("promoted"):
        return _ORIGINAL_BRIDGE_RELATION(mask, components, centroids, policy)
    slots = BASE.expected_slot_centroids(centroids)
    component_anchors: dict[int, set[str]] = {}
    audits: list[dict[str, Any]] = []
    assigned: set[str] = set()
    spacing_mm = float(BASE.TARGET_SPACING_XYZ[0])
    for component_index, component in enumerate(components):
        slices = component["slices"]
        starts = np.asarray([part.start for part in slices], dtype=np.float64)
        coordinates = np.argwhere(component["local_mask"]).astype(np.float64) + starts
        arch = BASE._component_arch(component, slots)
        candidates: list[tuple[float, int, str, str]] = []
        for fdi, (point, source) in slots.items():
            if BASE.arch_for_fdi(fdi) != arch:
                continue
            distance_mm = float(
                np.sqrt(
                    (
                        (coordinates - np.asarray(point, dtype=np.float64)) ** 2
                    ).sum(axis=1)
                ).min()
                * spacing_mm
            )
            if distance_mm <= float(policy["slot_surface_threshold_mm"]):
                candidates.append((distance_mm, int(fdi), fdi, source))
        candidates.sort(key=lambda row: (row[0], row[1]))
        selected = candidates[: int(policy["max_slots"])]
        anchors = {row[2] for row in selected}
        component_anchors[component_index] = anchors
        span = BASE.ordered_bridge_span(anchors)
        if not span:
            continue
        assigned.update(span)
        observed_anchors = anchors & set(centroids)
        audits.append(
            {
                "component_rank": component_index + 1,
                "selected_slots": [
                    {
                        "fdi": fdi,
                        "slot_surface_distance_mm": round(distance_mm, 4),
                        "slot_source": source,
                    }
                    for distance_mm, _, fdi, source in selected
                ],
                "abutment_teeth": BASE.order_fdis(observed_anchors),
                "span_teeth": span,
                "pontic_candidates": [
                    fdi for fdi in span if fdi not in observed_anchors
                ],
                "relation": "SLOT_SURFACE_ABUTS_PONTIC_AT_SPANS",
            }
        )
    for cluster_index, indices in enumerate(
        _fragment_clusters(
            components, slots, float(policy["fragment_merge_distance_mm"])
        ),
        start=1,
    ):
        if len(indices) < 2:
            continue
        anchors: set[str] = set()
        for component_index in indices:
            anchors.update(component_anchors.get(component_index, set()))
        span = BASE.ordered_bridge_span(anchors)
        if not span:
            continue
        observed_anchors = anchors & set(centroids)
        assigned.update(span)
        audits.append(
            {
                "cluster_rank": cluster_index,
                "component_ranks": [index + 1 for index in indices],
                "abutment_teeth": BASE.order_fdis(observed_anchors),
                "span_teeth": span,
                "pontic_candidates": [
                    fdi for fdi in span if fdi not in observed_anchors
                ],
                "relation": "MERGED_FRAGMENT_ABUTS_PONTIC_AT_SPANS",
            }
        )
    return sorted(assigned, key=int), audits


BASE._assign_bridge_relations = _assign_bridge_relations_r178


def extract_direct_objects(
    mask: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    assignments, audit = _ORIGINAL_EXTRACT_DIRECT_OBJECTS(mask, centroids)
    audit["schema_version"] = (
        "r178_r172_all_prosthetic_relations_slot_surface_bridge_span_v2"
    )
    audit["forced_experimental_promotion"] = True
    return assignments, audit


BASE.extract_direct_objects = extract_direct_objects


def run_segmentation(input_file: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R237 segmentation")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(
            f"GPU compute capability {capability[0]}.{capability[1]} is below R237 sm75 minimum"
        )
    cuda = {
        "device": device,
        "name": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mamba_execution": (
            "reference_sm75" if capability < (8, 0) else "fused_ampere_plus"
        ),
    }
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": "/tmp/nnunet_raw",
            "nnUNet_preprocessed": "/tmp/nnunet_preprocessed",
            "nnUNet_results": str(BASE.NNUNET_RESULTS),
        }
    )
    command = [
        sys.executable,
        "-m",
        "repgen.compat.predict",
        "-i",
        str(input_file.parent),
        "-o",
        str(output_dir),
        "-d",
        "317",
        "-c",
        BASE.CONFIGURATION,
        "-p",
        BASE.PLANS_NAME,
        "-tr",
        BASE.TRAINER_NAME,
        "-f",
        "0",
        "-chk",
        BASE.CHECKPOINT_NAME,
        "-npp",
        "0",
        "-nps",
        "0",
        "-step_size",
        "0.7",
        "-device",
        "cuda",
        "--disable_progress_bar",
        "--disable_tta",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"nnU-Net prediction failed with exit code {completed.returncode}"
        )
    return {
        "cuda": cuda,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "command": command,
        "execution_mode": "sequential_no_multiprocessing",
    }


def _ordered_teeth(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return sorted(
        {
            str(value)
            for value in values
            if len(str(value)) == 2 and str(value).isdigit()
        },
        key=int,
    )


DIRECT_CASE_ONLY_FACTS = {
    "bridge": ("bridge", "A fixed prosthetic bridge is present."),
    "crown": ("crown_oracle", "Prosthetic crowns are present."),
    "implant": ("implant", "Dental implants are present."),
}


def direct_component_score(direct: dict[str, Any], label: str) -> float:
    objects = (
        direct.get("direct_object_component_audit", {})
        .get("objects", {})
    )
    components = objects.get(label, {}).get("components", [])
    return max((float(row.get("voxels", 0.0)) for row in components), default=0.0)


def apply_direct_writer_scope(
    pack: dict[str, Any], direct_audit: dict[str, Any]
) -> list[str]:
    """Keep case presence while suppressing unstable FDI relations."""
    ready = dict(pack.get("report_ready_evidence") or {})
    labels = set(pack.get("positive_case_labels") or [])
    direct = dict(pack.get("image_predicted_case_evidence") or {})
    direct_labels = set(direct.get("direct_case_labels") or [])

    absence_present = "tooth_absence" in labels or bool(
        ready.get("tooth_absence")
    )
    labels.discard("tooth_absence")
    direct_labels.discard("tooth_absence")
    ready.pop("tooth_absence", None)

    sentences: list[str] = []
    if absence_present:
        labels.add("tooth_absence")
        sentences.append("Partial edentulism is present.")
    for label, (ready_key, sentence) in DIRECT_CASE_ONLY_FACTS.items():
        if label == "implant":
            score = direct_component_score(direct_audit, label)
            present = score >= R223_IMPLANT_MIN_COMPONENT_VOXELS
            labels.discard(label)
            if present:
                direct_labels.add(label)
            else:
                direct_labels.discard(label)
        else:
            present = label in labels or bool(ready.get(ready_key))
        ready.pop(ready_key, None)
        if present:
            labels.add(label)
            sentences.append(sentence)

    direct["direct_case_labels"] = sorted(direct_labels)
    pack["image_predicted_case_evidence"] = direct
    pack["positive_case_labels"] = sorted(labels)
    pack["report_ready_evidence"] = ready
    pack["direct_writer_scope_profile"] = (
        "R223_case_prosthetics_implant_component_guard"
    )
    pack["r223_direct_case_scores"] = {
        "implant_max_component_voxels": direct_component_score(
            direct_audit, "implant"
        ),
        "implant_min_component_voxels": R223_IMPLANT_MIN_COMPONENT_VOXELS,
        "implant_selected": "implant" in labels,
    }
    return sentences


def apply_exact_absence_writer_scope(
    pack: dict[str, Any],
    direct_sentences: list[str],
    absence_fdis: Iterable[str],
) -> list[str]:
    """Restore only R233 high-confidence tooth-level absence relations."""
    selected = _ordered_teeth(absence_fdis)
    if not selected:
        return direct_sentences
    labels = set(pack.get("positive_case_labels") or [])
    labels.add("tooth_absence")
    pack["positive_case_labels"] = sorted(labels)
    pack["r233_tooth_absence"] = {
        "selected_fdis": selected,
        "policy": "R218H_top2_plus_contiguous_raw_adaptive_top3",
    }
    return direct_sentences


def append_exact_absence_report(
    report: str, pack: dict[str, Any], absence_fdis: Iterable[str]
) -> str:
    """Insert exact absence in the same dental section used by the shadow."""
    selected = _ordered_teeth(absence_fdis)
    if not selected:
        return report
    sections: dict[str, str] = {}
    for block in report.split("\n\n"):
        heading, separator, body = block.strip().partition(":")
        if separator:
            sections[heading.strip()] = body.strip()
    heading = "Dental and prosthetic status"
    sentences = [
        item.strip() + "."
        for item in sections.get(heading, "").split(".")
        if item.strip()
    ]
    partial = "Partial edentulism of the dental arches is present."
    if partial not in sentences:
        sentences.insert(0, partial)
    sentences = [
        sentence
        for sentence in sentences
        if not sentence.startswith("Absence of teeth ")
    ]
    sentences.insert(
        sentences.index(partial) + 1,
        BASE.sentence_teeth("Absence of", selected),
    )
    sections[heading] = " ".join(sentences)
    preferred = (
        "Additional findings",
        heading,
        "Mandibular canal",
        "Maxillary sinus",
        "Scan volume",
    )
    rendered = [
        f"{name}: {sections[name]}" for name in preferred if name in sections
    ]
    rendered.extend(
        f"{name}: {body}"
        for name, body in sections.items()
        if name not in preferred
    )
    ready = dict(pack.get("report_ready_evidence") or {})
    ready["tooth_absence"] = selected
    pack["report_ready_evidence"] = ready
    return "\n\n".join(rendered)


def _writer_arch_name(fdi: str) -> str:
    return "maxillary" if str(fdi)[0] in "12" else "mandibular"


def apply_location_preserving_writer(
    report: str,
    absence_audit: dict[str, Any],
    score_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Preserve arch context without adding a new diagnosis fact."""
    output = report
    absence_arches = {
        _writer_arch_name(str(row["fdi"]))
        for row in absence_audit.get("scores", [])
        if row.get("eligible")
    }
    generic_absence = "Partial edentulism of the dental arches is present."
    absence_phrase = generic_absence
    if absence_arches == {"maxillary"}:
        absence_phrase = "Partial edentulism of the maxillary arch is present."
    elif absence_arches == {"mandibular"}:
        absence_phrase = "Partial edentulism of the mandibular arch is present."
    elif absence_arches:
        absence_phrase = (
            "Partial edentulism of the maxillary and mandibular arches is present."
        )
    output = output.replace(generic_absence, absence_phrase)

    endodontic_rows = sorted(
        (
            row
            for row in score_rows
            if row.get("label") == "endodontic"
            and row.get("tooth_fdi")
            and row.get("score") is not None
        ),
        key=lambda row: (-float(row["score"]), int(row["tooth_fdi"])),
    )[:4]
    endodontic_arch = ""
    if len(endodontic_rows) == 4:
        arches = [
            _writer_arch_name(str(row["tooth_fdi"]))
            for row in endodontic_rows
        ]
        winner = max(set(arches), key=arches.count)
        if arches.count(winner) >= 3:
            endodontic_arch = winner
            output = output.replace(
                "Sequelae of endodontic treatment are noted.",
                f"Sequelae of endodontic treatment are noted in the {winner} dentition.",
            )
    return output, {
        "absence_arches": sorted(absence_arches),
        "absence_phrase": absence_phrase,
        "endodontic_arch": endodontic_arch,
        "endodontic_policy": "top4_R101_scores_at_least_3_same_arch",
        "new_diagnosis_facts_added": False,
    }


def apply_sinus_coverage_policy(pack: dict[str, Any]) -> None:
    """Add a qualified sinus-coverage fact when upper anatomy is observed."""
    labels = set(pack.get("positive_case_labels") or [])
    direct = dict(pack.get("image_predicted_case_evidence") or {})
    direct_labels = set(direct.get("direct_case_labels") or [])
    present_teeth = _ordered_teeth(direct.get("present_teeth"))
    upper_teeth = [fdi for fdi in present_teeth if fdi[0] in "12"]
    direct_sinus = "sinus" in direct_labels or "sinus" in labels
    selected = direct_sinus or bool(upper_teeth)
    if selected:
        labels.add("sinus")
    direct["direct_case_labels"] = sorted(direct_labels)
    pack["image_predicted_case_evidence"] = direct
    pack["positive_case_labels"] = sorted(labels)
    pack["r223_sinus_coverage"] = {
        "policy": R223_SINUS_COVERAGE_POLICY,
        "direct_sinus": direct_sinus,
        "upper_teeth": upper_teeth,
        "selected": selected,
        "diagnostic_sinus_finding_enabled": False,
    }


def apply_fov_extent_policy(
    pack: dict[str, Any], geometry: dict[str, Any]
) -> None:
    """Map limited physical z coverage to a qualified FOV limitation fact."""
    labels = set(pack.get("positive_case_labels") or [])
    direct = dict(pack.get("image_predicted_case_evidence") or {})
    direct_labels = set(direct.get("direct_case_labels") or [])
    physical_z_mm = float(geometry["target_size_xyz"][2]) * float(
        geometry["target_spacing_xyz"][2]
    )
    limited = physical_z_mm <= R223_FOV_NOT_INCLUDED_MAX_Z_MM
    if limited:
        labels.discard("fov_included")
        labels.add("fov_not_included")
        direct_labels.discard("fov_included")
        direct_labels.add("fov_not_included")
    direct["direct_case_labels"] = sorted(direct_labels)
    pack["image_predicted_case_evidence"] = direct
    pack["positive_case_labels"] = sorted(labels)
    pack["r223_fov_extent"] = {
        "physical_z_mm": physical_z_mm,
        "limited_max_z_mm": R223_FOV_NOT_INCLUDED_MAX_Z_MM,
        "fov_not_included_selected": limited,
    }


def append_direct_case_report(report: str, sentences: list[str]) -> str:
    if not sentences:
        return report.rstrip()
    return report.rstrip() + "\nProsthetic status: " + " ".join(sentences)


def merge_weak_facts(pack: dict[str, Any], selected: dict[str, Any]) -> None:
    ready = dict(pack.get("report_ready_evidence") or {})
    labels = set(pack.get("positive_case_labels") or [])
    for label in (
        "bone_atrophy",
        "endodontic",
        "periapical",
        "periodontal",
    ):
        if selected.get(label):
            labels.add(label)
            ready[label] = True
    pack["positive_case_labels"] = sorted(labels)
    pack["report_ready_evidence"] = ready
    pack["weak_label_profile"] = (
        "R223_case_labels_plus_R225_impacted_consensus"
    )
    pack["weak_label_selected_facts"] = selected


def score_r212_bone_atrophy(
    image: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Run the five-fold R212 ensemble and fail closed on any artifact error."""
    try:
        contract = R212_CONTRACT.load_contract(R212_CONTRACT_PATH)
        contract = dict(contract)
        contract["case_threshold"] = R223_BONE_ATROPHY_THRESHOLD
        checkpoints = R212_CONTRACT.resolve_checkpoints(
            weak_runtime.MODEL_ROOT,
            contract,
            verify_hash=True,
        )
        references = [
            contract["selection_score_references"][str(fold)]
            for fold in range(5)
        ]
        score_result = R212_RUNTIME.score_ensemble(
            image,
            segmentation,
            checkpoints,
            device="cuda",
            candidate_batch_size=8,
            selection_score_references=references,
        )
        decision = R212_CONTRACT.decide(score_result, contract)
        return bool(decision["selected"]), {
            "schema_version": "r212w_case_atrophy_runtime_audit_v1",
            "available": bool(score_result.get("available")),
            "checkpoint_count": len(checkpoints),
            "case_score": score_result.get("case_score"),
            "calibrated_case_score": score_result.get(
                "calibrated_case_score"
            ),
            "regions": score_result.get("regions", []),
            "geometry": score_result.get("geometry", []),
            "decision": decision,
            "regional_output_enabled": False,
            "severity_output_enabled": False,
        }
    except Exception as error:
        return False, {
            "schema_version": "r212w_case_atrophy_runtime_audit_v1",
            "available": False,
            "decision": "abstain_artifact_or_runtime_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "regional_output_enabled": False,
            "severity_output_enabled": False,
        }


def select_bone_atrophy_region(audit: dict[str, Any]) -> str:
    if not audit.get("available"):
        return ""
    decision = dict(audit.get("decision") or {})
    regions = list(audit.get("regions") or [])
    if not decision.get("selected") or not regions:
        return ""
    top = dict(regions[0])
    region = str(top.get("region") or "")
    geometry = {
        str(row.get("region")): row for row in audit.get("geometry") or []
    }.get(region, {})
    checks = (
        float(audit.get("calibrated_case_score") or 0.0)
        >= R231_REGION_POLICY["minimum_calibrated_case_score"],
        float(top.get("score") or 0.0)
        >= R231_REGION_POLICY["minimum_top_region_score"],
        float(decision.get("top1_top2_margin") or 0.0)
        >= R231_REGION_POLICY["minimum_top1_top2_margin"],
        float(top.get("score_std") or 1.0)
        <= R231_REGION_POLICY["maximum_top_region_std"],
        int(geometry.get("observed_slots") or 0)
        >= R231_REGION_POLICY["minimum_observed_slots"],
    )
    return region if region in R231_REGION_TEXT and all(checks) else ""


def weak_evidence_sentences(selected: dict[str, Any]) -> list[str]:
    sentences: list[str] = []
    if selected.get("endodontic"):
        sentences.append("Endodontic treatment is present.")
    if selected.get("periodontal"):
        sentences.append("Periodontal bone loss is present.")
    if selected.get("periapical"):
        sentences.append("A periapical finding is present.")
    if selected.get("bone_atrophy"):
        region = str(selected.get("bone_atrophy_region") or "")
        if region in R231_REGION_TEXT:
            sentences.append(
                "Atrophy of the alveolar processes is observed, most pronounced "
                f"in the {R231_REGION_TEXT[region]}."
            )
        else:
            sentences.append("Alveolar process atrophy is present.")
    for fdi in _ordered_teeth(selected.get("impacted")):
        sentences.append(f"Tooth {fdi} is impacted.")
    implant_fdis = _ordered_teeth(selected.get("implant_exact_fdis"))
    if implant_fdis:
        sentences.append(
            "Endosseous implants are present at teeth "
            + ", ".join(implant_fdis)
            + "."
        )
    return sentences


def append_weak_report(report: str, selected: dict[str, Any]) -> str:
    sentences = weak_evidence_sentences(selected)
    if not sentences:
        return report.rstrip()
    return report.rstrip() + "\nAdditional evidence-gated findings: " + " ".join(
        sentences
    )


def apply_fact_preserving_reference_style(report: str) -> str:
    """Apply the R223 A20-stable style without changing supported facts."""
    lines: list[tuple[str, str]] = []
    for raw_line in report.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        heading, separator, body = raw_line.partition(":")
        if not separator:
            heading, body = "", raw_line
        for old, new in REFERENCE_STYLE_REPLACEMENTS.items():
            body = body.replace(old, new)
        lines.append((heading, body.strip()))
    order = {
        "Additional evidence-gated findings": 0,
        "Prosthetic status": 1,
        "Mandibular canal": 2,
        "Maxillary sinus": 3,
        "Scan volume": 4,
    }
    lines.sort(key=lambda item: order.get(item[0], 9))
    rendered = [
        f"{R223_HEADING_REPLACEMENTS.get(heading, heading)}: {body}"
        if heading
        else body
        for heading, body in lines
    ]
    return "\n\n".join(rendered)


def execute_case(
    input_path: Path, case_id: str, tmp_root: Path
) -> tuple[str, dict[str, Any]]:
    BASE.prepare_runtime_dirs(tmp_root)
    manifest = BASE.verify_model_bundle(verify_hash=True)
    if manifest.get("precomputed_case_scores_packaged") is not False:
        raise RuntimeError("R178 model bundle must explicitly exclude precomputed scores")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    resampled_path = tmp_root / "input" / f"{case_id}_0000.nii.gz"
    resample_started = time.perf_counter()
    geometry = BASE.resample_cbct(input_path, resampled_path)
    geometry["target_voxels"] = int(
        np.prod(tuple(geometry["target_size_xyz"]), dtype=np.int64)
    )
    geometry["estimated_dense_fp16_logit_gib"] = round(
        estimate_dense_logit_gib(geometry["target_size_xyz"]), 4
    )
    timings["resample_seconds"] = round(
        time.perf_counter() - resample_started, 4
    )
    print(
        json.dumps(
            {"event": "resample_complete", "case_id": case_id, **geometry},
            sort_keys=True,
        ),
        flush=True,
    )

    segmentation = run_segmentation(resampled_path, tmp_root / "prediction")
    timings["segmentation_seconds"] = segmentation["wall_seconds"]
    prediction_path = tmp_root / "prediction" / f"{case_id}.nii.gz"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"nnU-Net did not produce {prediction_path}")

    evidence_started = time.perf_counter()
    sitk = BASE._sitk()
    image_sitk = sitk.ReadImage(str(resampled_path))
    image = sitk.GetArrayFromImage(image_sitk).astype(np.float32, copy=False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path))).astype(
        np.int16, copy=False
    )
    direct, guard_audit = BASE.extract_direct_evidence(
        mask, BASE.load_volume_priors()
    )
    guarded_mask, _ = BASE.apply_r117_guard(mask, BASE.load_volume_priors())
    implant_fdis, implant_audit = R230_IMPLANT.score(guarded_mask, BASE)
    absence_fdis, absence_audit = R233_ABSENCE.score(mask, BASE)
    selected, score_rows, weak_audit = weak_runtime.score_weak_evidence(
        image,
        guarded_mask,
        case_id,
        device="cuda",
        geometry={
            "shape_zyx": tuple(int(value) for value in image.shape),
            "spacing_xyz": tuple(
                float(value) for value in image_sitk.GetSpacing()
            ),
            "origin_xyz": tuple(
                float(value) for value in image_sitk.GetOrigin()
            ),
            "direction": tuple(
                float(value) for value in image_sitk.GetDirection()
            ),
        },
        raw_mask=mask,
        labels=ACTIVE_WEAK_LABELS,
    )
    impacted_fdis, impacted_audit = R225_IMPACTED.score_consensus(
        image,
        mask,
        [row for row in score_rows if row.get("label") == "impacted"],
        BASE.load_volume_priors(),
        device="cuda",
    )
    # The legacy R134D2 result is only an independent guard. Only the R225
    # cross-model consensus may reach the report writer.
    selected["impacted"] = impacted_fdis
    selected["implant_exact_fdis"] = implant_fdis
    weak_audit["r225_impacted_consensus"] = impacted_audit
    score_rows.append(
        {
            "label": "impacted",
            "source": "R225_R198_R134D2_cross_model_consensus",
            "selected": bool(impacted_fdis),
            "selected_fdis": impacted_fdis,
            "r198_top_fdi": impacted_audit.get("r198", {}).get("top_fdi"),
            "r134d2_top_fdi": impacted_audit.get("r134d2", {}).get("top_fdi"),
            "cross_model_location_agreement": impacted_audit.get(
                "cross_model_location_agreement", False
            ),
        }
    )
    score_rows.append(
        {
            "label": "implant",
            "source": "R230_missing_slot_component_guard",
            "available": bool(implant_audit.get("available")),
            "selected": bool(implant_fdis),
            "selected_fdis": implant_fdis,
        }
    )
    score_rows.append(
        {
            "label": "tooth_absence",
            "source": "R218H_TF3_OOF_residual_occupancy_R234_adaptive_guard",
            "available": bool(absence_audit.get("available")),
            "selected": bool(absence_fdis),
            "selected_fdis": absence_fdis,
        }
    )
    weak_audit["r230_implant_relation"] = implant_audit
    weak_audit["r233_tooth_absence"] = absence_audit
    r212_selected, r212_audit = score_r212_bone_atrophy(image, mask)
    selected["bone_atrophy"] = r212_selected
    selected["bone_atrophy_region"] = select_bone_atrophy_region(r212_audit)
    score_rows.append(
        {
            "label": "bone_atrophy",
            "source": "R212_five_fold_case_ensemble",
            "available": bool(r212_audit.get("available")),
            "score": r212_audit.get("case_score"),
            "calibrated_score": r212_audit.get("calibrated_case_score"),
            "selected": r212_selected,
        }
    )
    weak_audit["r212_bone_atrophy"] = r212_audit
    pack = BASE.build_writer_evidence(direct, case_id)
    direct_case_sentences = apply_direct_writer_scope(pack, direct)
    direct_case_sentences = apply_exact_absence_writer_scope(
        pack, direct_case_sentences, absence_fdis
    )
    merge_weak_facts(pack, selected)
    apply_sinus_coverage_policy(pack)
    apply_fov_extent_policy(pack, geometry)
    report = append_direct_case_report(
        BASE.generate_report(pack), direct_case_sentences
    )
    report = append_weak_report(report, selected)
    report = apply_fact_preserving_reference_style(report)
    report = append_exact_absence_report(report, pack, absence_fdis)
    report, r236_location = apply_location_preserving_writer(
        report, absence_audit, score_rows
    )
    r238_evidence = R238_WRITER.build_detail_evidence(
        direct, selected, score_rows, absence_fdis
    )
    report, r238_detail = R238_WRITER.enhance_report(report, r238_evidence)
    pack["writer_profile"] = "R248_reference_phrase_crown_bridge_T4"
    timings["evidence_writer_seconds"] = round(
        time.perf_counter() - evidence_started, 4
    )
    timings["total_seconds"] = round(time.perf_counter() - started, 4)
    audit = {
        "case_id": case_id,
        "geometry": geometry,
        "cuda": segmentation["cuda"],
        "segmentation_execution_mode": segmentation["execution_mode"],
        "timings": timings,
        "direct_evidence": direct,
        "writer_evidence": pack,
        "weak_label_scores": score_rows,
        "weak_label_audit": weak_audit,
        "active_weak_labels": [
            *ACTIVE_WEAK_LABELS,
            "bone_atrophy",
            "implant",
            "tooth_absence",
        ],
        "writer_profile": "R248_reference_phrase_crown_bridge_T4",
        "r225_impacted_consensus": impacted_audit,
        "r212_bone_atrophy": r212_audit,
        "r230_implant_relation": implant_audit,
        "r233_tooth_absence": absence_audit,
        "r236_location_preserving_writer": r236_location,
        "r238_evidence_grounded_writer": r238_detail,
        "r231_bone_atrophy_region": {
            "policy": R231_REGION_POLICY,
            "selected_region": selected.get("bone_atrophy_region", ""),
            "scientific_support_gate_pass": False,
        },
        "weak_score_geometry": {
            "mode": "online_0p3mm_fail_closed",
            "shape_zyx": tuple(int(value) for value in image.shape),
            "spacing_xyz": tuple(
                float(value) for value in image_sitk.GetSpacing()
            ),
        },
        "r117_guard_pass_count": sum(
            int(row["guard_pass"]) for row in guard_audit
        ),
        "report_characters": len(report),
    }
    del image
    del image_sitk
    del mask
    del guarded_mask
    if os.environ.get("ODIN_KEEP_PREDICTION") != "1":
        shutil.rmtree(tmp_root / "prediction", ignore_errors=True)
    return report, audit
