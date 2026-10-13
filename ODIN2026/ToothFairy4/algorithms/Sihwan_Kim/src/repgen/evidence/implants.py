"""R230 precision-first implant-to-FDI relation runtime."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


MIN_COMPONENT_VOXELS = 3000
MAX_DISTANCE_MM = 16.0


def select_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["component_rank"])].append(row)
    selected: list[str] = []
    audit: list[dict[str, Any]] = []
    for component_rank, candidates in sorted(grouped.items()):
        voxels = max(int(row["component_voxels"]) for row in candidates)
        eligible = [
            row
            for row in candidates
            if not bool(row["candidate_observed_tooth"])
        ]
        eligible.sort(
            key=lambda row: (
                float(row["centroid_distance_mm"]),
                int(row["candidate_fdi"]),
            )
        )
        winner = eligible[0] if eligible else None
        distance = (
            float(winner["centroid_distance_mm"])
            if winner is not None
            else math.inf
        )
        passed = bool(
            voxels >= MIN_COMPONENT_VOXELS
            and winner is not None
            and distance <= MAX_DISTANCE_MM
        )
        fdi = str(winner["candidate_fdi"]) if passed else ""
        if fdi:
            selected.append(fdi)
        audit.append(
            {
                "component_rank": component_rank,
                "component_voxels": voxels,
                "candidate_fdi": fdi,
                "candidate_distance_mm": None if winner is None else distance,
                "component_guard_pass": voxels >= MIN_COMPONENT_VOXELS,
                "distance_guard_pass": distance <= MAX_DISTANCE_MM,
                "selected": passed,
            }
        )
    return sorted(set(selected), key=int), audit


def score(mask: np.ndarray, base: Any) -> tuple[list[str], dict[str, Any]]:
    try:
        centroids = base.tooth_centroids(mask)
        slots = base.expected_slot_centroids(centroids)
        components = base._component_records(
            mask,
            base.LABEL_GROUPS["implant"],
            1,
            connectivity=26,
        )
        rows: list[dict[str, Any]] = []
        spacing_mm = float(base.TARGET_SPACING_XYZ[0])
        for component_rank, component in enumerate(components, start=1):
            centroid = np.asarray(component["centroid"], dtype=np.float64)
            arch = base._component_arch(component, slots)
            for fdi, (point, source) in slots.items():
                if base.arch_for_fdi(fdi) != arch:
                    continue
                rows.append(
                    {
                        "component_rank": component_rank,
                        "component_voxels": int(component["voxels"]),
                        "candidate_fdi": str(fdi),
                        "candidate_observed_tooth": not str(source).startswith(
                            "interpolated:"
                        ),
                        "candidate_slot_source": str(source),
                        "centroid_distance_mm": float(
                            np.linalg.norm(centroid - np.asarray(point))
                            * spacing_mm
                        ),
                    }
                )
        selected, component_audit = select_candidate_rows(rows)
        return selected, {
            "schema_version": "r230_implant_relation_runtime_v1",
            "available": True,
            "selected_fdis": selected,
            "component_count": len(components),
            "policy": {
                "minimum_component_voxels": MIN_COMPONENT_VOXELS,
                "maximum_distance_mm": MAX_DISTANCE_MM,
                "missing_slot_only": True,
                "report_label_used": False,
            },
            "components": component_audit,
        }
    except Exception as error:
        return [], {
            "schema_version": "r230_implant_relation_runtime_v1",
            "available": False,
            "selected_fdis": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }
