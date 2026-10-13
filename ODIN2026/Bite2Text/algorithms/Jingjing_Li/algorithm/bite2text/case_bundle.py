"""Shared raw-input objects and construction for one Bite2Text case."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from .segmentation import DadsnSegmenter
from .tiff import load_five_view_tiff


@dataclass
class CaseBundle:
    """All reusable case state derived solely from two STL files and one TIFF."""

    case_id: str
    requested_case_id: str | None
    upper_stl_path: Path
    lower_stl_path: Path
    photo_tiff_path: Path
    upper_mesh: trimesh.Trimesh
    lower_mesh: trimesh.Trimesh
    upper_vertices: np.ndarray
    upper_faces: np.ndarray
    lower_vertices: np.ndarray
    lower_faces: np.ndarray
    photos: dict[str, Image.Image]
    upper_c17: np.ndarray | None = None
    lower_c17: np.ndarray | None = None
    upper_fdi: np.ndarray | None = None
    lower_fdi: np.ndarray | None = None
    upper_fdi_original: np.ndarray | None = None
    lower_fdi_original: np.ndarray | None = None
    upper_crowding_mask: np.ndarray | None = None
    lower_crowding_mask: np.ndarray | None = None
    segmentation_selection: dict[str, str] = field(default_factory=dict)
    crowding_qc_summary: list[dict[str, str]] = field(default_factory=list)
    crowding_qc_tooth_metrics: list[dict[str, str]] = field(default_factory=list)
    geometry_cache: dict[str, Any] = field(default_factory=dict)
    render_cache: dict[str, Image.Image] = field(default_factory=dict)
    feature_cache: dict[str, Any] = field(default_factory=dict)

    def require_segmentation(self) -> None:
        if any(value is None for value in (self.upper_c17, self.lower_c17, self.upper_fdi, self.lower_fdi)):
            raise RuntimeError("DADSN segmentation has not been added to this CaseBundle")
        if len(self.upper_fdi) != len(self.upper_vertices) or len(self.lower_fdi) != len(self.lower_vertices):
            raise RuntimeError("DADSN FDI labels are not aligned with original STL vertices")


def load_case(
    *,
    upper_stl_path: Path | str,
    lower_stl_path: Path | str,
    photo_tiff_path: Path | str,
    case_id: str | None = None,
) -> CaseBundle:
    """Load each raw modality once using the deployment-wide representations."""

    upper_path = _require_file(upper_stl_path, "upper STL")
    lower_path = _require_file(lower_stl_path, "lower STL")
    photo_path = _require_file(photo_tiff_path, "intraoral TIFF")
    upper_mesh = _load_mesh(upper_path, "upper")
    lower_mesh = _load_mesh(lower_path, "lower")
    requested = _normalize_requested_case_id(case_id)
    return CaseBundle(
        case_id=_content_case_id(upper_path, lower_path),
        requested_case_id=requested,
        upper_stl_path=upper_path,
        lower_stl_path=lower_path,
        photo_tiff_path=photo_path,
        upper_mesh=upper_mesh,
        lower_mesh=lower_mesh,
        upper_vertices=np.asarray(upper_mesh.vertices, dtype=np.float64),
        upper_faces=np.asarray(upper_mesh.faces, dtype=np.int64),
        lower_vertices=np.asarray(lower_mesh.vertices, dtype=np.float64),
        lower_faces=np.asarray(lower_mesh.faces, dtype=np.int64),
        photos=load_five_view_tiff(photo_path),
    )


def add_dadsn_segmentation(bundle: CaseBundle, segmenter: DadsnSegmenter) -> CaseBundle:
    """Run canonical DADSN once and attach C17/FDI arrays to the shared bundle."""

    if bundle.upper_fdi is not None or bundle.lower_fdi is not None:
        raise RuntimeError("DADSN segmentation is already attached to this CaseBundle")
    result = segmenter.predict(
        upper_stl_path=bundle.upper_stl_path,
        lower_stl_path=bundle.lower_stl_path,
        case_id=bundle.case_id,
    )
    if len(result.upper_fdi) != len(bundle.upper_vertices):
        raise RuntimeError(
            f"Upper DADSN vertex mismatch: {len(result.upper_fdi)} labels for {len(bundle.upper_vertices)} vertices"
        )
    if len(result.lower_fdi) != len(bundle.lower_vertices):
        raise RuntimeError(
            f"Lower DADSN vertex mismatch: {len(result.lower_fdi)} labels for {len(bundle.lower_vertices)} vertices"
        )
    bundle.upper_c17 = result.upper_c17
    bundle.lower_c17 = result.lower_c17
    bundle.upper_fdi = result.upper_fdi
    bundle.lower_fdi = result.lower_fdi
    bundle.upper_fdi_original = result.upper_original_fdi
    bundle.lower_fdi_original = result.lower_original_fdi
    bundle.upper_crowding_mask = result.upper_mask
    bundle.lower_crowding_mask = result.lower_mask
    bundle.segmentation_selection = result.selection
    bundle.crowding_qc_summary = result.qc_summary
    bundle.crowding_qc_tooth_metrics = result.qc_tooth_metrics
    return bundle


def _require_file(value: Path | str, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {description}: {path}")
    return path


def _load_mesh(path: Path, arch: str) -> trimesh.Trimesh:
    # This exact load form matches the canonical DADSN implementation.
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{arch} scan is not a triangular mesh: {path}")
    if len(mesh.vertices) < 100 or len(mesh.faces) == 0:
        raise ValueError(f"{arch} scan is empty or too small: {path}")
    return mesh


def _normalize_requested_case_id(case_id: str | None) -> str | None:
    if case_id is None:
        return None
    normalized = str(case_id).strip()
    return normalized or None


def _content_case_id(upper_path: Path, lower_path: Path) -> str:
    """Return the production identity from ordered raw STL bytes only.

    TIFF bytes are intentionally excluded: the identity keys geometry and
    segmentation workspaces, while photo semantics remain independent inputs.
    The role tags make swapping upper and lower scans produce a different ID.
    """

    digest = hashlib.sha256()
    for role, path in ((b"upper", upper_path), (b"lower", lower_path)):
        digest.update(b"bite2text-case-id-v1\0")
        digest.update(role)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"case_{digest.hexdigest()[:16]}"
