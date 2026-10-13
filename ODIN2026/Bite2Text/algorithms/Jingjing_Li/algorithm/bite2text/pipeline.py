"""Shared raw-only preprocessing orchestration for all Bite2Text field predictors."""

from __future__ import annotations

from pathlib import Path

from .case_bundle import CaseBundle, add_dadsn_segmentation, load_case
from .segmentation import DadsnSegmenter


def preprocess_case(
    *,
    upper_stl_path: Path | str,
    lower_stl_path: Path | str,
    photo_tiff_path: Path | str,
    case_id: str | None = None,
    segmenter: DadsnSegmenter | None = None,
) -> CaseBundle:
    """Create the reusable bundle and execute canonical DADSN exactly once."""

    bundle = load_case(
        upper_stl_path=upper_stl_path,
        lower_stl_path=lower_stl_path,
        photo_tiff_path=photo_tiff_path,
        case_id=case_id,
    )
    add_dadsn_segmentation(bundle, segmenter or DadsnSegmenter())
    bundle.require_segmentation()
    return bundle
