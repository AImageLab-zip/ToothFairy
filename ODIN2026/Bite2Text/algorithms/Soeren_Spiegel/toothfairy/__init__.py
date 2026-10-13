"""ToothFairy4 Qwen2.5-VL representation pipeline.

Port of HEADLINE's ``--recon video`` pipeline to dental CBCT: a panoramic (2D) image + a
cross-section (3D) video rendered from one CBCT volume per case, fed to Qwen2.5-VL. The package
covers the infra — loader, local-cache I/O, orientation, the Qwen video message path.

Runnable modules:
    python -m toothfairy.extract_cache        # extract raw volumes to local /cache
    python -m toothfairy.check_orientation    # audit affine/axcodes across the 4 families
    python -m toothfairy.verify_batch         # end-to-end batch + token-count checks
"""
from __future__ import annotations

from . import arch, config, render, volume_io
from .arch import ArchCurve, detect_arch_curve
from .dataset import (
    ToothFairyQwenDataset, assemble_messages, build_messages, extract_vision,
    load_split_records, processor_inputs, render_messages,
)
from .render import RenderedCase, render_case
from .volume_io import load_canonical_volume, reorient_to_canonical

__all__ = [
    "arch", "config", "render", "volume_io",
    "ToothFairyQwenDataset", "load_split_records", "render_messages", "build_messages",
    "assemble_messages", "extract_vision", "processor_inputs",
    "RenderedCase", "render_case",
    "ArchCurve", "detect_arch_curve",
    "load_canonical_volume", "reorient_to_canonical",
]
