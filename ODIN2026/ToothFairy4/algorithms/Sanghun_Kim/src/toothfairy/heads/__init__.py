"""Prediction heads for the factored pipeline.

findings — categorical claim heads: 10 axes per tooth plus the non-teeth anatomical
regions (maxilla, mandible, mandibular canals, arch summaries).
"""
from .findings import FindingsHead  # noqa: F401
