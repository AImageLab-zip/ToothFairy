"""Data pipeline: split construction, the region-feature cache dataset, per-region GT targets."""
from .dataset import (  # noqa: F401
    ReportDataset, make_collate, read_split, build_region_targets,
)
# In-loop dataset that serves volumes directly, without the cached region bags.
