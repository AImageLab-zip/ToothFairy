"""Categorical claim schema (per-tooth 10-axis + non-teeth region axes).

Single canonical definition of the claim output space shared by the findings classification and
report generation pipelines. Lowest-level component - it does not depend on any other toothfairy
module.
"""
from .claims import (  # noqa: F401
    FDI_UP, FDI_LO, FDI_ALL,
    Axis, AXES, N_AXES, AUX_DIM, SOFT_DIM,
    SOFTMAX_AXES, SIGMOID_AXES, AXIS_NAMES,
    logit_slices,
    build_axis_labels, compute_class_weights,
    MAXILLA_AXES, MANDIBLE_AXES, NERVE_AXES, NONTEETH_AXES,
    region_aux_dim, region_logit_slices, build_nonteeth_labels,
)
