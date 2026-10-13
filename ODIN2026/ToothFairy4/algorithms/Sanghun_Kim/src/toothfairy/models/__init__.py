"""Assembled report-generation model.

report_model — cached perception -> encoder -> [findings head] -> realizer + losses.
Experiment variants are selected through config axes
(perception.backbone, train.findings_head, train.use_siglip).
"""
from .report_model import ReportModel  # noqa: F401
# Classification-only model with no realizer — shares the encoder and the findings head.
from .findings_model import FindingsModel  # noqa: F401
