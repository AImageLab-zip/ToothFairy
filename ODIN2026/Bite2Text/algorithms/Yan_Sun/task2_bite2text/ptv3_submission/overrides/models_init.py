"""Minimal Pointcept model registry for Bite2Text PTv3 inference."""

from .builder import build_model
from .modules import PointModule, PointModel
from .point_transformer_v3.point_transformer_v3m1_base import *
from .multi_task_classifier.multi_task_classifier_v1m1_base import MultiTaskClassifier
