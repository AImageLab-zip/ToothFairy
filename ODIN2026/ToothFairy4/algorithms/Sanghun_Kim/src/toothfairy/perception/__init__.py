"""Perception components — frozen segmentation backbone → per-patch region features.

Parts:
  region_pooling — pool_patch·RegionBag·POOL_CLASSES (import-light, no nnU-Net)
  backbone       — build_predictor·hook (needs nnU-Net)
  cache          — extract_region_bag (needs nnU-Net)

pool_patch/RegionBag are re-exported here so they can be used without nnU-Net, while backbone
and cache are imported directly from their submodules so that the heavy nnU-Net dependency is
loaded lazily (`from toothfairy.perception.backbone import build_predictor`).
"""
from .region_pooling import pool_patch, RegionBag, POOL_CLASSES  # noqa: F401

