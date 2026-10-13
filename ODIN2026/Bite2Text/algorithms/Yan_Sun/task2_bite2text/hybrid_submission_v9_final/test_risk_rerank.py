#!/usr/bin/env python3
"""Unit tests for the conservative v9 retrieval gate."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# The selector tests exercise no tensor operations. A tiny module stub keeps
# source-level CI independent of the 5+ GB CUDA/PyTorch runtime image.
torch_stub = types.ModuleType("torch")
sys.modules.setdefault("torch", torch_stub)

# Import only the retrieval logic without loading the production mesh pipeline.
inference_stub = types.ModuleType("inference")
for name in (
    "env_path",
    "inference_sample",
    "input_files",
    "load_model",
    "postcorrect_triangles",
    "run_ios_normalizer",
    "sampled_coordinates",
):
    setattr(inference_stub, name, lambda *args, **kwargs: None)
sys.modules["inference"] = inference_stub

retrieval_stub = types.ModuleType("retrieval_inference")
retrieval_stub.descriptor_from_coord = lambda value: value
retrieval_stub.load_assets = lambda *args, **kwargs: None
sys.modules["retrieval_inference"] = retrieval_stub

photo_stub = types.ModuleType("photo_inference")
photo_stub.run_photo_inference = lambda *args, **kwargs: None
sys.modules["photo_inference"] = photo_stub

from hybrid_inference import CV_F1, select_report


class RiskRerankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.heads = list(CV_F1)
        self.predicted = {head: "a" for head in self.heads}
        self.confidence = {head: 0.9 for head in self.heads}
        baseline_labels = {head: None for head in self.heads}
        baseline_labels[self.heads[0]] = "b"
        safe_labels = {head: None for head in self.heads}
        self.labels = [baseline_labels, safe_labels]
        self.config = {
            "enabled": True,
            "margin": 0.02,
            "unsupported_penalty": 0.005,
            "unsupported_gate": 5,
            "contradiction_threshold": 0.65,
            "contradiction_gate": 0.01,
            "contradiction_penalty": 0.5,
            "min_contradiction_improvement": 0.015,
            "no_new_unsupported": True,
        }

    def select(self, *, enabled: bool) -> tuple[int, dict[str, object]]:
        config = {**self.config, "enabled": enabled}
        result = select_report(
            descriptor=np.asarray([1.0]),
            predicted_labels=self.predicted,
            patient_ids=np.asarray(["baseline", "safer"]),
            database=np.asarray([[1.0], [0.995]], dtype=np.float64),
            mean=np.asarray([0.0]),
            scale=np.asarray([1.0]),
            reports=[
                "The dental arches are described.",
                "The dental arches are described conservatively.",
            ],
            candidate_labels=self.labels,
            top_k=2,
            blend_lambda=0.5,
            confidence=self.confidence,
            risk_config=config,
        )
        return result[0], result[-1]

    def test_reranks_close_high_confidence_contradiction(self) -> None:
        index, summary = self.select(enabled=True)
        self.assertEqual(index, 1)
        self.assertTrue(summary["reranked"])
        self.assertEqual(summary["reason"], "contradiction")

    def test_disable_flag_is_exact_v8_fallback(self) -> None:
        index, summary = self.select(enabled=False)
        self.assertEqual(index, 0)
        self.assertFalse(summary["reranked"])

    def test_unsupported_route_does_not_require_contradiction(self) -> None:
        self.confidence = {head: 0.5 for head in self.heads}
        with patch("hybrid_inference.unsupported_sentence_count", side_effect=[5, 0]):
            index, summary = self.select(enabled=True)
        self.assertEqual(index, 1)
        self.assertEqual(summary["reason"], "unsupported")
        self.assertEqual(summary["baseline_contradiction"], 0)

    def test_below_both_activation_thresholds_keeps_baseline(self) -> None:
        self.confidence = {head: 0.5 for head in self.heads}
        with patch("hybrid_inference.unsupported_sentence_count", side_effect=[4, 0]):
            index, summary = self.select(enabled=True)
        self.assertEqual(index, 0)
        self.assertFalse(summary["reranked"])

    def test_candidate_outside_score_margin_is_rejected(self) -> None:
        self.config["margin"] = 0.0
        index, summary = self.select(enabled=True)
        self.assertEqual(index, 0)
        self.assertFalse(summary["reranked"])

    def test_new_unsupported_sentence_blocks_contradiction_improvement(self) -> None:
        # Isolate the hard veto from the separate soft penalty.
        self.config["unsupported_penalty"] = 0.0
        with patch("hybrid_inference.unsupported_sentence_count", side_effect=[0, 1]):
            index, summary = self.select(enabled=True)
        self.assertEqual(index, 0)
        self.assertFalse(summary["reranked"])
        self.config["no_new_unsupported"] = False
        with patch("hybrid_inference.unsupported_sentence_count", side_effect=[0, 1]):
            index, summary = self.select(enabled=True)
        self.assertEqual(index, 1)
        self.assertTrue(summary["reranked"])


if __name__ == "__main__":
    unittest.main()
