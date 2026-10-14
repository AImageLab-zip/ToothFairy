import json
from pathlib import Path

import numpy as np
import pytest

from bitemark.config import UnfittedConfigurationError, load_config
from bitemark.iop import REQUIRED_VIEWS, assign_required_views
from bitemark.pipeline import run_case, run_from_features
from bitemark.schema import CaseInput

ROOT = Path(__file__).resolve().parents[1]


def test_placeholder_is_fail_closed():
    config = load_config(ROOT / "configs/public_placeholder.yaml")
    payload = json.loads((ROOT / "examples/synthetic_features.json").read_text(encoding="utf-8"))
    with pytest.raises(UnfittedConfigurationError):
        run_from_features(payload, config)


def test_synthetic_demo_has_warning_and_report():
    config = load_config(ROOT / "configs/public_placeholder.yaml")
    payload = json.loads((ROOT / "examples/synthetic_features.json").read_text(encoding="utf-8"))
    result = run_from_features(payload, config, allow_unfitted=True)
    assert result.case_id == "SYNTHETIC-DEMO"
    assert result.report
    assert "UNFITTED PUBLIC PLACEHOLDER" in result.warnings[0]


def test_global_view_assignment_is_one_to_one():
    images = [Path(f"image-{index}.jpg") for index in range(6)]
    probabilities = np.zeros((6, 5), dtype=float)
    probabilities[:5, :] = np.eye(5)
    probabilities[5, :] = 0.2
    assigned = assign_required_views(images, probabilities)
    assert tuple(assigned) == REQUIRED_VIEWS
    assert len(set(assigned.values())) == 5


def test_private_backends_can_be_injected_without_weights_in_repo():
    class FakeIOP:
        def classify_views(self, image_paths):
            return np.eye(5)

        def classify_crowding(self, arch, image_path):
            return "none-to-mild"

        def measure_midline(self, frontal_image):
            return 0.0

    class FakeIOS:
        def extract_features(self, upper_stl, lower_stl):
            return json.loads(
                (ROOT / "examples/synthetic_features.json").read_text(encoding="utf-8")
            )["ios"]

    case = CaseInput(
        case_id="SYNTHETIC-INJECTED",
        image_paths=tuple(Path(f"image-{index}.jpg") for index in range(5)),
        upper_stl=Path("upper.stl"),
        lower_stl=Path("lower.stl"),
    )
    result = run_case(
        case,
        load_config(ROOT / "configs/public_placeholder.yaml"),
        FakeIOP(),
        FakeIOS(),
        allow_unfitted=True,
    )
    assert result.case_id == "SYNTHETIC-INJECTED"
