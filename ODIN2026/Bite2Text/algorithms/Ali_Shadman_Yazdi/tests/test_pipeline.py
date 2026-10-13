"""Tests. The end-to-end test needs all weights and takes ~2 min on a GPU.

    pytest -q                   # everything
    pytest -q -m "not slow"     # seconds, no weights needed
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "sample_data" / "F5535"


def _load_score_module():
    spec = importlib.util.spec_from_file_location("score", ROOT / "scripts" / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_meteor_lite_identical_text_scores_high():
    score = _load_score_module()
    text = "Sagittally, there is a bilateral Class I molar and canine relationship."
    assert score.meteor_lite([text], [text]) > 0.99
    assert score.meteor_lite([text], ["Completely different words here."]) < 0.2


def test_prior_photo_section_renders():
    from bite2text.photo_findings import load_prior
    from bite2text.render import render_photo_findings

    text = render_photo_findings(load_prior())
    assert text.strip()
    assert "sextant" not in text.lower()          # generic_only drops localised claims


def test_occlusal_render_is_complete():
    from bite2text.render import render_occlusal

    slots = {"constr": "none", "xbite_present": False, "deep": False, "overbite": "normal",
             "overjet": "normal", "mr_class": "I", "ml_class": "I", "cr_class": "I", "cl_class": "I",
             "mid": "deviated", "mid_mag": "slight", "spee": "normal", "wilson": "increased",
             "cru": "mild", "crl": "mild"}
    text = render_occlusal(slots)
    for word in ("transverse", "Class I", "midlines", "Curve of Spee", "crowding"):
        assert word.lower() in text.lower()


@pytest.mark.skipif(not (SAMPLE / "ios_upper.stl").exists(), reason="sample data not present")
def test_scan_format_is_sniffed_from_content(tmp_path):
    from bite2text.scan_io import load_scan, sniff_mesh_type

    mislabelled = tmp_path / "3d-upper-teeth-scan.obj"      # STL bytes, OBJ name (as on Grand Challenge)
    shutil.copyfile(SAMPLE / "ios_upper.stl", mislabelled)
    assert sniff_mesh_type(mislabelled) == "stl"
    assert len(load_scan(mislabelled).faces) > 100_000


def _weights_ready() -> bool:
    from bite2text import config
    return not config.missing_weights() and (config.photo_cnn_dir() / "best.pt").is_file()


@pytest.mark.slow
@pytest.mark.skipif(not (SAMPLE / "expected_output.json").exists(), reason="sample data not present")
@pytest.mark.skipif(not _weights_ready(), reason="weights missing: run scripts/download_weights.py")
def test_sample_case_reproduces_expected_report():
    from bite2text import ReportGenerator

    expected = json.loads((SAMPLE / "expected_output.json").read_text())
    photos = sorted(SAMPLE.glob("intraoral_*.jpg"))
    report = ReportGenerator(seed=0)(SAMPLE / "ios_upper.stl", SAMPLE / "ios_lower.stl", photos)

    assert report.photo_source == "model"
    assert report.slots == expected["slots"]
    assert report.text == expected["report"]
