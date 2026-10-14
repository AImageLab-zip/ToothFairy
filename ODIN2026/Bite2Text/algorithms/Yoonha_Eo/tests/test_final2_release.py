"""CPU-only tests; no claim of end-to-end GPU validation."""
import ast
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_final2_bundle import make_notebook, sha256
from download_models import safe_extract
from prepare_input import prepare


def test_pipeline_and_adapter_provenance():
    provenance = json.loads((ROOT / "release/provenance.json").read_text())
    assert sha256(ROOT / "final2/pipeline.py") == provenance["public_pipeline_code_sha256"]
    assert sha256(ROOT / "submission/inference.py") == provenance["adapter_sha256"]
    code = (ROOT / "final2/pipeline.py").read_text(encoding="utf-8")
    ast.parse(code)
    nb = make_notebook(code)
    assert "".join(nb["cells"][0]["source"]) == code
    assert nb["cells"][0]["outputs"] == []
    assert nb["cells"][0]["execution_count"] is None


def test_checkpoint_manifest_complete():
    entries = json.loads((ROOT / "release/weights.json").read_text())["weights"]
    assert len(entries) == 9
    assert len({e["p"] for e in entries}) == 9
    assert all(len(e["h"]) == 64 and e["b"] > 1_000_000 for e in entries)


def test_calibration_gate_inference_fields_only():
    gate = json.loads((ROOT / "final2/model/u/canine-eruption-gate.json").read_text())
    assert set(gate) == {"features", "mean", "scale", "coefficient", "intercept",
                         "probability_threshold"}
    assert len(gate["features"]) == len(gate["mean"]) == len(gate["scale"])
    assert len(gate["scale"]) == len(gate["coefficient"])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\escape"])
def test_unsafe_archive_rejected(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo(name)
        entry.size = 1
        tar.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_link_archive_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../outside"
        tar.addfile(entry)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_safe_archive(tmp_path):
    archive = tmp_path / "good.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("w/example.txt")
        entry.size = 2
        tar.addfile(entry, io.BytesIO(b"ok"))
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out/w/example.txt").read_bytes() == b"ok"


def test_staging_and_original_socket_handler(tmp_path):
    upper, lower, photo = [tmp_path / n for n in ("upper.stl", "lower.stl", "view.png")]
    for p in (upper, lower, photo):
        p.write_bytes(b"synthetic test payload, not clinical data")
    target = tmp_path / "input"
    prepare(upper, lower, [photo], target)
    assert (target / "files/ios-upper/ios_upper.stl").read_bytes() == upper.read_bytes()
    spec = importlib.util.spec_from_file_location("submission_adapter", ROOT / "submission/inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((target / "inputs.json").read_text())
    assert module.select_handler(manifest) is module.full_pipeline_handler
    with pytest.raises(FileExistsError):
        prepare(upper, lower, [photo], target)
