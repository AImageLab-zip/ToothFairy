"""CPU-only release checks; these do not assert successful GPU retraining."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_training_sources_parse():
    sources = json.loads((ROOT / "training/provenance/sources.json").read_text())
    for record in sources["sources"]:
        path = ROOT / record["public_source"]
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert len(record["retrieved_source_sha256"]) == 64


def test_all_released_checkpoints_have_training_provenance():
    weights = json.loads((ROOT / "release/weights.json").read_text())["weights"]
    facts = json.loads((ROOT / "training/provenance/checkpoints.json").read_text())
    assert {w["p"] for w in weights} == {w["path"] for w in facts["checkpoints"]}
    for fact in facts["checkpoints"]:
        if "training_source" in fact:
            assert (ROOT / fact["training_source"]).is_file()


def test_curve_cohort_profiles_keep_wilson_binary():
    tree = ast.parse((ROOT / "training/ios/train_curves.py").read_text())
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets)
    ) or (isinstance(n, ast.FunctionDef) and n.name == "configure_split_profile")]
    namespace = {}
    # Only the repository-owned constant and pure profile function, not GPU imports.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "profiles", "exec"), namespace)  # noqa: S102
    configure = namespace["configure_split_profile"]
    configure("wilson-final2")
    assert namespace["TASKS"]["spee"]["classes"] == ["Normal", "Increased", "Reversed"]
    assert namespace["TASKS"]["wilson"]["classes"] == ["Normal", "Increased"]
    configure("spee-final2")
    assert namespace["TASKS"]["spee"]["classes"] == ["Normal", "Increased"]
    with pytest.raises(ValueError):
        configure("unknown")
