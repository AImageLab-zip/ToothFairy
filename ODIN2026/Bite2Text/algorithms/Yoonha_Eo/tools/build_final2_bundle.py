"""Build the public model bundle from checked-in sources and exact final2 weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_notebook(code: str) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"id": "final2-inference", "cell_type": "code", "metadata": {},
                   "execution_count": None, "outputs": [], "source": code.splitlines(True)}],
    }


def build(weights: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    entries = json.loads((ROOT / "release/weights.json").read_text())["weights"]
    for entry in entries:
        source = weights / Path(entry["p"]).name
        if source.stat().st_size != entry["b"] or sha256(source) != entry["h"]:
            raise ValueError(f"Not the submitted final2 checkpoint: {source.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bitemark-release-") as temp:
        model = Path(temp) / "model"
        shutil.copytree(ROOT / "final2/model", model,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        code = (ROOT / "final2/pipeline.py").read_text(encoding="utf-8")
        write_json(model / "p/n.ipynb", make_notebook(code))
        (model / "w").mkdir()
        for entry in entries:
            shutil.copyfile(weights / Path(entry["p"]).name, model / entry["p"])
        shutil.copytree(ROOT / "LICENSES", model / "LICENSES")
        for name in ("LICENSE", "NOTICE.md"):
            shutil.copyfile(ROOT / name, model / name)
        files = [{"p": p.relative_to(model).as_posix(), "b": p.stat().st_size,
                  "h": sha256(p)} for p in sorted(model.rglob("*")) if p.is_file()]
        write_json(model / "m.json", {"v": 6, "n": sha256(model / "p/n.ipynb"), "f": files})
        # Regular files only; no symlinks, caches or patient inputs are bundled.
        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            for path in sorted(model.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(model).as_posix())
    return {"name": output.name, "bytes": output.stat().st_size, "sha256": sha256(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path,
                        help="Directory containing the nine original checkpoint files")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.weights, args.output), indent=2))
