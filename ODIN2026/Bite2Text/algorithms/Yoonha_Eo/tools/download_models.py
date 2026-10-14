"""Download, verify and safely extract the public final2 model bundle (stdlib only)."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from build_final2_bundle import ROOT, sha256


def safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            part = PurePosixPath(member.name)
            if (not member.isfile() or part.is_absolute() or ".." in part.parts
                    or "\\" in member.name or ":" in member.name):
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Duplicate archive member: {member.name}")
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def verify_model(model: Path) -> None:
    manifest = json.loads((model / "m.json").read_text())
    if manifest.get("v") != 6:
        raise ValueError("Unsupported model manifest")
    root = model.resolve()
    for entry in manifest["f"]:
        path = (root / entry["p"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes model directory")
        if path.stat().st_size != entry["b"] or sha256(path) != entry["h"]:
            raise ValueError(f"Invalid model file: {entry['p']}")
    if sha256(model / "p/n.ipynb") != manifest["n"]:
        raise ValueError("Notebook checksum mismatch")
    expected = json.loads((ROOT / "release/weights.json").read_text())["weights"]
    for entry in expected:
        if sha256(model / entry["p"]) != entry["h"]:
            raise ValueError(f"Weight differs from final2: {entry['p']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/final2")
    parser.add_argument("--archive", type=Path, help="Verify a previously downloaded archive")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_model(args.output)
        print("All model files and nine original checkpoint hashes verified.")
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --verify-only")
    release = json.loads((ROOT / "release/model_bundle.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bitemark-download-",
                                     dir=args.output.parent) as temp:
        temporary = Path(temp)
        archive = args.archive
        if archive is None:
            archive = temporary / release["name"]
            print(f"Downloading {release['url']}", flush=True)
            with urllib.request.urlopen(release["url"], timeout=120) as source, archive.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if archive.stat().st_size != release["bytes"] or sha256(archive) != release["sha256"]:
            raise ValueError("Archive integrity check failed; nothing installed")
        staged = temporary / "model"
        staged.mkdir()
        safe_extract(archive, staged)
        verify_model(staged)
        staged.rename(args.output)
    print(f"Verified final2 model installed at {args.output}")


if __name__ == "__main__":
    main()
