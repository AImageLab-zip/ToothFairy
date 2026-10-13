#!/usr/bin/env python3
"""Verify/recover the published v9 assets without importing pickle or PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

MANIFEST = Path(__file__).resolve().parents[1] / "reproducibility/v9_assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular file (symlinks not accepted): {path}")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Size/SHA-256 mismatch: {path}")


def verify_model(root: Path, manifest: dict) -> None:
    for name, expected in manifest["model_files"].items():
        verify_file(root / name, expected)


def safe_flat_name(name: str) -> str:
    # No links, directories, absolute paths or nested paths enter the model mount.
    path = PurePosixPath(name)
    if path.is_absolute() or len(path.parts) != 1 or path.name in ("", ".", "..") or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path.name


def extract_model(archive: Path, destination: Path, manifest: dict) -> None:
    verify_file(archive, manifest["model_archive"])
    if destination.exists():
        raise ValueError(f"Refusing to overwrite: {destination}")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if not (m.isdir() and m.name in (".", "./"))]
        names = [safe_flat_name(m.name) for m in members]
        if len(set(names)) != len(names) or set(names) != set(manifest["model_files"]):
            raise ValueError("Model archive must contain exactly the nine manifest files")
        if any(not m.isfile() for m in members):
            raise ValueError("Archive links/devices are not allowed")
        for member, name in zip(members, names):
            if member.size != manifest["model_files"][name]["bytes"]:
                raise ValueError(f"Unexpected member size: {name}")
        destination.mkdir(parents=True)
        for member, name in zip(members, names):
            with tar.extractfile(member) as source, (destination / name).open("xb") as target:
                shutil.copyfileobj(source, target)
    verify_model(destination, manifest)


def unpack_submission(archive: Path, destination: Path, manifest: dict) -> None:
    release = manifest["archives"]["submission"]
    verify_file(archive, release)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [safe_flat_name(m.filename) for m in members]
        # Published archive member names are immutable compatibility metadata.
        expected = {manifest["image_archive"]["filename"], manifest["model_archive"]["filename"],
                    *release["metadata_files"]}
        if len(set(names)) != len(names) or set(names) != expected:
            raise ValueError("Unexpected submission bundle members")
        destination.mkdir(parents=True)
        for member, name in zip(members, names):
            with bundle.open(member) as source, (destination / name).open("xb") as target:
                shutil.copyfileobj(source, target)
    verify_file(destination / manifest["image_archive"]["filename"], manifest["image_archive"])
    extract_model(destination / manifest["model_archive"]["filename"], destination / "model", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-archive")
    verify.add_argument("kind", choices=("submission", "weights"))
    verify.add_argument("archive", type=Path)
    unpack = sub.add_parser("unpack-submission")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    model = sub.add_parser("verify-model")
    model.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "verify-archive":
        verify_file(args.archive, manifest["archives"][args.kind])
    elif args.command == "unpack-submission":
        unpack_submission(args.archive, args.destination, manifest)
    else:
        verify_model(args.directory, manifest)
    print("PASS: published v9 asset integrity verified (no checkpoint deserialization)")


if __name__ == "__main__":
    main()
