#!/usr/bin/env python3
"""Download pinned official backbones; rerun safely after interrupted downloads.

Usage: python prepare_models.py [--model-root PATH]
Optional mirror: HF_ENDPOINT=https://your-huggingface-mirror python prepare_models.py
Existing files are verified, never silently trusted or replaced when invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


QWEN_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# Exact files only: no repository-wide snapshot, alternate weights, or training data.
QWEN_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
RESNET_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"


def verify_digest(path: Path, digest: str, size: int | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"Incorrect size: {path}. Remove this file explicitly and rerun.")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != digest:
        raise RuntimeError(f"SHA256 mismatch: {path}. Remove this file explicitly and rerun.")


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"Model configuration path escapes its directory: {relative}")
    return path


def prepare_qwen(model_root: Path) -> None:
    # Import after command-line parsing so HF_ENDPOINT is read from the user's environment.
    from huggingface_hub import hf_hub_download

    runtime_root = model_root / "report_runtime"
    config = json.loads((runtime_root / "RUNTIME_CONFIG.json").read_text(encoding="utf-8"))
    model_config = config["model"]
    manifest_path = contained_path(runtime_root, model_config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (model_config["model_id"], model_config["revision"]) != (QWEN_ID, QWEN_REVISION):
        raise RuntimeError("RUNTIME_CONFIG.json does not specify the supported pinned Qwen model.")
    if (manifest["model_id"], manifest["model_revision"]) != (QWEN_ID, QWEN_REVISION):
        raise RuntimeError("MODEL_MANIFEST.json does not match the pinned Qwen model.")
    destination = contained_path(runtime_root, model_config["local_path"])
    records = {record["path"]: record for record in manifest["model_files"]}
    missing = set(QWEN_FILES) - records.keys()
    if missing:
        raise RuntimeError(f"MODEL_MANIFEST.json lacks required files: {sorted(missing)}")
    endpoint = os.environ.get("HF_ENDPOINT")
    print(f"Qwen source: {endpoint or 'https://huggingface.co'}; revision {QWEN_REVISION}", flush=True)
    for filename in QWEN_FILES:
        target = destination / filename
        record = records[filename]
        if target.exists() or target.is_symlink():
            verify_digest(target, record["sha256"], record["bytes"])
            print(f"Verified existing: {target}", flush=True)
            continue
        print(f"Downloading Qwen: {filename}", flush=True)
        downloaded = Path(hf_hub_download(
            repo_id=QWEN_ID,
            filename=filename,
            revision=QWEN_REVISION,
            local_dir=str(destination),
            endpoint=endpoint,
        ))
        verify_digest(downloaded, record["sha256"], record["bytes"])
    print("Qwen runtime files verified.", flush=True)


def download_resnet(source: Path) -> None:
    if source.exists() or source.is_symlink():
        verify_digest(source, RESNET_SHA256)
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    # Never expose an interrupted download under the final filename.
    fd, temporary_name = tempfile.mkstemp(prefix=source.name + ".", suffix=".part", dir=source.parent)
    temporary = Path(temporary_name)
    try:
        print(f"Downloading official ResNet18: {RESNET_URL}", flush=True)
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(RESNET_URL, timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        verify_digest(temporary, RESNET_SHA256)
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_resnet(model_root: Path) -> None:
    import torch
    from torchvision.models import resnet18

    source = model_root / ".downloads" / "resnet18-f37072fd.pth"
    download_resnet(source)
    official_state = torch.load(source, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(official_state, strict=True)
    state = model.state_dict()
    outputs = (
        (model_root / "frozen/curves/models/shared/resnet18_imagenet_default_state_dict.pt", True),
        (model_root / "frozen/dental_hygiene/models/resnet18_imagenet_state_dict.pt", False),
    )
    for target, wrapped in outputs:
        if target.exists() or target.is_symlink():
            try:
                existing = torch.load(target, map_location="cpu", weights_only=True)
                if wrapped:
                    if existing["architecture"] != "resnet18" or existing["weights_name"] != WEIGHTS_NAME:
                        raise ValueError("backbone metadata mismatch")
                    existing = existing["state_dict"]
                if existing.keys() != state.keys() or any(
                    not isinstance(existing[key], torch.Tensor)
                    or existing[key].dtype != value.dtype
                    or existing[key].shape != value.shape
                    or not torch.equal(existing[key], value)
                    for key, value in state.items()
                ):
                    raise ValueError("backbone tensors differ from official weights")
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid existing backbone: {target}: {exc}. Remove this file explicitly and rerun."
                ) from exc
            print(f"Verified existing: {target}", flush=True)
            continue
        payload = {"architecture": "resnet18", "weights_name": WEIGHTS_NAME, "state_dict": state} if wrapped else state
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as output:
                torch.save(payload, output)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Prepared: {target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-root", type=Path, default=Path(__file__).resolve().parent / "model",
        help="Model bundle root (default: model beside this script, independent of working directory)",
    )
    args = parser.parse_args()
    try:
        model_root = args.model_root.expanduser().resolve()
        prepare_qwen(model_root)
        prepare_resnet(model_root)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Rerun the same command to continue safely.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Model preparation failed: {exc}", file=sys.stderr)
        print(
            "Install requirements.txt first and check network access/free disk space. "
            "For a Hugging Face mirror, set HF_ENDPOINT=https://your-mirror before launching. "
            "Rerun after fixing the error; verified files are retained and Hugging Face partial downloads can resume.",
            file=sys.stderr,
        )
        return 1
    print("Official model preparation complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
