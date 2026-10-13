#!/usr/bin/env python3
"""Official ToothFairy4 container adapter for one CBCT-to-report job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("/input")
DEFAULT_OUTPUT_DIR = Path("/output")
DEFAULT_CONFIG = REPO_ROOT / "evigre/configs/inference/evident_cbct.container.json"
OFFICIAL_INPUT_SLUG = "cbct-image"
OFFICIAL_OUTPUT_NAME = "diagnostic-imaging-report.json"
SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd")


class SubmissionError(RuntimeError):
    pass


def _input_slugs(input_root: Path) -> tuple[str, ...] | None:
    inputs_json = input_root / "inputs.json"
    if not inputs_json.is_file():
        return None
    try:
        payload = json.loads(inputs_json.read_text(encoding="utf-8"))
        slugs = [str(item["socket"]["slug"]) for item in payload]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SubmissionError(f"Invalid Grand Challenge inputs.json: {inputs_json}") from error
    return tuple(sorted(slugs))


def validate_interface(input_root: Path) -> None:
    slugs = _input_slugs(input_root)
    if slugs is not None and slugs != (OFFICIAL_INPUT_SLUG,):
        raise SubmissionError(
            f"Expected ToothFairy4 input socket {(OFFICIAL_INPUT_SLUG,)}, got {slugs}"
        )


def _has_supported_suffix(path: Path) -> bool:
    return any(path.name.lower().endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def resolve_submission_volume(input_root: Path) -> Path:
    root = input_root.expanduser().resolve()
    if root.is_file():
        candidates = [root] if _has_supported_suffix(root) else []
    elif root.is_dir():
        official_root = root / "images" / "cbct"
        search_root = official_root if official_root.is_dir() else root
        candidates = sorted(
            (path.resolve() for path in search_root.rglob("*") if path.is_file() and _has_supported_suffix(path)),
            key=str,
        )
    else:
        raise FileNotFoundError(root)

    if len(candidates) != 1:
        raise SubmissionError(
            f"Expected exactly one CBCT volume under {root}, found {len(candidates)}: "
            + ", ".join(str(path) for path in candidates[:8])
        )
    return candidates[0]


def infer_case_id(path: Path) -> str:
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def convert_to_nifti(source: Path, scratch_dir: Path) -> Path:
    if source.name.lower().endswith((".nii", ".nii.gz")):
        return source

    import SimpleITK as sitk

    try:
        image = sitk.ReadImage(str(source))
    except Exception as error:
        raise SubmissionError(f"Cannot read CBCT image {source}: {error}") from error
    if image.GetDimension() != 3:
        raise SubmissionError(f"Expected a 3D CBCT image, got dimension {image.GetDimension()}: {source}")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    target = scratch_dir / f"{infer_case_id(source)}.nii.gz"
    sitk.WriteImage(image, str(target), True)
    return target


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_submission(
    input_root: Path,
    output_dir: Path,
    config: Path,
    *,
    scratch_dir: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    input_root = input_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    validate_interface(input_root)
    source = resolve_submission_volume(input_root)
    case_id = infer_case_id(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_output = output_dir / OFFICIAL_OUTPUT_NAME
    official_output.unlink(missing_ok=True)

    scratch = (scratch_dir or Path(os.environ.get("EVIDENT_SUBMISSION_SCRATCH", "/tmp/evident-cbct-submission"))).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)

    try:
        nifti = convert_to_nifti(source, scratch)
        command: Sequence[str] = (
            sys.executable,
            "-m",
            "evigre.inference",
            str(nifti),
            str(output_dir),
            "--case-id",
            case_id,
            "--config",
            str(config),
            "--clean-work",
            "--overwrite",
        )
        print(f"ToothFairy4 input: {source}", flush=True)
        if nifti != source:
            print(f"Converted Grand Challenge MHA to NIfTI: {nifti}", flush=True)
        runner(command, check=True)

        report_path = output_dir / "report.txt"
        if not report_path.is_file():
            raise SubmissionError(f"Inference completed without report.txt: {report_path}")
        report = report_path.read_text(encoding="utf-8").strip()
        if not report:
            raise SubmissionError("Inference produced an empty clinical report")
        _atomic_write_json(official_output, {"report": report})
        print(f"Official ToothFairy4 output: {official_output}", flush=True)
        return 0
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one official ToothFairy4 CBCT-to-report job.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_submission(args.input_root, args.output_dir, args.config)
    except (FileNotFoundError, SubmissionError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
