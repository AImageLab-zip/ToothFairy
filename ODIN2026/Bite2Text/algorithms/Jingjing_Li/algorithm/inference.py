"""Grand Challenge entrypoint for the raw-only Bite2Text structured runtime.

The production path accepts exactly one upper STL, one lower STL, and one
five-frame TIFF, then emits the complete v3.1 23-field structured intermediate.
"""

from __future__ import annotations

import glob
import shutil
import tempfile
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from bite2text import FIELD_COUNT, generate_report, predict_structured, preprocess_case
from bite2text.runtime_paths import (
    DADSN_ASSET_ROOT,
    DADSN_CODE_ROOT,
    FROZEN_ASSET_ROOT,
    FROZEN_CODE_ROOT,
    MODEL_ROOT,
    REPORT_ASSET_ROOT,
    RUNTIME_ROOT,
    require_directory,
)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
FROZEN_PATH = FROZEN_ASSET_ROOT
DADSN_PATH = DADSN_ASSET_ROOT


def run():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    handler = select_handler(inputs)
    return handler()


def select_handler(inputs: list[dict]) -> callable:
    slug_set = {
        str((value.get("socket") or {}).get("slug") or "").strip()
        for value in inputs
    }

    lower_slugs = {"3d-lower-teeth-scan", "ios-lower-scan"}
    upper_slugs = {"3d-upper-teeth-scan", "ios-upper-scan"}
    photo_slugs = {"2d-intraoral-photographs", "intraoral-photo"}

    has_lower = any(slug in slug_set for slug in lower_slugs)
    has_upper = any(slug in slug_set for slug in upper_slugs)
    has_photo = any(slug in slug_set for slug in photo_slugs)

    if has_lower and has_upper and has_photo:
        return interf0_handler

    raise RuntimeError(
        "Unsupported input socket configuration. "
        f"Found slugs: {sorted(slug for slug in slug_set if slug)}"
    )


def interf0_handler() -> int:
    lower_scan = load_single_file_compat(
        preferred_files=(INPUT_PATH / "3d-lower-teeth-scan.obj",),
        fallback_locations=(INPUT_PATH / "files" / "ios-lower",),
        fallback_patterns=("*.stl",),
    )
    upper_scan = load_single_file_compat(
        preferred_files=(INPUT_PATH / "3d-upper-teeth-scan.obj",),
        fallback_locations=(INPUT_PATH / "files" / "ios-upper",),
        fallback_patterns=("*.stl",),
    )
    intraoral_tiff = load_single_file_compat(
        preferred_files=(),
        fallback_locations=(
            INPUT_PATH / "images" / "2d-intraoral-photographs",
            INPUT_PATH / "images" / "intraoral-photo",
        ),
        fallback_patterns=("*.tif", "*.tiff"),
    )

    with tempfile.TemporaryDirectory(prefix="bite2text_input_compat_") as temporary:
        compatibility_root = Path(temporary)
        lower_scan = normalize_scan_suffix(lower_scan, compatibility_root, "lower")
        upper_scan = normalize_scan_suffix(upper_scan, compatibility_root, "upper")
        require_production_cuda()
        result = run_model(
            lower_scan_path=lower_scan,
            upper_scan_path=upper_scan,
            intraoral_photo_path=intraoral_tiff,
            report_device=os.environ.get("BITE2TEXT_REPORT_DEVICE", "auto"),
            enable_report_llm=os.environ.get("BITE2TEXT_DISABLE_REPORT_LLM", "0") != "1",
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(
        location=OUTPUT_PATH / "diagnostic-imaging-report.json",
        content={"report": result["report"]},
    )
    return 0


def run_model(
    *,
    lower_scan_path: Path,
    upper_scan_path: Path,
    intraoral_photo_path: Path,
    case_id: str | None = None,
    device: str = "auto",
    report_device: str = "auto",
    enable_report_llm: bool = True,
    runtime_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Return final English, structured facts, and report-generation audit."""

    require_production_cuda()
    total_started = time.perf_counter()
    roots = runtime_roots or _runtime_roots()
    require_directory(MODEL_ROOT, "Grand Challenge model root")
    require_directory(RUNTIME_ROOT, "container runtime code root")
    os.environ["DADSN_CODE_ROOT"] = str(roots["dadsn_code"])
    os.environ["DADSN_ASSET_ROOT"] = str(roots["dadsn"])
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    segmenter = __import__("bite2text.segmentation", fromlist=["DadsnSegmenter"]).DadsnSegmenter(
        code_root=roots["dadsn_code"],
        asset_root=roots["dadsn"],
    )
    preprocess_started = time.perf_counter()
    bundle = preprocess_case(
        upper_stl_path=upper_scan_path,
        lower_stl_path=lower_scan_path,
        photo_tiff_path=intraoral_photo_path,
        case_id=case_id,
        segmenter=segmenter,
    )
    preprocess_seconds = time.perf_counter() - preprocess_started
    dadsn_seconds = segmenter.last_predict_seconds
    print(f"B2T_TIMING preprocess_total_seconds={preprocess_seconds:.6f}")
    print(f"B2T_TIMING dadsn_seconds={dadsn_seconds:.6f}")
    print(f"B2T_TIMING preprocess_excluding_dadsn_seconds={max(0.0, preprocess_seconds - dadsn_seconds):.6f}")
    structured_started = time.perf_counter()
    structured = predict_structured(
        bundle,
        runtime_roots=roots,
        device=device,
        include_debug=False,
    )
    structured_seconds = time.perf_counter() - structured_started
    print(f"B2T_TIMING structured_predictors_seconds={structured_seconds:.6f}")
    field_count = sum(len(fields) for fields in structured.values())
    if field_count != FIELD_COUNT or FIELD_COUNT != 23:
        raise RuntimeError(f"Unexpected structured field count: {field_count}")
    report_started = time.perf_counter()
    print(f"B2T_STAGE_START report elapsed_total={report_started - total_started:.6f}", flush=True)
    print("B2T_STAGE_REPORT_MODEL_LOAD start", flush=True)
    report_result = generate_report(
        structured,
        runtime_root=roots["report_runtime"],
        device=report_device,
        enable_llm=enable_report_llm,
    )
    print(f"B2T_STAGE_REPORT_MODEL_LOAD seconds={report_result.model_load_seconds:.6f}", flush=True)
    print(f"B2T_STAGE_REPORT_GENERATE seconds={report_result.generation_seconds:.6f}", flush=True)
    print(
        f"B2T_STAGE_DONE report seconds={time.perf_counter() - report_started:.6f} "
        f"elapsed_total={time.perf_counter() - total_started:.6f}",
        flush=True,
    )
    print(f"B2T_TIMING qwen_load_seconds={report_result.model_load_seconds:.6f}", flush=True)
    print(f"B2T_TIMING qwen_generation_seconds={report_result.generation_seconds:.6f}", flush=True)
    print(f"B2T_TIMING total_seconds={time.perf_counter() - total_started:.6f}", flush=True)
    print(f"B2T_ELAPSED_TOTAL seconds={time.perf_counter() - total_started:.6f}", flush=True)
    return {
        "report": report_result.report,
        "structured_intermediate": structured,
        "schema_version": "3.1-preserve-midline-crowding",
        "field_count": field_count,
        "report_generation": report_result.to_dict(),
    }


def _runtime_roots() -> dict[str, Path]:
    return {
        "sagittal_code": FROZEN_CODE_ROOT / "sagittal",
        "sagittal": FROZEN_ASSET_ROOT / "sagittal",
        "transverse_code": FROZEN_CODE_ROOT / "transverse",
        "transverse": FROZEN_ASSET_ROOT / "transverse",
        "vertical_code": FROZEN_CODE_ROOT / "vertical",
        "vertical": FROZEN_ASSET_ROOT / "vertical",
        "crowding": FROZEN_ASSET_ROOT / "crowding",
        "curves_code": FROZEN_CODE_ROOT / "curves",
        "curves": FROZEN_ASSET_ROOT / "curves",
        "dental_hygiene_code": FROZEN_CODE_ROOT / "dental_hygiene",
        "dental_hygiene": FROZEN_ASSET_ROOT / "dental_hygiene",
        "report_runtime": REPORT_ASSET_ROOT,
        "midline": DADSN_CODE_ROOT / "midline",
        "dadsn_code": DADSN_CODE_ROOT,
        "dadsn": DADSN_ASSET_ROOT,
    }


def load_json_file(*, location):
    with open(location, encoding="utf-8") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w", encoding="utf-8") as f:
        f.write(json.dumps(content, indent=4))


def load_single_file_compat(
    *,
    preferred_files: tuple[Path, ...],
    fallback_locations: tuple[Path, ...],
    fallback_patterns: tuple[str, ...],
) -> Path:
    for file_path in preferred_files:
        if file_path.exists():
            return file_path

    for location in fallback_locations:
        for pattern in fallback_patterns:
            input_files = sorted(glob.glob(str(location / pattern)))
            if not input_files:
                continue
            if len(input_files) > 1:
                print(f"Multiple files found in {location}; using {Path(input_files[0]).name}")
            return Path(input_files[0])

    fallback_desc = ", ".join(str(path) for path in fallback_locations)
    raise RuntimeError(
        "No compatible input file found. "
        f"Checked preferred files: {[str(path) for path in preferred_files]}, "
        f"fallback locations: {fallback_desc}, patterns: {fallback_patterns}"
    )
def normalize_scan_suffix(path: Path, destination_root: Path, arch: str) -> Path:
    """Expose extension-mislabeled binary STL socket files as `.stl`."""

    if path.suffix.lower() != ".obj" or not _looks_like_binary_stl(path):
        return path
    destination = destination_root / f"{arch}.stl"
    shutil.copy2(path, destination)
    return destination


def _looks_like_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as handle:
        header = handle.read(84)
    triangle_count = int.from_bytes(header[80:84], "little")
    return 84 + triangle_count * 50 == size




def require_production_cuda() -> None:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print("=+" * 10)
    print("Bite2Text production CUDA diagnostic")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unset')}")
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available()={available}")
    print(f"torch.cuda.device_count()={count}")
    if available and count:
        print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
        print("production_device=cuda:0 (first GPU visible inside container)")
    print("=+" * 10)
    if not available or count < 1:
        raise RuntimeError(
            "Grand Challenge GPU submission requires at least one visible CUDA GPU. "
            "CPU fallback is disabled to prevent MaxRuntimeExceeded."
        )


def _show_torch_cuda_info() -> None:
    """Backward-compatible alias for diagnostics used by local harnesses."""
    require_production_cuda()


if __name__ == "__main__":
    raise SystemExit(run())
