"""
ToothFairy4 algorithm entrypoint (Grand Challenge).

  /input/images/cbct/*.mha  ->  /output/diagnostic-imaging-report.json

Pipeline: segment the RAW volume (no affine reorientation; auto-orient handled
inside run_segmentation, mask returned in input space) -> extract facts ->
optional handoff reorientation -> report. Runs offline, single GPU.

Orientation note: per the ToothFairy4 dataset correction, the reports align with
the RAW image content, so we do NOT reorient by affine. See segmentation.py.
"""

import glob
import json
import time
from pathlib import Path

import SimpleITK as sitk

from config import (
    HANDOFF_ORIENTATION,
    VERIFY_ORIENTATION,
    ORIENTATION_CHECK_FATAL,
    UPPER_TOOTH_LABELS,
    LOWER_TOOTH_LABELS,
)
from orientation import reorient, current_axcodes, maxilla_above_mandible
from segmentation import run_segmentation
from facts import run_fact_extraction
from report_side import generate_report

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def run() -> int:
    _t_case = time.time()
    _t = time.time()
    cbct = load_cbct(INPUT_PATH / "images" / "cbct")
    print(f"=+= input axcodes: {current_axcodes(cbct)}  size={cbct.GetSize()}")

    # 1) segment the RAW volume; mask comes back in input space
    _t = time.time()
    mask = run_segmentation(cbct)
    _t_seg = time.time() - _t
    print(f"=+= TIMING segmentation      {_t_seg:8.1f}s", flush=True)

    # 2) anatomical sanity check (physical space; frame-agnostic)
    if VERIFY_ORIENTATION:
        if not maxilla_above_mandible(mask, UPPER_TOOTH_LABELS, LOWER_TOOTH_LABELS):
            msg = "maxilla-above-mandible check FAILED — orientation suspect"
            if ORIENTATION_CHECK_FATAL:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

    # 3) facts (extract_facts is orientation-agnostic; runs on the input-space mask)
    _t = time.time()
    facts = run_fact_extraction(cbct, mask)
    _t_facts = time.time() - _t
    print(f"=+= TIMING fact extraction   {_t_facts:8.1f}s", flush=True)

    # 4) handoff frame for her VLM (default INPUT = raw, report-aligned)
    cbct_out, mask_out = apply_handoff_frame(cbct, mask)

    # 5) her half
    _t = time.time()
    report = generate_report(cbct_out, mask_out, facts)
    _t_rep = time.time() - _t
    print(f"=+= TIMING report generation {_t_rep:8.1f}s", flush=True)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_PATH / "diagnostic-imaging-report.json", {"report": report})
    print("=+= wrote diagnostic-imaging-report.json")
    _tot = time.time() - _t_case
    print("=+= ========== CASE TIMING SUMMARY ==========", flush=True)
    print(f"=+=   segmentation      {_t_seg:8.1f}s", flush=True)
    print(f"=+=   fact extraction   {_t_facts:8.1f}s", flush=True)
    print(f"=+=   report generation {_t_rep:8.1f}s", flush=True)
    print(f"=+=   CASE TOTAL        {_tot:8.1f}s  ({_tot/60:.1f} min)", flush=True)
    print(f"=+=   GC LIMIT           900.0s  (15.0 min)", flush=True)
    print("=+= ==========================================", flush=True)
    return 0


def apply_handoff_frame(cbct: sitk.Image, mask: sitk.Image):
    """Both cbct and mask are in raw input space. Default: hand over as-is
    (raw = report-aligned). Reorient only if HANDOFF_ORIENTATION names a frame."""
    target = HANDOFF_ORIENTATION
    if target in ("", "INPUT", "RAW"):
        return cbct, mask
    print(f"=+= handoff reorient -> {target}")
    return reorient(cbct, target), reorient(mask, target)


def load_cbct(location: Path) -> sitk.Image:
    files = sorted(glob.glob(str(location / "*.mha")))
    if not files:
        raise RuntimeError(f"No .mha CBCT found in {location}")
    if len(files) > 1:
        print(f"Multiple .mha in {location}; using {Path(files[0]).name}")
    return sitk.ReadImage(files[0])


def write_json(location: Path, content: dict) -> None:
    location.write_text(json.dumps(content, indent=4), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())