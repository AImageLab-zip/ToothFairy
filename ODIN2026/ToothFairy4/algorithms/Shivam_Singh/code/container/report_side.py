"""
Report generation: delegates to the collaborator's competition_runner.py.

competition_runner.py owns the whole VQA half -- it starts vLLM, renders the
images while the server loads, runs the VQA calls at a concurrency it sizes from
the server's reported KV pool, post-processes and synthesizes the report. This
module only has to hand it the two things the segmentation half produces
(mask.nii.gz + facts.json) and read back reports/<case>.txt.

Measured by the collaborator on a 24 GiB A30, cold compile cache:
    vLLM startup   190 s   (images finish 119 s before the server is ready)
    images          71 s
    inference       96 s   (concurrency ~10, from 65,472 KV tokens)
    TOTAL          287 s   = 32% of the 900 s budget

Two settings that are NOT free to change:
  * --max-model-len 32768. The whole-jaw calls request 8192 output tokens on
    top of a ~12.5k-character prompt. At 8192 they all fail with a 400 while
    the per-tooth calls still succeed, so the run exits 0 and writes a report
    with the whole-jaw findings silently missing.
  * AWQ weights, A10G or better. Marlin kernels need compute capability 8.0+;
    a T4 (7.5) cannot run them.

INTERFACE (unchanged): generate_report(cbct_image, mask_image, facts) -> str
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import SimpleITK as sitk

from config import (
    VLM_PYTHON, VLM_CODE_DIR, VLM_SCHEMA, VLM_MODEL_PATH, VLM_PORT,
    VLM_MAX_MODEL_LEN, VLM_MAX_IMAGES_PER_PROMPT, VLM_GPU_UTIL,
    VLM_STARTUP_TIMEOUT_S, VLM_ENABLED, REPORT_FALLBACK_ON_ERROR,
    VLM_MAX_CONCURRENCY,
)


def _stage(work: Path, case_id: str, cbct, mask, facts):
    """Write the handover artefacts: the volume, mask.nii.gz and facts.json."""
    vol_p = work / f"{case_id}.nii.gz"
    msk_p = work / "mask.nii.gz"
    fct_p = work / "facts.json"

    sitk.WriteImage(cbct, str(vol_p), True)
    sitk.WriteImage(mask, str(msk_p), True)

    # The renderer keys off structured.teeth_present: it draws and labels only
    # those teeth, so a segmentation false positive never becomes a tooth in
    # the report.
    rec = dict(facts) if isinstance(facts, dict) and "structured" in facts \
        else {"structured": facts}
    rec.setdefault("case_id", case_id)
    fct_p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return vol_p, msk_p, fct_p


def generate_report(cbct_image: sitk.Image, mask_image: sitk.Image, facts) -> str:
    if not VLM_ENABLED:
        return _template_fallback(facts)

    case_id = os.getenv("CASE_ID", "CASE")
    work = Path(tempfile.mkdtemp(prefix="vqarun_"))
    t0 = time.time()
    try:
        vol_p, msk_p, fct_p = _stage(work, case_id, cbct_image, mask_image, facts)
        out_dir = work / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            VLM_PYTHON, str(VLM_CODE_DIR / "competition_runner.py"),
            "--case-id", case_id,
            "--volume", str(vol_p),
            "--mask", str(msk_p),
            "--facts-file", str(fct_p),
            "--out-dir", str(out_dir),
            "--model", str(VLM_MODEL_PATH),
            "--schema", str(VLM_SCHEMA),
            "--project-dir", str(VLM_CODE_DIR.parent),
            "--port", str(VLM_PORT),
            "--max-model-len", str(VLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(VLM_GPU_UTIL),
            "--max-images-per-prompt", str(VLM_MAX_IMAGES_PER_PROMPT),
            "--startup-timeout", str(float(VLM_STARTUP_TIMEOUT_S)),
            "--max-concurrency", str(VLM_MAX_CONCURRENCY),
        ]
        # State of the compile cache, logged explicitly: a miss is silent
        # otherwise, and the difference is ~200s of startup.
        _cr = os.getenv("VLLM_CACHE_ROOT", "")
        print(f"=+= VLLM_CACHE_ROOT={_cr!r} exists={os.path.isdir(_cr) if _cr else False}",
              flush=True)
        if _cr and os.path.isdir(_cr):
            for _root, _dirs, _files in os.walk(_cr):
                print(f"=+=   cache: {_root} ({len(_files)} files)", flush=True)
        print("=+= competition_runner: " + " ".join(cmd), flush=True)

        # stdout/stderr are INHERITED so the runner's phase timings and the
        # vLLM startup breakdown land in the platform log. A PIPE nobody drains
        # is invisible and deadlocks once the ~64KB buffer fills.
        rc = subprocess.run(cmd, cwd=str(VLM_CODE_DIR.parent)).returncode
        print(f"=+= competition_runner exited {rc} "
              f"after {time.time() - t0:.1f}s", flush=True)
        if rc != 0:
            raise RuntimeError(f"competition_runner failed (rc={rc})")

        rep_p = out_dir / "reports" / f"{case_id}.txt"
        if not rep_p.exists():
            cands = sorted((out_dir / "reports").glob("*.txt"))
            if not cands:
                raise RuntimeError(f"no report under {out_dir / 'reports'}")
            rep_p = cands[0]
        report = rep_p.read_text(encoding="utf-8").strip()
        if not report:
            raise RuntimeError("synthesized report is empty")
        return report

    except Exception as e:
        print(f"=+= VQA pipeline FAILED: {e}", flush=True)
        if REPORT_FALLBACK_ON_ERROR:
            print("=+= falling back to template report", flush=True)
            return _template_fallback(facts)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# fallback: never submit an empty report
# ---------------------------------------------------------------------------
def _template_fallback(facts) -> str:
    s = facts.get("structured", facts) if isinstance(facts, dict) else facts
    absent = set(s.get("teeth_absent", []) or [])
    canals = s.get("canals", {}) or {}
    crowns = s.get("crowns", []) or []
    implants = [p for p in (s.get("implants") or []) if p]
    ian = s.get("ian_close_teeth", []) or []
    fov = s.get("fov", {}) or {}

    def tl(ids):
        return ", ".join(str(t) for t in sorted(ids))

    mand, maxi = [], []
    for side in ("right", "left"):
        if canals.get(side) == "present":
            mand.append(f"{side.capitalize()} mandibular canal with a regular course.")
    low = sorted(t for t in absent if t >= 30)
    up = sorted(t for t in absent if t < 30)
    if low:
        mand.append(f"Absence of teeth {tl(low)}.")
    if up:
        maxi.append(f"Absence of teeth {tl(up)}.")
    for t in crowns:
        (mand if t >= 30 else maxi).append(f"Prosthetic crown on tooth {t}.")
    for p in implants:
        (mand if p >= 30 else maxi).append(f"Endosseous implant in position {p}.")
    if s.get("bridge_present"):
        mand.append("A fixed prosthetic bridge is present.")
    for t in ian:
        mand.append(f"Tooth {t} is in close proximity to the mandibular canal.")
    mand.append("No definite osteolytic or osteosclerotic lesions.")
    if fov.get("maxilla") == "partial":
        maxi.insert(0, "The maxilla is partially included in the acquisition.")

    parts = ["Mandible:", *mand, "", "Maxilla:", *(maxi or ["Unremarkable."])]
    if fov.get("condyles") == "excluded":
        parts += ["", "Mandibular condyles:",
                  "Excluded from the acquisition and not visible."]
    return "\n".join(parts).strip()