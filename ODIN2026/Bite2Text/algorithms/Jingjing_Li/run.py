#!/usr/bin/env python3
"""Generate a dental report using local models and a Python/CUDA environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    inputs = [input_dir / name for name in ("upper.stl", "lower.stl", "intraoral.tiff")]
    missing = [str(p) for p in inputs if not p.is_file() or p.stat().st_size == 0]
    if missing:
        parser.error("Missing or empty input files: " + ", ".join(missing))
    if not (ROOT / "model" / "report_runtime").is_dir():
        parser.error("Missing local model assets")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "structured_23.json", "report.txt"):
        if (output / name).exists():
            parser.error(f"Refusing to overwrite {output / name}")
    # Set paths before importing modules that read them at import time.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["B2T_MODEL_ROOT"] = str(ROOT / "model")
    os.environ["DADSN_ASSET_ROOT"] = str(ROOT / "model" / "dadsn")
    os.environ["B2T_REPORT_ASSET_ROOT"] = str(ROOT / "model" / "report_runtime")
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    # Segmentation writes work files beside its source; keep the installed tree clean.
    with tempfile.TemporaryDirectory(prefix="bite2text-") as temporary:
        source = Path(temporary) / "algorithm"
        shutil.copytree(ROOT / "algorithm", source)
        os.environ["B2T_RUNTIME_ROOT"] = str(source / "runtime")
        os.environ["DADSN_CODE_ROOT"] = str(source / "runtime" / "dadsn")
        os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, str(source))
        from inference import run_model
        result = run_model(upper_scan_path=inputs[0], lower_scan_path=inputs[1],
                           intraoral_photo_path=inputs[2], case_id="case")
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output / "structured_23.json").write_text(json.dumps(result["structured_intermediate"], ensure_ascii=False, indent=2) + "\n")
        (output / "report.txt").write_text(result["report"] + "\n")


if __name__ == "__main__":
    main()
