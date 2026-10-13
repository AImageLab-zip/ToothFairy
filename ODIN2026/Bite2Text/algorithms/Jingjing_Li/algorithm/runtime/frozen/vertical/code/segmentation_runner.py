#!/usr/bin/env python3
"""Run the deterministic two-pass DADSN segmentation bundled with Vertical."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class FrozenTwoPassSegmenter:
    def __init__(
        self,
        code_root: str | Path,
        asset_root: str | Path,
        device: str = "cuda:0",
    ):
        self.code_root = Path(code_root).expanduser().resolve()
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.device = device
        self.script = self.code_root / "code" / "predict1_retry_configurable.py"
        self.model = self.asset_root / "models" / "dadsn_all_align_best_model.pth"
        self.label_dir = self.asset_root / "label_c17"
        self.model_source = self.code_root / "code" / "dadsn_model.py"
        self.iops_source = self.code_root / "code" / "dadsn_iops.py"
        for path in (self.script, self.model, self.model_source, self.iops_source):
            if not path.is_file():
                raise FileNotFoundError(path)
        for name in ("color_dict.json", "label_decoder_upper.json", "label_decoder_lower.json"):
            if not (self.label_dir / name).is_file():
                raise FileNotFoundError(self.label_dir / name)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_stl(path: str | Path) -> Path:
        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        return source

    def segment_pair(
        self,
        upper_stl: str | Path,
        lower_stl: str | Path,
    ) -> dict[str, Any]:
        upper = self._validate_stl(upper_stl)
        lower = self._validate_stl(lower_stl)
        temporary = tempfile.TemporaryDirectory(prefix="vertical_segmentation_")
        work = Path(temporary.name)
        runtime = work / "runtime"
        input_dir = work / "input"
        output_dir = work / "output"
        debug_dir = work / "debug"
        runtime.mkdir()
        input_dir.mkdir()
        output_dir.mkdir()

        case = "generic_case"
        shutil.copy2(upper, input_dir / f"{case}_ios_upper.stl")
        shutil.copy2(lower, input_dir / f"{case}_ios_lower.stl")
        shutil.copy2(self.model_source, runtime / "model.py")
        shutil.copy2(self.iops_source, runtime / "iops.py")
        shutil.copytree(self.label_dir, runtime / "label_c17")
        models = runtime / "outputs" / "All_align" / "models"
        models.mkdir(parents=True)
        shutil.copy2(self.model, models / "best_model.pth")

        env = os.environ.copy()
        python_paths = [str(runtime)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        if self.device == "cpu":
            raise RuntimeError("Grand Challenge Vertical runtime requires CUDA; CPU fallback is disabled")
        if self.device.startswith("cuda:") and self.device != "cuda:0":
            raise RuntimeError(f"Vertical must use the first visible GPU (cuda:0), got {self.device}")

        command = [
            os.environ.get("VERTICAL_PYTHON", os.sys.executable),
            str(self.script),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--debug_dir",
            str(debug_dir),
            "--align_seed",
            "0",
            "--pass1_seed",
            "11",
            "--pass2_seed",
            "29",
            "--retry_pass1_seed",
            "47",
            "--align_trials",
            "4",
            "--no_debug",
        ]
        completed = subprocess.run(
            command,
            cwd=runtime,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            temporary.cleanup()
            raise RuntimeError(
                f"Two-pass segmentation failed ({completed.returncode}):\n{completed.stdout[-4000:]}"
            )

        result: dict[str, Any] = {"_temporary": temporary}
        arch_debug: dict[str, Any] = {}
        for arch in ("upper", "lower"):
            prefix = f"{case}_ios_{arch}"
            raw_ply = output_dir / f"{prefix}_pass2_full_raw.ply"
            labels_json = output_dir / f"{prefix}.json"
            internal_npy = output_dir / f"{prefix}.npy"
            alignment = output_dir / f"{prefix}_alignment_selection.json"
            if not raw_ply.is_file() or not labels_json.is_file() or not internal_npy.is_file() or not alignment.is_file():
                temporary.cleanup()
                raise RuntimeError(f"Missing two-pass segmentation output for {arch}")
            labels = json.loads(labels_json.read_text(encoding="utf-8"))
            alignment_payload = json.loads(alignment.read_text(encoding="utf-8"))
            result[arch] = {
                "raw_ply": str(raw_ply),
                "labels_json": str(labels_json),
                "internal_npy": str(internal_npy),
            }
            arch_debug[arch] = {
                "vertex_labels": len(labels),
                "chosen_alignment_trial": alignment_payload.get("chosen_trial"),
                "chosen_alignment_seed": alignment_payload.get("chosen_seed"),
                "chosen_alignment_score": alignment_payload.get("chosen_score"),
                "labels_sha256": self._sha256(labels_json),
            }
        result["debug"] = {
            "parameters": {
                "align_seed": 0,
                "pass1_seed": 11,
                "pass2_seed": 29,
                "retry_pass1_seed": 47,
                "align_trials": 4,
            },
            "arches": arch_debug,
            "log_tail": completed.stdout[-4000:],
        }
        return result
