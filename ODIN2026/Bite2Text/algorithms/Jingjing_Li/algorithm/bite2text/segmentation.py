"""Live adapter for the canonical DADSN two-pass segmenter.

`predict1.py` remains the segmentation implementation of record. This adapter
only stages one case for that script, validates its full-vertex output, and
decodes C17 class indices to FDI labels for the shared pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
import time
import csv
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DADSN_CODE_ROOT = Path("/opt/app/runtime/dadsn")
DEFAULT_DADSN_ASSET_ROOT = Path("/opt/ml/model/dadsn")
OUTPUT_DIRECTORY = Path("work/output_two_pass_geodesic_recall_guard")


@dataclass(frozen=True)
class DadsnSegmentationResult:
    """Final labels aligned with the original upper/lower STL vertex order."""

    upper_c17: np.ndarray
    lower_c17: np.ndarray
    upper_fdi: np.ndarray
    lower_fdi: np.ndarray
    upper_original_c17: np.ndarray
    lower_original_c17: np.ndarray
    upper_original_fdi: np.ndarray
    lower_original_fdi: np.ndarray
    upper_mask: np.ndarray
    lower_mask: np.ndarray
    selection: dict[str, str]
    qc_summary: list[dict[str, str]]
    qc_tooth_metrics: list[dict[str, str]]


class DadsnSegmenter:
    """Run one canonical two-pass DADSN segmentation job inside the container."""

    def __init__(
        self,
        *,
        code_root: Path | str | None = None,
        asset_root: Path | str | None = None,
        cuda_visible_devices: str | None = None,
        keep_workspace: bool = False,
    ) -> None:
        self.code_root = Path(code_root or os.environ.get("DADSN_CODE_ROOT", DEFAULT_DADSN_CODE_ROOT)).resolve()
        self.asset_root = Path(asset_root or os.environ.get("DADSN_ASSET_ROOT", DEFAULT_DADSN_ASSET_ROOT)).resolve()
        self.root = self.code_root
        if cuda_visible_devices is not None:
            raise ValueError("Production DADSN may not override CUDA_VISIBLE_DEVICES; use cuda:0 within the container")
        self.cuda_visible_devices = None
        self.keep_workspace = keep_workspace
        self.last_predict_seconds = 0.0

    def predict(
        self,
        *,
        upper_stl_path: Path | str,
        lower_stl_path: Path | str,
        case_id: str | None = None,
    ) -> DadsnSegmentationResult:
        started = time.perf_counter()
        upper_stl_path = Path(upper_stl_path).resolve()
        lower_stl_path = Path(lower_stl_path).resolve()
        self._validate_runtime(upper_stl_path, lower_stl_path)
        case_stem = self._case_stem(case_id)

        work_dir = self.root / "work"
        try:
            self._stage_case(
                work_dir=work_dir,
                upper_stl_path=upper_stl_path,
                lower_stl_path=lower_stl_path,
                case_stem=case_stem,
            )
            self._run_predict1()
            original_output = self.root / OUTPUT_DIRECTORY
            upper_original_c17 = self._load_c17(f"{case_stem}_ios_upper", original_output)
            lower_original_c17 = self._load_c17(f"{case_stem}_ios_lower", original_output)
            selected_output, selected_qc_summary, selected_tooth_metrics = self._select_final_segmentation()
            self._render_crowding_masks(selected_output)
            upper_c17 = self._load_c17(f"{case_stem}_ios_upper", selected_output)
            lower_c17 = self._load_c17(f"{case_stem}_ios_lower", selected_output)
            upper_fdi = self._decode_fdi(upper_c17, arch="upper")
            lower_fdi = self._decode_fdi(lower_c17, arch="lower")
            masks_dir = self.root / "work" / "crowding_masks"
            selection = self._load_selection(case_stem)
            return DadsnSegmentationResult(
                upper_c17=upper_c17,
                lower_c17=lower_c17,
                upper_fdi=upper_fdi,
                lower_fdi=lower_fdi,
                upper_original_c17=upper_original_c17,
                lower_original_c17=lower_original_c17,
                upper_original_fdi=self._decode_fdi(upper_original_c17, arch="upper"),
                lower_original_fdi=self._decode_fdi(lower_original_c17, arch="lower"),
                upper_mask=np.asarray(Image.open(masks_dir / f"{case_stem}_ios_upper_label.png").convert("L"), dtype=np.uint8),
                lower_mask=np.asarray(Image.open(masks_dir / f"{case_stem}_ios_lower_label.png").convert("L"), dtype=np.uint8),
                selection=selection,
                qc_summary=self._read_csv(selected_qc_summary),
                qc_tooth_metrics=self._read_csv(selected_tooth_metrics),
            )
        finally:
            self.last_predict_seconds = time.perf_counter() - started
            if not self.keep_workspace:
                shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _case_stem(case_id: str | None) -> str:
        stem = "case" if case_id is None else str(case_id).strip()
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not stem or any(character not in allowed for character in stem):
            raise ValueError("case_id must contain only letters, digits, underscores, or hyphens")
        return stem

    def _validate_runtime(self, upper_stl_path: Path, lower_stl_path: Path) -> None:
        required = (
            self.code_root / "predict1.py",
            self.code_root / "model.py",
            self.code_root / "iops.py",
            self.asset_root / "label_c17" / "label_decoder_upper.json",
            self.asset_root / "label_c17" / "label_decoder_lower.json",
            self.asset_root / "outputs" / "All_align" / "models" / "best_model.pth",
            self.code_root / "segmentation_qc_fdi.py",
            self.code_root / "build_segmentation_qc_v2.py",
            self.code_root / "run_predict1_retry.py",
            self.code_root / "select_best_segmentation.py",
            self.code_root / "crowding" / "final_freeze_v1" / "render_occlusal_label_masks_compat.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete split DADSN runtime: {missing}")
        for path, description in ((upper_stl_path, "upper STL"), (lower_stl_path, "lower STL")):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty {description}: {path}")

    def _stage_case(
        self,
        *,
        work_dir: Path,
        upper_stl_path: Path,
        lower_stl_path: Path,
        case_stem: str,
    ) -> None:
        shutil.rmtree(work_dir, ignore_errors=True)
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True)
        # The canonical script infers arch from these stable filename tokens.
        shutil.copy2(upper_stl_path, input_dir / f"{case_stem}_ios_upper.stl")
        shutil.copy2(lower_stl_path, input_dir / f"{case_stem}_ios_lower.stl")

    def _run_predict1(self) -> None:
        environment = self._runtime_environment()
        command = [sys.executable, "predict1.py", "--no_debug"]
        try:
            subprocess.run(command, cwd=self.code_root, env=environment, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Canonical DADSN predict1.py failed with exit code {error.returncode}") from error

    def _select_final_segmentation(self) -> tuple[Path, Path, Path]:
        work = self.root / "work"
        original = self.root / OUTPUT_DIRECTORY
        qc = work / "segmentation_qc"
        qc_v2 = work / "segmentation_qc_v2"
        self._run_command([
            sys.executable, "segmentation_qc_fdi.py", "--output-dir", str(original),
            "--input-dir", str(work / "input"), "--qc-dir", str(qc),
        ])
        self._run_command([
            sys.executable, "build_segmentation_qc_v2.py", "--summary", str(qc / "segmentation_qc_summary.csv"),
            "--tooth-metrics", str(qc / "segmentation_qc_tooth_metrics.csv"), "--output-dir", str(qc_v2),
        ])
        retry_cases = (qc_v2 / "retry_cases_v2.txt").read_text(encoding="utf-8").split()
        selected = original
        if retry_cases:
            retry_output = work / "output2_retry1"
            retry_qc = work / "segmentation_qc_retry1"
            retry_qc_v2 = work / "segmentation_qc_retry1_v2"
            self._run_command([
                sys.executable, "run_predict1_retry.py", "--qc-v2-summary", str(qc_v2 / "segmentation_qc_v2_summary.csv"),
                "--source-input-dir", str(work / "input"), "--retry-input-dir", str(work / "input_retry1"),
                "--retry-output-dir", str(retry_output), "--retry-debug-dir", str(work / "debug_retry1"),
                "--patched-script", "predict1_retry_configurable.py", "--execute", "--clean-retry-input",
            ])
            self._run_command([
                sys.executable, "segmentation_qc_fdi.py", "--output-dir", str(retry_output),
                "--input-dir", str(work / "input_retry1"), "--qc-dir", str(retry_qc),
            ])
            self._run_command([
                sys.executable, "build_segmentation_qc_v2.py", "--summary", str(retry_qc / "segmentation_qc_summary.csv"),
                "--tooth-metrics", str(retry_qc / "segmentation_qc_tooth_metrics.csv"), "--output-dir", str(retry_qc_v2),
            ])
            selected = work / "output2_selected"
            self._run_command([
                sys.executable, "select_best_segmentation.py", "--original-qc", str(qc_v2 / "segmentation_qc_v2_summary.csv"),
                "--retry-qc", str(retry_qc_v2 / "segmentation_qc_v2_summary.csv"), "--original-output", str(original),
                "--retry-output", str(retry_output), "--selected-output", str(selected), "--mode", "symlink", "--clean",
            ])
        if selected == original:
            print(
                "B2T_QC_REUSE selected segmentation is original; reusing identical original QC/QCv2",
                flush=True,
            )
            return original, qc_v2 / "segmentation_qc_v2_summary.csv", qc / "segmentation_qc_tooth_metrics.csv"
        selected_qc = work / "segmentation_qc_selected"
        selected_qc_v2 = work / "segmentation_qc_selected_v2"
        self._run_command([
            sys.executable, "segmentation_qc_fdi.py", "--output-dir", str(selected),
            "--input-dir", str(work / "input"), "--qc-dir", str(selected_qc),
        ])
        self._run_command([
            sys.executable, "build_segmentation_qc_v2.py", "--summary", str(selected_qc / "segmentation_qc_summary.csv"),
            "--tooth-metrics", str(selected_qc / "segmentation_qc_tooth_metrics.csv"), "--output-dir", str(selected_qc_v2),
        ])
        return selected, selected_qc_v2 / "segmentation_qc_v2_summary.csv", selected_qc / "segmentation_qc_tooth_metrics.csv"

    def _render_crowding_masks(self, selected_output: Path) -> None:
        self._run_command([
            sys.executable, "crowding/final_freeze_v1/render_occlusal_label_masks_compat.py",
            "--segmentation-dir", str(selected_output), "--output-dir", str(self.root / "work" / "crowding_masks"),
            "--target-size", "1024", "--render-size", "1400", "--padding", "12", "--overwrite",
        ])

    def _load_selection(self, case_stem: str) -> dict[str, str]:
        manifest = self.root / "work" / "output2_selected" / "selection_manifest.csv"
        if not manifest.is_file():
            return {"upper": "original", "lower": "original"}
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0].split(",")
        result: dict[str, str] = {}
        for line in lines[1:]:
            values = line.split(",")
            row = dict(zip(header, values, strict=True))
            if row["case"].startswith(f"{case_stem}_ios_"):
                result[row["case"].rsplit("_", 1)[-1]] = row["selected_source"]
        return result

    def _run_command(self, command: list[str]) -> None:
        environment = self._runtime_environment()
        try:
            subprocess.run(command, cwd=self.code_root, env=environment, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"DADSN selection command failed with exit code {error.returncode}: {command}") from error

    def _runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["DADSN_ASSET_ROOT"] = str(self.asset_root)
        python_paths = [str(self.code_root)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        return environment

    def _load_c17(self, stem: str, output_dir: Path) -> np.ndarray:
        path = output_dir / f"{stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"DADSN did not produce final full-vertex labels: {path}")
        labels = np.asarray(np.load(path), dtype=np.int64)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"Invalid DADSN C17 output at {path}: shape={labels.shape}")
        if labels.min() < 0 or labels.max() > 16:
            raise ValueError(f"DADSN C17 output is outside [0, 16] at {path}")
        return labels

    def _decode_fdi(self, c17: np.ndarray, *, arch: str) -> np.ndarray:
        decoder_path = self.asset_root / "label_c17" / f"label_decoder_{arch}.json"
        decoder_payload = json.loads(decoder_path.read_text(encoding="utf-8"))["label"]
        decoder = np.asarray([decoder_payload[str(index)] for index in range(17)], dtype=np.int64)
        return decoder[c17]
    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
