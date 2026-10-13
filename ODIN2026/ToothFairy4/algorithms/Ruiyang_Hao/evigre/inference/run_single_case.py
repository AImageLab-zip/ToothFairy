#!/usr/bin/env python3
"""Run EviDent-CBCT from one preprocessed CBCT volume to one clinical report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import nibabel as nib
import numpy as np
import torch

from evigre.inference.evidence_filter import SUPPORTED_FILTER_VERSIONS


REPO_ROOT = Path(__file__).resolve().parents[2]
UMAMBA2_ROOT = Path(os.environ.get("UMAMBA2_DIR", REPO_ROOT / "third_party" / "U-Mamba2")).expanduser().resolve()
MODELS_ROOT = Path(os.environ.get("EVIDENT_MODELS", REPO_ROOT / "models")).expanduser().resolve()
DEFAULT_CONFIG = REPO_ROOT / "evigre/configs/inference/evident_cbct.example.json"
KNOWN_OUTPUTS = {
    "report.txt",
    "report.json",
    "template_report.txt",
    "evidence.json",
    "evidence_filtered.json",
    "evidence_raw.json",
    "evidence_filter_audit.json",
    "consistency_audit.json",
    "pipeline_manifest.json",
    "pipeline.log",
}


class PipelineError(RuntimeError):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _expand_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config(item) for item in value]
    if isinstance(value, str):
        expanded = value.format(repo_root=REPO_ROOT, umamba2_root=UMAMBA2_ROOT, models_root=MODELS_ROOT)
        return os.path.expandvars(os.path.expanduser(expanded))
    return value


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = _expand_config(json.loads(path.read_text()))
    for section in ("pipeline_version", "stage1", "stage2", "stage3"):
        if section not in config:
            raise PipelineError(f"Missing {section!r} in inference config: {path}")
    filter_version = str(config["stage3"].get("evidence_filter", ""))
    if filter_version not in SUPPORTED_FILTER_VERSIONS:
        raise PipelineError(
            f"Unsupported stage3.evidence_filter {filter_version!r} in {path}; "
            f"expected one of {sorted(SUPPORTED_FILTER_VERSIONS)}"
        )
    if not isinstance(config["stage3"].get("add_preamble", False), bool):
        raise PipelineError(f"stage3.add_preamble must be a boolean in {path}")
    return config


def resolve_input_volume(input_path: Path) -> Path:
    source = input_path.expanduser().resolve()
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        preferred = [
            source / "cbct/volume.nii.gz",
            source / "volume.nii.gz",
            source / "cbct/volume.nii",
            source / "volume.nii",
        ]
        candidates = [path for path in preferred if path.is_file()]
        if not candidates:
            candidates = sorted(source.rglob("*.nii.gz")) + sorted(source.rglob("*.nii"))
    else:
        raise FileNotFoundError(source)

    unique = list(dict.fromkeys(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise PipelineError(
            f"Expected exactly one NIfTI CBCT volume under {source}, found {len(unique)}: "
            + ", ".join(str(path) for path in unique[:8])
        )
    volume = unique[0]
    if not (volume.name.endswith(".nii.gz") or volume.name.endswith(".nii")):
        raise PipelineError(f"Input must be a .nii or .nii.gz NIfTI volume: {volume}")
    return volume


def infer_case_id(volume: Path) -> str:
    if volume.name == "volume.nii.gz" or volume.name == "volume.nii":
        if volume.parent.name.lower() == "cbct":
            return volume.parent.parent.name
        return volume.parent.name
    if volume.name.endswith(".nii.gz"):
        return volume.name[:-7]
    return volume.stem


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def inspect_volume(path: Path) -> Dict[str, Any]:
    try:
        image = nib.load(str(path))
    except Exception as error:
        raise PipelineError(f"Cannot read NIfTI volume {path}: {error}") from error
    if len(image.shape) != 3:
        raise PipelineError(f"Expected a 3D CBCT volume, got shape {image.shape}: {path}")
    if any(int(size) <= 0 for size in image.shape):
        raise PipelineError(f"Invalid CBCT shape {image.shape}: {path}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(spacing) != 3 or not all(np.isfinite(value) and value > 0 for value in spacing):
        raise PipelineError(f"Invalid voxel spacing {spacing}: {path}")
    affine = np.asarray(image.affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise PipelineError(f"Invalid NIfTI affine: {path}")
    return {
        "shape_xyz": [int(value) for value in image.shape],
        "spacing_xyz_mm": list(spacing),
        "dtype": str(image.get_data_dtype()),
        "orientation": list(nib.aff2axcodes(image.affine)),
    }


def _path(config: Mapping[str, Any], section: str, key: str) -> Path:
    return Path(str(config[section][key]))


def preflight(
    config: Mapping[str, Any],
    config_path: Path,
    input_volume: Path,
    output_dir: Path,
    device: str,
    skip_rewrite: bool,
) -> Dict[str, Any]:
    if not device.startswith("cuda"):
        raise PipelineError("The released pipeline requires a CUDA device.")
    if not torch.cuda.is_available():
        raise PipelineError("CUDA was requested but torch.cuda.is_available() is false.")

    if ":" in device:
        index = int(device.split(":", 1)[1])
    else:
        index = 0
    if index < 0 or index >= torch.cuda.device_count():
        raise PipelineError(f"CUDA device {device} is unavailable; found {torch.cuda.device_count()} device(s).")

    stage2 = config["stage2"]
    required_files = {
        "inference_config": config_path,
        "stage1_checkpoint": _path(config, "stage1", "checkpoint"),
        "stage1_task1_inference": _path(config, "stage1", "task1_inference_py"),
        "stage2_global_checkpoint": Path(stage2["global_folder"]) / stage2["checkpoint_name"],
        "stage2_global_thresholds": Path(stage2["global_thresholds"]),
        "stage2_relation_checkpoint": Path(stage2["relation_folder"]) / stage2["checkpoint_name"],
        "stage2_relation_thresholds": Path(stage2["relation_thresholds"]),
        "stage2_tooth_checkpoint": Path(stage2["tooth_folder"]) / stage2["checkpoint_name"],
        "stage2_tooth_thresholds": Path(stage2["tooth_thresholds"]),
        "stage2_tooth_region_mask": Path(stage2["tooth_region_mask_config"]),
        "stage2_preprocessing": REPO_ROOT / "evigre/inference/stage2_preprocessing.py",
    }
    if not skip_rewrite:
        adapter = _path(config, "stage3", "qwen_adapter")
        required_files.update(
            {
                "qwen_adapter_config": adapter / "adapter_config.json",
                "qwen_adapter_weights": adapter / "adapter_model.safetensors",
                "qwen_adapter_tokenizer": adapter / "tokenizer.json",
            }
        )

    missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise PipelineError("Missing required inference artifact(s):\n  " + "\n  ".join(missing))

    for key in ("global_thresholds", "relation_thresholds", "tooth_thresholds"):
        threshold_path = Path(stage2[key])
        try:
            payload = json.loads(threshold_path.read_text())
        except Exception as error:
            raise PipelineError(f"Invalid threshold JSON {threshold_path}: {error}") from error
        if not payload:
            raise PipelineError(f"Empty threshold JSON: {threshold_path}")

    if not skip_rewrite:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(config["stage3"]["qwen_base"], local_files_only=True)
        except Exception as error:
            raise PipelineError(
                f"Qwen base model is not available in the local Hugging Face cache: "
                f"{config['stage3']['qwen_base']}"
            ) from error

    output_parent = output_dir.parent if output_dir.parent.exists() else REPO_ROOT
    free_bytes = shutil.disk_usage(output_parent).free
    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(index),
        "cuda_device_count": torch.cuda.device_count(),
        "input": inspect_volume(input_volume),
        "required_artifacts": {name: str(path) for name, path in required_files.items()},
        "free_disk_gib": round(free_bytes / (1024**3), 2),
        "offline_rewrite": not skip_rewrite,
        "evidence_filter": str(config["stage3"]["evidence_filter"]),
    }


class CommandRunner:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def run(self, stage: str, command: Iterable[str], env: Mapping[str, str]) -> float:
        command = [str(value) for value in command]
        rendered = shlex.join(command)
        started = time.monotonic()
        header = f"\n[{stage}] $ {rendered}\n"
        print(header.strip(), flush=True)
        with self.log_path.open("a") as log:
            log.write(header)
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        elapsed = time.monotonic() - started
        if return_code != 0:
            raise PipelineError(f"Stage {stage!r} failed with exit code {return_code}. See {self.log_path}")
        return elapsed


def _read_single_report(csv_path: Path) -> Dict[str, str]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise PipelineError(f"Expected one generated report in {csv_path}, found {len(rows)}")
    report = str(rows[0].get("generated_report", "")).strip()
    if not report:
        raise PipelineError(f"Generated report is empty: {csv_path}")
    return {"case_id": str(rows[0].get("case_id", "")), "generated_report": report}


def _template_command(
    stage3: Mapping[str, Any],
    projected_dir: Path,
    template_csv: Path,
) -> list[Any]:
    command: list[Any] = [
        sys.executable,
        "-m",
        "evigre.stage3.render_report",
        "--merged-json-dir",
        projected_dir,
        "--min-prob",
        str(stage3["min_probability"]),
        "--out-csv",
        template_csv,
    ]
    if stage3.get("add_preamble", False):
        command.append("--add-preamble")
    return command


def _stage_input(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if source.name.endswith(".nii.gz"):
        destination.symlink_to(source)
        return
    image = nib.load(str(source))
    nib.save(image, str(destination))


def _prepare_environment() -> Dict[str, str]:
    python_path = [str(UMAMBA2_ROOT), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(value for value in python_path if value),
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _existing_result_matches(
    manifest_path: Path,
    input_sha256: str,
    config_sha256: str,
    report_path: Path,
) -> bool:
    if not manifest_path.is_file() or not report_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("input", {}).get("sha256") == input_sha256
        and manifest.get("config", {}).get("sha256") == config_sha256
    )


def run_pipeline(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    input_volume = resolve_input_volume(args.input_path)
    output_dir = args.output_dir.expanduser().resolve()
    case_id = args.case_id or infer_case_id(input_volume)
    input_sha256 = hash_file(input_volume)
    internal_case_id = f"I{input_sha256[:15]}"
    config_sha256 = hash_json(config)

    preflight_result = preflight(
        config,
        config_path,
        input_volume,
        output_dir,
        args.device,
        args.skip_rewrite,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "pipeline_version": config["pipeline_version"],
                    "case_id": case_id,
                    "input": str(input_volume),
                    "input_sha256": input_sha256,
                    "preflight": preflight_result,
                },
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pipeline_manifest.json"
    report_path = output_dir / "report.txt"
    if _existing_result_matches(manifest_path, input_sha256, config_sha256, report_path) and not args.overwrite:
        print(f"Complete matching result already exists: {report_path}")
        return 0

    if manifest_path.exists() and not args.overwrite:
        previous = json.loads(manifest_path.read_text())
        same_input = previous.get("input", {}).get("sha256") == input_sha256
        same_config = previous.get("config", {}).get("sha256") == config_sha256
        if not (same_input and same_config and previous.get("status") in {"running", "failed"}):
            raise PipelineError(
                f"Output directory contains a different or completed run: {output_dir}. "
                "Use --overwrite to replace the pipeline-owned outputs."
            )

    work_dir = (args.work_dir or (output_dir / "_work")).expanduser().resolve()
    if args.overwrite:
        for name in KNOWN_OUTPUTS:
            path = output_dir / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        if work_dir.exists():
            shutil.rmtree(work_dir)

    input_root = work_dir / "input"
    staged_volume = input_root / internal_case_id / "cbct/volume.nii.gz"
    stage1_dir = work_dir / "stage1"
    stage1_csv = work_dir / "stage1_input.csv"
    stage2_root = work_dir / "stage2_preprocessed"
    stage2_preprocessed = stage2_root / config["stage2"]["dataset_name"]
    manifest_csv = stage2_preprocessed / "validation_case_merged.csv"
    merged_dir = work_dir / "stage2_merged"
    raw_merged_json_dir = merged_dir / "merged_json"
    filtered_dir = work_dir / "stage2_filtered/merged_json"
    filter_audit_dir = work_dir / "stage2_filtered/audit"
    projected_dir = work_dir / "stage2_projected/merged_json"
    projection_audit_dir = work_dir / "stage2_projected/audit"
    template_csv = work_dir / "reports_template.csv"
    final_csv = work_dir / "reports_final.csv"

    work_dir.mkdir(parents=True, exist_ok=True)
    _stage_input(input_volume, staged_volume)
    with stage1_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["identifier", "image_path"])
        writer.writeheader()
        writer.writerow({"identifier": internal_case_id, "image_path": str(staged_volume)})

    stage2 = config["stage2"]
    stage3 = config["stage3"]
    manifest: Dict[str, Any] = {
        "status": "running",
        "pipeline_version": config["pipeline_version"],
        "case_id": case_id,
        "internal_case_id": internal_case_id,
        "input": {"path": str(input_volume), "sha256": input_sha256, **preflight_result["input"]},
        "config": {"path": str(config_path), "sha256": config_sha256},
        "preflight": preflight_result,
        "models": {
            "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
            "stage2_global_folder": str(stage2["global_folder"]),
            "stage2_relation_folder": str(stage2["relation_folder"]),
            "stage2_tooth_folder": str(stage2["tooth_folder"]),
            "qwen_base": None if args.skip_rewrite else str(stage3["qwen_base"]),
            "qwen_adapter": None if args.skip_rewrite else str(stage3["qwen_adapter"]),
        },
        "parameters": {
            "device": args.device,
            "min_probability": float(stage3["min_probability"]),
            "add_preamble": bool(stage3.get("add_preamble", False)),
            "evidence_filter": stage3["evidence_filter"],
            "consistency_projection": stage3["consistency_projection"],
            "skip_rewrite": bool(args.skip_rewrite),
        },
        "work_dir": str(work_dir),
        "current_stage": "initialization",
        "stage_seconds": {},
    }
    _write_json(manifest_path, manifest)
    log_path = output_dir / "pipeline.log"
    _atomic_write_text(log_path, "")
    runner = CommandRunner(log_path)
    env = _prepare_environment()

    def run_stage(name: str, command: Iterable[str]) -> None:
        manifest["current_stage"] = name
        _write_json(manifest_path, manifest)
        manifest["stage_seconds"][name] = round(runner.run(name, command, env), 3)
        _write_json(manifest_path, manifest)

    try:
        run_stage(
            "stage1",
            [
                sys.executable, "-m", "evigre.inference.inference_umamba2_task1",
                "--csv-path", stage1_csv,
                "--checkpoint-path", config["stage1"]["checkpoint"],
                "--task1-inference-py", config["stage1"]["task1_inference_py"],
                "--output", stage1_dir,
                "--output-label-space", "compact",
                "--tile-step-size", str(config["stage1"]["tile_step_size"]),
                "--reference-spacing", str(config["stage1"]["reference_spacing"]),
                "--accumulation-device", str(config["stage1"].get("accumulation_device", "cpu")),
                "--save-visual-tokens",
                "--visual-token-levels", "2,3,4",
            ],
        )
        stage1_mask = stage1_dir / "predictions" / f"{internal_case_id}.nii.gz"
        visual_tokens = stage1_dir / "visual_structure_tokens" / f"{internal_case_id}.npz"
        run_stage(
            "stage2_preprocessing",
            [
                sys.executable, "-m", "evigre.inference.stage2_preprocessing",
                "--case-id", internal_case_id,
                "--image", staged_volume,
                "--stage1-mask", stage1_mask,
                "--visual-tokens", visual_tokens,
                "--out-dir", stage2_preprocessed,
            ],
        )

        if not manifest_csv.is_file():
            raise PipelineError(f"Single-case Stage-2 manifest was not produced: {manifest_csv}")

        merge_command = [
            sys.executable, "-m", "evigre.inference.merge_stage2_branch_val_results",
            "--csv-path", manifest_csv,
            "--global-folder", stage2["global_folder"],
            "--global-thresholds", stage2["global_thresholds"],
            "--relation-folder", stage2["relation_folder"],
            "--relation-thresholds", stage2["relation_thresholds"],
            "--tooth-folder", stage2["tooth_folder"],
            "--tooth-thresholds", stage2["tooth_thresholds"],
            "--tooth-threshold-view", stage2["tooth_threshold_view"],
            "--tooth-region-mask-config", stage2["tooth_region_mask_config"],
            "--checkpoint-name", stage2["checkpoint_name"],
            "--out-dir", merged_dir,
            "--device", args.device,
        ]
        run_stage("stage2_merge", merge_command)

        raw_json = raw_merged_json_dir / f"{internal_case_id}.json"
        if not raw_json.is_file():
            raise PipelineError(f"Merged evidence was not produced: {raw_json}")
        raw_case = json.loads(raw_json.read_text())
        raw_case["case_id"] = case_id
        raw_case["report_id"] = case_id
        raw_case["source_report_ids"] = []
        _write_json(output_dir / "evidence_raw.json", raw_case)

        run_stage(
            "evidence_filter",
            [
                sys.executable,
                "-m",
                "evigre.inference.evidence_filter",
                "--input-dir",
                raw_merged_json_dir,
                "--output-dir",
                filtered_dir,
                "--audit-dir",
                filter_audit_dir,
                "--filter-version",
                stage3["evidence_filter"],
            ],
        )
        filtered_json = filtered_dir / f"{internal_case_id}.json"
        if not filtered_json.is_file():
            raise PipelineError(f"Filtered evidence was not produced: {filtered_json}")

        run_stage(
            "consistency_projection",
            [
                sys.executable,
                "-m",
                "evigre.inference.consistency_projection",
                "--input-dir",
                filtered_dir,
                "--output-dir",
                projected_dir,
                "--audit-dir",
                projection_audit_dir,
                "--min-probability",
                str(stage3["min_probability"]),
            ],
        )
        filtered_case = json.loads(filtered_json.read_text())
        filtered_case["case_id"] = case_id
        filtered_case["report_id"] = case_id
        filtered_case["source_report_ids"] = []
        _write_json(output_dir / "evidence_filtered.json", filtered_case)
        projected_json = projected_dir / f"{internal_case_id}.json"
        projected_case = json.loads(projected_json.read_text())
        projected_case["case_id"] = case_id
        projected_case["report_id"] = case_id
        projected_case["source_report_ids"] = []
        _write_json(projected_json, projected_case)
        _write_json(output_dir / "evidence.json", projected_case)
        shutil.copy2(
            filter_audit_dir / f"{internal_case_id}.json",
            output_dir / "evidence_filter_audit.json",
        )
        shutil.copy2(projection_audit_dir / f"{internal_case_id}.json", output_dir / "consistency_audit.json")

        run_stage(
            "stage3_template",
            _template_command(stage3, projected_dir, template_csv),
        )
        template = _read_single_report(template_csv)["generated_report"]
        _atomic_write_text(output_dir / "template_report.txt", template + "\n")

        if args.skip_rewrite:
            final_report = template
        else:
            run_stage(
                "stage3_rewrite",
                [
                    sys.executable,
                    "-m",
                    "evigre.stage3.sft.generate",
                    "--model",
                    stage3["qwen_base"],
                    "--adapter",
                    stage3["qwen_adapter"],
                    "--in-csv",
                    template_csv,
                    "--out-csv",
                    final_csv,
                    "--id-col",
                    "case_id",
                    "--text-col",
                    "generated_report",
                    "--batch-size",
                    "1",
                    "--max-new-tokens",
                    str(stage3["max_new_tokens"]),
                ],
            )
            final_report = _read_single_report(final_csv)["generated_report"]

        _atomic_write_text(report_path, final_report + "\n")
        _write_json(
            output_dir / "report.json",
            {
                "case_id": case_id,
                "generated_report": final_report,
                "template_report": template,
                "pipeline_version": config["pipeline_version"],
                "template_add_preamble": bool(stage3.get("add_preamble", False)),
                "evidence_filter": stage3["evidence_filter"],
                "consistency_projection": stage3["consistency_projection"],
                "rewriter": None if args.skip_rewrite else "Qwen2.5-7B-Instruct LoRA",
            },
        )
        manifest["status"] = "complete"
        manifest["current_stage"] = None
        manifest["outputs"] = {
            "report_text": str(report_path),
            "report_json": str(output_dir / "report.json"),
            "template_report": str(output_dir / "template_report.txt"),
            "projected_evidence": str(output_dir / "evidence.json"),
            "filtered_evidence": str(output_dir / "evidence_filtered.json"),
            "raw_evidence": str(output_dir / "evidence_raw.json"),
            "evidence_filter_audit": str(output_dir / "evidence_filter_audit.json"),
            "consistency_audit": str(output_dir / "consistency_audit.json"),
            "log": str(log_path),
        }
        _write_json(manifest_path, manifest)
        if args.clean_work:
            shutil.rmtree(work_dir)
            manifest["work_dir_removed"] = True
            _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        _write_json(manifest_path, manifest)
        raise

    print(f"\nInference complete: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one preprocessed 3D CBCT volume through Stage 1, Stage 2, and Stage 3."
    )
    parser.add_argument("input_path", type=Path, help="NIfTI volume or a case directory containing it")
    parser.add_argument("output_dir", type=Path, help="Directory in which report.txt will be written")
    parser.add_argument("--case-id", help="External case identifier; inferred from the input path by default")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--skip-rewrite", action="store_true", help="Use the deterministic template as report.txt")
    parser.add_argument("--preflight-only", action="store_true", help="Validate input and artifacts without inference")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs owned by a prior pipeline run")
    parser.add_argument("--clean-work", action="store_true", help="Remove intermediate tensors after success")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_pipeline(args)
    except (PipelineError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
