#!/usr/bin/env python3
"""Dispatch EviDent-CBCT inference for one NIfTI file or a directory of cases."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from evigre.inference.run_single_case import (
    PipelineError,
    build_parser as build_single_parser,
    infer_case_id,
    run_pipeline,
)


def discover_volumes(input_path: Path) -> Tuple[str, List[Path]]:
    source = input_path.expanduser().resolve()
    if source.is_file():
        if not (source.name.endswith(".nii.gz") or source.name.endswith(".nii")):
            raise PipelineError(f"Input file must be .nii.gz or .nii: {source}")
        return "single", [source]
    if not source.is_dir():
        raise FileNotFoundError(source)

    volumes = sorted(
        {
            path.resolve()
            for pattern in ("*.nii.gz", "*.nii")
            for path in source.rglob(pattern)
            if path.is_file()
        },
        key=str,
    )
    if not volumes:
        raise PipelineError(f"No .nii.gz or .nii volumes found recursively under: {source}")
    return "directory", volumes


def output_name(case_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id)).strip("._")
    if not name:
        raise PipelineError(f"Case ID cannot be converted to a safe output name: {case_id!r}")
    return name


def build_case_table(volumes: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    seen_case_ids: Dict[str, Path] = {}
    seen_output_names: Dict[str, str] = {}
    for volume in volumes:
        case_id = infer_case_id(volume)
        name = output_name(case_id)
        if case_id in seen_case_ids:
            raise PipelineError(
                f"Duplicate inferred case ID {case_id!r}: {seen_case_ids[case_id]} and {volume}. "
                "Rename the files or enclosing case directories."
            )
        if name in seen_output_names:
            raise PipelineError(
                f"Case IDs {seen_output_names[name]!r} and {case_id!r} map to the same output name {name!r}."
            )
        seen_case_ids[case_id] = volume
        seen_output_names[name] = case_id
        rows.append({"case_id": case_id, "output_name": name, "input_path": volume})
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_reports_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["case_id", "input_path", "output_dir", "report_path", "status", "generated_report", "error"]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def run_directory(args, volumes: List[Path]) -> int:
    input_root = args.input_path.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise PipelineError("For directory inference, OUTPUT_DIR must be outside INPUT_DIR.")
    if args.case_id:
        raise PipelineError("--case-id is only valid when INPUT is a single NIfTI file.")
    if args.work_dir:
        raise PipelineError("--work-dir is only valid for single-file inference.")

    cases = build_case_table(volumes)
    if not args.preflight_only:
        output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch_manifest.json"
    reports_csv = output_root / "reports.csv"
    manifest: Dict[str, Any] = {
        "status": "preflight" if args.preflight_only else "running",
        "input_mode": "directory",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_cases": len(cases),
        "cases": [],
    }
    if not args.preflight_only:
        _write_json(manifest_path, manifest)

    result_rows: List[Dict[str, Any]] = []
    failures = 0
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        volume = Path(case["input_path"])
        case_output = output_root / str(case["output_name"])
        print(f"\n=== [{index}/{len(cases)}] {case_id}: {volume} ===", flush=True)
        case_args = copy.copy(args)
        case_args.input_path = volume
        case_args.output_dir = case_output
        case_args.case_id = case_id
        case_args.work_dir = None
        try:
            run_pipeline(case_args)
            if args.preflight_only:
                row = {
                    "case_id": case_id,
                    "input_path": str(volume),
                    "output_dir": str(case_output),
                    "report_path": "",
                    "status": "ready",
                    "generated_report": "",
                    "error": "",
                }
            else:
                report_payload = json.loads((case_output / "report.json").read_text())
                row = {
                    "case_id": case_id,
                    "input_path": str(volume),
                    "output_dir": str(case_output),
                    "report_path": str(case_output / "report.txt"),
                    "status": "complete",
                    "generated_report": str(report_payload["generated_report"]),
                    "error": "",
                }
        except Exception as error:
            failures += 1
            row = {
                "case_id": case_id,
                "input_path": str(volume),
                "output_dir": str(case_output),
                "report_path": "",
                "status": "failed",
                "generated_report": "",
                "error": str(error),
            }
            if not args.continue_on_error:
                result_rows.append(row)
                if not args.preflight_only:
                    manifest["status"] = "failed"
                    manifest["cases"] = result_rows
                    _write_reports_csv(reports_csv, result_rows)
                    _write_json(manifest_path, manifest)
                raise
        result_rows.append(row)
        if not args.preflight_only:
            manifest["cases"] = result_rows
            _write_reports_csv(reports_csv, result_rows)
            _write_json(manifest_path, manifest)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ready" if failures == 0 else "failed",
                    "input_mode": "directory",
                    "num_cases": len(cases),
                    "failures": failures,
                    "cases": result_rows,
                },
                indent=2,
            )
        )
    else:
        manifest["status"] = "complete" if failures == 0 else "partial"
        manifest["num_complete"] = len(cases) - failures
        manifest["num_failed"] = failures
        manifest["cases"] = result_rows
        _write_reports_csv(reports_csv, result_rows)
        _write_json(manifest_path, manifest)
        print(f"\nBatch inference {manifest['status']}: {reports_csv}")
    return 0 if failures == 0 else 1


def build_parser():
    parser = build_single_parser()
    parser.description = (
        "Run EviDent-CBCT for one NIfTI volume or recursively for every NIfTI volume in a directory."
    )
    for action in parser._actions:
        if action.dest == "input_path":
            action.help = "One NIfTI volume, or a directory recursively containing NIfTI volumes"
        elif action.dest == "output_dir":
            action.help = (
                "Single input: report output directory; directory input: batch root with one subdirectory per case"
            )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="For directory input, record a failed case and continue with the remaining cases",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode, volumes = discover_volumes(args.input_path)
        if mode == "single":
            return run_pipeline(args)
        return run_directory(args, volumes)
    except (PipelineError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
