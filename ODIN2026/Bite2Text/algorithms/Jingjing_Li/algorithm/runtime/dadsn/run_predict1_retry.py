#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare and optionally run one selective predict1.py retry.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PATH_ARGUMENTS = """
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("work/input"),
        help="Input mesh directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("work/output_two_pass_geodesic_recall_guard"),
        help="Segmentation output directory",
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("work/debug_two_pass_geodesic_recall_guard"),
        help="Debug output directory",
    )
""".rstrip()


DIRECTORY_BLOCK = """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir)
""".rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-script", type=Path, default=Path("predict1.py"))
    parser.add_argument(
        "--qc-v2-summary",
        type=Path,
        default=Path(
            "work/segmentation_qc_v2/segmentation_qc_v2_summary.csv"
        ),
    )
    parser.add_argument(
        "--source-input-dir",
        type=Path,
        default=Path("work/input"),
    )
    parser.add_argument(
        "--retry-input-dir",
        type=Path,
        default=Path("work/input_retry1"),
    )
    parser.add_argument(
        "--retry-output-dir",
        type=Path,
        default=Path("work/output2_retry1"),
    )
    parser.add_argument(
        "--retry-debug-dir",
        type=Path,
        default=Path("work/debug_retry1"),
    )
    parser.add_argument(
        "--patched-script",
        type=Path,
        default=Path("predict1_retry_configurable.py"),
    )
    parser.add_argument(
        "--file-mode",
        choices=("symlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--align-seed", type=int, default=200003)
    parser.add_argument("--pass1-seed", type=int, default=200011)
    parser.add_argument("--pass2-seed", type=int, default=200029)
    parser.add_argument("--retry-pass1-seed", type=int, default=200047)
    parser.add_argument("--align-trials", type=int, default=4)
    parser.add_argument("--keep-debug", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clean-retry-input", action="store_true")
    return parser


def patch_predict_script(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")

    if "--input_dir" not in text:
        marker = "    return parser.parse_args()"

        if text.count(marker) != 1:
            raise RuntimeError(
                "Could not uniquely locate return parser.parse_args()"
            )

        text = text.replace(
            marker,
            PATH_ARGUMENTS + "\n\n" + marker,
            1,
        )

    if "input_dir = Path(args.input_dir)" not in text:
        pattern = re.compile(
            r"""
            [ \t]*experiment_dir[ \t]*=[ \t]*Path\(["']work["']\)[ \t]*\n
            [ \t]*input_dir[ \t]*=[ \t]*experiment_dir[ \t]*/[ \t]*["'][^"']+["'][ \t]*\n
            [ \t]*output_dir[ \t]*=[ \t]*\([ \t]*\n
            [ \t]*experiment_dir[ \t]*\n
            [ \t]*/[ \t]*["'][^"']+["'][ \t]*\n
            [ \t]*\)[ \t]*\n
            [ \t]*debug_dir[ \t]*=[ \t]*\([ \t]*\n
            [ \t]*experiment_dir[ \t]*\n
            [ \t]*/[ \t]*["'][^"']+["'][ \t]*\n
            [ \t]*\)
            """,
            flags=re.VERBOSE,
        )
        matches = list(pattern.finditer(text))

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely patch predict1 directory block; "
                f"found {len(matches)} matches"
            )

        text = pattern.sub(
            "\n" + DIRECTORY_BLOCK,
            text,
            count=1,
        )

    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8")


def find_input_mesh(source_dir: Path, case: str) -> Path:
    for suffix in (".stl", ".ply", ".obj"):
        candidate = source_dir / f"{case}{suffix}"
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in source_dir.glob(f"{case}*")
        if path.suffix.lower() in {".stl", ".ply", ".obj"}
    )

    if not candidates:
        raise FileNotFoundError(f"No input mesh for {case}")

    return candidates[0]


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path.cwd().resolve()
    qc = pd.read_csv(
        args.qc_v2_summary.resolve(),
        dtype={"case": str},
    )

    if "retry_recommended_v2" not in qc.columns:
        raise ValueError("QC v2 summary lacks retry_recommended_v2")

    retry_rows = qc[
        pd.to_numeric(
            qc["retry_recommended_v2"],
            errors="coerce",
        ).fillna(0).astype(int) == 1
    ]

    source_input = args.source_input_dir.resolve()
    retry_input = args.retry_input_dir.resolve()
    retry_output = args.retry_output_dir.resolve()
    retry_debug = args.retry_debug_dir.resolve()
    patched_script = args.patched_script.resolve()

    if args.clean_retry_input and retry_input.exists():
        shutil.rmtree(retry_input)

    retry_input.mkdir(parents=True, exist_ok=True)
    retry_output.mkdir(parents=True, exist_ok=True)
    retry_debug.mkdir(parents=True, exist_ok=True)

    prepared = []
    missing = []

    for case in retry_rows["case"].astype(str):
        try:
            source_mesh = find_input_mesh(source_input, case)
        except FileNotFoundError:
            missing.append(case)
            continue

        target_mesh = retry_input / source_mesh.name

        if target_mesh.exists() or target_mesh.is_symlink():
            target_mesh.unlink()

        if args.file_mode == "copy":
            shutil.copy2(source_mesh, target_mesh)
        else:
            target_mesh.symlink_to(source_mesh)

        prepared.append(case)

    patch_predict_script(
        args.predict_script.resolve(),
        patched_script,
    )

    command = [
        sys.executable,
        str(patched_script),
        "--input_dir",
        str(retry_input),
        "--output_dir",
        str(retry_output),
        "--debug_dir",
        str(retry_debug),
        "--align_seed",
        str(args.align_seed),
        "--pass1_seed",
        str(args.pass1_seed),
        "--pass2_seed",
        str(args.pass2_seed),
        "--retry_pass1_seed",
        str(args.retry_pass1_seed),
        "--align_trials",
        str(args.align_trials),
    ]

    if not args.keep_debug:
        command.append("--no_debug")

    (retry_output / "retry_command.txt").write_text(
        subprocess.list2cmdline(command) + "\n",
        encoding="utf-8",
    )

    if missing:
        (retry_output / "missing_retry_inputs.txt").write_text(
            "\n".join(missing) + "\n",
            encoding="utf-8",
        )

    print("=" * 72)
    print("Selective predict1 retry")
    print("=" * 72)
    print(f"QC retry cases: {len(retry_rows)}")
    print(f"Prepared inputs: {len(prepared)}")
    print(f"Missing inputs: {len(missing)}")
    print(f"Patched script: {patched_script}")
    print(f"Retry output: {retry_output}")
    print()
    print(subprocess.list2cmdline(command))

    if args.execute:
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
        )
    else:
        print()
        print("Dry run only. Add --execute to start segmentation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
