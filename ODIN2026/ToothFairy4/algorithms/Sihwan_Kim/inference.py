"""Grand Challenge ToothFairy4 Task 1 R237 T4-compatible entrypoint."""

from __future__ import annotations

import glob
import json
import os
import re
import resource
from pathlib import Path
from typing import Any

from repgen.pipeline import execute_case


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
TMP_PATH = Path("/tmp/odin2026")
REPORT_OUTPUT = OUTPUT_PATH / "diagnostic-imaging-report.json"
INPUT_SOCKET_SLUG = "cbct-image"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input() -> tuple[Path, str]:
    inputs = load_json(INPUT_PATH / "inputs.json")
    if tuple(sorted(item["socket"]["slug"] for item in inputs)) != (INPUT_SOCKET_SLUG,):
        raise RuntimeError(f"Expected only the {INPUT_SOCKET_SLUG!r} input socket")
    matches = sorted(glob.glob(str(INPUT_PATH / "images" / "cbct" / "*.mha")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one .mha CBCT input, found {len(matches)}")
    path = Path(matches[0])
    case_id = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "case"
    return path, case_id


def run() -> int:
    input_path, case_id = resolve_input()
    report, audit = execute_case(input_path, case_id, TMP_PATH)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    audit["max_rss_kib_main_process"] = int(usage.ru_maxrss)
    audit["max_rss_kib_child_processes"] = int(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(
        json.dumps({"report": report}, indent=4) + "\n", encoding="utf-8"
    )
    if os.environ.get("ODIN_WRITE_DEBUG_AUDIT") == "1":
        (OUTPUT_PATH / "runtime-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"case_id": case_id, "output": str(REPORT_OUTPUT), **audit["timings"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
