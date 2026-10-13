"""Grand Challenge entrypoint for the ODIN 2026 Bite2Text track.

    /input/inputs.json                          socket declarations
    /input/files/ios-lower/*                    3d-lower-teeth-scan
    /input/files/ios-upper/*                    3d-upper-teeth-scan
    /input/images/intraoral-photo/*             2d-intraoral-photographs
    /opt/ml/model/                              contents of weights/ (the model tarball)
 -> /output/diagnostic-imaging-report.json      {"report": "<plain text>"}

Both scans are required. If either cannot be resolved the job exits non-zero
and writes nothing -- a failed job can be resubmitted, a plausible wrong report
cannot be taken back. The photographs are optional: without them the photo
section comes from the cohort prior, and the log says so.

For a local run without Docker, point the three paths elsewhere:

    BITE2TEXT_INPUT=/tmp/case BITE2TEXT_OUTPUT=/tmp/out BITE2TEXT_WEIGHTS=weights python inference.py
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("BITE2TEXT_WEIGHTS", "/opt/ml/model")

import glob
import json
import logging
import sys
from pathlib import Path

INPUT_PATH = Path(os.environ.get("BITE2TEXT_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("BITE2TEXT_OUTPUT", "/output"))

LOWER_SLUGS = ("3d-lower-teeth-scan", "ios-lower-scan")
UPPER_SLUGS = ("3d-upper-teeth-scan", "ios-upper-scan")
PHOTO_SLUGS = ("2d-intraoral-photographs", "intraoral-photo")
MESH_PATTERNS = ("*.stl", "*.obj", "*.ply", "*.STL", "*.OBJ", "*.PLY")
ARCH_HINTS = {"ios-lower": ("lower", "mandib"), "ios-upper": ("upper", "maxill")}

log = logging.getLogger("bite2text")


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def input_tree(limit: int = 40) -> list:
    try:
        files = sorted(str(p.relative_to(INPUT_PATH)) for p in INPUT_PATH.rglob("*") if p.is_file())
    except OSError:
        return ["<unreadable>"]
    return files[:limit] + ([f"... +{len(files) - limit} more"] if len(files) > limit else [])


def declared_paths(slugs) -> list:
    """Directories declared for these socket slugs in inputs.json."""
    path = INPUT_PATH / "inputs.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except ValueError:
        log.warning("could not parse %s", path)
        entries = []
    found = []
    for entry in entries if isinstance(entries, list) else []:
        socket = entry.get("socket") or {}
        if str(socket.get("slug") or "").strip() in slugs:
            rel = str(socket.get("relative_path") or "").strip()
            if rel:
                found.append(INPUT_PATH / rel)
    return found


def find_one(slugs, default_dir: str, exclude=()):
    """First scan for one arch, most specific location first.

    The bare /input glob must stay last: Grand Challenge can place both
    ``3d-*-teeth-scan.obj`` files side by side at the root, and a bare glob
    would return the same (lower) file for both arches. ``exclude`` holds the
    file already claimed by the other arch.
    """
    excluded = {Path(p).resolve() for p in exclude if p}

    roots = declared_paths(slugs) + [INPUT_PATH / "files" / default_dir, INPUT_PATH / default_dir]
    roots += [INPUT_PATH / f"{slug}{ext}" for slug in slugs for ext in (".obj", ".stl", ".ply")]
    roots.append(INPUT_PATH)

    for root in roots:
        if root.is_file() and root.resolve() not in excluded:
            return root
        if not root.is_dir():
            continue
        for pattern in MESH_PATTERNS:
            hits = [Path(h) for h in sorted(glob.glob(str(root / pattern)))
                    if Path(h).resolve() not in excluded]
            if hits:
                return hits[0]

    # Unfamiliar layout: search recursively for an arch word in the path.
    for pattern in MESH_PATTERNS:
        for p in sorted(INPUT_PATH.rglob(pattern)):
            if p.is_file() and p.resolve() not in excluded and \
                    any(h in str(p).lower() for h in ARCH_HINTS[default_dir]):
                return p
    return None


def find_photos(slugs, suffixes) -> list:
    """Every photograph, most specific location first, then a sweep of /input."""
    def photos_in(root: Path, recursive: bool) -> list:
        try:
            entries = root.rglob("*") if recursive else root.iterdir()
            return [p for p in entries if p.is_file() and p.suffix.lower() in suffixes]
        except OSError:
            return []

    roots = declared_paths(slugs) + [INPUT_PATH / "images" / s for s in slugs]
    roots += [INPUT_PATH / s for s in slugs] + [INPUT_PATH / "images"]
    for root in roots:
        if root.is_file() and root.suffix.lower() in suffixes:
            return [root]
        if root.is_dir():
            hits = photos_in(root, recursive=False) or photos_in(root, recursive=True)
            if hits:
                return sorted(hits, key=lambda p: p.name)
    return sorted(photos_in(INPUT_PATH, recursive=True), key=lambda p: p.name)


def describe(path) -> str:
    return "MISSING" if path is None else f"{path.name} ({path.stat().st_size / 1e6:.1f} MB)"


# ---------------------------------------------------------------------------

def run() -> int:
    logging.basicConfig(level=logging.INFO, format="[bite2text] %(message)s")

    from bite2text import ReportGenerator, config
    from bite2text.photo_io import PHOTO_SUFFIXES

    missing = config.missing_weights()
    log.info("weights at %s: %s", config.weights_dir(),
             "OK" if not missing else f"MISSING {[str(p) for p in missing]}")
    photo_ckpt = config.photo_cnn_dir() / "best.pt"
    log.info("photo checkpoint: %s", "OK" if photo_ckpt.is_file() else "MISSING (cohort prior only)")

    lower = find_one(LOWER_SLUGS, "ios-lower")
    upper = find_one(UPPER_SLUGS, "ios-upper", exclude=(lower,))
    photos = find_photos(PHOTO_SLUGS, PHOTO_SUFFIXES)
    log.info("lower=%s upper=%s photos=%d", describe(lower), describe(upper), len(photos))

    if lower is None or upper is None:
        raise RuntimeError(f"could not resolve both IOS scans (lower={describe(lower)}, "
                           f"upper={describe(upper)}). /input contains: {input_tree()}")

    report = ReportGenerator()(upper=upper, lower=lower, photos=photos)
    log.info("photo section source: %s", report.photo_source)
    if not report.text.strip():
        raise RuntimeError("generated report is empty")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PATH / "diagnostic-imaging-report.json"
    out.write_text(json.dumps({"report": report.text}, indent=4), encoding="utf-8")
    log.info("wrote %s (%d chars)", out, len(report.text))
    return 0


if __name__ == "__main__":
    sys.exit(run())
