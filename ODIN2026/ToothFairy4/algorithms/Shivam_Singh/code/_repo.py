"""Where the repo root is, and where the code groups are.

Every module under code/ is flat-imported by name -- `import postprocess_pred`,
not `from pipeline.postprocess import postprocess_pred`. That is on purpose: it
keeps `python3 code/<group>/<module>.py` working as a plain script invocation,
which is how every job script and every doc example calls this code, and it
means the grouping can be rearranged without touching a single import.

The cost is that sys.path has to be told where the groups are, which is what
add_code_paths() does. Call it once, at the top of any module that imports a
sibling.

REPO_ROOT is resolved from this file's own location rather than counted in
`.parent.parent` hops, and module_path() searches the groups rather than
assuming a flat directory. Both replace idioms that failed SILENTLY when the
tree was reorganised: a default path one directory too shallow, and a
subprocess path to a script that had moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent

#: Directories under code/ that hold flat-importable modules.
GROUPS = (
    "pipeline/segmentation",
    "pipeline/preprocess",
    "pipeline/vqa",
    "pipeline/postprocess",
    "ground_truth",
    # The audit half: mechanical checks on the stage-1 extraction.
    "ground_truth/audit_gt",
    "eval",
    "train",
    # The visual-evidence pass: a research-only side arm, kept out of the
    # released subset. Its own group so make_release.py's closure does not
    # reach it and code/train/ reads as the arm-6 path alone.
    "train/visual_evidence",
    "arms",
    "studies",
    "data",
)

DEFAULT_SCHEMA = REPO_ROOT / "schema" / "schema.json"


def add_code_paths() -> None:
    """Put every code group on sys.path, nearest-first, without duplicates."""
    for group in GROUPS:
        d = CODE_ROOT / group
        if d.is_dir():
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)


def module_path(filename: str) -> Path:
    """Absolute path of a module, by filename, searched across the groups.

    For subprocess invocations -- `[sys.executable, str(module_path(
    "create_panoramic.py"))]` -- where an import will not do. Raises rather
    than returning a path that does not exist, because the failure it replaces
    was a subprocess that could not find its script at run time.
    """
    for group in GROUPS:
        p = CODE_ROOT / group / filename
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"{filename} is in none of the code groups under {CODE_ROOT}")


def check() -> None:
    """Fail loudly if the tree is not shaped the way REPO_ROOT assumes."""
    if not DEFAULT_SCHEMA.is_file():
        raise RuntimeError(
            f"repo root resolved to {REPO_ROOT}, which has no "
            f"schema/schema.json -- code/_repo.py has been moved out of code/")
