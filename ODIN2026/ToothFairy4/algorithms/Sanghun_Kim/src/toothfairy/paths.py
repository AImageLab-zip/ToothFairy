"""The one place that answers "where is the repository root".

Every module that has to reach outside the package — feature caches, splits, model
directories, the reconstructed per-structure targets — resolves its paths against
``REPO``.  Relative paths in the configs are read as relative to it.

The layout is ``<REPO>/src/toothfairy/paths.py``, so the root is two levels up from the
package.  Keeping this in a single module means moving the package needs one edit, not
one per file.

⚠️ ``REPO`` is the checkout, not the installation: the data, the feature caches and the
model directories are read relative to it, so run from a checkout even when the package is
installed elsewhere.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1]                 # <REPO>/src


def resolve(p: str | Path) -> Path:
    """Absolute paths are used as they are; relative ones hang off the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p
