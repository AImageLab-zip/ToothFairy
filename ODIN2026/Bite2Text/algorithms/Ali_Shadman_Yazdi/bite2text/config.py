"""Paths, label conventions and defaults.

Weights are read from ``$BITE2TEXT_WEIGHTS`` (default ``<repo>/weights``). The
variable is read at call time, so it can be set after this module is imported.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_WEIGHTS_DIR = PACKAGE_DIR.parent / "weights"

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

CLIK_FILES = {
    "incisor":  "[Crown]incisor-e837.pt",
    "cuspid":   "[Crown]cuspid-e879.pt",
    "premolar": "[Crown]premolar-e443.pt",
    "molar":    "[Crown]molar-e178.pt",
}


def weights_dir() -> Path:
    return Path(os.environ.get("BITE2TEXT_WEIGHTS", DEFAULT_WEIGHTS_DIR))


def seg_checkpoint(jaw: str) -> Path:
    return weights_dir() / "tsgcnet" / f"{jaw_key(jaw)}.pth"


def clik_checkpoint(kind: str) -> Path:
    return weights_dir() / "clik" / CLIK_FILES[kind]


def photo_cnn_dir() -> Path:
    return weights_dir() / "photo_cnn"


def required_weights() -> List[Path]:
    """Files the scan branch cannot run without."""
    return ([seg_checkpoint(j) for j in ("upper", "lower")]
            + [clik_checkpoint(k) for k in CLIK_FILES])


def missing_weights() -> List[Path]:
    return [p for p in required_weights() if not p.is_file()]


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jaw_key(jaw: str) -> str:
    key = str(jaw).strip().lower()
    if key in ("upper", "maxilla", "maxillary", "u"):
        return "upper"
    if key in ("lower", "mandible", "mandibular", "l"):
        return "lower"
    raise ValueError(f"jaw must be 'upper' or 'lower', got {jaw!r}")


# ---------------------------------------------------------------------------
# Segmentation (TSGCNet): 17 classes, 0 = gingiva, 1..16 = teeth in FDI order.
# Fixed by how the training data was built; must not be reordered.
# ---------------------------------------------------------------------------

NUM_CLASSES = 17

LABEL_TO_FDI = {
    "upper": (0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28),
    "lower": (0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48),
}

TARGET_FACES = 16_000      # face count TSGCNet was trained on
MIN_COMPONENT_AREA = 50.0  # mm^2; smaller label islands are absorbed
UPSAMPLE_NEIGHBORS = 8     # kNN used to lift coarse labels back to full resolution

# ---------------------------------------------------------------------------
# Landmarks (CLIK works in Universal numbering 1..32)
# ---------------------------------------------------------------------------

FPS_SAMPLES = 2_048        # points per tooth the CLIK models were trained on

FDI_TO_UNIVERSAL = {
    18: 1,  17: 2,  16: 3,  15: 4,  14: 5,  13: 6,  12: 7,  11: 8,
    21: 9,  22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
    38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,
}

LANDMARK_NAMES = {
    "0":  "Centroid",              "3":  "Labial low",
    "4":  "Lingual low",           "5":  "Cusp tip",
    "6":  "Buccal cusp",           "7":  "Lingual cusp",
    "8":  "Mesiobuccal cusp",      "9":  "Distobuccal cusp",
    "10": "Mesiolingual cusp",     "11": "Distolingual cusp",
    "13": "Incisal center",        "15": "Mesioincisal",
    "16": "Distoincisal",          "19": "Mesiolinguoversion",
    "20": "Distolingoversion",     "21": "Mesial marginal ridge",
    "22": "Distal marginal ridge", "23": "Groove center",
    "25": "Buccal groove peak",    "30": "Fovea center",
    "34": "Mesial contact",        "35": "Distal contact",
}

# CLIK model -> landmark ids it predicts (output channels, in order).
CLIK_LANDMARK_IDS = {
    "incisor":  ["3", "4", "13", "15", "16", "34", "35"],
    "cuspid":   ["3", "4", "5", "19", "20", "34", "35"],
    "premolar": ["4", "6", "7", "21", "22", "23", "34", "35"],
    "molar":    ["8", "9", "10", "11", "21", "22", "25", "30", "34", "35"],
}

# Universal tooth id -> CLIK model. Third molars (1, 16, 17, 32) have none.
UNIVERSAL_TO_CLIK = {
    **{t: "incisor" for t in (7, 8, 9, 10, 23, 24, 25, 26)},
    **{t: "cuspid" for t in (6, 11, 22, 27)},
    **{t: "premolar" for t in (4, 5, 12, 13, 20, 21, 28, 29)},
    **{t: "molar" for t in (2, 3, 14, 15, 18, 19, 30, 31)},
}
