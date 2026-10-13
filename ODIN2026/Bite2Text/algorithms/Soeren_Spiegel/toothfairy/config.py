"""Central config for the ToothFairy4 Qwen2.5-VL representation pipeline.

Paths, splits, render params, Qwen params, LoRA/training params.

"""
from __future__ import annotations

import os
from pathlib import Path

# ── Repo layout ────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[1]
SPLITS_DIR  = REPO / "splits"
PLOTS_DIR   = REPO / "plots"
RUNS_DIR    = REPO / "experiments" / "finetune_qwen_v0" / "runs"
# Streaming zip access + report/metadata helpers live with the exploration code.
EXPLORE_DIR = REPO / "eval" / "explore"
REPORTS_CSV = EXPLORE_DIR / "out" / "reports.csv"   # metadata index (junk flags, langs)

# ── Data ───────────────────────────────────────────────────────────────────────
# v03 (2026-08-04 re-release): fixed P/F/S orientation metadata, L/R-corrected A
# volumes, duplicates removed, FDI normalized to non-dotted. 622 cases, all with
# volumes.
# The dataset is a single ~48 GB zip, read without extraction (eval/explore/tf_zip.py):
#   <CASE>/cbct/volume.nii.gz
#   <CASE>/reports_en/<rid>.txt   (1..N independent EN annotations per case)
ZIP_PATH = os.environ.get(
    "TF_ZIP_PATH", "/path/to/toothfairy4_v03.zip"
)

# local-cache I/O: raw volumes may be byte-copied once to a local disk and every read goes
# local-first, falling back to the zip. ODIN_CACHE_DIR is
# accepted as a legacy alias for TF_CACHE_DIR.
CACHE_DIR = Path(os.environ.get("TF_CACHE_DIR", os.environ.get("ODIN_CACHE_DIR", "/path/to/cache")))
VOLUME_CACHE_DIR = CACHE_DIR / "volumes_v03"        # <case_id>.nii.gz  (v03; NOT "volumes")
RENDER_CACHE_DIR = CACHE_DIR / "render"             # render cache, keyed by config hash

# ── Reports / splits ──────────────────────────────────────────────────────────
REPORT_LANG = "en"                                  # challenge Task 1 is English reports
# One training record per (case, EN report): multi-report cases are independent annotations,
# so each is its own supervision target, not a duplicate.
MULTI_REPORT_AS_SEPARATE_TARGETS = True
MIN_REPORT_CHARS = 40   # drop junk/near-empty targets (v03 has no junk reports left)

# v03 splits: per-family stratified 80/10/10 (eval/explore/freeze_splits_v03.py, seed 2026).
#   train = 498 (A=75, F=51, P=330, S=42)
#   val   =  62 (A=10, F=6,  P=41,  S=5)
#   test  =  62 (A=10, F=6,  P=41,  S=5)
# The pre-v03 val_id / val_ood (F-family holdout) scheme is retired; its files are
# archived in splits/legacy_v02/ and are NOT a partition of the v03 case list. For an
# OOD-style read, score the F subset of `test`.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# ── Volume preprocessing ───────────────────────────────────────────────────────
# Orientation is NOT uniform across families (v03: A = LPS, F/P/S = LAS); everything is
# reoriented to LAS so x=left, y=anterior, z=superior and FDI quadrants never mirror.
CANONICAL_AXCODES = ("L", "A", "S")
RESAMPLE_MM = 0.3       # isotropic target grid (P/F/S are already 0.3 mm; family A varies)
WINDOW_PCT  = (0.5, 99.5)   # per-volume percentile window — CBCT has no HU calibration
FOV_LIMITED_MM = 100.0  # below this Z-extent, tell the model condyles/upper maxilla may be cropped

# Pathology-derived RGB windows on the calibrated HU scale (baseline_hu_v1 Task 2).
# Single source of truth lives in toothfairy.hu; re-exported here for convenience.
# R = soft/lucent-lesion, G = dentin/periodontal/IAN, B = cortical/enamel/restoration.
from .hu import RGB_WINDOWS  # noqa: E402,F401  (re-export; do not redefine)

# ── Rendered representation (the two views) ────────────────────────────────────
FRAME_TILE       = 448  # square tile for video frames / MPR stills. A multiple of 28, so
                        # Qwen's ViT (patch 14, 2x2 merge) does not re-resize -> 256 tokens.
N_CROSS_SECTIONS = 40   # cap on buccolingual cross-section video frames (5 mm arc step)
MAX_VIDEO_FRAMES_INFER = 96   # defensive cap at inference (no-op in-distribution)
VIDEO_FPS = 1.0         # nominal fps for Qwen's temporal position ids (frames are spatial)

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# Processor pixel budget. The panoramic is wide (up to PANO_MAX_WIDTH=1568 x 448 = 703k px),
# so max_pixels must sit above that or the processor silently downsamples it.
MAX_PIXELS_SIDE = 1280      # budget is MAX_PIXELS_SIDE**2 = 1.64M px > 703k
MIN_PIXELS_SIDE = 256

# ── LoRA / training defaults ───────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
SAVE_STEPS       = 250
RANDOM_SEED      = 42

# ── Inference / decoding ───────────────────────────────────────────────────────
MAX_NEW_TOKENS = 512
DO_SAMPLE      = False

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_ENTITY  = None
WANDB_PROJECT = "rgbite-toothfairy4"
