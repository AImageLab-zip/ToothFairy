"""
Central configuration + the knobs.

Everything here is env-overridable so you can flip behaviour at `docker run`
time without rebuilding the image (e.g. `-e HANDOFF_ORIENTATION=RAS`).
"""

import os
from pathlib import Path


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


# ── Orientation ───────────────────────────────────────────────────────────
# The segmentation model was trained on RPI *array* orientation and ALWAYS
# receives RPI. This is fixed by the trained weights and is NOT a knob.
MODEL_ORIENTATION = "RPI"

# THE KNOB (frame in which the aligned (CBCT, mask) pair is handed to her half).
# Default INPUT = raw input space, which per the ToothFairy4 correction is what
# aligns with the reports (do NOT apply the affine). Override only if her cropper
# needs a specific frame:
#   "INPUT" -> hand over raw input space (default)
#   "RPI" / "RAS" / "LPS" -> reorient the aligned pair to that frame
HANDOFF_ORIENTATION = os.getenv("HANDOFF_ORIENTATION", "INPUT").strip().upper()

# Post-segmentation anatomical sanity check (maxilla physically above mandible).
# Runs in world/physical space so it is frame-agnostic. Only *warns* by
# default; set ORIENTATION_CHECK_FATAL=1 to raise instead.
VERIFY_ORIENTATION = _flag("VERIFY_ORIENTATION", "1")
ORIENTATION_CHECK_FATAL = _flag("ORIENTATION_CHECK_FATAL", "0")

# Segmentation label ids for upper (maxillary) vs lower (mandibular) structures.
# From Dataset703 dataset.json (sequential anatomical labels, NOT FDI).
# Jawbones (2 upper, 1 lower) are included so the check still fires on
# edentulous arches where the tooth labels would be empty.
UPPER_TOOTH_LABELS: tuple[int, ...] = (2,) + tuple(range(11, 27))   # Upper Jawbone + maxillary teeth 11-26
LOWER_TOOTH_LABELS: tuple[int, ...] = (1,) + tuple(range(27, 43))   # Lower Jawbone + mandibular teeth 27-42

# Remap model (sequential 0-46) labels into extract_facts's FDI scheme BEFORE
# extraction. DEFAULT OFF, deliberately: the facts her VLM was trained on were
# produced WITHOUT this remap (extract_facts read sequential masks as if FDI,
# mislabeling 3 of 4 quadrants). Reproducing that exactly keeps train/inference
# consistent. Turn ON only if her VLM is retrained on corrected FDI facts.
REMAP_MODEL_TO_FDI = _flag("REMAP_MODEL_TO_FDI", "0")

# ── Paths (model.tar.gz is mounted read-only at /opt/ml/model) ────────────
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/opt/ml/model"))
# Folder that contains fold_all/checkpoint_final.pth, plans.json, dataset.json
NNUNET_MODEL_FOLDER = MODEL_ROOT / os.getenv("NNUNET_SUBDIR", "nnunet_seg")
NNUNET_CHECKPOINT = os.getenv("NNUNET_CHECKPOINT", "checkpoint_final.pth")
NNUNET_FOLDS = ("all",)

# ── Runtime ───────────────────────────────────────────────────────────────
# Run nnU-Net's final resample on CPU to avoid the 40GB-OOM you already hit.
RESAMPLE_ON_CPU = _flag("RESAMPLE_ON_CPU", "1")
# Test-time mirroring augmentation. Off = faster, slightly lower Dice.
USE_MIRRORING = _flag("USE_TTA", "1")


# ── VLM report arm (CBCT_VQA pipeline) ────────────────────────────────────
# Her code + schema are copied into the image; weights ship in model.tar.gz.
# Her stack needs torch 2.11+cu130 / vllm 0.22, which is ABI-incompatible with
# the torch 2.5.1+cu124 the mamba kernels were compiled against. The two live in
# SEPARATE venvs in the same image; every VLM script runs under this
# interpreter, never the segmentation one.
VLM_PYTHON = os.getenv("VLM_PYTHON", "/opt/vlmenv/bin/python")

VLM_CODE_DIR = Path(os.getenv("VLM_CODE_DIR", "/opt/app/vqa/code"))
VLM_SCHEMA   = Path(os.getenv("VLM_SCHEMA", "/opt/app/vqa/schema/schema.json"))

# Qwen3.5-VL weights, mounted read-only with the seg model.
VLM_MODEL_PATH = Path(os.getenv("VLM_MODEL_PATH", str(MODEL_ROOT / "vlm")))
VLM_SERVED_NAME = os.getenv("VLM_SERVED_NAME", "qwen3.5-vl")

VLM_PORT = int(os.getenv("VLM_PORT", "8000"))
VLM_MAX_MODEL_LEN = int(os.getenv("VLM_MAX_MODEL_LEN", "32768"))
VLM_MAX_IMAGES_PER_PROMPT = int(os.getenv("VLM_MAX_IMAGES_PER_PROMPT", "3"))

# GPU budget. Grand Challenge gives T4 (16GB) or A10G (24GB); the segmentation
# stage has already released its VRAM by the time vLLM starts.
VLM_GPU_UTIL = float(os.getenv("VLM_GPU_UTIL", "0.85"))
# T4 is Turing: no native bf16. Use float16 there, bfloat16 on A10G.
VLM_DTYPE = os.getenv("VLM_DTYPE", "float16")
VLM_STARTUP_TIMEOUT_S = int(os.getenv("VLM_STARTUP_TIMEOUT_S", "900"))

# Skip torch.compile + CUDA-graph capture. Measured on a 16GB card: compilation
# alone cost 342s and warmup/profiling ~90s, i.e. ~8.5 min before the first
# request -- against a 15 min/case Grand Challenge budget, paid per case because
# each case is its own job. Eager mode trades ~10-20% throughput for ~6 min of
# startup, which is the right trade for ~40 short VQA calls.
VLM_ENFORCE_EAGER = _flag("VLM_ENFORCE_EAGER", "1")

# Force a specific quantization kernel. Empty = let vLLM autodetect (it picks
# awq_marlin on Ampere+). Set to "awq" on Turing (T4, sm75), where the marlin
# kernel is unavailable -- the checkpoint is AWQ-GEMM, which does run on sm75.
VLM_QUANTIZATION = os.getenv("VLM_QUANTIZATION", "").strip()

# Prefix caching reserves extra KV blocks. Worth it when many prompts share a
# prefix, but on a 16GB T4 the weights plus a KV reservation are already tight,
# so it can be turned off to free memory.
VLM_PREFIX_CACHING = _flag("VLM_PREFIX_CACHING", "1")

# 0 = let competition_runner size concurrency from the KV pool the server
# reports (it measured ~10 from 65,472 tokens on a 24 GiB card). Only override
# if a run shows the auto value is wrong.
VLM_MAX_CONCURRENCY = int(os.getenv("VLM_MAX_CONCURRENCY", "0"))

# Worker processes for the per-tooth composites. Steps 1-4 already run
# concurrently, so leave headroom: 4 generators + this many tooth workers
# should not greatly exceed the instance's vCPU count (GC g5.2xlarge = 8).
VLM_TOOTH_WORKERS = int(os.getenv("VLM_TOOTH_WORKERS", "4"))

# Master switch: VLM_ENABLED=0 falls back to the template renderer (useful for
# a fast plumbing-only submission, or if the VLM won't fit the GPU).
VLM_ENABLED = _flag("VLM_ENABLED", "1")
# On any VLM failure, still emit a template report rather than no report at all.
REPORT_FALLBACK_ON_ERROR = _flag("REPORT_FALLBACK_ON_ERROR", "1")