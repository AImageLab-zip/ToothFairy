"""Environment shims — import FIRST, before peft / transformers heavy imports.

Copy of experiments/baseline_hu_v1/03_lora_train/_env.py with the repo-root depth
adjusted for this directory (experiments/finetune_qwen_v1/ -> parents[1] is the repo).

This devcontainer ships a broken ``transformer_engine`` (its .so has an undefined
cublasLt symbol against the installed CUDA). ``peft`` eagerly probes for it at import
(``is_te_pytorch_available``), which crashes with an ``OSError`` on ``ctypes.CDLL``.
We don't use TE, so we install a lightweight stub module (with a valid ``__spec__`` and
NO ``pytorch`` attribute) into ``sys.modules`` so peft's probe returns False cleanly.
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import types
import warnings
from pathlib import Path

# Make the repo root importable (so `import toothfairy...` resolves) and this dir
# (so `import dataset_fiveviews` resolves regardless of cwd).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _block_transformer_engine() -> None:
    if "transformer_engine" in sys.modules and getattr(
        sys.modules["transformer_engine"], "_tf_stub", False
    ):
        return
    stub = types.ModuleType("transformer_engine")
    stub.__spec__ = importlib.machinery.ModuleSpec("transformer_engine", loader=None)
    stub._tf_stub = True  # marker so we don't double-install
    # Deliberately NO `pytorch` attribute -> peft.is_te_pytorch_available() -> False.
    sys.modules["transformer_engine"] = stub


def quiet() -> None:
    _block_transformer_engine()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


quiet()
