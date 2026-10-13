#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL-7B on the Bite2Text (ODIN 2026 Task 2) representation.

Adapted from experiments/finetune_qwen_v1/train.py (the ToothFairy4 fiveviews run). Same
training loop -- single-GPU bf16, LoRA on the LM + ViT-block/merger projections,
completion-only label masking via the ChatML response template, AdamW + cosine schedule,
gradient checkpointing. What changed is the INPUT and everything CBCT-specific:

  * dataset: bite2text_pipeline.dataset.Bite2TextDataset (7 mesh-view renders of the
    bite-registered IOS pair + up to 7 intraoral photos -> Qwen chat messages, all stills,
    no video item) instead of ToothFairyFiveViewsDataset (cross-section video + 5 stills).
  * no video/n_cross/z-jitter/mirror-augment CLI args -- there is no video and no rendering
    RNG to jitter in this pipeline (mesh_views.render_views is deterministic per case).
  * no per-tooth auxiliary head (tooth_labels/_pop_tooth/aux_loss) -- that head reads
    ToothFairy4's FDI tooth_state records, which have no Bite2Text analogue.
  * no family logic (--family-weights/--sampler-seed/WeightedRandomSampler) -- Bite2Text
    records carry no source-family field to balance over.
  * no superior_truncated prompt plumbing -- Bite2TextDataset always sets it False (kept
    only for dict-shape parity with the TF4 collator); Bite2Text's FOV note is about camera
    framing, not CBCT z-extent, and dataset.py does not thread it into the prompt at all.

--chart-concat is kept as a flag, GUARDED: bite2text_pipeline/dataset.py already imports
``data_bite2text.chart_render.concat_target`` lazily "once it lands", and this file mirrors
that -- the chart-token-weighting hookup (experiments/finetune_qwen_v1/chart_loss.py, reused
verbatim aside from repointing its CHART_CLASSES/DELIMITER module globals at the Bite2Text
format) only imports anything when --chart-concat is actually passed, and raises a clear,
immediate SystemExit if data_bite2text/chart_render.py does not exist yet.

Usage:
    python -m bite2text_pipeline.train --split train --val-split val --epochs 8 \
        --batch 1 --accum 8 --lr 1e-4 --save-each-epoch --out runs/bite2text_v1
"""
from __future__ import annotations

# ── Environment bootstrap (inlined _env.py equivalent) ───────────────────────────
# experiments/finetune_qwen_v1/_env.py does three things before any peft/transformers
# import: (1) put the repo root + its own directory on sys.path, (2) stub out a broken
# `transformer_engine` so peft's is_te_pytorch_available() probe returns False instead of
# crashing on this devcontainer's mismatched CUDA .so, (3) quiet a couple of warning
# categories. This file can't add a 4th sibling file (the task scopes the deliverable to
# train.py/infer.py/entrypoint.py), so the same three steps are inlined here instead of
# `import _env`. Must run before `import torch.multiprocessing`, `transformers`, `peft`.
import importlib.machinery
import os
import sys
import types
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (str(_REPO_ROOT), str(_HERE)):
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
    sys.modules["transformer_engine"] = stub  # deliberately no `pytorch` attr


_block_transformer_engine()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Normal imports ────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import time
from functools import partial

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# Package-relative import, with a sys.path fallback so this file runs both as
# `python -m bite2text_pipeline.train` (relative import resolves normally) and as
# `python bite2text_pipeline/train.py` (no parent package -> ImportError -> fall back to
# an absolute import with the repo root already on sys.path from the bootstrap above).
try:
    from .dataset import Bite2TextDataset, load_split_records
    from . import dataset as b2t_dataset
except ImportError:
    from bite2text_pipeline.dataset import Bite2TextDataset, load_split_records
    from bite2text_pipeline import dataset as b2t_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# LoRA hyperparams -- unchanged from experiments/finetune_qwen_v1/train.py. Qwen2.5-VL's ViT
# block MLPs use the same SwiGLU names as the LM (gate/up/down_proj), so the first row adapts
# both; the second row is the vision-only stuff that would otherwise stay frozen: ViT block
# attention (fused `qkv` + output `attn.proj`) and the merger connector (`mlp.0`/`mlp.2`, the
# PatchMerger's `nn.Sequential`). NOT adapted: patch_embed Conv3d, all norms.
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",   # LM (+ ViT block MLPs via name collision)
                "qkv", "attn.proj", "mlp.0", "mlp.2"]  # ViT block attention + merger connector


# ── completion-only label masking (Qwen2.5 ChatML) ──────────────────────────────
# Copied verbatim from experiments/finetune_qwen_v1/train.py -- the ChatML response
# template and the backward-search for it are model/tokenizer behavior, not TF4-specific.

def _response_template_ids(tokenizer) -> list[int]:
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    return [im_start] + tokenizer.encode("assistant\n", add_special_tokens=False)


def _find_response_start(ids: list[int], template_ids: list[int]) -> int | None:
    n = len(template_ids)
    for i in range(len(ids) - n, -1, -1):
        if ids[i:i + n] == template_ids:
            return i + n
    return None


# ── processor call (images only -- Bite2Text never emits a video vision item) ────
# bite2text_pipeline/dataset.py does not define a processor_inputs helper (unlike
# dataset_fiveviews.py, which re-exports toothfairy.dataset.processor_inputs). Importing
# that TF4 function would work mechanically (it already branches on "if images: ... if
# videos: ..."), but it would also pull in toothfairy.dataset's module-level import graph
# (render.py, volume_io.py, config.py -- the whole CBCT rendering pipeline) for a Bite2Text
# run that never touches any of it. This is the images-only subset of that function,
# specific to this file so infer.py can import it alongside MODEL_NAME.

def processor_inputs(processor, batch_msgs: list[list[dict]], add_generation_prompt: bool):
    """Templatize + run the processor for a batch of Bite2Text message-lists. Returns a
    BatchFeature (no labels; the train collate adds them)."""
    texts, images = [], []
    for msgs in batch_msgs:
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt))
        try:
            from qwen_vl_utils import process_vision_info
            imgs, _vids = process_vision_info(msgs)
        except ImportError:
            imgs = [it["image"] for m in msgs for it in (m.get("content") or [])
                    if isinstance(it, dict) and it.get("type") == "image"]
        if imgs:
            images.extend(imgs)
    kwargs = {"images": images} if images else {}
    return processor(text=texts, padding=True, return_tensors="pt", **kwargs)


def collate_fn(batch, processor, resp_template_ids, chart_tagger=None):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    inputs = processor_inputs(
        processor, [b["messages"] for b in batch], add_generation_prompt=False)
    labels = inputs["input_ids"].clone()
    # Per-token loss weights + per-token chart class ids: only allocated when a chart lever
    # is armed (--chart-concat with --chart-loss-weight != 1.0 or --chart-class-weights).
    # Both stay None on the default path, so the trainer takes the model's own CE.
    tw = cid = None
    if chart_tagger is not None:
        cid = torch.full_like(labels, -1)
        if chart_tagger.weighted:
            tw = torch.ones_like(labels, dtype=torch.float32)
    for i in range(len(batch)):
        ids = labels[i].tolist()
        pos = _find_response_start(ids, resp_template_ids)
        if pos is not None:
            labels[i, :pos] = -100          # mask system+user+vision; loss on report only
            if cid is not None:
                w, c = chart_tagger(ids[pos:])
                cid[i, pos:] = c
                if tw is not None:
                    tw[i, pos:] = w
        else:
            labels[i, :] = -100             # assistant turn missing -> drop from loss
    labels[inputs["attention_mask"] == 0] = -100
    if tw is not None:
        inputs["chart_token_weights"] = tw
    if cid is not None:
        inputs["chart_class_ids"] = cid
    # If NO row has a supervised token the batch loss is NaN, which permanently NaNs the
    # LoRA weights. Drop such a batch. (A mixed batch is fine -- an all-(-100) row just
    # contributes no tokens.)
    if not (labels != -100).any():
        return None
    inputs["labels"] = labels
    return inputs


def _pop_chart(batch: dict) -> dict:
    """Splits the chart token weights / class ids out of the batch before model(**batch) --
    Qwen raises on unknown kwargs. No-op (returns {}) unless --chart-concat armed the
    weighting lever, since collate_fn only adds these keys in that case."""
    return {k: batch.pop(k) for k in list(batch) if k.startswith("chart_")}


@torch.inference_mode()
def _val_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _pop_chart(batch)   # val loss stays the LM loss, comparable across runs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        tot += out.loss.item(); n += 1
    model.train()
    return tot / n if n else float("nan")


def _val_subset(records, n):
    """First ``n`` records. TF4's counterpart (_stratified_subset) round-robins over a
    `family` field to keep a small val slice from being all one source family; Bite2Text
    records carry no family field (load_split_records returns {case_id, target} only), so
    this is a plain prefix slice. n<=0 or n>=len -> all records."""
    if not records or n <= 0 or n >= len(records):
        return records
    return records[:n]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--split", default="train")
    p.add_argument("--split-dir", default="data_bite2text/splits",
                   help="directory holding <split>.json case-id lists (and, once it lands, "
                        "splits/canonical/<split>.jsonl target prose). "
                        "bite2text_pipeline.dataset.SPLITS_DIR is a module constant rather "
                        "than a load_split_records() parameter, so a non-default value here "
                        "is applied by overriding that module global at startup.")
    p.add_argument("--max-cases", type=int, default=0, help="0 = all records in the split")
    p.add_argument("--max-steps", type=int, default=0, help="0 = run full --epochs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--chart-concat", action="store_true",
                   help="target = chart block + '\\n\\nReport:\\n' + canonical prose "
                        "(data_bite2text/chart_render.py, once it exists). Both segments "
                        "supervised. Raises at startup if that module is missing.")
    p.add_argument("--chart-loss-weight", type=float, default=1.0,
                   help="weight on the chart segment's tokens. 1.0 leaves plain LM CE.")
    p.add_argument("--chart-class-weights", action="store_true",
                   help="up-weight the tokens spelling a rare chart class by inverse class "
                        "frequency (only meaningful once the chart module can report class "
                        "counts).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--vision-lr", type=float, default=None,
                   help="separate (usually lower) LR for the vision-tower LoRA params "
                        "(attention + merger); defaults to --lr when unset")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026,
                   help="seeds torch + the DataLoader shuffle generator")
    p.add_argument("--log-every", type=int, default=1, help="log train loss every N opt steps")
    p.add_argument("--val-split", default=None, help="split for periodic val loss (optional)")
    p.add_argument("--val-cases", type=int, default=0, help="0 = use every case in --val-split")
    p.add_argument("--val-every", type=int, default=0, help="opt steps between val-loss checks (0=off)")
    p.add_argument("--out", required=True, help="output dir (adapter + loss.csv + config)")
    p.add_argument("--init-from", default=None,
                   help="continue from an existing LoRA adapter dir (loaded trainable, "
                        "fresh optimizer/scheduler) instead of a fresh adapter")
    p.add_argument("--save-each-epoch", action="store_true",
                   help="also save an adapter checkpoint after every epoch "
                        "(out/adapter_ep{epoch_offset+epoch})")
    p.add_argument("--epoch-offset", type=int, default=0,
                   help="added to the loop epoch when naming per-epoch checkpoints")
    # ── Weights & Biases ─────────────────────────────────────────────────────
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="rgbite-bite2text",
                   help="W&B project (created on first run)")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name (default: --out basename)")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                   help="online needs `wandb login`; offline logs locally to sync later")
    return p.parse_args()


def _require_chart_module(args) -> None:
    """Fast-fail guard, called immediately after parse_args() (before any record/processor/
    model load): if --chart-concat is passed, data_bite2text/chart_render.py must exist.
    bite2text_pipeline/dataset.py's own Bite2TextDataset.__getitem__ already does
    ``from data_bite2text.chart_render import concat_target  # once it lands`` lazily under
    this same flag; this mirrors that, but fails in milliseconds instead of deep inside the
    first training step (or after a 16GB model pull)."""
    if not args.chart_concat:
        return
    try:
        import data_bite2text.chart_render  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "--chart-concat requires data_bite2text/chart_render.py (the Bite2Text chart "
            "block renderer + concat_target -- bite2text_pipeline/dataset.py's own "
            "Bite2TextDataset.__getitem__ already imports it lazily under this same flag). "
            "It does not exist yet; see data_canonical/chart_render.py (TF4) as the "
            "template.") from e


def _build_chart_tagger(args, repo_root: Path, records: list[dict], tokenizer):
    """Wires up chart_loss.py's token-weighting/tagging helpers, reused verbatim aside from
    repointing its module-level CHART_CLASSES/DELIMITER (imported from toothfairy.chart_parse
    at that file's own top) at the Bite2Text format -- those functions read the module
    globals rather than taking them as parameters, so they're repointed here instead of
    editing that (existing) file. Returns (chart_tagger, CHART_CLASSES, weighted_ce); all
    None/() when --chart-concat is off.

    NOTE: unexercised path -- data_bite2text/chart_render.py does not exist yet, so
    _require_chart_module() above would already have raised before this is ever called with
    args.chart_concat=True. This function body is forward scaffolding, written to the same
    shape as TF4's train.py chart_tagger, not something that can be run or verified in this
    environment today.
    """
    if not args.chart_concat:
        return None, (), None
    import data_bite2text.chart_render as b2t_chart_render

    chart_classes = getattr(b2t_chart_render, "CHART_CLASSES", None)
    chart_delim = getattr(b2t_chart_render, "DELIMITER", None)
    if chart_classes is None or chart_delim is None:
        try:
            import data_bite2text.chart_parse as b2t_chart_parse
            chart_classes = chart_classes or getattr(b2t_chart_parse, "CHART_CLASSES", None)
            chart_delim = chart_delim or getattr(b2t_chart_parse, "DELIMITER", None)
        except ImportError:
            pass
    if chart_classes is None or chart_delim is None:
        raise SystemExit(
            "--chart-concat: data_bite2text/chart_render.py exists but does not expose "
            "CHART_CLASSES/DELIMITER (directly, or via a data_bite2text/chart_parse.py "
            "sibling, mirroring toothfairy/chart_parse.py) -- the token-weighting helpers "
            "reused from experiments/finetune_qwen_v1/chart_loss.py need both.")

    sys.path.insert(0, str(repo_root / "experiments" / "finetune_qwen_v1"))
    import chart_loss as _chart_loss
    _chart_loss.CHART_CLASSES = chart_classes
    _chart_loss.DELIMITER = chart_delim
    from chart_loss import (TokenTexts, chart_span, class_id_tensor, class_occurrences,
                            class_weights, token_weights, weighted_ce)

    chart_state = getattr(b2t_chart_render, "chart_state", None)
    weighted = args.chart_loss_weight != 1.0 or args.chart_class_weights
    chart_cls_w = None
    if args.chart_class_weights:
        if chart_state is None:
            log.warning("--chart-class-weights requested but data_bite2text.chart_render "
                        "has no chart_state(record) -- falling back to unweighted classes")
        else:
            counts = {c: 0 for c in chart_classes}
            for r in records:
                struct = (r.get("target") or {}).get("record")
                if struct is None:
                    continue
                for cls in chart_state(struct).values():
                    counts[cls] = counts.get(cls, 0) + 1
            if any(counts.values()):
                chart_cls_w = class_weights(counts)
                log.info("chart class counts %s", counts)
                log.info("chart class weights %s",
                         {c: round(w, 2) for c, w in chart_cls_w.items()})

    class _ChartTagger:
        """ids of one supervised run -> (per-token weight, per-token class id)."""

        def __init__(self):
            self._texts = TokenTexts(tokenizer)
            self._miss = 0
            self.weighted = weighted

        def __call__(self, ids):
            texts = self._texts(ids)
            span = chart_span(texts)
            if span is None:     # no delimiter in the target -> nothing to tag, and say so
                self._miss += 1
                log.warning("chart tagging: no delimiter in target (%d so far)", self._miss)
                return (torch.ones(len(texts)),
                        torch.full((len(texts),), -1, dtype=torch.long))
            occ = class_occurrences(texts, span)
            w = (token_weights(texts, span, occ, args.chart_loss_weight, chart_cls_w)
                 if self.weighted else torch.ones(len(texts)))
            return w, class_id_tensor(len(texts), occ)

    log.info("chart-concat ON: token weighting=%s (chart_loss_weight=%.3f, "
             "class_weights=%s)", weighted, args.chart_loss_weight, args.chart_class_weights)
    return _ChartTagger(), chart_classes, weighted_ce


def main():
    args = parse_args()
    _require_chart_module(args)   # fast-fail before any record/processor/model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── --split-dir: patch the module global (see parse_args help) ───────────
    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = _REPO_ROOT / split_dir
    if split_dir != b2t_dataset.SPLITS_DIR:
        log.info(f"overriding SPLITS_DIR: {b2t_dataset.SPLITS_DIR} -> {split_dir}")
        b2t_dataset.SPLITS_DIR = split_dir

    # ── Records ──────────────────────────────────────────────────────────────
    records = load_split_records(args.split)
    if args.max_cases:
        records = records[:args.max_cases]
    # F1 guard: a record without a canonical target trains NOTHING (dataset returns
    # None, collate drops it). Without this check a targets-less tree "trains" for
    # 0 steps and saves an identity adapter with exit code 0.
    n_supervised = sum(
        1 for r in records
        if r.get("target") and len(r["target"].get("prose") or "") >= b2t_dataset.MIN_REPORT_CHARS)
    log.info(f"split={args.split}  split_dir={split_dir}  records={len(records)}  "
             f"supervised={n_supervised}  chart_concat={args.chart_concat}")
    if n_supervised == 0:
        raise SystemExit(
            f"no supervised targets: none of the {len(records)} '{args.split}' cases has "
            f"a canonical prose target under {b2t_dataset.SPLITS_DIR / 'canonical'} — "
            f"build data_bite2text canonical targets first (splits/canonical/<split>.jsonl)")
    train_ds = Bite2TextDataset(records, training=True, chart_concat=args.chart_concat,
                                seed=args.seed)

    val_loader = None
    if args.val_split:
        vrec_all = load_split_records(args.val_split)
        vrec = _val_subset(vrec_all, args.val_cases)
        log.info(f"val: {len(vrec)}/{len(vrec_all)} cases")
        val_ds = Bite2TextDataset(vrec, training=True, chart_concat=args.chart_concat)

    # ── Processor + model ────────────────────────────────────────────────────
    log.info(f"Loading processor: {args.model}")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model, min_pixels=256 * 256, max_pixels=1280 * 1280)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model)
    resp_ids = _response_template_ids(processor.tokenizer)

    log.info(f"Loading model (bf16): {args.model}  [first load pulls ~16GB from HF]")
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    log.info(f"attention: {attn}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=attn)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if args.init_from:
        from peft import PeftModel
        log.info(f"init-from: continuing from LoRA adapter (trainable): {args.init_from}")
        model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    if not args.init_from:
        _EXPECT_TRAINABLE = 51_824_640
        if trainable != _EXPECT_TRAINABLE:
            log.warning("trainable=%d != expected %d -- LoRA targeting may have changed "
                        "(check LORA_TARGETS against the current Qwen2.5-VL module names)",
                        trainable, _EXPECT_TRAINABLE)

    # ── chart token tagging (guarded; see _require_chart_module/_build_chart_tagger) ────
    chart_tagger, CHART_CLASSES, weighted_ce = _build_chart_tagger(
        args, _REPO_ROOT, records, processor.tokenizer)

    _collate = partial(collate_fn, processor=processor, resp_template_ids=resp_ids,
                       chart_tagger=chart_tagger)
    g = torch.Generator().manual_seed(args.seed)
    # persistent_workers=False: workers respawn each epoch and so pick up set_epoch (kept
    # for interface parity with the TF4 dataset contract even though Bite2TextDataset does
    # not yet vary anything per-epoch).
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=_collate,
                        pin_memory=False, drop_last=True,
                        persistent_workers=False, generator=g)
    if args.val_split:
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=_collate)

    # Two param groups so the vision LoRA (attention + merger) can train at a different LR
    # than the LM. A param is "vision" iff its module path contains 'visual'.
    vis_lr = args.vision_lr if args.vision_lr is not None else args.lr
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    vis_params = [p for n, p in named if "visual" in n]
    lm_params = [p for n, p in named if "visual" not in n]
    log.info(f"trainable param tensors: {len(lm_params)} LM @ lr={args.lr}, "
             f"{len(vis_params)} vision @ lr={vis_lr}")

    groups = [{"params": lm_params, "lr": args.lr},
              {"params": vis_params, "lr": vis_lr}]
    optim = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    log.info(f"opt steps: {total_steps}  warmup: {warmup}  steps/epoch: {steps_per_epoch}")

    run_config = {
        "model": args.model, "split": args.split, "split_dir": str(split_dir),
        "n_records": len(records),
        "representation": "mesh_views(<=7 stills) + intraoral_photos(<=7 stills)",
        "chart_concat": args.chart_concat,
        "chart_loss_weight": args.chart_loss_weight,
        "chart_class_weights": args.chart_class_weights,
        "max_cases": args.max_cases, "batch": args.batch,
        "accum": args.accum, "effective_batch": args.batch * args.accum,
        "lr": args.lr, "vision_lr": vis_lr, "max_steps": args.max_steps, "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio, "init_from": args.init_from,
        "total_opt_steps": total_steps, "warmup_steps": warmup,
        "seed": args.seed,
        "val_split": args.val_split, "val_every": args.val_every, "val_cases": args.val_cases,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "targets": LORA_TARGETS},
        "trainable_params": trainable, "total_params": total,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # ── W&B (opt-in; no-op unless --wandb) ────────────────────────────────────
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name or out_dir.name, mode=args.wandb_mode,
            config=run_config, dir=str(out_dir))
        log.info(f"W&B: project={args.wandb_project} run={wb.name} mode={args.wandb_mode}")

    loss_csv = out_dir / "loss.csv"
    cf = open(loss_csv, "w", newline="")
    cw = csv.writer(cf)
    cw.writerow(["step", "loss", "lr", "grad_norm", "sec_per_step"])

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train(); optim.zero_grad()
    global_step, done = 0, False
    accum_loss, accum_n = 0.0, 0
    # Per-class chart hit/total counts over the logging window (only meaningful under
    # --chart-concat; CHART_CLASSES is () otherwise, so these loops are no-ops).
    chart_hit = torch.zeros(max(len(CHART_CLASSES), 1), device=device)
    chart_tot = torch.zeros(max(len(CHART_CLASSES), 1), device=device)
    t_step = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainable_for_clip = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(1, args.epochs + 1):
        if done:
            break
        train_ds.set_epoch(epoch)
        optim.zero_grad(); accum_loss, accum_n = 0.0, 0
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            extra = _pop_chart(batch)
            tw = extra.pop("chart_token_weights", None)
            cid = extra.pop("chart_class_ids", None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if tw is None:
                    out_lm = model(**batch)
                    lm_loss = out_lm.loss
                else:
                    # Our own CE so the chart segment can carry a different weight. Labels
                    # are pulled out first -- letting the model compute its CE too would
                    # double the largest activation in the step for nothing.
                    lbl = batch.pop("labels")
                    out_lm = model(**batch)
                    lm_loss = weighted_ce(out_lm.logits, lbl, tw)  # bound by _build_chart_tagger
                    batch["labels"] = lbl
            if cid is not None:
                # Teacher-forced per-class chart recall. Position j predicts token j+1, so a
                # class id at position j is scored against the argmax at j-1. Detached and
                # counted only -- it never touches the loss.
                with torch.no_grad():
                    sel = cid[:, 1:] >= 0
                    if sel.any():
                        pred = out_lm.logits[:, :-1, :].argmax(-1)[sel]
                        gold = batch["input_ids"][:, 1:][sel]
                        cls = cid[:, 1:][sel]
                        hit = pred == gold
                        for c in range(len(CHART_CLASSES)):
                            m = cls == c
                            chart_hit[c] += (hit & m).sum()
                            chart_tot[c] += m.sum()
            del out_lm
            loss = lm_loss / args.accum
            loss.backward()
            accum_loss += loss.item() * args.accum
            accum_n += 1

            if (step + 1) % args.accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_for_clip, 1.0).item()
                lr_now = sched.get_last_lr()[0]        # LR used for THIS step, pre-advance
                optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
                sps = (time.time() - t_step) / args.accum
                t_step = time.time()

                if global_step % args.log_every == 0:
                    avg = accum_loss / accum_n
                    lr_vis = sched.get_last_lr()[1]
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    log.info(f"step={global_step:4d}/{total_steps}  loss={avg:.4f}  "
                             f"lr={lr_now:.2e}  lr_vis={lr_vis:.2e}  gnorm={gnorm:.2f}  "
                             f"{sps:.1f}s/step  peakVRAM={peak:.1f}GB")
                    cw.writerow([global_step, round(avg, 5), lr_now, round(gnorm, 3),
                                 round(sps, 2)]); cf.flush()
                    wlog = {"train/loss": avg, "train/lr": lr_now, "train/lr_vision": lr_vis,
                            "train/grad_norm": gnorm, "perf/sec_per_step": sps,
                            "perf/peak_vram_gb": peak, "epoch": epoch}
                    if args.chart_concat:
                        rec_txt = []
                        for c, name in enumerate(CHART_CLASSES):
                            r = (float(chart_hit[c] / chart_tot[c])
                                 if chart_tot[c] > 0 else float("nan"))
                            wlog[f"chart/recall_{name.replace(' ', '_')}"] = r
                            wlog[f"chart/n_{name.replace(' ', '_')}"] = float(chart_tot[c])
                            rec_txt.append(f"{name.split()[0][:4]}={r:.2f}")
                        log.info("   chart recall: %s  (n=%s)", " ".join(rec_txt),
                                 [int(x) for x in chart_tot])
                        chart_hit.zero_(); chart_tot.zero_()
                    if wb is not None:
                        wb.log(wlog, step=global_step)
                    accum_loss, accum_n = 0.0, 0

                if val_loader is not None and args.val_every and global_step % args.val_every == 0:
                    vl = _val_loss(model, val_loader, device)
                    log.info(f"  [val] step={global_step}  val_loss={vl:.4f}")
                    if wb is not None:
                        wb.log({"val/loss": vl}, step=global_step)

                if args.max_steps and global_step >= args.max_steps:
                    done = True
                    break

        # ── per-epoch checkpoint ──────────────────────────────────────────────
        if args.save_each_epoch and not done:
            ep_id = args.epoch_offset + epoch
            ep_dir = out_dir / f"adapter_ep{ep_id}"
            model.save_pretrained(ep_dir)
            processor.save_pretrained(ep_dir)
            cf.flush()
            if val_loader is not None:
                vl = _val_loss(model, val_loader, device)
                log.info(f"epoch {epoch} (global ep{ep_id}) val_loss={vl:.4f}")
                if wb is not None:
                    wb.log({"val/loss_epoch": vl, "epoch": epoch}, step=global_step)
            log.info(f"epoch {epoch} (global ep{ep_id}) checkpoint -> {ep_dir}  "
                     f"step={global_step}")

    cf.close()
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info(f"training done: {global_step} opt steps, peak VRAM {peak:.1f} GB")

    # F1 guard: never ship an adapter that took zero optimizer steps — an untrained
    # LoRA (B=0) is an identity adapter and downstream code cannot tell it apart.
    if global_step == 0:
        raise SystemExit("0 optimizer steps taken (every batch was dropped) — "
                         "refusing to save an untrained identity adapter")

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    log.info(f"adapter -> {adapter_dir}")
    log.info(f"loss curve -> {loss_csv}")

    if wb is not None:
        wb.summary["final_opt_steps"] = global_step
        wb.summary["peak_vram_gb"] = peak
        wb.finish()


if __name__ == "__main__":
    main()
