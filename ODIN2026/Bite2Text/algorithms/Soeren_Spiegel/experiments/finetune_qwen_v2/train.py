#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL-7B on the ToothFairy4 cross-section video + five-view stills.

Same training loop as experiments/baseline_hu_v1/03_lora_train/train.py (single-GPU bf16,
LoRA on the LM projections, completion-only label masking, AdamW + cosine schedule, gradient
checkpointing, W&B train+val logging). The ONLY change is the input: this run feeds the
``dataset_fiveviews`` representation (cross-section video + five_views panoramic stills)
instead of the axial RGB video.

``--frames`` sets the cross-section video budget (N_CROSS_SECTIONS), passed to the dataset as
``n_cross``. The five stills are always DEFAULT_VIEWS.

Usage (FULL, this run):
    python train.py --split train --val-split val --epochs 6 --warmup-ratio 0.03 \
        --batch 1 --grad-accum 8 --frames 40 --val-every 50 --val-cases 8 \
        --save-each-epoch --wandb --out runs/fiveviews_v1
"""
from __future__ import annotations

import _env  # noqa: F401  (blocks broken transformer_engine — MUST be first)

import argparse
import contextlib
import csv
import json
import logging
import time
from functools import partial
from pathlib import Path

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

from dataset_fiveviews import (
    ToothFairyFiveViewsDataset, load_split_records, processor_inputs,
)
from toothfairy.config import FRAME_TILE
from toothfairy.views import (
    DEFAULT_VIEWS, STILL_TOKEN_BUDGET_PANO, STILL_TOKEN_BUDGET_V2, TILE_V2, VIEWS_PANO,
    VIEWS_V2, ALL_VIEWS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# LoRA hyperparams (as in HEADLINE).
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
# The first row is the LM projections, and — because Qwen2.5-VL's
# ViT MLP uses the SAME SwiGLU names (gate/up/down_proj) — it has ALSO been adapting the 32
# vision-block MLPs since v0 (the old "LM only" comment was wrong; verified by the trainable
# count 47,589,376 = 40,370,176 LM + 7,219,200 ViT-MLP). The second row adds the vision side
# that was genuinely frozen: ViT block attention (fused `qkv` + output `attn.proj`) and the
# merger connector (`mlp.0`/`mlp.2`, the only indexed-child MLP in the model — the LM's mlp
# children are named, so these do not leak into it). PEFT matches on `.<key>` suffix, so:
#   "qkv"       -> visual.blocks.*.attn.qkv          (LM has no fused qkv)
#   "attn.proj" -> visual.blocks.*.attn.proj         (not patch_embed.proj, not LM o_proj)
#   "mlp.0/2"   -> visual.merger.mlp.0 / .mlp.2       (PatchMerger nn.Sequential)
# NOT adapted: patch_embed Conv3d (input tokenizer, high-risk) and all norms.
# Expected trainable after this change: 51,824,640 (0.62%). Assert it in pre-flight.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",   # LM (+ ViT block MLPs via name collision)
                "qkv", "attn.proj", "mlp.0", "mlp.2"]  # ViT block attention + merger connector


# ── completion-only label masking (Qwen2.5 ChatML) ──────────────────────────────

def _response_template_ids(tokenizer) -> list[int]:
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    return [im_start] + tokenizer.encode("assistant\n", add_special_tokens=False)


def _find_response_start(ids: list[int], template_ids: list[int]) -> int | None:
    n = len(template_ids)
    for i in range(len(ids) - n, -1, -1):
        if ids[i:i + n] == template_ids:
            return i + n
    return None


def collate_fn(batch, processor, resp_template_ids, chart_tagger=None):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    inputs = processor_inputs(
        processor, [b["messages"] for b in batch], add_generation_prompt=False)
    labels = inputs["input_ids"].clone()
    # per-token loss weights (only when a chart lever is armed) and the per-token chart
    # class ids (always under --chart-concat — they are the smoke collapse gate). Both stay None
    # on the default path, so the trainer takes the model's own CE and the run is
    # bit-comparable to combined_v1.
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
    # stack the per-tooth labels alongside. They ride in the same dict so the loop's
    # .to(device) sweep picks them up, and the loop POPS them before model(**batch) — Qwen
    # raises on unknown kwargs.
    for key in ("tooth_status", "tooth_attr", "tooth_attr_mask"):
        if key in batch[0]:
            inputs[key] = torch.stack([b[key] for b in batch])
    # if NO row has a supervised token the batch loss is NaN, which permanently NaNs the
    # LoRA weights. Drop such a batch. (A mixed batch is fine — an all-(-100) row just
    # contributes no tokens; partial-filtering is avoided because Qwen's video/image tensors
    # are flattened across the batch and are not row-indexable by an input-id mask.)
    if not (labels != -100).any():
        return None
    inputs["labels"] = labels
    return inputs


def _stratified_subset(records, n):
    """First ``n`` records, round-robin over families, so a small val slice is not all-A
    (the sorted split otherwise makes ``[:8]`` eight family-A cases). n>=len -> all."""
    if not records or n >= len(records):
        return records
    from collections import defaultdict
    by_fam = defaultdict(list)
    for r in records:
        by_fam[r["family"]].append(r)
    fams = sorted(by_fam)
    out = []
    i = 0
    while len(out) < n:
        f = fams[i % len(fams)]
        if by_fam[f]:
            out.append(by_fam[f].pop(0))
        i += 1
        if all(not v for v in by_fam.values()):
            break
    return out[:n]


def _parse_family_weights(spec: str) -> dict:
    """'P=1.0,A=1.7,F=2.0,S=1.3' -> {'P':1.0,...}."""
    out = {}
    for tok in spec.split(","):
        k, v = tok.split("=")
        out[k.strip()] = float(v)
    return out



def _eff_still_budget(args) -> int:
    """The still token budget that `dataset_fiveviews` will actually apply.

    Must stay in step with the same expression in `dataset_fiveviews.render_case_to_messages`.
    It is a function so the log line and `run_config.json` cannot drift from each other or from
    the renderer — they did: the run log printed `still_tokens=4800` for `chartv2_v1` while the
    effective budget was 0, and only `run_config.json` recorded the truth."""
    return args.still_tokens if (args.render_v2 or args.still_per_view) else 0


def _resolve_views(args) -> tuple[str, ...]:
    """--views wins; otherwise VIEWS_V2 under --render-v2, else DEFAULT_VIEWS.

the arm is a command-line argument, not an edit. Unknown keys fail loudly here rather
    than being silently omitted by five_views (which drops a view it cannot build, so a typo
    would look like a legitimately missing view)."""
    if not getattr(args, "views", None):
        return VIEWS_V2 if args.render_v2 else DEFAULT_VIEWS
    keys = tuple(v.strip() for v in args.views.split(",") if v.strip())
    known = set(ALL_VIEWS) | set(DEFAULT_VIEWS) | set(VIEWS_V2) | set(VIEWS_PANO)
    bad = [k for k in keys if k not in known]
    if bad:
        raise SystemExit(f"unknown --views key(s) {bad}; known: {sorted(known)}")
    return keys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--split", default="train")
    p.add_argument("--max-cases", type=int, default=0, help="0 = all records in the split")
    p.add_argument("--max-steps", type=int, default=0, help="0 = run full --epochs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--frames", type=int, default=40)
    # Representation v2. One flag, because
    # the levers are not independent: the crops and the crown bands are what make the larger
    # tile affordable, and the hybrid reduction is what the larger tile is for.
    p.add_argument("--render-v2", action="store_true",
                   help="4 cropped RGB stills at --tile, hybrid ray reduction, validated crown "
                        "bands, and a slab-reduced HU-windowed video with empty frames dropped")
    p.add_argument("--tile", type=int, default=FRAME_TILE,
                   help=f"STILL height in px, multiple of 28 (v2 uses {TILE_V2})")
    # The video is deliberately NOT given the resolution bump: it already gains the slab
    # reduction and the HU windows at zero token cost, and it is half the budget, so raising
    # its tile too would be a second change to the token bill in the same run.
    # HU-colouring the video is NOT part of --render-v2: measured, a buccolingual cross-section
    # has a median voxel of +42 HU, so RGB_WINDOWS (validated on whole-head axial slices) puts
    # 75% of the frame in R alone and the frame reads red. Three alternative window sets and a
    # foreground-only stretch all scored equal or worse on channel balance. The slab MIP and the
    # empty-frame drop are independent of colour and are the measured wins; this flag is the
    # unproven half, kept separable.
    p.add_argument("--video-rgb", action="store_true",
                   help="render the video through hu.RGB_WINDOWS (unproven; see --render-v2)")
    p.add_argument("--video-tile", type=int, default=FRAME_TILE,
                   help="VIDEO frame height in px, multiple of 28")
    # The v2 view set is case-dependent (a duplicate mandible panoramic is dropped, a
    # single-jaw scan gets one occlusal tile), so a fixed tile makes the sparser cases the
    # smallest inputs. A budget spends the freed tokens on patch density instead.
    p.add_argument("--still-tokens", type=int, default=STILL_TOKEN_BUDGET_V2,
                   help="per-case image-token budget across the stills (v2 only; 0 = off)")
    # The per-IMAGE cap (views.MAX_MPX = 2028 tokens) is what --still-tokens cannot cross, so
    # past ~2 k tokens on one field the only lever is more images of it. --pano-split k cuts
    # each panoramic into k overlapping arc tiles, each at the cap: k-fold ViT patch density
    # over the same anatomy (views.split_wide — the tokens are patch density, NOT data).
    p.add_argument("--pano-split", type=int, default=1,
                   help="emit each panoramic as k overlapping arc tiles (1 = off). Tiling is "
                        "the fallback lever; prefer --still-mpx, which magnifies the WHOLE "
                        "field in one image instead of splitting it across k ViT contexts")
    # The 1.64 Mpx we have always passed to the processor is OUR tightening, not the model's
    # limit: Qwen2.5-VL's own default is 12.8 Mpx = 16384 tokens per image. Raising it is what
    # lets one panoramic carry more than 2028 tokens without being cut into tiles.
    # ⚠ This value is passed to BOTH the processor (`max_pixels`) and the renderer (`max_mpx`).
    # They must agree: render above the processor's cap and it silently downsamples the frame;
    # render below it and the tokens are wasted. Inference and the container must use the same
    # number as training.
    p.add_argument("--still-mpx", type=float, default=1.59,
                   help="per-IMAGE megapixel cap for the stills AND the processor's max_pixels "
                        "(1.59 = the shipped 1280x1280 budget; 12.8 = Qwen's own default)")
    p.add_argument("--still-per-view", action="store_true",
                   help="scale each still to its own cap instead of by a factor common to all. "
                        "Without it a view already at the cap pins the factor at ~1.0 and a "
                        "small view stays small — pano_union shipped at 133 tokens that way")
    # the view set is selectable rather than implied by --render-v2, so a renderer arm is
    # one argument instead of an edit. `--frames 0` drops the video entirely (panoramic-only):
    # the arch fit still runs, but no frames are rendered and no video item reaches the
    # processor — see dataset_fiveviews.render_fiveviews_messages.
    p.add_argument("--views", default=None,
                   help="comma-separated view keys (default: VIEWS_V2 under --render-v2, else "
                        f"DEFAULT_VIEWS). Panoramic-only is '{','.join(VIEWS_PANO)}' "
                        f"with --frames 0 --still-tokens {STILL_TOKEN_BUDGET_PANO}")
    p.add_argument("--video-axis", default="buccolingual", choices=["buccolingual"],
                   help="cross-section video slicing direction. Only buccolingual is implemented "
                        "(the axial path was not ported; video-sweep). Kept for launcher parity.")
    p.add_argument("--canonical", action="store_true",
                   help="train on the merged canonical-prose target "
                        "(splits/canonical/<split>.jsonl) instead of the raw EN reports; "
                        "one record per case, multi-report duplication removed")
    # chart: concatenated interrogate-then-report target.
    p.add_argument("--chart-concat", action="store_true",
                   help=" target = chart block + '\\n\\nReport:\\n' + canonical prose "
                        "(data_canonical/chart_render.py). Both segments supervised. "
                        "Requires --canonical.")
    p.add_argument("--chart-loss-weight", type=float, default=1.0,
                   help="weight on the chart segment's tokens. 1.0 (the default and the "
                        "shipped setting) leaves plain LM CE — same head, same cardinality, so "
                        "there is no gradient-ratio tuning round to run here.")
    # the multilabel chart-only target.
    p.add_argument("--chart-v2", action="store_true",
                   help=" target = the 32-slot MULTILABEL chart ALONE, no prose "
                        "(data_canonical/chart_v2.py). The report is rendered deterministically "
                        "from the chart at inference. Requires --canonical.")
    p.add_argument("--chart-vocab", choices=("v2", "v3"), default="v2",
                   help="non-tooth chart vocabulary. v3 = the VOCABULARY round: IAN split into "
                        "`course` + `position` (the mixed enum agreed 38%% between annotators; "
                        "split and restricted to pairs where both claimed the axis it agrees "
                        "97.8%% / 87.2%% — data_canonical/chart_v3.py --harmonise), and "
                        "annotator silence spelled `n/a` and MASKED from the loss instead of "
                        "trained as a class. The tooth half is byte-identical to v2.")
    p.add_argument("--chart-v2-class-weights", action="store_true",
                   help=" inverse-frequency weighting over the 10 surface classes. The "
                        "corpus imbalance is 122x (n/a 9288 vs root remnant 76); unweighted, "
                        "the cheapest policy is to emit n/a and healthy everywhere.")
    p.add_argument("--chart-v2-weight-power", type=float, default=0.5,
                   help="softening exponent for --chart-v2-class-weights. 1.0 = plain inverse "
                        "frequency (122x spread, 27.6x on root remnant); 0.5 = inverse sqrt "
                        "(11x spread) and the pre-registered default.")
    # non-tooth weighting: the same lever for the four NON-tooth lines, kept on its own
    # knob and its own normalisation so it can be A/B'd against chartv2_pano_v1 ep7.
    p.add_argument("--nontooth-class-weights", action="store_true",
                   help=" inverse-frequency weighting over the non-tooth chart lines, "
                        "keyed by (FIELD, value) — `included` is 83%% of FOV.mandible and a "
                        "minority of FOV.maxillary_sinuses, so pooling the value alone would "
                        "under-weight it exactly where it is rare. Unweighted (ep7) these four "
                        "lines collapsed to their majority value: Sinuses mucosal_thickening "
                        "gold 6 -> predicted 0.")
    p.add_argument("--nontooth-weight-power", type=float, default=0.5,
                   help="softening exponent for --nontooth-class-weights, separate from the "
                        "tooth knob on purpose. 0.5 = inverse sqrt, the pre-registered default.")
    p.add_argument("--nontooth-weight-cap", type=float, default=8.0,
                   help="max inverse-frequency ratio before normalisation. FOV.mandible "
                        "excluded occurs in ONE train case; uncapped it would take ~15x the "
                        "mean non-tooth gradient. 8.0 sits just above the tooth side's realised "
                        "7.38, so both groups keep the same dynamic range.")
    p.add_argument("--chart-class-weights", action="store_true",
                   help="smoke contingency ONLY: up-weight the tokens spelling a rare chart "
                        "class by inverse class frequency. Arm this if the smoke shows the "
                        "chart collapsing to the all-present solution.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--vision-lr", type=float, default=None,
                   help=" separate (usually lower) LR for the vision-tower LoRA params "
                        "(attention + merger); defaults to --lr when unset")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026,
                   help=" seeds torch + the DataLoader sampler for reproducibility")
    # family balancing toward the hidden (complete-scan, cross-centre) test distribution.
    p.add_argument("--family-weights", default=None,
                   help='per-family sampling weights, e.g. "P=1.0,A=1.7,F=2.0,S=1.3"; '
                        "enables a WeightedRandomSampler (num_samples=len(records), so the "
                        "step count is unchanged). Unset -> uniform shuffle.")
    p.add_argument("--sampler-seed", type=int, default=2026)
    # augmentation (mirror + occlusal-z jitter), threaded to the dataset.
    p.add_argument("--augment", action="store_true",
                   help=" enable online augmentation (LR mirror + occlusal-z jitter)")
    p.add_argument("--mirror-p", type=float, default=0.5,
                   help="probability of the laterality-consistent LR mirror per sample")
    p.add_argument("--z-jitter-mm", type=float, default=0.0,
                   help="uniform +/- occlusal-plane jitter in mm applied before rendering")
    # auxiliary per-tooth head on pooled vision features. OFF by default (0.0) so the
    # existing training path is bit-identical — at 0 the head is never built, so it does not
    # even perturb the torch RNG stream.
    p.add_argument("--aux-tooth-weight", type=float, default=0.0,
                   help=" weight of the per-tooth auxiliary loss. 0 = head disabled "
                        "(existing path, bit-identical). Requires --canonical.")
    p.add_argument("--aux-lr", type=float, default=1e-3,
                   help="LR for the auxiliary head's own parameters (third optimizer group); "
                        "it is trained from scratch, so it wants more than the LoRA groups")
    p.add_argument("--aux-hidden", type=int, default=512, help="hidden width of the tooth head")
    p.add_argument("--aux-class-weight-power", type=float, default=1.0,
                   help="exponent on the inverse-frequency status weights; 1.0 = full inverse "
                        "frequency, 0.5 = tempered (use it if the rare classes dominate)")
    p.add_argument("--log-every", type=int, default=1, help="log train loss every N opt steps")
    p.add_argument("--val-split", default=None, help="split for periodic val loss (optional)")
    p.add_argument("--val-cases", type=int, default=62,
                   help=" was 8, which is 8/8 family A on the sorted split; use the full "
                        "62-case val (or a family-stratified subset when smaller)")
    p.add_argument("--val-every", type=int, default=0, help="opt steps between val-loss checks (0=off)")
    p.add_argument("--out", required=True, help="output dir (adapter + loss.csv + config)")
    p.add_argument("--init-from", default=None,
                   help="continue from an existing LoRA adapter dir (loaded trainable, "
                        "fresh optimizer/scheduler) instead of a fresh adapter")
    p.add_argument("--save-each-epoch", action="store_true",
                   help="also save an adapter checkpoint after every epoch "
                        "(out/adapter_ep{epoch_offset+epoch})")
    p.add_argument("--epoch-offset", type=int, default=0,
                   help="added to the loop epoch when naming per-epoch checkpoints "
                        "(e.g. 3 for a continuation → adapter_ep4/5/6)")
    # ── Weights & Biases ─────────────────────────────────────────────────────
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="rgbite-toothfairy4",
                   help="W&B project (created on first run)")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team (default: your account)")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name (default: --out basename)")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                   help="online needs `wandb login`; offline logs locally to sync later")
    return p.parse_args()


def _save_head(head, pooler, args, ep_dir) -> None:
    """ the auxiliary head next to the adapter it was trained with, so eval_tooth_head.py can
    reload the pair. Written as a plain state_dict + geometry — it is not a PEFT artifact and the
    submission path must never load it. No-op when the head is off."""
    if head is None:
        return
    torch.save({"state_dict": head.state_dict(), "d_in": pooler.dim,
                "hidden": args.aux_hidden, "hook": pooler.path},
               Path(ep_dir) / "tooth_head.pt")


def _pop_tooth(batch: dict) -> dict:
    """ split the per-tooth labels and the chart token weights out of the batch. Qwen
    raises on unknown kwargs, so this MUST run before every model(**batch) — the training step
    and the val loss both."""
    return {k: batch.pop(k) for k in list(batch)
            if k.startswith("tooth_") or k.startswith("chart_")}


@torch.inference_mode()
def _val_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _pop_tooth(batch)   # val loss stays the LM loss, comparable across runs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        tot += out.loss.item(); n += 1
    model.train()
    return tot / n if n else float("nan")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B is optional; without it the run is recorded in loss.csv + run_config.json only ──
    if not args.wandb and not args.max_steps:
        log.warning("running without --wandb: metrics go to loss.csv only")
    if args.wandb and args.wandb_mode == "online":
        import wandb
        if wandb.api.api_key is None:
            raise SystemExit(
                "W&B credential missing — run `wandb login` and re-launch.\n"
                "Not falling back to offline: an unsynced local directory is how a run ends up "
                "with no record at all.")

    # seed everything derivable from Python/torch RNG.
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Records ──────────────────────────────────────────────────────────────
    records = load_split_records(args.split, training=True, canonical=args.canonical)
    if args.max_cases:
        records = records[:args.max_cases]
    log.info(f"split={args.split}  train records={len(records)}  frames/video={args.frames}"
             f"  video_axis={args.video_axis}  canonical={args.canonical}  "
             f"augment={args.augment} (mirror_p={args.mirror_p}, z_jitter={args.z_jitter_mm}mm)")
    aux_on = args.aux_tooth_weight > 0
    if args.chart_v2 and not args.canonical:
        raise SystemExit("--chart-v2 requires --canonical: the split rows carry the case ids "
                         "the per-annotator records are keyed by")
    if args.chart_v2 and args.chart_concat:
        raise SystemExit("--chart-v2 and --chart-concat are different targets; pick one")
    if args.chart_vocab == "v3" and not args.chart_v2:
        raise SystemExit("--chart-vocab v3 needs --chart-v2: it changes the non-tooth half of "
                         "that target, and no v1 path has one")
    if args.nontooth_class_weights and not args.chart_v2:
        raise SystemExit("--nontooth-class-weights needs --chart-v2: the v1 target has no "
                         "FOV/IAN/Sinuses/Bone lines to weight")
    if args.chart_concat and not args.canonical:
        raise SystemExit("--chart-concat requires --canonical (the chart is rendered from the "
                         "structured record, which only the canonical splits carry)")
    view_set = _resolve_views(args)
    log.info(f"views={view_set}  video={'OFF (panoramic-only)' if args.frames <= 0 else args.frames}"
             f"  tile={args.tile}  still_tokens={_eff_still_budget(args)}"
             f"(asked {args.still_tokens})"
             f"  pano_split={args.pano_split}  still_mpx={args.still_mpx}"
             f"  per_view={args.still_per_view}")
    train_ds = ToothFairyFiveViewsDataset(
        records, training=True, n_cross=args.frames, video_axis=args.video_axis,
        augment=args.augment, mirror_p=args.mirror_p, z_jitter_mm=args.z_jitter_mm,
        canonical=args.canonical, seed=args.seed, tooth_labels=aux_on,
        chart_concat=args.chart_concat, chart_v2=args.chart_v2,
        chart_vocab=args.chart_vocab,
        views=view_set,
        render_v2=args.render_v2, video_rgb=args.video_rgb,
        tile=args.tile, video_tile=args.video_tile,
        still_token_budget=args.still_tokens, pano_split=args.pano_split,
        max_mpx=args.still_mpx * 1e6, still_per_view=args.still_per_view)

    # build the val loader whenever a val split is given (not only when --val-every),
    # so the end-of-epoch val loss runs even with mid-epoch checks off. Val uses a
    # family-stratified subset so a small --val-cases is not 8/8 family A.
    val_loader = None
    if args.val_split:
        vrec_all = load_split_records(args.val_split, training=True, canonical=args.canonical)
        vrec = _stratified_subset(vrec_all, args.val_cases)
        log.info(f"val: {len(vrec)}/{len(vrec_all)} cases, families "
                 f"{sorted({r['family'] for r in vrec})}")
        val_ds = ToothFairyFiveViewsDataset(
            vrec, training=True, n_cross=args.frames, video_axis=args.video_axis,
            canonical=args.canonical,
            chart_concat=args.chart_concat, chart_v2=args.chart_v2,
            chart_vocab=args.chart_vocab,
            views=view_set,
            render_v2=args.render_v2, video_rgb=args.video_rgb,
        tile=args.tile, video_tile=args.video_tile,
            still_token_budget=args.still_tokens,
            pano_split=args.pano_split, max_mpx=args.still_mpx * 1e6,
            still_per_view=args.still_per_view)  # never augment val

    # ── Processor + model ────────────────────────────────────────────────────
    log.info(f"Loading processor: {args.model}")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model, min_pixels=256 * 256, max_pixels=int(args.still_mpx * 1e6))
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
    # guard the LoRA targeting against a silent transformers module rename that would drop
    # the vision adapters. Only asserted for the exact shipped config (fresh r=16 adapter, all
    # 11 target keys); skipped for --init-from or a reduced --max-cases smoke where r may differ.
    if not args.init_from:
        _EXPECT_TRAINABLE = 51_824_640
        if trainable != _EXPECT_TRAINABLE:
            log.warning("trainable=%d != expected %d — LoRA targeting may have changed "
                        "(check LORA_TARGETS against the current Qwen2.5-VL module names)",
                        trainable, _EXPECT_TRAINABLE)

    # ── chart token tagging ───────────────────────────────────────────────
    # Under --chart-concat every target is tagged so the trainer can log per-class chart
    # recall (the smoke gate: an all-PRESENT chart drives the loss down while the other eight
    # recalls sit at zero, and only the second of those is visible). The loss WEIGHTS are a
    # separate, opt-in thing — without a lever armed the model's own CE is used and the run
    # stays comparable to combined_v1's curve.
    chart_tagger = None
    chart_cls_w = None
    CHART_CLASSES = ()
    # where the non-tooth classes sit inside CHART_CLASSES, so the log line can print the
    # tooth classes per class and the non-tooth ones per FIELD. Empty on every v1 path.
    n_tooth_classes = 0
    nontooth_fields_idx: list = []
    NONTOOTH_ABBR: dict = {}
    if args.chart_v2:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "data_canonical"))
        # The TOOTH half is v2's under both vocabularies (chart_v3._test asserts the rendered
        # tooth slots are identical on all 622 cases). Only the non-tooth vocabulary, its span
        # scanner and its weights come from v3 — and v3 forces every `n/a` weight to 0.0, which
        # `weighted_ce` turns into an exact mask because it divides by ww.sum().
        from chart_loss_v2 import (CHART_CLASS_NAMES, NONTOOTH_FIELD_ABBR, NONTOOTH_FIELDS,
                                   NONTOOTH_INDEX, N_SURFACE, TokenTexts, chart_class_counts,
                                   class_id_tensor, class_occurrences, class_weights,
                                   nontooth_class_counts, nontooth_class_weights,
                                   nontooth_occurrences, nt_name, supervised_span,
                                   token_weights, weighted_ce)
        if args.chart_vocab == "v3":
            from chart_v3 import chart_text as _v2_chart_text
            # ⚠ `token_weights` MUST come from v3 too: v2's resolves a non-tooth index through
            # its own NONTOOTH_NAMES, so mixing them raises KeyError inside a DataLoader worker
            # (caught by the first chartv3 smoke).
            from chart_loss_v3 import (CHART_CLASS_NAMES, NONTOOTH_FIELD_ABBR,  # noqa: F811
                                       NONTOOTH_FIELDS, NONTOOTH_INDEX,
                                       nontooth_class_counts, nontooth_class_weights,
                                       nontooth_occurrences, token_weights)
        else:
            from chart_v2 import chart_text as _v2_chart_text
        CHART_CLASSES = CHART_CLASS_NAMES
        n_tooth_classes = N_SURFACE
        NONTOOTH_ABBR = NONTOOTH_FIELD_ABBR
        nontooth_fields_idx = [
            (field, [N_SURFACE + NONTOOTH_INDEX[nt_name(field, v)] for v in vocab])
            for field, vocab in NONTOOTH_FIELDS]
        if args.chart_v2_class_weights:
            counts = chart_class_counts([r["case_id"] for r in records], _v2_chart_text)
            chart_cls_w = class_weights(counts, power=args.chart_v2_weight_power)
            log.info("chart-v2 class counts %s", counts)
            log.info("chart-v2 class weights (power=%.2f) %s", args.chart_v2_weight_power,
                     {c: round(w, 2) for c, w in chart_cls_w.items()})
            log.info("chart-v2 weight spread %.1fx",
                     max(chart_cls_w.values()) / min(chart_cls_w.values()))
        # the non-tooth half, on its own knob and its own (per-field) normalisation, so
        # arming it moves gradient WITHIN the four non-tooth lines and never between them and
        # the teeth. Both vectors are printed here because "a weight of 6.5x on
        # FOV.mandibular_condyles partial" is a fact the run should state, not one epoch 3
        # should reveal.
        nontooth_cls_w = None
        if args.nontooth_class_weights:
            nt_counts = nontooth_class_counts([r["case_id"] for r in records], _v2_chart_text)
            nontooth_cls_w = nontooth_class_weights(
                nt_counts, power=args.nontooth_weight_power, cap=args.nontooth_weight_cap)
            for field, vocab in NONTOOTH_FIELDS:
                names = [nt_name(field, v) for v in vocab]
                n_f = sum(nt_counts[k] for k in names)
                log.info("non-tooth weighting %-24s n=%-4d %s", field, n_f,
                         "  ".join(f"{v}:{nt_counts[nt_name(field, v)]}"
                                   f"(w{nontooth_cls_w[nt_name(field, v)]:.2f})"
                                   for v in vocab if nt_counts[nt_name(field, v)]))
            log.info("non-tooth weighting ON (power=%.2f, cap=%.1f): %d classes over "
                     "%d fields, %d occurrences, max weight %.2fx",
                     args.nontooth_weight_power, args.nontooth_weight_cap, len(nt_counts),
                     len(NONTOOTH_FIELDS), sum(nt_counts.values()),
                     max(nontooth_cls_w.values()))
            log.info("non-tooth weighting ⚠ IAN is not a harmonised enum (annotator agreement 37%/39%, "
                     "`regular` is a different axis from `lingual`) — read any IAN movement "
                     "against that ceiling. See chart_loss_v2.__doc__.")
        _texts = TokenTexts(processor.tokenizer)

        class _ChartV2Tagger:
            """ids of one supervised run -> (per-token weight, per-token class id).

            The v2 target is chart-only, so the supervised span is the whole run and there is
            no delimiter to miss — the `no delimiter in target` failure mode of the v1 tagger
            cannot occur here.

            The non-tooth ids are tagged whether or not their weights are armed: per-class
            teacher-forced recall on the four non-tooth lines is the diagnostic that would have
            made ep7's collapse visible during training instead of two runs later."""
            weighted = False

            def __call__(self, ids):
                texts = _texts(ids)
                span = supervised_span(texts)
                occ = class_occurrences(texts, span)
                nt_occ = nontooth_occurrences(texts, span)
                w = (token_weights(texts, span, occ, chart_cls_w, nt_occ, nontooth_cls_w)
                     if self.weighted else torch.ones(len(texts)))
                return w, class_id_tensor(len(texts), occ, nt_occ)

        chart_tagger = _ChartV2Tagger()
        chart_tagger.weighted = bool(args.chart_v2_class_weights or
                                     args.nontooth_class_weights)
        log.info("chart-v2 ON: %d classes (%d tooth + %d non-tooth), token weighting=%s "
                 "(teeth=%s, non-tooth=%s)", len(CHART_CLASSES), N_SURFACE,
                 len(CHART_CLASSES) - N_SURFACE, chart_tagger.weighted,
                 bool(args.chart_v2_class_weights), bool(args.nontooth_class_weights))
    elif args.chart_concat:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "data_canonical"))
        from toothfairy.chart_parse import CHART_CLASSES
        from chart_render import chart_state
        n_tooth_classes = len(CHART_CLASSES)      # v1 has no non-tooth vocabulary at all
        from chart_loss import (TokenTexts, chart_class_counts, chart_span, class_id_tensor,
                                class_occurrences, class_weights, token_weights, weighted_ce)
        weighted = args.chart_loss_weight != 1.0 or args.chart_class_weights
        if args.chart_class_weights:
            counts = chart_class_counts(records, chart_state)
            chart_cls_w = class_weights(counts)
            log.info("chart class counts %s", counts)
            log.info("chart class weights %s",
                     {c: round(w, 2) for c, w in chart_cls_w.items()})
        _texts = TokenTexts(processor.tokenizer)
        _miss = [0]

        class _ChartTagger:
            """ids of one supervised run -> (per-token weight, per-token class id)."""
            weighted = False

            def __call__(self, ids):
                texts = _texts(ids)
                span = chart_span(texts)
                if span is None:     # no delimiter in the target -> nothing to tag, and say so
                    _miss[0] += 1
                    log.warning("chart tagging: no delimiter in target (%d so far)", _miss[0])
                    return (torch.ones(len(texts)),
                            torch.full((len(texts),), -1, dtype=torch.long))
                occ = class_occurrences(texts, span)
                w = (token_weights(texts, span, occ, args.chart_loss_weight, chart_cls_w)
                     if self.weighted else torch.ones(len(texts)))
                return w, class_id_tensor(len(texts), occ)

        chart_tagger = _ChartTagger()
        chart_tagger.weighted = weighted
        log.info("chart-concat ON: token weighting=%s (chart_loss_weight=%.3f, "
                 "class_weights=%s)", weighted, args.chart_loss_weight,
                 args.chart_class_weights)

    _collate = partial(collate_fn, processor=processor, resp_template_ids=resp_ids,
                       chart_tagger=chart_tagger)
    # family-weighted sampling toward the hidden test distribution. num_samples =
    # len(records) so steps_per_epoch (hence the cosine schedule and the six checkpoints)
    # is unchanged; only WHICH cases are drawn changes.
    g = torch.Generator().manual_seed(args.sampler_seed)
    sampler = None
    if args.family_weights:
        fw = _parse_family_weights(args.family_weights)
        weights = torch.tensor([fw.get(r["family"], 1.0) for r in records], dtype=torch.double)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(records),
                                        replacement=True, generator=g)
        log.info(f"family-weighted sampler: {fw}  (num_samples={len(records)})")
    # persistent_workers=False (MF1): workers respawn each epoch and so pick up set_epoch, which
    # is what makes the per-sample augmentation RNG actually vary across epochs.
    loader = DataLoader(train_ds, batch_size=args.batch,
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.workers, collate_fn=_collate,
                        pin_memory=False, drop_last=True,
                        persistent_workers=False, generator=g)
    if args.val_split:  # val_ds was built above whenever --val-split was given
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=_collate)

    # two param groups so the newly-adapted, more-fragile vision LoRA (attention +
    # merger) can train at a lower LR than the LM. A param is "vision" iff its module path
    # contains 'visual'. The cosine scheduler scales every group by the same factor.
    vis_lr = args.vision_lr if args.vision_lr is not None else args.lr
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    vis_params = [p for n, p in named if "visual" in n]
    lm_params = [p for n, p in named if "visual" not in n]
    log.info(f"trainable param tensors: {len(lm_params)} LM @ lr={args.lr}, "
             f"{len(vis_params)} vision @ lr={vis_lr}")
    # ── auxiliary per-tooth head ──────────────────────────────────────────
    # Built ONLY when the weight is positive. At 0 nothing below runs, so no module init
    # consumes the torch RNG and the run is bit-identical to the aux-head code path.
    pooler = head = None
    status_w = pos_w = None
    ignore_index = -100
    if aux_on:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "data_canonical"))
        from tooth_state import (tooth_state, N_SLOTS, N_STATUS, N_ATTR, STATUS_CLASSES,
                                 ATTRIBUTES, IGNORE_INDEX)
        from tooth_head import (VisionPooler, ToothHead, label_stats, loss_weights,
                                aux_loss)
        ignore_index = IGNORE_INDEX
        pooler = VisionPooler(model, model.config.vision_config.spatial_merge_size)
        head = ToothHead(pooler.dim, N_SLOTS, N_STATUS, N_ATTR,
                         hidden=args.aux_hidden).to(device=device, dtype=torch.float32)
        counts, attr_pos, attr_n = label_stats(records, tooth_state, N_SLOTS, N_STATUS, N_ATTR)
        status_w, pos_w = loss_weights(counts, attr_pos, attr_n,
                                       power=args.aux_class_weight_power)
        status_w, pos_w = status_w.to(device), pos_w.to(device)
        log.info("aux-head tooth head: hook at %s (dim=%d), %d params, aux_weight=%.3f aux_lr=%.1e",
                 pooler.path, pooler.dim,
                 sum(p.numel() for p in head.parameters()), args.aux_tooth_weight, args.aux_lr)
        log.info("aux-head status counts %s -> weights %s",
                 {c: int(n) for c, n in zip(STATUS_CLASSES, counts)},
                 {c: round(float(w), 2) for c, w in zip(STATUS_CLASSES, status_w)})
        log.info("aux-head attribute pos_weight %s",
                 {a: round(float(w), 1) for a, w in zip(ATTRIBUTES, pos_w)})

    groups = [{"params": lm_params, "lr": args.lr},
              {"params": vis_params, "lr": vis_lr}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": args.aux_lr})
    optim = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    log.info(f"opt steps: {total_steps}  warmup: {warmup}  steps/epoch: {steps_per_epoch}")

    # ⚠ Derive every renderer field from `args`. This string used to be hardcoded to
    # "five_views(DEFAULT_VIEWS)", which made fiveviews_reprv2_1's run_config.json claim the v1
    # representation for a --render-v2 run — and since infer_all_epochs.py takes the same flags
    # by hand, a wrong record here is a train/inference mismatch waiting to happen.
    run_config = {
        "model": args.model, "split": args.split, "n_records": len(records),
        "representation": (f"cross_section_video({args.video_axis}, rgb={args.video_rgb}, "
                           f"tile={args.video_tile}) + "
                           f"stills({'VIEWS_V2' if args.render_v2 else 'DEFAULT_VIEWS'}, "
                           f"tile={args.tile}, budget={_eff_still_budget(args)}, "
                           f"pano_split={args.pano_split})"),
        "render_v2": args.render_v2,
        "views": list(view_set),
        "video_rgb": args.video_rgb,
        "tile": args.tile,
        "video_tile": args.video_tile,
        "still_token_budget": _eff_still_budget(args),
        "pano_split": args.pano_split,
        "still_mpx": args.still_mpx,
        "still_per_view": args.still_per_view,
        "processor_max_pixels": int(args.still_mpx * 1e6),
        "video_axis": args.video_axis,
        "canonical_target": args.canonical,
        "target": ("chart_v2" if args.chart_v2 else
                   "chart_concat" if args.chart_concat else
                   "canonical_prose" if args.canonical else "raw_en_reports"),
        "chart_concat": args.chart_concat,
        "chart_loss_weight": args.chart_loss_weight,
        "chart_class_weights": args.chart_class_weights,
        # chart-v2 / non-tooth weighting levers. Added 2026-08-14; `chartv2_pano_v1`'s run_config
        # predates them and does not record which loss levers were armed.
        "chart_v2": args.chart_v2,
        "chart_v2_class_weights": args.chart_v2_class_weights,
        "chart_v2_weight_power": args.chart_v2_weight_power,
        "nontooth_class_weights": args.nontooth_class_weights,
        "nontooth_weight_power": args.nontooth_weight_power,
        "nontooth_weight_cap": args.nontooth_weight_cap,
        "max_cases": args.max_cases, "frames": args.frames, "batch": args.batch,
        "grad_accum": args.grad_accum, "effective_batch": args.batch * args.grad_accum,
        "lr": args.lr, "vision_lr": vis_lr, "max_steps": args.max_steps, "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio, "init_from": args.init_from,
        "total_opt_steps": total_steps, "warmup_steps": warmup,
        "seed": args.seed, "sampler_seed": args.sampler_seed,
        "family_weights": args.family_weights,
        "augment": args.augment, "mirror_p": args.mirror_p, "z_jitter_mm": args.z_jitter_mm,
        "aux_tooth_weight": args.aux_tooth_weight, "aux_lr": args.aux_lr,
        "aux_hidden": args.aux_hidden,
        "aux_hook": pooler.path if pooler is not None else None,
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
    cw.writerow(["step", "loss", "lm_loss", "aux_loss", "lr", "grad_norm", "sec_per_step"])

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train(); optim.zero_grad()
    if head is not None:
        head.train()
    global_step, done = 0, False
    accum_loss, accum_lm, accum_aux, accum_n = 0.0, 0.0, 0.0, 0
    # per-class hit/total/predicted counts over the logging window. Logged as recall so an
    # all-PRESENT head is visible immediately instead of hiding inside a falling aux_loss.
    aux_hit = aux_tot = aux_pred = None
    if head is not None:
        aux_hit = torch.zeros(N_STATUS, device=device)
        aux_tot = torch.zeros(N_STATUS, device=device)
        aux_pred = torch.zeros(N_STATUS, device=device)
    # the same idea for the chart segment. Chart recall for `absent` staying near zero
    # while the loss falls IS the all-present collapse, and the loss alone cannot show it.
    chart_hit = torch.zeros(max(len(CHART_CLASSES), 1), device=device)
    chart_tot = torch.zeros(max(len(CHART_CLASSES), 1), device=device)
    t_step = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainable_for_clip = [p for p in model.parameters() if p.requires_grad]
    # the head is clipped SEPARATELY, not folded into the model's clip. Measured: at unit aux
    # weight the head's gradient norm is 7.6 against 2.7 for the whole adapted model, so a joint
    # clip at 1.0 would scale the LM+vision update down by 0.47x at aux_weight=0.5 — i.e. adding
    # the head would quietly halve the LM's effective learning rate, degrade the LM loss for a
    # reason that has nothing to do with the auxiliary task, and make the run uninterpretable.
    # Clipping the two groups independently keeps the LM update identical to the aux-off run.
    head_for_clip = list(head.parameters()) if head is not None else []

    for epoch in range(1, args.epochs + 1):
        if done:
            break
        train_ds.set_epoch(epoch)   # MF1: vary the augmentation RNG per epoch
        # drop any grad accumulated from the previous epoch's tail micro-batches
        # (len(loader) is not divisible by grad_accum), which would otherwise leak into
        # this epoch's first optimizer step. Reset the running loss too.
        optim.zero_grad(); accum_loss, accum_lm, accum_aux, accum_n = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            tooth = _pop_tooth(batch)
            # The pooler is armed ONLY around this forward. Every other forward in the process —
            # the end-of-epoch val loop above all — must leave the hook inert, or its captures
            # accumulate and land on the next training step.
            tw = tooth.pop("chart_token_weights", None)
            cid = tooth.pop("chart_class_ids", None)
            with (pooler.capturing() if pooler is not None else contextlib.nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if tw is None:
                        out_lm = model(**batch)
                        lm_loss = out_lm.loss
                    else:
                        # our own CE so the chart segment can carry a different weight.
                        # Labels are pulled out first — letting the model compute its CE too
                        # would double the largest activation in the step for nothing.
                        lbl = batch.pop("labels")
                        out_lm = model(**batch)
                        lm_loss = weighted_ce(out_lm.logits, lbl, tw)
                        batch["labels"] = lbl
            if cid is not None:
                # teacher-forced per-class chart recall. Position j predicts token j+1, so
                # a class id at position j is scored against the argmax at j-1. Detached and
                # counted only — it never touches the loss.
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
            aux = torch.zeros((), device=device)
            if head is not None:
                # take() asserts the capture's shape against the processor grids before the head
                # ever sees it.
                status_logits, attr_logits = head(pooler.take(batch))
                aux, parts = aux_loss(status_logits, attr_logits, tooth,
                                      status_w, pos_w, ignore_index)
                aux_hit += parts["hit"]; aux_tot += parts["tot"]; aux_pred += parts["pred"]
            loss = (lm_loss + args.aux_tooth_weight * aux) / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            accum_lm += lm_loss.item()
            accum_aux += float(aux)
            accum_n += 1

            if (step + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_for_clip, 1.0).item()
                gnorm_head = (torch.nn.utils.clip_grad_norm_(head_for_clip, 1.0).item()
                              if head_for_clip else 0.0)
                lr_now = sched.get_last_lr()[0]        # LR used for THIS step, pre-advance
                optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
                sps = (time.time() - t_step) / args.grad_accum
                t_step = time.time()

                if global_step % args.log_every == 0:
                    avg = accum_loss / accum_n
                    avg_lm, avg_aux = accum_lm / accum_n, accum_aux / accum_n
                    # index, not [-1]: the aux-head is a THIRD group, so [-1] would silently
                    # start reporting the head's LR under the vision label.
                    lr_vis = sched.get_last_lr()[1]
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    aux_txt = f"  lm={avg_lm:.4f} aux={avg_aux:.4f}" if head is not None else ""
                    log.info(f"step={global_step:4d}/{total_steps}  loss={avg:.4f}{aux_txt}  "
                             f"lr={lr_now:.2e}  lr_vis={lr_vis:.2e}  gnorm={gnorm:.2f}  "
                             f"{sps:.1f}s/step  peakVRAM={peak:.1f}GB")
                    cw.writerow([global_step, round(avg, 5), round(avg_lm, 5),
                                 round(avg_aux, 5), lr_now, round(gnorm, 3),
                                 round(sps, 2)]); cf.flush()
                    wlog = {"train/loss": avg, "train/lm_loss": avg_lm,
                            "train/lr": lr_now, "train/lr_vision": lr_vis,
                            "train/grad_norm": gnorm, "perf/sec_per_step": sps,
                            "perf/peak_vram_gb": peak, "epoch": epoch}
                    if head is not None:
                        # aux_loss AND per-class recall: a head that predicts PRESENT for every
                        # slot drives aux_loss down and recall for the other three to 0, and
                        # only the second of those is visible.
                        wlog["train/aux_loss"] = avg_aux
                        wlog["train/lr_aux"] = sched.get_last_lr()[2]
                        # separate from train/grad_norm, which must stay comparable to the
                        # aux-off runs — that is the whole point of clipping them apart.
                        wlog["train/grad_norm_head"] = gnorm_head
                        rec_txt = []
                        for c, name in enumerate(STATUS_CLASSES):
                            r = float(aux_hit[c] / aux_tot[c]) if aux_tot[c] > 0 else float("nan")
                            wlog[f"aux/recall_{name}"] = r
                            wlog[f"aux/predicted_{name}"] = float(aux_pred[c])
                            rec_txt.append(f"{name[:4]}={r:.2f}")
                        log.info("   aux recall: %s  (n=%s)", " ".join(rec_txt),
                                 [int(x) for x in aux_tot])
                        aux_hit.zero_(); aux_tot.zero_(); aux_pred.zero_()
                    if args.chart_concat or args.chart_v2:
                        rec_txt = []
                        for c, name in enumerate(CHART_CLASSES):
                            r = (float(chart_hit[c] / chart_tot[c])
                                 if chart_tot[c] > 0 else float("nan"))
                            wlog[f"chart/recall_{name.replace(' ', '_')}"] = r
                            wlog[f"chart/n_{name.replace(' ', '_')}"] = float(chart_tot[c])
                            if c < n_tooth_classes:
                                rec_txt.append(f"{name.split()[0][:4]}={r:.2f}")
                        log.info("   chart recall: %s  (n=%s)", " ".join(rec_txt),
                                 [int(x) for x in chart_tot[:n_tooth_classes]])
                        # the non-tooth lines, aggregated PER FIELD. Per (field, value)
                        # is 44 numbers and unreadable in a log line; the field mean is the one
                        # that would have shown ep7's collapse (a field pinned to its majority
                        # value reads ~1.00 on that value and ~0.00 on every other).
                        if nontooth_fields_idx:
                            nt_txt = []
                            for field, idxs in nontooth_fields_idx:
                                tot = float(sum(chart_tot[i] for i in idxs))
                                hit = float(sum(chart_hit[i] for i in idxs))
                                r = hit / tot if tot > 0 else float("nan")
                                wlog[f"chart/recall_field_{field}"] = r
                                nt_txt.append(f"{NONTOOTH_ABBR[field]}={r:.2f}")
                            log.info("   non-tooth recall (by field): %s", " ".join(nt_txt))
                        chart_hit.zero_(); chart_tot.zero_()
                    if wb is not None:
                        wb.log(wlog, step=global_step)
                    accum_loss, accum_lm, accum_aux, accum_n = 0.0, 0.0, 0.0, 0

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
            _save_head(head, pooler, args, ep_dir)
            cf.flush()
            # end-of-epoch val loss so the epoch checkpoints are directly comparable
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

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    _save_head(head, pooler, args, adapter_dir)
    log.info(f"adapter -> {adapter_dir}")
    log.info(f"loss curve -> {loss_csv}")

    if wb is not None:
        wb.summary["final_opt_steps"] = global_step
        wb.summary["peak_vram_gb"] = peak
        wb.finish()


if __name__ == "__main__":
    main()
