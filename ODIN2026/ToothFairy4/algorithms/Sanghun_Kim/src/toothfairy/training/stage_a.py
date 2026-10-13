"""Perception-stage trainer — the aggregation encoder and the findings head (stage 1).

The stage adapts perception through **claim classification alone**; no language model is loaded,
which is what makes it cheap (about a minute per epoch). Both segmentation networks stay frozen,
so their features are read from the cache built by `toothfairy.cli.cache` and the only trainable
parameters are the 0.41M aggregator and the head. Once this stage has finished the encoder is
frozen too, and decoding runs on top of it through the prefix and LLM LoRA stages.

Selection metric: **validation classification cross-entropy (l_gate)** — selection by validation
CE, reporting by argmax F1.

reproducibility: seed, git commit, env / requirements, resolved config and split hash are all
recorded (shared with trainer.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_config, ReportModelConfig
from ..models import FindingsModel
from ..data import ReportDataset, make_collate, read_split
from ..schema.claims import compute_class_weights
from .trainer import REPO, seed_everything, snapshot_repro, to_device


# --- data ------------------------------------------------------------------ #
def build_loaders(cfg: ReportModelConfig):
    """train / val loaders over the cached perception features."""
    train_pids = read_split(REPO / "data/splits/train.txt")
    val_pids = read_split(REPO / "data/splits/val.txt")
    train_ds = ReportDataset(train_pids, cfg, repo_root=REPO, tokenizer=None, resample=True)
    val_ds = ReportDataset(val_pids, cfg, repo_root=REPO, tokenizer=None, resample=False)
    for ds, name in ((train_ds, "train"), (val_ds, "val")):
        if len(ds) == 0:
            raise FileNotFoundError(
                f"no samples for {name} split — one of the three paths is wrong:\n"
                f"  cache     = {ds.feat_dir}\n  dense (claim labels) = {ds.dense_dir}\n"
                f"  anat (region text) = {ds.anat_dir}")
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    seed = cfg.train.seed

    def worker_init(wid):                        # same seeding convention as trainer.py
        np.random.seed(seed + wid)
        random.seed(seed + wid)

    collate = make_collate(cfg.perception.feat_dim)
    common = dict(collate_fn=collate, num_workers=2, worker_init_fn=worker_init, generator=g)
    bs = cfg.train.batch_size
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False, **common)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **common)
    return train_ld, val_ld, len(train_ds), len(val_ds)


# --- eval / checkpoint ------------------------------------------------------ #
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Sample-weighted means. `loss` = the classification CE, which is the selection objective."""
    model.eval()
    agg: dict[str, float] = defaultdict(float)
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        b = len(batch["pid"])
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 0:
                agg[k] += v.item() * b
        n += b
    res = {k: v / max(n, 1) for k, v in agg.items()}
    res["loss"] = res.get("l_gate", float("nan"))
    return res


def save_checkpoint(path: Path, model, cfg: ReportModelConfig, metrics: dict) -> None:
    """Saves the trainable parameters of the classifier (aggregation encoder + findings head).
    The next stage picks this up through `init_from`."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
    torch.save({"trainable_state": state, "metrics": metrics,
                "run_name": cfg.train.run_name}, path)


# --- train loop ------------------------------------------------------------- #
def train(cfg: ReportModelConfig, device, out_dir: Path, wandb_run, max_steps: int = 0) -> dict:
    model = FindingsModel(cfg).to(device)
    train_ld, val_ld, n_tr, n_val = build_loaders(cfg)
    train_ds = train_ld.dataset

    if cfg.train.use_siglip:
        # **Inject** the text encoder: the trainer builds the frozen PubMedBERT and attaches it
        # to the model. The model does not build it itself so that runs without SigLIP are not
        # dragged through a heavy HF load (same convention as in trainer.py).
        from ..losses.contrastive import FrozenTextEncoder
        model.text_encoder = FrozenTextEncoder(cfg.train.siglip_text_encoder, device=device,
                                               cache=True).to(device)
        if model.text_encoder.dim != cfg.train.siglip_text_dim:
            raise ValueError(f"siglip_text_dim={cfg.train.siglip_text_dim} != "
                             f"encoder dim {model.text_encoder.dim}")
        # WARNING: SigLIP aligns to the **region text** in anat_dir. If that path is empty or
        # wrong, `_siglip_loss` silently returns 0 without raising and the run looks healthy —
        # the kind of failure that burns days while you believe SigLIP is on. Pull a real sample
        # here to rule it out. (This stage otherwise reads its claim labels from dense_dir, so a
        # stale anat_dir has no effect at all until SigLIP is switched on.)
        anat = REPO / cfg.train.anat_dir if not Path(cfg.train.anat_dir).is_absolute() \
            else Path(cfg.train.anat_dir)
        if not anat.is_dir() or not any(anat.glob("*.json")):
            raise FileNotFoundError(
                f"use_siglip=true but anat_dir holds no GT: {anat} — SigLIP aligns to this text, "
                "so an empty directory makes the loss silently 0")
        probe = train_ds[0].get("region_targets_text") or {}
        n_txt = sum(1 for v in probe.values() if v and v.strip())
        if n_txt < 2:
            raise ValueError(
                f"the first sample of anat_dir={anat} has only {n_txt} region texts, which is "
                "not enough for a pairwise SigLIP loss (below 2 the loss is 0). Check the GT "
                "path.")
        print(f"[stage-a] SigLIP on (text={cfg.train.siglip_text_encoder} "
              f"dim={model.text_encoder.dim} λ={cfg.train.lambda_siglip} · "
              f"anat={cfg.train.anat_dir} · {n_txt} texts in the first sample)", flush=True)


    if cfg.train.gate_class_weight:                       # effective-number (beta), as in decode
        ds = train_ld.dataset
        dense_paths = [ds.dense_dir / Path(ap).name for _, ap in ds.samples]
        model.class_weights = compute_class_weights(
            dense_paths, beta=cfg.train.gate_class_weight_beta)
        print(f"[stage-a] findings class-weight on (β={cfg.train.gate_class_weight_beta}, "
              f"{len(dense_paths)} reports)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    print(f"[stage-a] train={n_tr} val={n_val} "
          f"trainable={sum(p.numel() for p in params)/1e6:.2f}M lr={cfg.train.lr}", flush=True)

    accum = max(cfg.train.grad_accum, 1)
    patience = cfg.train.early_stopping_patience
    clip = 1.0                                  # gradient clipping, as in the other trainers
    best = {"loss": float("inf"), "epoch": -1}
    epochs_since_best = 0
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        if hasattr(train_ld.dataset, "set_epoch"):
            train_ld.dataset.set_epoch(epoch)
        t0 = time.time()
        run: dict[str, float] = defaultdict(float)
        nb = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_ld):
            batch = to_device(batch, device)
            out = model(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):         # a silent NaN/Inf would waste days of training
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} step {step} pid={batch['pid'][0]}: "
                    f"l_gate={float(out['l_gate'])}")
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            run["loss"] += float(loss.detach())
            run["l_gate"] += float(out["l_gate"].detach())
            nb += 1
            step += 1
            if max_steps and step >= max_steps:
                break
        if nb % accum:                                    # apply the trailing gradients
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        tr = {k: v / max(nb, 1) for k, v in run.items()}
        val = evaluate(model, val_ld, device)
        dt = time.time() - t0
        sig = (f" l_siglip {val['l_siglip']:.4f}" if 'l_siglip' in val else "")
        print(f"[stage-a] epoch {epoch} train l_gate {tr['l_gate']:.4f}{sig} "
              f"| val l_gate {val.get('l_gate', float('nan')):.4f} "
              f"obj {val['loss']:.4f} | {dt:.0f}s", flush=True)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **{f"train/{k}": v for k, v in tr.items()},
                           **{f"val/{k}": v for k, v in val.items()}, "sec_per_epoch": dt})

        score = val["loss"]                     # validation classification CE
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite val objective at epoch {epoch}: {val}")
        if score < best["loss"]:
            best = {"loss": score, "epoch": epoch, **{f"val_{k}": v for k, v in val.items()},
                    **{f"train_{k}": v for k, v in tr.items()}}
            epochs_since_best = 0
            if not max_steps:
                save_checkpoint(out_dir / "best.pt", model, cfg, best)
        else:
            epochs_since_best += 1
        if not max_steps:
            save_checkpoint(out_dir / "last.pt", model, cfg, {"epoch": epoch, **val})
        if patience and epochs_since_best >= patience:
            print(f"[stage-a] early stop at epoch {epoch} (best {best.get('epoch', -1)} "
                  f"val obj {best['loss']:.4f})", flush=True)
            break
        if max_steps and step >= max_steps:
            if torch.cuda.is_available():
                print(f"[dry-run] peak CUDA mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
                      f"(reserved {torch.cuda.max_memory_reserved()/1e9:.2f} GB)", flush=True)
            print("[stage-a] max_steps reached (dry-run); stopping.", flush=True)
            break
    return best


def main():
    ap = argparse.ArgumentParser(description="perception stage: claim-classification training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 = dry-run (no done.flag)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    if (out_dir / "done.flag").exists():                  # never overwrite a finished run
        print(f"[stage-a] {out_dir}/done.flag exists — skipping (finished run protected).",
              flush=True)
        return
    meta = snapshot_repro(out_dir, cfg)
    meta.update({"stage": "A"})
    print(f"[stage-a] repro: {json.dumps(meta, ensure_ascii=False)}", flush=True)

    wandb_run = None
    if args.wandb:
        import os
        import wandb
        wandb_run = wandb.init(project="toothfairy",
                               entity=os.environ.get("WANDB_ENTITY") or None,
                               name=cfg.train.run_name or "stage-a",
                               config={"config": args.config})

    best = train(cfg, device, out_dir, wandb_run, args.max_steps)
    (out_dir / "metrics.json").write_text(
        json.dumps({**meta, "best": best}, ensure_ascii=False, indent=2))
    if not args.max_steps:
        (out_dir / "done.flag").write_text(f"best epoch {best.get('epoch','?')} "
                                           f"val obj {best.get('loss', float('nan')):.4f}\n")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
