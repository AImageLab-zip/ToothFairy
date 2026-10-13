"""Trainer — align→generate 2-stage (frozen backbone), reproducible.

Trains ReportModel on top of the cached perception features (region feature cache). Two stages:
stage1 = projector **align** (LoRA frozen, freeze_llm_epochs epochs, optional val-plateau early
stop) → stage2 = **generate** (LoRA on). The findings head and the contrastive loss keep training
through both stages. On a stage switch the previous stage's best.pt is reloaded, so the next
stage starts from the lowest-val weights.

Reproducibility: fixed seed, git commit, env/requirements, resolved config dump and split hashes
are written into the experiment dir. The checkpoint holds the trainable modules
(encoder/head/projector) plus every LoRA tensor (the frozen LLM base is reloaded from HF).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ..config import load_config, dump_config, ReportModelConfig
from ..models import ReportModel
from ..data import ReportDataset, make_collate, read_split
from ..schema.claims import compute_class_weights
from ..losses import FrozenTextEncoder
from . import ddp

from ..paths import REPO  # noqa: E402


# --- reproducibility ------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256_file(path: Path, n: int = 16) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def snapshot_repro(out_dir: Path, cfg: ReportModelConfig) -> dict:
    """env / requirements / resolved config + hashes; returns the meta dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # The env lock must capture the environment of the
    # **running interpreter**. A venv created by `uv venv` has no pip module, so
    # `python -m pip freeze` fails there → fall back to uv (same prefix passed as VIRTUAL_ENV).
    freeze = None
    prefix = sys.prefix
    for cmd, env in ((["python", "-m", "pip", "freeze"], None),
                     (["uv", "pip", "freeze"], {**os.environ, "VIRTUAL_ENV": prefix})):
        if cmd[0] == "python":
            cmd = [sys.executable, *cmd[1:]]
        try:
            freeze = subprocess.check_output(cmd, cwd=REPO, env=env,
                                             stderr=subprocess.DEVNULL)
            break
        except Exception:  # noqa: BLE001
            continue
    (out_dir / "requirements.txt").write_bytes(
        freeze if freeze else b"# pip/uv freeze failed\n")
    try:                                                    # with conda present, keep env.yaml
        (out_dir / "env.yaml").write_bytes(
            subprocess.check_output(["conda", "env", "export", "--no-builds"], cwd=REPO))
    except Exception:  # noqa: BLE001 — venv: record interpreter/accelerator info instead
        (out_dir / "env.yaml").write_text(
            "# no conda on this host (venv). requirements.txt = pip freeze of the interpreter below.\n"
            f"python: {sys.version.split()[0]}\n"
            f"executable: {sys.executable}\n"
            f"torch: {torch.__version__}\n"
            f"cuda: {torch.version.cuda}\n"
            f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")
    dump_config(cfg, out_dir / "config.resolved.yaml")
    return {
        "git_commit": _git_commit(),
        "split_train_hash": _sha256_file(REPO / "data/splits/train.txt"),
        "split_val_hash": _sha256_file(REPO / "data/splits/val.txt"),
        "torch": torch.__version__,
        "seed": cfg.train.seed,
        "run_name": cfg.train.run_name or "report-model",
        "findings_head": cfg.train.findings_head,
        "use_siglip": cfg.train.use_siglip,
        "backbone": cfg.perception.backbone,
    }


# --- batch device move ----------------------------------------------------- #
def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "regions":
            out[k] = {c: {kk: t.to(device) for kk, t in d.items()} for c, d in v.items()}
        elif k == "teeth":
            out[k] = {a: {kk: t.to(device) for kk, t in d.items()} for a, d in v.items()}
        elif k == "teeth_tf2":                                 # hybrid backbone per-tooth vectors
            out[k] = {a: {kk: t.to(device) for kk, t in d.items()} for a, d in v.items()}
        elif k == "nonteeth_labels":
            out[k] = {kk: t.to(device) for kk, t in v.items()}
        elif k == "region_targets":                           # list[dict[str, LongTensor]]
            out[k] = [{rid: t.to(device) for rid, t in d.items()} for d in v]
        elif k == "report_ids":                               # joint: list[LongTensor|None]
            out[k] = [t.to(device) if torch.is_tensor(t) else t for t in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _set_lora_trainable(model, on: bool) -> int:
    """Toggle requires_grad on LoRA (peft) params: off in align (connector only), on in generate."""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = on
            n += 1
    return n


# --- curriculum stages ------------------------------------------------------ #
# Which modules each stage **trains**; everything else is frozen (the LLM base is never trained).
# soft prefix = realizer.projector (+ region_type_emb / prob_proj, part of the visual prefix).
_PREFIX_MODULES = ("realizer.projector.", "realizer.region_type_emb", "realizer.prob_proj.")
STAGE_TRAINABLE: dict[str, tuple[tuple[str, ...], bool]] = {
    # stage: (name prefixes of the parameters to train, whether LoRA is trained)
    "prefix": (_PREFIX_MODULES, False),                        # prefix only
    "perception_prefix": (("encoder.",) + _PREFIX_MODULES, False),   # perception + prefix
    "qlora": ((), True),                                       # LLM LoRA only
    "prefix_qlora": (_PREFIX_MODULES, True),                   # prefix + LLM LoRA (encoder frozen)
}


def _build_optimizer(model, cfg):
    """Build AdamW over the trainable parameters. `prefix_lr`>0 puts the prefix in its own group.

    Why split: `prefix_qlora` trains the projector and LoRA together, but the projector has
    already converged in the prefix stage while LoRA starts from scratch. One shared lr shakes
    the projector hard and can re-flatten the soft prefix, a collapse mode seen repeatedly in
    this project. With prefix_lr=0 no group is split and the behaviour is unchanged.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    plr = float(getattr(cfg.train, "prefix_lr", 0.0) or 0.0)
    is_pre = lambda n: any(n.startswith(m) for m in _PREFIX_MODULES)   # noqa: E731
    if plr <= 0:
        groups = [{"params": [p for _, p in named], "lr": cfg.train.lr}]
    else:
        pre = [p for n, p in named if is_pre(n)]
        rest = [p for n, p in named if not is_pre(n)]
        groups = [g for g in ({"params": rest, "lr": cfg.train.lr},
                              {"params": pre, "lr": plr}) if g["params"]]
        print(f"[train] split lr — prefix {sum(p.numel() for p in pre)/1e6:.2f}M @{plr:g} · "
              f"rest {sum(p.numel() for p in rest)/1e6:.2f}M @{cfg.train.lr:g}", flush=True)
    return torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def _apply_stage_freeze(model, stage: str) -> tuple[float, list[str]]:
    """Set requires_grad for the stage. Returns (trainable params in millions, module list)."""
    prefixes, lora_on = STAGE_TRAINABLE[stage]
    n_train = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = lora_on
        else:
            p.requires_grad = any(name.startswith(pre) for pre in prefixes)
        if p.requires_grad:
            n_train += p.numel()
    groups = list(prefixes) + (["lora_*"] if lora_on else [])
    return n_train / 1e6, groups


def _load_init_from(model, path: str, device) -> None:
    """Seed weights from the previous stage checkpoint (strict=False); fails loudly on a break."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        raise FileNotFoundError(f"init_from checkpoint not found: {p}")
    ck = torch.load(p, map_location=device)
    sd = ck.get("trainable_state", ck)
    # A perception-stage checkpoint carries findings_head weights, while later stages build no
    # head (injection off). Drop such intended mismatches; fail loudly on any other one.
    model_keys = set(model.state_dict())
    usable = {k: v for k, v in sd.items() if k in model_keys}
    dropped = [k for k in sd if k not in model_keys]
    # `contrastive.` is an auxiliary head that exists only in the perception stage with SigLIP on
    # (it aligns anatomical tokens with region text) — scaffolding for that stage's objective,
    # not part of the perception path. Later stages legitimately lack it, so dropping it is fine.
    _OPTIONAL = ("findings_head.", "contrastive.")
    unexpected_other = [k for k in dropped if not k.startswith(_OPTIONAL)]
    if dropped and not unexpected_other:
        print(f"[train] init_from: dropped {len(dropped)} stage-only keys "
              f"({sorted({k.split('.')[0] for k in dropped})})", flush=True)
    if unexpected_other:
        raise RuntimeError(f"init_from keys not in model (arch mismatch): {unexpected_other[:5]}")
    shape_bad = [k for k, v in usable.items() if tuple(v.shape) != tuple(model.state_dict()[k].shape)]
    if shape_bad:
        raise RuntimeError(f"init_from shape mismatch: {shape_bad[:5]}")
    model.load_state_dict(usable, strict=False)
    n_enc = sum(1 for k in usable if k.startswith("encoder."))
    n_proj = sum(1 for k in usable if k.startswith("realizer.projector."))
    n_lora = sum(1 for k in usable if "lora_" in k)
    print(f"[train] init_from {p.name}: loaded {len(usable)} tensors "
          f"(encoder {n_enc} · projector {n_proj} · lora {n_lora})"
          + (f" · ignored {len(dropped)} head keys (stage has no head)" if dropped else "")
          + f" · run={ck.get('run_name','?')}", flush=True)
    if n_enc == 0:
        raise RuntimeError(f"init_from has no encoder weights — the stage chain is broken: {p}")


def _stage_advance(elapsed: int, cap: int, early: bool, patience: int, since: int) -> tuple[bool, str]:
    """Decide when the align stage ends. early=True: val plateau (patience) or cap; False: cap."""
    if early and patience and since >= patience:
        return True, "val plateau"
    if elapsed >= cap:
        return True, f"cap {cap} epoch"
    return False, ""


# --- data ------------------------------------------------------------------ #
def build_loaders(cfg: ReportModelConfig, tokenizer, dist=None):
    """train/val loaders. With `dist` on, the data is split per rank.

    train uses DistributedSampler (it pads; `set_epoch` keeps the shuffle in sync), val is a
    **stride shard without padding** — padding val would let duplicated samples enter the mean
    twice and make the metric wobble.
    """
    train_pids = read_split(REPO / "data/splits/train.txt")
    val_pids = read_split(REPO / "data/splits/val.txt")
    train_ds = ReportDataset(train_pids, cfg, repo_root=REPO, tokenizer=tokenizer, resample=True)
    val_ds = ReportDataset(val_pids, cfg, repo_root=REPO, tokenizer=tokenizer, resample=False)
    for ds, name in ((train_ds, "train"), (val_ds, "val")):
        if len(ds) == 0:
            raise FileNotFoundError(
                f"no cached region features / GT for {name} split under {ds.feat_dir}. "
                f"Run toothfairy.cli.cache first.")
    seed = cfg.train.seed
    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init(wid):
        np.random.seed(seed + wid)
        random.seed(seed + wid)

    collate = make_collate(cfg.perception.feat_dim)
    common = dict(collate_fn=collate, num_workers=2, worker_init_fn=worker_init,
                  generator=g, persistent_workers=False)
    on = bool(dist and dist.enabled)
    sampler = (DistributedSampler(train_ds, num_replicas=dist.world_size, rank=dist.rank,
                                  shuffle=True, drop_last=False, seed=seed) if on else None)
    train_ld = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          shuffle=(sampler is None), sampler=sampler,
                          drop_last=False, **common)
    val_shard = (Subset(val_ds, list(range(dist.rank, len(val_ds), dist.world_size)))
                 if on else val_ds)
    val_ld = DataLoader(val_shard, batch_size=cfg.train.batch_size, shuffle=False,
                        drop_last=False, **common)
    return train_ld, val_ld, len(train_ds), len(val_ds)


@torch.no_grad()
def evaluate(model, loader, device, dist=None) -> dict:
    """Sample-weighted mean of every scalar the model returns. With `dist` on the ranks are
    **summed first** and then divided — per-shard means would make best/early-stop diverge."""
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
    if dist and dist.enabled:
        sums, total = ddp.reduce_scalar_dict(dist, dict(agg), float(n))
        return {k: v / max(total, 1.0) for k, v in sums.items()}
    return {k: v / max(n, 1) for k, v in agg.items()}


def _is_llm_base(name: str) -> bool:
    """Is this an LLM base weight (reloaded from HF)? LoRA tensors are not base weights."""
    return name.startswith("realizer.llm.") and "lora_" not in name


def _atomic_save(obj, path: Path) -> None:
    """Write to a temp file and rename — a power cut mid-save cannot corrupt the existing file.

    A power cut did interrupt a prefix-stage run once (the file survived). rename is atomic
    within one filesystem, so the worst case leaves the previous epoch's state intact.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)          # do not leave a half-written temp file behind
        raise


def save_resume(path: Path, model, cfg: ReportModelConfig, opt, state: dict) -> None:
    """Restart state — weights + **optimizer** + progress counters + RNG.

    best.pt/last.pt hold weights only and cannot resume a run: without the AdamW moments, the
    epoch and the early-stop counters the trajectory breaks. This single file continues it.
    """
    keep = {n for n, _ in model.named_parameters() if not _is_llm_base(n)}
    payload = {
        "trainable_state": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep},
        "optimizer": opt.state_dict(),
        "state": state,                       # epoch / best / early-stop / stage counters
        "run_name": cfg.train.run_name,
        "rng": {                              # reproduces data order and dropout
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    _atomic_save(payload, path)


def _restore_rng(rng: dict) -> None:
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:  # noqa: BLE001 — a failed RNG restore is not fatal (warn only)
        print(f"[resume][WARN] RNG restore failed ({type(e).__name__}: {e}); continuing", flush=True)


def save_checkpoint(path: Path, model, cfg: ReportModelConfig, metrics: dict) -> None:
    # Save **everything except the LLM base** (encoder, findings head, projector,
    # region_type_emb, prob_proj, contrastive + ALL LoRA). Selecting by requires_grad would drop
    # the modules frozen in that stage (e.g. the encoder during the prefix stage) and break the
    # init_from chain into the next stage. LoRA is always saved regardless of requires_grad too:
    # if it were missing, loading would silently evaluate a base-init LoRA (lora_B = 0).
    keep = {n for n, _ in model.named_parameters() if not _is_llm_base(n)}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
    _atomic_save({"trainable_state": state, "metrics": metrics,
                  "run_name": cfg.train.run_name}, path)      # survives a power cut mid-save


# --- train loop (align → generate 2-stage) --------------------------------- #
def train(cfg: ReportModelConfig, device, out_dir: Path, wandb_run, max_steps: int = 0,
          dist=None,
          resume: bool = False) -> dict:
    model = ReportModel(cfg).to(device)
    tokenizer = model.realizer.tokenizer
    train_ld, val_ld, n_tr, n_val = build_loaders(cfg, tokenizer, dist)

    # findings class weights (effective number; only when findings_head is on)
    if cfg.train.findings_head and cfg.train.gate_class_weight:
        ds = train_ld.dataset
        dense_paths = [ds.dense_dir / Path(ap).name for _, ap in ds.samples]
        model.class_weights = compute_class_weights(
            dense_paths, beta=cfg.train.gate_class_weight_beta)
        print(f"[train] findings class-weight on (β={cfg.train.gate_class_weight_beta}, "
              f"{len(dense_paths)} reports)", flush=True)

    # false-EOS region-type weight: normal EOS slots dominate → up-weight the finding slots
    n_normal, n_finding = train_ld.dataset.region_type_counts()
    beta, lam = cfg.train.false_eos_weight_beta, cfg.train.false_eos_lambda

    def _en(n):                                                # effective-number raw weight
        return (1.0 - beta) / (1.0 - beta ** n) if n > 0 else None

    wn, wf = _en(n_normal), _en(n_finding)
    w_finding = (wf / wn) * lam if (wn and wf) else lam        # multiple of normal=1.0, times λ
    model.region_type_weights = {"normal": 1.0, "finding": w_finding}
    print(f"[train] false-EOS region-weight: normal={n_normal} finding={n_finding} "
          f"→ w_finding={w_finding:.3f} (β={beta}, λ={lam})", flush=True)
    # SigLIP text encoder (frozen; only when use_siglip is on)
    if cfg.train.use_siglip:
        model.text_encoder = FrozenTextEncoder(cfg.train.siglip_text_encoder, device=device,
                                 max_len=cfg.train.siglip_text_max_len)
        if model.text_encoder.dim != cfg.train.siglip_text_dim:
            raise ValueError(f"siglip_text_dim {cfg.train.siglip_text_dim} != "
                             f"encoder dim {model.text_encoder.dim}")
        print(f"[train] SigLIP on (text={cfg.train.siglip_text_encoder} "
              f"dim={model.text_encoder.dim}, λ={cfg.train.lambda_siglip})", flush=True)

    patience = cfg.train.early_stopping_patience
    s_early = cfg.train.stage_early_stop
    s_patience = cfg.train.stage_patience or patience
    stage_name = cfg.train.stage

    if cfg.train.init_from:                    # seed from the previous stage's weights
        _load_init_from(model, cfg.train.init_from, device)

    if stage_name == "align_generate":
        # two-stage schedule: align (LoRA frozen, freeze_llm_epochs) → generate (LoRA on)
        freeze = cfg.train.freeze_llm_epochs
        if freeze >= cfg.train.epochs:
            raise ValueError(f"freeze_llm_epochs({freeze}) >= epochs({cfg.train.epochs}): "
                             "the generate stage would never run.")
        in_align = freeze > 0
        if in_align:
            n_lora = _set_lora_trainable(model, False)
            print(f"[train] align stage: LoRA frozen ({n_lora} tensors)", flush=True)
        stage_label = "align" if in_align else "generate"
    else:
        # single curriculum stage: no stage switch, train only the modules named for the stage
        if stage_name not in STAGE_TRAINABLE:
            raise ValueError(f"unknown train.stage={stage_name!r} — expected one of "
                             f"{['align_generate'] + list(STAGE_TRAINABLE)}")
        freeze = 0
        in_align = False
        n_m, groups = _apply_stage_freeze(model, stage_name)
        stage_label = stage_name
        print(f"[train] stage={stage_name} training {groups} trainable={n_m:.2f}M", flush=True)
        if n_m == 0:
            raise RuntimeError(f"stage={stage_name} has no trainable parameters")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(model, cfg)
    print(f"[train] train={n_tr} val={n_val} stage={stage_label} "
          f"trainable={sum(p.numel() for p in params)/1e6:.2f}M", flush=True)

    accum = max(cfg.train.grad_accum, 1)
    chunk = cfg.train.region_chunk_size or 0
    use_chunk = chunk > 0 and hasattr(model, "train_step_chunked")
    if use_chunk:
        print(f"[train] region sub-batching on (chunk_size={chunk})", flush=True)

    best = {"loss": float("inf")}
    step = 0
    epochs_since_best = 0
    stage_start = 0
    align_best, align_since = float("inf"), 0

    def _to_generate(epoch_next: int):
        nonlocal params, opt, best, epochs_since_best, stage_start
        bp = out_dir / "best.pt"
        # Only rank0 writes best.pt, so a rank arriving first would read the previous epoch's
        # file (or none) and start generate from different weights. Read only after the write.
        if dist:
            ddp.barrier(dist)
        if bp.exists():                                    # reload align best → seed generate
            sd = torch.load(bp, map_location=device)["trainable_state"]
            _, unexpected = model.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"best.pt reload unexpected keys: {list(unexpected)[:5]}")
            print(f"[train] reload align best.pt (epoch {best.get('epoch','?')}, "
                  f"val {best.get('loss', float('nan')):.4f}) → seed generate", flush=True)
        _set_lora_trainable(model, True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = _build_optimizer(model, cfg)
        best = {"loss": float("inf")}
        epochs_since_best = 0
        stage_start = epoch_next
        print(f"[train] → generate at epoch {epoch_next} "
              f"(trainable {sum(p.numel() for p in params)/1e6:.2f}M)", flush=True)

    # --- restart recovery (--resume) -----------------------------------------
    # A power cut once cost 12.7 hours of prefix-stage training. resume.pt stores weights +
    # optimizer + counters + RNG every epoch, and is picked up here as-is.
    start_epoch = 0
    resume_path = out_dir / "resume.pt"
    if resume and resume_path.exists():
        # weights_only=False is required — the payload holds numpy RNG state, which fails to load
        # under the torch>=2.6 default (it would die with UnpicklingError exactly when resuming).
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        st = ck["state"]
        # 1) restore the stage first — in the generate stage LoRA must be enabled and the
        #    optimizer rebuilt before its state is loaded, or the param groups will not match.
        if stage_name == "align_generate" and not st.get("in_align", True) and in_align:
            _set_lora_trainable(model, True)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = _build_optimizer(model, cfg)
            in_align = False
        # 2) weights and optimizer
        _, unexpected = model.load_state_dict(ck["trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"resume.pt unexpected keys: {list(unexpected)[:5]}")
        opt.load_state_dict(ck["optimizer"])
        # 3) progress counters
        start_epoch = int(st["epoch"])            # next epoch to run (0-based)
        best = st["best"]
        epochs_since_best = int(st.get("epochs_since_best", 0))
        stage_start = int(st.get("stage_start", 0))
        align_best = float(st.get("align_best", float("inf")))
        align_since = int(st.get("align_since", 0))
        step = int(st.get("step", 0))
        _restore_rng(ck.get("rng", {}))
        print(f"[resume] {resume_path} — resuming at epoch {start_epoch} "
              f"(best epoch {best.get('epoch','?')} · stage={'align' if in_align else 'generate/single'} "
              f"· optimizer state {len(opt.state_dict().get('state', {}))} tensors)", flush=True)
    elif resume:
        print(f"[resume] no {resume_path} — training from scratch", flush=True)

    if dist and dist.enabled:
        # Align the starting weights across ranks. The seed differs per rank (`seed + rank`), so
        # any module that init_from does not overwrite starts from different random values, and
        # then averaging gradients means averaging different models. This must run **after every
        # weight load (init_from, resume)** — called earlier, the loads overwrite the broadcast.
        #
        # Broadcast **everything except the LLM base**, not just the trainable parameters (the
        # same set save_checkpoint stores): frozen modules still run in the forward pass, so a
        # per-rank difference splits the gradients. The base is excluded because broadcasting
        # 4bit (bnb `Params4bit`) tensors is unsafe (see the ddp.py header).
        ddp.broadcast_parameters(
            dist, [p for n, p in model.named_parameters() if not _is_llm_base(n)])

    for epoch in range(start_epoch, cfg.train.epochs):
        if hasattr(train_ld.dataset, "set_epoch"):
            train_ld.dataset.set_epoch(epoch)
        if train_ld.sampler is not None and hasattr(train_ld.sampler, "set_epoch"):
            train_ld.sampler.set_epoch(epoch)      # sync shuffling across ranks (else same split)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, run_loss = time.time(), 0.0
        for i, batch in enumerate(train_ld):
            batch = to_device(batch, device)
            if use_chunk:
                out = model.train_step_chunked(batch, chunk, loss_scale=1.0 / accum)
            else:
                out = model(batch)
                (out["loss"] / accum).backward()
            run_loss += out["loss"].item()
            if (i + 1) % accum == 0:
                # ⚠️ average **before** clipping. Clipping with a per-rank norm changes the
                # average itself and lets the ranks drift apart.
                if dist and dist.enabled:
                    ddp.average_gradients(dist, params)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if max_steps and step >= max_steps:
                break
        val = evaluate(model, val_ld, device, dist)
        dt = time.time() - t0
        tr_loss = run_loss / max(len(train_ld), 1)
        extra = " ".join(f"{k}={val[k]:.4f}" for k in val if k != "loss")
        _main = (dist is None) or dist.is_main
        if _main:
            print(f"[epoch {epoch+1}/{cfg.train.epochs}] train_loss={tr_loss:.4f} "
                  f"val_loss={val['loss']:.4f} ({extra}) {dt:.0f}s", flush=True)
        if wandb_run is not None and _main:
            wandb_run.log({"epoch": epoch + 1, "train/loss": tr_loss,
                           **{f"val/{k}": val[k] for k in val}})
        # best / early-stop metric. 'finding' looks only at the token CE of finding regions
        # (the many normal EOS tokens mechanically lower the total loss and would skew selection).
        _mk = "l_region_finding" if cfg.train.best_metric == "finding" else "loss"
        _cur = float(val.get(_mk, val["loss"]))
        # val is already summed across ranks, so `_cur` is identical everywhere and the decision
        # agrees by construction. Only rank0 writes files (concurrent writes to one path corrupt).
        if _cur < best.get(_mk, best["loss"]):
            best = {"epoch": epoch + 1, **val}
            if _main:
                save_checkpoint(out_dir / "best.pt", model, cfg, best)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if _main:
            save_checkpoint(out_dir / "last.pt", model, cfg, {"epoch": epoch + 1, **val})
            if getattr(cfg.train, "save_every_epoch", False):
                save_checkpoint(out_dir / f"epoch{epoch + 1}.pt", model, cfg,
                                {"epoch": epoch + 1, **val})

        if in_align:
            elapsed = epoch + 1 - stage_start
            v = float(val.get(_mk, val["loss"]))
            if v < align_best:
                align_best, align_since = v, 0
            else:
                align_since += 1
            advance, reason = _stage_advance(elapsed, freeze, s_early, s_patience, align_since)
            if advance:
                print(f"[train] align done ({reason}; {elapsed} epochs, best val {align_best:.4f})",
                      flush=True)
                _to_generate(epoch + 1)
                in_align = False
        else:
            if patience and epochs_since_best >= patience:
                print(f"[early-stop] no val improvement({_mk}) in {patience} epochs "
                      f"(best epoch {best['epoch']}, {_mk} "
                      f"{float(best.get(_mk, best['loss'])):.4f}); stopping.", flush=True)
                break
        # Restart state — saved **after** the stage switch is handled, so it captures the state
        # just after the switch and resuming does not delay it by one epoch. dry-runs write none.
        if not max_steps and _main:
            save_resume(out_dir / "resume.pt", model, cfg, opt,
                        {"epoch": epoch + 1, "best": best, "epochs_since_best": epochs_since_best,
                         "step": step, "in_align": in_align, "stage_start": stage_start,
                         "align_best": align_best, "align_since": align_since})
        if max_steps and step >= max_steps:
            if torch.cuda.is_available():
                print(f"[dry-run] peak CUDA mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
                      f"(reserved {torch.cuda.max_memory_reserved()/1e9:.2f} GB)", flush=True)
            print("[train] max_steps reached (dry-run); stopping.", flush=True)
            break
    return best


def main():
    ap = argparse.ArgumentParser(description="Train the report-generation model.")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--out-dir", required=True, help="output dir (checkpoints/logs)")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 = dry-run (no done.flag)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out-dir>/resume.pt if it exists (restores weights, "
                         "optimizer, counters and RNG); otherwise start from scratch")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Under torchrun this returns per-rank info, otherwise single-process info (no-op path).
    dist = ddp.setup()
    # Give each rank a different seed — with one seed, dropout and sampling would be identical
    # on every rank, weakening data parallelism and making every rank pick the same reports.
    seed_everything(cfg.train.seed + dist.rank)
    device = dist.device if dist.enabled else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    if (out_dir / "done.flag").exists():                       # skip a run that already finished
        if dist.is_main:
            print(f"[train] {out_dir}/done.flag exists — skipping (already finished).", flush=True)
        ddp.cleanup(dist)
        return
    meta = snapshot_repro(out_dir, cfg) if dist.is_main else {}
    ddp.barrier(dist)                       # others enter only after rank0 has created out_dir
    if dist.is_main:
        print(f"[train] repro: {json.dumps(meta, ensure_ascii=False)} · ranks={dist.world_size}",
              flush=True)

    wandb_run = None
    if args.wandb and dist.is_main:
        import os
        import wandb
        wandb_run = wandb.init(project="toothfairy",
                               entity=os.environ.get("WANDB_ENTITY") or None,
                               name=cfg.train.run_name or "report-model",
                               config={"config": args.config})

    best = train(cfg, device, out_dir, wandb_run, args.max_steps, dist=dist, resume=args.resume)
    if dist.is_main:
        (out_dir / "metrics.json").write_text(
            json.dumps({**meta, "best": best}, ensure_ascii=False, indent=2))
        if not args.max_steps:                                 # a dry-run must not write done.flag
            (out_dir / "done.flag").write_text(f"best epoch {best.get('epoch','?')} "
                                               f"val {best.get('loss', float('nan')):.4f}\n")
            (out_dir / "resume.pt").unlink(missing_ok=True)   # resume state no longer needed
        print(f"[train] done. best={json.dumps(best, ensure_ascii=False)}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    ddp.cleanup(dist)


if __name__ == "__main__":
    main()
