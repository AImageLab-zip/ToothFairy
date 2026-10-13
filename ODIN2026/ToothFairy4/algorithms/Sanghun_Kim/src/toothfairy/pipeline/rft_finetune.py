#!/usr/bin/env python3
"""The second half of an RFT round: **finetune the LoRA on the accepted samples**.

The per-region generations accepted by `pipeline/rft_rollout.py` (`rft_selected.jsonl`) are used as
**supervised targets** to continue training **only the LoRA** of the LLM LoRA stage checkpoint.
Perception and the projector stay frozen: a linear probe showed the projector is not what needs
fixing, and with a training set of only a few thousand tokens, unfreezing more only adds
overfitting risk.

⚠️ **Do not misread the metrics.**
- `val_reward` (fast scorer, greedy decoding) is **exactly the quantity being optimised**. It is
  expected to go up, so it is not evidence of success: use it only to confirm that training took
  effect and as a first pass when comparing arms.
- `val_ce` (report CE against the original GT) can get worse, because RFT learns from the
  **model's own output**. A worse value is not a failure; read it as a drift indicator.
- The real verdict is **RadFact** (run separately). A rising reward with a flat RadFact is
  reward hacking.
- **Regression watch**: regions left out of the reward (field-of-view statements, negated
  findings) get no training signal and, this being RFT, no KL term either. The rate at which the
  field-of-view vocabulary is still uttered is reported for every arm.

usage:
  python -m toothfairy.pipeline.rft_finetune --selected experiments/analysis/rft-round1/rft_selected.jsonl \\
      --lr 5e-5 --epochs 2 --out-dir models/sinus_rft_lr5e-5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from toothfairy.paths import REPO  # noqa: E402

from toothfairy.config import load_config                                    # noqa: E402
from toothfairy.data import ReportDataset, make_collate, read_split          # noqa: E402
from toothfairy.generation.postprocess import clean_gen                      # noqa: E402
from toothfairy.models import ReportModel                                    # noqa: E402
from toothfairy.rl.rollout import eos_token_ids, evaluate_reward, rollout    # noqa: E402
from toothfairy.training.trainer import _apply_stage_freeze, to_device       # noqa: E402

FOV_RX = re.compile(r"not included|excluded|not available|not assessab|not evaluab|not visible", re.I)


@torch.no_grad()
def eval_val(model, cfg, tok, dev, val_pids, chunk: int) -> dict:
    """Validation report CE over finding regions, against the original GT. RFT can make this
    worse; it is a drift indicator."""
    ds = ReportDataset(val_pids, cfg, repo_root=REPO, tokenizer=tok, resample=False)
    ld = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                     collate_fn=make_collate(cfg.perception.feat_dim))
    model.eval()
    num = den = 0.0
    for b in ld:
        out = model.eval_loss_chunked(to_device(b, dev), chunk)
        n = float(out.get("ntok_finding", 0.0))
        if n:
            num += float(out["l_region_finding"]) * n
            den += n
    return {"val_ce_finding": num / den if den else float("nan")}


@torch.no_grad()
def eval_fov(model, cfg, gc_, tok, dev, val_pids, eos_ids, pad_id, n: int) -> dict:
    """Regression watch: does the model still utter field-of-view wording in the regions that
    the reward leaves out (greedy decoding)?"""
    ds = ReportDataset(val_pids, cfg, repo_root=REPO, tokenizer=None, resample=False)
    collate = make_collate(cfg.perception.feat_dim)
    union = ds.union_targets_by_pid() if gc_.reward_gt == "union" else None
    n_hit = n_gt = n_say = 0
    for i in range(min(n, len(ds))):
        batch = to_device(collate([ds[i]]), dev)
        groups = rollout(model, batch, gc_, random.Random(f"fov:{i}"), eos_ids, pad_id,
                         greedy=True, union_targets=union)
        for g in groups:
            if g.region_id.startswith("fdi:") or not g.comps:
                continue
            gt = (union or {}).get(batch["pid"][0], {}).get(g.region_id, "") or ""
            txt = clean_gen(tok.decode(g.comps[0], skip_special_tokens=True))
            if FOV_RX.search(gt):
                n_gt += 1
                n_hit += bool(FOV_RX.search(txt))
            n_say += bool(FOV_RX.search(txt))
    return {"fov_recall": n_hit / n_gt if n_gt else float("nan"),
            "fov_gt_regions": n_gt, "fov_said": n_say}


def _tokenize(tok, text: str, max_len: int) -> torch.Tensor:
    """Target tokens. An **empty string stays an empty tensor**: that is the normal signal for
    learning to emit EOS immediately (silence)."""
    if not (text or "").strip():
        return torch.empty(0, dtype=torch.long)
    return tok(text, truncation=True, max_length=max_len, add_special_tokens=False,
               return_tensors="pt")["input_ids"][0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="toothfairy/configs/sinus_rft_rollout.yaml")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--selected", required=True, help="rft_selected.jsonl written by `toothfairy.cli.rft select`")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--fov-patients", type=int, default=20)
    ap.add_argument("--maint-weight", default="auto",
                    help="loss weight of the maintenance (normal) samples; auto matches their "
                         "loss contribution to the sample-count ratio")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    cfg = load_config(REPO / a.config if not Path(a.config).is_absolute() else a.config)
    gc_ = cfg.grpo
    out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else REPO / a.out_dir
    if (out_dir / "done.flag").exists():                 # protect an already finished run
        print(f"[rft-ft] {out_dir}/done.flag exists - skipping (finished run protected).",
              flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = "cuda"

    sel = [json.loads(l) for l in Path(a.selected).open()]
    by_pid: dict[str, list[dict]] = {}
    for r in sel:
        by_pid.setdefault(r["pid"], []).append(r)
    n_find = sum(1 for r in sel if r.get("kind", "finding") == "finding")
    n_maint = len(sel) - n_find
    print(f"[rft-ft] {len(sel)} accepted samples (finding {n_find} · maintenance {n_maint}) · "
          f"{len(by_pid)} patients · lr={a.lr} · {a.epochs} epochs", flush=True)

    # Maintenance-sample weight: `auto` **matches the loss contribution to the sample-count
    # ratio**. Left alone, maintenance targets are short (1-5 EOS tokens) and drown in the finding
    # targets (23 tokens on average), so the model stops respecting the silence rule.
    if a.maint_weight == "auto":
        tf = sum(len((r["chosen"] or "").split()) or 1 for r in sel
                 if r.get("kind", "finding") == "finding")
        tm = sum(len((r["chosen"] or "").split()) or 1 for r in sel
                 if r.get("kind", "finding") == "normal")
        w_maint = (tf / max(n_find, 1)) / max(tm / max(n_maint, 1), 1e-8) if n_maint else 0.0
    else:
        w_maint = float(a.maint_weight)
    print(f"[rft-ft] maintenance-sample loss weight w_maint={w_maint:.2f}", flush=True)

    model = ReportModel(cfg).to(dev)
    ipath = Path(a.init_from or gc_.init_from)
    ipath = ipath if ipath.is_absolute() else REPO / ipath
    ck = torch.load(ipath, map_location="cpu", weights_only=False)
    sd = ck.get("trainable_state", ck)
    # The claim-schema axis merge changed SOFT_DIM from 31 to 22, so in checkpoints from before
    # the merge only `realizer.prob_proj` has a mismatching shape. When soft injection is off that
    # tensor is not on the forward path, so it is dropped and the rest is loaded; if soft injection
    # is on, or any other tensor mismatches, this fails immediately (the same guard the rollout
    # module applies).
    _cur = model.state_dict()
    _drop = [k for k, v in sd.items()
             if k in _cur and tuple(_cur[k].shape) != tuple(v.shape)]
    if _drop:
        _bad = [k for k in _drop if not k.startswith("realizer.prob_proj")]
        if _bad or model._soft_on():
            raise RuntimeError(
                f"shape mismatch - different architecture (soft={model._soft_on()}): "
                f"{_bad or _drop}")
        print(f"[rft-ft] dropped on shape mismatch (soft injection off): {_drop}", flush=True)
        sd = {k: v for k, v in sd.items() if k not in _drop}
    _m, unexp = model.load_state_dict(sd, strict=False)
    if unexp:
        raise RuntimeError(f"init_from does not match this model architecture: {list(unexp)[:5]}")
    n_tr, groups_ = _apply_stage_freeze(model, "qlora")   # **LoRA only**
    tok = model.realizer.tokenizer
    eos_ids, pad_id = eos_token_ids(tok)
    print(f"[rft-ft] starting from {ipath.parent.name} · training {groups_} {n_tr:.2f}M",
          flush=True)

    tr_pids = read_split(REPO / "data/splits/train.txt")
    va_pids = read_split(REPO / "data/splits/val.txt")
    ds = ReportDataset([p for p in tr_pids if p in by_pid], cfg, repo_root=REPO,
                       tokenizer=None, resample=False)
    collate = make_collate(cfg.perception.feat_dim)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    val_ds = ReportDataset(va_pids, cfg, repo_root=REPO, tokenizer=None, resample=False)
    val_union = val_ds.union_targets_by_pid() if gc_.reward_gt == "union" else None

    def evaluate() -> dict:
        """Three validation metrics: reward (the optimised quantity, for a first pass over the
        arms), CE (drift indicator) and field-of-view recall (regression watch)."""
        model.eval()
        m = evaluate_reward(model, val_ds, collate, gc_, dev, eos_ids, pad_id,
                            len(val_ds), val_union)
        m |= eval_val(model, cfg, tok, dev, va_pids, a.chunk)
        m |= eval_fov(model, cfg, gc_, tok, dev, va_pids, eos_ids, pad_id, a.fov_patients)
        return m

    base = evaluate()
    print(f"[rft-ft] before training: {json.dumps(base, ensure_ascii=False)}", flush=True)

    hist = [{"epoch": 0, **base}]
    for ep in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = list(range(len(ds)))
        random.Random(a.seed + ep).shuffle(order)
        t0, run, nstep = time.time(), 0.0, 0
        for i, idx in enumerate(order):
            batch = to_device(collate([ds[idx]]), dev)
            pid = batch["pid"][0]
            ctx, aux = model._encode(batch)
            spec = {"finding": [], "normal": []}
            for r in by_pid.get(pid, []):
                ids = _tokenize(tok, r["chosen"], cfg.realizer.max_text_len)
                kind = r.get("kind", "finding")
                if kind == "finding" and ids.numel() == 0:
                    continue                                  # empty target for a finding = bug
                tokens, prob = model._slot_tokens(ctx[0], r["region_id"], None, False)
                spec[kind].append({"tokens": tokens, "region_id": r["region_id"],
                                   "prob": prob, "target_ids": ids.to(dev)})
            if not spec["finding"] and not spec["normal"]:
                continue
            # Finding and maintenance samples are computed **separately** and combined with a
            # weight. In one block they would be weighted by token count, and the maintenance
            # samples (usually 1-5 EOS tokens) would drown in the finding ones (23 on average).
            lf, nf = model.realizer.decode_regions(spec["finding"], return_count=True)
            ln, nn_ = model.realizer.decode_regions(spec["normal"], return_count=True)
            loss = (lf * nf + w_maint * ln * nn_) / max(nf + w_maint * nn_, 1e-8)
            (loss / a.grad_accum).backward()
            run += float(loss.item()); nstep += 1
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                                if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        m = {"epoch": ep, "train_loss": run / max(nstep, 1), "sec": time.time() - t0}
        m |= evaluate()
        hist.append(m)
        print(f"[rft-ft] epoch {ep}: {json.dumps(m, ensure_ascii=False)}", flush=True)

    keep = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"trainable_state": keep, "run_name": out_dir.name, "lr": a.lr,
                "epochs": a.epochs, "init_from": str(ipath), "hist": hist,
                "n_finding": n_find, "n_maint": n_maint, "w_maint": w_maint},
               out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"lr": a.lr, "epochs": a.epochs, "n_selected": len(sel), "n_finding": n_find,
         "n_maint": n_maint, "w_maint": w_maint, "hist": hist},
        ensure_ascii=False, indent=2))
    (out_dir / "done.flag").write_text(f"rft lr={a.lr} epochs={a.epochs}\n")
    print(f"[rft-ft] wrote {out_dir}/best.pt · metrics.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
