#!/usr/bin/env python
"""Assemble a RadFact-scorable checkpoint from an RFT run.

`rft_finetune.py` saves the LoRA tensors alone — that is what it trains — but
`pipeline/generate.py` builds a fresh ReportModel and loads one state dict into it, so it
needs the perception and projector weights too.  This joins the base policy's
trainable state with the round's LoRA (verified: merging the base policy with its
round's LoRA this way reproduces a hand-assembled checkpoint tensor for tensor,
0 differences over all 317 tensors).

The base's `config.resolved.yaml` is copied alongside because `pipeline/generate.py` reads
it from the run directory to rebuild the model.

Usage:
    python -m toothfairy.pipeline.rft_merge \
        --base models/sinus_llm_lora \
        --lora models/cast_rft_r3_w1 --out models/cast_rft_r3_w1_eval
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from toothfairy.paths import REPO  # noqa: E402


def _state(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("trainable_state", ck)


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise — never raises on a temp path."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="starting-policy directory (perception, projector and LoRA)")
    ap.add_argument("--lora", required=True, help="rft_finetune.py output directory (LoRA only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args(argv)

    base, lora, out = (Path(p) if Path(p).is_absolute() else REPO / p
                       for p in (a.base, a.lora, a.out))
    base_sd, lora_sd = _state(base / a.ckpt), _state(lora / a.ckpt)

    stray = [k for k in lora_sd if "lora_" not in k]
    if stray:
        raise SystemExit(f"non-LoRA tensors mixed in ({len(stray)}): {stray[:3]} — "
                         "check whether rft_finetune changed what it saves")
    missing = [k for k in lora_sd if k not in base_sd]
    if missing:
        raise SystemExit(f"{len(missing)} LoRA keys are missing from the starting policy: {missing[:3]} — "
                         "base and lora are different model structures")

    merged = {**base_sd, **lora_sd}
    changed = sum(1 for k in lora_sd if not torch.equal(base_sd[k], lora_sd[k]))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": merged, "run_name": out.name,
                "rft": {"base": _rel(base), "lora": _rel(lora)}},
               out / a.ckpt)
    shutil.copy2(base / "config.resolved.yaml", out / "config.resolved.yaml")
    print(f"[merge] {len(merged)} tensors (LoRA {len(lora_sd)} · {changed} of them changed by training) "
          f"→ {_rel(out)}/{a.ckpt}", flush=True)
    if changed == 0:
        print("[merge] ⚠️ not a single LoRA tensor changed — training had no effect", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
