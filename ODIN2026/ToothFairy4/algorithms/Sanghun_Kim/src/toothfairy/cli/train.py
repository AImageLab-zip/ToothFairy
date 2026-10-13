#!/usr/bin/env python3
"""Train SINUS: perception -> visual prefix -> LLM LoRA.

Each stage optimises exactly one thing and starts from the previous stage's checkpoint
(``init_from`` in its config), so they have to run in this order:

  1. ``perception``  models/sinus_perception  toothfairy/configs/sinus_perception.yaml
     trains the 0.41M aggregator + findings head on the categorical claim loss plus a
     region-text SigLIP term. No language model is loaded (about a minute per epoch).
  2. ``prefix``      models/sinus_prefix      toothfairy/configs/sinus_prefix.yaml
     trains only the 2.18M projector that maps the anatomical tokens into the embedding
     space of the language model, on the per-slot token cross-entropy.
  3. ``llm_lora``    models/sinus_llm_lora    toothfairy/configs/sinus_llm_lora.yaml
     trains only the 29.1M LoRA adapter of the frozen 4-bit language model.

Rejection-sampling fine-tuning, the last stage of the submitted model, is a separate
pipeline — see ``toothfairy.cli.rft`` and TRAIN.md.

    python -m toothfairy.cli.train --gpus 0                 # single GPU, all three stages
    python -m toothfairy.cli.train --gpus 0,1               # 2-rank DDP for stages 2 and 3
    python -m toothfairy.cli.train --gpus 0 --stage prefix  # one stage only

A stage whose output directory already holds ``done.flag`` is skipped, so an interrupted
run can be restarted with the same command.

With more than one GPU, stages 2 and 3 run under ``torch.distributed.run`` and read the
``*_ddp<N>.yaml`` config, whose only difference is ``grad_accum`` halved so that the
effective batch (world_size x batch_size x grad_accum = 4) matches the single-GPU recipe.
Without that the effective batch becomes 8 and a change of batch size gets mixed into the
change of parallelism. Stage 1 always runs on one card: it is a classifier over cached
features and finishes in minutes.

Do not run the same stage twice at once — both invocations write the same output directory.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from toothfairy.paths import REPO, SRC

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# stage -> (module, config stem, output dir, resumable)
STAGES = {
    "perception": ("toothfairy.training.stage_a", "sinus_perception", "models/sinus_perception", False),
    "prefix":     ("toothfairy.training.trainer", "sinus_prefix",     "models/sinus_prefix",     True),
    "llm_lora":   ("toothfairy.training.trainer", "sinus_llm_lora",   "models/sinus_llm_lora",   True),
}
ORDER = ("perception", "prefix", "llm_lora")

# TF4_TORCH_GDN_FALLBACK=conv — the causal-conv1d fast path of the language model segfaults on
#   this stack, so that one kernel falls back to the reference implementation. Numerically
#   equivalent (the loss differs by less than 1e-3 relative).
# HF_HUB_OFFLINE=1 — with more than one rank, concurrent hub lookups intermittently fail on a
#   shard that is present in the local cache. Read from the cache only.
TRAIN_ENV = {
    "TF4_TORCH_GDN_FALLBACK": "conv",
    "HF_HUB_OFFLINE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
}


def _config_for(stem: str, nproc: int) -> Path:
    """The DDP variant if one exists for this rank count, otherwise the single-GPU config."""
    if nproc > 1:
        ddp = CONFIG_DIR / f"{stem}_ddp{nproc}.yaml"
        if ddp.is_file():
            return ddp
    return CONFIG_DIR / f"{stem}.yaml"


def run_stage(stage: str, gpus: list[str], master_port: int) -> int:
    module, stem, out_dir, resumable = STAGES[stage]
    out = REPO / out_dir
    if (out / "done.flag").is_file():
        print(f"[train] {out_dir} already finished — skip", flush=True)
        return 0

    # Stage 1 is single-card by construction; only stages 2 and 3 have DDP wiring.
    devices = gpus if stage != "perception" else gpus[:1]
    nproc = len(devices)
    cfg = _config_for(stem, nproc)
    if not cfg.is_file():
        print(f"[train] FATAL no config: {cfg}", file=sys.stderr)
        return 1

    cmd = [sys.executable]
    if nproc > 1:
        cmd += ["-m", "torch.distributed.run",
                f"--nproc_per_node={nproc}", f"--master_port={master_port}"]
    cmd += ["-m", module, "--config", str(cfg), "--out-dir", out_dir]
    if resumable:
        cmd.append("--resume")

    env = {**os.environ, **TRAIN_ENV, "CUDA_VISIBLE_DEVICES": ",".join(devices)}
    # The stages run as child processes, so they need to find the package too. With
    # `pip install -e .` they already would; putting src on their path as well means an
    # uninstalled checkout works the same way.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH", "")]))
    print(f"[train] {out_dir} (config={cfg.name}, ranks={nproc}, gpus={','.join(devices)})",
          flush=True)
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    if rc:
        print(f"[train] FATAL {out_dir} exited with {rc}", file=sys.stderr)
        return rc

    # Stage 1 writes best.pt but no done.flag (it has no --resume path); the later stages do.
    produced = (out / "done.flag").is_file() or (out / "best.pt").is_file()
    if not produced:
        print(f"[train] FATAL {out_dir} produced no checkpoint", file=sys.stderr)
        return 1
    print(f"[train] {out_dir} finished", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", required=True,
                    help="comma-separated CUDA device ids, e.g. 0 or 0,1")
    ap.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    ap.add_argument("--master-port", type=int, default=29510,
                    help="rendezvous port for the DDP stages")
    args = ap.parse_args(argv)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        ap.error("--gpus must name at least one device")

    for stage in (ORDER if args.stage == "all" else (args.stage,)):
        rc = run_stage(stage, gpus, args.master_port)
        if rc:
            return rc

    print("[train] all requested stages done — evaluate with: "
          "python -m toothfairy.cli.eval generate --run-dir models/sinus_llm_lora", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
