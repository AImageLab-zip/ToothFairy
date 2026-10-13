#!/usr/bin/env python3
"""Merge a vision LoRA adapter into the AWQ checkpoint, at the tensor level.

`docs/vision_sft_plan.md` §3 and its log entry 'AWQ quantizes almost nothing'. The arm is trained against bf16
`models/Qwen3.5-9B` and served from `models/Qwen3.5-9B-AWQ`, and this is the
script that makes those the same model.

WHY THIS IS EXACT AND NOT AN APPROXIMATION. AWQ's `modules_to_not_convert`
covers `visual`, `linear_attn`, `self_attn`, `model.layers.0.` and `mtp`, so
every `model.visual.*` tensor is bit-identical between the two checkpoints --
the same sha256, not merely the same shape. Arm 2 targets `model.visual.*` and
nothing else. So the weights the LoRA was fit against ARE the weights it is
merged into, and the merge introduces no numerical question of its own.

That claim is the load-bearing assumption of the whole light plan, so this
script VERIFIES it per tensor rather than citing it (`--allow-base-mismatch`
to override, loudly). If someone later points an arm at `model.language_model`
the check fires on the first quantized module instead of producing a silently
wrong checkpoint: you cannot add a bf16 delta to a 4-bit packed tensor.

That guard turned out to be exactly the right shape when arm 5
(`vision+language`) was added on 2026-08-14. It targets language weights and
still passes, because AWQ's skip list leaves `self_attn` and `linear_attn` in
bf16 -- so those tensors are bit-identical too. What the check blocks is
`mlp.{down,up,gate}_proj`, the only group AWQ actually quantizes. The fence
between a shippable arm and an unshippable one is therefore enforced here,
per tensor, rather than by remembering which modules were safe.

WHY TENSOR SURGERY RATHER THAN peft's merge_and_unload(). That path loads the
whole model -- which, on the AWQ checkpoint, means the gptqmodel/Marlin stack
that cost four jobs on 2026-08-14 and never reached a forward pass. Here
nothing is instantiated: shards in, shards out, no model class, no quantizer,
no GPU. Only 2 of the 5 AWQ shards hold visual tensors, so the other 3 are
hardlinked and never read.

WHAT THIS DOES NOT PROVE. That the merged tensors are right arithmetically is
checked here; that the served model behaves like the adapter-applied one is
not, and cannot be without inference. Run the ~20-call check the script prints
at the end before scoring an arm with it.

Runs in `cbct_sft_cu128` (cbct_base has no `safetensors`). CPU only, ~1 min.
Peak RAM is one shard, ~3 GB -- fine on the login node, but the cpu partition
is the safe habit:

    srun --partition=cpu --qos=cpu --mem=16G \\
      ~/miniconda3/envs/cbct_sft_cu128/bin/python code/train/merge_vision_lora.py \\
        --adapter outputs/vsft_arm2/adapter \\
        --out models/Qwen3.5-9B-AWQ-vsft-arm2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

ROOT = REPO_ROOT

import lora_arms  # noqa: E402  (after sys.path)

# peft stores every adapter weight under this prefix, and appends the adapter
# name when the model was saved with more than the default one. Both are
# stripped to recover the base module path the tensor belongs to.
PEFT_PREFIX = "base_model.model."
ADAPTER_SUFFIXES = (".lora_A.weight", ".lora_B.weight",
                    ".lora_A.default.weight", ".lora_B.default.weight")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


# ── The adapter ───────────────────────────────────────────────────────────

def load_adapter(adapter_dir: Path):
    """Adapter tensors keyed by BASE module path, plus the merge scaling.

    Returns (pairs, scaling, cfg) where pairs maps
    `model.visual.blocks.0.attn.qkv` -> {"A": tensor, "B": tensor}.
    """
    import torch
    from safetensors.torch import load_file

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[FAIL] no adapter_config.json in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Everything here changes the arithmetic below, so none of it is assumed.
    if cfg.get("use_dora"):
        raise SystemExit("[FAIL] DoRA adapter: magnitude vectors are not a "
                         "plain B@A delta and this script does not implement "
                         "them. Nothing in lora_arms.py sets use_dora.")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("[FAIL] fan_in_fan_out=True -- the delta would need "
                         "transposing. nn.Linear never sets it.")
    if cfg.get("lora_bias") not in (None, False, "none"):
        raise SystemExit(f"[FAIL] lora_bias={cfg.get('lora_bias')!r}: this "
                         f"script merges weights only.")

    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"[FAIL] no adapter_model.safetensors in {adapter_dir}")
    raw = load_file(str(weights))

    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    # peft: alpha/r normally, alpha/sqrt(r) under rank-stabilised LoRA.
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

    pairs: dict[str, dict[str, "torch.Tensor"]] = defaultdict(dict)
    skipped = []
    for key, tensor in raw.items():
        if "lora_magnitude_vector" in key or "lora_embedding" in key:
            raise SystemExit(f"[FAIL] unsupported adapter tensor {key!r}")
        for suffix in ADAPTER_SUFFIXES:
            if key.endswith(suffix):
                mod = key[:-len(suffix)]
                if mod.startswith(PEFT_PREFIX):
                    mod = mod[len(PEFT_PREFIX):]
                pairs[mod]["A" if ".lora_A" in suffix else "B"] = tensor
                break
        else:
            skipped.append(key)
    if skipped:
        raise SystemExit(f"[FAIL] {len(skipped)} adapter tensor(s) matched no "
                         f"known LoRA suffix, e.g. {skipped[0]!r}. Refusing "
                         f"rather than merging a partial adapter.")

    dead = []
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"[FAIL] {mod}: has only {sorted(ab)}, needs both")
        # peft initialises B to zero, so B@A is zero until something has
        # trained. Caught here rather than after the merge: an untrained
        # adapter would otherwise write ~3 GB of shards before being spotted.
        if not ab["B"].any():
            dead.append(mod)
    if dead:
        raise SystemExit(
            f"[FAIL] lora_B is all-zero in {len(dead)}/{len(pairs)} module(s), "
            f"e.g. {dead[0]}.\n"
            f"       B starts at zero and stays there if the module never got "
            f"a gradient, so the delta is exactly zero and the merged "
            f"checkpoint IS the baseline -- scoring identically while looking "
            f"like an arm.")

    return dict(pairs), scaling, cfg


def check_against_arm(pairs: dict, arm: str, rank: int | None) -> None:
    """Refuse unless the adapter's modules are exactly the arm's declared set.

    The same failure lora_arms.assert_arm() catches at train time, caught
    again at merge time -- because an adapter can also be mismatched by being
    the WRONG one, and `outputs/vsft_arm2/adapter` is a path, not a proof.
    """
    import re

    spec = lora_arms.ARMS[arm]
    pattern = re.compile(spec["pattern"])
    bad = sorted(m for m in pairs if not pattern.fullmatch(m))
    if bad:
        raise SystemExit(
            f"[FAIL] {len(bad)} adapter module(s) do not match arm {arm!r}:\n"
            f"       e.g. {bad[0]}\n       pattern: {spec['pattern']}")
    if len(pairs) != spec["n_modules"]:
        raise SystemExit(
            f"[FAIL] arm {arm!r}: adapter has {len(pairs)} module(s), "
            f"declared {spec['n_modules']}. A short adapter merges silently "
            f"and serves a partly-trained tower.")

    r = rank or spec["r"]
    n_params = sum(t.numel() for ab in pairs.values() for t in ab.values())
    want = lora_arms.expected_params(arm, r)
    if n_params != want:
        raise SystemExit(f"[FAIL] arm {arm!r} r={r}: adapter holds "
                         f"{n_params:,} parameter(s), declared {want:,}")
    print(f"[PASS] arm {arm!r}: {len(pairs)} module(s), {n_params:,} adapter "
          f"parameter(s) -- matches lora_arms.py")


# ── The merge ─────────────────────────────────────────────────────────────

def merge(args) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_dir = Path(args.adapter).resolve()
    train_base = Path(args.train_base).resolve()
    target_base = Path(args.target_base).resolve()
    final = Path(args.out).resolve()

    if final.exists() and not args.force:
        raise SystemExit(f"[FAIL] {final} exists -- pass --force to replace it")

    # Build under a name nothing will serve, and rename only once every check
    # has passed. Several of those checks run after the shards are written, so
    # without this a walltime kill or a Ctrl-C in between leaves a checkpoint
    # that is complete to `ls` and wrong to vLLM.
    out = final.parent / (final.name + ".incomplete")
    shutil.rmtree(out, ignore_errors=True)

    # Which arm, from the run's own metadata rather than from the caller.
    meta_path = adapter_dir.parent / "train_meta.json"
    train_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    arm = args.arm or train_meta.get("arm")
    if not arm:
        raise SystemExit("[FAIL] no arm: train_meta.json is missing and "
                         "--arm was not given")
    if arm not in lora_arms.ARMS:
        raise SystemExit(f"[FAIL] unknown arm {arm!r}")
    rank = args.rank or train_meta.get("rank")
    print(f"[INFO] arm {arm!r} rank {rank or lora_arms.ARMS[arm]['r']} "
          f"from {'train_meta.json' if train_meta else 'CLI'}")

    pairs, scaling, adapter_cfg = load_adapter(adapter_dir)
    print(f"[INFO] adapter: {len(pairs)} module(s), r={adapter_cfg['r']}, "
          f"alpha={adapter_cfg['lora_alpha']}, "
          f"rslora={bool(adapter_cfg.get('use_rslora'))} -> scaling {scaling:g}")
    check_against_arm(pairs, arm, rank)

    tgt_index = json.loads((target_base / "model.safetensors.index.json")
                           .read_text(encoding="utf-8"))
    trn_index = json.loads((train_base / "model.safetensors.index.json")
                           .read_text(encoding="utf-8"))
    tgt_map, trn_map = tgt_index["weight_map"], trn_index["weight_map"]

    # Every adapter module must name a real weight in BOTH checkpoints.
    targets = {}
    for mod in pairs:
        name = f"{mod}.weight"
        if name not in tgt_map:
            raise SystemExit(f"[FAIL] {name} is not in {target_base.name} -- "
                             f"the adapter does not belong to this checkpoint")
        if name not in trn_map:
            raise SystemExit(f"[FAIL] {name} is not in {train_base.name}")
        targets[name] = mod

    shards = sorted({tgt_map[n] for n in targets})
    print(f"[INFO] {len(targets)} tensor(s) to rewrite across {len(shards)} of "
          f"{len(set(tgt_map.values()))} shard(s): {', '.join(shards)}")

    out.mkdir(parents=True, exist_ok=True)
    stats, identical, mismatched = [], 0, []

    for shard in shards:
        tgt_path, tensors = target_base / shard, {}
        with safe_open(str(tgt_path), framework="pt") as fh:
            metadata = fh.metadata() or {}
            for name in fh.keys():
                tensors[name] = fh.get_tensor(name)

        for name in sorted(n for n in targets if tgt_map[n] == shard):
            mod = targets[name]
            base = tensors[name]

            # The identity check: is the weight this LoRA was fit against the
            # weight it is being merged into? Bitwise, not approximately.
            with safe_open(str(train_base / trn_map[name]), framework="pt") as tf:
                trained_on = tf.get_tensor(name)
            if trained_on.shape != base.shape or trained_on.dtype != base.dtype:
                mismatched.append(f"{name}: {train_base.name} "
                                  f"{tuple(trained_on.shape)}/{trained_on.dtype} vs "
                                  f"{target_base.name} {tuple(base.shape)}/{base.dtype}")
            elif torch.equal(trained_on, base):
                identical += 1
            else:
                delta = (trained_on.float() - base.float()).abs().max().item()
                mismatched.append(f"{name}: differs, max |Δ| {delta:.3e}")
            del trained_on

            A, B = pairs[mod]["A"].float(), pairs[mod]["B"].float()
            if A.shape[1] != base.shape[1] or B.shape[0] != base.shape[0]:
                raise SystemExit(
                    f"[FAIL] {name}: base {tuple(base.shape)} but "
                    f"B@A is [{B.shape[0]}, {A.shape[1]}]")
            delta = (B @ A) * scaling
            if not torch.isfinite(delta).all():
                raise SystemExit(f"[FAIL] {name}: delta is not finite")

            merged = (base.float() + delta).to(base.dtype)
            changed = int((merged != base).sum().item())
            if changed == 0:
                shutil.rmtree(out, ignore_errors=True)
                raise SystemExit(
                    f"[FAIL] {name}: the merge changed nothing. The delta is "
                    f"non-zero in fp32 but vanishes on the cast back to "
                    f"{base.dtype} -- it is smaller than the weights' own "
                    f"resolution, so serving this would score as the baseline.")
            nb, nd = base.float().norm().item(), delta.norm().item()
            stats.append({
                "tensor": name,
                "rel": nd / nb if nb else float("inf"),
                "max_abs": delta.abs().max().item(),
                "changed": changed,
                "numel": base.numel(),
            })
            tensors[name] = merged
            del base, delta, merged, A, B

        save_file(tensors, str(out / shard), metadata=metadata or {"format": "pt"})
        print(f"[INFO] wrote {shard}")
        del tensors

    # ── the refusals ──────────────────────────────────────────────────────
    if mismatched:
        head = "\n       ".join(mismatched[:5])
        msg = (f"[FAIL] {len(mismatched)} tensor(s) differ between "
               f"{train_base.name} and {target_base.name}:\n       {head}\n"
               f"       The adapter was fit against weights that are not the "
               f"ones being served, so this merge is an approximation of "
               f"unknown size (light plan §3). Quantized modules land here.")
        if not args.allow_base_mismatch:
            shutil.rmtree(out, ignore_errors=True)
            raise SystemExit(msg)
        print(msg.replace("[FAIL]", "[WARN]"))
    else:
        print(f"[PASS] all {identical} merged tensor(s) bit-identical between "
              f"{train_base.name} and {target_base.name} -- the merge is exact")

    rels = sorted(s["rel"] for s in stats)
    med = rels[len(rels) // 2]
    print(f"[INFO] ||Δ||/||W||: min {rels[0]:.2e}  median {med:.2e}  "
          f"max {rels[-1]:.2e}")
    if med > args.max_rel:
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(
            f"[FAIL] median relative delta {med:.2e} exceeds --max-rel "
            f"{args.max_rel:g}. A vision LoRA that moves the tower this far "
            f"is more likely a scaling bug than training.")

    # ── everything not rewritten ──────────────────────────────────────────
    linked = copied = 0
    for src in sorted(target_base.iterdir()):
        if src.is_dir() or src.name in shards:
            continue
        dst = out / src.name
        if src.suffix == ".safetensors":
            # Untouched shard: hardlink. Nothing in this repo edits a
            # safetensors file in place, and it saves ~8 GB per arm.
            try:
                os.link(src, dst)
                linked += 1
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        copied += 1
    print(f"[INFO] sidecars: {linked} shard(s) hardlinked, {copied} file(s) copied")

    # The index is unchanged -- same names, same shards, same shapes, same
    # dtypes, so the same total_size.
    for required in ("config.json", "chat_template.jinja",
                     "preprocessor_config.json", "tokenizer.json",
                     "model.safetensors.index.json"):
        if not (out / required).exists():
            raise SystemExit(f"[FAIL] {required} did not reach {out}")
    print("[PASS] config, chat_template, preprocessor, tokenizer and index all present")

    (out / "merge_meta.json").write_text(json.dumps({
        "arm": arm,
        "rank": rank or lora_arms.ARMS[arm]["r"],
        "scaling": scaling,
        "adapter": str(adapter_dir),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "train_base": str(train_base),
        "target_base": str(target_base),
        "bases_bit_identical": not mismatched,
        "modules_merged": len(stats),
        "rel_delta": {"min": rels[0], "median": med, "max": rels[-1]},
        "shards_rewritten": shards,
        "git_commit": git_commit(),
        "train_meta": train_meta,
    }, indent=2), encoding="utf-8")

    out.rename(final)
    out = final

    print(f"\n[PASS] merged -> {out}")
    # This used to print a two-line recipe, as if the check were a step someone
    # had skipped. It is not: the adapter-applied half has never existed. Audited
    # 2026-08-17 -- no --enable-lora / LoRARequest / load_adapter anywhere in
    # anywhere under code/, and arms 1, 2, 5 and 6 were all scored without it. Printing
    # a command that cannot be completed reads as a chore; printing the gap reads
    # as what it is. See code/train/README.md.
    print("\n  NOT VERIFIED, AND NOT CURRENTLY VERIFIABLE: that this checkpoint "
          "behaves like\n  the adapter-applied model. Merge arithmetic is "
          "checked above; serving is not.\n\n"
          "  Nothing in this repo applies a LoRA adapter at load time, so the "
          "reference half\n  of that diff has no harness. Running the merged "
          "half alone proves only that the\n  checkpoint serves:\n\n"
          f"    QWEN_MODEL_NAME={out.name} RUN_NAME=vsft_verify \\\n"
          f"      sbatch scripts/pool_infer.sh\n\n"
          "  A real check needs --enable-lora on vLLM 0.19.0 against "
          "Qwen3.5-VL (unverified\n  as supported), a stated pass/fail "
          "criterion, and TEMPERATURE=0 RETRY_TEMPERATURE=0\n  with a pinned "
          "CONCURRENCY on both sides -- decoding is nondeterministic by "
          "default.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True,
                    help="peft adapter dir, e.g. outputs/vsft_arm2/adapter")
    ap.add_argument("--out", required=True, help="checkpoint dir to write")
    ap.add_argument("--train-base", default=str(ROOT / "models/Qwen3.5-9B"),
                    help="checkpoint the LoRA was FIT against (default: bf16)")
    ap.add_argument("--target-base", default=str(ROOT / "models/Qwen3.5-9B-AWQ"),
                    help="checkpoint to merge INTO and serve (default: AWQ)")
    ap.add_argument("--arm", choices=list(lora_arms.ARMS),
                    help="override the arm in train_meta.json")
    ap.add_argument("--rank", type=int, help="override the rank in train_meta.json")
    ap.add_argument("--max-rel", type=float, default=0.5,
                    help="refuse if the median ||Δ||/||W|| exceeds this (0.5)")
    ap.add_argument("--allow-base-mismatch", action="store_true",
                    help="merge even where train and target weights differ. "
                         "Makes the merge an approximation -- see §3.")
    ap.add_argument("--force", action="store_true", help="replace --out if it exists")
    args = ap.parse_args()

    if args.force and Path(args.out).exists():
        shutil.rmtree(args.out)
    return merge(args)


if __name__ == "__main__":
    sys.exit(main())
