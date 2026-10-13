#!/usr/bin/env python3
"""The **generation and selection** half of one RFT round.

RFT (rejection-sampling fine-tuning) = draw samples from the policy and **fine-tune on the
good ones only**. It is run before GRPO because **the selected samples stay on disk and can
be judged clinically by hand** (the reward is validated as a scorer, but not as a training
signal).

This script only does the front half of a round — **generate -> select -> dump to file +
statistics**. The fine-tuning is run separately once the threshold and the lr are fixed.

Selection rule
--------------
For every finding-region group the highest-reward sample is a candidate, and it is kept only
when **all** of the following hold.

  (a) reward >= `--threshold`   (0.3)
  (b) **strictly above the group median**
  (c) **reward above the greedy output**

Why (b) is needed: measured, **16.2% of finding groups give all 6 samples the same reward.**
Taking "the best one" there is the same as taking one at random, which puts noise into the
training data.

Why (c) is needed: (b) only compares **inside a group**, so even when all 6 samples are worse
than greedy the best of them is kept — in a 100-patient pilot **6-7% of groups were like
that**, and one such sample really did **teach the flip** from `Tooth 18 extruded` (greedy,
correct) to `impacted` (kept). (c) drives that regression to zero.

Why the threshold is 0.3 and not 0.5: rare findings come out together with other content, so
claim F1 becomes a partial score, and **raising the threshold removes the very findings we are
trying to teach first** (pilot: from 0.3 to 0.7, implant 10->3, caries 9->2, crown 51->24).

Normal regions = **maintenance samples**
--------------------------------------------------------------------------
The first attempt dropped normal regions from training (silence already scores full marks, so
adding them looked pointless). **That was wrong** — a model that learned only the 1,855 finding
samples picks up "speak" and loses the norm of silence: the reward on normal regions collapsed
from 0.8879 to 0.65 / 0.47 / 0.42 / 0.14 (by lr), taking the overall reward below baseline.

So **normal regions whose greedy reward is 1.0** are mixed back in as maintenance samples.
WARNING: the target is **not an empty string, it is the model's own greedy output** — on normal
regions the model currently writes silence 76.2% : presence statement 23.8%, and forcing an
empty target would **flip** those 459 cases into "do not speak". Neutralising presence
statements in scoring while forcing one side of them in training does not add up, and above all
**RadFact does not neutralise presence statements** (staying silent about something that is in
the reference costs recall).
= this adds no new information; it is **self-distillation that pins the current behaviour**.

**The greedy output is generated as well** for comparison (once per group, +1/6 cost). The dump
has to show "what it says now -> what we are about to teach" side by side to be judgeable.

usage:
  python -m toothfairy.pipeline.rft_rollout --n-patients 100 --out-dir experiments/analysis/rft-pilot
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from toothfairy.paths import REPO  # noqa: E402

from toothfairy.config import load_config                                    # noqa: E402
from toothfairy.data import ReportDataset, make_collate, read_split          # noqa: E402
from toothfairy.generation.postprocess import clean_gen                      # noqa: E402
from toothfairy.models import ReportModel                                    # noqa: E402
from toothfairy.rl.extract import FOV_AXES, extract_claims                   # noqa: E402
from toothfairy.rl.llm_judge import split_phrases                            # noqa: E402
from toothfairy.rl.presence import is_presence_line                          # noqa: E402
from toothfairy.rl.reward import region_reward                               # noqa: E402
from toothfairy.rl.rollout import (_epoch_order, eos_token_ids, reward_targets,  # noqa: E402
                                   rollout)
from toothfairy.training.trainer import to_device                            # noqa: E402

SWEEP = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _beats_greedy(r: dict) -> bool:
    """(c) Is the reward above greedy? If greedy is not comparable (nan), let it pass."""
    g = r.get("greedy_reward")
    return g is None or (isinstance(g, float) and g != g) or r["best_reward"] > g


def is_cls(r: dict) -> bool:
    """Is this an arch CLS slot (cls_upper / cls_lower)?"""
    return str(r.get("region_id", "")).startswith("cls_")


def select(fin: list[dict], threshold: float, beat_greedy: bool = True) -> list[dict]:
    """Finding groups that pass (a)+(b)+(c)."""
    out = [r for r in fin if r["best_reward"] >= threshold and r["beats_median"]]
    return [r for r in out if _beats_greedy(r)] if beat_greedy else out


def _maint_keep(greedy_text: str | None) -> bool:
    """May the greedy output of a normal region **stay as it is**? This decides the
    maintenance-sample target.

    Silence, or nothing but presence statements, is kept; anything else is over-generation and
    is corrected to silence (the rule verbatim: "silence stays silence, **a presence statement
    stays a presence statement**").

    WARNING: this used to be decided by `greedy_reward >= 0.999`. Under the older scorer a
    normal region was left with 0 candidate phrases once presence statements were neutralised,
    so it scored 1.0 and the two tests meant the same thing. But `--pure-radfact` does not erase
    presence statements and always gives **0.0** to a region with 0 reference claims — so that
    condition **can never hold any more**, and every maintenance target silently turned into
    silence. Measured: of 13,802 normal regions, **2,981 (21.6%)** flipped from "keep the
    presence statement" to "stay silent", taking the share of silent targets from 78.4% to 100%.
    Silence pressure from the maintenance samples bleeding into the finding regions is an
    already-measured failure path (an earlier round: silence rate on finding regions 7.4% ->
    29.4%, recall 0.3410 -> 0.2794).

    So the test is detached from the reward and computed from the rule itself. Being independent
    of the scoring method, it behaves identically under both scorers — on the 41,406 normal
    regions of three earlier scoring dumps it agreed with `greedy_reward >= 0.999` in
    **100.000%** of cases (0 disagreements).
    """
    return all(is_presence_line("- " + p) for p in split_phrases(greedy_text or ""))


def _maint_keep_row(r: dict) -> bool:
    """Maintenance-target test — filter-aware for a filtered dump, the previous rule otherwise.

    If the dump was scored with the filter on (`greedy_kept` is present), the test has to move
    with it to stay consistent. The challenge scorer **removes** normal confirmations,
    field-of-view statements and presence statements before scoring, so if greedy said only such
    things the region has 0 candidate phrases in the official metric = no penalty -> **keep it**.
    If a phrase survives the filter, it is something that really is scored in a region with an
    empty reference, i.e. an ungrounded finding -> correct it to silence.

    The previous rule (`_maint_keep`) treated only tooth presence statements as keepable. Under
    the filter that rule counts the other harmless phrases it erases (normal confirmations,
    field of view) as things to correct, and so teaches silence to dodge a penalty that does not
    exist.
    """
    if "greedy_kept" in r:
        return not r["greedy_kept"]
    return _maint_keep(r.get("greedy_text"))


def select_maintenance(rows: list[dict], n: int | None = None, seed: int = 42,
                       drop_noop: bool = False) -> list[dict]:
    """Normal-region samples — **all of them** (n=None) or n of them.

    Target rule:
      greedy at full marks -> that greedy output verbatim (**keep**: silence stays silence,
                              a presence statement stays a presence statement)
      greedy wrong         -> **corrected to the empty string (silence)** — the region is normal,
                              so anything said there is over-generation.
    The first round used only the 928 full-mark samples, which made finding:normal = 2:1 against
    the real ratio (1:1.82) and pushed the model to speak more overall (utterance rate on normal
    regions 30.0% -> 32.4%, RadFact precision 0.4243 -> 0.3917).

    `drop_noop=True`: **drop samples that are already silent and whose target is silence too.**
    In an earlier round these were identified as the source of a silence bias — 3,367 of the
    4,288 maintenance samples said "stay silent", and since only 335 of them were corrections,
    3,032 were **re-teaching what the model already does**; the LoRA is shared by every region,
    so that pressure bled into the finding regions. Measured: silence rate on finding regions
    7.4% -> **29.4%**, recall 0.3410 -> 0.2794, silence rate on normal regions 68.5% -> 83.1%.
    WARNING: corrections (greedy said something but the target is silence) and kept presence
    statements are **retained** — those differ from the model's current behaviour, so they do
    teach something.
    """
    pool = [r for r in rows if not r["is_finding"]]
    pool.sort(key=lambda r: (r["pid"], r["region_id"]))          # deterministic
    for r in pool:
        r["_target"] = (r["greedy_text"] or "") if _maint_keep_row(r) else ""
    if drop_noop:
        pool = [r for r in pool
                if (r["_target"] or "").strip() or (r["greedy_text"] or "").strip()]
    if n is not None and n < len(pool):
        pool = sorted(random.Random(seed).sample(pool, n), key=lambda r: (r["pid"], r["region_id"]))
    return pool


def rescore(rows: list[dict], excluded_axes) -> list[dict]:
    """Re-score the groups with the excluded axes applied (from the dump only, no re-generation).

    A region left with 0 scorable claims (e.g. one that only held field-of-view statements) is
    **dropped as a whole group** — greedy already gets it 100% right there, so no training sample
    can come out of it and it only burns rollout budget.
    """
    out = []
    for r in rows:
        ctx = {"pid": r["pid"], "region_id": r["region_id"]}
        if r["is_finding"]:
            g, _ = region_reward(r["greedy_text"] or "", r["gt"], ctx, excluded_axes=excluded_axes)
            if g != g:                                            # nan = outside the scoring set
                continue
            sc = [region_reward(t, r["gt"], ctx, excluded_axes=excluded_axes)[0] for t in r["texts"]]
            b = int(np.argmax(sc))
            r = {**r, "greedy_reward": g, "rewards": sc, "best_i": b, "best_reward": sc[b],
                 "median_reward": float(statistics.median(sc)),
                 "beats_median": sc[b] > float(statistics.median(sc)),
                 "zero_var": max(sc) == min(sc), "best_text": r["texts"][b]}
        out.append(r)
    return out


def report(rows: list[dict], out_dir: Path, threshold: float,
           beat_greedy: bool = True, maint_ratio: float = 0.5,
           drop_noop_maint: bool = False, exclude_cls: bool = False) -> int:
    """Group list -> threshold sweep, rejection reasons, make-up of the kept set + sample dump."""
    if exclude_cls:
        # Drop the arch CLS slots from training entirely.
        # Reason: only **245 of the 26,194 training GT sentences (0.94%)** fall on CLS, so the
        # model never learned what to say there, and on validation it **speaks in 14.5% of the
        # slots while only 8.9% have a reference** — it speaks more often than there is anything
        # to say. 12 of the 18 slots it spoke in have an empty reference, and the content is an
        # arch-level assertion such as `Complete edentulism of the mandible`, which **directly
        # contradicts the individual FDI slots describing the teeth one by one**. Pushing a slot
        # with no basis to teach through reinforcement only strengthens that contradiction.
        n0 = len(rows)
        rows = [r for r in rows if not is_cls(r)]
        print(f"[rft] CLS slots excluded — groups {n0} -> {len(rows)}", flush=True)
    fin = [r for r in rows if r["is_finding"]]
    print(f"\n[rft] groups {len(rows)} (finding {len(fin)} · "
          f"normal {len(rows) - len(fin)})", flush=True)

    lines = [f"# keep rate by threshold (finding-region groups · (b) above median + "
             f"{'(c) above greedy' if beat_greedy else '(c) off'})",
             f"  {'thresh':>6s} {'kept':>6s} {'rate':>7s} {'mean rew':>9s} "
             f"{'mean tok':>9s} {'delta vs greedy':>16s}"]
    for thr in SWEEP:
        keep = select(fin, thr, beat_greedy)
        if not keep:
            lines.append(f"  {thr:6.2f} {0:6d} {0.0:7.1%}")
            continue
        dg = [r["best_reward"] - r["greedy_reward"] for r in keep
              if not np.isnan(r["greedy_reward"])]
        lines.append(f"  {thr:6.2f} {len(keep):6d} {len(keep) / len(fin):7.1%} "
                     f"{np.mean([r['best_reward'] for r in keep]):9.4f} "
                     f"{np.mean([r['n_tok'] for r in keep]):9.1f} "
                     f"{np.mean(dg):+16.4f}")

    n_zero = sum(1 for r in fin if r["zero_var"])
    n_nomed = sum(1 for r in fin if not r["beats_median"])
    n_nogr = sum(1 for r in fin if r["beats_median"] and not _beats_greedy(r))
    lines += ["", "# rejection reasons (finding regions)",
              f"  reward variance 0 (all 6 tied) : {n_zero:5d} ({n_zero / max(len(fin),1):5.1%})",
              f"  (b) not above median          : {n_nomed:5d} ({n_nomed / max(len(fin),1):5.1%})",
              f"  (c) not above greedy          : {n_nogr:5d} ({n_nogr / max(len(fin),1):5.1%})"
              "  <- without this condition we teach regressions"]

    keep = select(fin, threshold, beat_greedy)
    comp = Counter()
    for r in keep:
        for c in extract_claims(r["best_text"], r["region_id"]):
            comp[f"{c[1]}:{c[2]}"] += 1
    lines += ["", f"# claim make-up of the kept set (threshold {threshold} · {len(keep)} groups · "
                  f"{sum(r['n_tok'] for r in keep):,} training tokens)",
              f"  {'claim (axis:value)':32s} {'count':>6s}"]
    for k, v in comp.most_common(20):
        lines.append(f"  {k:32s} {v:6d}")

    # --- maintenance samples from the normal regions ------------------------ #
    maint = select_maintenance(rows, None if maint_ratio <= 0 else int(round(len(keep) * maint_ratio)),
                               drop_noop=drop_noop_maint)
    n_sil = sum(1 for r in maint if not (r.get("_target") or "").strip())
    # A real correction = greedy said something but the target is silence. This used to be counted
    # with `greedy_reward < 0.999`, but `--pure-radfact` always gives 0.0 on normal regions, so
    # that expression counts **every** sample as a correction (one run marked all 12,818 of them
    # as 'correction'). Count it from the rule instead, so it is independent of the scoring
    # method — the rest are samples that re-teach silence the model already produces.
    n_fix = sum(1 for r in maint
                if (r.get("greedy_text") or "").strip() and not (r.get("_target") or "").strip())
    n_noop = len(maint) - n_fix - (len(maint) - n_sil)
    tok_f = sum(r["n_tok"] for r in keep)
    tok_m = sum(len((r["greedy_text"] or "").split()) for r in maint)   # rough count (words)
    lines += ["", f"# maintenance samples (normal regions · ratio {maint_ratio:.2f} of "
                  f"full-mark greedy)",
              f"  kept {len(maint)} — target silence {n_sil} ({n_sil / max(len(maint),1):.1%}) · "
              f"presence statement kept {len(maint) - n_sil} · "
              f"**corrections (greedy wrong -> silence) {n_fix}** · "
              f"already silent (no-op) {n_noop}",
              f"  finding:normal = {len(keep)} : {len(maint)} = "
              f"1 : {len(maint)/max(len(keep),1):.2f} "
              f"(the ratio actually fed is 1 : 1.82)",
              f"  WARNING: finding targets average {tok_f / max(len(keep),1):.1f} tokens against "
              f"{tok_m / max(len(maint),1):.1f} words for maintenance targets — at equal "
              f"counts **their loss contribution is far smaller**. Correct with --maint-weight "
              f"during fine-tuning."]

    text = "\n".join(lines)
    print("\n" + text, flush=True)
    with (out_dir / "rft_selected.jsonl").open("w") as f:
        for r in keep:
            f.write(json.dumps({"kind": "finding", "pid": r["pid"], "region_id": r["region_id"],
                                "gt": r["gt"], "greedy": r["greedy_text"],
                                "chosen": r["best_text"], "reward": r["best_reward"],
                                "greedy_reward": r["greedy_reward"]}, ensure_ascii=False) + "\n")
        for r in maint:
            f.write(json.dumps({"kind": "normal", "pid": r["pid"], "region_id": r["region_id"],
                                "gt": r["gt"], "greedy": r["greedy_text"],
                                "chosen": r.get("_target", r["greedy_text"]),
                                "reward": r["greedy_reward"],
                                "greedy_reward": r["greedy_reward"]}, ensure_ascii=False) + "\n")
    (out_dir / "rft_report.txt").write_text(text + "\n")
    print(f"\n[rft] wrote {out_dir}/rft_selected.jsonl · rft_report.txt", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="toothfairy/configs/sinus_rft_rollout.yaml")
    ap.add_argument("--init-from", default="", help="empty = grpo.init_from from the config")
    ap.add_argument("--split", default="data/splits/train.txt")
    ap.add_argument("--n-patients", type=int, default=100, help="0 = all")
    ap.add_argument("--threshold", type=float, default=0.3, help="selection rule (a): reward floor")
    ap.add_argument("--no-beat-greedy", action="store_true",
                    help="turn off condition (c), reward above greedy — for a control run")
    ap.add_argument("--maint-ratio", type=float, default=0.0,
                    help="maintenance samples / finding samples ratio. **0 = keep all of them**")
    ap.add_argument("--exclude-cls", action="store_true",
                    help="exclude the arch CLS slots (cls_upper/cls_lower) from the training "
                         "samples (0.94% of the training GT, and they contradict the "
                         "individual FDI slots).")
    ap.add_argument("--drop-noop-maint", action="store_true",
                    help="drop maintenance samples that are already silent and whose target is "
                         "silence too. An earlier round traced a silence bias to them — 3,032 "
                         "samples re-teaching what the model already does.")
    ap.add_argument("--exclude-fov", action="store_true",
                    help="exclude the field-of-view axes from scoring "
                         "(greedy gets them 100%% right)")
    ap.add_argument("--out-dir", default="experiments/analysis/rft-pilot")
    ap.add_argument("--all-regions", action="store_true",
                    help="generate **all** 38 slots (no region sampling). Removes sampling seed, "
                         "coverage and sample-ratio bias in one go — the training distribution "
                         "then matches the inference distribution.")
    ap.add_argument("--reward-gt", choices=("", "union", "sampled"), default="",
                    help="override grpo.reward_gt from the config. Empty = the config value.")
    ap.add_argument("--groups", default="",
                    help="path of the group file to merge. Empty = rft_groups.shard*.jsonl in "
                         "out-dir. Use it to feed in a dump re-scored by the entailment judge "
                         "(produced by `toothfairy.cli.rft rescore`, which runs where RadFact is installed, "
                         "where torch is not installed).")
    # Generation is per patient, so it shards straight over GPUs (same order split by stride,
    # so the shards never overlap).
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--report-only", action="store_true",
                    help="skip generation, merge the shard files in out-dir and only report")
    a = ap.parse_args(argv)

    cfg = load_config(REPO / a.config if not Path(a.config).is_absolute() else a.config)
    gc_ = cfg.grpo
    if a.reward_gt:
        gc_.reward_gt = a.reward_gt
    if a.all_regions:
        # There is never a slot beyond 38, so simply lifting the caps makes sample_region_ids
        # return all of them.
        # preserve_ratio_when_short trims the normal count down to the finding count, so it must
        # be switched off.
        gc_.n_finding_regions = 38
        gc_.n_normal_regions = 38
        gc_.preserve_ratio_when_short = False
    out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else REPO / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    if a.report_only:                                   # merge the shards and only report
        if a.groups:
            src = Path(a.groups) if Path(a.groups).is_absolute() else REPO / a.groups
            rows = [json.loads(l) for l in src.open() if l.strip()]
            print(f"[rft] group file {src.name}: {len(rows)}", flush=True)
        else:
            rows = [json.loads(l) for f in sorted(out_dir.glob("rft_groups.shard*.jsonl"))
                    for l in f.open()]
            print(f"[rft] merged shards: {len(rows)} groups", flush=True)
        if not rows:
            raise FileNotFoundError(f"no groups to read in {out_dir}")
        if a.exclude_fov and a.groups:
            # Axis exclusion is a rule defined over extracted claims, so it does not carry over
            # to sentence-level entailment judging. Ignoring it quietly would look as if it had
            # been applied, so refuse instead.
            raise SystemExit("--exclude-fov cannot be used with a dump re-scored by the "
                             "entailment judge (axis exclusion is a claim-level notion of the "
                             "rule-based scorer)")
        if a.exclude_fov:
            n0 = len(rows)
            rows = rescore(rows, FOV_AXES)
            print(f"[rft] re-scored without the field-of-view axes: {n0} -> {len(rows)} groups "
                  f"(dropped {n0 - len(rows)} outside the scoring set)", flush=True)
        return report(rows, out_dir, a.threshold, not a.no_beat_greedy, a.maint_ratio,
                      drop_noop_maint=a.drop_noop_maint, exclude_cls=a.exclude_cls)

    model = ReportModel(cfg).to(dev).eval()
    ipath = Path(a.init_from or gc_.init_from)
    ipath = ipath if ipath.is_absolute() else REPO / ipath
    ck = torch.load(ipath, map_location="cpu", weights_only=False)
    sd = ck.get("trainable_state", ck)
    # A schema axis merge took SOFT_DIM from 31 to 22, so a checkpoint trained before it differs
    # in shape on `realizer.prob_proj` alone. When soft probability injection is off that tensor
    # is not on the inference path, so it is skipped while loading. **If injection is on, or any
    # other tensor mismatches, fail at once** — that stops the run from quietly becoming a
    # different model.
    cur = model.state_dict()
    dropped = [k for k, v in sd.items()
               if k in cur and tuple(cur[k].shape) != tuple(v.shape)]
    if dropped:
        bad = [k for k in dropped if not k.startswith("realizer.prob_proj")]
        if bad or model._soft_on():
            raise RuntimeError(
                f"shape mismatch — different architecture "
                f"(soft={model._soft_on()}): {bad or dropped}")
        print(f"[rft] skipped on shape mismatch (soft injection off, "
              f"unused at inference): {dropped}", flush=True)
        sd = {k: v for k, v in sd.items() if k not in dropped}
    _miss, unexp = model.load_state_dict(sd, strict=False)
    if unexp:
        raise RuntimeError(f"init_from does not match this model architecture: {list(unexp)[:5]}")
    tok = model.realizer.tokenizer
    eos_ids, pad_id = eos_token_ids(tok)
    print(f"[rft] start policy {ipath.parent.name} · G={gc_.group_size} · T={gc_.temperature} "
          f"· top_p={gc_.top_p} · {gc_.n_finding_regions} finding:{gc_.n_normal_regions} normal "
          f"· reward_gt={gc_.reward_gt} · threshold {a.threshold}", flush=True)

    pids = read_split(REPO / a.split)
    ds = ReportDataset(pids, cfg, repo_root=REPO, tokenizer=None, resample=False)
    collate = make_collate(cfg.perception.feat_dim)
    union = ds.union_targets_by_pid() if gc_.reward_gt == "union" else None
    n_pat = len(ds) if a.n_patients <= 0 else min(a.n_patients, len(ds))
    order = _epoch_order(len(ds), 0, gc_.seed)[:n_pat]
    if a.num_shards > 1:
        order = order[a.shard::a.num_shards]
    print(f"[rft] {a.split} {len(ds)} patients, taking {n_pat} · shard {a.shard}/{a.num_shards} "
          f"-> {len(order)} patients in this process", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for n, idx in enumerate(order):
            batch = to_device(collate([ds[idx]]), dev)
            pid = batch["pid"][0]
            sel = lambda: random.Random(f"rft:{gc_.seed}:{idx}")   # same regions in both rollouts
            # Reference to store in the dump — the **same dict** the rollout scored against
            # (cleaning and neutralisation included). Reading the union dict directly used to
            # (1) store gt as an empty string throughout when reward_gt=sampled, because union is
            # None there, which makes re-scoring impossible, and (2) even for union, keep the raw
            # text from before _clean_targets, which then disagreed with what was scored.
            dump_targets = reward_targets(batch, gc_, union)
            g_greedy = rollout(model, batch, gc_, sel(), eos_ids, pad_id, greedy=True,
                               union_targets=union)
            g_sample = rollout(model, batch, gc_, sel(), eos_ids, pad_id, greedy=False,
                               union_targets=union)
            gtxt = {g.region_id: (clean_gen(tok.decode(g.comps[0], skip_special_tokens=True)),
                                  g.rewards[0]) for g in g_greedy if g.comps}
            for g in g_sample:
                texts = [clean_gen(tok.decode(c, skip_special_tokens=True)) for c in g.comps]
                r = list(g.rewards)
                b = int(np.argmax(r))
                med = float(statistics.median(r))
                gt_txt, gt_r = gtxt.get(g.region_id, ("", float("nan")))
                rows.append({
                    "pid": pid, "region_id": g.region_id, "is_finding": bool(g.is_finding),
                    "gt": dump_targets.get(g.region_id, ""),
                    "rewards": r, "best_i": b, "best_reward": float(r[b]),
                    "median_reward": med, "beats_median": bool(r[b] > med),
                    "zero_var": bool(max(r) == min(r)),
                    "best_text": texts[b], "texts": texts,
                    "greedy_text": gt_txt, "greedy_reward": gt_r,
                    "n_tok": int(g.comps[b].numel()),
                })
            if (n + 1) % 10 == 0:
                print(f"[rft] {n + 1}/{len(order)} patients · {len(rows)} groups "
                      f"({time.time() - t0:.0f}s)", flush=True)

    with (out_dir / f"rft_groups.shard{a.shard}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rft] shard {a.shard}: {len(rows)} groups -> "
          f"{out_dir}/rft_groups.shard{a.shard}.jsonl ({time.time() - t0:.0f}s)", flush=True)
    if a.num_shards > 1:
        print("[rft] when the other shards finish, merge with --report-only.", flush=True)
        return 0
    return report(rows, out_dir, a.threshold, not a.no_beat_greedy, a.maint_ratio,
                  drop_noop_maint=a.drop_noop_maint, exclude_cls=a.exclude_cls)


if __name__ == "__main__":
    raise SystemExit(main())
