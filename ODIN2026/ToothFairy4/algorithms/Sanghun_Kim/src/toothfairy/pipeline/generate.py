"""End-to-end validation report generation + scoring input (predictions.csv).

Whichever regions the model decides to emit (self-gating) is what gets scored: every imaged
region — normal teeth included — is fed to the LLM and generated; a normal slot emits EOS
immediately (empty output), a slot with a finding emits text. Empty outputs are dropped from
the report.

The generated region texts are concatenated into a single report as bullets in a fixed anatomical
order (no section headers), and the reference text is written as the target in
predictions.csv/jsonl. BLEU/METEOR are computed by `toothfairy.cli.eval captioning` and RadFact by the
RadFact scorer, both separately.

usage:
  python -m toothfairy.pipeline.generate --run-dir models/<model>       # --limit N = first N
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toothfairy.config import load_config                                   # noqa: E402
from toothfairy.data import ReportDataset, make_collate, read_split         # noqa: E402
from toothfairy.models import ReportModel                                   # noqa: E402
from toothfairy.training.trainer import to_device                          # noqa: E402
from toothfairy.schema.claims import FDI_ALL, AXES                          # noqa: E402
from toothfairy.losses import axis_soft_probs                              # noqa: E402
from toothfairy.generation.postprocess import clean_gen                    # noqa: E402,F401

from toothfairy.paths import REPO  # noqa: E402

# Fixed anatomical order of the assembled report: maxilla -> mandible -> canals (R,L) -> teeth.
_NONTEETH_ORDER = ["maxilla", "mandible", "nerve_right", "nerve_left"]
# 38 anatomical tokens = 4 non-teeth + 2 arch summaries + 32 teeth.
# Evaluation generates for **every** token; the set is never picked from the ground truth, so
# there is no leakage. cls_upper/cls_lower carry the broad per-arch dental summary.
ALL_TOKENS = _NONTEETH_ORDER + ["cls_upper", "cls_lower"] + [f"fdi:{f}" for f in FDI_ALL]


def tokens_for(cfg) -> list:
    """With `encoder.drop_arch_cls` the two arch summary slots are dropped, leaving 36. Querying a
    slot the model was never fed during training is an out-of-distribution input and silently
    produces unrelated text."""
    if bool(getattr(cfg.encoder, "drop_arch_cls", False)):
        return [t for t in ALL_TOKENS if t not in ("cls_upper", "cls_lower")]
    return list(ALL_TOKENS)
_STATE_NOT_AVAILABLE = AXES[0].categories.index("not_available")


def load_reference(pid: str) -> tuple[str, str]:
    """(reference_text, source). One reference text per patient.

    Priority: (1) the per-patient union of all reports (`experiments/GT/report_GT`, written by
    build_report_gt_union.py), (2) the manually reconciled text (`reconciled_final`), (3) the
    merged reference (`report_GT_val_merged`, build_val_merged_reports.py), (4) a single original
    English report.

    (1) comes first because (2) is untracked and does not follow the code onto another machine. If
    (1) is missing and we silently fall back to (4), a multi-report patient is scored against only
    its first report, which is exactly the hole (2) and (3) close. The union variants (1) and (3)
    keep any contradiction between the reports in the reference (per-patient conflict counts are
    recorded in `_summary.json`).

    (1) and (3) were verified to produce character-identical reference text on every validation
    patient, so (3) is kept only as a fallback for when (1) is absent.
    """
    rg = REPO / "experiments/GT/report_GT" / f"{pid}.txt"
    if rg.exists():
        return " ".join(rg.read_text().split()), "report_gt_union"
    rf = REPO / "experiments/audit_archive/multireport-conflict/reconciled_final" / f"{pid}.json"
    if rf.exists():
        d = json.load(open(rf))
        sents = [s.strip() for s in d.get("sentences", []) if s and s.strip()]
        return " ".join(sents), "reconciled"
    mg = REPO / "experiments/GT/report_GT_val_merged" / f"{pid}.json"
    if mg.exists():
        d = json.load(open(mg))
        sents = [s.strip() for s in d.get("sentences", []) if s and s.strip()]
        return " ".join(sents), d.get("source", "merged_union")
    rep_dir = REPO / "data" / pid / "reports_en"
    txts = sorted(rep_dir.glob("*.txt")) if rep_dir.exists() else []
    if txts:
        return " ".join(txts[0].read_text().split()), "single_report"
    return "", "missing"


# clean_gen lives in toothfairy.generation.postprocess (behaviour unchanged) so that the string the
# RFT reward scores and the string the evaluation scores are built by the same function. It is
# re-exported here, so `from toothfairy.pipeline.generate import clean_gen` keeps working.


def _region_order_key(rid: str):
    """Sort key for the assembled report: non-teeth < teeth (FDI order) < arch summaries."""
    if rid in _NONTEETH_ORDER:
        return (0, _NONTEETH_ORDER.index(rid))
    if rid.startswith("fdi:"):
        f = int(rid.split(":")[1])
        return (1, FDI_ALL.index(f) if f in FDI_ALL else 99)
    return (2, 0)


def assemble_report(region_texts: dict[str, str]) -> str:
    """{region_id: generated_text} (non-empty entries only) -> bullets in the fixed order, no
    headers. The model already emits '- a - b' bullets, so split them per line and merge."""
    lines: list[str] = []
    for rid in sorted(region_texts, key=_region_order_key):
        t = " ".join(region_texts[rid].replace("\n", " ").split()).strip()
        if not t:
            continue
        for seg in t.split(" - "):                        # split the model's own bullets
            seg = seg.lstrip("-").strip()
            if seg:
                lines.append(f"- {seg}")
    return "\n".join(lines).strip()


def select_region_ids(cfg) -> list[str]:
    """The region_ids to generate: **all** anatomical tokens, letting the model self-gate
    through EOS.

    This used to read the batch["region_targets"] keys, which are derived from the state axis of
    the ground truth, and so told the model which teeth were imaged. The hidden test set has no
    such information and inference already queries all 32 teeth, so evaluation is pinned to the
    full token list as well."""
    return tokens_for(cfg)


def score_regions(gt_targets: dict, gen_texts: dict) -> list[dict]:
    """Per-token scoring rows. Every token is generated, and we look at
      - tokens whose GT carries finding text -> was a finding actually generated?
      - tokens whose GT is empty (normal or out of field) -> was EOS (empty output) emitted?
    gt_targets is the output of build_region_targets_eos (empty string = EOS expected). Computed
    independently of the assembled-report scoring (RadFact/BLEU/METEOR)."""
    rows = []
    for rid in ALL_TOKENS:
        gt = (gt_targets.get(rid) or "").strip()
        gen = (gen_texts.get(rid) or "").strip()
        gt_has, gen_has = bool(gt), bool(gen)
        if gt_has and gen_has:      cat = "hit"          # finding present and spoken
        elif gt_has and not gen_has: cat = "miss"         # finding present but silent (false EOS)
        elif not gt_has and gen_has: cat = "false_alarm"  # no finding but spoken (hallucination)
        else:                        cat = "correct_silence"
        rows.append({"region_id": rid, "category": cat, "gt": gt, "gen": gen})
    return rows


def _region_group(rid: str) -> str:
    return "teeth" if rid.startswith("fdi:") else ("nerve" if rid.startswith("nerve") else rid)


def aggregate_region_metrics(all_rows: list[dict]) -> dict:
    """All per-region rows -> per-group gate metrics (precision/recall/F1) and raw counts."""
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in all_rows:
        agg[_region_group(r["region_id"])][r["category"]] += 1
        agg["ALL"][r["category"]] += 1
    out = {}
    for g, c in agg.items():
        hit, miss, fa = c["hit"], c["miss"], c["false_alarm"]
        prec = hit / (hit + fa) if (hit + fa) else 0.0
        rec = hit / (hit + miss) if (hit + miss) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[g] = {"hit": hit, "miss": miss, "false_alarm": fa,
                  "correct_silence": c["correct_silence"],
                  "gate_precision": round(prec, 4), "gate_recall": round(rec, 4),
                  "gate_f1": round(f1, 4)}
    return out


def build_infer_specs(model, ctx_i, aux_i, region_ids):
    """region_ids -> generation specs (tokens, region_id, prob). No GT target (inference)."""
    soft = model._soft_on()
    prob_i = axis_soft_probs(aux_i, model._logit_slices) if (soft and aux_i is not None) else None
    ptext = model._prob_text_on()
    nt_i = None
    if ptext:
        nt_i = {k: v[0] for k, v in
                model.findings_head.nonteeth_logits(ctx_i.unsqueeze(0)).items()}
    specs = []
    for rid in region_ids:
        tokens, prob = model._slot_tokens(ctx_i, rid, prob_i, soft)
        spec = {"tokens": tokens, "region_id": rid, "prob": prob}
        if ptext:
            spec["prob_text"] = model._slot_prob_text(rid, aux_i, nt_i)
        specs.append(spec)
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="experiments/feature_cache/<arm>")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--limit", type=int, default=0, help=">0 = first N patients only (smoke test)")
    # For overfitting diagnostics: scoring the patients the model was trained on with the same
    # checkpoint means swapping the evaluation set. The defaults are unchanged, so existing
    # invocations behave exactly as before.
    ap.add_argument("--split-file", default="data/splits/val.txt",
                    help="patients to evaluate (default: val); change only to score training data")
    ap.add_argument("--out-dir", default="",
                    help="output location (default: run-dir); pass one to keep run-dir outputs")
    args = ap.parse_args(argv)

    run_dir = (REPO / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out_dir = run_dir if not args.out_dir else (
        (REPO / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, cfg_path = run_dir / args.ckpt, run_dir / "config.resolved.yaml"
    for p in (ckpt_path, cfg_path):
        if not p.exists():
            ap.error(f"file not found: {p}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(cfg_path)

    model = ReportModel(cfg, device=device).to(device).eval()
    tok = model.realizer.tokenizer
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["trainable_state"]
    # Merging claim axes changed SOFT_DIM, so checkpoints trained before that mismatch only on
    # `realizer.prob_proj`. When soft probability injection is off that tensor is not on the
    # inference path, so it is dropped. If injection is on, or any other tensor mismatches, fail
    # immediately - scoring silently with a different model is worse than not scoring at all.
    cur = model.state_dict()
    dropped = [k for k, v in sd.items()
               if k in cur and tuple(cur[k].shape) != tuple(v.shape)]
    if dropped:
        bad = [k for k in dropped if not k.startswith("realizer.prob_proj")]
        if bad or model._soft_on():
            raise RuntimeError(
                f"shape mismatch - different architecture "
                f"(soft={model._soft_on()}): {bad or dropped}")
        print(f"[eval] dropped on shape mismatch (soft injection off, "
              f"unused at inference): {dropped}", flush=True)
        sd = {k: v for k, v in sd.items() if k not in dropped}
    _, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{args.ckpt} unexpected keys: {list(unexpected)[:5]}")
    try:
        model.realizer.llm.config.use_cache = True                 # generation speed
    except Exception:
        pass

    split_path = (REPO / args.split_file) if not Path(args.split_file).is_absolute() \
        else Path(args.split_file)
    val_pids = read_split(split_path)
    if args.split_file != "data/splits/val.txt":
        print(f"[eval] evaluation set replaced: {split_path} "
              f"({len(val_pids)} patients) → {out_dir}", flush=True)
    val_ds = ReportDataset(val_pids, cfg, repo_root=REPO, tokenizer=tok, resample=False)
    collate = make_collate(cfg.perception.feat_dim)
    if len(val_ds) == 0:
        raise FileNotFoundError("no validation samples - check the feature cache and GT paths.")
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    # The chat model leaves config.eos_token_id=None, so generation never stops at turn end;
    # pass eos=<|im_end|> explicitly to stop it.
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list({im_end, tok.eos_token_id})
    gkw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
               repetition_penalty=args.rep_penalty, num_beams=1,
               eos_token_id=eos_ids, pad_token_id=(tok.pad_token_id or im_end))
    rows, records, region_rows = [], [], []
    n_total = len(val_ds) if args.limit <= 0 else min(args.limit, len(val_ds))
    for i, b in enumerate(val_ld):
        if args.limit and i >= args.limit:
            break
        pid = b["pid"][0]
        target, src = load_reference(pid)
        b = to_device(b, device)
        with torch.no_grad():
            ctx, aux = model._encode(b)
            region_ids = select_region_ids(cfg)
            specs = build_infer_specs(model, ctx[0], aux[0] if aux is not None else None, region_ids)
            region_texts = {}
            for s in specs:
                gen = model.realizer.generate_region(
                    s["tokens"], s["region_id"], prob_text=s.get("prob_text"), **gkw)
                txt = clean_gen(tok.decode(gen[0], skip_special_tokens=True))
                if txt:                                            # empty output = EOS self-gate
                    region_texts[s["region_id"]] = txt
        prediction = assemble_report(region_texts)
        # per-region scoring over all tokens - independent of the assembled-report scoring
        rrows = score_regions(b["region_targets_text"][0], region_texts)
        for rr in rrows:
            rr["pid"] = pid
        region_rows.extend(rrows)
        nmiss = sum(1 for r in rrows if r["category"] == "miss")
        nfa = sum(1 for r in rrows if r["category"] == "false_alarm")
        rows.append({"example_id": pid, "prediction": prediction, "target": target})
        records.append({"pid": pid, "source": src, "n_fed": len(specs),
                        "n_emitted": len(region_texts), "region_texts": region_texts,
                        "prediction": prediction, "target": target})
        print(f"[{i + 1}/{n_total}] {pid}({src}): fed {len(specs)} → emitted "
              f"{len(region_texts)} · miss {nmiss} · false-alarm {nfa} · "
              f"pred {len(prediction)}c gt {len(target)}c", flush=True)

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "prediction", "target"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    (out_dir / "per_region.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in region_rows) + "\n")
    rm = aggregate_region_metrics(region_rows)
    (out_dir / "per_region_metrics.json").write_text(
        json.dumps(rm, ensure_ascii=False, indent=2))
    print("\n[per-region gate] per-token finding emission (all tokens generated)")
    print(f"  {'group':10s} {'hit':>5s} {'miss':>5s} {'falseAl':>7s} {'silence':>7s} "
          f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
    for g in sorted(rm, key=lambda x: (x != "ALL", x)):
        v = rm[g]
        print(f"  {g:10s} {v['hit']:5d} {v['miss']:5d} {v['false_alarm']:7d} "
              f"{v['correct_silence']:7d} {v['gate_precision']:6.3f} "
              f"{v['gate_recall']:6.3f} {v['gate_f1']:6.3f}")
    m = ck.get("metrics", {})
    print(f"\n[generate] {cfg.train.run_name} ({args.ckpt}, ep {m.get('epoch','?')}): "
          f"{len(rows)} reports → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
