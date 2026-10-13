"""Compute RadFact locally against an LM Studio (or Ollama) endpoint.

Grand Challenge's evaluation container has no internet, so `evaluate.py` skips
RadFact entirely there (`disabled_on_grand_challenge`) and the leaderboard shows
only BLEU-4 and METEOR. The clinical score -- 80% of the final ranking -- has to
be measured here.

Reference phrases are cached to disk: the reference side never changes, and
parsing is half the LLM cost, so every rerun after the first is ~2x cheaper.

Usage:
    # LM Studio serving on :1234 with any instruct model loaded
    python tf4/run_radfact.py --n 30 --model <model-id>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "work", "radfact_ref_phrases.json")


def build_predictions(cases: list[str]) -> dict[str, str]:
    """Run the real pipeline to produce candidate reports."""
    import nibabel as nib

    from tf4.mask_features import derive_findings
    from tf4.render import render
    from tf4.trinity_infer import (
        from_training_frame, load_model, predict_volume, resample_to_target,
        to_canonical, to_training_frame,
    )

    model = load_model(os.path.join(ROOT, "algorithm/model/trinity_weights_fp16.pth"))
    norm = json.load(open(os.path.join(ROOT, "algorithm/model/norm_stats.json")))

    out = {}
    for i, case in enumerate(cases, 1):
        data, spacing = to_canonical(nib.load(f"{ROOT}/toothfairy4_v03/{case}/cbct/volume.nii.gz"))
        data, spacing = resample_to_target(data, spacing)
        mask = from_training_frame(predict_volume(model, to_training_frame(data), norm))
        out[case] = render(derive_findings(mask, spacing=float(np.mean(spacing)), case_id=case))
        print(f"  [{i}/{len(cases)}] {case}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default=os.getenv("RADFACT_MODEL", "local-model"))
    ap.add_argument("--base-url", default=os.getenv("RADFACT_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--prior-only", action="store_true",
                    help="score the fallback report instead of the pipeline (baseline)")
    ap.add_argument("--drop", default="", help="comma-separated slots to suppress at render "
                                               "time for ablations, e.g. canal_contact,impacted")
    ap.add_argument("--from-cache", action="store_true",
                    help="render from work/pred_findings.json instead of re-running the "
                         "segmentation -- needs no GPU, so the card is free for the LLM")
    args = ap.parse_args()

    from radfact_lite import ModelConfig, PipelineModels, RadFactLitePipeline, ReportType
    import radfact_lite.clients as rf_clients

    # radfact_lite never sets max_tokens. Against a hosted API that is fine, but a
    # local model that falls into a repetition loop ("own own own ...") then runs
    # unbounded and returns multi-KB garbage that fails JSON parsing and kills the
    # whole run. Reports are ~100 words, so 2000 tokens is generous.
    rf_clients._LOWEST_SETTINGS = {**rf_clients._LOWEST_SETTINGS, "max_tokens": 2000}

    # Reasoning models served by LM Studio put their whole answer in
    # `reasoning_content` and leave `content` empty, so radfact_lite sees "" and
    # raises. The JSON is there, just in the other field.
    _orig_create = rf_clients._ChatJSONClient.complete_json

    def _complete_json(self, messages, response_format):
        try:
            return _orig_create(self, messages, response_format)
        except ValueError:
            create_kwargs = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "response_format": response_format,
                "max_tokens": 2000,
                "temperature": 0,
            }
            msg = self.client.chat.completions.create(**create_kwargs).choices[0].message
            body = (msg.content or "") or (getattr(msg, "reasoning_content", "") or "")
            return json.loads(body)

    rf_clients._ChatJSONClient.complete_json = _complete_json

    cases = [c for c in sorted(os.listdir(f"{ROOT}/toothfairy4_v03")) if c.startswith(("P", "F", "S"))]
    random.seed(0)
    cases = random.sample(cases, args.n)

    references = {}
    for case in cases:
        reports = sorted(glob.glob(f"{ROOT}/toothfairy4_v03/{case}/reports_en/*.txt"))
        if reports:
            references[case] = open(reports[0]).read().strip()
    cases = [c for c in cases if c in references]

    if args.prior_only:
        from tf4.render import render
        from tf4.schema import Findings
        prior = render(Findings(), use_priors=True)
        candidates = {c: prior for c in cases}
        print(f"scoring PRIOR-ONLY baseline on {len(cases)} cases")
    elif args.from_cache:
        from tf4.render import render
        from tf4.schema import Findings
        cached = json.load(open(os.path.join(ROOT, "work", "pred_findings.json")))
        candidates = {}
        for case in cases:
            rec = cached.get(case)
            if rec is None:
                continue
            f = Findings(
                case_id=case,
                teeth_present={int(k): v for k, v in rec["teeth"].items()},
                canal_contact={k: int(v) for k, v in rec["canal_contact"].items()},
                impacted=set(rec["impacted"]),
                maxilla_coverage=rec["maxilla"],
                canal_course=rec["canal_course"],
                edentulism=rec["edentulism"],
                has_implant=rec["has_implant"],
                has_bridge=rec["has_bridge"],
                has_crown=rec["has_crown"],
                sinuses_included=rec.get("sinuses"),
                crowns=set(rec.get("crowns") or []),
            )
            for slot in (s.strip() for s in args.drop.split(",") if s.strip()):
                if slot == "canal_contact":
                    f.canal_contact = {}
                elif slot == "impacted":
                    f.impacted = set()
                elif slot == "absent_teeth":
                    f.teeth_present = {k: v for k, v in f.teeth_present.items() if v}
                elif slot == "restorations":
                    f.has_implant = f.has_bridge = f.has_crown = None
                else:
                    raise SystemExit(f"unknown --drop slot: {slot}")
            candidates[case] = render(f, use_priors=True)
        cases = [c for c in cases if c in candidates]
        print(f"rendered {len(candidates)} cases from cached findings (no GPU used)")
    else:
        print(f"generating pipeline reports for {len(cases)} cases")
        candidates = build_predictions(cases)

    # provider="ollama" just means "OpenAI-compatible endpoint with a dummy key",
    # which is exactly what LM Studio exposes.
    cfg = ModelConfig(
        model=args.model, provider="ollama", base_url=args.base_url,
        timeout=args.timeout, max_retries=1,
    )
    pipeline = RadFactLitePipeline(
        models=PipelineModels(parse_model=cfg, entailment_model=cfg, filtering_model=cfg),
        report_type=ReportType.TOOTHFAIRY,
    )

    print(f"\nRunning RadFact via {args.base_url} (model={args.model}) -- this is the slow part.")
    aggregate, per_case = pipeline.compute_radfact(
        candidates_by_id=candidates,
        references_by_id={c: references[c] for c in cases},
        is_narrative_text=True,
        remove_normal_findings=False,
    )

    agg = asdict(aggregate)
    print("\n=== RadFact ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    clinical = agg.get("logical_f1", 0.0)
    print(f"\n  clinical score (0.8 weight) contribution: {0.8*clinical:.4f}")

    out = os.path.join(ROOT, "work", "radfact_results.json")
    with open(out, "w") as fh:
        json.dump({"aggregate": agg, "per_case": [asdict(r) for r in per_case]}, fh, indent=2)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
