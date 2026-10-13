#!/usr/bin/env python
"""Re-score an RFT rollout dump with the per-region entailment judge.

Split out from `pipeline/rft_rollout.py` because the two halves want different
environments: that module imports torch and the model, this one needs the RadFact prompt
and an OpenAI client, and the two dependency sets do not have to live in the same
environment. Keeping them apart also means re-scoring costs no GPU beyond the judge itself.

    python -m toothfairy.pipeline.rft_rescore ...                            (this)
    python -m toothfairy.pipeline.rft_rollout --report-only --groups ...     (the selection)

No regeneration: the 6 samples and the greedy output of every group are already text
in the dump, so a new scorer only has to read them.  That is the 3 hours saved.

Usage:
    API_KEY_CHAT_0=dummy python -m toothfairy.pipeline.rft_rescore \
        --out-dir experiments/analysis/rft --url http://localhost:8001/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402

from toothfairy.rl.llm_judge import EntailmentJudge, FindingFilter, rescore_llm  # noqa: E402


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="directory holding rft_groups.shard*.jsonl")
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--out-name", default="rft_groups_llmjudge.jsonl")
    ap.add_argument("--pure-radfact", action="store_true",
                    help="score with the RadFact entailment judgement alone — a region with "
                         "zero reference claims no longer gets a full mark for staying silent, "
                         "and tooth presence statements are not stripped.")
    ap.add_argument("--filter-cache", default=None,
                    help="cache of the official finding_filter decisions (produced by "
                         "`toothfairy.cli.rft filter-cache`). When given, both reference and candidate are "
                         "passed through the filter before scoring, so only the sentences the "
                         "official metric sees enter the reward. If the reference empties out "
                         "entirely, the region stops counting as a finding region and moves to "
                         "the maintenance samples.")
    ap.add_argument("--fbeta", type=float, default=1.0,
                    help="beta of F_beta. 2 weights recall four times as heavily as precision. "
                         "The judgements are cached, so this only recomputes and never calls "
                         "the LLM.")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else REPO / a.out_dir
    shards = sorted(out_dir.glob("rft_groups.shard*.jsonl"))
    if not shards:
        raise SystemExit(f"no rft_groups.shard*.jsonl found in {out_dir}")
    rows = [json.loads(line) for s in shards for line in s.open() if line.strip()]
    print(f"[rescore] {len(shards)} shards · {len(rows):,} groups", flush=True)

    empty_gt = sum(1 for r in rows if r["is_finding"] and not (r["gt"] or "").strip())
    if empty_gt:
        # This condition once cost an 8-hour rollout; fail immediately instead of re-scoring.
        raise SystemExit(f"{empty_gt} finding groups have an empty gt — this dump cannot be "
                         "re-scored. Check that the rollout that produced the dump writes "
                         "reward_targets()")

    # The judge cache must be kept separate per scoring mode: for the same (sentence,
    # reference) pair, what is actually judged differs depending on whether presence statements
    # are neutralised. Sharing one file would leak old judgements into the new scoring.
    cache = out_dir / ("judge_cache_pure.json" if a.pure_radfact else "judge_cache.json")
    judge = EntailmentJudge(url=a.url, concurrency=a.concurrency,
                            cache_path=cache, beta=a.fbeta,
                            pure_radfact=a.pure_radfact)
    if a.pure_radfact:
        print("[rescore] RadFact entailment only — no free mark for silence · presence kept",
              flush=True)
    # With the filter on, the judge cache still uses the same file: the key is a content hash
    # of (reference list, hypothesis sentence), so filtering the list changes the key too.
    # Nothing stale is reused, while pairs that do coincide are reused and save judge calls.
    ffilter = None
    if a.filter_cache:
        p = Path(a.filter_cache)
        ffilter = FindingFilter(p if p.is_absolute() else REPO / p)
        print(f"[rescore] official finding_filter applied — {len(ffilter.cache):,} cached "
              "decisions", flush=True)
    rows = await rescore_llm(rows, judge, ffilter=ffilter)
    dest = out_dir / a.out_name
    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    fin = [r for r in rows if r["is_finding"]]
    print(f"[rescore] -> {dest.relative_to(REPO)}", flush=True)
    print(f"[rescore] {len(fin):,} finding groups · mean best reward "
          f"{sum(r['best_reward'] for r in fin) / max(len(fin), 1):.4f} · "
          f"mean greedy reward {sum(r['greedy_reward'] for r in fin) / max(len(fin), 1):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
