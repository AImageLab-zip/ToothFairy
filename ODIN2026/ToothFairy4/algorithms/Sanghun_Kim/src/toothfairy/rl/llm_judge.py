"""Per-region entailment judge — a RadFact-shaped reward.

Why this exists
---------------
The rule-based scorer (`reward.region_reward`) turned out not to proxy RadFact:
per-patient Spearman +0.19~+0.27, change-direction agreement 60.7% (coin flip is
50%), rule extractor recall 0.582.  Two RFT rounds raised the rule score and left
RadFact flat or worse, so the objective itself was the problem.

This module scores a region the way RadFact scores a report: ask an LLM whether
each generated claim is *entailed* by the reference, and vice versa, then take the
harmonic mean.  The phrase-extraction stage RadFact needs is skipped — per-structure
decode already emits one bullet per claim.

Two deliberate departures from RadFact:

1. **Reference = that region's ground truth only**, not the whole report.  Cheaper
   and stricter.  It also means a sentence whose support lives in another region is
   rejected, so this is a *different* scorer than RadFact, not a reimplementation.
   How far it diverges was measured, not assumed.
2. **Judge = the local Qwen3-32B**, the same backbone RadFact scores with.  Measured
   at 13.4 calls/s on one H100 with vLLM prefix caching at 99.7%, i.e. ~1.5 h for a
   full RFT epoch (73,566 deduplicated calls) at zero cost and pinned for reproducibility.

Normal regions (empty ground truth) keep the existing convention, which is not
revisited here: silence scores 1.0, saying anything scores 0.0.  No LLM call is
made for them.
"""
from __future__ import annotations

import asyncio
import hashlib
import statistics
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
from pathlib import Path

from ..paths import REPO  # noqa: E402
_RADFACT_SRC = REPO / "tools" / "RadFact" / "src"
if str(_RADFACT_SRC) not in sys.path:
    sys.path.insert(0, str(_RADFACT_SRC))

from .presence import is_presence_line

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B-AWQ"
# Saturates here; 256 buys nothing because the workload is decode-bound (measured).
DEFAULT_CONCURRENCY = 128


def split_phrases(text: str) -> list[str]:
    """Split a region output into bullet phrases ('- a - b' or newline separated)."""
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?:^|\s)-\s+", text) if p.strip()]


@dataclass
class RegionScore:
    """Entailment score for one region, with the counts the F1 was built from."""

    precision: float
    recall: float
    f1: float
    n_candidate: int
    n_reference: int
    n_candidate_entailed: int
    n_reference_entailed: int
    llm_calls: int = 0
    verdicts: list[dict] = field(default_factory=list)


class EntailmentJudge:
    """Batched entailment verification against a local OpenAI-compatible endpoint.

    Identical (reference, hypothesis) pairs recur heavily across a rollout group —
    73% of precision-direction pairs and 77% of recall-direction pairs are repeats —
    so results are memoised in-process and, optionally, on disk.
    """

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 concurrency: int = DEFAULT_CONCURRENCY, max_tokens: int = 256,
                 cache_path: Path | None = None, beta: float = 1.0,
                 pure_radfact: bool = False) -> None:
        from openai import AsyncOpenAI
        from radfact.llm_utils.nli.processor import get_ev_processor_singlephrase
        from radfact.llm_utils.prompt_tasks import ReportType

        self.model = model
        # **Score with the RadFact entailment verdict alone**: for each region compare the
        # generated sentences with that region's split ground truth claim by claim, 1 if
        # entailed and 0 otherwise, with no further conventions. Off (default) keeps the
        # legacy conventions.
        self.pure_radfact = pure_radfact
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self._url = url
        self._client = AsyncOpenAI(base_url=url,
                                   api_key=os.environ.get("API_KEY_CHAT_0", "dummy"),
                                   timeout=600)
        self._processor = get_ev_processor_singlephrase(ReportType.CT, Path("/tmp"))
        self._cache: dict[str, bool] = {}
        self._cache_path = cache_path
        if cache_path is not None and cache_path.exists():
            self._cache = json.loads(cache_path.read_text())
        self.n_calls = 0
        self.n_errors = 0
        # Give up quickly when the judge is down as a whole. A re-scoring run once pointed at
        # port 8003 while the server listened on 8000: **all** 118,537 verdicts failed and it
        # still ran for two and a half hours. That many failures without a single success is a
        # transport problem, not a verdict.
        self._circuit_open = False
        # F_beta — beta>1 weights recall more heavily. Measured motivation: the policy already
        # writes 24.3 phrases, more than the reference, yet recall is 0.3396, and the 872 misses
        # pile up on crowns (95.8%), root canals (94.9%) and implants (100%). Saying more does
        # not lift it (+27% length bought +0.014 recall for -0.037 precision); *what* is said
        # has to change.
        self.beta = float(beta)

    # -- prompt -----------------------------------------------------------------
    @staticmethod
    def _key(reference: list[str], hypothesis: str) -> str:
        blob = json.dumps([reference, hypothesis], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _messages(self, reference: list[str], hypothesis: str) -> list[dict]:
        from radfact.llm_utils.nli.schema import ComparisonQuerySinglePhrase

        query = ComparisonQuerySinglePhrase(reference=reference, hypothesis=hypothesis)
        role_of = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}
        return [{"role": role_of[type(m).__name__], "content": m.content}
                for m in self._processor.query_template.format_messages(query=query)]

    # -- inference --------------------------------------------------------------
    async def _judge_one(self, reference: list[str], hypothesis: str, sem) -> bool:
        key = self._key(reference, hypothesis)
        if key in self._cache:
            return self._cache[key]
        if self._circuit_open:
            # The judge is already known to be down — firing the remaining hundred thousand
            # calls would only burn time.
            self.n_errors += 1
            return False
        async with sem:
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(reference, hypothesis),
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    # Qwen3 is a thinking model; RadFact disables thinking for this proxy
                    # (tools/RadFact/src/radfact/llm_utils/engine/arguments.py:68).
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception:  # noqa: BLE001
                # RadFact treats an unanswered phrase as not entailed; match that so a
                # transport hiccup can never inflate the reward.
                #
                # ⚠️ Return False but **do not cache it**. Caching freezes a transient outage
                # into a permanent wrong answer that poisons later runs — one wrong port once
                # cached 118,537 False verdicts, which put the reward of all 20,202 groups at
                # 0.0 (zero_var 100%) and still handed 12,818 "selected" samples to training.
                # Uncached, the next run reuses the successful verdicts and re-asks the rest.
                self.n_errors += 1
                if self.n_calls == 0 and self.n_errors >= 200:
                    self._circuit_open = True
                return False
        self.n_calls += 1
        body = (resp.choices[0].message.content or "").lower()
        # The schema emits exactly one `status:` line; `not_entailment` must not match
        # the positive class, so test for the negative first.
        entailed = "status: entailment" in body and "status: not_entailment" not in body
        self._cache[key] = entailed
        return entailed

    async def score_regions(self, regions: list[tuple[list[str], list[str]]],
                            progress_every: int = 0) -> list[RegionScore]:
        """Score many regions concurrently.

        `regions` is a list of (candidate phrases, reference phrases).  Every pair is
        dispatched at once so the endpoint stays saturated across regions, which is
        what gives the measured 13.4 calls/s.
        """
        sem = asyncio.Semaphore(self.concurrency)
        jobs: list[tuple[int, str, asyncio.Task]] = []
        for i, (candidate, reference) in enumerate(regions):
            if candidate and reference:
                for phrase in candidate:
                    jobs.append((i, "p", asyncio.ensure_future(
                        self._judge_one(reference, phrase, sem))))
                for phrase in reference:
                    jobs.append((i, "r", asyncio.ensure_future(
                        self._judge_one(candidate, phrase, sem))))

        if jobs:
            done = 0
            for _, _, task in jobs:
                await task
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"[judge] {done}/{len(jobs)} verdicts (calls {self.n_calls})", flush=True)

        entailed: dict[tuple[int, str], list[bool]] = {}
        for i, side, task in jobs:
            entailed.setdefault((i, side), []).append(task.result())

        out: list[RegionScore] = []
        for i, (candidate, reference) in enumerate(regions):
            out.append(self._assemble(i, candidate, reference, entailed))
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
        # Never let failures pass as results. A failure has the same value as "not entailed",
        # so a burst of them silently drags the score towards 0 while leaving a plausible file
        # behind (that is how 20,202 groups once ended at reward 0 and were selected and trained
        # on). Keep the successful verdicts in the cache, then fail loudly.
        if jobs and self.n_errors:
            rate = self.n_errors / len(jobs)
            if self._circuit_open or rate > 0.02:
                raise RuntimeError(
                    f"{self.n_errors:,}/{len(jobs):,} verdicts failed ({rate:.1%}) — check the "
                    f"judge endpoint ({self._url}). A failure counts the same as not-entailed, "
                    f"so leaving it in drags the reward silently to 0. Successful verdicts stay "
                    f"in the cache, so a rerun after fixing the server re-asks only the rest.")
            print(f"[judge] WARN {self.n_errors:,}/{len(jobs):,} verdicts failed ({rate:.1%}) — "
                  f"treated as not-entailed (not cached, re-asked on the next run)", flush=True)
        return out

    def _assemble(self, i, candidate, reference, entailed) -> RegionScore:
        n_c, n_r = len(candidate), len(reference)
        if not reference:
            if self.pure_radfact:
                # With zero reference claims there is **nothing to get right**: silence earns
                # no credit, and whatever was said is unsupported, so precision is 0.
                # ⚠️ The legacy convention (silence = 1.0) fired on **12,699 of 17,561 (72%)**
                # tooth groups, i.e. it taught "say nothing about teeth for full marks" that
                # often. Every RFT round lost recall while the silence rate rose from 7.4% to
                # 29-34%, which fits.
                return RegionScore(0.0, 0.0, 0.0, n_c, 0, 0, 0)
            # Legacy convention: silence is right, saying anything is wrong.
            score = 0.0 if candidate else 1.0
            return RegionScore(score, score, score, n_c, 0, 0, 0)
        if not candidate:
            # Nothing said about a region that has findings: no precision to measure,
            # zero recall. RadFact's harmonic mean is 0 either way.
            return RegionScore(0.0, 0.0, 0.0, 0, n_r, 0, 0)
        c_hits = sum(entailed.get((i, "p"), []))
        r_hits = sum(entailed.get((i, "r"), []))
        precision = c_hits / n_c
        recall = r_hits / n_r
        b2 = self.beta ** 2
        denom = b2 * precision + recall
        score = 0.0 if denom == 0 else (1 + b2) * precision * recall / denom
        return RegionScore(precision, recall, score, n_c, n_r, c_hits, r_hits,
                           llm_calls=n_c + n_r)




def _judge_phrases(text: str | None, neutralise_presence: bool = True) -> list[str]:
    """Region text → claim phrases, with tooth-presence statements neutralised.

    Presence neutralisation is a fixed scoring rule: readers record
    "tooth N present" only 45.5% of the time on teeth the ground truth marks normal,
    so the same input carried opposite answers.  `reward.region_reward` applies it via
    `strip_presence`, which works line by line; a generation often puts several bullets
    on one line ("- a - b"), so the same rule is applied here **per phrase** instead.
    Same rule, applied where this scorer can see it.
    """
    if not neutralise_presence:
        # RadFact does not strip tooth-presence statements — if the reference carries one and
        # the generation stays silent, RadFact counts that as a recall loss.
        return split_phrases(text)
    return [p for p in split_phrases(text) if not is_presence_line("- " + p)]


def _phrase_key(phrases: list[str]) -> str:
    """Cache key used by `src/toothfairy/pipeline/rft_filter_cache.py` — the phrase list itself."""
    return hashlib.sha1(json.dumps(list(phrases), ensure_ascii=False).encode()).hexdigest()


class FindingFilter:
    """The organizers' `finding_filter` verdict, replayed from a cache.

    The official scorer drops normal-status confirmations, tooth-presence statements and
    field-of-view sentences before judging anything — 78% of our generated phrases
    (48.2 → 10.8) and a third of the reference. Scoring the reward on unfiltered text
    therefore pays the policy for sentences the metric never sees. This class puts the
    reward on the same footing.

    Verdicts come from a cache built once with the LLM filter itself (63,662 distinct
    phrase lists across both dumps, ~7 minutes on the local judges), not from a rule:
    a rule would need its own validation and would still be approximating a call we can
    simply make. A cache miss is an error rather than a pass-through — silently letting
    unfiltered text through would reintroduce exactly the mismatch this exists to remove.
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache: dict[str, list[str]] = json.loads(Path(cache_path).read_text())
        self.n_hit = 0
        self.n_miss = 0

    def __call__(self, phrases: list[str]) -> list[str]:
        if not phrases:
            return []
        got = self.cache.get(_phrase_key(phrases))
        if got is None:
            self.n_miss += 1
            return phrases
        self.n_hit += 1
        return list(got)

    def check(self) -> None:
        if self.n_miss:
            raise RuntimeError(
                f"{self.n_miss:,} filter-cache misses ({self.n_hit:,} hits) — the cache does "
                f"not cover this dump. Rebuild it with `toothfairy.cli.rft filter-cache` on the same dump. "
                f"Letting a miss through scores that sample unfiltered and mixes the reward.")


async def rescore_llm(rows: list[dict], judge: EntailmentJudge,
                      ffilter: "FindingFilter | None" = None) -> list[dict]:
    """Re-score every group with the per-region entailment judge, no regeneration.

    Replaces the rule reward (claim 0.7 + term 0.2 + polarity 0.1 with an FDI gate)
    by the entailment F1 of that region: does each generated claim follow from the
    region's ground truth, and does each ground-truth claim follow from the generation.
    The rule scorer was retired because it did not proxy RadFact — per-patient Spearman
    +0.19~+0.27, change-direction agreement 60.7%.

    ⚠️ The FDI gate is **not** carried over.
    A claim naming the wrong tooth simply fails to be entailed by that region's ground
    truth, so the judge already rejects it; keeping a multiplicative gate on top would
    penalise it twice.

    Every sample of every group is dispatched in one pass so the endpoint stays
    saturated — that is what gives the measured 13.4 calls/s.

    Lives here rather than in the driver module so it can run where RadFact is installed,
    which has the judge's dependencies but no torch.
    """
    index: list[tuple[int, object]] = []
    regions: list[tuple[list[str], list[str]]] = []
    kept_by_row: dict[int, dict[object, list[str]]] = {}
    for i, r in enumerate(rows):
        keep_presence = getattr(judge, "pure_radfact", False)
        prep = (lambda t: ffilter(_judge_phrases(t, neutralise_presence=False))) if ffilter \
            else (lambda t: _judge_phrases(t, neutralise_presence=not keep_presence))
        ref = prep(r["gt"])
        index.append((i, "g"))
        greedy = prep(r.get("greedy_text"))
        regions.append((greedy, ref))
        kept_by_row.setdefault(i, {})["gt"] = ref
        kept_by_row[i]["g"] = greedy
        for k, txt in enumerate(r["texts"]):
            index.append((i, k))
            cand = prep(txt)
            regions.append((cand, ref))
            kept_by_row[i][k] = cand

    n_calls = sum(len(c) + len(ref) for c, ref in regions if c and ref)
    print(f"[rft] LLM judge re-scoring — {len(regions):,} region-samples, up to {n_calls:,} "
          f"verdict calls (before deduplication)", flush=True)
    scores = await judge.score_regions(regions, progress_every=5000)

    by_row: dict[int, dict[object, float]] = {}
    for (i, slot), s in zip(index, scores):
        # `RegionScore.f1` — **not** `region_reward_llm`, which pools phrase counts and so
        # drops the normal-region convention (silence 1.0 / speech 0.0): a silent region
        # has no phrases to pool and would come out 0.0, teaching the model that staying
        # quiet where the ground truth is empty is worthless. The pooling aggregator is
        # for patient-level numbers comparable to RadFact; a group is a single region.
        by_row.setdefault(i, {})[slot] = s.f1

    out = []
    n_demoted = 0
    for i, r in enumerate(rows):
        got = by_row[i]
        sc = [float(got[k]) for k in range(len(r["texts"]))]
        b = int(np.argmax(sc))
        med = float(statistics.median(sc))
        row = {**r, "greedy_reward": float(got["g"]), "rewards": sc,
               "best_i": b, "best_reward": sc[b], "median_reward": med,
               "beats_median": sc[b] > med, "zero_var": max(sc) == min(sc),
               "best_text": r["texts"][b]}
        if ffilter is not None:
            kept = kept_by_row[i]
            # ⚠️ When the filter empties the reference completely, that region **stops being a
            # finding region**: it moves over to the maintenance samples, changing both sides of
            # the finding:maintenance ratio. A keep ratio r after filtering therefore does not
            # describe the same sample mix as the same r before filtering.
            row["is_finding_raw"] = bool(r["is_finding"])
            row["is_finding"] = bool(kept["gt"])
            row["gt_kept"] = kept["gt"]
            # `rft_rollout._maint_keep` picks the maintenance target from this: if no phrase
            # survives the filter the sentence is never scored, so it is **left as is**; if one
            # survives it is an unsupported finding on a normal region and is corrected to
            # silence.
            row["greedy_kept"] = kept["g"]
            n_demoted += int(row["is_finding_raw"] and not row["is_finding"])
        out.append(row)
    if ffilter is not None:
        ffilter.check()
        fin_now = sum(1 for r in out if r["is_finding"])
        print(f"[rft] filter applied — findings {sum(1 for r in rows if r['is_finding']):,} -> "
              f"{fin_now:,} ({n_demoted:,} regions moved to maintenance, reference emptied), "
              f"filter-cache hits {ffilter.n_hit:,}", flush=True)
    print(f"[rft] re-scoring done — {judge.n_calls:,} calls, {judge.n_errors} errors", flush=True)
    return out
