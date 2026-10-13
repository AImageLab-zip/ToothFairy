#!/usr/bin/env python
"""Cache the organizers' `finding_filter` verdict for every phrase list in an RFT dump.

Why.  The organizers' scorer runs `finding_filter.remove_normal_findings` before it
judges anything: normal-status confirmations, tooth-presence statements and
field-of-view sentences are dropped from BOTH the candidate and the reference.  Measured
on our own generations that filter deletes 78% of the phrases (48.2 -> 10.8).  Our RFT
reward, meanwhile, scores exactly those deleted phrases and pays for them — so the reward
and the metric disagree about what the task is.  This script removes that disagreement by
letting the reward see the same filtered text the official metric sees.

Why the LLM filter rather than a rule approximation.  A regex approximation for the
candidate side looks necessary if filtering every rollout sample costs 6 x 20,202 calls.
It does not: identical phrase lists recur heavily across samples, patients and arms, so
after de-duplication the dumps together need **63,662** calls (4,505 reference + 59,157
candidate), about 35 minutes across three local judges and free.  A rule would have needed
its own validation and would still have approximated the thing we can simply call.

The cache is keyed by the phrase list, so a later arm that generates the same text pays
nothing.  Shards write separate files; `--merge` folds them into one.

Usage (one shard per judge port):
    python -m toothfairy.pipeline.rft_filter_cache \
        --dumps experiments/analysis/<rollout-dump-dir>/rft_groups_pure.jsonl \
        --out experiments/analysis/rft-filter-cache --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-32B-AWQ --shard 0 --num-shards 3

    ... --merge          # after all shards finish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from toothfairy.paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "tools" / "radfact_lite" / "src"))

from radfact_lite import ReportType, remove_normal_findings  # noqa: E402
from radfact_lite.clients import _ChatJSONClient  # noqa: E402

# Same split as `toothfairy.rl.llm_judge.split_phrases`; duplicated because that module
# imports torch-free but may live in a separate environment.
_SPLIT = re.compile(r"(?:^|\s)-\s+")


def split_phrases(text: str | None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SPLIT.split(text) if p.strip()]


def key_of(phrases: list[str] | tuple[str, ...]) -> str:
    blob = json.dumps(list(phrases), ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


def collect(dumps: list[Path], sides: str) -> dict[str, list[str]]:
    """Every distinct phrase list appearing in the dumps, keyed by its hash."""
    out: dict[str, list[str]] = {}
    for path in dumps:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                texts: list[str | None] = []
                if sides in ("reference", "both"):
                    texts.append(row.get("gt"))
                if sides in ("candidate", "both"):
                    texts.extend(row.get("texts") or [])
                    texts.append(row.get("greedy_text"))
                for t in texts:
                    ph = split_phrases(t)
                    if ph:
                        out.setdefault(key_of(ph), ph)
    return out


def load_cache(out_dir: Path) -> dict[str, list[str]]:
    """Merged cache plus any shard files already on disk."""
    cache: dict[str, list[str]] = {}
    for path in sorted(out_dir.glob("filter_cache*.json")):
        try:
            cache.update(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"[filter] WARN {path.name} is corrupt, skipping it", flush=True)
    return cache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=[], help="rft_groups*.jsonl")
    ap.add_argument("--out", default="experiments/analysis/rft-filter-cache")
    ap.add_argument("--sides", default="both", choices=["reference", "candidate", "both"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="probe: only this many new lists")
    ap.add_argument("--merge", action="store_true", help="fold shard files into filter_cache.json")
    a = ap.parse_args(argv)

    out_dir = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.merge:
        merged = load_cache(out_dir)
        (out_dir / "filter_cache.json").write_text(json.dumps(merged, ensure_ascii=False))
        for path in sorted(out_dir.glob("filter_cache.shard*.json")):
            path.rename(path.with_suffix(".json.merged"))
        print(f"[filter] merged {len(merged):,} entries → {out_dir/'filter_cache.json'}")
        return 0

    dumps = [Path(d) if Path(d).is_absolute() else REPO / d for d in a.dumps]
    missing = [str(d) for d in dumps if not d.exists()]
    if missing:
        raise SystemExit(f"dumps not found: {missing}")

    want = collect(dumps, a.sides)
    have = load_cache(out_dir)
    todo_keys = sorted(k for k in want if k not in have)
    todo_keys = [k for i, k in enumerate(todo_keys) if i % a.num_shards == a.shard]
    if a.limit:
        todo_keys = todo_keys[: a.limit]
    print(f"[filter] unique phrase lists {len(want):,} · already cached {len(have):,} · "
          f"this shard {len(todo_keys):,} (shard {a.shard}/{a.num_shards})", flush=True)
    if not todo_keys:
        return 0

    client = _ChatJSONClient(a.model, base_url=a.base_url, api_key="dummy", timeout=180,
                             max_retries=5)
    lock = threading.Lock()
    done: dict[str, list[str]] = {}
    n_err = 0
    t0 = time.time()

    def work(k: str) -> None:
        nonlocal n_err
        try:
            kept = remove_normal_findings(client, want[k], ReportType.TOOTHFAIRY)
        except Exception as exc:  # noqa: BLE001
            with lock:
                n_err += 1
                if n_err <= 3:
                    print(f"[filter] failed: {type(exc).__name__} {exc}"[:200], flush=True)
            return
        with lock:
            done[k] = kept
            n = len(done)
            if n % 500 == 0:
                rate = n / max(time.time() - t0, 1e-9) * 60
                left = (len(todo_keys) - n) / max(rate, 1e-9)
                print(f"[filter] {n:,}/{len(todo_keys):,} · {rate:.0f}/min · {left:.0f} min left",
                      flush=True)
                shard_path.write_text(json.dumps(done, ensure_ascii=False))

    shard_path = out_dir / f"filter_cache.shard{a.shard}.json"
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo_keys))
    shard_path.write_text(json.dumps(done, ensure_ascii=False))

    rate = n_err / max(len(todo_keys), 1)
    kept_n = sum(len(v) for v in done.values())
    src_n = sum(len(want[k]) for k in done)
    print(f"[filter] done {len(done):,} · failed {n_err:,} ({rate:.1%}) · "
          f"phrases {src_n:,} → {kept_n:,} ({kept_n/max(src_n,1):.1%} kept) · "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    if rate > 0.02:
        # A silent failure here would look identical to "the filter deleted everything",
        # and that would teach the reward to say nothing at all.
        raise SystemExit(f"failure rate {rate:.1%} — the cache cannot be trusted. Check the judge "
                         f"and re-run (what succeeded is kept in {shard_path.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
