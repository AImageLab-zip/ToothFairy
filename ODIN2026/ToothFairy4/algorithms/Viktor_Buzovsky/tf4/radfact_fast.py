"""RadFact, made fast enough and quiet enough to actually run experiments on.

`run_radfact.py` works but cannot resolve the differences we need to measure. Its
per-case F1 has sd ~0.15, so at n=30 the standard error on the mean is ~0.03; the
paired human-vs-pipeline gap measured +0.075 with a 95% CI of [-0.018, +0.165].
If a 30-case run cannot separate this pipeline from a human expert, it certainly
cannot resolve a -0.007 ablation. Every earlier conclusion that rested
on a delta under ~0.08 was noise.

Four changes, and they compound:

  * every LLM call is cached on disk by content hash, so the reference side is
    paid for once and ablations only pay for what actually changed;
  * calls run concurrently (LM Studio serves several at a time), which is what
    makes n >= 150 affordable at all;
  * every per-phrase decision is written out, so ONE run yields the whole
    per-claim-class attribution table instead of needing an ablation per class;
  * a run that had LLM failures refuses to report a score, because a failed call
    is scored as "not entailed" and quietly looks like a bad result.

Usage
-----
    # baseline, and the attribution table that comes with it
    python tf4/radfact_fast.py --mode pipeline --n 150 --tag base

    # what two human experts score against each other -- the ceiling
    python tf4/radfact_fast.py --mode human --n 30 --tag human

    # paired comparison of two runs over their common cases
    python tf4/radfact_fast.py --compare work/rf_base.json work/rf_v2.json

LM Studio gotcha: the context window is DIVIDED across parallel slots. Loading
with `--parallel 12 -c 16384` gives each slot 1365 tokens, and 94% of calls die
with a bare 400 while the run still prints a plausible-looking aggregate. Use

    lms load qwen3.6-35b-a3b -c 32768 --parallel 6
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "work", "rfcache")
DATA = os.path.join(ROOT, "toothfairy4_v03")


# --------------------------------------------------------------------------
# client patches (same two as run_radfact.py -- see its docstring)
# --------------------------------------------------------------------------
def _build_client(model: str, base_url: str, timeout: float):
    import radfact_lite.clients as rc
    from radfact_lite import ModelConfig
    from radfact_lite.clients import build_json_client

    rc._LOWEST_SETTINGS = {**rc._LOWEST_SETTINGS, "max_tokens": 2000}

    original = rc._ChatJSONClient.complete_json

    def complete_json(self, messages, response_format):
        try:
            return original(self, messages, response_format)
        except ValueError:
            # Reasoning models served by LM Studio put the whole answer in
            # `reasoning_content` and leave `content` empty.
            msg = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                response_format=response_format,
                max_tokens=2000,
                temperature=0,
            ).choices[0].message
            body = (msg.content or "") or (getattr(msg, "reasoning_content", "") or "")
            return json.loads(body)

    rc._ChatJSONClient.complete_json = complete_json
    return build_json_client(
        ModelConfig(model=model, provider="ollama", base_url=base_url,
                    timeout=timeout, max_retries=1)
    )


# --------------------------------------------------------------------------
# disk cache: append-only JSONL, loaded once, so a rerun costs nothing
# --------------------------------------------------------------------------
class Cache:
    def __init__(self, name: str) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f"{name}.jsonl")
        self.lock = threading.Lock()
        self.mem: dict[str, object] = {}
        if os.path.exists(self.path):
            with open(self.path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a run killed mid-write leaves one torn line
                    self.mem[rec["k"]] = rec["v"]

    def get(self, key: str):
        return self.mem.get(key)

    def put(self, key: str, value) -> None:
        with self.lock:
            self.mem[key] = value
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


def _key(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:24]


# --------------------------------------------------------------------------
# corpus access
# --------------------------------------------------------------------------
def reports_for(case: str) -> list[str]:
    """Report paths for a case, ordered by annotator id.

    The evaluator ships exactly one .txt per case, and the lowest annotator id is
    the rule we infer; `reports_for(c)[0]` is therefore the reference, and
    `[1]` is a second expert's opinion on the same scan.
    """
    return sorted(
        glob.glob(os.path.join(DATA, case, "reports_en", "*.txt")),
        key=lambda p: int(re.search(r"cbct_(\d+)_", p).group(1)),
    )


def select_cases(n: int, paired: bool, seed: int = 0, sets: str = "PFS") -> list[str]:
    """Cases to score, restricted to `sets` (e.g. "F", "PFS", "AFPS").

    The default excludes A because it had no derived findings when this was
    written. That default also means every headline number so far is 78% set P,
    which matters: absence is entailed 0.53 on P and 0.06 on F, and the lesion
    prior runs the other way (0.29 on P, 0.94 on F). Score on one set, report on
    that set.
    """
    cases = [c for c in sorted(os.listdir(DATA)) if c[0] in set(sets.upper())]
    cases = [c for c in cases if reports_for(c)]
    if paired:
        # the human-ceiling run needs two annotators; using the same pool for the
        # pipeline run makes the two directly comparable case-for-case
        cases = [c for c in cases if len(reports_for(c)) >= 2]
    # `Random(seed).sample` rather than a shuffle, so the case set matches the one
    # run_radfact.py picks for the same seed and n -- otherwise every cached
    # decision from an earlier run misses.
    return sorted(random.Random(seed).sample(cases, min(n, len(cases))))


def render_cached(case: str, rec: dict, drop: set[str]) -> str:
    from tf4.render import render
    from tf4.schema import Findings

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
        endodontic=set(rec.get("endodontic") or []),
        restorations=set(rec.get("restorations") or []),
    )
    for slot in drop:
        if slot == "canal_contact":
            f.canal_contact = {}
        elif slot == "impacted":
            f.impacted = set()
        elif slot == "absent_teeth":
            f.teeth_present = {k: v for k, v in f.teeth_present.items() if v}
        elif slot == "crowns":
            f.crowns = set()
        elif slot == "restorations":
            f.has_implant = f.has_bridge = f.has_crown = None
        else:
            raise SystemExit(f"unknown --drop slot: {slot}")
    return render(f)


# --------------------------------------------------------------------------
# phrase taxonomy -- shared by the attribution tables
# --------------------------------------------------------------------------
CANDIDATE_CLASSES = [
    ("endodontic",          r"endodontic"),
    ("restoration, per site", r"conservative restoration"),
    ("lesion prior",        r"osteolytic|osteocondens"),
    ("canal-molar contact", r"canal.{0,60}(close relationship|in relation)"),
    ("canal course",        r"canal"),
    ("tooth absence",       r"absen|edentul"),
    ("impacted tooth",      r"impacted"),
    ("implant",             r"implant"),
    ("bridge",              r"bridge"),
    ("crown, per site",     r"crown on tooth|crown.{0,20}tooth \d"),
    ("crown, site-less",    r"prosthetic crowns"),
    ("crowns, maxillary",   r"crowns of the maxillary"),
    ("condyles",            r"condyl|coronoid"),
    ("coverage / FOV",      r"maxill|acquisition|scan volume|assessab|mandib"),
]

REFERENCE_CLASSES = [
    ("endodontic",   r"endodont|root canal|post[- ]and[- ]core|intracanal|gutta|beyond the apex"),
    ("crown/prosth", r"crown|prosthe|abutment|bridge|pontic"),
    ("filling",      r"filling|restorat|black class|composite|conservative|amalgam"),
    ("implant",      r"implant"),
    ("absence",      r"absen|missing|edentul|avuls|agenes"),
    ("impaction",    r"impact|unerupt|semi-includ|disodont"),
    ("lesion",       r"radiolucen|granulom|cyst|osteolytic|osteocondens|rarefact|lesion|periapical"),
    ("perio/bone",   r"periodont|furcation|bone loss|atroph|trophism|pocket|Cawood|dimension"),
    ("caries",       r"cari"),
    ("canal/IAN",    r"canal|IAN|alveolar nerve|foramen|emergence"),
    ("condyle/TMJ",  r"condyl|coronoid|glenoid"),
    ("sinus",        r"sinus"),
    ("coverage/FOV", r"maxill|acquisition|scan volume|included|assessab|mandib"),
]


def classify(phrase: str, table) -> str:
    for name, rx in table:
        if re.search(rx, phrase, re.I):
            return name
    return "other"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def f1_of(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def run(args) -> int:
    from radfact_lite.entailment import assess_entailment
    from radfact_lite.phrase_parser import parse_report_to_phrases
    from radfact_lite.rf_types import ReportType

    client = _build_client(args.model, args.base_url, args.timeout)
    parse_cache, entail_cache = Cache("parse"), Cache("entail")

    def parse(text: str) -> list[str]:
        k = _key("parse", text)
        hit = parse_cache.get(k)
        if hit is not None:
            return hit
        phrases = parse_report_to_phrases(client, text, ReportType.TOOTHFAIRY)
        parse_cache.put(k, phrases)
        return phrases

    def entail(reference: list[str], hypothesis: str) -> str:
        k = _key("entail", hypothesis, "\n".join(reference))
        hit = entail_cache.get(k)
        if hit is not None:
            return hit
        status = assess_entailment(client, reference, hypothesis, ReportType.TOOTHFAIRY).status
        entail_cache.put(k, status)
        return status

    paired = args.mode == "human" or args.paired
    cases = select_cases(args.n, paired, args.seed, args.sets)

    references, candidates = {}, {}
    if args.mode == "human":
        for c in cases:
            paths = reports_for(c)
            references[c] = open(paths[0]).read().strip()
            candidates[c] = open(paths[1]).read().strip()
    else:
        from tf4.render import render
        from tf4.schema import Findings

        cached = json.load(open(os.path.join(ROOT, "work", "pred_findings.json")))
        drop = {s.strip() for s in args.drop.split(",") if s.strip()}
        prior = render(Findings())
        for c in list(cases):
            if c not in cached:
                cases.remove(c)
                continue
            references[c] = open(reports_for(c)[0]).read().strip()
            candidates[c] = prior if args.mode == "prior" else render_cached(c, cached[c], drop)

    print(f"mode={args.mode} cases={len(cases)} workers={args.workers} model={args.model}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        parsed_ref = dict(zip(cases, pool.map(lambda c: parse(references[c]), cases)))
        parsed_cand = dict(zip(cases, pool.map(lambda c: parse(candidates[c]), cases)))
    print("parsed; running entailment", flush=True)

    jobs = []
    for c in cases:
        jobs += [(c, "prec", p, parsed_ref[c]) for p in parsed_cand[c]]
        jobs += [(c, "rec", p, parsed_cand[c]) for p in parsed_ref[c]]

    done = [0]
    lock = threading.Lock()

    def one(job):
        case, direction, phrase, against = job
        try:
            status = entail(against, phrase)
        except Exception as exc:  # noqa: BLE001 -- recorded, then fails the run
            status = f"error: {type(exc).__name__}"
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)
        return {"case": case, "dir": direction, "phrase": phrase, "status": status}

    with ThreadPoolExecutor(args.workers) as pool:
        decisions = list(pool.map(one, jobs))

    histogram = Counter(d["status"].split(":")[0] for d in decisions)
    print(f"\ndecision statuses: {dict(histogram)}")
    failures = sum(v for k, v in histogram.items() if k == "error")
    if failures:
        # A failed call scores as "not entailed", so a broken run looks like a bad
        # result rather than a broken one. That cost a whole measurement once.
        print(f"\n!! {failures}/{len(jobs)} LLM calls FAILED -- score suppressed.")
        print("   Most likely the context window: LM Studio divides it across parallel")
        print("   slots, so `--parallel N -c C` gives each call only C/N tokens.")
        print("   Try:  lms load <model> -c 32768 --parallel 6")
        return 1

    per_case = defaultdict(lambda: {"ec": 0, "c": 0, "er": 0, "r": 0})
    for d in decisions:
        slot = per_case[d["case"]]
        if d["dir"] == "prec":
            slot["c"] += 1
            slot["ec"] += d["status"] == "entailment"
        else:
            slot["r"] += 1
            slot["er"] += d["status"] == "entailment"

    precisions = [per_case[c]["ec"] / per_case[c]["c"] if per_case[c]["c"] else 0.0 for c in cases]
    recalls = [per_case[c]["er"] / per_case[c]["r"] if per_case[c]["r"] else 0.0 for c in cases]
    P, R = statistics.mean(precisions), statistics.mean(recalls)

    print(f"\n=== {args.tag or args.mode}:  P={P:.4f}  R={R:.4f}  F1={f1_of(P, R):.4f}   (n={len(cases)})")
    print(f"    candidate {statistics.mean(per_case[c]['c'] for c in cases):.1f} phrases/case"
          f"   reference {statistics.mean(per_case[c]['r'] for c in cases):.1f} phrases/case")
    if len(cases) < 100:
        f1s = [f1_of(p, r) for p, r in zip(precisions, recalls)]
        se = statistics.stdev(f1s) / len(f1s) ** 0.5
        print(f"    per-case F1 sd {statistics.stdev(f1s):.3f} -> SE {se:.3f};"
              f" deltas under {2 * se:.2f} are not resolvable at this n")

    attribution(decisions, len(cases))

    out = args.out or os.path.join(ROOT, "work", f"rf_{args.tag or args.mode}.json")
    json.dump({"mode": args.mode, "tag": args.tag, "model": args.model, "seed": args.seed,
               "P": P, "R": R, "F1": f1_of(P, R), "n": len(cases),
               "per_case": {c: per_case[c] for c in cases}, "decisions": decisions},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return 0


def attribution(decisions: list[dict], n_cases: int) -> None:
    """The point of the whole exercise: one run, the entire per-class table."""
    cand = defaultdict(lambda: [0, 0])
    for d in decisions:
        if d["dir"] != "prec":
            continue
        row = cand[classify(d["phrase"], CANDIDATE_CLASSES)]
        row[0] += 1
        row[1] += d["status"] == "entailment"

    print(f"\n--- what each phrase we emit is worth "
          f"(break-even entailment ~0.24; below that a claim costs F1) ---")
    print(f"{'class':22s}{'/case':>7s}{'entailed':>10s}{'gain/case':>11s}")
    for name, (count, ent) in sorted(cand.items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0])):
        print(f"{name:22s}{count / n_cases:7.2f}{ent / count:10.2f}{ent / n_cases:11.2f}")
    total_c = sum(v[0] for v in cand.values())
    total_e = sum(v[1] for v in cand.values())
    print(f"{'TOTAL':22s}{total_c / n_cases:7.2f}{total_e / total_c:10.2f}{total_e / n_cases:11.2f}")

    ref = defaultdict(lambda: [0, 0])
    for d in decisions:
        if d["dir"] != "rec":
            continue
        row = ref[classify(d["phrase"], REFERENCE_CLASSES)]
        row[0] += 1
        row[1] += d["status"] == "entailment"

    print(f"\n--- what the reference says that we miss (ranked by recoverable mass) ---")
    print(f"{'category':22s}{'/case':>7s}{'recalled':>10s}{'lost/case':>11s}")
    for name, (count, ent) in sorted(ref.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        print(f"{name:22s}{count / n_cases:7.2f}{ent / count:10.2f}{(count - ent) / n_cases:11.2f}")


def compare(path_a: str, path_b: str, iterations: int = 10000, seed: int = 0) -> int:
    """Paired bootstrap over the cases both runs share.

    Unpaired aggregate comparison is what made every earlier ablation unreadable:
    per-case F1 sd is ~0.15, so two 30-case aggregates differ by ~0.05 for free.
    """
    a, b = json.load(open(path_a)), json.load(open(path_b))
    common = sorted(set(a["per_case"]) & set(b["per_case"]))
    if not common:
        raise SystemExit("the two runs share no cases")

    def case_f1(run_, case):
        v = run_["per_case"][case]
        return f1_of(v["ec"] / v["c"] if v["c"] else 0.0, v["er"] / v["r"] if v["r"] else 0.0)

    deltas = [case_f1(b, c) - case_f1(a, c) for c in common]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(iterations))
    lo, hi = means[int(0.025 * iterations)], means[int(0.975 * iterations)]

    print(f"paired over {len(common)} common cases")
    print(f"  A {os.path.basename(path_a):28s} F1 {a['F1']:.4f}")
    print(f"  B {os.path.basename(path_b):28s} F1 {b['F1']:.4f}")
    print(f"\n  B - A per-case F1: {statistics.mean(deltas):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant -- indistinguishable at this n"
    print(f"  {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "human", "prior"],
                    help="pipeline: the shipped renderer. human: a second annotator's report, "
                         "i.e. the ceiling. prior: the fallback report with no findings.")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--paired", action="store_true",
                    help="restrict to cases with >=2 annotators so the run is "
                         "case-for-case comparable with --mode human")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="PFS",
                    help='which sets to score, e.g. "F", "PFS", "AFPS". Every number '
                         'before this flag existed was PFS, i.e. 78%% set P.')
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--drop", default="", help="slots to suppress for an ablation, comma separated")
    ap.add_argument("--tag", default="", help="names the output file work/rf_<tag>.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=os.getenv("RADFACT_MODEL", "qwen3.6-35b-a3b"))
    ap.add_argument("--base-url", default=os.getenv("RADFACT_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="paired bootstrap of two result files instead of running")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare, seed=args.seed)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
