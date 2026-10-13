#!/usr/bin/env python3
"""
code/eval/eval.py

Score a run: the official ranking, and the fact-level survey underneath it.

This is the evaluation ENTRY POINT: `scripts/run_eval.sh` is the shell runner
that submits it. It drives two modules and merges neither of them --
`official_ranking.py` and `structured_findings_evaluation.py` are both imported
elsewhere (per_dataset_breakdown, interrater_merge, compare_sources,
rate_matched_null and two studies), so they stay where they are and this file
is the thing that runs them together and finds what they need.

TWO NUMBERS, AND WHY YOU WANT BOTH
──────────────────────────────────
  rank    official_ranking.py -- RadFact Logical F1 + BLEU-4 + METEOR, the
          leaderboard metric. Needs the RadFact judge, so it needs a GPU
          somewhere -- though not this process's.
  survey  structured_findings_evaluation.py -- per finding, what the raw VQA
          read (PRED) against what survived postprocess (SUMMARY) against the
          generated ground truth. CPU only, seconds.

The ranking says whether an arm is better. The survey says which finding moved
and in which direction, which is the only thing you can act on. `--stage all`
(default) runs both against one run dir; either stage runs alone.

THE JUDGE IS NOT STARTED HERE, AND THAT IS DELIBERATE
─────────────────────────────────────────────────────
`scripts/judge_server.sh` holds a persistent vLLM RadFact judge on a GPU node
and advertises itself in `.judge/{url,ready,jobid}`. This process runs on the
LOGIN node and finds it. Folding the two together would mean paying a 15-25
minute model load on every evaluation, which is exactly what `evaluation.sh`
did and exactly why it was abandoned. Discovery order: --judge-url, then
`.judge/url`, then asking SLURM for the node running the `judge` job.

TWO PREFLIGHT CHECKS THAT LOOK PEDANTIC AND ARE NOT
───────────────────────────────────────────────────
Both exist because their failure mode is a COMPLETE-LOOKING result that is
wrong, discovered hours later:

  * A judge that is up but unreachable yields logical_f1=None for every case.
    The results file is full and entirely garbage. So /health is probed before
    a single case is scored.
  * nltk installs no corpora, and a missing wordnet is not an error -- METEOR
    silently changes implementation, and scale, on 10% of the Final Score.
    official_ranking.py warns either way and records `meteor_backend`, but it
    says so at the END of a run that takes hours. This says it at the start,
    while the fix is still cheap.

--survey-splits: ONE PREDICTION SET, THREE SURVEYS
──────────────────────────────────────────────────
A survey over all 582 training cases is dominated by cases the arm memorised,
so its headline number measures fit, not skill. The same predictions hold two
clean slices, and splitting them costs no GPU and no re-inference:

  trained   in the arm's training rows -- memorisation
  evalonly  the --eval-cases split: seen as eval loss, never trained on
  unseen    in the payload, in no training row at all

`unseen` is the honest number. `trained` minus `unseen` is a direct read on how
much of an arm's gain is memorisation -- the question the aggregate cannot
answer. `evalonly` is the control between them: if it tracks `unseen`, eval
loss was an honest generalisation signal; if it tracks `trained`, watching it
during training leaked more than intended.

Case lists come from the arm's own `cases_{trained,evalonly,unseen}.txt`, never
retyped, so a survey cannot be labelled with a split it does not describe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(
    p for p in _Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths, module_path  # noqa: E402

add_code_paths()

DEFAULT_SERVED_NAME = "qwen3-14b-text"
SPLITS = ("trained", "evalonly", "unseen")


# ── finding the judge ───────────────────────────────────────────────────────
def discover_judge(explicit: Optional[str]) -> Optional[str]:
    """--judge-url, then .judge/url, then SLURM. First hit wins."""
    if explicit:
        return explicit
    url_file = REPO_ROOT / ".judge" / "url"
    if url_file.is_file():
        url = url_file.read_text(encoding="utf-8").strip()
        if url:
            return url
    try:
        node = subprocess.run(
            ["squeue", "-u", __import__("getpass").getuser(),
             "-n", "judge", "-h", "-t", "RUNNING", "-o", "%N"],
            capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"http://{node[0]}:8001/v1" if node else None


def judge_reachable(url: str, timeout: float = 10.0) -> bool:
    """Probe /health. A judge that is up but unreachable produces a full
    results file with logical_f1=None on every case -- complete-looking and
    entirely garbage. Cheaper to find out now."""
    health = url[:-3] + "health" if url.endswith("/v1") else url + "/health"
    try:
        urllib.request.urlopen(health, timeout=timeout).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def check_nltk() -> None:
    """nltk itself is required; its wordnet corpus only silently matters."""
    try:
        import nltk  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[FAIL] nltk is not importable -- official_ranking.py needs it "
            "plus radfact_lite, which live in cbct_base. Activate that env, "
            "or run with --stage survey, which needs neither.")
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    except Exception:  # noqa: BLE001 -- nltk raises several unrelated types
        print("[WARN] wordnet corpus missing -- METEOR falls back to an "
              "implementation that does NOT score the same, on 10% of the "
              "Final Score. Fix before a run that takes hours:\n"
              "[WARN]   python3 -m nltk.downloader wordnet omw-1.4",
              file=sys.stderr)


# ── the two stages ──────────────────────────────────────────────────────────
def run_rank(args, reports_dir: Path, out_dir: Path) -> int:
    """official_ranking.py: RadFact + BLEU-4 + METEOR."""
    cmd = [sys.executable, str(module_path("official_ranking.py")),
           "--schema", str(args.schema),
           "--synthesized-dir", str(reports_dir),
           "--reports-dir", str(args.reports_dir),
           "--out-dir", str(out_dir),
           "--batch-size", str(args.batch_size)]
    if args.resume:
        cmd.append("--resume")
    if args.multi_reference:
        cmd.append("--multi-reference")
    # The normal-finding filter is ON by default in official_ranking.py: normal
    # boilerplate is dropped from BOTH sides before entailment so it cannot
    # prop up precision and recall alike. This reproduces the pre-2026-08
    # unfiltered scores, which are NOT comparable to the filtered ones.
    if args.no_filter_normal:
        cmd.append("--no-filter-normal-findings")
    if args.case_ids:
        cmd += ["--case-ids"] + args.case_ids

    if args.no_radfact:
        print("[INFO] --no-radfact: BLEU/METEOR only, Clinical Score will be "
              "0.0", file=sys.stderr)
        cmd.append("--no-radfact")
    else:
        url = discover_judge(args.judge_url)
        if not url:
            print("[FAIL] no judge server found. Start one with:\n"
                  "[FAIL]   sbatch --partition=gpu --qos=a100 "
                  "--gres=gpu:a100:1 scripts/judge_server.sh\n"
                  "[FAIL] ...or pass --no-radfact for BLEU/METEOR only.",
                  file=sys.stderr)
            return 2
        if not judge_reachable(url):
            print(f"[FAIL] judge not answering at {url}", file=sys.stderr)
            if ((REPO_ROOT / ".judge" / "url").is_file()
                    and not (REPO_ROOT / ".judge" / "ready").is_file()):
                print("[FAIL] the job is queued or loading -- vLLM takes 15-25 "
                      "min. Watch:  tail -f logs/judge_server_*.log",
                      file=sys.stderr)
            return 2
        print(f"[INFO] judge: {url}  (model {args.served_name})",
              file=sys.stderr)
        cmd += ["--judge-backend", "vllm", "--vllm-url", url,
                "--model", args.served_name]

    subprocess.run(cmd, check=True)
    return 0


def run_survey(args, run_dir: Path, case_ids: Optional[List[str]],
               label: Optional[str] = None) -> None:
    """structured_findings_evaluation.py over one run dir, optionally a slice.

    Text and JSON both land under survey/, so successive runs diff directly
    against each other -- that comparison is the whole reason to re-run this
    after a flag change.
    """
    survey_dir = run_dir / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    stem = f"survey_{label}" if label else \
        f"survey_facts_{time.strftime('%Y%m%d_%H%M%S')}"

    cmd = [sys.executable,
           str(module_path("structured_findings_evaluation.py")), str(run_dir),
           "--gt-dir", str(args.gt_dir), "--schema", str(args.schema),
           "--json-out", str(survey_dir / f"{stem}.json")]
    if case_ids:
        cmd += ["--case-ids"] + case_ids

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (survey_dir / f"{stem}.txt").write_text(out, encoding="utf-8")
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print(f"[WARN] survey exited {proc.returncode}; "
              f"{survey_dir / (stem + '.txt')} has the output", file=sys.stderr)
    else:
        print(f"[PASS] survey -> {survey_dir / (stem + '.txt')}",
              file=sys.stderr)


def read_case_list(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def run_split_surveys(args, run_dir: Path, arm_dir: Path) -> None:
    """Three surveys over one prediction set, by what the arm saw."""
    for split in SPLITS:
        lst = arm_dir / f"cases_{split}.txt"
        if not lst.is_file():
            raise SystemExit(f"[FAIL] no case list: {lst}")
        cases = read_case_list(lst)
        # Only cases that actually have a prediction. A list naming cases the
        # run never wrote would silently score a smaller split than its label
        # claims -- and the label is the whole point of splitting.
        have = [c for c in cases
                if (run_dir / "predictions" / f"{c}_pred.json").is_file()]
        print(f"\n===== {split} -- {len(have)} of {len(cases)} case(s) present "
              f"=====")
        if not have:
            print("[WARN] nothing to score, skipping", file=sys.stderr)
            continue
        run_survey(args, run_dir, have, label=split)

    print(f"\n[PASS] three surveys -> {run_dir}/survey/"
          f"survey_{{{','.join(SPLITS)}}}.{{txt,json}}")
    print("[INFO] read 'unseen' as the arm's real number; trained-minus-unseen "
          "is the memorisation gap.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="a run directory: synthesized_reports/ is scored by "
                         "the ranking, predictions/ and summaries/ by the "
                         "survey. One argument, because both read the same run.")
    ap.add_argument("--split", choices=("training", "validate"),
                    default="validate",
                    help="which dataset/<split> the reference reports and "
                         "ground truth come from")
    ap.add_argument("--stage", choices=("all", "rank", "survey"), default="all",
                    help="`rank` is the leaderboard metric and needs the "
                         "judge; `survey` is the per-finding diagnostic and "
                         "needs neither GPU nor model")

    ap.add_argument("--reports-dir", type=Path,
                    help="reference reports (default: dataset/<split>/reports)")
    ap.add_argument("--gt-dir", type=Path,
                    help="generated ground truth (default: "
                         "dataset/<split>/outputs/ground_truth)")
    ap.add_argument("--synthesized-dir", type=Path,
                    help="override <run_dir>/synthesized_reports")
    ap.add_argument("--out-dir", type=Path,
                    help="ranking results (default: <run_dir>/official_ranking)")
    ap.add_argument("--schema", type=Path,
                    default=REPO_ROOT / "schema" / "schema.json")
    ap.add_argument("--case-ids", nargs="+")

    ap.add_argument("--no-radfact", action="store_true",
                    help="BLEU/METEOR only. No judge needed at all.")
    ap.add_argument("--judge-url", help="skip discovery, use this URL")
    ap.add_argument("--served-name", default=DEFAULT_SERVED_NAME)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="cases per incremental write, so summary.json is "
                         "readable mid-run")
    ap.add_argument("--multi-reference", action="store_true",
                    help="score BLEU/METEOR against EVERY reference report for "
                         "a case, not just the first. 59.7%% of cases have 2+. "
                         "Diagnostic only -- leave off for leaderboard parity.")
    ap.add_argument("--no-filter-normal", action="store_true",
                    help="reproduce the pre-2026-08 unfiltered scores. Not "
                         "comparable to the filtered ones.")

    ap.add_argument("--survey-splits", type=Path, nargs="?", const=Path("."),
                    metavar="ARM_DIR",
                    help="three surveys instead of one, from ARM_DIR's "
                         "cases_{trained,evalonly,unseen}.txt (default: the "
                         "run dir's parent). See the module docstring.")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"[FAIL] no such run dir: {run_dir}")
    base = REPO_ROOT / "dataset" / args.split
    args.reports_dir = args.reports_dir or base / "reports"
    args.gt_dir = args.gt_dir or base / "outputs" / "ground_truth"
    reports_dir = args.synthesized_dir or run_dir / "synthesized_reports"
    out_dir = args.out_dir or run_dir / "official_ranking"

    do_rank = args.stage in ("all", "rank")
    do_survey = args.stage in ("all", "survey")

    if do_rank:
        if not reports_dir.is_dir():
            raise SystemExit(
                f"[FAIL] no reports dir: {reports_dir}\n"
                f"[HINT] build them with: scripts/run_infer.sh (STAGE=post)")
        if not args.reports_dir.is_dir():
            raise SystemExit(f"[FAIL] no reference reports: {args.reports_dir}")
        check_nltk()
        n = len(list(reports_dir.glob("*.txt")))
        print(f"[INFO] reports : {reports_dir}  ({n} cases)", file=sys.stderr)
        print(f"[INFO] refs    : {args.reports_dir}", file=sys.stderr)
        print(f"[INFO] out     : {out_dir}", file=sys.stderr)
        rc = run_rank(args, reports_dir, out_dir)
        if rc:
            return rc

    if do_survey:
        if not args.gt_dir.is_dir():
            print(f"[WARN] no ground truth at {args.gt_dir} -- survey skipped. "
                  f"Generate it with scripts/run_gen_gt.sh.", file=sys.stderr)
            return 0
        if args.survey_splits is not None:
            arm_dir = (run_dir.parent if args.survey_splits == Path(".")
                       else args.survey_splits)
            run_split_surveys(args, run_dir, arm_dir)
        else:
            run_survey(args, run_dir, args.case_ids)

    return 0


if __name__ == "__main__":
    sys.exit(main())
