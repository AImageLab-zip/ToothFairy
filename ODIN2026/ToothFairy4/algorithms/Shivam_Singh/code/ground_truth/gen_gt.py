#!/usr/bin/env python3
"""
code/ground_truth/gen_gt.py

Reference reports -> the labels everything else is scored against.

This is the ground-truth ENTRY POINT: `scripts/run_gen_gt.sh` is the shell
runner that submits it. It drives one module, `parse_reports_to_gt.py`, and
what it adds is the thing that module cannot do for itself -- stand up the text
model it calls, prove the port belongs to it, and say afterwards whether the
output is the shape every consumer expects.

TWO STAGES, AND ONLY THE FIRST NEEDS A GPU
──────────────────────────────────────────
  stage 1 (LLM)   report text -> {case}_report_facts.json, a small
                  report-shaped intermediate. TWO calls per report -- mandible
                  and maxilla -- not the 34 the per-fact/per-tooth design
                  needed.
  stage 2 (CPU)   report_facts -> {case}_gt.json, the full schema shape,
                  derived deterministically so the arch map, the per-tooth
                  facts and the wisdom-tooth facts cannot contradict each other
                  the way the LLM-answers-everything version did.

`--stage all` (default) runs both and needs a server. `--stage derive` re-runs
stage 2 alone, anywhere, with no GPU and no queue -- which is what makes a
change to the deterministic half cheap to test. Both files land in --out-dir;
only *_gt.json is read downstream.

THE FILENAME IS THE WHOLE CONTRACT
──────────────────────────────────
`structured_findings_evaluation.py` looks up exactly one file per case:
`{case}_gt.json`. A multi-report case run with neither --consensus nor
--first-report-only writes only `{case}_{radiologist}_gt.json`, so that case is
skipped with NO ERROR and the run reports a plausible score over a subset. On
dataset/validate that is 26 of 40 cases. This script counts both shapes when it
finishes and says so, because nothing else will.

WHY THE PORT IS CHECKED THREE TIMES
───────────────────────────────────
vLLM binds with SO_REUSEPORT, so a second server on a busy port does NOT fail
with "address already in use" -- both bind, and the kernel round-robins
connections between them. scripts/judge_server.sh defaults to 8001 and can hold
the same node, serving `qwen3-14b-text` while this job asks for `qwen3-text`,
so roughly half of every extraction call landed on the judge and returned 404:
logs/gen_gt_549316.log has 342 x 200 against 379 x 404 in
logs/judge_server_549303.log at the same timestamps. A report needs all of its
calls to succeed, so at a ~50% collision rate essentially no case survived.

Hence: scan for a port nobody answers on BEFORE starting; wait for readiness on
/v1/models rather than /health, because /health proves *a* server is there and
not that it is ours; then probe repeatedly, because one probe can pass by luck
when connections are handed out round-robin.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits. See code/_repo.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(
    p for p in _Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths, module_path  # noqa: E402

add_code_paths()

SERVED_NAME = "qwen3-text"


# ── the text server ─────────────────────────────────────────────────────────
class TextServer:
    """A text-only vLLM, started either in a pyxis container or in-process.

    Deliberately NOT infer.py's VLLMServer: that one launches `python3 -m vllm`
    in its own interpreter with vision flags (--limit-mm-per-prompt) and waits
    on /health. This one is launched from the HOST into a container via srun --
    the client then calls it over HTTP from the host, because extraction makes
    network calls and loads nothing itself -- and it has to prove the port is
    its own. Sharing one class would mean a parameter for every one of those
    differences.
    """

    def __init__(self, model_path: str, port: int, max_model_len: int,
                 gpu_mem_util: float, container: Optional[Path],
                 model_dir: Path, log_path: Path):
        vllm = ["python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path, "--served-model-name", SERVED_NAME,
                "--port", str(port), "--dtype", "bfloat16",
                "--max-model-len", str(max_model_len),
                "--reasoning-parser", "qwen3",
                "--gpu-memory-utilization", str(gpu_mem_util),
                "--enable-prefix-caching"]
        if container:
            # --overlap: this job already holds the allocation, and the step
            # must share it rather than queue behind itself.
            self.cmd = ["srun", "--overlap",
                        f"--container-image={container}",
                        f"--container-mounts={model_dir}:/models"] + vllm
        else:
            self.cmd = [sys.executable] + vllm[1:]
        self.port, self.log_path = port, log_path
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1)
        threading.Thread(target=self._tail, daemon=True).start()

    def _tail(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._fh.write(line)
            self._fh.flush()

    def wait_ready(self, timeout_s: float, poll_s: float = 1.0) -> float:
        """Poll until /v1/models answers as SERVED_NAME. Returns seconds waited.

        Polling every second rather than `sleep 60` first: a 14B text model on
        an A100 takes 10-20 minutes, but the point of not sleeping is that when
        it takes four, you get four.
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with {self.proc.returncode} before becoming "
                    f"ready; {self.log_path} has its output")
            if serves(self.port, SERVED_NAME):
                return time.time() - t0
            time.sleep(poll_s)
        raise TimeoutError(f"vLLM not ready after {timeout_s:.0f}s "
                           f"({self.log_path})")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def serves(port: int, name: str, timeout: float = 5.0) -> bool:
    """True if /v1/models on this port lists `name`."""
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False
    return any(m.get("id") == name for m in body.get("data", []))


def answers(port: int, timeout: float = 3.0) -> bool:
    """True if ANYTHING answers /health here -- i.e. the port is taken."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health",
                               timeout=timeout).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def free_port(start: int, scan: int) -> int:
    """First port in [start, start+scan) that nobody is already serving on.

    Probed BEFORE starting, because SO_REUSEPORT means a busy port does not
    announce itself at bind time.
    """
    for port in range(start, start + scan):
        if not answers(port):
            return port
        print(f"[WARN] port {port} is already serving; trying {port + 1}",
              file=sys.stderr)
    raise SystemExit(
        f"[FAIL] no free port in {start}..{start + scan - 1}. Another vLLM job "
        f"is likely on this node -- check `squeue -u $USER`, or pass --port.")


def prove_exclusive(port: int, probes: int = 8) -> None:
    """Require EVERY probe to answer as ours.

    One /v1/models probe can pass by luck if a second server co-bound this port
    between the scan and now: connections are handed out round-robin, so a
    single success proves only that one of them is ours.
    """
    for i in range(1, probes + 1):
        if not serves(port, SERVED_NAME):
            raise SystemExit(
                f"[FAIL] port {port} is shared with another vLLM server "
                f"(probe {i}/{probes} did not answer as {SERVED_NAME}). "
                f"Extraction calls would split between the two and 404 on the "
                f"wrong one.\n[HINT] rerun with an explicit free --port, or "
                f"wait for the other job.")
    print(f"[PASS] port {port} verified exclusive to this job's "
          f"{SERVED_NAME} server", file=sys.stderr)


# ── extraction ──────────────────────────────────────────────────────────────
def extract(args, vllm_url: Optional[str]) -> None:
    """One call to parse_reports_to_gt.py, with the flags this run implies."""
    cmd = [sys.executable, str(module_path("parse_reports_to_gt.py")),
           "--reports-dir", str(args.reports_dir),
           "--schema", str(args.schema),
           "--out-dir", str(args.out_dir)]
    if args.stage == "derive":
        cmd.append("--from-report-facts")
    if vllm_url:
        cmd += ["--vllm-url", vllm_url, "--model", SERVED_NAME]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.case_ids:
        cmd += ["--case-ids"] + args.case_ids
    if args.resume:
        cmd.append("--resume")
    if args.consensus:
        cmd.append("--consensus")
    if args.first_report_only:
        cmd.append("--first-report-only")
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)


def auto_resolve(args) -> int:
    """The audit, with no human in it: repair, mask, replay, gate.

    The audit's four repairs are mechanical (FDI numbering or one stated
    range decides them). Whatever still raises an ERROR after them is a claim
    the report text cannot settle, and --mask-unresolved takes it out of its
    reader's file; stage 2 then nulls every label that claim decides. Returns the final audit's exit code,
    which is 0 exactly when nothing is left for anyone to decide.
    """
    audit = [sys.executable, str(module_path("audit_report_facts.py")),
             "--gt-dir", str(args.out_dir), "--reports-dir", str(args.reports_dir),
             "--quiet"]
    if args.case_ids:
        audit += ["--case-ids"] + args.case_ids
    subprocess.run(audit + ["--fix-laterality", "--fix-arch-range",
                            "--fix-canal-adjacency", "--fix-intrasinusal",
                            "--mask-unresolved"])
    replay = argparse.Namespace(**{**vars(args), "stage": "derive",
                                   "resume": False, "limit": None})
    extract(replay, vllm_url=None)
    return subprocess.run(audit).returncode


# ── what got written, and whether it is usable ──────────────────────────────
_PER_READER = re.compile(r"_[^_]+_gt\.json$")


def summarise(out_dir: Path, consensus: bool, first_only: bool) -> None:
    """Count both filename shapes and warn if the usable one is missing.

    This is the check the shell version could only approximate with `find`
    regexes, and it is the one that matters: a run can succeed loudly and still
    produce nothing the evaluation will look up.
    """
    gts = sorted(out_dir.glob("*_gt.json"))
    per_reader = [p for p in gts if _PER_READER.search(p.name)]
    per_case = [p for p in gts if p not in per_reader]

    print(f"\n[INFO] {out_dir}:")
    print(f"       {len(per_case)} x {{case}}_gt.json"
          f"            <- what the evaluation reads")
    print(f"       {len(per_reader)} x {{case}}_{{radiologist}}_gt.json  "
          f"<- per reader, not looked up")

    if per_reader and not (consensus or first_only):
        cases = {_PER_READER.sub("", p.name) for p in per_reader}
        orphans = sorted(c for c in cases
                         if not (out_dir / f"{c}_gt.json").exists())
        if orphans:
            print(f"\n[WARN] {len(orphans)} multi-report case(s) have NO "
                  f"{{case}}_gt.json and will be SILENTLY SKIPPED by "
                  f"structured_findings_evaluation.py:", file=sys.stderr)
            print(f"[WARN]   {', '.join(orphans[:12])}"
                  f"{' ...' if len(orphans) > 12 else ''}", file=sys.stderr)
            print(f"[WARN] Re-run with --consensus (merge the readers) or "
                  f"--first-report-only (take the lowest-numbered one).",
                  file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("training", "validate"),
                    help="shorthand for the dataset/<split> layout; "
                         "--reports-dir and --out-dir override it")
    ap.add_argument("--reports-dir", type=Path,
                    help="one {case}_{n}.txt per radiologist")
    ap.add_argument("--out-dir", type=Path,
                    help="where *_report_facts.json and *_gt.json land")
    ap.add_argument("--schema", type=Path,
                    default=REPO_ROOT / "schema" / "schema.json")

    ap.add_argument("--stage", choices=("all", "derive"), default="all",
                    help="`all` (default) runs the LLM extraction and then "
                         "derives the schema shape from it, and needs a "
                         "server. `derive` re-runs only the deterministic "
                         "second half from existing *_report_facts.json -- no "
                         "GPU, no queue, which is what makes a change to that "
                         "half cheap to test.")

    ap.add_argument("--consensus", action="store_true",
                    help="merge every radiologist of a multi-report case into "
                         "one {case}_gt.json, field by field")
    ap.add_argument("--first-report-only", action="store_true",
                    help="one report per case, the lowest-numbered "
                         "radiologist, written straight to {case}_gt.json. "
                         "The cheapest way to get the filename the evaluation "
                         "looks up.")
    ap.add_argument("--no-auto-resolve", dest="auto_resolve",
                    action="store_false",
                    help="skip the default last step: the audit's mechanical "
                         "repairs, masking every claim they leave unsettled, a "
                         "stage-2 replay and a gate on a clean audit -- which "
                         "is what makes the ground truth free of human "
                         "decisions. Exit code 1 if anything is still "
                         "unsettled. Without it the output is the raw "
                         "extraction.")
    ap.add_argument("--limit", type=int, help="smoke test: N cases, not files")
    ap.add_argument("--case-ids", nargs="+")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases whose output already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="group and validate the report files and print the "
                         "plan. No model call, so no server is started -- "
                         "there is no reason to pay a GPU allocation to check "
                         "inputs.")

    ap.add_argument("--vllm-url",
                    help="use an ALREADY-RUNNING server instead of starting "
                         "one. The port checks are skipped: you named it.")
    ap.add_argument("--container", type=Path,
                    help="pyxis image to start vLLM in. Omitted, vLLM is "
                         "started in this interpreter -- which is right when "
                         "gen_gt.py is itself already inside the container.")
    ap.add_argument("--model-dir", type=Path,
                    default=REPO_ROOT / "models")
    ap.add_argument("--model-name", default="Qwen3-14B")
    ap.add_argument("--port", type=int, default=8011,
                    help="8011, not 8001: judge_server.sh defaults to 8001 and "
                         "can hold the same node. See the module docstring.")
    ap.add_argument("--port-scan", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=24576,
                    help="24576, not 32768: --qos=a100 can land on an "
                         "A100-40GB, where Qwen3-14B's 27.5 GiB of weights "
                         "plus CUDA graphs leave ~3.9 GiB for KV while 32768 "
                         "tokens need 5.0 -- vLLM then refuses to start at "
                         "all. This job's worst case is ~4.5k tokens anyway.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--startup-timeout", type=float, default=1800.0)
    args = ap.parse_args()

    if args.split:
        base = REPO_ROOT / "dataset" / args.split
        args.reports_dir = args.reports_dir or base / "reports"
        args.out_dir = args.out_dir or base / "outputs" / "ground_truth"
    if not args.reports_dir or not args.out_dir:
        ap.error("give --split, or both --reports-dir and --out-dir")
    if args.consensus and args.first_report_only:
        ap.error("--consensus and --first-report-only are mutually exclusive: "
                 "consensus needs every radiologist's report")
    if not args.reports_dir.is_dir():
        raise SystemExit(f"[FAIL] no reports dir: {args.reports_dir}\n"
                         f"[HINT] expected one {{case}}_{{n}}.txt per reader")
    if not args.schema.is_file():
        raise SystemExit(f"[FAIL] no schema: {args.schema}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_reports = len(list(args.reports_dir.glob("*.txt")))
    print(f"[INFO] stage={args.stage}  {n_reports} report file(s) in "
          f"{args.reports_dir}  ->  {args.out_dir}", file=sys.stderr)

    # Neither of these touches a model, so neither pays for a server.
    if args.dry_run or args.stage == "derive":
        extract(args, vllm_url=None)
        if args.dry_run:
            return 0
        rc = auto_resolve(args) if args.auto_resolve else 0
        summarise(args.out_dir, args.consensus, args.first_report_only)
        return rc

    if args.vllm_url:
        extract(args, vllm_url=args.vllm_url)
        rc = auto_resolve(args) if args.auto_resolve else 0
        summarise(args.out_dir, args.consensus, args.first_report_only)
        return rc

    model_path = (f"/models/{args.model_name}" if args.container
                  else str(args.model_dir / args.model_name))
    if not (args.model_dir / args.model_name).is_dir():
        raise SystemExit(f"[FAIL] no model: {args.model_dir / args.model_name}")
    if args.container and not args.container.is_file():
        raise SystemExit(f"[FAIL] no container: {args.container}")

    port = free_port(args.port, args.port_scan)
    server = TextServer(model_path, port, args.max_model_len,
                        args.gpu_memory_utilization, args.container,
                        args.model_dir, args.out_dir / "vllm_startup.log")
    print(f"[INFO] starting {args.model_name} on port {port}", file=sys.stderr)
    server.start()
    try:
        waited = server.wait_ready(args.startup_timeout)
        print(f"[PASS] vLLM ready after {waited:.0f}s", file=sys.stderr)
        prove_exclusive(port)
        extract(args, vllm_url=f"http://localhost:{port}/v1")
    finally:
        server.stop()

    # After the server is gone: the audit is CPU and should not hold the GPU.
    rc = auto_resolve(args) if args.auto_resolve else 0
    summarise(args.out_dir, args.consensus, args.first_report_only)
    return rc


if __name__ == "__main__":
    sys.exit(main())
