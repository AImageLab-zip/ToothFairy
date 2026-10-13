#!/usr/bin/env python3
"""
code/train/sft.py

The LoRA SFT pipeline, one stage at a time.

This is the training ENTRY POINT: `scripts/run_sft.sh` submits it. It is NOT
the trainer -- `trainer.py` is the training loop, and this file only decides
what runs, where, and under which interpreter. See code/train/README.md for
what the modules are and code/train/dataset.py for what a row is.

    STAGE=pool     sft.py    select the training pool
    STAGE=targets  sft.py    build sft_targets.jsonl
    STAGE=parity   sft.py    token-id equality gate              [GPU]
    STAGE=train    sft.py    the arm                             [GPU]
    STAGE=merge    sft.py    adapter -> servable AWQ checkpoint

WHY THIS FILE EXISTS, WHICH IS NOT "TO HAVE A DRIVER"
─────────────────────────────────────────────────────
The stages do NOT agree on where their client runs or which Python runs it.
That map is the whole content of this file, and it was previously spread
across five shell scripts where nothing stated it:

  stage    server    client runs      interpreter
  ------------------------------------------------------------------
  pool     -         host             base   (nibabel / PIL / schema)
  targets  -         host             base
  parity   student   host             SFT    (torch 2.11 / transformers 5.9)
  train    -         host             SFT
  merge    -         host             SFT

Two more -- `draft` and `screen`, the visual-evidence pass -- register
themselves from code/train/visual_evidence/ when that directory is present.
It is a research side arm and is not part of the released path.

The container/host split is not stylistic: vLLM exists only inside the image,
and the training stack (`torch==2.11.0`, `transformers==5.9.0`) exists only in
`cbct_sft_cu128`, which is a different env from the `cbct_base` that holds
nibabel and the schema tooling. Running a stage under the wrong one of the
three fails late and confusingly -- a missing nibabel forty minutes in, or a
peft/transformers version error after the model has loaded. Here it is one
table, checked before anything starts.

PASS-THROUGH, RATHER THAN RE-DECLARING FIFTY FLAGS
──────────────────────────────────────────────────
Each underlying module has its own tuned defaults -- TOKENS_PER_REQUEST,
MIN_VISIBLE, the arm table, the LoRA hyper-parameters. This file does not
mirror them: everything after `--` goes to the stage's module verbatim.

    sft.py --stage train -- --arm vision+merger --rows sft_targets.jsonl

That way a flag this file has never heard of still works, and a default only
ever lives in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
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

# ── the stage table ─────────────────────────────────────────────────────────
# module     what to run
# python     "base" | "sft" | "container"  -- see the docstring's table
# server     None, or the served-model-name the stage needs on :PORT
STAGES = {
    "pool":    dict(module="select_sft_pool.py",           python="base",      server=None),
    "targets": dict(module="build_sft_targets.py",         python="base",      server=None),
    "parity":  dict(module="check_prompt_parity.py",       python="sft",       server="student"),
    "train":   dict(module="trainer.py",                   python="sft",       server=None),
    "merge":   dict(module="merge_vision_lora.py",         python="sft",       server=None),
}

# Which model each served name is. The teacher is the big one that writes
# evidence; the student is the model that will actually be trained and served,
# and the screen MUST use it -- asking the teacher whether its own prose is
# visible is not a test of anything.
SERVER_MODEL = {"teacher": "TEACHER_MODEL", "student": "STUDENT_MODEL"}


class VLLM:
    """A vLLM started inside the pyxis image, from the host.

    Same shape as gen_gt.TextServer and for the same reason -- vLLM lives in
    the container and the thing that calls it may not. Kept separate rather
    than shared because that one serves text and proves a port against a
    co-bound judge, and this one serves vision and does not.
    """

    def __init__(self, model_path: str, served: str, port: int,
                 container: Path, model_dir: Path, project_dir: Path,
                 gpu_mem_util: float, max_model_len: int, log_path: Path):
        self.cmd = [
            "srun", "--overlap", f"--container-image={container}",
            f"--container-mounts={model_dir}:/models,{project_dir}:/project",
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", served,
            "--port", str(port), "--max-model-len", str(max_model_len),
            "--limit-mm-per-prompt", json.dumps({"image": 3}),
            "--gpu-memory-utilization", str(gpu_mem_util),
            "--enable-prefix-caching",
        ]
        self.port, self.served, self.log_path = port, served, log_path
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
        """Poll /v1/models for OUR served name. Never `sleep N`: a fixed wait
        after a server that is already up is that many seconds of a GPU
        allocation spent doing nothing."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited {self.proc.returncode} before becoming "
                    f"ready; see {self.log_path}")
            try:
                with urllib.request.urlopen(
                        f"http://localhost:{self.port}/v1/models",
                        timeout=5) as r:
                    body = json.loads(r.read().decode("utf-8"))
                if any(m.get("id") == self.served for m in body.get("data", [])):
                    return time.time() - t0
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(poll_s)
        raise TimeoutError(f"vLLM not ready after {timeout_s:.0f}s "
                           f"({self.log_path})")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def interpreter(kind: str, args) -> List[str]:
    """The argv prefix that runs a stage's module.

    Checked here, before a server is started or a GPU is touched, because all
    three failure modes are late and unhelpful otherwise.
    """
    if kind == "base":
        return [args.base_python]
    if kind == "sft":
        if not Path(args.sft_python).is_file():
            raise SystemExit(
                f"[FAIL] no training interpreter: {args.sft_python}\n"
                f"[HINT] this stage needs torch 2.11 / transformers 5.9, which "
                f"live in cbct_sft_cu128 and NOT in cbct_base. Set SFT_PY.")
        return [args.sft_python]
    # container: the module runs inside the image, next to the server
    return ["srun", "--overlap", f"--container-image={args.container}",
            f"--container-mounts={args.model_dir}:/models,"
            f"{args.project_dir}:/project",
            "--container-workdir=/project", "python3"]


def module_arg(kind: str, module: str) -> str:
    """Where the module is, from the point of view of the interpreter.

    Both answers come from module_path(), which searches the code groups, so a
    module that moves between them needs no edit here. The container path used
    to be spelled `/project/code/train/{module}` -- a second, silently stale
    copy of the layout the moment a stage's module lived anywhere else.
    """
    p = module_path(module)
    if kind == "container":
        return f"/project/{p.relative_to(REPO_ROOT).as_posix()}"
    return str(p)


def register_research_stages() -> None:
    """Add the visual-evidence stages, if this repo has them.

    Lazy, and inside a function, for one reason beyond tidiness:
    tools/make_release.py derives the released file set from the MODULE-SCOPE
    import closure of the entry points, so an import in here is how a stage
    stays out of it. The released tree has no visual_evidence/ directory and
    `--stage draft` there is an argparse error, which is the honest answer.
    """
    try:
        import evidence_stages           # code/train/visual_evidence/
    except ImportError:
        return
    STAGES.update(evidence_stages.STAGES)


def main() -> int:
    register_research_stages()          # before --stage choices are built
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=tuple(STAGES),
                    help="which step of the pipeline to run")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--container", type=Path,
                    default=Path.home() / "containers" / "vllm019_cu128.sqsh")
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--project-dir", type=Path, default=REPO_ROOT)
    ap.add_argument("--teacher-model", default="Qwen3.5-27B",
                    help="writes the evidence (STAGE=draft, research only)")
    ap.add_argument("--student-model", default="Qwen3.5-9B-AWQ",
                    help="the model that will be trained and served. The "
                         "screen and the parity gate MUST use it: asking the "
                         "teacher whether its own prose is visible tests "
                         "nothing.")
    ap.add_argument("--base-python", default="python3",
                    help="cbct_base: nibabel, PIL, the schema tooling")
    ap.add_argument("--sft-python",
                    default=str(Path.home() / "miniconda3" / "envs"
                                / "cbct_sft_cu128" / "bin" / "python3"),
                    help="cbct_sft_cu128: torch 2.11, transformers 5.9")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--startup-timeout", type=float, default=2700.0)
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="everything after `--` goes to the stage's module "
                         "verbatim")
    args = ap.parse_args()

    spec = STAGES[args.stage]
    passthrough = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    # Resolve the interpreter FIRST. A missing training env should cost a
    # second, not a queue wait plus a model load.
    prefix = interpreter(spec["python"], args)
    cmd = prefix + [module_arg(spec["python"], spec["module"])] + passthrough

    server: Optional[VLLM] = None
    if spec["server"]:
        name = spec["server"]
        model = (args.teacher_model if name == "teacher"
                 else args.student_model)
        if not (args.model_dir / model).is_dir():
            raise SystemExit(f"[FAIL] no model: {args.model_dir / model}")
        if not args.container.is_file():
            raise SystemExit(f"[FAIL] no container: {args.container}")
        args.log_dir.mkdir(parents=True, exist_ok=True)
        server = VLLM(f"/models/{model}", name, args.port, args.container,
                      args.model_dir, args.project_dir,
                      args.gpu_memory_utilization, args.max_model_len,
                      args.log_dir / f"sft_{args.stage}_vllm.log")
        print(f"[INFO] stage={args.stage}: starting {model} as '{name}' on "
              f"port {args.port}", file=sys.stderr)
        server.start()
        try:
            waited = server.wait_ready(args.startup_timeout)
        except BaseException:
            server.stop()
            raise
        print(f"[PASS] {name} ready after {waited:.0f}s", file=sys.stderr)
        # The module reaches the server over HTTP; on this cluster both sides
        # are on the same node, so localhost is right in the container too.
        cmd += ["--vllm-url", f"http://localhost:{args.port}/v1",
                "--model", name]

    print(f"[INFO] {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd).returncode
    finally:
        if server:
            server.stop()
    print(f"[{'PASS' if rc == 0 else 'FAIL'}] stage={args.stage} exited {rc} "
          f"after {time.time() - t0:.0f}s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
