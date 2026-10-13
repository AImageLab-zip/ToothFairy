"""Frozen schema-to-English report runtime with closed-world verification."""

from __future__ import annotations

import json
import re
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .report_template import REPORT_GENERATOR_VERSION, generate_report as generate_canonical_report
from .report_verifier import VerificationResult, verify_polished_report

REPORT_RUNTIME_VERSION = "bite2text-final-report-runtime-v1"
POLISHER_VERSION = "qwen3.5-2b-closed-world-polisher-v1"
_DEFAULT_RUNTIME_ROOT = Path(os.environ.get("B2T_REPORT_ASSET_ROOT", "/opt/ml/model/report_runtime"))
_POLISHER_CACHE: dict[tuple[str, str], "QwenReportPolisher"] = {}
_POLISHER_LOCK = threading.Lock()


@dataclass(frozen=True)
class FinalReportResult:
    report: str
    canonical_report: str
    candidate_report: str | None
    generation_path: str
    candidate_accepted: bool
    used_template_fallback: bool
    fallback_reason: str | None
    verification_reasons: tuple[str, ...]
    canonical_seconds: float
    model_load_seconds: float
    generation_seconds: float
    verification_seconds: float
    total_seconds: float
    runtime_version: str = REPORT_RUNTIME_VERSION
    canonical_builder_version: str = REPORT_GENERATOR_VERSION
    polisher_version: str = POLISHER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_config(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / "RUNTIME_CONFIG.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("runtime_version") != REPORT_RUNTIME_VERSION:
        raise RuntimeError(f"Unexpected report runtime version in {path}")
    return payload


def _clean_model_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^<think>.*?</think>\s*", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^(?:report|output)\s*:\s*", "", cleaned, flags=re.I)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1]
    return re.sub(r"\s+", " ", cleaned).strip()


class QwenReportPolisher:
    def __init__(self, runtime_root: Path, device: str) -> None:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.runtime_root = runtime_root
        self.device = torch.device(device)
        self.config = _load_config(runtime_root)
        model_path = runtime_root / self.config["model"]["local_path"]
        generation = self.config["generation"]
        self.max_new_tokens = int(generation["max_new_tokens"])
        self.enable_thinking = bool(generation["enable_thinking"])
        self.do_sample = bool(generation["do_sample"])
        self.use_cache = bool(generation["use_cache"])
        self.system_prompt = (runtime_root / "SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (runtime_root / "USER_PROMPT_TEMPLATE.txt").read_text(encoding="utf-8").strip()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval()
        print(
            f"B2T_MODEL_DEVICE qwen hf_device_map={getattr(self.model, 'hf_device_map', None)}",
            flush=True,
        )

    def generate_candidate(self, canonical_report: str) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(canonical_report=canonical_report)},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = {key: value.to(self.device) for key, value in self.tokenizer(rendered, return_tensors="pt").items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return _clean_model_output(self.tokenizer.decode(new_tokens, skip_special_tokens=True)), elapsed


def _resolve_device(requested: str) -> str:
    if requested not in {"auto", "cuda", "cuda:0"}:
        raise RuntimeError(f"Production report runtime must use cuda:0, got {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Grand Challenge report runtime requires CUDA; CPU fallback is disabled")
    return "cuda:0"


def _get_polisher(runtime_root: Path, device: str) -> tuple[QwenReportPolisher, float]:
    key = (str(runtime_root.resolve()), device)
    with _POLISHER_LOCK:
        cached = _POLISHER_CACHE.get(key)
        if cached is not None:
            return cached, 0.0
        started = time.perf_counter()
        polisher = QwenReportPolisher(runtime_root, device)
        elapsed = time.perf_counter() - started
        _POLISHER_CACHE[key] = polisher
        return polisher, elapsed


def clear_report_runtime_cache() -> None:
    with _POLISHER_LOCK:
        _POLISHER_CACHE.clear()


def generate_report(
    structured: Mapping[str, Any],
    *,
    runtime_root: str | Path | None = None,
    device: str = "auto",
    enable_llm: bool = True,
) -> FinalReportResult:
    """Generate one final plain-English report from structured facts only.

    The selected frozen policy performs exactly one greedy candidate attempt. It
    does not perform correction retries: any load/generation/verifier failure
    returns the deterministic canonical template.
    """

    total_started = time.perf_counter()
    canonical_started = time.perf_counter()
    canonical = generate_canonical_report(dict(structured))
    canonical_seconds = time.perf_counter() - canonical_started
    canonical_verification = verify_polished_report(structured, canonical)
    if not canonical_verification.accepted:
        raise RuntimeError(f"Canonical report failed closed-world verification: {canonical_verification.reasons}")

    if not enable_llm:
        return FinalReportResult(
            report=canonical,
            canonical_report=canonical,
            candidate_report=None,
            generation_path="template_disabled",
            candidate_accepted=False,
            used_template_fallback=True,
            fallback_reason="llm_disabled",
            verification_reasons=(),
            canonical_seconds=canonical_seconds,
            model_load_seconds=0.0,
            generation_seconds=0.0,
            verification_seconds=0.0,
            total_seconds=time.perf_counter() - total_started,
        )

    root = Path(runtime_root) if runtime_root is not None else _DEFAULT_RUNTIME_ROOT
    if not root.is_dir():
        raise RuntimeError(f"Missing mounted report runtime assets: {root}")
    resolved_device = _resolve_device(device)
    try:
        polisher, model_load_seconds = _get_polisher(root, resolved_device)
        candidate, generation_seconds = polisher.generate_candidate(canonical)
    except Exception as error:
        return FinalReportResult(
            report=canonical,
            canonical_report=canonical,
            candidate_report=None,
            generation_path="template_runtime_fallback",
            candidate_accepted=False,
            used_template_fallback=True,
            fallback_reason=f"{type(error).__name__}: {error}",
            verification_reasons=("report runtime failure",),
            canonical_seconds=canonical_seconds,
            model_load_seconds=locals().get("model_load_seconds", 0.0),
            generation_seconds=0.0,
            verification_seconds=0.0,
            total_seconds=time.perf_counter() - total_started,
        )

    verification_started = time.perf_counter()
    verification: VerificationResult = verify_polished_report(structured, candidate)
    verification_seconds = time.perf_counter() - verification_started
    accepted = verification.accepted
    return FinalReportResult(
        report=candidate if accepted else canonical,
        canonical_report=canonical,
        candidate_report=candidate,
        generation_path="qwen_verified" if accepted else "template_verifier_fallback",
        candidate_accepted=accepted,
        used_template_fallback=not accepted,
        fallback_reason=None if accepted else "closed_world_verifier_rejected_candidate",
        verification_reasons=verification.reasons,
        canonical_seconds=canonical_seconds,
        model_load_seconds=model_load_seconds,
        generation_seconds=generation_seconds,
        verification_seconds=verification_seconds,
        total_seconds=time.perf_counter() - total_started,
    )
