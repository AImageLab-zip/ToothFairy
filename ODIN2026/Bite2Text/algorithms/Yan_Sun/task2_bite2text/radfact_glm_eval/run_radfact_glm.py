#!/usr/bin/env python3
"""Run RadFact-Lite with an OpenAI-compatible GLM endpoint.

The API key is read only from an environment variable. Every successful LLM
response is cached by request content, making long evaluations resumable and
allowing reference parsing to be reused across candidate versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate
from openai import BadRequestError, OpenAI
from radfact_lite.clients import Message
from radfact_lite.entailment import assess_entailment
from radfact_lite.finding_filter import remove_normal_findings
from radfact_lite.metric import aggregate_results, sample_result
from radfact_lite.phrase_parser import parse_report_to_phrases
from radfact_lite.rf_types import ReportType


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "photo_pipeline"
    / "fact_correction_crossfit_v1"
    / "crossfit_predictions.jsonl"
)
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.2"
DEFAULT_KEY_ENV = "RADFACT_API_KEY"
UPSTREAM_COMMIT = "053f680be1c57225f94d67b198a34aa871b1127d"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response contains no JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed


class CachedGLMJSONClient:
    """JSON client compatible with radfact_lite's small JSONClient protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        cache_dir: Path,
        timeout: float,
        max_retries: int,
        min_interval: float,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_request_at = 0.0
        self._unsupported: set[str] = set()

    def _schema_instruction(self, response_format: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        json_schema = response_format.get("json_schema", {})
        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        name = str(json_schema.get("name", "response"))
        instruction = (
            "\n\nReturn only one valid JSON object. It must satisfy this JSON Schema "
            f"for '{name}':\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
        )
        return instruction, schema

    def _request_messages(
        self, messages: list[Message], response_format: dict[str, Any]
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        instruction, schema = self._schema_instruction(response_format)
        rendered = [{"role": message.role, "content": message.content} for message in messages]
        for item in rendered:
            if item["role"] == "system":
                item["content"] += instruction
                break
        else:
            rendered.insert(0, {"role": "system", "content": instruction.strip()})
        return rendered, schema

    def _cache_path(
        self,
        rendered_messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> Path:
        request_identity = {
            "adapter": "glm-json-object-v1",
            "model": self.model,
            "base_url": self.base_url,
            "messages": rendered_messages,
            "schema": schema,
            "temperature": 0,
            "thinking": "disabled",
        }
        digest = hashlib.sha256(canonical_json(request_identity).encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _throttle(self) -> None:
        remaining = self.min_interval - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if "temperature" not in self._unsupported:
            kwargs["temperature"] = 0
        if "response_format" not in self._unsupported:
            kwargs["response_format"] = {"type": "json_object"}
        if "thinking" not in self._unsupported:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        while True:
            self._throttle()
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._last_request_at = time.monotonic()
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Model returned empty content")
                return content
            except BadRequestError as exc:
                message = str(exc).lower()
                removed = None
                for name in ("thinking", "response_format", "temperature"):
                    if name in kwargs or (name == "thinking" and "extra_body" in kwargs):
                        if name in message or "unsupported" in message or "invalid parameter" in message:
                            removed = name
                            break
                if removed is None:
                    raise
                self._unsupported.add(removed)
                if removed == "thinking":
                    kwargs.pop("extra_body", None)
                else:
                    kwargs.pop(removed, None)

    def complete_json(
        self, messages: list[Message], response_format: dict[str, Any]
    ) -> dict[str, Any]:
        rendered, schema = self._request_messages(messages, response_format)
        cache_path = self._cache_path(rendered, schema)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            validate(instance=cached["response"], schema=schema)
            self.cache_hits += 1
            return cached["response"]

        last_error: Exception | None = None
        attempt_messages = list(rendered)
        for attempt in range(1, self.max_retries + 2):
            try:
                content = self._chat_completion(attempt_messages)
                parsed = extract_json_object(content)
                validate(instance=parsed, schema=schema)
                atomic_write_json(
                    cache_path,
                    {
                        "model": self.model,
                        "base_url": self.base_url,
                        "response": parsed,
                    },
                )
                self.cache_misses += 1
                return parsed
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                attempt_messages = rendered + [
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid. Return only a JSON object that "
                            "strictly satisfies the schema; do not use Markdown fences."
                        ),
                    }
                ]
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Unable to obtain schema-valid JSON: {last_error}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"patient_id", "prediction", "reference"} - row.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing keys: {sorted(missing)}")
            patient_id = str(row["patient_id"])
            if patient_id in seen:
                raise ValueError(f"Duplicate patient_id: {patient_id}")
            seen.add(patient_id)
            rows.append(
                {
                    "patient_id": patient_id,
                    "prediction": str(row["prediction"]),
                    "reference": str(row["reference"]),
                }
            )
    return rows


def select_rows(
    rows: list[dict[str, Any]], *, sample_size: int | None, seed: int, limit: int | None
) -> list[dict[str, Any]]:
    if sample_size is not None:
        if sample_size <= 0 or sample_size > len(rows):
            raise ValueError(f"--sample-size must be between 1 and {len(rows)}")
        selected_ids = {
            row["patient_id"] for row in random.Random(seed).sample(rows, sample_size)
        }
        rows = [row for row in rows if row["patient_id"] in selected_ids]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    return rows


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                completed[str(item["patient_id"])] = item
    return completed


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_sample(
    client: CachedGLMJSONClient,
    row: dict[str, Any],
    *,
    filter_normal: bool,
    report_type: ReportType,
) -> dict[str, Any]:
    candidate_phrases = parse_report_to_phrases(client, row["prediction"], report_type)
    reference_phrases = parse_report_to_phrases(client, row["reference"], report_type)
    parsed_candidate_phrases = list(candidate_phrases)
    parsed_reference_phrases = list(reference_phrases)

    if filter_normal:
        candidate_phrases = remove_normal_findings(client, candidate_phrases, report_type)
        reference_phrases = remove_normal_findings(client, reference_phrases, report_type)

    precision_decisions = [
        asdict(assess_entailment(client, reference_phrases, phrase, report_type))
        for phrase in candidate_phrases
    ]
    recall_decisions = [
        asdict(assess_entailment(client, candidate_phrases, phrase, report_type))
        for phrase in reference_phrases
    ]
    entailed_candidate = sum(x["status"] == "entailment" for x in precision_decisions)
    entailed_reference = sum(x["status"] == "entailment" for x in recall_decisions)
    metric = sample_result(
        row["patient_id"],
        entailed_candidate,
        entailed_reference,
        len(candidate_phrases),
        len(reference_phrases),
    )
    return {
        "patient_id": row["patient_id"],
        **asdict(metric),
        "parsed_candidate_phrases": parsed_candidate_phrases,
        "parsed_reference_phrases": parsed_reference_phrases,
        "scored_candidate_phrases": candidate_phrases,
        "scored_reference_phrases": reference_phrases,
        "precision_decisions": precision_decisions,
        "recall_decisions": recall_decisions,
    }


def write_summary(
    path: Path,
    results: list[dict[str, Any]],
    *,
    failures: int,
    client: CachedGLMJSONClient,
    config: dict[str, Any],
) -> None:
    metrics = [
        sample_result(
            item["patient_id"],
            int(item["entailed_candidate_count"]),
            int(item["entailed_reference_count"]),
            int(item["candidate_count"]),
            int(item["reference_count"]),
        )
        for item in results
    ]
    aggregate = aggregate_results(metrics, failures)
    atomic_write_json(
        path,
        {
            "aggregate": asdict(aggregate),
            "completed_samples": len(results),
            "failed_samples": failures,
            "cache_hits_this_process": client.cache_hits,
            "api_calls_this_process": client.cache_misses,
            "configuration": config,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-dir", type=Path, default=SCRIPT_DIR / "runs" / "v7_glm52")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional shared content-addressed LLM cache (defaults to RUN_DIR/llm_cache).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default=DEFAULT_KEY_ENV)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=0.15)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--report-type",
        choices=[item.value for item in ReportType],
        default=ReportType.BITE2TEXT.value,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--prewarm-references",
        action="store_true",
        help="Parse and cache reference phrases only, without scoring candidates.",
    )
    parser.add_argument(
        "--remove-normal-findings",
        dest="filter_normal",
        action="store_true",
        default=True,
        help="Filter normal findings before entailment (default).",
    )
    parser.add_argument(
        "--keep-normal-findings",
        dest="filter_normal",
        action="store_false",
        help="Score normal and abnormal findings together.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    run_dir = args.run_dir.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else run_dir / "llm_cache"
    report_type = ReportType(args.report_type)
    rows = load_jsonl(input_path)
    rows = select_rows(rows, sample_size=args.sample_size, seed=args.seed, limit=args.limit)
    config = {
        "input": str(input_path),
        "selected_patient_ids": [row["patient_id"] for row in rows],
        # Completed samples are keyed by patient ID. Bind the run to report
        # contents as well, so replacing predictions at the same path cannot
        # silently reuse scores from an earlier model.
        "selected_input_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "api_key_env": args.api_key_env,
        "report_type": report_type.value,
        "remove_normal_findings": args.filter_normal,
        "radfact_lite_commit": UPSTREAM_COMMIT,
        "adapter": "glm-json-object-v1",
    }
    signature = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    config["signature"] = signature

    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "num_samples": len(rows), **config}, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"Missing API key environment variable: {args.api_key_env}. "
            "The key is intentionally never read from a project file.",
            file=sys.stderr,
        )
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            print(
                f"Run configuration differs from {config_path}. Use a new --run-dir.",
                file=sys.stderr,
            )
            return 2
    else:
        atomic_write_json(config_path, config)

    client = CachedGLMJSONClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        cache_dir=cache_dir,
        timeout=args.timeout,
        max_retries=args.max_retries,
        min_interval=args.min_interval,
    )

    if args.probe:
        probe_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "ok"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
        response = client.complete_json(
            [Message(role="system", content="You are a JSON API probe."), Message(role="user", content="Return status ok.")],
            probe_schema,
        )
        print(json.dumps({"status": "probe_ok", "response": response}, indent=2))
        return 0

    if args.prewarm_references:
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            phrases = parse_report_to_phrases(client, row["reference"], report_type)
            print(
                f"[{index}/{total}] {row['patient_id']}: reference_phrases={len(phrases)} "
                f"API={client.cache_misses} cache={client.cache_hits}"
            )
        atomic_write_json(
            run_dir / "prewarm_summary.json",
            {
                "completed_references": total,
                "cache_hits_this_process": client.cache_hits,
                "api_calls_this_process": client.cache_misses,
                "configuration": config,
            },
        )
        return 0

    results_path = run_dir / "per_sample.jsonl"
    failures_path = run_dir / "failures.jsonl"
    completed = load_completed(results_path)
    failures = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        patient_id = row["patient_id"]
        if patient_id in completed:
            print(f"[{index}/{total}] {patient_id}: cached sample")
            continue
        try:
            result = evaluate_sample(
                client,
                row,
                filter_normal=args.filter_normal,
                report_type=report_type,
            )
            append_jsonl(results_path, result)
            completed[patient_id] = result
            print(
                f"[{index}/{total}] {patient_id}: "
                f"P={result['logical_precision']:.4f} "
                f"R={result['logical_recall']:.4f} "
                f"F1={result['logical_f1']:.4f} "
                f"API={client.cache_misses} cache={client.cache_hits}"
            )
        except Exception as exc:  # Continue so a long run remains useful.
            failures += 1
            append_jsonl(
                failures_path,
                {
                    "patient_id": patient_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            print(f"[{index}/{total}] {patient_id}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

        ordered_results = [completed[row["patient_id"]] for row in rows if row["patient_id"] in completed]
        write_summary(
            run_dir / "summary.json",
            ordered_results,
            failures=failures,
            client=client,
            config=config,
        )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
