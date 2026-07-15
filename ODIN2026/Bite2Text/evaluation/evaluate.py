from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
import importlib
import sys


INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
GROUND_TRUTH_DIR = Path("/opt/ml/input/data/ground_truth")

PREDICTIONS_JSON = INPUT_DIR / "predictions.json"
REPORT_OUTPUT_SLUG = "diagnostic-imaging-report"
REPORT_KEY = "report"
CASE_INPUT_SLUG = "intraoral-photo"
CASE_INPUT_SLUGS = {"intraoral-photo", "2d-intraoral-photographs"}
GENERIC_FILENAMES = {
    "ios_lower",
    "ios_upper",
    "3d-lower-teeth-scan",
    "3d-upper-teeth-scan",
    "intraoral-photo",
    "2d-intraoral-photographs",
}

TRUTHY = {"1", "true", "yes", "on"}
METRICS = {}
UNAVAILABLE_METRICS: dict[str, str] = {}
FALLBACK_LOGGED: set[str] = set()
WORDNET_AVAILABLE: bool | None = None


def main() -> int:
    jobs = read_predictions()
    references = read_references()
    pending_cases = []

    for job in jobs:
        if job.get("status") and job["status"] != "Succeeded":
            raise RuntimeError(f"Prediction job {job.get('pk')} has status {job['status']!r}")

        case_id = get_case_id(job)
        pending_cases.append((case_id, read_report(job)))

    cases = resolve_cases(pending_cases, references)

    metrics = offline_metrics(cases)
    online = online_metrics(cases)
    if online.get("enabled") and "error" not in online:
        merge_online_metrics(metrics, online)
    
    keys_to_filter = {"aggregates", "results"}
    metrics["online_metrics"] = {k: v for k, v in online.items() if k not in keys_to_filter}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"Evaluated {len(cases)} cases")
    return 0


def resolve_cases(
    pending_cases: list[tuple[str, str]], references: dict[str, str]
) -> list[tuple[str, str, str]]:
    resolved_cases: list[tuple[str, str, str]] = []
    unresolved_predictions: list[str] = []
    unresolved_case_ids: list[str] = []
    used_reference_ids: set[str] = set()

    for case_id, prediction in pending_cases:
        if case_id in references and case_id not in used_reference_ids:
            resolved_cases.append((case_id, prediction, references[case_id]))
            used_reference_ids.add(case_id)
            continue

        unresolved_case_ids.append(case_id)
        unresolved_predictions.append(prediction)

    remaining_reference_ids = [
        reference_id for reference_id in sorted(references) if reference_id not in used_reference_ids
    ]

    if unresolved_case_ids and len(unresolved_case_ids) != len(remaining_reference_ids):
        raise RuntimeError(
            "Could not match prediction jobs to reference reports. "
            f"Unresolved case ids: {unresolved_case_ids}; "
            f"remaining references: {remaining_reference_ids}"
        )

    for fallback_case_id, prediction, reference_id in zip(
        unresolved_case_ids,
        unresolved_predictions,
        remaining_reference_ids,
    ):
        print(
            "Falling back to reference-order case matching for "
            f"job case id {fallback_case_id!r} -> reference {reference_id!r}",
            flush=True,
        )
        resolved_cases.append((reference_id, prediction, references[reference_id]))

    return resolved_cases


def read_predictions() -> list[dict[str, Any]]:
    jobs = json.loads(PREDICTIONS_JSON.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"{PREDICTIONS_JSON} must contain a non-empty list")
    return jobs


def get_case_id(job: dict[str, Any]) -> str:
    # Keep local test compatibility where job ids are formatted as job-<case-id>.
    job_pk = str(job.get("pk") or "").strip().lower()
    if job_pk.startswith("job-") and len(job_pk) > 4:
        return normalize_id(job_pk[4:])

    # Prefer photo-based identifiers when present.
    preferred_candidates: list[str] = []
    fallback_candidates: list[str] = []

    for value in job.get("inputs") or []:
        socket = value.get("socket") or {}
        slug = str(socket.get("slug") or "").strip().lower()

        image = value.get("image") or {}
        file = value.get("file") or {}

        if isinstance(image, str):
            image = {"name": image}
        if isinstance(file, str):
            file = {"name": file}

        if image.get("name"):
            candidate = normalize_id(image["name"])
            if slug in CASE_INPUT_SLUGS:
                preferred_candidates.append(candidate)
            else:
                fallback_candidates.append(candidate)

        if file.get("name"):
            fallback_candidates.append(normalize_id(file["name"]))

    for candidate in preferred_candidates + fallback_candidates:
        if candidate and candidate not in GENERIC_FILENAMES:
            return candidate

    # Grand Challenge may only expose UUID-like records; use stable ids as last resort.
    for key in ("pk", "id", "uuid"):
        raw = job.get(key)
        if raw:
            return normalize_id(raw)

    raise RuntimeError(f"Could not determine case id for job {job.get('pk')!r}")


def read_report(job: dict[str, Any]) -> str:
    path = report_path(job)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(REPORT_KEY), str):
        raise RuntimeError(f"{path} must contain {{\"{REPORT_KEY}\": string}}")
    return payload[REPORT_KEY].strip()


def report_path(job: dict[str, Any]) -> Path:
    for value in job.get("outputs") or []:
        if value["socket"]["slug"] == REPORT_OUTPUT_SLUG:
            return INPUT_DIR / job["pk"] / "output" / value["socket"]["relative_path"]
    raise RuntimeError(f"No output socket {REPORT_OUTPUT_SLUG!r} found for job {job.get('pk')}")


def read_references() -> dict[str, str]:
    if not GROUND_TRUTH_DIR.exists():
        raise RuntimeError(f"Missing ground truth directory: {GROUND_TRUTH_DIR}")

    references = {}
    for report in sorted(GROUND_TRUTH_DIR.glob("*.txt")):
        references[normalize_id(report.name)] = report.read_text(encoding="utf-8", errors="replace").strip()

    if not references:
        raise RuntimeError(f"No references found at {GROUND_TRUTH_DIR}/*.txt")
    return references


def offline_metrics(cases: list[tuple[str, str, str]]) -> dict[str, Any]:
    rows = []
    references = []
    predictions = []

    for case_id, prediction, reference in cases:
        references.append(reference)
        predictions.append(prediction)
        rows.append(
            {
                "case_id": case_id,
                "bleu_4": bleu_4([prediction], [reference]),
                "meteor": meteor([prediction], [reference]),
            }
        )

    return {
        "results": rows,
        "aggregates": {
            "bleu_4": bleu_4(predictions, references),
            "meteor": meteor(predictions, references),
            "num_cases": len(rows),
        },
    }


def online_metrics(cases: list[tuple[str, str, str]]) -> dict[str, Any]:
    if enabled("RUNNING_ON_GRAND_CHALLENGE"):
        return {"enabled": False, "reason": "disabled_on_grand_challenge"}
    if not enabled("ENABLE_ONLINE_METRICS"):
        return {"enabled": False, "reason": "not_enabled"}

    try:
        print("Computing RadFact online metrics with OpenAI; this can take several minutes.", flush=True)
        from radfact_lite import ModelConfig, PipelineModels, RadFactLitePipeline, ReportType

        model = ModelConfig(
            model=os.getenv("RADFACT_MODEL", "gpt-4o-mini"),
            timeout=float(os.getenv("RADFACT_TIMEOUT", "30")),
            max_retries=int(os.getenv("RADFACT_MAX_RETRIES", "0")),
        )
        pipeline = RadFactLitePipeline(
            models=PipelineModels(parse_model=model, entailment_model=model, filtering_model=model),
            report_type=ReportType.TOOTHFAIRY,
        )
        aggregate, per_case = pipeline.compute_radfact(
            candidates_by_id={case_id: prediction for case_id, prediction, _ in cases},
            references_by_id={case_id: reference for case_id, _, reference in cases},
            is_narrative_text=True,
            filter_negatives=False,
        )
        return {
            "enabled": True,
            "aggregates": asdict(aggregate),
            "results": [asdict(result) for result in per_case],
        }
    except Exception as error:
        return {"enabled": True, "error": str(error)}


def merge_online_metrics(metrics: dict[str, Any], online: dict[str, Any]) -> None:
    radfact_by_id = {result["sample_id"]: result for result in online["results"]}
    for row in metrics["results"]:
        radfact = radfact_by_id[row["case_id"]]
        row.update(
            {
                "logical_precision": radfact["logical_precision"],
                "logical_recall": radfact["logical_recall"],
                "logical_f1": radfact["logical_f1"],
                "entailed_candidate_count": radfact["entailed_candidate_count"],
                "entailed_reference_count": radfact["entailed_reference_count"],
                "candidate_count": radfact["candidate_count"],
                "reference_count": radfact["reference_count"],
            }
        )

    metrics["aggregates"].update(
        {
            "logical_precision": online["aggregates"]["logical_precision"],
            "logical_recall": online["aggregates"]["logical_recall"],
            "logical_f1": online["aggregates"]["logical_f1"],
            "num_samples": online["aggregates"]["num_samples"],
            "num_llm_failures": online["aggregates"]["num_llm_failures"],
        }
    )


def bleu_4(predictions: list[str], references: list[str]) -> float:
    if use_local_metrics():
        return bleu_4_local(predictions, references)

    try:
        result = load_metric("bleu").compute(
            predictions=predictions,
            references=[[reference] for reference in references],
            max_order=4,
        )
        return float(result["bleu"])
    except Exception as error:
        if "bleu" not in FALLBACK_LOGGED:
            print(f"BLEU metric fallback enabled: {error}", flush=True)
            FALLBACK_LOGGED.add("bleu")
        return bleu_4_local(predictions, references)


def meteor(predictions: list[str], references: list[str]) -> float:
    if use_local_metrics():
        return meteor_lite_batch(predictions, references)

    try:
        return float(load_metric("meteor").compute(predictions=predictions, references=references)["meteor"])
    except Exception as error:
        if "meteor" not in FALLBACK_LOGGED:
            print(f"METEOR metric fallback enabled: {error}", flush=True)
            FALLBACK_LOGGED.add("meteor")
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

        if not predictions:
            return 0.0

        use_wordnet = has_wordnet()
        if not use_wordnet and "meteor_wordnet" not in FALLBACK_LOGGED:
            print("METEOR wordnet fallback enabled: wordnet corpus not available", flush=True)
            FALLBACK_LOGGED.add("meteor_wordnet")

        scores = []
        for prediction, reference in zip(predictions, references):
            prediction_tokens = tokenize(prediction)
            reference_tokens = tokenize(reference)
            if not prediction_tokens and not reference_tokens:
                scores.append(1.0)
                continue
            if not prediction_tokens or not reference_tokens:
                scores.append(0.0)
                continue
            if use_wordnet:
                scores.append(float(nltk_meteor_score([reference_tokens], prediction_tokens)))
            else:
                scores.append(meteor_lite_score(prediction_tokens, reference_tokens))

        return float(sum(scores) / len(scores)) if scores else 0.0


def bleu_4_local(predictions: list[str], references: list[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    predicted_tokens = [tokenize(prediction) for prediction in predictions]
    reference_tokens = [[tokenize(reference)] for reference in references]
    if not predicted_tokens:
        return 0.0

    return float(
        corpus_bleu(
            list_of_references=reference_tokens,
            hypotheses=predicted_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def meteor_lite_batch(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0

    scores = []
    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize(prediction)
        reference_tokens = tokenize(reference)
        if not prediction_tokens and not reference_tokens:
            scores.append(1.0)
            continue
        if not prediction_tokens or not reference_tokens:
            scores.append(0.0)
            continue
        scores.append(meteor_lite_score(prediction_tokens, reference_tokens))
    return float(sum(scores) / len(scores)) if scores else 0.0


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+|[^\w\s]", text.lower()) if token.strip()]


def meteor_lite_score(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    matched_reference_indices = greedy_match_indices(prediction_tokens, reference_tokens)
    matches = len(matched_reference_indices)
    if matches == 0:
        return 0.0

    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    denominator = recall + 9.0 * precision
    if denominator == 0.0:
        return 0.0

    f_mean = (10.0 * precision * recall) / denominator
    chunks = chunk_count(matched_reference_indices)
    penalty = 0.5 * (chunks / matches) ** 3
    return float((1.0 - penalty) * f_mean)


def greedy_match_indices(prediction_tokens: list[str], reference_tokens: list[str]) -> list[int]:
    used = [False] * len(reference_tokens)
    indices: list[int] = []
    for token in prediction_tokens:
        for index, ref_token in enumerate(reference_tokens):
            if used[index] or token != ref_token:
                continue
            used[index] = True
            indices.append(index)
            break
    return indices


def chunk_count(indices: list[int]) -> int:
    if not indices:
        return 0
    chunks = 1
    for current, previous in zip(indices[1:], indices[:-1]):
        if current != previous + 1:
            chunks += 1
    return chunks


def has_wordnet() -> bool:
    global WORDNET_AVAILABLE
    if WORDNET_AVAILABLE is not None:
        return WORDNET_AVAILABLE

    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        WORDNET_AVAILABLE = True
    except LookupError:
        WORDNET_AVAILABLE = False
    return WORDNET_AVAILABLE


def use_local_metrics() -> bool:
    return enabled("RUNNING_ON_GRAND_CHALLENGE") or enabled("FORCE_LOCAL_METRICS")


def load_metric(name: str):
    if name in METRICS:
        return METRICS[name]
    if name in UNAVAILABLE_METRICS:
        raise RuntimeError(UNAVAILABLE_METRICS[name])
    if enabled("RUNNING_ON_GRAND_CHALLENGE"):
        reason = "disabled in Grand Challenge runtime; using local fallback"
        UNAVAILABLE_METRICS[name] = reason
        raise RuntimeError(reason)

    app_dir = Path(__file__).resolve().parent
    old_path = sys.path[:]
    shadow = sys.modules.get("evaluate")
    shadow_file = getattr(shadow, "__file__", None)
    if shadow_file is not None and Path(shadow_file).resolve() == Path(__file__).resolve():
        del sys.modules["evaluate"]

    try:
        sys.path = [path for path in sys.path if Path(path or os.getcwd()).resolve() != app_dir]
        METRICS[name] = importlib.import_module("evaluate").load(name)
        return METRICS[name]
    except Exception as error:
        UNAVAILABLE_METRICS[name] = str(error)
        raise
    finally:
        sys.path = old_path


def normalize_id(value: Any) -> str:
    return Path(str(value).strip()).stem.lower()


def enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


if __name__ == "__main__":
    raise SystemExit(main())
