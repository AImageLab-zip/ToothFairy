from __future__ import annotations

import json
import os
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

TRUTHY = {"1", "true", "yes", "on"}
METRICS = {}


def main() -> int:
    jobs = read_predictions()
    references = read_references()
    cases = []

    for job in jobs:
        if job.get("status") and job["status"] != "Succeeded":
            raise RuntimeError(f"Prediction job {job.get('pk')} has status {job['status']!r}")

        case_id = get_case_id(job)
        if case_id not in references:
            raise RuntimeError(f"Missing English reference report for case {case_id!r}")
        cases.append((case_id, read_report(job), references[case_id]))

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


def read_predictions() -> list[dict[str, Any]]:
    jobs = json.loads(PREDICTIONS_JSON.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"{PREDICTIONS_JSON} must contain a non-empty list")
    return jobs


def get_case_id(job: dict[str, Any]) -> str:
    job_pk = str(job.get("pk") or "").strip().lower()
    if job_pk.startswith("job-") and len(job_pk) > 4:
        return normalize_id(job_pk[4:])

    for value in job.get("inputs") or []:
        if value["socket"]["slug"] != CASE_INPUT_SLUG:
            continue
        image = value.get("image") or {}
        if image.get("name"):
            return normalize_id(image["name"])
    raise RuntimeError(f"No input socket {CASE_INPUT_SLUG!r} found for job {job.get('pk')}")


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
    result = load_metric("bleu").compute(
        predictions=predictions,
        references=[[reference] for reference in references],
        max_order=4,
    )
    return float(result["bleu"])


def meteor(predictions: list[str], references: list[str]) -> float:
    return float(load_metric("meteor").compute(predictions=predictions, references=references)["meteor"])


def load_metric(name: str):
    if name in METRICS:
        return METRICS[name]

    app_dir = Path(__file__).resolve().parent
    old_path = sys.path[:]
    shadow = sys.modules.get("evaluate")
    shadow_file = getattr(shadow, "__file__", None)
    if shadow_file is not None and Path(shadow_file).resolve() == Path(__file__).resolve():
        del sys.modules["evaluate"]

    try:
        sys.path = [path for path in sys.path if Path(path or os.getcwd()).resolve() != app_dir]
        if name == "meteor":
            import nltk

            nltk.download = lambda *_args, **_kwargs: True
        METRICS[name] = importlib.import_module("evaluate").load(name)
        return METRICS[name]
    finally:
        sys.path = old_path


def normalize_id(value: Any) -> str:
    return Path(str(value).strip()).stem.lower()


def enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


if __name__ == "__main__":
    raise SystemExit(main())
