#!/usr/bin/env python3
"""Grand Challenge inference entrypoint for the Bite2Text IOS ensemble.

The official Bite2Text container contract mounts one case at ``/input`` and
expects ``/output/diagnostic-imaging-report.json`` with exactly one English
``report`` field.  This implementation intentionally uses only the two IOS
STL sockets; the required photograph socket is accepted but is not yet used.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from train import PairedPointNet, load_stl_vertices, normalize_pair, sample_points


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
DEFAULT_MODEL_PATH = Path("/opt/ml/model")
DEFAULT_NUM_POINTS = 4096
DEFAULT_TTA_SAMPLES = 4
# This is the first deterministic validation sampling stream used by the
# ensemble evaluator (20260807 + 300000).  It can be overridden per run.
DEFAULT_SEED = 20_560_807


RELATION_TEXT = {
    "class_i": "Class I",
    "class_ii_edge_to_edge": "edge-to-edge Class II",
    "class_ii_full": "full Class II",
    "class_ii_unspecified": "Class II",
    "class_iii": "Class III",
}
OVERJET_TEXT = {
    "normal": "within normal limits",
    "increased": "increased",
    "reduced": "reduced",
    "negative": "negative",
    "edge_to_edge": "edge-to-edge",
}
VERTICAL_SENTENCES = {
    "normal": "From a vertical standpoint, the overbite is within normal limits.",
    "increased": "From a vertical standpoint, the overbite is increased.",
    "reduced": "From a vertical standpoint, the overbite is reduced.",
    "deep_bite": "From a vertical standpoint, there is a deep bite.",
    "open_bite": "From a vertical standpoint, there is an open bite.",
}
MIDLINE_SENTENCES = {
    "coincident": "The dental midlines are coincident.",
    "slightly_deviated": "The dental midlines are slightly deviated.",
    "deviated": "The dental midlines are deviated.",
}


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def input_files() -> tuple[Path, Path]:
    """Resolve the official socket paths, retaining compatibility fallbacks."""
    input_manifest = INPUT_PATH / "inputs.json"
    if input_manifest.is_file():
        payload = read_json(input_manifest)
        if not isinstance(payload, list):
            raise RuntimeError(f"{input_manifest} must contain a JSON list")
        slugs = {str((item.get("socket") or {}).get("slug") or "") for item in payload if isinstance(item, dict)}
        required = {"3d-lower-teeth-scan", "3d-upper-teeth-scan"}
        if not required.issubset(slugs):
            raise RuntimeError(f"Missing required IOS socket(s); found {sorted(slugs)}")

    lower = find_single_file(
        preferred=(INPUT_PATH / "3d-lower-teeth-scan.obj", INPUT_PATH / "3d-lower-teeth-scan.stl"),
        locations=(INPUT_PATH / "files" / "ios-lower", INPUT_PATH / "ios-lower"),
    )
    upper = find_single_file(
        preferred=(INPUT_PATH / "3d-upper-teeth-scan.obj", INPUT_PATH / "3d-upper-teeth-scan.stl"),
        locations=(INPUT_PATH / "files" / "ios-upper", INPUT_PATH / "ios-upper"),
    )
    return upper, lower


def find_single_file(*, preferred: Iterable[Path], locations: Iterable[Path]) -> Path:
    for path in preferred:
        if path.is_file():
            return path
    matches: list[Path] = []
    for location in locations:
        for pattern in ("*.stl", "*.STL", "*.obj", "*.OBJ"):
            matches.extend(Path(path) for path in glob.glob(str(location / pattern)))
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        checked = [str(path) for path in locations]
        raise RuntimeError(f"No IOS mesh found; checked {checked}")
    if len(unique) > 1:
        print(f"Multiple IOS meshes found; using {unique[0].name}", flush=True)
    return unique[0]


def checkpoint_paths() -> list[Path]:
    """Obtain checkpoint paths from an explicit env var or the model bundle."""
    configured = os.getenv("BITE2TEXT_CHECKPOINTS", "").strip()
    if configured:
        paths = [Path(value).expanduser() for value in configured.split(":") if value]
    else:
        paths = sorted(DEFAULT_MODEL_PATH.glob("checkpoints/*.pt"))
        if not paths:
            paths = sorted(DEFAULT_MODEL_PATH.glob("*.pt"))
    if not paths:
        raise RuntimeError(
            "No checkpoints found. Set BITE2TEXT_CHECKPOINTS to a colon-separated "
            "list, or package *.pt files below /opt/ml/model/checkpoints/."
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing checkpoint(s): {missing}")
    return paths


def optional_extension_checkpoint_paths() -> list[Path]:
    """Return a separately trained report-extension ensemble when configured."""
    configured = os.getenv("BITE2TEXT_EXTENSION_CHECKPOINTS", "").strip()
    if not configured:
        return []
    paths = [Path(value).expanduser() for value in configured.split(":") if value]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing extension checkpoint(s): {missing}")
    return paths


def load_models(paths: list[Path], device: torch.device) -> tuple[list[PairedPointNet], dict[str, list[str]]]:
    models: list[PairedPointNet] = []
    head_vocabs: dict[str, list[str]] | None = None
    expected_config: dict[str, object] | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        vocabs = checkpoint.get("head_vocabs")
        if not isinstance(vocabs, dict):
            raise RuntimeError(f"Checkpoint has no valid head vocabularies: {path}")
        config = checkpoint.get("config") or {}
        model_config = {
            "geometry_features": bool(config.get("geometry_features", False)),
            "contact_points": int(config.get("contact_points", 256)),
            "direct_geometry_heads": bool(config.get("direct_geometry_heads", False)),
            "direct_geometry_exclude_heads": str(config.get("direct_geometry_exclude_heads", "")),
        }
        if head_vocabs is None:
            head_vocabs = vocabs
            expected_config = model_config
        elif vocabs != head_vocabs or model_config != expected_config:
            raise RuntimeError(f"Incompatible ensemble checkpoint: {path}")
        model = PairedPointNet(head_vocabs, **model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    if head_vocabs is None:
        raise RuntimeError("No ensemble model loaded")
    return models, head_vocabs


def ensemble_probabilities(
    models: list[PairedPointNet],
    upper_vertices: np.ndarray,
    lower_vertices: np.ndarray,
    num_points: int,
    tta_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if num_points < 2 or tta_samples < 1:
        raise ValueError("num_points must be at least two and tta_samples at least one")
    summed: dict[str, torch.Tensor] | None = None
    with torch.inference_mode():
        for view in range(tta_samples):
            rng = np.random.default_rng(seed + view * 1_000_003)
            upper = sample_points(upper_vertices, num_points, rng)
            lower = sample_points(lower_vertices, num_points, rng)
            upper, lower = normalize_pair(upper, lower)
            upper_tensor = torch.from_numpy(upper).unsqueeze(0).to(device)
            lower_tensor = torch.from_numpy(lower).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits_by_model = [model(upper_tensor, lower_tensor) for model in models]
            averaged = {
                head: torch.stack([logits[head].float().softmax(dim=1) for logits in logits_by_model]).mean(dim=0)
                for head in logits_by_model[0]
            }
            if summed is None:
                summed = {head: values for head, values in averaged.items()}
            else:
                for head, values in averaged.items():
                    summed[head] += values
    if summed is None:
        raise RuntimeError("No inference views executed")
    return {head: values / tta_samples for head, values in summed.items()}


def labels_from_probabilities(probabilities: dict[str, torch.Tensor], head_vocabs: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, float]]:
    labels: dict[str, str] = {}
    confidences: dict[str, float] = {}
    for head, values in probabilities.items():
        index = int(values.argmax(dim=1).item())
        labels[head] = head_vocabs[head][index]
        confidences[head] = float(values[0, index].item())
    return labels, confidences


def render_report(labels: dict[str, str]) -> str:
    """Render only supported findings, using report language seen in training."""
    right = (
        f"{RELATION_TEXT[labels['right_molar_relation']]} molar relationship and "
        f"{RELATION_TEXT[labels['right_canine_relation']]} canine relationship on the right"
    )
    left = (
        f"{RELATION_TEXT[labels['left_molar_relation']]} molar relationship and "
        f"{RELATION_TEXT[labels['left_canine_relation']]} canine relationship on the left"
    )
    extension_sentences: list[str] = []
    if "crossbite" in labels:
        extension_sentences.append(
            {
                "none": "No crossbite is present.",
                "anterior": "An anterior crossbite is present.",
                "posterior": "A posterior crossbite is present.",
                "present_unspecified": "A crossbite is present.",
            }[labels["crossbite"]]
        )
    crowding_heads = ("upper_crowding", "lower_crowding")
    if all(head in labels for head in crowding_heads):
        upper_crowding = labels["upper_crowding"].replace("-", " ")
        lower_crowding = labels["lower_crowding"].replace("-", " ")
        if upper_crowding == lower_crowding:
            extension_sentences.append(f"There is {upper_crowding} crowding in both arches.")
        else:
            extension_sentences.append(
                f"There is {upper_crowding} crowding in the upper arch and {lower_crowding} crowding in the lower arch."
            )
    curve_heads = ("curve_spee", "curve_wilson")
    if all(head in labels for head in curve_heads):
        spee = "within normal limits" if labels["curve_spee"] == "normal" else "increased"
        wilson = "within normal limits" if labels["curve_wilson"] == "normal" else "increased"
        if spee == wilson:
            extension_sentences.append(f"The Curve of Spee and the Curve of Wilson are {spee}.")
        else:
            extension_sentences.append(f"The Curve of Spee is {spee}, while the Curve of Wilson is {wilson}.")
    return " ".join(
        extension_sentences
        + [
            f"From a sagittal standpoint, there is a {right}, and a {left}.",
            f"The overjet is {OVERJET_TEXT[labels['overjet']]}.",
            VERTICAL_SENTENCES[labels["vertical_relation"]],
            MIDLINE_SENTENCES[labels["midline_relation"]],
        ]
    )


def run() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_points = int(os.getenv("BITE2TEXT_NUM_POINTS", str(DEFAULT_NUM_POINTS)))
    tta_samples = int(os.getenv("BITE2TEXT_TTA_SAMPLES", str(DEFAULT_TTA_SAMPLES)))
    seed = int(os.getenv("BITE2TEXT_SEED", str(DEFAULT_SEED)))
    upper_path, lower_path = input_files()
    models, head_vocabs = load_models(checkpoint_paths(), device)
    upper_vertices = load_stl_vertices(upper_path)
    lower_vertices = load_stl_vertices(lower_path)
    probabilities = ensemble_probabilities(
        models=models,
        upper_vertices=upper_vertices,
        lower_vertices=lower_vertices,
        num_points=num_points,
        tta_samples=tta_samples,
        seed=seed,
        device=device,
    )
    labels, confidences = labels_from_probabilities(probabilities, head_vocabs)
    extension_paths = optional_extension_checkpoint_paths()
    if extension_paths:
        extension_models, extension_vocabs = load_models(extension_paths, device)
        extension_probabilities = ensemble_probabilities(
            models=extension_models,
            upper_vertices=upper_vertices,
            lower_vertices=lower_vertices,
            num_points=num_points,
            tta_samples=tta_samples,
            seed=seed,
            device=device,
        )
        extension_labels, extension_confidences = labels_from_probabilities(extension_probabilities, extension_vocabs)
        overlap = sorted(set(labels).intersection(extension_labels))
        if overlap:
            raise RuntimeError(f"Main and extension head vocabularies overlap: {overlap}")
        labels.update(extension_labels)
        confidences.update(extension_confidences)
    report = render_report(labels)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "diagnostic-imaging-report.json").write_text(
        json.dumps({"report": report}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "models": len(models),
                "extension_models": len(extension_paths),
                "num_points": num_points,
                "tta_samples": tta_samples,
                "labels": labels,
                "confidence": {head: round(value, 4) for head, value in confidences.items()},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
