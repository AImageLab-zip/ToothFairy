#!/usr/bin/env python3
"""Raw-only single-case inference for the frozen Curves predictors."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parent
CODE = CODE_ROOT / "code"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(command: list[str]) -> None:
    import subprocess

    subprocess.run(command, check=True)


def _write_slot_images(photos: dict[str, Image.Image], directory: Path) -> list[Path]:
    slot_views = ("frontal", "right", "left", "lower_occlusal", "upper_occlusal")
    missing = [view for view in slot_views if view not in photos]
    if missing:
        raise KeyError(f"Missing semantic TIFF views: {missing}")
    paths = []
    for index, view in enumerate(slot_views, 1):
        path = directory / f"intraoral_{index}.bmp"
        photos[view].convert("RGB").save(path, format="BMP")
        paths.append(path)
    return paths


def _spee_preprocess(frame: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    columns = checkpoint["feature_columns"]
    preprocessing = checkpoint["preprocessing"]
    x = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    medians = np.asarray(preprocessing["imputer_statistics"], dtype=np.float32)
    missing = ~np.isfinite(x)
    x = np.where(missing, medians[None, :], x)
    indicator_columns = np.asarray(preprocessing["imputer_indicator_features"], dtype=np.int64)
    if len(indicator_columns):
        x = np.concatenate((x, missing[:, indicator_columns].astype(np.float32)), axis=1)
    mean = np.asarray(preprocessing["scaler_mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["scaler_scale"], dtype=np.float32)
    if x.shape[1] != len(mean) or len(mean) != len(scale):
        raise RuntimeError("Frozen Spee preprocessing dimensionality mismatch")
    return (x - mean[None, :]) / scale[None, :]

def _embedding_encoder(mod04, device: torch.device, assets: Path):
    from torchvision.models import resnet18

    state = torch.load(assets / "models" / "shared" / "resnet18_imagenet_default_state_dict.pt", map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(state["state_dict"], strict=True)
    return torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()


def _encode_paths(mod04, encoder, paths: list[Path], device: torch.device) -> dict[str, np.ndarray]:
    out = {}
    with torch.inference_mode():
        for path in paths:
            value = encoder(mod04.letterbox_tensor(str(path)).unsqueeze(0).to(device)).flatten().cpu().numpy().astype(np.float32)
            out[str(path)] = value
    return out


def _majority(labels: list[str]) -> str:
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def predict(
    upper_stl: Path,
    lower_stl: Path,
    upper_fdi: np.ndarray,
    lower_fdi: np.ndarray,
    photos: dict[str, Image.Image],
    case_id: str,
    *,
    asset_root: Path | str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Predict all frozen Curves fields from raw STLs, original FDI, and semantic TIFF views."""
    curves_started = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - curves_started

    print(
        f"B2T_CURVES_START requested_device={device} elapsed_total={_elapsed():.6f}",
        flush=True,
    )

    setup_started = time.perf_counter()
    assets = Path(asset_root).expanduser().resolve()
    schema = json.loads((assets / "config" / "runtime_feature_schema.json").read_text())
    mod02 = _module("curves_eval02_runtime", CODE / "02_evaluate_curve_geometry.py")
    mod04 = _module("curves_multimodal_runtime", CODE / "04_evaluate_curve_multimodal_v3.py")
    mod05 = _module("curves_spee_runtime", CODE / "05_finetune_curve_targeted.py")
    mod06 = _module("curves_wilson_runtime", CODE / "06_evaluate_wilson_hierarchical_v2.py")
    target = torch.device(device)

    def _sync() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    print(
        f"B2T_CURVES_STAGE_DONE runtime_setup "
        f"seconds={time.perf_counter() - setup_started:.6f} "
        f"device={target} elapsed_total={_elapsed():.6f}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="curves_live_") as temp_text:
        workspace_started = time.perf_counter()
        temp = Path(temp_text)
        raw_root, label_root, geometry_out, render_out = (temp / "raw", temp / "labels", temp / "geometry", temp / "renders")
        ios = raw_root / case_id / "ios"
        ios.mkdir(parents=True)
        label_root.mkdir()
        shutil.copy2(upper_stl, ios / "ios_upper.stl")
        shutil.copy2(lower_stl, ios / "ios_lower.stl")
        (label_root / f"{case_id}_ios_upper.json").write_text(json.dumps(np.asarray(upper_fdi).tolist()))
        (label_root / f"{case_id}_ios_lower.json").write_text(json.dumps(np.asarray(lower_fdi).tolist()))
        print(
            f"B2T_CURVES_STAGE_DONE workspace_setup "
            f"seconds={time.perf_counter() - workspace_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        stage_started = time.perf_counter()
        print(
            f"B2T_CURVES_STAGE_START geometry elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        _run([
            sys.executable, str(CODE / "01_extract_curve_geometry.py"),
            "--raw-root", str(raw_root), "--label-root", str(label_root),
            "--geometry-helper", str(CODE / "transverse_01_extract_crossbite_geometry.py"),
            "--output-dir", str(geometry_out), "--patient-id", case_id,
        ])

        print(
            f"B2T_CURVES_STAGE_DONE geometry "
            f"seconds={time.perf_counter() - stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        stage_started = time.perf_counter()
        print(
            f"B2T_CURVES_STAGE_START prepare_renders elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        _run([
            sys.executable, str(CODE / "03_prepare_curve_multimodal_inputs_v3.py"),
            "--raw-root", str(raw_root), "--label-root", str(label_root),
            "--curve-script", str(CODE / "01_extract_curve_geometry.py"),
            "--geometry-helper", str(CODE / "transverse_01_extract_crossbite_geometry.py"),
            "--output-dir", str(render_out), "--patient-id", case_id,
        ])

        print(
            f"B2T_CURVES_STAGE_DONE prepare_renders "
            f"seconds={time.perf_counter() - stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        stage_started = time.perf_counter()
        print(
            f"B2T_CURVES_STAGE_START feature_prepare elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        geometry = pd.read_csv(geometry_out / "curve_geometry_features_v1.csv").reindex(columns=["patient_id", *schema["raw_geometry_columns"]])
        # The batch evaluator selects numeric columns cohort-wide. Reproduce its
        # derived formulas, then reindex to the checkpoint's frozen 101D schema.
        spee_derived, _ = mod02.add_spee_relative_features(geometry)
        spee_source = pd.concat((geometry.reset_index(drop=True), spee_derived.reset_index(drop=True)), axis=1)
        spee_3d = spee_source.reindex(columns=schema["spee_geometry_columns"])
        wilson_3d = geometry.reindex(columns=schema["wilson_geometry_columns"])
        pair_3d = mod06.get_pair_focused_geometry(geometry).reindex(columns=schema["wilson_pair_columns"])
        print(
            f"B2T_CURVES_STAGE_DONE feature_prepare "
            f"seconds={time.perf_counter() - stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        image_io_started = time.perf_counter()
        photo_dir = temp / "semantic_slots"
        photo_dir.mkdir()
        photo_paths = _write_slot_images(photos, photo_dir)
        render_dir = render_out / "renders" / case_id
        render_paths = [render_dir / name for name in ("spee_left.png", "spee_right.png", "wilson_frontal.png", "wilson_oblique.png")]
        if not all(path.is_file() for path in render_paths):
            raise FileNotFoundError(f"Frozen Curves renderer missed outputs for {case_id}")
        print(
            f"B2T_CURVES_STAGE_DONE semantic_image_io "
            f"seconds={time.perf_counter() - image_io_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        # Spee retains its five independently trained CNN folds and fold-local preprocessors.
        spee_probs = []
        spee_class_names: list[str] | None = None

        spee_input_started = time.perf_counter()
        photo_a, photo_b = (
            mod04.letterbox_tensor(str(photo_paths[1])).unsqueeze(0).to(target),
            mod04.letterbox_tensor(str(photo_paths[2])).unsqueeze(0).to(target),
        )
        spee_load_seconds = 0.0
        render_a, render_b = (
            mod04.letterbox_tensor(str(render_paths[0])).unsqueeze(0).to(target),
            mod04.letterbox_tensor(str(render_paths[1])).unsqueeze(0).to(target),
        )
        _sync()
        print(
            f"B2T_CURVES_STAGE_DONE spee_input_tensors "
            f"seconds={time.perf_counter() - spee_input_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        spee_stage_started = time.perf_counter()
        print(
            f"B2T_CURVES_STAGE_START spee_5fold elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        for fold in range(5):
            fold_started = time.perf_counter()

            # ----------------------------------------------------------
            # checkpoint load
            # ----------------------------------------------------------
            load_started = time.perf_counter()

            checkpoint = torch.load(
                assets / "models" / "spee" / f"spee_fold{fold}.pt",
                map_location="cpu",
                weights_only=False,
            )

            checkpoint_load_seconds = time.perf_counter() - load_started
            spee_load_seconds += checkpoint_load_seconds

            print(
                f"B2T_CURVES_SPEE fold={fold} checkpoint_loaded "
                f"seconds={checkpoint_load_seconds:.6f} "
                f"elapsed_total={_elapsed():.6f}",
                flush=True,
            )

            if spee_class_names is None:
                spee_class_names = list(checkpoint["class_names"])

            # ----------------------------------------------------------
            # preprocessing + model construction + state_dict + GPU move
            # ----------------------------------------------------------
            model_started = time.perf_counter()

            x3d = torch.from_numpy(
                _spee_preprocess(spee_3d, checkpoint)
            ).to(target)

            model = mod05.build_model(
                "spee",
                x3d.shape[1],
                len(checkpoint["class_names"]),
                checkpoint["training_config"]["dropout"],
            )

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            model.to(target).eval()
            _sync()

            model_prepare_seconds = time.perf_counter() - model_started

            print(
                f"B2T_CURVES_SPEE fold={fold} model_ready "
                f"seconds={model_prepare_seconds:.6f} "
                f"elapsed_total={_elapsed():.6f}",
                flush=True,
            )

            # ----------------------------------------------------------
            # inference
            # ----------------------------------------------------------
            _sync()
            infer_started = time.perf_counter()

            with torch.inference_mode():
                fold_prob = torch.softmax(
                    model(
                        photo_a,
                        photo_b,
                        render_a,
                        render_b,
                        x3d,
                    ),
                    dim=1,
                ).cpu().numpy()[0]

            _sync()
            infer_seconds = time.perf_counter() - infer_started

            spee_probs.append(fold_prob)

            print(
                f"B2T_CURVES_SPEE fold={fold} inference_done "
                f"seconds={infer_seconds:.6f} "
                f"fold_total={time.perf_counter() - fold_started:.6f} "
                f"elapsed_total={_elapsed():.6f}",
                flush=True,
            )

            del model

        print(
            f"B2T_CURVES_STAGE_DONE spee_5fold "
            f"seconds={time.perf_counter() - spee_stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        mean_spee = np.mean(np.vstack(spee_probs), axis=0)
        if spee_class_names is None:
            raise RuntimeError("Frozen Spee ensemble has no folds")
        curve_spee = spee_class_names[int(mean_spee.argmax())]
        print(f"B2T_MODEL_LOAD curves_spee count=5 seconds={spee_load_seconds:.6f}", flush=True)

        # Wilson uses exactly its five frozen outer-fold pipelines and thresholds.
        print(
            f"B2T_CURVES_STAGE_START shared_resnet elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        encoder_started = time.perf_counter()

        encoder = _embedding_encoder(
            mod04,
            target,
            assets,
        )

        _sync()
        encoder_load_seconds = time.perf_counter() - encoder_started

        print(
            f"B2T_CURVES_RESNET model_loaded "
            f"seconds={encoder_load_seconds:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        embedding_started = time.perf_counter()
        _sync()

        embedding_map = _encode_paths(
            mod04,
            encoder,
            [*photo_paths, *render_paths[2:]],
            target,
        )

        _sync()

        print(
            f"B2T_CURVES_RESNET embeddings_done "
            f"seconds={time.perf_counter() - embedding_started:.6f} "
            f"count={len(photo_paths) + len(render_paths[2:])} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        print(
            f"B2T_CURVES_STAGE_DONE shared_resnet "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        print(
            f"B2T_MODEL_LOAD curves_shared_resnet count=1 "
            f"seconds={encoder_load_seconds:.6f}",
            flush=True,
        )
        wilson_feature_started = time.perf_counter()
        wilson_frame = pd.DataFrame({"patient_id": [case_id]})
        for index, path in enumerate(photo_paths, 1):
            wilson_frame[f"strict_photo_{index}_path"] = str(path)
        wilson_frame["wilson_frontal_render_path"] = str(render_paths[2])
        wilson_frame["wilson_oblique_render_path"] = str(render_paths[3])
        quality, _ = mod06.build_quality_matrix(wilson_frame)
        photo_concat, photo_stats = mod06.make_photo_embedding_features(wilson_frame, embedding_map, mod04)
        render_embeddings = mod06.make_render_embedding_features(wilson_frame, embedding_map, mod04)
        gate_features, visual_features = mod06.build_feature_families(
            wilson_3d.to_numpy(np.float32),
            pair_3d.to_numpy(np.float32),
            quality,
            photo_concat,
            photo_stats,
            render_embeddings,
        )
        gate_specs, visual_specs = mod06.candidate_specs(gate_features, visual_features)
        print(
            f"B2T_CURVES_STAGE_DONE wilson_feature_prepare "
            f"seconds={time.perf_counter() - wilson_feature_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )
        wilson_labels = []
        wilson_load_seconds = 0.0
        wilson_stage_started = time.perf_counter()

        print(
            f"B2T_CURVES_STAGE_START wilson_5fold "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        for fold in range(5):
            fold_started = time.perf_counter()

            load_started = time.perf_counter()

            artifact = joblib.load(
                assets
                / "models"
                / "wilson"
                / f"wilson_hierarchical_fold{fold}.joblib"
            )

            fold_load_seconds = time.perf_counter() - load_started
            wilson_load_seconds += fold_load_seconds

            print(
                f"B2T_CURVES_WILSON fold={fold} model_loaded "
                f"seconds={fold_load_seconds:.6f} "
                f"elapsed_total={_elapsed():.6f}",
                flush=True,
            )

            predict_started = time.perf_counter()

            prediction, _, _ = mod06.predict_full_outer_models(
                indices=np.asarray([0]),
                gate_model=artifact["gate_model_object"],
                visual_model=artifact["visual_model_object"],
                gate_features=gate_features,
                visual_features=visual_features,
                gate_specs=gate_specs,
                visual_specs=visual_specs,
                selected=artifact["selection"],
            )

            predict_seconds = time.perf_counter() - predict_started
            wilson_labels.append(str(prediction[0]))

            print(
                f"B2T_CURVES_WILSON fold={fold} prediction_done "
                f"seconds={predict_seconds:.6f} "
                f"fold_total={time.perf_counter() - fold_started:.6f} "
                f"elapsed_total={_elapsed():.6f}",
                flush=True,
            )

        print(
            f"B2T_CURVES_STAGE_DONE wilson_5fold "
            f"seconds={time.perf_counter() - wilson_stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        print(
            f"B2T_MODEL_LOAD curves_wilson count=5 "
            f"seconds={wilson_load_seconds:.6f}",
            flush=True,
        )
        curve_wilson = _majority(wilson_labels)

        location_stage_started = time.perf_counter()
        print(
            f"B2T_CURVES_STAGE_START wilson_location "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        location_load_started = time.perf_counter()
        location = joblib.load(
            assets / "models" / "wilson_location" / "wilson_location_fulltrain.joblib"
        )
        location_load_seconds = time.perf_counter() - location_load_started
        print(
            f"B2T_MODEL_LOAD curves_wilson_location count=1 "
            f"seconds={location_load_seconds:.6f}",
            flush=True,
        )

        location_predict_started = time.perf_counter()
        curve_wilson_location = str(location["model_object"].predict(wilson_3d)[0])
        location_predict_seconds = time.perf_counter() - location_predict_started

        print(
            f"B2T_CURVES_STAGE_DONE wilson_location "
            f"predict_seconds={location_predict_seconds:.6f} "
            f"stage_seconds={time.perf_counter() - location_stage_started:.6f} "
            f"elapsed_total={_elapsed():.6f}",
            flush=True,
        )

        print(
            f"B2T_CURVES_DONE total_seconds={_elapsed():.6f}",
            flush=True,
        )

        return {
            "curve_spee": curve_spee,
            "curve_wilson": curve_wilson,
            "curve_wilson_location": curve_wilson_location,
            "_debug": {
                "spee_geometry_columns": list(spee_3d.columns),
                "wilson_geometry_columns": list(wilson_3d.columns),
                "wilson_pair_columns": list(pair_3d.columns),
                "wilson_fold_labels": wilson_labels,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper-stl", required=True, type=Path)
    parser.add_argument("--lower-stl", required=True, type=Path)
    parser.add_argument("--upper-fdi-json", required=True, type=Path)
    parser.add_argument("--lower-fdi-json", required=True, type=Path)
    parser.add_argument("--photo-tiff", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from PIL import Image
    import sys as _sys
    _sys.path.insert(0, str(Path("/opt/app")))
    from bite2text.tiff import load_five_view_tiff
    result = predict(args.upper_stl, args.lower_stl, np.asarray(json.loads(args.upper_fdi_json.read_text())), np.asarray(json.loads(args.lower_fdi_json.read_text())), load_five_view_tiff(args.photo_tiff), args.case_id, device=args.device)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
