"""Grand Challenge entrypoint for the final BiteMark Bite2Text pipeline.

The model tarball contains the validated notebook, all checkpoints, 3DTeethLand,
Pointcept, and the minimal YOLO tooth-boundary runtime. For each invocation this
entrypoint stages the three challenge inputs as one temporary Bite2Text case,
runs the notebook pipeline without relying on workspace caches, and writes the
rule-based report to the ``diagnostic-imaging-report`` output socket.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
NOTEBOOK_PATH = MODEL_PATH / "p" / "n.ipynb"
MANIFEST_PATH = MODEL_PATH / "m.json"
TEETHLAND_PYTHON = Path("/opt/teethland-python")
CASE_ID = "F0000"


def run() -> int:
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    handler = select_handler(inputs)
    return handler(inputs)


def select_handler(inputs: list[dict]):
    slug_set = {
        str((value.get("socket") or {}).get("slug") or "").strip()
        for value in inputs
    }
    has_lower = bool({"3d-lower-teeth-scan", "ios-lower-scan"} & slug_set)
    has_upper = bool({"3d-upper-teeth-scan", "ios-upper-scan"} & slug_set)
    has_photo = bool({"2d-intraoral-photographs", "intraoral-photo"} & slug_set)
    if has_lower and has_upper and has_photo:
        return full_pipeline_handler
    raise RuntimeError(
        "Unsupported input socket configuration. "
        f"Found slugs: {sorted(slug for slug in slug_set if slug)}"
    )


def full_pipeline_handler(inputs: list[dict]) -> int:
    lower_scan = load_single_file_compat(
        arch_keywords=("lower", "mandible", "mandibular", "ios_lower"),
        manifest_files=files_from_socket_manifest(
            inputs, {"3d-lower-teeth-scan", "ios-lower-scan"}, ("*.stl", "*.obj")
        ),
        official_locations=(INPUT_PATH / "files" / "ios-lower",),
        fallback_patterns=("*.stl", "*.obj"),
    )
    upper_scan = load_single_file_compat(
        arch_keywords=("upper", "maxilla", "maxillary", "ios_upper"),
        manifest_files=files_from_socket_manifest(
            inputs, {"3d-upper-teeth-scan", "ios-upper-scan"}, ("*.stl", "*.obj")
        ),
        official_locations=(INPUT_PATH / "files" / "ios-upper",),
        fallback_patterns=("*.stl", "*.obj"),
    )
    photo_inputs = load_files_compat(
        manifest_files=files_from_socket_manifest(
            inputs,
            {"2d-intraoral-photographs", "intraoral-photo"},
            ("*.mha", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png", "*.webp"),
        ),
        preferred_locations=(
            INPUT_PATH / "images" / "intraoral-photo",
            INPUT_PATH / "images" / "2d-intraoral-photographs",
        ),
        fallback_locations=(),
        patterns=("*.mha", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png", "*.webp"),
    )

    if not getattr(validate_model_bundle, "_complete", False):
        validate_model_bundle()
    with tempfile.TemporaryDirectory(prefix="bite2text_gc_") as temporary_text:
        temporary = Path(temporary_text)
        runtime_root, bits_root = stage_runtime_roots(
            temporary=temporary,
            lower_scan=lower_scan,
            upper_scan=upper_scan,
            photo_inputs=photo_inputs,
        )
        report = execute_validated_notebook(runtime_root, bits_root)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(
        OUTPUT_PATH / "diagnostic-imaging-report.json",
        {"report": report},
    )
    print(report, flush=True)
    return 0


def validate_model_bundle() -> None:
    required = [
        NOTEBOOK_PATH,
        MANIFEST_PATH,
        TEETHLAND_PYTHON,
        *(MODEL_PATH / "w" / name for name in (
            "view.pt", "crowding-upper.pt", "crowding-lower.pt", "yolo11m.pt",
            "instseg.ckpt", "landmarks.ckpt", "spee.pt", "wilson.pt", "overjet.pt",
        )),
        MODEL_PATH / "u" / "canine-calibration.json",
        MODEL_PATH / "u" / "canine-eruption-gate.json",
        MODEL_PATH / "r" / "toothseg_primary_arch_roi.py",
        *(MODEL_PATH / "q" / name for name in (
            "build_ras_landmark_pointsets.py", "infer_resilient.py",
        )),
        MODEL_PATH / "s" / "a" / "pointcept",
        MODEL_PATH / "s" / "b" / "a.py",
        MODEL_PATH / "s" / "b" / "b",
        MODEL_PATH / "c" / "a.yml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete model/runtime bundle: " + "; ".join(missing))

    manifest = load_json_file(MANIFEST_PATH)
    if manifest.get("v") != 6:
        raise RuntimeError(f"Unsupported model bundle manifest: {MANIFEST_PATH}")
    validate_manifest_files(manifest)
    expected_notebook_hash = manifest.get("n")
    actual_notebook_hash = sha256_file(NOTEBOOK_PATH)
    if expected_notebook_hash != actual_notebook_hash:
        raise RuntimeError(
            "The validated notebook was modified after model bundling: "
            f"expected={expected_notebook_hash}, actual={actual_notebook_hash}"
        )

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is missing from the algorithm container") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This complete Bite2Text pipeline requires a CUDA GPU; refusing to skip "
            "the 3DTeethLand/PTv3 stages."
        )
    try:
        import pointgroup_ops  # noqa: F401
        import pointops  # noqa: F401
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The T4-compatible Pointcept CUDA extensions are unavailable"
        ) from error
    validate_model_bundle._complete = True


def validate_manifest_files(manifest: dict) -> None:
    """Reject truncated, altered, or path-escaping model archive members."""
    model_root = MODEL_PATH.resolve()
    for item in manifest.get("f", []):
        relative = Path(str(item.get("p", "")))
        candidate = (MODEL_PATH / relative).resolve()
        if not relative.parts or not candidate.is_relative_to(model_root):
            raise RuntimeError(f"Unsafe model manifest path: {relative}")
        if item.get("t"):
            if not candidate.is_dir():
                raise FileNotFoundError(candidate)
            actual_bytes = sum(
                path.stat().st_size for path in candidate.rglob("*") if path.is_file()
            )
        else:
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            actual_bytes = candidate.stat().st_size
        if actual_bytes != int(item["b"]):
            raise RuntimeError(
                f"Model member size mismatch: {relative}: "
                f"expected={item['b']}, actual={actual_bytes}"
            )
        expected_hash = item.get("h")
        if expected_hash and sha256_file(candidate) != expected_hash:
            raise RuntimeError(f"Model member SHA-256 mismatch: {relative}")


def stage_runtime_roots(
    *, temporary: Path, lower_scan: Path, upper_scan: Path, photo_inputs: list[Path]
) -> tuple[Path, Path]:
    runtime_root = temporary / "b"
    bits_root = temporary / "x"
    runtime_root.mkdir()

    # Reconstruct the exact paths referenced by the validated notebook while
    # keeping the uploaded archive compact and independent of the workspace.
    aliases = {
        "submission/models/view/view_efficientnet_b3_5class.pt": MODEL_PATH / "w/view.pt",
        "submission/models/crowding/upper_efficientnet_v2_s.pt": MODEL_PATH / "w/crowding-upper.pt",
        "submission/models/crowding/lower_efficientnet_v2_s.pt": MODEL_PATH / "w/crowding-lower.pt",
        "submission/models/3dteethland/instseg_full.ckpt": MODEL_PATH / "w/instseg.ckpt",
        "submission/models/3dteethland/landmarks_full.ckpt": MODEL_PATH / "w/landmarks.ckpt",
        "submission/models/pointcept/curve/curve_spee_toothmask_2class_run369367253595_best_macro_f1.pth": MODEL_PATH / "w/spee.pt",
        "submission/models/pointcept/curve/curve_wilson_toothmask_run369367253578_best_accuracy.pth": MODEL_PATH / "w/wilson.pt",
        "submission/models/pointcept/overjet/overjet_raw_stl_balanced_run369367248013_best_accuracy.pth": MODEL_PATH / "w/overjet.pt",
        "submission/runtime/toothseg2d/toothseg_primary_arch_roi.py": MODEL_PATH / "r/toothseg_primary_arch_roi.py",
        "submission/config/3dteethland_config_full.yaml": MODEL_PATH / "c/a.yml",
        "complexity_aihub_dental/yolo11m_seg_fdi24_20260816/outputs/yolo11m_seg_fdi24_seed42/weights/best.pt": MODEL_PATH / "w/yolo11m.pt",
        "experiments/canine_occlusion_bite2text_calibration/calibration_result.json": MODEL_PATH / "u/canine-calibration.json",
        "experiments/canine_occlusion_bite2text_calibration/eruption_gate_calibration.json": MODEL_PATH / "u/canine-eruption-gate.json",
    }
    for relative, source in aliases.items():
        destination = runtime_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)

    pointcept_root = runtime_root / "submission/runtime/pointcept"
    tools_root = pointcept_root
    tools_root.mkdir(parents=True, exist_ok=True)
    for name, source in {
        "build_ras_landmark_pointsets.py": MODEL_PATH / "q/build_ras_landmark_pointsets.py",
        "infer_resilient.py": MODEL_PATH / "q/infer_resilient.py",
    }.items():
        (tools_root / name).symlink_to(source)
    (pointcept_root / "landmarks/predictions").mkdir(parents=True)
    (pointcept_root / "data/landmark_only_ras_fdi/all_cases").mkdir(parents=True)

    (bits_root / "bits2bites_pointcept").parent.mkdir(parents=True, exist_ok=True)
    (bits_root / "bits2bites_pointcept").symlink_to(MODEL_PATH / "s/a", target_is_directory=True)
    teethland_alias = bits_root / "bits2bites_input_representation/external/3dteethland"
    teethland_alias.parent.mkdir(parents=True, exist_ok=True)
    teethland_alias.mkdir()
    (teethland_alias / "infer.py").symlink_to(MODEL_PATH / "s/b/a.py")
    (teethland_alias / "teethland").symlink_to(MODEL_PATH / "s/b/b", target_is_directory=True)
    case_root = runtime_root / "dataset" / CASE_ID
    ios_dir = case_root / "ios"
    photo_dir = case_root / "intraoral-photo"
    ios_dir.mkdir(parents=True)
    photo_dir.mkdir(parents=True)
    materialize_scan_as_stl(lower_scan, ios_dir / "ios_lower.stl")
    materialize_scan_as_stl(upper_scan, ios_dir / "ios_upper.stl")
    materialize_intraoral_frames(photo_inputs, photo_dir)

    # The notebook checks for this directory before running the 3D classifiers,
    # but challenge inference intentionally has no ground-truth report files.
    (runtime_root / "combined_reports_en_20260715").mkdir(exist_ok=True)
    return runtime_root, bits_root


def materialize_scan_as_stl(source: Path, destination: Path) -> None:
    """Stage the official OBJ/STL socket as a clean STL used by 3DTeethLand."""
    payload = inspect_mesh_payload(source)
    if payload["kind"] in {"binary_stl", "ascii_stl"}:
        shutil.copyfile(source, destination)
        print(
            f"Staged {source.name} -> {destination.name} as detected "
            f"{payload['kind']} ({payload.get('faces', 'unknown')} faces)",
            flush=True,
        )
        return

    staged = materialize_scan_with_pymeshlab(source, destination)
    if staged is not None:
        vertices, faces = staged
        print(
            f"Staged {source.name} -> {destination.name} "
            f"with pymeshlab ({vertices} vertices, {faces} faces)",
            flush=True,
        )
        return

    import trimesh

    mesh = trimesh.load_mesh(source, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            raise RuntimeError(f"No triangular geometry found in {source}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected one triangular mesh in {source}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(
            f"Empty or face-less 3D scan for {source}: "
            f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
        )
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(
            f"3D scan became empty after cleanup for {source}: "
            f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
        )
    mesh.export(destination, file_type="stl")
    print(
        f"Staged {source.name} -> {destination.name} "
        f"({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)",
        flush=True,
    )


def materialize_scan_with_pymeshlab(source: Path, destination: Path) -> tuple[int, int] | None:
    payload = inspect_mesh_payload(source)
    detected_suffix = {
        "obj": ".obj",
        "binary_stl": ".stl",
        "ascii_stl": ".stl",
        "ply": ".ply",
    }.get(str(payload["kind"]), source.suffix.lower())
    staged_source = source
    temporary_source: Path | None = None
    try:
        import pymeshlab

        if detected_suffix and detected_suffix != source.suffix.lower():
            descriptor, temporary_name = tempfile.mkstemp(suffix=detected_suffix)
            os.close(descriptor)
            temporary_source = Path(temporary_name)
            shutil.copyfile(source, temporary_source)
            staged_source = temporary_source
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(staged_source))
        ms.meshing_remove_duplicate_vertices()
        mesh = ms.current_mesh()
        mesh.compact()
        vertices = int(mesh.vertex_number())
        faces = int(mesh.face_number())
        if vertices <= 0 or faces <= 0:
            print(
                f"pymeshlab read {source} as vertices={vertices}, faces={faces}; "
                "trying fallback loader",
                flush=True,
            )
            return None
        ms.save_current_mesh(str(destination))
        return vertices, faces
    except Exception as error:
        print(f"pymeshlab could not stage {source}: {error}", flush=True)
        return None
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)


def materialize_intraoral_frames(photo_inputs: list[Path], output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    frame_index = 0
    for source in photo_inputs:
        suffix = source.suffix.lower()
        if suffix == ".mha":
            frames = read_mha_frames(source)
        else:
            with Image.open(source) as image:
                frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
        for frame in frames:
            output_path = output_dir / f"intraoral_{frame_index:02d}.png"
            frame.convert("RGB").save(output_path)
            output_paths.append(output_path)
            frame_index += 1
    if len(output_paths) < 5:
        raise RuntimeError(
            "The Bite2Text pipeline requires at least five intraoral frames "
            f"(left/right/upper/front/lower); extracted {len(output_paths)}."
        )
    return output_paths


def read_mha_frames(path: Path) -> list[Image.Image]:
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise RuntimeError("SimpleITK is required for .mha intraoral inputs") from error
    array = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
    if array.ndim == 2:
        array = array[None, ..., None]
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = array[None, ...]
    elif array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4:
        raise RuntimeError(f"Unsupported MHA intraoral array shape: {array.shape}")
    frames = []
    for frame in array:
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        frame = np.asarray(frame[..., :3])
        if frame.dtype != np.uint8:
            lo, hi = np.percentile(frame, (0.5, 99.5))
            frame = np.clip((frame - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame, mode="RGB"))
    return frames


def execute_validated_notebook(runtime_root: Path, bits_root: Path) -> str:
    notebook = load_json_file(NOTEBOOK_PATH)
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if not code_cells:
        raise RuntimeError(f"No code cell found in {NOTEBOOK_PATH}")
    # The notebook also contains tiny convenience cells with IPython shell
    # commands. The validated algorithm lives in the single largest code cell.
    source = max(code_cells, key=len)

    source, case_replacements = re.subn(
        r"(?m)^CASE_NUMBER\s*=\s*[^\n#]+",
        f'CASE_NUMBER = "{CASE_ID}"',
        source,
        count=1,
    )
    if case_replacements != 1:
        raise RuntimeError("Validated notebook CASE_NUMBER marker changed or missing")
    root_marker = 'BITE2TEXT_ROOT = Path("/data/MICCAI/Bite2Text")'
    if root_marker not in source:
        raise RuntimeError(f"Validated notebook marker changed or missing: {root_marker}")
    source = source.replace(
        root_marker,
        f'BITE2TEXT_ROOT = Path({str(runtime_root)!r})',
        1,
    )
    bits_marker = "/data/MICCAI/Bits2Bites"
    if bits_marker not in source:
        raise RuntimeError(f"Validated notebook marker changed or missing: {bits_marker}")
    source = source.replace(bits_marker, str(bits_root))

    # Preserve the notebook's separate, validated 3DTeethLand environment.
    source = re.sub(
        r'TEETHLAND_PYTHON = Path\("[^"\n]+"\)',
        f'TEETHLAND_PYTHON = Path({str(TEETHLAND_PYTHON)!r})',
        source,
        count=1,
    )
    namespace = {
        "__name__": "__bite2text_notebook__",
        "__file__": str(NOTEBOOK_PATH),
    }
    bundled_packages = str(MODEL_PATH / "z")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    with patched_environment(
        PYTHONPATH=(
            bundled_packages
            if not existing_pythonpath
            else bundled_packages + os.pathsep + existing_pythonpath
        ),
        MPLBACKEND="Agg",
        TOKENIZERS_PARALLELISM="false",
        YOLO_CONFIG_DIR="/tmp/Ultralytics",
    ):
        sys.path.insert(0, bundled_packages)
        try:
            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
        finally:
            with suppress_value_error():
                sys.path.remove(bundled_packages)

    summary_rows = namespace.get("summary_rows")
    generator = namespace.get("generate_rule_based_report")
    if not isinstance(summary_rows, list) or not callable(generator):
        raise RuntimeError("Notebook did not expose summary_rows and generate_rule_based_report")
    return str(generator(CASE_ID, summary_rows))


@contextmanager
def patched_environment(**updates: str):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def suppress_value_error():
    try:
        yield
    except ValueError:
        pass


def load_json_file(location: Path):
    with location.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_file(location: Path, content) -> None:
    with location.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)


def sha256_file(location: Path) -> str:
    digest = hashlib.sha256()
    with location.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_from_socket_manifest(
    inputs: list[dict],
    slugs: set[str],
    patterns: tuple[str, ...],
) -> tuple[Path, ...]:
    files: list[Path] = []
    for value in inputs:
        socket = value.get("socket") or {}
        slug = str(socket.get("slug") or "").strip()
        if slug not in slugs:
            continue
        relative_path = socket.get("relative_path")
        if not relative_path:
            continue
        candidate = INPUT_PATH / str(relative_path).lstrip("/\\")
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            for pattern in patterns:
                files.extend(Path(path) for path in glob.glob(str(candidate / pattern)))
    return tuple(sorted({path.resolve(): path for path in files}.values(), key=lambda path: path.name))


def load_single_file_compat(
    *,
    arch_keywords: tuple[str, ...],
    manifest_files: tuple[Path, ...],
    official_locations: tuple[Path, ...],
    fallback_patterns: tuple[str, ...],
) -> Path:
    candidates: list[Path] = []
    for location in official_locations:
        for pattern in fallback_patterns:
            candidates.extend(sorted(Path(path) for path in glob.glob(str(location / pattern))))
    candidates.extend(file_path for file_path in manifest_files if file_path.is_file())
    candidates.extend(discover_scan_candidates(arch_keywords, fallback_patterns))
    unique = sorted(
        {path.resolve(): path for path in candidates}.values(),
        key=lambda path: scan_candidate_rank(path, arch_keywords),
    )
    rejected: list[str] = []
    for file_path in unique:
        if scan_has_mesh_geometry(file_path):
            print(f"Using {arch_keywords[0]} scan: {file_path}", flush=True)
            return file_path
        rejected.append(str(file_path))
    raise RuntimeError(
        "No compatible non-empty scan file found. Checked "
        + ", ".join(
            str(path)
            for path in (
                *manifest_files,
                *official_locations,
            )
        )
        + (f". Empty/unreadable candidates: {', '.join(rejected)}" if rejected else "")
        + ". /input files: "
        + "; ".join(describe_input_files())
    )


def discover_scan_candidates(
    arch_keywords: tuple[str, ...],
    patterns: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(path) for path in glob.glob(str(INPUT_PATH / "**" / pattern), recursive=True))
    lowered_keywords = tuple(keyword.lower() for keyword in arch_keywords)
    return [
        path
        for path in files
        if path.is_file()
        and any(keyword in path.as_posix().lower() for keyword in lowered_keywords)
        and "intraoral" not in path.as_posix().lower()
        and "photo" not in path.as_posix().lower()
    ]


def scan_candidate_rank(path: Path, arch_keywords: tuple[str, ...]) -> tuple[int, int, str]:
    text = path.as_posix().lower()
    in_official_dir = "/files/ios-" in text
    is_stl = path.suffix.lower() == ".stl"
    has_keyword = any(keyword.lower() in text for keyword in arch_keywords)
    return (
        0 if in_official_dir else 1,
        0 if is_stl and has_keyword else 1,
        text,
    )


def describe_input_files() -> list[str]:
    descriptions: list[str] = []
    for path in sorted(INPUT_PATH.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            descriptions.append(f"{path} ({size} bytes)")
    return descriptions[:80]


def inspect_mesh_payload(path: Path) -> dict[str, object]:
    """Identify mesh bytes independently of the challenge-provided suffix."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(4096)
            if file_size >= 84:
                stream.seek(80)
                triangle_count = struct.unpack("<I", stream.read(4))[0]
                expected_size = 84 + triangle_count * 50
                if triangle_count > 0 and expected_size == file_size:
                    return {"kind": "binary_stl", "faces": triangle_count}
    except (OSError, struct.error) as error:
        return {"kind": "unreadable", "error": str(error)}

    lowered = prefix.lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
    if lowered.startswith(b"ply"):
        vertices = 0
        faces = 0
        try:
            with path.open("rb") as stream:
                for raw_line in stream:
                    line = raw_line.decode("ascii", errors="ignore").strip().lower()
                    if line.startswith("element vertex "):
                        vertices = int(line.rsplit(maxsplit=1)[-1])
                    elif line.startswith("element face "):
                        faces = int(line.rsplit(maxsplit=1)[-1])
                    elif line == "end_header":
                        break
        except (OSError, ValueError):
            pass
        return {"kind": "ply", "vertices": vertices, "faces": faces}

    vertices = 0
    faces = 0
    stl_facets = 0
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                line = raw_line.lstrip()
                if line.startswith(b"v "):
                    vertices += 1
                elif line.startswith(b"f "):
                    faces += 1
                elif line.lower().startswith(b"facet normal"):
                    stl_facets += 1
    except OSError as error:
        return {"kind": "unreadable", "error": str(error)}
    if vertices > 0 and faces > 0:
        return {"kind": "obj", "vertices": vertices, "faces": faces}
    if lowered.startswith(b"solid") and stl_facets > 0:
        return {"kind": "ascii_stl", "faces": stl_facets}
    return {
        "kind": "unknown",
        "size": file_size,
        "prefix_hex": prefix[:32].hex(),
        "obj_vertices": vertices,
        "obj_faces": faces,
        "stl_facets": stl_facets,
    }


def scan_has_mesh_geometry(path: Path) -> bool:
    payload = inspect_mesh_payload(path)
    if payload["kind"] in {"binary_stl", "ascii_stl", "obj", "ply"}:
        vertices = int(payload.get("vertices", 1))
        faces = int(payload.get("faces", 0))
        if vertices > 0 and faces > 0:
            print(f"Detected scan payload for {path}: {payload}", flush=True)
            return True
    print(f"Mesh payload inspection for {path}: {payload}", flush=True)

    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        mesh = ms.current_mesh()
        vertices = int(mesh.vertex_number())
        faces = int(mesh.face_number())
        if vertices > 0 and faces > 0:
            return True
        print(
            f"Skipping scan candidate {path}: pymeshlab vertices={vertices}, faces={faces}",
            flush=True,
        )
    except Exception as error:
        print(f"pymeshlab could not read scan candidate {path}: {error}", flush=True)

    try:
        import trimesh

        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = [
                geometry
                for geometry in mesh.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
                and geometry.vertices.size > 0
                and geometry.faces.size > 0
            ]
            return bool(meshes)
        return (
            isinstance(mesh, trimesh.Trimesh)
            and mesh.vertices.size > 0
            and mesh.faces.size > 0
        )
    except Exception as error:
        print(f"Skipping unreadable scan candidate {path}: {error}", flush=True)
        return False


def load_files_compat(
    *,
    manifest_files: tuple[Path, ...],
    preferred_locations: tuple[Path, ...],
    fallback_locations: tuple[Path, ...],
    patterns: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = list(manifest_files)
    for location in (*preferred_locations, *fallback_locations):
        for pattern in patterns:
            files.extend(Path(path) for path in glob.glob(str(location / pattern)))
    unique = sorted({path.resolve(): path for path in files}.values(), key=lambda path: path.name)
    if not unique:
        raise RuntimeError(
            "No intraoral photograph found. Checked manifest paths plus "
            + ", ".join(str(path) for path in (*preferred_locations, *fallback_locations))
        )
    return unique


if __name__ == "__main__":
    raise SystemExit(run())
