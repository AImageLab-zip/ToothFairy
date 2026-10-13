"""Dataset + Qwen2.5-VL message path for ToothFairy4 (finetune_qwen_v0).

Port of HEADLINE's dataset ``--recon video`` path (Qwen video +
labeled still images + layout-hint prompt) to ToothFairy Step 0:
  - the HEADLINE **soft-tissue video slot** is filled by our **cross-section stack**
    (render.render_case.frames);
  - the HEADLINE **bone-MIP image slot** is filled by our **panoramic** (+ future MPR
    stills);
  - volumes are streamed from the 44 GB zip OR read from the local /cache extract
    (volume_io.load_canonical_volume), reoriented to canonical + resampled isotropic.

The message shape and the ``process_vision_info(return_video_kwargs=True)`` +
``processor_inputs`` plumbing are HEADLINE's, so an adapted Qwen train.py / inference.py
consumes it unchanged. Prompt is shared by train + inference so preprocessing is identical.

Public interface (mirrors finetune_v0 / HEADLINE):
  load_split_records(split, training) -> list[dict]
  ToothFairyQwenDataset(records, training, ...) -> __getitem__ -> {messages, case_id, report}
  assemble_messages / build_messages / render_messages
  processor_inputs(processor, batch_msgs, add_generation_prompt) -> BatchFeature
"""
from __future__ import annotations

# Must be set before any DataLoader is created (small /dev/shm in the devcontainer).
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

import json
import os
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from .config import (
    ZIP_PATH, VOLUME_CACHE_DIR, REPORTS_CSV, REPORT_LANG, SPLITS_DIR,
    EXPLORE_DIR, MIN_REPORT_CHARS, FRAME_TILE, VIDEO_FPS,
    N_CROSS_SECTIONS, FOV_LIMITED_MM, MAX_VIDEO_FRAMES_INFER,
)
from .render import render_case
from .volume_io import load_canonical_volume

# Reuse the proven zip index/text helpers from the exploration code (same pattern as
# experiments/finetune_v0/dataset.py). Low-level zip access lives in eval/explore/tf_zip.py.
sys.path.insert(0, str(EXPLORE_DIR))
from tf_zip import index_cases, read_text            # noqa: E402


# ── Prompts (shared by train + inference so the prompt is identical) ────────────
# Same task framing as the v0 baseline (experiments/finetune_v0/dataset.py) so the Qwen
# arm builds on the same floor; only the *layout hint* changes (video + panoramic, not
# 14 slices). The assistant target is the reference report.
SYSTEM_PROMPT = (
    "You are an experienced maxillofacial radiologist reporting on a CBCT (cone-beam CT) "
    "scan of the jaws. Write a structured clinical report in English covering the mandible "
    "and maxilla (dentition in FDI notation, alveolar bone, pathologies, endodontic status, "
    "implants, inferior alveolar nerve canal), the mandibular condyles/TMJ, and the maxillary "
    "sinuses.\n"
    "Rules: if a structure (condyles, upper maxilla, nasal floor) is outside the scan volume, "
    'state "[structure] not included in scan volume" — do NOT fabricate findings for unseen '
    'anatomy. If a finding is ambiguous, write "not assessable" rather than guessing. Use FDI '
    "notation throughout. Output only the report text, no preamble."
)

# recon="video": the cross-section stack as a Qwen video + the panoramic (and later MPR)
# as labeled still images. The block ORDER is fixed by render_case/video_content; the hint
# tells the model what each vision block is. Generated from which stills are present so a
# missing panoramic never leaves a dangling reference.
_VIDEO_HINT = (
    # the frames are buccolingual cross-sections taken along the arch (a sweep across the
    # dental arch), NOT an inferior->superior axial stack. The old "(inferior to superior)"
    # parenthetical was false and is dropped; laterality is stated by the caller's hint.
    "This CBCT of the jaws is shown as a video of {n_frames} buccolingual cross-sectional "
    "slices taken along the dental arch"
)
_STILL_LABEL = {
    "panoramic": "a panoramic reconstruction of the dental arch",
    "sinus_mpr": "a coronal MPR through the maxillary sinuses",
    "condyle_l": "a sagittal MPR through the left condyle",
    "condyle_r": "a sagittal MPR through the right condyle",
}
_FOV_NOTE = (
    " Note: this scan has a limited superior-inferior field of view (~{z_mm:.0f} mm); the "
    "condyles and upper maxilla may be partially or fully outside the volume — state this "
    "rather than describing unseen structures."
)
_TRUNC_NOTE = (
    " The superior aspect appears truncated — the mandibular condyles may not be fully "
    "included; report them as not included in the scan volume if absent."
)
_TASK = " Report the findings on this CBCT of the jaws."


def _user_text(n_frames: int, still_keys: list[str], z_mm: float | None,
               superior_truncated: bool = False) -> str:
    txt = _VIDEO_HINT.format(n_frames=n_frames)
    labels = [_STILL_LABEL.get(k, k) for k in still_keys]
    if labels:
        txt += ", followed by " + ", ".join(labels)
    txt += "."
    if z_mm is not None and z_mm < FOV_LIMITED_MM:
        txt += _FOV_NOTE.format(z_mm=z_mm)
    if superior_truncated:
        txt += _TRUNC_NOTE
    return txt + _TASK


# ── Message building (Qwen ChatML: content is a list of vision/text dicts) ──────

def video_content(frames: list, stills: list) -> list[dict]:
    """Qwen vision content: one video item (no-signal-loss knobs) + the still images.

    No-signal-loss knobs: all frames are passed explicitly (no fps sampling /
    token-cap subsample), ``fps`` is set for the temporal position ids, and the frame count is
    padded to even so the temporal-pair merge (temporal_patch=2) drops nothing.

``resized_height``/``width`` are read from the ACTUAL frame size
    (both already multiples of 28 from ``to_wide_rgb``) instead of being pinned to FRAME_TILE².
    Pinning to 448x448 re-stretched the new aspect-preserving ~448x168 frames back to square,
    which would have undone the geometry fix. Falls back to FRAME_TILE² if a frame lacks .size.

``frames=[]`` emits the stills alone with **no video item at all**. An empty video item
    is not the same thing — the processor would still insert the vision placeholder tokens and
    the prompt would promise a video that carries nothing."""
    if not frames:
        return [{"type": "image", "image": img} for _, img in stills]
    try:
        fw, fh = frames[0].size            # PIL (width, height)
    except (AttributeError, IndexError, TypeError):
        fw, fh = FRAME_TILE, FRAME_TILE
    content: list[dict] = [{
        "type":  "video",
        "video": frames,
        "fps":   VIDEO_FPS,
        "resized_height": fh,
        "resized_width":  fw,
    }]
    content += [{"type": "image", "image": img} for _, img in stills]
    return content


def assemble_messages(frames: list, stills: list, z_mm: float | None,
                      report: str | None, superior_truncated: bool = False) -> list[dict]:
    """Wrap rendered views into Qwen chat messages. ``report=None`` -> inference (messages
    end at the user turn so the model generates the continuation)."""
    still_keys = [k for k, _ in stills]
    vision = video_content(frames, stills)
    user_text = _user_text(len(frames), still_keys, z_mm, superior_truncated)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": vision + [{"type": "text", "text": user_text}]},
    ]
    if report is not None:
        msgs.append({"role": "assistant", "content": report})
    return msgs


def render_messages(volume_member: str, zf: zipfile.ZipFile | None = None,
                    local_path: str | Path | None = None, report: str | None = None,
                    n_cross: int = N_CROSS_SECTIONS) -> list[dict]:
    """Load + render a case, then build full chat messages. ``report=None`` -> inference."""
    arr, spacing = load_canonical_volume(volume_member, zf=zf, local_path=local_path)
    rc = render_case(arr, spacing, n_cross=n_cross, tile=FRAME_TILE)
    return assemble_messages(rc.frames, rc.stills, rc.z_mm, report,
                             superior_truncated=rc.superior_truncated)


# Alias kept for parity with HEADLINE/finetune_v0 naming.
build_messages = render_messages


# ── Shared vision extraction + processor call (images AND video) ────────────────

def _content_items(msgs):
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for it in c:
                if isinstance(it, dict):
                    yield it


def extract_vision(msgs):
    """(images, videos, video_kwargs) for a message list. Prefers the official Qwen2.5-VL
    path process_vision_info(return_video_kwargs=True) (converts explicit frames + returns
    fps); degrades gracefully across qwen_vl_utils versions."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        imgs = [it["image"] for it in _content_items(msgs) if it.get("type") == "image"]
        vids = [it["video"] for it in _content_items(msgs) if it.get("type") == "video"]
        return (imgs or None), (vids or None), {}
    try:
        return process_vision_info(msgs, return_video_kwargs=True)
    except TypeError:  # older signature: (images, videos)
        imgs, vids = process_vision_info(msgs)
        return imgs, vids, {}


def processor_inputs(processor, batch_msgs: list[list[dict]], add_generation_prompt: bool):
    """Templatize + run the processor for a batch of message-lists, handling images and
    video uniformly. Returns a BatchFeature (no labels; the train collate adds them). Used
    by train collate and both inference paths so preprocessing is identical (HEADLINE parity)."""
    texts, images, videos, fps = [], [], [], []
    for msgs in batch_msgs:
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt))
        imgs, vids, vkw = extract_vision(msgs)
        if imgs:
            images.extend(imgs)
        if vids:
            videos.extend(vids)
        f = (vkw or {}).get("fps")
        if f is not None:
            fps.extend(f if isinstance(f, (list, tuple)) else [f])
    kwargs = {}
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    if fps:
        # transformers 5.x strict-validates the processor's `fps` video kwarg as a scalar
        # (int|float|None) and rejects a list; the processor then broadcasts it across all
        # videos. Every clip here uses the same VIDEO_FPS, so collapse to one scalar.
        kwargs["fps"] = float(fps[0])
    return processor(text=texts, padding=True, return_tensors="pt", **kwargs)


# ── Record loading ──────────────────────────────────────────────────────────────

def _junk_report_members() -> set[str]:
    """Zip member paths of junk reports (junk=True in reports.csv), e.g. A006 'Unable to
    load CT images.'. Matched by <case>/reports_<lang>/<rid>.txt."""
    if not Path(REPORTS_CSV).exists():
        return set()
    df = pd.read_csv(REPORTS_CSV)
    junk = df[(df["lang"] == REPORT_LANG) & (df["junk"] == True)]  # noqa: E712
    return {f"{r.case_id}/reports_{REPORT_LANG}/{r.rid}.txt" for r in junk.itertuples()}


def _local_volume_path(case_id: str) -> str | None:
    """Local /cache extract path for a case, or None if not extracted yet."""
    p = VOLUME_CACHE_DIR / f"{case_id}.nii.gz"
    return str(p) if p.exists() else None


def load_split_records(split: str, training: bool = True,
                       canonical: bool = False) -> list[dict]:
    """Build records for a frozen split.

    training=True  -> one record per (case, EN report), dropping junk/near-empty reports
                      and volumeless cases (multi-report = independent supervision targets).
    training=False -> one record per case (report=None) for inference/scoring.

    canonical=True -> use the merged canonical-prose target (splits/canonical/<split>.jsonl,
                      built by data_canonical/) instead of the raw EN reports. ONE record per
                      case even for training (contradictory multi-report targets are already
                      merged into one), which also removes the multi-report duplication. For
                      inference (training=False) report stays None; canonical is a no-op there.

    Each record: {case_id, family, volume_member, report_member, report}. Volume rendering
    is deferred to __getitem__ (only the small report text is read here)."""
    case_ids: list[str] = json.loads((SPLITS_DIR / f"{split}.json").read_text())
    cases = index_cases(ZIP_PATH)
    junk = _junk_report_members()

    if canonical and training:
        canon_path = SPLITS_DIR / "canonical" / f"{split}.jsonl"
        prose: dict[str, str] = {}
        struct: dict[str, dict] = {}
        for line in canon_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                prose[obj["case_id"]] = obj["prose"]
                struct[obj["case_id"]] = obj.get("record")   # for the mirror augmentation
        records: list[dict] = []
        for cid in case_ids:
            case = cases.get(cid)
            if case is None or case.volume_member is None:
                continue
            text = prose.get(cid, "").strip()
            if not text:
                continue  # no canonical target for this case
            reps = sorted(m for m in case.reports_en if m not in junk)
            records.append({
                "case_id": cid, "family": case.family,
                "volume_member": case.volume_member,
                "report_member": reps[0] if reps else None, "report": text,
                "record": struct.get(cid),   # structured record (mirror source); None if absent
            })
        assert case_ids, f"empty split {split!r}"
        return records

    records: list[dict] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for cid in case_ids:
            case = cases.get(cid)
            if case is None or case.volume_member is None:
                continue  # A076 (no volume) and any missing case
            reps = sorted(m for m in case.reports_en if m not in junk)
            if not training:
                records.append({
                    "case_id": cid, "family": case.family,
                    "volume_member": case.volume_member,
                    "report_member": reps[0] if reps else None, "report": None,
                })
                continue
            for member in reps:
                text = read_text(zf, member).strip()
                if len(text) < MIN_REPORT_CHARS:
                    continue
                records.append({
                    "case_id": cid, "family": case.family,
                    "volume_member": case.volume_member,
                    "report_member": member, "report": text,
                })
    assert case_ids, f"empty split {split!r}"
    return records


# ── Per-worker zip handle (ZipFile is not fork-safe to share) ───────────────────

@lru_cache(maxsize=None)
def _zip_for_pid(pid: int) -> zipfile.ZipFile:
    return zipfile.ZipFile(ZIP_PATH)


def _zip() -> zipfile.ZipFile:
    return _zip_for_pid(os.getpid())


# ── Dataset ──────────────────────────────────────────────────────────────────────

class ToothFairyQwenDataset(Dataset):
    """Returns dicts: {messages, case_id, report}. Training records include the assistant
    turn; inference records end at the user turn.

    Volumes are read local-cache-first (VOLUME_CACHE_DIR, a plain file read that is safe and
    fast under DataLoader workers) and fall back to the per-worker zip handle only for cases
    not yet extracted. Rendering failures return None (collate drops them)."""

    def __init__(self, records: list[dict], training: bool = True,
                 n_cross: int = N_CROSS_SECTIONS):
        self.records = records
        self.training = training
        # Inference uses the defensive frame cap (no-op in-distribution); training keeps
        # the configured budget so the input distribution is fixed.
        self.n_cross = n_cross if training else min(n_cross, MAX_VIDEO_FRAMES_INFER)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict | None:
        rec = self.records[idx]
        report = rec["report"] if self.training else None
        local_path = _local_volume_path(rec["case_id"])
        try:
            msgs = render_messages(
                rec["volume_member"],
                zf=None if local_path else _zip(),
                local_path=local_path,
                report=report,
                n_cross=self.n_cross,
            )
        except Exception:  # noqa: BLE001 — a corrupt/unreadable volume shouldn't kill training
            return None
        return {"messages": msgs, "case_id": rec["case_id"], "report": report}
