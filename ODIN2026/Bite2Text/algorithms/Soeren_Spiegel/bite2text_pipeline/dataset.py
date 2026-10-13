"""Bite2Text dataset: case -> Qwen2.5-VL chat messages.

Mirrors the TF4 contract (experiments/finetune_qwen_v1/dataset_fiveviews.py) so
the training loop and collator carry over unchanged:
  - messages = [system, user(vision items + one text item), assistant?]
  - vision items embed PIL Image objects directly ({"type": "image", "image": im})
  - report=None  => inference (no assistant turn)
  - Dataset.__getitem__ -> dict | None; None = skip (collate drops)

No video element: the Bite2Text input is 7 rendered mesh views + up to 7 photos,
all stills. Chart-concat (chart block + "\\n\\nReport:\\n" + prose) hooks in once
the canonical Bite2Text targets land in data_bite2text/.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import case_frame, read_stl
from .mesh_views import render_views
from .photos import load_photos_from_zip

def _resolve_zip():
    """Env override (B2T_ZIP) first, then the default placeholder path."""
    import os
    cands = [os.environ.get("B2T_ZIP"),
             "/path/to/Bite2Text.zip"]
    for c in cands:
        if c and Path(c).exists():
            return c
    return cands[-1]


ZIP_PATH = _resolve_zip()
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data_bite2text" / "splits"
MIN_REPORT_CHARS = 40      # drop degenerate/empty targets (TF4 parity)
MIN_MESH_VERTS = 1000      # below this a "mesh" is a zero-byte/garbage STL

RENDER_ORDER = ("frontal", "buccal_right", "buccal_left", "occlusal_upper",
                "occlusal_lower", "cutaway", "occlusogram")

SYSTEM_PROMPT = (
    "You are an experienced orthodontist reporting on a patient's intraoral records "
    "(upper and lower 3D dental scans and clinical photographs). Write a concise "
    "orthodontic report in English covering: transverse relationships (maxillary "
    "constriction, crossbite), vertical relationships (overbite), sagittal "
    "relationships (molar and canine class on each side, overjet), dental midlines, "
    "the curves of Spee and Wilson, crowding and spacing in each arch, dentition "
    "stage if not permanent, and — where visible in the photographs — soft-tissue "
    "and tooth-surface findings (gingiva, plaque, staining, sealants, caries risk).\n"
    "Rules: use FDI tooth notation (\"tooth 36\"). If a relationship cannot be "
    "judged from the available views, write \"not assessable\" rather than "
    "guessing. Output only the report text, no preamble."
)

_VIEW_LABEL = {
    "frontal": ("a rendered frontal view of the scans in occlusion "
                "(semi-transparent: the farther jaw shows through the nearer one)"),
    "buccal_right": "a rendered right buccal view in occlusion (semi-transparent)",
    "buccal_left": "a rendered left buccal view in occlusion (semi-transparent)",
    "occlusal_upper": "a rendered occlusal view of the upper arch",
    "occlusal_lower": "a rendered occlusal view of the lower arch",
    "cutaway": "a mid-sagittal cutaway of the occlusion (upper tinted red, lower green)",
    "occlusogram": (
        "an occlusal overlay map of both arches (red = upper-arch height, green = "
        "lower-arch height, yellow = arch overlap, blue-to-white = occlusal "
        "proximity with white = contact, dark engraved lines = tooth edges)"),
}

_LATERALITY = (
    " In the frontal, occlusal and overlay renders the left of the image is the "
    "patient's RIGHT side; in the right buccal render anterior points to image "
    "right, in the left buccal render to image left.")

_PHOTO_NOTE = (
    " After the renders, {n} clinical intraoral photographs of the same patient "
    "follow in arbitrary order (frontal, buccal and occlusal-mirror views; the "
    "occlusal photographs are mirror shots, so their left-right is flipped "
    "relative to the renders).")

_TASK = " Report the findings."


def _user_text(view_keys, n_photos, measured=None):
    labels = [_VIEW_LABEL[k] for k in view_keys if k in _VIEW_LABEL]
    txt = (f"This patient's records are shown as {len(labels)} rendered views of "
           f"the bite-registered upper and lower intraoral 3D scans: "
           + ", ".join(labels) + ".")
    txt += _LATERALITY
    if n_photos:
        txt += _PHOTO_NOTE.format(n=n_photos)
    if measured:
        txt += (" Measured from the registered scans: "
                + "; ".join(f"{k} ≈ {v:.1f} mm" for k, v in measured.items())
                + ".")
    txt += _TASK
    return txt


def assemble_messages(renders, photos, report=None, measured=None):
    """renders: dict name->uint8 array or PIL; photos: list[PIL]."""
    vision = []
    keys = [k for k in RENDER_ORDER if k in renders]
    for k in keys:
        im = renders[k]
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        vision.append({"type": "image", "image": im})
    for ph in photos:
        vision.append({"type": "image", "image": ph})
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": vision + [
            {"type": "text", "text": _user_text(keys, len(photos), measured)}]},
    ]
    if report is not None:
        msgs.append({"role": "assistant", "content": report})
    return msgs


def build_case_inputs(case_id, zf=None, zip_path=ZIP_PATH):
    """Training-side loader: case id -> (renders, photos). Renders may be {} on
    mesh failure (photos-only fallback); photos may be [] likewise."""
    close = False
    if zf is None:
        zf = zipfile.ZipFile(zip_path)
        close = True
    try:
        renders = {}
        try:
            names = [n for n in zf.namelist()
                     if n.startswith(f"{case_id}/ios/") and n.lower().endswith(".stl")]
            up = [n for n in names if "upper" in n.lower()]
            lo = [n for n in names if "lower" in n.lower()]
            if not (up and lo):
                names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                up, lo = names[:1], names[1:2]
            vu, nu = read_stl(zf.read(up[0]))
            vl, nl = read_stl(zf.read(lo[0]))
            if len(vu) > MIN_MESH_VERTS and len(vl) > MIN_MESH_VERTS:
                renders = render_views(vu, nu, vl, nl, case_frame(vu, vl))
        except Exception:
            renders = {}
        photos = load_photos_from_zip(zf, case_id)
        return renders, photos
    finally:
        if close:
            zf.close()


def render_case_messages(case_id, report=None, zf=None):
    """-> (messages, n_renders, n_photos). Photos-only and renders-only cases
    both produce valid messages; a case with neither returns (None, 0, 0)."""
    renders, photos = build_case_inputs(case_id, zf=zf)
    if not renders and not photos:
        return None, 0, 0
    return assemble_messages(renders, photos, report=report), len(renders), len(photos)


def load_split_records(split, canonical_dir=None):
    """Reads data_bite2text/splits/<split>.json (case-id list) and attaches the
    canonical target prose from splits/canonical jsonl once it exists."""
    ids = json.loads((SPLITS_DIR / f"{split}.json").read_text())
    targets = {}
    can = Path(canonical_dir) if canonical_dir else SPLITS_DIR / "canonical"
    jl = can / f"{split}.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            rec = json.loads(line)
            targets[rec["case_id"]] = rec
    return [{"case_id": cid, "target": targets.get(cid)} for cid in ids]


class Bite2TextDataset:
    """torch-style Dataset; __getitem__ -> dict | None (collate drops None)."""

    def __init__(self, records, training=True, chart_concat=False, seed=0):
        self.records = records
        self.training = training
        self.chart_concat = chart_concat
        self.seed = seed
        self.epoch = 0
        self._zf = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def _zip(self):
        if self._zf is None:      # one handle per worker process
            self._zf = zipfile.ZipFile(ZIP_PATH)
        return self._zf

    def __getitem__(self, idx):
        rec = self.records[idx]
        target = rec.get("target")
        report = None
        try:
            if self.training:
                if not target:
                    return None
                report = target.get("prose") or ""
                if len(report) < MIN_REPORT_CHARS:
                    return None
                if self.chart_concat:
                    from data_bite2text.chart_render import concat_target
                    report = concat_target(target["record"], report)
            messages, n_r, n_p = render_case_messages(
                rec["case_id"], report=report, zf=self._zip())
        except Exception:
            return None
        if messages is None or (self.training and n_r == 0 and n_p == 0):
            return None
        return {"messages": messages, "case_id": rec["case_id"],
                "report": report or "", "superior_truncated": False}
