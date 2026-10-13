"""Scans (+ photographs) in, orthodontic report out.

    upper.stl, lower.stl
      -> TSGCNet tooth segmentation            (segmentation.py)
      -> CLIK crown landmarks per tooth        (landmarks.py)
      -> 25 pose-free occlusal measurements    (features.py)
      -> 15 occlusal slot predictions          (slots.py)
      -> occlusal paragraph                    (render.py)

    intraoral_1..5.jpg
      -> multi-view photo CNN + cohort prior   (photo_findings.py)
      -> photo-finding sentences               (render.py)

No language model runs at inference: sentences come from a bank mined from the
training reports, selected by the predicted slot values.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from . import config
from .landmarks import get_landmarks
from .photo_findings import PhotoFindingPredictor, load_prior
from .photo_io import resolve_photo_views
from .render import render_occlusal, render_photo_findings
from .scan_io import load_scan
from .segmentation import segment_faces, split_teeth
from .slots import predict_slots

log = logging.getLogger("bite2text")


@dataclass
class Report:
    text: str
    occlusal_text: str
    photo_text: str
    photo_source: str                   # "model" (patient's photos) or "prior" (cohort prior)
    slots: dict
    photo_slots: dict
    landmarks: Dict[str, Dict[str, list]]
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"report": self.text, "occlusal_text": self.occlusal_text,
                "photo_text": self.photo_text, "photo_source": self.photo_source,
                "slots": self.slots, "photo_slots": self.photo_slots, "timings": self.timings}


class ReportGenerator:
    """Loads models lazily on first use and keeps them for subsequent patients."""

    def __init__(self, device: Optional[str] = None, seed: int = 0):
        self.device = config.resolve_device(device)
        self.seed = seed
        self._photo_predictor: Optional[PhotoFindingPredictor] = None

    # -- stages --------------------------------------------------------------

    def scan_landmarks(self, path, jaw: str) -> Dict[str, Dict[str, list]]:
        """One jaw scan -> ``{fdi_str: {landmark name: [x, y, z]}}``."""
        mesh = load_scan(path)
        labels = segment_faces(mesh, jaw, device=self.device)
        teeth = split_teeth(mesh, labels, jaw)
        # PointMLP's grouping layers sample with torch's global RNG
        torch.manual_seed(self.seed)
        found = get_landmarks(teeth, device=self.device, seed=self.seed)
        log.info("%s jaw: %d teeth segmented, landmarks on %d", jaw, len(teeth), len(found))
        return {str(fdi): {name: [float(c) for c in xyz] for name, xyz in lms.items()}
                for fdi, lms in found.items()}

    def photo_predictor(self) -> PhotoFindingPredictor:
        if self._photo_predictor is None:
            self._photo_predictor = PhotoFindingPredictor(device=self.device)
        return self._photo_predictor

    def photo_findings(self, photos: Iterable) -> tuple:
        """-> ``(photo slots, "model" | "prior")``.

        Photographs are optional. If they are missing or unreadable, the cohort
        prior is used and the log says so; the scans alone still yield a report.
        """
        photos = list(photos or [])
        try:
            if not photos:
                raise RuntimeError("no intraoral photographs supplied")
            views = resolve_photo_views(photos)
            if not views:
                raise RuntimeError(f"no photo views could be decoded from {[Path(p).name for p in photos]}")
            predictor = self.photo_predictor()
            slots = predictor.predict(views)
            if not render_photo_findings(slots).strip():
                raise RuntimeError("model and prior together produced no photo sentences")
            log.info("photo findings from %d view(s); model heads %s, prior for the rest",
                     len(views), predictor.trusted_heads)
            return slots, "model"
        except Exception as exc:
            log.warning("photo findings from the cohort prior (NO photographs were read): %s", exc)
            return load_prior(), "prior"

    # -- end to end ----------------------------------------------------------

    def __call__(self, upper, lower, photos: Optional[Iterable] = None) -> Report:
        timings: Dict[str, float] = {}

        landmarks: Dict[str, Dict[str, list]] = {}
        for jaw, path in (("upper", upper), ("lower", lower)):
            t0 = time.perf_counter()
            landmarks.update(self.scan_landmarks(path, jaw))
            timings[f"{jaw}_scan"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        slots = predict_slots(landmarks)
        if slots is None:
            raise RuntimeError(f"too few teeth to build an anatomical frame ({len(landmarks)} with landmarks)")
        occlusal = render_occlusal(slots)
        timings["occlusal_report"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        photo_slots, source = self.photo_findings(photos)
        photo_text = render_photo_findings(photo_slots, generic_only=True)
        timings["photo_report"] = round(time.perf_counter() - t0, 2)

        text = f"{occlusal} {photo_text}".strip() if photo_text else occlusal
        return Report(text=text, occlusal_text=occlusal, photo_text=photo_text,
                      photo_source=source, slots=slots, photo_slots=photo_slots,
                      landmarks=landmarks, timings=timings)


def generate_report(upper, lower, photos: Optional[Iterable] = None, **kwargs) -> str:
    """One-shot convenience: report text for one patient."""
    return ReportGenerator(**kwargs)(upper, lower, photos).text
