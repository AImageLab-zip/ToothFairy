"""Five intraoral photographs -> photo-finding slots.

Ten findings are predicted: gingivitis, plaque, calculus, hygiene, restorations,
sealants, caries, discoloration, recession, enamel_defect. Only heads that beat
their majority baseline on the hand-labelled dev set are trusted (recorded in
``weights/photo_cnn/results.json`` -> ``use_model``: gingivitis, restorations
and recession). The remaining heads are served from the cohort prior
(``data/photo_prior.json``), the finding set selected against real training reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

from . import config
from .nets.photo_cnn import MultiViewPhotoNet

VIEW_ORDER = ["front", "right", "left", "lower", "upper"]

# head -> class names; index 0 is always "none" (the report is silent on it)
HEADS: Dict[str, list] = {
    "gingivitis":    ["none", "true", "false", "uncertain"],
    "plaque":        ["none", "present"],
    "calculus":      ["none", "present"],
    "hygiene":       ["none", "poor"],
    "restorations":  ["none", "restoration", "crown"],
    "sealants":      ["none", "present"],
    "caries":        ["none", "present", "suspected"],
    "discoloration": ["none", "pigment", "stain", "white_spot"],
    "recession":     ["none", "present"],
    "enamel_defect": ["none", "hypoplasia", "fracture", "wear"],
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_prior() -> dict:
    """Cohort prior: photo-finding slots asserted when a patient's photos cannot be used."""
    return json.loads((config.DATA_DIR / "photo_prior.json").read_text(encoding="utf-8"))


def decode_slots(pred: Dict[str, str]) -> dict:
    """Predicted class name per head -> slot dict consumed by ``render_photo_findings``."""
    def on(head: str) -> bool:
        return pred.get(head, "none") != "none"

    slots = {
        "hygiene": "poor" if pred.get("hygiene") == "poor" else None,
        "plaque": True if on("plaque") else None,
        "calculus": True if on("calculus") else None,
        "sealants": True if on("sealants") else None,
        "recession": {} if on("recession") else None,
        "restorations": {"type": [pred["restorations"]]} if on("restorations") else None,
        "caries": {"status": pred["caries"]} if on("caries") else None,
        "discoloration": {"type": [pred["discoloration"]]} if on("discoloration") else None,
        "enamel_defect": {"type": [pred["enamel_defect"]]} if on("enamel_defect") else None,
        "gingivitis": {"true": True, "false": False, "uncertain": "uncertain"}.get(
            pred.get("gingivitis", "none")),
    }
    return slots


class PhotoFindingPredictor:
    def __init__(self, run_dir: Optional[Path] = None, device=None):
        run_dir = Path(run_dir) if run_dir else config.photo_cnn_dir()
        checkpoint = run_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"photo CNN checkpoint not found: {checkpoint}")

        self.device = config.resolve_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        self.image_size = int(cfg.get("image_size", 224))

        self.model = MultiViewPhotoNet({h: len(c) for h, c in HEADS.items()},
                                       backbone=cfg.get("backbone", "resnet18"),
                                       pooling=cfg.get("pooling", "mean"),
                                       dropout=cfg.get("dropout", 0.3))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.use_model = json.loads((run_dir / "results.json").read_text())["use_model"]
        self.prior = load_prior()

    @property
    def trusted_heads(self) -> list:
        return [h for h, keep in self.use_model.items() if keep]

    @torch.no_grad()
    def predict(self, views: Dict[str, Image.Image]) -> dict:
        """``{view name: PIL image}`` (any subset of VIEW_ORDER) -> photo slots."""
        size = self.image_size
        images, mask = [], []
        for view in VIEW_ORDER:
            image = views.get(view)
            if image is not None:
                images.append(self.transform(image.convert("RGB")))
                mask.append(True)
            else:
                images.append(torch.zeros(3, size, size))
                mask.append(False)
        tensor = torch.stack(images).unsqueeze(0).to(self.device)
        view_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        logits = self.model(tensor, view_mask)
        names = {h: HEADS[h][int(logits[h].argmax(dim=1))] for h in HEADS}
        predicted = decode_slots(names)
        return {h: predicted.get(h) if self.use_model.get(h, False) else self.prior.get(h)
                for h in HEADS}
