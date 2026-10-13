"""Parser for the Bite2Text fixed-slot chart block — inverse of chart_render.chart_block.

Stdlib-only (vendored into the container without numpy/PIL), mirroring
toothfairy/chart_parse.py's role for TF4: parse a generated chart back into slot
values for slot-level accuracy, chart<->report consistency, and the class-recall
eval. STRICT: any malformed line, unknown token, or bad FDI fails the WHOLE block
(returns None) — partial credit hides format drift.

parse_chart(text) -> {"slots": {slot: value_token}, "degrees": {side_slot: degree|None},
                      "teeth": {fdi: [finding_token, ...]}} | None
compare_charts(pred, gold) -> per-slot + per-class tallies for aggregation.
"""
from __future__ import annotations

import re

CHART_HEADER = "Chart:"
DELIMITER = "\n\nReport:\n"

_CLASS_TOKENS = {"class1", "class2", "class3", "not-assessable"}
_DEGREES = {"full", "end-to-end", "half", "one-quarter", "three-quarter", "tendency"}
_SIMPLE = {
    "dentition": {"permanent", "mixed", "deciduous"},
    "transverse": {"normal", "constricted", "expanded", "not-assessable"},
    "xbite": {"none", "present", "not-assessable"},
    "overbite": {"normal", "increased", "reduced", "open-bite", "edge-to-edge",
                 "not-assessable"},
    "overjet": {"normal", "increased", "reduced", "zero", "reverse", "not-assessable"},
    "midline U": {"coincident", "deviated", "dev-right", "dev-left", "not-assessable"},
    "midline L": {"coincident", "deviated", "dev-right", "dev-left", "not-assessable"},
    "spee": {"normal", "increased", "flat", "not-assessable"},
    "wilson": {"normal", "increased", "flat", "not-assessable"},
    "crowding U": {"none", "mild", "moderate", "severe", "not-assessable"},
    "crowding L": {"none", "mild", "moderate", "severe", "not-assessable"},
    "spacing U": {"none", "present", "diastema", "not-assessable"},
    "spacing L": {"none", "present", "diastema", "not-assessable"},
}
_TOOTH_TOKENS = {
    "absent", "unerupted", "erupting", "impacted", "retained-primary", "crossbite",
    "scissor-bite", "sealed", "caries-risk", "restored", "crowned", "worn",
    "fractured", "rotated", "displaced", "infra-occlusion",
}
_SLOT_KEY = {"midline U": "midline_u", "midline L": "midline_l",
             "crowding U": "crowding_u", "crowding L": "crowding_l",
             "spacing U": "spacing_u", "spacing L": "spacing_l"}
_SIDE_KEY = {"molar R": "molar_r", "molar L": "molar_l",
             "canine R": "canine_r", "canine L": "canine_l"}
_FDI = re.compile(r"^(1[1-8]|2[1-8]|3[1-8]|4[1-8]|5[1-5]|6[1-5]|7[1-5]|8[1-5])$")

# fixed line order (chart_render.chart_block); index -> expected label prefix set
_EXPECTED = [
    ("dentition",), ("transverse", "xbite"), ("overbite",),
    ("molar R", "molar L"), ("canine R", "canine L"), ("overjet",),
    ("midline U", "midline L"), ("spee", "wilson"),
    ("crowding U", "crowding L"), ("spacing U", "spacing L"), ("teeth",),
]


def _parse_side(val):
    parts = val.split()
    if not parts or parts[0] not in _CLASS_TOKENS:
        return None
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2 and parts[1] in _DEGREES and parts[0] != "not-assessable":
        return parts[0], parts[1]
    return None


def parse_chart(text: str):
    if not text:
        return None
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines or lines[0].strip() != CHART_HEADER:
        return None
    lines = lines[1:]
    if len(lines) != len(_EXPECTED):
        return None
    slots, degrees, teeth = {}, {}, {}
    for ln, expected in zip(lines, _EXPECTED):
        segs = [s.strip() for s in ln.split("|")]
        if len(segs) != len(expected):
            return None
        for seg, label in zip(segs, expected):
            if not seg.startswith(label + ":"):
                return None
            val = seg[len(label) + 1:].strip()
            if label == "teeth":
                if val == "unremarkable":
                    continue
                for bit in val.split(","):
                    parts = bit.strip().split()
                    if len(parts) < 2 or not _FDI.match(parts[0]):
                        return None
                    toks = parts[1:]
                    if any(t not in _TOOTH_TOKENS for t in toks):
                        return None
                    if parts[0] in teeth:
                        return None
                    teeth[parts[0]] = toks
            elif label in _SIDE_KEY:
                parsed = _parse_side(val)
                if parsed is None:
                    return None
                slots[_SIDE_KEY[label]], degrees[_SIDE_KEY[label]] = parsed
            else:
                if val not in _SIMPLE[label]:
                    return None
                slots[_SLOT_KEY.get(label, label)] = val
    return {"slots": slots, "degrees": degrees, "teeth": teeth}


def compare_charts(pred, gold):
    """-> dict with slot_correct/slot_total, per-class {tp, fp, fn}, teeth exact/jaccard.
    pred may be None (parse failure): everything counts as wrong/missed."""
    per_class: dict[str, dict] = {}

    def _cls(tok):
        return per_class.setdefault(tok, {"tp": 0, "fp": 0, "fn": 0})

    slot_total = len(gold["slots"])
    slot_correct = 0
    for k, gv in gold["slots"].items():
        pv = (pred or {"slots": {}})["slots"].get(k)
        if pv == gv:
            slot_correct += 1
            _cls(gv)["tp"] += 1
        else:
            _cls(gv)["fn"] += 1
            if pv is not None:
                _cls(pv)["fp"] += 1
    g_teeth = {(f, t) for f, toks in gold["teeth"].items() for t in toks}
    p_teeth = set() if pred is None else {(f, t) for f, toks in pred["teeth"].items()
                                          for t in toks}
    for f, t in g_teeth & p_teeth:
        _cls(t)["tp"] += 1
    for f, t in g_teeth - p_teeth:
        _cls(t)["fn"] += 1
    for f, t in p_teeth - g_teeth:
        _cls(t)["fp"] += 1
    union = len(g_teeth | p_teeth)
    return {"slot_correct": slot_correct, "slot_total": slot_total,
            "teeth_jaccard": (len(g_teeth & p_teeth) / union) if union else 1.0,
            "per_class": per_class, "parse_failed": pred is None}
