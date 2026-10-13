"""Laterality-consistent left/right mirror of a canonical record.

A left-right image flip swaps every FDI quadrant, so the target
report must be transformed to match or the augmentation teaches the wrong quadrant — the exact
error the v03 re-release existed to fix. Because the canonical record is structured, the
transform is exact rather than a regex over prose:

  - ``teeth`` FDI keys and every FDI code inside a free-text string are quadrant-swapped
    (1<->2, 3<->4, 5<->6, 7<->8 — permanent and deciduous), tooth position unchanged;
  - ``ian_canal.left`` / ``.right`` dicts are swapped;
  - side words ("left"<->"right", case preserved) and quadrant tokens ("Q1"<->"Q2",
    "Q3"<->"Q4") are swapped in every string.

``mirror_is_safe`` is the gate: a 2-digit FDI-shaped token that is actually a MEASUREMENT
("14-15 mm", "~48 mm") or a cardinal COUNT ("13 crowns", "11 implants") would be corrupted by the
FDI remap, so any record carrying one is declared unsafe and not mirrored (a handful of cases).
The gate is best-effort, not a proof: it covers the token forms observed across all 622 records,
so a novel non-tooth 2-digit form could still slip through — re-run the mirror check if the
corpus changes.

Usage (online, in the dataset): mirror the raw volume on x, and use
``render_prose.render(mirror_record(rec))`` as the target.
"""
from __future__ import annotations

import re
from copy import deepcopy

# FDI quadrant under a left-right mirror. Second digit (tooth position) is unchanged.
_QSWAP = {"1": "2", "2": "1", "3": "4", "4": "3", "5": "6", "6": "5", "7": "8", "8": "7"}

_FDI_RE = re.compile(r"\b([1-8][1-8])\b")
_SIDE_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)
_QUAD_RE = re.compile(r"\bQ([1-4])\b")
# A measurement whose value is FDI-SHAPED ([1-8][1-8]) and would therefore be corrupted by the
# FDI remap. Only such tokens matter: single-digit ("5 mm") and non-FDI-shaped ("10 mm")
# measurements are never touched by ``_FDI_RE``, so they must not trigger a skip. If any string
# in the record matches, the record is left un-mirrored.
_MEAS_RE = re.compile(
    r"~\s*[1-8][1-8]\b"                                # ~14
    r"|\b[1-8][1-8]\s*[-–]\s*\d{1,2}\s*(?:mm|cm)"      # 14-15 mm (first token is FDI-shaped)
    r"|\b\d{1,2}\s*[-–]\s*[1-8][1-8]\s*(?:mm|cm)"      # 9-14 mm  (second token is FDI-shaped)
    r"|\b[1-8][1-8]\s*(?:mm|cm|%)"                     # 48 mm
    # A cardinal COUNT that is FDI-shaped: "13 crowns", "11 implants", "28 teeth". The FDI remap
    # would corrupt the number (13->23). Only plural count-nouns trigger this; a singular site
    # ref ("tooth 13", "implant 37") has the noun BEFORE the number and is unaffected. Measured:
    # only S0007 (crowns) and P119 (implants) match across all 622 records.
    r"|\b[1-8][1-8]\s+(?:implants|crowns|teeth|bridges|cantilevers)\b",
    re.IGNORECASE,
)


def _swap_fdi_code(code: str) -> str:
    return _QSWAP[code[0]] + code[1]


def _swap_fdi_in_str(s: str) -> str:
    return _FDI_RE.sub(lambda m: _swap_fdi_code(m.group(1)), s)


def _swap_sides(s: str) -> str:
    def rep(m):
        w = m.group(0)
        other = "right" if w.lower() == "left" else "left"
        if w.isupper():
            return other.upper()
        if w[0].isupper():
            return other.capitalize()
        return other
    return _SIDE_RE.sub(rep, s)


def _swap_quadrant_tokens(s: str) -> str:
    return _QUAD_RE.sub(lambda m: "Q" + _QSWAP[m.group(1)], s)


def _mirror_str(s):
    if not isinstance(s, str) or not s:
        return s
    return _swap_quadrant_tokens(_swap_sides(_swap_fdi_in_str(s)))


def _iter_strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for vv in v.values():
            yield from _iter_strings(vv)
    elif isinstance(v, list):
        for vv in v:
            yield from _iter_strings(vv)


def mirror_is_safe(rec) -> bool:
    """False when any string carries a measurement-shaped 2-digit token the FDI remap would
    corrupt. Those records are left un-mirrored rather than mangled."""
    return not any(_MEAS_RE.search(s) for s in _iter_strings(rec))


def _mirror_obj(v):
    """Recursively mirror every string in a JSON-like structure (keys handled by caller)."""
    if isinstance(v, str):
        return _mirror_str(v)
    if isinstance(v, list):
        return [_mirror_obj(x) for x in v]
    if isinstance(v, dict):
        return {k: _mirror_obj(x) for k, x in v.items()}
    return v


def mirror_record(rec):
    """Return a deep-copied, left-right-mirrored canonical record. Caller should check
    ``mirror_is_safe`` first; mirroring an unsafe record is still laterality-correct for teeth
    but may corrupt a measurement string."""
    r = deepcopy(rec)

    # teeth: quadrant-swap the FDI KEYS, then mirror the values' strings.
    teeth = r.get("teeth", {})
    r["teeth"] = {_swap_fdi_code(k): _mirror_obj(v) for k, v in teeth.items()}

    # ian_canal: swap the left/right dicts, then mirror their strings.
    canal = r.get("ian_canal", {})
    if isinstance(canal, dict) and ("left" in canal or "right" in canal):
        new = {k: v for k, v in canal.items() if k not in ("left", "right")}
        if "left" in canal:
            new["right"] = _mirror_obj(canal["left"])
        if "right" in canal:
            new["left"] = _mirror_obj(canal["right"])
        r["ian_canal"] = new

    # everything else: mirror strings in place (fov, bone_and_perio, prostheses, sinuses,
    # other_findings, merge_notes, ...). teeth and ian_canal already handled above.
    for k, v in list(r.items()):
        if k in ("teeth", "ian_canal", "case_id", "family", "source_rids"):
            continue
        r[k] = _mirror_obj(v)
    return r


if __name__ == "__main__":
    import json, sys
    rec = json.load(open(sys.argv[1]))
    print(json.dumps(mirror_record(rec), indent=2, ensure_ascii=False))
