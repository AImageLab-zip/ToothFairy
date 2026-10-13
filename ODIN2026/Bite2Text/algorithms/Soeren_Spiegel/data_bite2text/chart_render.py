"""Bite2Text chart + prose rendering from extracted case records (SCHEMA.md v1.2).

Three jobs, mirroring data_canonical/{chart_render,render_prose}.py for TF4:
  complete_record(rec) -> all core slots with healthy defaults filled in
      ("unmentioned => normal", the adopted completion rule; per-tooth and
      soft-tissue facts stay stated-only)
  chart_block(rec)     -> the fixed-slot chart text (training target prefix)
  render_prose(rec)    -> deterministic one-paragraph report in corpus phrasing
  concat_target(rec, report) -> chart_block + DELIMITER + report  (training target)
  chart_state(rec)     -> {slot: class_token} for chart_loss class weighting

Design constraints inherited from experiments/finetune_qwen_v1/chart_loss.py, which
scans the decoded chart text for class-name occurrences (longest-first):
  * CHART_CLASSES tokens are mutually substring-free (e.g. class degrees are
    "one-quarter"/"three-quarter", never bare "quarter"; midline sides are
    "dev-right"/"dev-left" so "deviated" stays collision-free; tooth crowns are
    "crowned" not "crown"; the transverse sub-slot label is "xbite" so the tooth
    finding "crossbite" is scannable).
  * scaffold words (labels, "unremarkable") are NOT class tokens.

Extraction `note` fields carry provenance (vote-group/tie explanations) and are
NEVER rendered — only enum values, FDI lists and the whitelisted free-text fields
(appliances[].v, other_findings[].v, soft_tissue.other.v) reach the prose, the
latter gated by _clean() against leaked rule references.
"""
from __future__ import annotations

import re

DELIMITER = "\n\nReport:\n"
CHART_HEADER = "Chart:"

# ── class-token vocabulary (substring-free; audited pairwise) ────────────────────
_CLASS = {"I": "class1", "II": "class2", "III": "class3",
          "not_assessable": "not-assessable"}
_DEGREE = {"full": "full", "end_to_end": "end-to-end", "half": "half",
           "quarter": "one-quarter", "three_quarter": "three-quarter",
           "tendency": "tendency"}
_TOOTH_FINDING = {
    "absent": "absent", "unerupted": "unerupted", "erupting": "erupting",
    "impacted": "impacted", "retained_deciduous": "retained-primary",
    "crossbite": "crossbite", "scissor_bite": "scissor-bite", "sealed": "sealed",
    "caries_suspect": "caries-risk", "restored": "restored",
    "prosthetic_crown": "crowned", "worn": "worn", "fractured": "fractured",
    "rotated": "rotated", "displaced": "displaced",
    "infra_occlusion": "infra-occlusion",
}
_MIDLINE = {"coincident": "coincident", "deviated": "deviated",
            "deviated_right": "dev-right", "deviated_left": "dev-left",
            "not_assessable": "not-assessable"}
_PLAIN = lambda v: {"open_bite": "open-bite", "edge_to_edge": "edge-to-edge",
                    "not_assessable": "not-assessable"}.get(v, v)

CHART_CLASSES = (
    "class1", "class2", "class3",
    "full", "end-to-end", "half", "one-quarter", "three-quarter", "tendency",
    "normal", "constricted", "expanded", "increased", "reduced",
    "open-bite", "edge-to-edge", "zero", "reverse",
    "coincident", "deviated", "dev-right", "dev-left", "flat",
    "none", "present", "mild", "moderate", "severe", "diastema",
    "permanent", "mixed", "deciduous",
    "absent", "unerupted", "erupting", "impacted", "retained-primary",
    "crossbite", "scissor-bite", "sealed", "caries-risk", "restored", "crowned",
    "worn", "fractured", "rotated", "displaced", "infra-occlusion",
    "not-assessable",
)
assert not any(a != b and a in b for a in CHART_CLASSES for b in CHART_CLASSES), \
    "CHART_CLASSES must be substring-free for chart_loss.class_occurrences"


def _v(rec, *path, default=None):
    cur = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if isinstance(cur, dict) and "v" in cur:
        return cur["v"]
    return cur if cur is not None else default


def _hedged(rec, *path):
    cur = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return bool(isinstance(cur, dict) and cur.get("hedged"))


def complete_record(rec) -> dict:
    """Core slots with healthy defaults; enum values from the raw record verbatim."""
    g = lambda *p: _v(rec, *p)
    return {
        "dentition": g("dentition") or "permanent",
        "maxilla": g("transverse", "maxilla") or "normal",
        "crossbite": g("transverse", "crossbite") or "none",
        "crossbite_teeth": _v(rec, "transverse", "crossbite", "teeth", default=[]) or [],
        "crossbite_region": (rec.get("transverse", {}).get("crossbite", {}) or {}).get("region", "unstated"),
        "scissor": g("transverse", "scissor_bite"),          # None unless stated
        "scissor_teeth": _v(rec, "transverse", "scissor_bite", "teeth", default=[]) or [],
        "overbite": g("vertical", "overbite") or "normal",
        "molar_r": g("sagittal", "molar_class_right") or "I",
        "molar_r_deg": (rec.get("sagittal", {}).get("molar_class_right", {}) or {}).get("degree", "unstated"),
        "molar_l": g("sagittal", "molar_class_left") or "I",
        "molar_l_deg": (rec.get("sagittal", {}).get("molar_class_left", {}) or {}).get("degree", "unstated"),
        "canine_r": g("sagittal", "canine_class_right") or "I",
        "canine_r_deg": (rec.get("sagittal", {}).get("canine_class_right", {}) or {}).get("degree", "unstated"),
        "canine_l": g("sagittal", "canine_class_left") or "I",
        "canine_l_deg": (rec.get("sagittal", {}).get("canine_class_left", {}) or {}).get("degree", "unstated"),
        "overjet": g("sagittal", "overjet") or "normal",
        "midline_u": g("midline", "upper") or "coincident",
        "midline_l": g("midline", "lower") or "coincident",
        "spee": g("curve_spee") or "normal",
        "spee_arch": (rec.get("curve_spee", {}) or {}).get("arch", "unstated"),
        "wilson": g("curve_wilson") or "normal",
        "wilson_arch": (rec.get("curve_wilson", {}) or {}).get("arch", "unstated"),
        "crowding_u": g("arch_upper", "crowding") or "none",
        "crowding_l": g("arch_lower", "crowding") or "none",
        "spacing_u": g("arch_upper", "spacing") or "none",
        "spacing_l": g("arch_lower", "spacing") or "none",
    }


def chart_state(rec) -> dict:
    """{slot: CHART_CLASSES token} over the core slots + per-tooth findings —
    consumed by train.py's class-frequency counting for --chart-class-weights."""
    c = complete_record(rec)
    out = {
        "dentition": _PLAIN(c["dentition"]),
        "transverse": _PLAIN(c["maxilla"]),
        "xbite": _PLAIN(c["crossbite"]),
        "overbite": _PLAIN(c["overbite"]),
        "molar_r": _CLASS.get(c["molar_r"], "not-assessable"),
        "molar_l": _CLASS.get(c["molar_l"], "not-assessable"),
        "canine_r": _CLASS.get(c["canine_r"], "not-assessable"),
        "canine_l": _CLASS.get(c["canine_l"], "not-assessable"),
        "overjet": _PLAIN(c["overjet"]),
        "midline_u": _MIDLINE.get(c["midline_u"], "not-assessable"),
        "midline_l": _MIDLINE.get(c["midline_l"], "not-assessable"),
        "spee": _PLAIN(c["spee"]),
        "wilson": _PLAIN(c["wilson"]),
        "crowding_u": _PLAIN(c["crowding_u"]),
        "crowding_l": _PLAIN(c["crowding_l"]),
        "spacing_u": _PLAIN(c["spacing_u"]),
        "spacing_l": _PLAIN(c["spacing_l"]),
    }
    for fdi, entry in sorted((rec.get("teeth") or {}).items()):
        for i, f in enumerate(entry.get("findings") or []):
            if f in _TOOTH_FINDING:
                out[f"tooth_{fdi}_{i}"] = _TOOTH_FINDING[f]
    return out


def _side_token(cls, deg):
    tok = _CLASS.get(cls, "not-assessable")
    dtok = _DEGREE.get(deg)
    return f"{tok} {dtok}" if (dtok and tok != "not-assessable") else tok


def chart_block(rec) -> str:
    c = complete_record(rec)
    teeth_bits = []
    for fdi, entry in sorted((rec.get("teeth") or {}).items()):
        toks = [_TOOTH_FINDING[f] for f in (entry.get("findings") or [])
                if f in _TOOTH_FINDING]
        if toks:
            teeth_bits.append(f"{fdi} " + " ".join(toks))
    lines = [
        CHART_HEADER,
        f"dentition: {_PLAIN(c['dentition'])}",
        f"transverse: {_PLAIN(c['maxilla'])} | xbite: {_PLAIN(c['crossbite'])}",
        f"overbite: {_PLAIN(c['overbite'])}",
        f"molar R: {_side_token(c['molar_r'], c['molar_r_deg'])} | "
        f"molar L: {_side_token(c['molar_l'], c['molar_l_deg'])}",
        f"canine R: {_side_token(c['canine_r'], c['canine_r_deg'])} | "
        f"canine L: {_side_token(c['canine_l'], c['canine_l_deg'])}",
        f"overjet: {_PLAIN(c['overjet'])}",
        f"midline U: {_MIDLINE.get(c['midline_u'], 'not-assessable')} | "
        f"midline L: {_MIDLINE.get(c['midline_l'], 'not-assessable')}",
        f"spee: {_PLAIN(c['spee'])} | wilson: {_PLAIN(c['wilson'])}",
        f"crowding U: {_PLAIN(c['crowding_u'])} | crowding L: {_PLAIN(c['crowding_l'])}",
        f"spacing U: {_PLAIN(c['spacing_u'])} | spacing L: {_PLAIN(c['spacing_l'])}",
        "teeth: " + (", ".join(teeth_bits) if teeth_bits else "unremarkable"),
    ]
    return "\n".join(lines)


def concat_target(rec, report: str) -> str:
    return chart_block(rec) + DELIMITER + report


# ── prose ────────────────────────────────────────────────────────────────────────
_PROVENANCE = re.compile(r"rule \d|vote group|dedup|unresolved tie|merge_note|group [AB]\b",
                         re.IGNORECASE)


def _clean(text) -> str | None:
    if not text or _PROVENANCE.search(text):
        return None
    return " ".join(str(text).split())


def _fdi_list(fdis) -> str:
    fdis = sorted(fdis, key=int)
    if len(fdis) == 1:
        return f"tooth {fdis[0]}"
    return "teeth " + ", ".join(fdis[:-1]) + f" and {fdis[-1]}"


_DEG_PROSE = {"full": "full ", "end_to_end": "end-to-end ", "half": "half ",
              "quarter": "quarter ", "three_quarter": "three-quarter ",
              "tendency": "mild "}


def _art(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _side_prose(mcls, mdeg, ccls, cdeg):
    def one(kind, cls, deg):
        if cls == "not_assessable":
            return f"the {kind} relationship cannot be assessed"
        body = f"{_DEG_PROSE.get(deg, '')}Class {cls} {kind} relationship"
        return f"{_art(body)} {body}"
    if mcls == ccls and mdeg == cdeg and mcls != "not_assessable":
        body = f"{_DEG_PROSE.get(mdeg, '')}Class {mcls} molar and canine relationship"
        return f"{_art(body)} {body}"
    return one("molar", mcls, mdeg) + " and " + one("canine", ccls, cdeg)


_OVERJET = {"normal": "overjet within normal limits", "increased": "increased overjet",
            "reduced": "reduced overjet", "zero": "zero overjet",
            "reverse": "reverse overjet"}


def render_prose(rec) -> str:
    c = complete_record(rec)
    s: list[str] = []

    if c["dentition"] == "mixed":
        s.append("The patient is in mixed dentition.")
    elif c["dentition"] == "deciduous":
        s.append("The patient is in deciduous dentition.")

    # transverse + crossbite
    hedge = "appear " if _hedged(rec, "transverse", "maxilla") else ""
    t = {"normal": f"The transverse relationships {hedge}within normal limits"
                   if hedge else "The patient presents correct transverse relationships",
         "constricted": "The patient presents a transverse constriction of the maxilla",
         "expanded": "The patient presents a transverse expansion of the maxilla",
         "not_assessable": "The transverse relationships are not assessable"}[c["maxilla"]]
    if c["crossbite"] == "present":
        t += (f", with a crossbite on {_fdi_list(c['crossbite_teeth'])}"
              if c["crossbite_teeth"] else ", with a crossbite")
    elif c["crossbite"] == "none":
        region = {"posterior": "posterior ", "anterior": "anterior "}.get(
            c["crossbite_region"], "")
        t += f", in the absence of {region}crossbite"
    s.append(t + ".")
    if c["scissor"] == "present":
        s.append("A scissor bite is present"
                 + (f" on {_fdi_list(c['scissor_teeth'])}" if c["scissor_teeth"] else "")
                 + ".")

    # vertical
    ob = c["overbite"]
    hedge = "would appear to be " if _hedged(rec, "vertical", "overbite") else "is "
    s.append({"normal": f"From a vertical standpoint, the overbite {hedge}within normal limits.",
              "increased": f"From a vertical standpoint, the overbite {hedge}increased.",
              "reduced": f"From a vertical standpoint, the overbite {hedge}reduced.",
              "open_bite": "From a vertical standpoint, there is an open bite.",
              "edge_to_edge": "From a vertical standpoint, the incisors are in an "
                              "edge-to-edge relationship.",
              "not_assessable": "From a vertical standpoint, the overbite is not "
                                "assessable."}[ob])

    # sagittal
    right = _side_prose(c["molar_r"], c["molar_r_deg"], c["canine_r"], c["canine_r_deg"])
    left = _side_prose(c["molar_l"], c["molar_l_deg"], c["canine_l"], c["canine_l_deg"])
    oj = c["overjet"]
    oj_txt = "" if oj == "not_assessable" else f", with {_OVERJET[oj]}"
    if right == left:
        s.append(f"Sagittally, there is {right} bilaterally{oj_txt}.")
    else:
        s.append(f"Sagittally, there is {right} on the right, and {left} on the "
                 f"left{oj_txt}.")
    if oj == "not_assessable":
        s.append("The overjet is not assessable.")

    # midlines
    mu, ml = c["midline_u"], c["midline_l"]
    if mu == ml == "coincident":
        s.append("The dental midlines are coincident.")
    elif mu == ml == "deviated":
        s.append("The dental midlines are deviated relative to each other.")
    elif mu == ml == "not_assessable":
        s.append("The dental midlines are not assessable.")
    else:
        side = {"deviated_right": "deviated to the right",
                "deviated_left": "deviated to the left", "deviated": "deviated",
                "coincident": "centered", "not_assessable": "not assessable"}
        s.append(f"The upper midline is {side[mu]}; the lower midline is {side[ml]}.")

    # curves
    cv = {"normal": "within normal limits", "increased": "increased",
          "flat": "flattened", "not_assessable": "not assessable"}
    arch = {"lower": "lower ", "upper": "upper "}
    sp_a, wi_a = arch.get(c["spee_arch"], ""), arch.get(c["wilson_arch"], "")
    if c["spee"] == c["wilson"] and sp_a == wi_a:
        s.append(f"The {sp_a}curves of Spee and Wilson are {cv[c['spee']]}.")
    else:
        s.append(f"The {sp_a}curve of Spee is {cv[c['spee']]}; the {wi_a}curve of "
                 f"Wilson is {cv[c['wilson']]}.")

    # crowding + spacing
    cr = {"none": "no", "mild": "mild", "moderate": "moderate", "severe": "severe",
          "not_assessable": None}
    cu, cl = c["crowding_u"], c["crowding_l"]
    if cu == cl and cr[cu] is not None:
        s.append(f"There is {cr[cu]} crowding in the upper and lower arches."
                 if cu != "none" else "There is no crowding in the upper or lower arch.")
    else:
        def one_arch(v, name):
            if cr.get(v) is None:
                return f"crowding in the {name} arch is not assessable"
            return (f"{cr[v]} crowding in the {name} arch" if v != "none"
                    else f"no crowding in the {name} arch")
        s.append(("There is " + one_arch(cu, "upper") + " and "
                  + one_arch(cl, "lower") + ".").replace(" and no crowding",
                                                         ", and no crowding"))
    for v, name in ((c["spacing_u"], "upper"), (c["spacing_l"], "lower")):
        if v == "present":
            s.append(f"Spacing is present in the {name} arch.")
        elif v == "diastema":
            s.append(f"Diastemas are present in the {name} arch.")

    # per-tooth findings (grouped by finding; crossbite teeth already stated above)
    groups: dict[str, list[str]] = {}
    for fdi, entry in sorted((rec.get("teeth") or {}).items()):
        for f in entry.get("findings") or []:
            if f in ("crossbite", "scissor_bite", "other"):
                continue
            groups.setdefault(f, []).append(fdi)
    tooth_txt = {
        "absent": "{} {} absent",
        "unerupted": "{} {} not yet erupted",
        "erupting": "{} {} not yet fully erupted",
        "impacted": "{} {} impacted",
        "retained_deciduous": "{} {} retained deciduous",
        "fractured": "{} {} fractured",
        "rotated": "{} {} rotated",
        "displaced": "{} {} displaced",
        "infra_occlusion": "{} {} in infra-occlusion",
    }
    for f, fdis in groups.items():
        lst = _fdi_list(fdis)
        verb = "is" if len(fdis) == 1 else "are"
        if f in tooth_txt:
            s.append((tooth_txt[f].format(lst, verb)).capitalize() + ".")
        elif f == "sealed":
            s.append(f"The occlusal fissures of {lst} have been sealed.")
        elif f == "caries_suspect":
            s.append(f"The fissures of {lst} are pigmented and should be assessed "
                     f"for caries risk.")
        elif f == "restored":
            s.append(f"{lst.capitalize()} present{'s' if len(fdis) == 1 else ''} "
                     f"restorations.")
        elif f == "prosthetic_crown":
            s.append(f"{lst.capitalize()} carr{'ies' if len(fdis) == 1 else 'y'} "
                     f"prosthetic crowns." if len(fdis) > 1 else
                     f"{lst.capitalize()} carries a prosthetic crown.")
        elif f == "worn":
            s.append(f"The incisal edges of {lst} are worn.")

    # appliances / soft tissue / other findings (whitelisted free text only)
    for a in rec.get("appliances") or []:
        v = _clean(a.get("v"))
        if v:
            s.append(f"There is a {v[0].lower() + v[1:]}."
                     if not v.endswith(".") else f"There is a {v[0].lower() + v[1:]}")
    st = rec.get("soft_tissue") or {}
    if _v(st, "plaque") == "present":
        s.append("Visible plaque is noted.")
    if _v(st, "staining") == "present":
        s.append("Extrinsic staining is noted.")
    gi = _v(st, "gingival_inflammation")
    if gi == "present":
        s.append("There appear to be signs of gingival inflammation."
                 if _hedged(st, "gingival_inflammation")
                 else "There are signs of gingival inflammation.")
    elif gi == "none":
        s.append("No clear signs of gingival inflammation are observed.")
    rec_v = _v(st, "recession")
    if rec_v == "present":
        teeth = (st.get("recession") or {}).get("teeth") or []
        s.append("Gingival recessions are present"
                 + (f" on {_fdi_list(teeth)}" if teeth else "") + ".")
    hy = _v(st, "hygiene")
    if hy in ("good", "fair", "poor"):
        s.append(f"Oral hygiene is {hy}.")
    ot = _clean(_v(st, "other"))
    if ot:
        s.append(ot[0].upper() + ot[1:] + ("" if ot.endswith(".") else "."))
    for o in rec.get("other_findings") or []:
        v = _clean(o.get("v"))
        if v:
            s.append(v[0].upper() + v[1:] + ("" if v.endswith(".") else "."))

    return " ".join(s)
