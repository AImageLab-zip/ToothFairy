"""Deterministic canonical-prose renderer: merged record -> report text.

Uses the corpus's own section layout
(Mandible / Maxilla / Maxillary sinuses / Mandibular condyles) and its common
sentence forms, so captioning score is not paid for invented wording. Fully
deterministic: same record -> same prose, byte for byte.

every free-text field is passed through ``_clean`` which
strips the ``(rN)`` annotator-provenance markers. Previously the ``r1/r2/r3``
substring guard *dropped* whole tooth/atrophy/osteolytic notes that carried a
marker (189 findings deleted), while ``other_findings`` / ``prostheses`` /
``ian_canal.*.relationship`` had no guard at all and leaked ``(rN)`` into 45
train targets. Stripping-not-dropping fixes both with one helper. Also emits the
previously-silent condyle (``not_assessable`` / ``included``), sinus
(``opacified``) and bare-present-tooth notes, and files upper-arch prostheses
under Maxilla.
"""
from __future__ import annotations

import re

_RMARK = re.compile(r"\s*\(r\d+\)")


def _clean(s):
    """Strip ``(rN)`` provenance markers and collapse whitespace. No-op on clean text."""
    if not s:
        return s
    s = _RMARK.sub("", s)
    return re.sub(r"  +", " ", s).strip()


FOV_TXT = {
    "included": "included in the scan volume",
    "partial": "partially included in the acquisition",
    "excluded": "not included in the scan volume",
    "not_assessable": "partially included in the acquisition and not assessable",
    "not_mentioned": None,
}
COURSE_TXT = {
    "regular": "a regular course", "lingual": "a predominantly lingual course",
    "apico_lingual": "a regular course in an apico-lingual position",
    "buccal": "a predominantly buccal course",
    "not_assessable": "a course that cannot be fully assessed", "not_mentioned": None,
}
RESTOR_TXT = {
    "filling": "a conservative restoration", "prosthetic_crown": "a prosthetic crown",
    "post_and_core": "a post-and-core", "bridge_abutment": "a bridge abutment",
    "prosthetic_pontic": "a prosthetic pontic", "implant": "an endosseous implant",
    "implant_crown": "a prosthetic crown on the implant",
}
ENDO_TXT = {"adequate": "adequate endodontic treatment",
            "inadequate": "inadequate endodontic treatment",
            "present": "endodontic treatment"}
LESION_TXT = {
    "periapical_granuloma": "a periapical granuloma", "periapical_cyst": "a periapical cyst",
    "periapical_lesion": "a periapical lesion", "root_fracture": "a root fracture",
    "caries": "a carious lesion", "peri_implant_resorption": "peri-implant bone resorption",
    "root_remnant": "a root remnant", "other": "a further finding",
}


def _tooth_clause(fdi, t):
    parts = []
    st = t["status"]
    note = _clean(t.get("note", ""))
    if st == "absent":
        return f"tooth {fdi} is absent"
    if st == "root_remnant":
        parts.append(f"a root remnant of tooth {fdi}")
    elif st == "impacted":
        imp = t.get("impaction") or {}
        ori = imp.get("orientation")
        parts.append(f"tooth {fdi} is impacted" + (f", {ori}" if ori and ori != "impacted" else ""))
    else:
        parts.append(f"tooth {fdi}")
    extra = []
    if t.get("endo"): extra.append(ENDO_TXT.get(t["endo"], "endodontic treatment"))
    for r in t.get("restorations", []): extra.append(RESTOR_TXT.get(r, r))
    if t.get("lesion"): extra.append(LESION_TXT.get(t["lesion"], t["lesion"]))
    head = parts[0]
    if extra:
        return head + " with " + ", ".join(extra)
    if st == "present":
        # a bare present tooth adds nothing UNLESS it carries a note (227 findings).
        if note:
            return f"tooth {fdi}"
        return None
    return head


def _teeth_sentences(teeth):
    # group absents together, then per-tooth findings in FDI order
    order = sorted(teeth.keys(), key=lambda k: int(k))
    absents = [k for k in order if teeth[k]["status"] == "absent" and not (teeth[k].get("restorations") or teeth[k].get("endo") or teeth[k].get("lesion"))]
    out = []
    if absents:
        out.append("Absence of teeth " + ", ".join(absents) + ".")
    for k in order:
        t = teeth[k]
        if k in absents:
            continue
        c = _tooth_clause(k, t)
        if c:
            s = c[0].upper() + c[1:] + "."
            note = _clean(t.get("note", ""))
            if note:
                s = s[:-1] + f" ({note})."
            out.append(s)
    return out


def _canal_sentence(canal):
    L = canal.get("left", {}) or {}
    R = canal.get("right", {}) or {}
    lc, rc = COURSE_TXT.get(L.get("course", "not_mentioned")), COURSE_TXT.get(R.get("course", "not_mentioned"))
    if not lc and not rc:
        return None
    if lc == rc:
        s = f"The mandibular canal shows {lc or rc} bilaterally"
    else:
        segs = []
        if rc: segs.append(f"the right mandibular canal shows {rc}")
        if lc: segs.append(f"the left mandibular canal shows {lc}")
        s = "; ".join(segs).capitalize()
    rels = []
    rrel, lrel = _clean(R.get("relationship", "")), _clean(L.get("relationship", ""))
    if rrel: rels.append(f"on the right {rrel}")
    if lrel: rels.append(f"on the left {lrel}")
    if rels:
        s += ", " + "; ".join(rels)
    return s + "."


def _prosthesis_sentence(pr):
    span = "-".join(pr["teeth"]) if pr.get("teeth") else ""
    note = _clean(pr.get("note", ""))
    return (f"{pr['type'].capitalize()} {span}: {note}").strip().rstrip(":") + "."


def render(rec) -> str:
    lines = []
    fov = rec["fov"]
    teeth = rec["teeth"]
    prostheses = rec.get("prostheses", [])
    # file a prosthesis under Maxilla iff every tooth it spans is upper (quadrant 1-2).
    upper_pros = [p for p in prostheses if p.get("teeth") and all(t[0] in "12" for t in p["teeth"])]
    lower_pros = [p for p in prostheses if p not in upper_pros]

    # ---- Mandible block ----
    mand = ["Mandible:"]
    m_stat = fov.get("mandible", {}).get("status", "not_mentioned")
    if m_stat == "partial":
        mand.append("The mandible is partially included in the acquisition.")
    lower = {k: v for k, v in teeth.items() if k[0] in "34"}
    mand += _teeth_sentences(lower)
    bp = rec.get("bone_and_perio", {})
    if bp.get("atrophy", {}).get("present"):
        an = _clean(bp["atrophy"].get("note", ""))
        mand.append(f"Atrophy is noted ({an})." if an else "Atrophy is noted.")
    if bp.get("periodontitis"):
        p = bp["periodontitis"]
        mand.append(f"Periodontal disease ({p.get('stage','')}, {p.get('distribution','')}).".replace(" ,", ""))
    if bp.get("post_extraction_sockets"):
        mand.append("Post-extraction sockets at " + ", ".join(bp["post_extraction_sockets"]) + ".")
    ol = bp.get("osteolytic_lesions", {})
    if ol.get("status") == "none":
        mand.append("No definite osteolytic or osteosclerotic lesions.")
    elif ol.get("status") == "present":
        on = _clean(ol.get("note", ""))
        mand.append(f"An osteolytic lesion is present ({on})." if on else "An osteolytic lesion is present.")
    cs = _canal_sentence(rec.get("ian_canal", {}))
    if cs:
        mand.append(cs)
    for pr in lower_pros:
        mand.append(_prosthesis_sentence(pr))
    for o in rec.get("other_findings", []):
        txt = _clean(o["text"])
        if txt:
            mand.append(txt[0].upper() + txt[1:] + ".")
    lines.append("\n".join(mand))

    # ---- Maxilla block ----
    mx_stat = fov.get("maxilla", {}).get("status", "not_mentioned")
    mx = ["Maxilla:"]
    if FOV_TXT.get(mx_stat):
        mx.append("The maxilla is " + FOV_TXT[mx_stat] + ".")
    upper = {k: v for k, v in teeth.items() if k[0] in "12"}
    if upper:
        mx += _teeth_sentences(upper)
    for pr in upper_pros:
        mx.append(_prosthesis_sentence(pr))
    sin = rec.get("sinuses", {})
    if sin.get("status") == "clear":
        mx.append("The maxillary sinuses are normally pneumatized.")
    elif sin.get("status") == "mucosal_thickening":
        mx.append("Mucosal thickening of the maxillary sinuses (" + _clean(sin.get("note", "")) + ").")
    elif sin.get("status") == "opacified":
        sn = _clean(sin.get("note", ""))
        mx.append(f"Opacification of the maxillary sinuses ({sn})." if sn else "Opacification of the maxillary sinuses.")
    elif sin.get("status") in ("not_assessable", "excluded"):
        mx.append("The maxillary sinuses are not included in the acquisition and not assessable.")
    if len(mx) > 1:
        lines.append("\n".join(mx))

    # ---- Condyles ----
    cond = fov.get("mandibular_condyles", {}).get("status", "not_mentioned")
    if cond == "excluded":
        lines.append("Mandibular condyles: not included in the scan volume.")
    elif cond == "partial":
        lines.append("Mandibular condyles: only partially included in the scan volume.")
    elif cond == "not_assessable":
        lines.append("Mandibular condyles: partially included in the acquisition and not assessable.")
    elif cond == "included":
        lines.append("Mandibular condyles: included in the scan volume.")

    return "\n\n".join(lines)


if __name__ == "__main__":
    import json, sys
    rec = json.load(open(sys.argv[1]))
    print(render(rec))
