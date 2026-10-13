#!/usr/bin/env python3
"""Stage-3 (template): render a CBCT report from Stage-2 evidence — deterministic,
faithful, and complete, in the register of the ToothFairy4 `reports_en`.

Design: the *template* decides WHAT to say (every predicted finding, nothing invented)
and organizes it like the reference reports (Mandible / Maxilla / Mandibular condyles,
the structure used by ~90% of cases — the P/F/S subsets). An optional LLM stylizer
(stylize_llm.py) then only rewrites the phrasing — it never adds or drops findings.

Input: the per-case evidence JSON produced by merge_stage2_branch_val_results.py
(`coverage`, `facts`, `merged_missing_teeth`, `warnings`). Output: report text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

FDI_ORDER = [f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)]


def _upper(t: str) -> bool:
    return str(t)[:1] in ("1", "2")


def _order(teeth: List[str]) -> List[str]:
    seen, out = set(), []
    for t in sorted(set(str(x) for x in teeth), key=lambda x: FDI_ORDER.index(x) if x in FDI_ORDER else 99):
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def _saddle_spans(present: List[str], missing: List[str]) -> List[str]:
    """Intercalated edentulous saddles: within each quadrant, a maximal run of missing teeth
    flanked on BOTH sides by present teeth -> "A-B" (the bounding present teeth). Derivable
    from present/missing sets; matches the reference phrasing "edentulous saddles intercalated
    between 33-37, 45-47, ...". Free-end (distal) gaps are excluded (no distal flank)."""
    pres, miss = set(present), set(missing)
    spans = []
    for q in "1234":
        seq = [q + str(i) for i in range(1, 9)]  # central incisor -> third molar
        idx_present = [j for j, t in enumerate(seq) if t in pres]
        for a, b in zip(idx_present, idx_present[1:]):
            gap = seq[a + 1:b]
            if gap and all(t in miss for t in gap):
                spans.append(f"{seq[a]}-{seq[b]}")
    return spans


def _fmt_tooth(t: str, dotted: bool) -> str:
    return f"{t[0]}.{t[1]}" if dotted and len(t) == 2 else t


def _teeth_phrase(teeth: List[str], dotted: bool = False) -> str:
    teeth = [_fmt_tooth(t, dotted) for t in _order(teeth)]
    if len(teeth) == 1:
        return teeth[0]
    return ", ".join(teeth[:-1]) + " and " + teeth[-1]


# Per-tooth finding -> (singular verb phrase, plural verb phrase). "{t}" = teeth phrase.
TOOTH_SENTENCE = {
    "implant":            ("Endosseous implant in position {t}, apparently osseointegrated.",
                           "Endosseous implants in positions {t}, apparently osseointegrated."),
    # v2: crown / restoration split. "prosthetic crown" (377), "coronal restoration" (99).
    "crown":              ("Prosthetic crown on tooth {t}.",
                           "Prosthetic crowns on teeth {t}."),
    "restoration":        ("Coronal restoration of tooth {t}.",
                           "Coronal restorations of teeth {t}."),
    # Phrasing aligned to the dominant report form (corpus counts in parentheses):
    # "endodontically treated" (553) >> "endodontic treatment of" (141).
    "endodontic_treatment": ("Tooth {t} endodontically treated.",
                             "Teeth {t} endodontically treated."),
    "prosthesis":         ("Prosthetic (bridge/pontic) rehabilitation involving tooth {t}.",
                           "Prosthetic (bridge/pontic) rehabilitation involving teeth {t}."),
    "impaction":          ("Impacted tooth {t}.", "Impacted teeth {t}."),   # "impacted" (386)
    "caries":             ("Caries on tooth {t}.", "Caries on teeth {t}."),
    "root_remnant":       ("Root remnant of tooth {t}.", "Root remnants of teeth {t}."),
    "root_fracture":      ("Root fracture of tooth {t}.", "Root fractures of teeth {t}."),
    # "associated with tooth" never occurs in the corpus; use the plain periapical form.
    "lesion":             ("Periapical lesion of tooth {t}.",
                           "Periapical lesions of teeth {t}."),
}
# Order in which per-tooth findings are stated within an arch.
TOOTH_ORDER = ["implant", "prosthesis", "crown", "restoration", "endodontic_treatment",
               "root_remnant", "root_fracture", "caries", "impaction", "lesion"]


def _tooth_sentences(by_type: Dict[str, List[str]], arch_upper: bool, dotted: bool = False) -> List[str]:
    out = []
    for typ in TOOTH_ORDER:
        teeth = [t for t in by_type.get(typ, []) if _upper(t) == arch_upper]
        if not teeth:
            continue
        sing, plur = TOOTH_SENTENCE[typ]
        tp = _teeth_phrase(teeth, dotted)
        out.append((sing if len(_order(teeth)) == 1 else plur).format(t=tp))
    return out


def _canal_sentences(canal: Dict[str, Any], findings_only: bool = False, dotted: bool = False,
                     drop_close: bool = False) -> List[str]:
    out = []
    for side in ("right", "left"):
        pred = canal.get(f"{side}_pred")  # 'lingual' | 'buccal' | None
        # "mandibular canal with a ... course" (625) >> "canal follows a ... course" (39).
        if pred:
            out.append(f"{side.capitalize()} mandibular canal with a regular course, predominantly {pred}.")
        elif not findings_only:  # a bare "regular course" is a normal statement -> drop when findings_only
            out.append(f"{side.capitalize()} mandibular canal with a regular course.")
    # The "close relationship with tooth X" / "lesion close" sub-claims are over-specific: 39%
    # of canal precision misses; references report the canal course but usually NOT this relation.
    # drop_close removes them (reference-blind precision filter).
    if not drop_close:
        close = _order(canal.get("close_teeth", []))
        if close:
            out.append(f"The mandibular canal is in close relationship with tooth {_teeth_phrase(close, dotted)}.")
        if canal.get("lesion_close"):
            out.append("A lesion is in close relationship with the mandibular canal.")
    return out


def _preamble(coverage: Dict[str, str]) -> str:
    """Acquisition-volume preamble, derived deterministically from coverage. Matches the
    reference family "CT including within the acquisition volume the mandible ...", which is
    ~13% of reference sentences and something the current template emits nothing for.
    coverage holds only *positive* coverage predictions (a structure absent from the dict is
    fully included), so: condyles/sinuses present-in-dict => not_included; maxilla present =>
    partial or excluded; maxilla absent => fully included."""
    parts = ["CT including within the acquisition volume the mandible"]
    if coverage.get("mandibular_condyles") == "not_included":
        parts.append(" excluding the condylar and coronoid processes bilaterally")
    max_status = coverage.get("maxilla")
    if max_status == "not_included":
        pass  # maxilla out of the field -> do not claim it
    elif max_status == "partially_included":
        parts.append(", and the maxilla partially")
    else:
        parts.append(", and the maxilla")
        if coverage.get("maxillary_sinuses") != "not_included":
            parts.append(" including the caudal portion of the maxillary sinuses bilaterally")
    return "".join(parts) + "."


_QUAD_ORD = {"1": "first", "2": "second", "3": "third", "4": "fourth"}


def _absence_lines(missing_arch: List[str], arch: str, group_edentulism: bool,
                   group_regions: bool = False, dotted: bool = False) -> List[str]:
    """Absence statement(s) for one arch. A near-complete arch (>=14/16 absent) collapses to
    "Complete edentulism of the ...". With ``group_regions`` a whole quadrant or a fully-absent
    premolar-molar region is stated by anatomy ("Edentulism of the premolar and molar region of
    the third quadrant") — the way references phrase it — with only the remaining scattered teeth
    enumerated. Both are derivable from the missing-teeth set. Returns a list (may be several)."""
    if group_edentulism and len(missing_arch) >= 14:
        return [f"Complete edentulism of the {arch}."]
    if not missing_arch:
        return []
    if not group_regions:
        return [f"Absence of teeth {_teeth_phrase(missing_arch, dotted)}."]
    miss = set(missing_arch)
    quads = ["1", "2"] if arch == "maxilla" else ["3", "4"]
    lines: List[str] = []
    leftover: List[str] = []
    for q in quads:
        pos = {int(t[1]) for t in miss if t[:1] == q and len(t) == 2}
        if not pos:
            continue
        if pos >= {1, 2, 3, 4, 5, 6, 7, 8}:
            lines.append(f"Edentulism of the {_QUAD_ORD[q]} quadrant.")
        elif {4, 5, 6, 7, 8} <= pos:   # entire premolar-molar region missing
            lines.append(f"Edentulism of the premolar and molar region of the {_QUAD_ORD[q]} quadrant.")
            leftover += [t for t in miss if t[:1] == q and int(t[1]) in {1, 2, 3}]
        else:
            leftover += [t for t in miss if t[:1] == q]
    if leftover:
        lines.append(f"Absence of teeth {_teeth_phrase(leftover, dotted)}.")
    return lines


def render_report(case: Dict[str, Any], min_prob: float = 0.0, findings_only: bool = False,
                  add_preamble: bool = False, add_present_teeth: bool = False,
                  group_edentulism: bool = True, dotted_fdi: bool = False,
                  drop_absent_third_molars: bool = False, add_saddle_atrophy: bool = False,
                  group_absences_by_region: bool = False, drop_canal_close: bool = False) -> str:
    """Render structured evidence into the deterministic challenge report.

    Findings below ``min_prob`` are omitted; findings without an associated
    probability, such as mask-derived missing teeth, are retained.
    ``findings_only`` suppresses normal and coverage boilerplate."""
    coverage = {c["structure"]: c["status"] for c in case.get("coverage", [])
                if c.get("probability", 1.0) >= min_prob}
    # Mask-present teeth that are not (predicted/known) missing -> "teeth X are present."
    merged_missing = set(str(t) for t in case.get("merged_missing_teeth", []))
    tooth_teeth = ((case.get("predictions") or {}).get("tooth") or {}).get("teeth") or {}
    present_teeth = [str(t) for t in tooth_teeth if str(t) not in merged_missing]

    globals_ = set()
    atrophy_arches: set = set()   # v2: {'maxilla','mandible'} (or 'unspecified')
    atrophy_severe = False        # v2
    lesion_lytic = lesion_sclerotic = False  # v2: osteolytic / osteosclerotic
    by_type: Dict[str, List[str]] = defaultdict(list)
    canal: Dict[str, Any] = {"close_teeth": []}
    for f in case.get("facts", []):
        if f.get("probability") is not None and f["probability"] < min_prob:
            continue
        typ = f.get("type"); teeth = list(f.get("teeth") or [])
        if typ in ("periodontitis", "sinus_pathology"):
            globals_.add(typ)
        elif typ == "atrophy":
            reg = f.get("region") or f.get("target")
            atrophy_arches.add(reg if reg in ("maxilla", "mandible") else "unspecified")
        elif typ == "atrophy_severe":
            atrophy_severe = True
        elif typ == "osteolytic_lesion":
            lesion_lytic = True
        elif typ == "osteosclerotic_lesion":
            lesion_sclerotic = True
        elif typ == "canal_relationship":
            fnd = str(f.get("finding", ""))
            if "predominantly lingual" in fnd or "predominantly buccal" in fnd:
                side = "left" if "left" in fnd else "right"
                canal[f"{side}_pred"] = "lingual" if "lingual" in fnd else "buccal"
            elif "lesion close" in fnd:
                canal["lesion_close"] = True
            else:
                canal["close_teeth"].extend(teeth)
        elif typ == "missing_tooth":
            by_type["missing"].extend(teeth)
        elif teeth:
            by_type[typ].extend(teeth)

    # A crown is the more specific statement; a crowned tooth is trivially also "restored",
    # so drop restoration on teeth already stated as crowned to avoid a redundant phrase.
    if by_type.get("crown"):
        crowned = set(by_type["crown"])
        by_type["restoration"] = [t for t in by_type.get("restoration", []) if t not in crowned]

    # Radiologists routinely omit absent third molars (18/28/38/48) as an expected finding;
    # our Stage-1 mask flags them, so reporting their absence mostly hits non-entailed candidates.
    if drop_absent_third_molars:
        by_type["missing"] = [t for t in by_type.get("missing", []) if not t.endswith("8")]

    # ---- Mandible ----
    mand: List[str] = []
    mand += _canal_sentences(canal, findings_only, dotted_fdi, drop_canal_close)
    if atrophy_arches:
        names = [n for a, n in (("maxilla", "maxillary"), ("mandible", "mandibular")) if a in atrophy_arches]
        who = " and ".join(names).capitalize() if names else "Alveolar"
        head = f"Severe {who.lower()} atrophy" if atrophy_severe else f"{who} atrophy"
        spans = _saddle_spans(present_teeth, by_type.get("missing", [])) if add_saddle_atrophy else []
        where = (f" at the level of the edentulous saddles intercalated between {_teeth_phrase(spans, dotted_fdi)}"
                 if spans else " of the alveolar processes in the edentulous regions")
        mand.append(f"{head}{where}.")
    pres_l = [t for t in present_teeth if not _upper(t)]
    if add_present_teeth and pres_l:
        mand.append(f"Teeth {_teeth_phrase(pres_l, dotted_fdi)} are present.")
    miss_l = _order([t for t in by_type.get("missing", []) if not _upper(t)])
    if miss_l:
        mand += _absence_lines(miss_l, "mandible", group_edentulism, group_absences_by_region, dotted_fdi)
    mand += _tooth_sentences(by_type, arch_upper=False, dotted=dotted_fdi)
    if "periodontitis" in globals_:
        # "periodontal disease/periodontopathy" (116) >> "chronic periodontitis" (2).
        mand.append("Periodontal disease involving the present teeth.")
    if lesion_lytic and lesion_sclerotic:
        mand.append("Areas of osteolysis and osteocondensation are present.")
    elif lesion_lytic:
        mand.append("Areas of osteolysis are present.")
    elif lesion_sclerotic:
        mand.append("Areas of osteocondensation are present.")
    elif not findings_only:
        # "osteocondensing" (163) >> "osteosclerotic" (38).
        mand.append("No definite osteolytic or osteocondensing lesions.")

    # ---- Maxilla ----
    maxil: List[str] = []
    max_status = coverage.get("maxilla")
    if max_status == "not_included":
        if not findings_only:
            maxil.append("Not included in the acquisition.")
    else:
        if max_status == "partially_included" and not findings_only:
            maxil.append("Partially included in the acquisition.")
        pres_u = [t for t in present_teeth if _upper(t)]
        if add_present_teeth and pres_u:
            maxil.append(f"Teeth {_teeth_phrase(pres_u, dotted_fdi)} are present.")
        miss_u = _order([t for t in by_type.get("missing", []) if _upper(t)])
        if miss_u:
            maxil += _absence_lines(miss_u, "maxilla", group_edentulism, group_absences_by_region, dotted_fdi)
        maxil += _tooth_sentences(by_type, arch_upper=True, dotted=dotted_fdi)
        if coverage.get("maxillary_sinuses") == "not_included":
            if not findings_only:
                maxil.append("Maxillary sinuses not included in the acquisition.")
        elif "sinus_pathology" in globals_:
            maxil.append("Maxillary sinus pathology is present.")
        elif not findings_only:
            maxil.append("Maxillary sinuses normally pneumatized.")

    # ---- Mandibular condyles ----
    cond = ("Excluded from the acquisition and not visible."
            if coverage.get("mandibular_condyles") == "not_included"
            else "Included in the acquisition.")

    head = [_preamble(coverage), ""] if add_preamble else []

    if findings_only:
        # Drop empty sections and their headers; references are flowing prose, not sectioned.
        blocks = []
        if mand:
            blocks += ["Mandible:", *mand]
        if maxil:
            blocks += (["", "Maxilla:"] if blocks else ["Maxilla:"]) + maxil
        body = blocks if blocks else ["No significant findings."]
        return "\n".join(head + body).strip() + "\n"

    lines = head + ["Mandible:", *mand, "", "Maxilla:", *maxil, "", "Mandibular condyles:", cond]
    return "\n".join(lines).strip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render Stage-2 evidence into a CBCT report (template)")
    ap.add_argument("--merged-json-dir", help="Directory of per-case evidence JSONs (from merge)")
    ap.add_argument("--case-json", help="A single evidence JSON to render (prints to stdout)")
    ap.add_argument("--out-csv", help="Write generated_reports.csv (report_id, generated_report)")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="Drop findings below this probability (precision-leaning rendering)")
    ap.add_argument("--findings-only", action="store_true",
                    help="Emit only positive findings; drop normal/coverage boilerplate")
    ap.add_argument("--add-preamble", action="store_true",
                    help="Prepend a coverage-derived acquisition-volume preamble sentence")
    ap.add_argument("--add-present-teeth", action="store_true",
                    help="Enumerate present teeth per arch (\"Teeth X are present.\")")
    ap.add_argument("--no-group-edentulism", action="store_true",
                    help="Disable collapsing a near-complete arch to \"Complete edentulism\"")
    ap.add_argument("--dotted-fdi", action="store_true",
                    help="Render tooth numbers in dotted FDI notation (e.g. 3.8 instead of 38)")
    ap.add_argument("--drop-absent-third-molars", action="store_true",
                    help="Do not report absence of third molars (18/28/38/48)")
    ap.add_argument("--add-saddle-atrophy", action="store_true",
                    help="Localize atrophy to intercalated edentulous saddles (\"between 33-37, ...\")")
    ap.add_argument("--group-absences-by-region", action="store_true",
                    help="State whole-quadrant / premolar-molar-region edentulism by anatomy")
    ap.add_argument("--drop-canal-close", action="store_true",
                    help="Drop the over-specific canal 'close relationship with tooth X' sub-claims")
    args = ap.parse_args(argv)
    common_kw = dict(min_prob=args.min_prob, findings_only=args.findings_only,
                     add_preamble=args.add_preamble, add_present_teeth=args.add_present_teeth,
                     group_edentulism=not args.no_group_edentulism, dotted_fdi=args.dotted_fdi,
                     drop_absent_third_molars=args.drop_absent_third_molars,
                     add_saddle_atrophy=args.add_saddle_atrophy,
                     group_absences_by_region=args.group_absences_by_region,
                     drop_canal_close=args.drop_canal_close)

    if args.case_json:
        print(render_report(json.loads(Path(args.case_json).read_text()), **common_kw))
        return 0
    if not args.merged_json_dir:
        ap.error("provide --case-json or --merged-json-dir")

    rows = []
    for jf in sorted(Path(args.merged_json_dir).glob("*.json")):
        case = json.loads(jf.read_text())
        rows.append({"report_id": case.get("report_id", jf.stem),
                     "case_id": case.get("case_id", jf.stem),
                     "generated_report": render_report(case, **common_kw)})
    if args.out_csv:
        import pandas as pd
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"wrote {len(rows)} reports -> {args.out_csv}")
    else:
        print(json.dumps(rows[:2], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
