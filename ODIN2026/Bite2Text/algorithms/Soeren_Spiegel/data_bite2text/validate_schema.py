#!/usr/bin/env python3
"""Schema validator for the Bite2Text chart extraction (schema v1.2).

Checks every record in data_bite2text/raw_cases/*.json against the spec in
data_bite2text/SCHEMA.md:

  - file parses as JSON
  - case_id matches the filename
  - enum values are legal per the spec (v-values, degree, region, arch,
    tooth findings), including "not_assessable" wherever the schema's Global
    Conventions section makes it first-class even if a given inline doc line
    doesn't spell it out
  - "hedged" is boolean wherever it appears
  - FDI codes are valid (teeth object keys, and teeth arrays under
    crossbite/scissor_bite/recession)
  - every recorded field object carrying "v" also carries a non-empty "src"
  - every src entry has the shape "ios:<rid>"/"photo:<rid>" and that rid is
    present in the record's sources.<modality> list
  - no empty arrays anywhere (sources.ios/sources.photo excepted: those are
    a real report-count registry, not a "recorded field", so an empty list
    there just means the case has zero reports of that modality)
  - merge_notes entries carry a field path and a values dict (plus a
    best-effort check that merge_notes value keys are legal enum members
    for the field they annotate, and that their src lists resolve)

stdlib only. Usage: python3 validate_schema.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw_cases"
CASE_LIST_PATH = HERE / "case_list.txt"

# --------------------------------------------------------------------------
# Enum tables (from SCHEMA.md)
# --------------------------------------------------------------------------

FDI_PERMANENT = {f"{q}{t}" for q in "1234" for t in "12345678"}
FDI_DECIDUOUS = {f"{q}{t}" for q in "5678" for t in "12345"}
VALID_FDI = FDI_PERMANENT | FDI_DECIDUOUS

DEGREE_ENUM = {"full", "end_to_end", "half", "quarter", "three_quarter", "tendency", "unstated"}
REGION_ENUM = {"right", "left", "bilateral", "anterior", "posterior", "unstated"}
ARCH_ENUM = {"upper", "lower", "both", "unstated"}
FINDINGS_ENUM = {
    "absent", "unerupted", "erupting", "impacted", "retained_deciduous", "rotated",
    "displaced", "infra_occlusion", "crossbite", "scissor_bite", "sealed",
    "caries_suspect", "restored", "prosthetic_crown", "worn", "fractured", "other",
}

# dotted-path -> legal "v" values.
# Per SCHEMA.md Global conventions: "not_assessable" is first-class in EVERY
# enum in the Record section, even where an inline doc line is abbreviated
# ("same") or doesn't spell it out. transverse.crossbite's inline doc line
# literally only shows "none|present" yet the corpus itself uses
# not_assessable there -- confirming the global rule overrides the
# abbreviated per-field line, not the other way around.
V_ENUMS = {
    "dentition": {"permanent", "mixed", "deciduous", "not_assessable"},
    "transverse.maxilla": {"normal", "constricted", "expanded", "not_assessable"},
    "transverse.crossbite": {"none", "present", "not_assessable"},
    "transverse.scissor_bite": {"none", "present", "not_assessable"},
    "vertical.overbite": {"normal", "increased", "reduced", "open_bite", "edge_to_edge", "not_assessable"},
    "sagittal.molar_class_right": {"I", "II", "III", "not_assessable"},
    "sagittal.molar_class_left": {"I", "II", "III", "not_assessable"},
    "sagittal.canine_class_right": {"I", "II", "III", "not_assessable"},
    "sagittal.canine_class_left": {"I", "II", "III", "not_assessable"},
    "sagittal.overjet": {"normal", "increased", "reduced", "zero", "reverse", "not_assessable"},
    "midline.upper": {"coincident", "deviated_right", "deviated_left", "deviated", "not_assessable"},
    "midline.lower": {"coincident", "deviated_right", "deviated_left", "deviated", "not_assessable"},
    "curve_spee": {"normal", "increased", "flat", "not_assessable"},
    "curve_wilson": {"normal", "increased", "flat", "not_assessable"},
    "arch_upper.crowding": {"none", "mild", "moderate", "severe", "not_assessable"},
    "arch_upper.spacing": {"none", "present", "diastema", "not_assessable"},
    "arch_lower.crowding": {"none", "mild", "moderate", "severe", "not_assessable"},
    "arch_lower.spacing": {"none", "present", "diastema", "not_assessable"},
    "soft_tissue.plaque": {"none", "present", "not_assessable"},
    "soft_tissue.staining": {"none", "present", "not_assessable"},
    "soft_tissue.gingival_inflammation": {"none", "present", "not_assessable"},
    "soft_tissue.recession": {"none", "present", "not_assessable"},
    "soft_tissue.hygiene": {"good", "fair", "poor", "not_assessable"},
    # soft_tissue.other is free text -- no enum, but still checked for src.
}

DEGREE_FIELDS = {
    "sagittal.molar_class_right", "sagittal.molar_class_left",
    "sagittal.canine_class_right", "sagittal.canine_class_left",
}
ARCH_FIELDS = {"curve_spee", "curve_wilson"}
# Fields whose sibling "teeth" array holds FDI codes.
FDI_ARRAY_HOLDERS = {"transverse.crossbite", "transverse.scissor_bite", "soft_tissue.recession"}

SRC_RE = re.compile(r"^(ios|photo):(.+)$")

# Field paths shown for the per-field stated-rate coverage table.
CORE_FIELDS = [
    "dentition",
    "transverse.maxilla", "transverse.crossbite", "transverse.scissor_bite",
    "vertical.overbite",
    "sagittal.molar_class_right", "sagittal.molar_class_left",
    "sagittal.canine_class_right", "sagittal.canine_class_left", "sagittal.overjet",
    "midline.upper", "midline.lower",
    "curve_spee", "curve_wilson",
    "arch_upper.crowding", "arch_upper.spacing", "arch_lower.crowding", "arch_lower.spacing",
    "soft_tissue.plaque", "soft_tissue.staining", "soft_tissue.gingival_inflammation",
    "soft_tissue.recession", "soft_tissue.hygiene", "soft_tissue.other",
    "teeth", "appliances", "other_findings", "merge_notes",
]

# Keys that never get followed by the "find every 'v'-bearing dict" walker
# because they're payload, not nesting.
_LEAF_KEYS = {"v", "src", "note", "contested", "hedged", "degree", "region", "arch", "teeth"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def get_path(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


class Violations:
    def __init__(self):
        self.counts = Counter()
        self.rows = defaultdict(list)  # check -> list[(file, field, detail)]

    def add(self, check, file_id, field, detail):
        self.counts[check] += 1
        self.rows[check].append((file_id, field, detail))


def find_v_fields(obj, path, out):
    """Collect (path, obj) for every dict carrying a 'v' key, indexing list
    positions so appliances[2]/other_findings[0]/etc. are addressable."""
    if isinstance(obj, dict):
        if "v" in obj:
            out.append((path, obj))
        for k, val in obj.items():
            if k in _LEAF_KEYS:
                continue
            new_path = f"{path}.{k}" if path else k
            find_v_fields(val, new_path, out)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_v_fields(item, f"{path}[{i}]", out)


def find_empty_arrays(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_empty_arrays(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        if len(obj) == 0:
            out.append(path)
        else:
            for item in obj:
                find_empty_arrays(item, path, out)


def find_hedged(obj, path, out):
    if isinstance(obj, dict):
        if "hedged" in obj:
            out.append((path, obj["hedged"]))
        for k, v in obj.items():
            find_hedged(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        for item in obj:
            find_hedged(item, path, out)


def check_src(v, file_id, field, sources, viol, allow_missing=False):
    """Validate a 'src' list: present, non-empty, well-formed, rids resolve."""
    src = v.get("src") if isinstance(v, dict) else None
    if src is None:
        if not allow_missing:
            viol.add("field_missing_src", file_id, field, "field has 'v' but no 'src' key")
        return
    if not isinstance(src, list) or len(src) == 0:
        viol.add("field_missing_src", file_id, field, f"src present but empty/not-a-list: {src!r}")
        return
    for s in src:
        check_one_src(s, file_id, field, sources, viol)


def check_one_src(s, file_id, field, sources, viol):
    if not isinstance(s, str):
        viol.add("src_rid_invalid", file_id, field, f"non-string src entry: {s!r}")
        return
    m = SRC_RE.match(s)
    if not m:
        viol.add("src_rid_invalid", file_id, field, f"malformed src {s!r} (want 'ios:<rid>' or 'photo:<rid>')")
        return
    modality, rid = m.group(1), m.group(2)
    allowed = sources.get(modality) if isinstance(sources, dict) else None
    if not isinstance(allowed, list) or rid not in allowed:
        viol.add("src_rid_not_in_sources", file_id, field, f"{s!r} -> rid {rid!r} not in sources.{modality}")


def resolve_merge_note_enum(field):
    """Best-effort: map a merge_notes 'field' path to the enum its values
    dict keys must belong to. Returns None if unresolvable (handled leniently,
    never flagged)."""
    if field in V_ENUMS:
        return V_ENUMS[field]
    if field.endswith(".degree"):
        base = field[: -len(".degree")]
        if base in DEGREE_FIELDS:
            return DEGREE_ENUM
        return None
    if field.endswith(".region"):
        base = field[: -len(".region")]
        if base == "transverse.crossbite":
            return REGION_ENUM
        return None
    if field.endswith(".arch"):
        base = field[: -len(".arch")]
        if base in ARCH_FIELDS:
            return ARCH_ENUM
        return None
    if field.startswith("teeth."):
        return FINDINGS_ENUM
    return None


def _is_fdi_list_key(key):
    """'25' or '25,36' -- a teeth-array diff (crossbite/scissor_bite/recession
    .teeth disagreed, or a contested 'teeth.<FDI>/teeth.<FDI>' identity
    conflict) recorded as tooth codes instead of the field's own v enum."""
    parts = key.split(",")
    return bool(parts) and all(p in VALID_FDI for p in parts)


def _is_compound_class_token(v_enum, token):
    """'end_to_end_II' / 'II_end_to_end' -- on a molar/canine class field, a
    disagreement that spans both v and degree gets logged as one compound
    token rather than two. True if some underscore split point divides the
    token into a legal degree value and a legal v value (either order)."""
    parts = token.split("_")
    for split in range(1, len(parts)):
        left = "_".join(parts[:split])
        right = "_".join(parts[split:])
        if (left in DEGREE_ENUM and right in v_enum) or (right in DEGREE_ENUM and left in v_enum):
            return True
    return False


def merge_note_key_is_plausible(field, legal_keys, key):
    """A merge_notes values-dict key is judged invalid only when it matches
    none of: the field's plain enum, the '..._self_split' within-group-
    disagreement sentinel (rule 5), a compound v+degree token on a class
    field, or a teeth-list diff. Anything else (e.g. a stray 'end-to-end'
    hyphen typo where every other record spells it 'end_to_end') still gets
    flagged."""
    if legal_keys is not None and key in legal_keys:
        return True
    if key.endswith("_self_split"):
        return True
    if legal_keys is not None and field in DEGREE_FIELDS and _is_compound_class_token(legal_keys, key):
        return True
    if _is_fdi_list_key(key):
        return True
    return legal_keys is None  # unresolvable field path -> don't guess


# --------------------------------------------------------------------------
# Per-record validation
# --------------------------------------------------------------------------


def validate_record(file_id, d, viol, stated_counts, total_records):
    # -- case_id --------------------------------------------------------
    case_id = d.get("case_id")
    if case_id is None:
        viol.add("case_id_missing", file_id, "case_id", "no case_id field")
    elif case_id != file_id:
        viol.add("case_id_mismatch", file_id, "case_id", f"case_id={case_id!r} != filename {file_id!r}")

    # -- sources ----------------------------------------------------------
    sources = d.get("sources")
    if not isinstance(sources, dict):
        viol.add("sources_malformed", file_id, "sources", f"sources is {type(sources).__name__}, expected dict")
        sources = {}
    else:
        for modality in ("ios", "photo"):
            if modality not in sources:
                viol.add("sources_missing_modality_key", file_id, f"sources.{modality}",
                          f"sources dict has no {modality!r} key at all")

    # -- coverage bookkeeping for stated-rate table -----------------------
    for field in CORE_FIELDS:
        if get_path(d, field) is not None:
            stated_counts[field] += 1

    # -- generic "v"-bearing field walk: enum legality, src, degree/region/arch, FDI teeth arrays --
    vfields = []
    find_v_fields(d, "", vfields)
    for path, obj in vfields:
        # strip list index suffix (e.g. "appliances[0]") to look up the enum table
        base_path = re.sub(r"\[\d+\]$", "", path)

        v = obj.get("v")
        if base_path in V_ENUMS:
            if v not in V_ENUMS[base_path]:
                viol.add("enum_invalid", file_id, path, f"v={v!r} not in legal set for {base_path}")
        # free-text fields (appliances[], other_findings[], soft_tissue.other,
        # teeth findings' note, etc.) carry no enum -- only structural checks apply.

        # src required + must resolve against sources
        check_src(obj, file_id, path, sources, viol)

        # degree (molar/canine class fields)
        if base_path in DEGREE_FIELDS and "degree" in obj:
            deg = obj["degree"]
            if deg not in DEGREE_ENUM:
                viol.add("enum_invalid", file_id, f"{path}.degree", f"degree={deg!r} not in {sorted(DEGREE_ENUM)}")

        # region (crossbite only per spec)
        if "region" in obj:
            reg = obj["region"]
            if base_path == "transverse.crossbite":
                if reg not in REGION_ENUM:
                    viol.add("enum_invalid", file_id, f"{path}.region", f"region={reg!r} not in {sorted(REGION_ENUM)}")
            elif base_path == "transverse.scissor_bite":
                # SCHEMA.md spells scissor_bite out fully as {v, teeth} with no
                # region key (unlike crossbite) -- a "region" here is an
                # undocumented field, not a v1.2 enum violation. Flagged
                # separately so it doesn't get conflated with real enum bugs.
                viol.add("undocumented_field", file_id, f"{path}.region",
                          f"scissor_bite carries 'region'={reg!r}; not in the v1.2 schema for scissor_bite")
                if reg not in REGION_ENUM:
                    viol.add("enum_invalid", file_id, f"{path}.region", f"region={reg!r} not in {sorted(REGION_ENUM)}")

        # arch (curve_spee / curve_wilson only)
        if base_path in ARCH_FIELDS and "arch" in obj:
            arch = obj["arch"]
            if arch not in ARCH_ENUM:
                viol.add("enum_invalid", file_id, f"{path}.arch", f"arch={arch!r} not in {sorted(ARCH_ENUM)}")

        # FDI arrays living alongside v (crossbite.teeth, scissor_bite.teeth, recession.teeth)
        if "teeth" in obj:
            teeth_arr = obj["teeth"]
            if isinstance(teeth_arr, list):
                for t in teeth_arr:
                    if not isinstance(t, str) or t not in VALID_FDI:
                        viol.add("fdi_invalid", file_id, f"{path}.teeth", f"invalid FDI code {t!r}")

        # contested must be boolean wherever present on a v-field (rare, but cheap to check)
        if "contested" in obj and not isinstance(obj["contested"], bool):
            viol.add("contested_not_bool", file_id, path, f"contested={obj['contested']!r} is not boolean")

    # -- hedged must be boolean, anywhere --------------------------------
    hedged_hits = []
    find_hedged(d, "", hedged_hits)
    for path, val in hedged_hits:
        if not isinstance(val, bool):
            viol.add("hedged_not_bool", file_id, path, f"hedged={val!r} is not boolean")

    # -- empty arrays anywhere (sources.ios/sources.photo excepted) -----
    empties = []
    find_empty_arrays(d, "", empties)
    for path in empties:
        if path in ("sources.ios", "sources.photo"):
            continue
        viol.add("empty_array", file_id, path, "empty array present (spec rule 21: omit the key instead)")

    # -- teeth object: keys are valid FDI, findings are legal, src required --
    teeth = d.get("teeth")
    if isinstance(teeth, dict):
        for fdi, entry in teeth.items():
            if fdi not in VALID_FDI:
                viol.add("fdi_invalid", file_id, f"teeth.{fdi}", f"invalid FDI key {fdi!r}")
            if not isinstance(entry, dict):
                viol.add("teeth_entry_malformed", file_id, f"teeth.{fdi}", f"entry is {type(entry).__name__}, expected dict")
                continue
            findings = entry.get("findings")
            if not isinstance(findings, list) or len(findings) == 0:
                viol.add("teeth_entry_malformed", file_id, f"teeth.{fdi}.findings",
                          f"findings missing/empty: {findings!r}")
            else:
                for fnd in findings:
                    if fnd not in FINDINGS_ENUM:
                        viol.add("enum_invalid", file_id, f"teeth.{fdi}.findings", f"finding={fnd!r} not in FINDINGS_ENUM")
            check_src(entry, file_id, f"teeth.{fdi}", sources, viol)
            if "contested" in entry and not isinstance(entry["contested"], bool):
                viol.add("contested_not_bool", file_id, f"teeth.{fdi}", f"contested={entry['contested']!r} is not boolean")
    elif teeth is not None:
        viol.add("teeth_entry_malformed", file_id, "teeth", f"teeth is {type(teeth).__name__}, expected dict")

    # -- merge_notes: structural + best-effort enum-key/src validation ---
    merge_notes = d.get("merge_notes")
    if isinstance(merge_notes, list):
        for i, mn in enumerate(merge_notes):
            tag = f"merge_notes[{i}]"
            if not isinstance(mn, dict):
                viol.add("merge_notes_malformed", file_id, tag, f"entry is {type(mn).__name__}, expected dict")
                continue
            field = mn.get("field")
            if not isinstance(field, str) or not field:
                viol.add("merge_notes_malformed", file_id, tag, f"missing/empty 'field': {field!r}")
                field = None
            values = mn.get("values")
            if not isinstance(values, dict) or len(values) == 0:
                viol.add("merge_notes_malformed", file_id, tag, f"missing/empty 'values' dict: {values!r}")
                values = {}

            legal_keys = resolve_merge_note_enum(field) if field else None
            for key, srcs in values.items():
                if field and not merge_note_key_is_plausible(field, legal_keys, key):
                    viol.add("merge_notes_value_key_invalid", file_id, f"{tag}.values.{key}",
                              f"key {key!r} not a legal enum member of field {field!r} "
                              f"(and not a recognized self_split/compound/teeth-list pattern)")
                if not isinstance(srcs, list) or len(srcs) == 0:
                    viol.add("merge_notes_malformed", file_id, f"{tag}.values.{key}",
                              f"values[{key!r}] empty/not-a-list: {srcs!r}")
                    continue
                for s in srcs:
                    check_one_src(s, file_id, f"{tag}.values.{key}", sources, viol)
    elif merge_notes is not None:
        viol.add("merge_notes_malformed", file_id, "merge_notes", f"merge_notes is {type(merge_notes).__name__}, expected list")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    case_list = [ln.strip() for ln in CASE_LIST_PATH.read_text().splitlines() if ln.strip()]
    case_list_set = set(case_list)

    on_disk = {p.stem for p in RAW_DIR.glob("*.json")}

    missing = sorted(case_list_set - on_disk)
    extra = sorted(on_disk - case_list_set)
    dup_lines = [c for c, n in Counter(case_list).items() if n > 1]

    viol = Violations()
    stated_counts = Counter()
    parsed = 0
    parse_errors = []

    total_records = len(case_list)

    for case_id in case_list:
        fpath = RAW_DIR / f"{case_id}.json"
        if not fpath.exists():
            continue  # already counted in `missing`
        raw = fpath.read_text(encoding="utf-8")
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            parse_errors.append((case_id, str(e)))
            viol.add("json_parse_error", case_id, "<file>", str(e))
            continue
        parsed += 1
        if not isinstance(d, dict):
            viol.add("json_parse_error", case_id, "<file>", f"top-level JSON is {type(d).__name__}, expected object")
            continue
        validate_record(case_id, d, viol, stated_counts, total_records)

    # ---------------------------------------------------------------- report
    print("=" * 78)
    print("BITE2TEXT SCHEMA VALIDATOR (v1.2)")
    print("=" * 78)

    print(f"\ncase_list.txt entries: {total_records}")
    print(f"raw_cases/*.json on disk: {len(on_disk)}")
    print(f"parsed successfully: {parsed}")
    print(f"duplicate lines in case_list.txt: {len(dup_lines)}" + (f" {dup_lines[:10]}" if dup_lines else ""))
    print(f"missing (listed, no file): {len(missing)}" + (f" e.g. {missing[:10]}" if missing else ""))
    print(f"extra (file, not listed): {len(extra)}" + (f" e.g. {extra[:10]}" if extra else ""))

    print("\n" + "-" * 78)
    print("VIOLATION COUNTS BY CHECK")
    print("-" * 78)
    if not viol.counts:
        print("(none -- every check passed clean)")
    else:
        total_violations = sum(viol.counts.values())
        for check, n in viol.counts.most_common():
            print(f"  {n:6d}  {check}")
        print(f"  {total_violations:6d}  TOTAL")

    print("\n" + "-" * 78)
    print("SAMPLE OFFENDING ROWS (up to 40, round-robin across checks)")
    print("-" * 78)
    checks_order = [c for c, _ in viol.counts.most_common()]
    printed = 0
    cursors = {c: 0 for c in checks_order}
    while printed < 40 and checks_order:
        progressed = False
        for c in checks_order:
            if printed >= 40:
                break
            rows = viol.rows[c]
            cur = cursors[c]
            if cur < len(rows):
                file_id, field, detail = rows[cur]
                print(f"  [{c}] {file_id} | {field} | {detail}")
                cursors[c] += 1
                printed += 1
                progressed = True
        if not progressed:
            break

    print("\n" + "-" * 78)
    print("PER-FIELD STATED-RATE (fraction of records where field is present)")
    print("-" * 78)
    denom = total_records if total_records else 1
    for field in CORE_FIELDS:
        n = stated_counts.get(field, 0)
        pct = 100.0 * n / denom
        print(f"  {field:38s} {n:4d}/{denom} ({pct:5.1f}%)")

    print("\n" + "=" * 78)
    ok = not viol.counts and not missing and not extra and not parse_errors
    print("RESULT: " + ("PASS -- no violations found" if ok else "VIOLATIONS FOUND -- see above"))
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
