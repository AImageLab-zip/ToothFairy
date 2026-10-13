#!/usr/bin/env python3
"""
code/pipeline/postprocess/normalize_pred.py

Repairs schema-violating VLM output in {case_id}_pred.json BEFORE
postprocess_pred.py consumes it. Schema-driven (reads object_fields/
enum_values straight from whatever schema.json it's pointed at), so v6.1
needed no rewrite here -- only two additions: "object" as a recognized
type head (dental_arch_findings_{arch}.findings is a free-keyed FDI ->
finding map) and, with it, accepting a parsed map whose keys are data
rather than declared field names.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Repo bootstrap. Finds code/ by walking up for _repo.py, so this file does not
# care how deep it sits, and puts every code group on sys.path so the flat
# `import postprocess_pred` works across groups. See code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import REPO_ROOT, add_code_paths  # noqa: E402
add_code_paths()

DEFAULT_SCHEMA = REPO_ROOT / "schema" / "schema.json"

# v4: dental_arch no longer exists as its own section (everything moved
# into mandible/maxilla, or is derived rather than asked). Scanning it
# here would just be a silent no-op since schema.get() returns [] for a
# missing key -- kept explicit rather than left to fail silently.
SCHEMA_SECTIONS = ("mandible", "maxilla", "dental_elements")


# ── Schema lookup ─────────────────────────────────────────────────────────

_SPEC_CACHE: Dict[str, Dict[str, Dict]] = {}


def load_field_specs(schema_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Map output_field -> fact spec. Per-tooth facts keep their "{fdi}"
    placeholder in the key; resolve_spec() substitutes the concrete FDI.
    """
    path = str(schema_path or DEFAULT_SCHEMA)
    if path in _SPEC_CACHE:
        return _SPEC_CACHE[path]
    with open(path) as f:
        schema = json.load(f)
    specs: Dict[str, Dict] = {}
    for section in SCHEMA_SECTIONS:
        for fact in schema.get(section, []) or []:
            specs[fact["output_field"]] = fact
    _SPEC_CACHE[path] = specs
    return specs


_TOOTH_FIELD_RE = re.compile(r"^tooth_(\d+)_(.+)$")


def resolve_spec(field: str, specs: Dict[str, Dict]) -> Optional[Dict]:
    """Find the spec for a concrete field name (tooth_11_eruption -> tooth_{fdi}_eruption)."""
    if field in specs:
        return specs[field]
    m = _TOOTH_FIELD_RE.match(field)
    if m:
        return specs.get(f"tooth_{{fdi}}_{m.group(2)}")
    return None


# An object_fields value is a type head optionally followed by prose, e.g.
# "bool -- true if ...", "list[int], only when scope != 'not_included' -- ...",
# "enum:none|mild|moderate|severe -- judged as bone loss relative to ...".
#
# "object" is in the list for v6.1's dental_arch_findings_{arch}.findings, a
# free-keyed FDI -> finding map. It must stay AFTER list[object] in the
# alternation, or "list[object] -- ..." would never match its own head.
_SPEC_HEAD_RE = re.compile(
    r"^\s*(list\[object\]|list\[int\]|list\[string\]"
    r"|enum:[A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*|object|bool|string|int)")


def enum_values_of(spec_str: str) -> List[str]:
    """
    'enum:fully_included|partially_included|not_included -- prose' -> [...]
    Stops at the first non-enum character, so trailing prose on the last
    member ("...|eroded, only answer when scope != not_included") is not
    mistaken for part of that member's value.
    """
    if not isinstance(spec_str, str):
        return []
    m = _SPEC_HEAD_RE.match(spec_str)
    if not m or not m.group(1).startswith("enum:"):
        return []
    return m.group(1)[len("enum:"):].split("|")


def parse_field_spec(spec_str: str) -> Optional[Dict]:
    """
    Turn an object_fields spec string into a spec dict coerce_field() accepts,
    so sub-fields inside an object-typed fact get the same type repair the
    top-level facts get. Returns None when the type head is unrecognized.
    """
    if not isinstance(spec_str, str):
        return None
    m = _SPEC_HEAD_RE.match(spec_str)
    if not m:
        return None
    head = m.group(1)
    if head.startswith("enum:"):
        return {"type": "enum", "enum_values": enum_values_of(spec_str)}
    return {"type": head}


# ── Tolerant parser for the pseudo-JSON the VLM emits ─────────────────────

def _split_top_level(text: str, sep: str = ",") -> List[str]:
    parts, buf, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "{[":
            depth += 1
            buf.append(ch)
        elif ch in "}]":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _parse_scalar(text: str) -> Any:
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "n/a", "na", ""):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _parse_value(text: str) -> Any:
    t = text.strip()
    if t.startswith("{"):
        return parse_loose_object(t)
    if t.startswith("["):
        inner = t[1:-1] if t.endswith("]") else t[1:]
        items = [_parse_value(p) for p in _split_top_level(inner) if p.strip()]
        return items
    return _parse_scalar(t)


def parse_loose_object(text: str) -> Optional[Dict]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass

    t = re.sub(r"^object\s*", "", t, flags=re.IGNORECASE).strip()
    if t.startswith("{"):
        t = t[1:]
        if t.endswith("}"):
            t = t[:-1]
    if ":" not in t:
        return None

    out: Dict[str, Any] = {}
    last_key = None
    for frag in _split_top_level(t):
        if not frag.strip():
            continue
        if ":" not in frag:
            if last_key is not None and isinstance(out.get(last_key), str):
                out[last_key] = f"{out[last_key]},{frag.rstrip()}"
            continue
        key, _, value = frag.partition(":")
        key = key.strip().strip("\"'")
        if not key:
            continue
        out[key] = _parse_value(value)
        last_key = key
    return out or None


# ── Type coercion, driven by the fact spec ────────────────────────────────

def parse_colonless_pairs(text: str, obj_fields: Dict[str, str]) -> Optional[Dict]:
    if not isinstance(text, str) or not obj_fields:
        return None
    aliases = {k.replace("_", " ").lower(): k for k in obj_fields}
    aliases.update({k.lower(): k for k in obj_fields})

    out: Dict[str, Any] = {}
    for frag in _split_top_level(text):
        f = frag.strip().strip("{}").strip()
        low = f.lower()
        for alias in sorted(aliases, key=len, reverse=True):
            if low.startswith(alias):
                rest = f[len(alias):].strip()
                if rest:
                    out[aliases[alias]] = _parse_scalar(rest)
                break
    return out or None


KEY_MATCH_CUTOFF = 0.8


def repair_object_keys(value: Dict, obj_fields: Dict[str, str]) -> Tuple[Dict, List[str]]:
    if not obj_fields:
        return value, []
    renamed: List[str] = []
    out = dict(value)
    for key in list(out):
        if key in obj_fields:
            continue
        candidates = [k for k in obj_fields if k not in out]
        match = difflib.get_close_matches(key, candidates, n=1, cutoff=KEY_MATCH_CUTOFF)
        if match:
            out[match[0]] = out.pop(key)
            renamed.append(f"{key} -> {match[0]}")
    return out, renamed


def repair_sub_field_types(value: Dict, obj_fields: Dict[str, str]) -> Tuple[Dict, List[str]]:
    """
    Type-repair the sub-fields of an object fact (the VLM violates the inner
    types as often as the outer ones -- e.g. answering a list[object]
    sub-field with a bare enum string). list[object] sub-fields are left
    alone on purpose: what a bare string there means is fact-specific, so
    postprocess_pred.py recovers those where it knows the domain, and a
    generic drop here would throw that information away first.
    """
    notes: List[str] = []
    out = dict(value)
    for key, spec_str in obj_fields.items():
        if key not in out:
            continue
        sub_spec = parse_field_spec(spec_str)
        if sub_spec is None or sub_spec["type"] == "list[object]":
            continue
        new_value, note = coerce_field(key, out[key], sub_spec)
        if note is not None:
            out[key] = new_value
            notes.append(f"{key}: {note}")
    return out, notes


def _coerce_object(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    obj_fields: Dict[str, str] = spec.get("object_fields", {}) or {}

    if isinstance(value, dict):
        fixed, renamed = repair_object_keys(value, obj_fields)
        fixed, retyped = repair_sub_field_types(fixed, obj_fields)
        notes = ([("misspelled key(s) renamed: " + ", ".join(renamed))] if renamed else []) + retyped
        if notes:
            return fixed, "; ".join(notes)
        return value, None
    if value in (None, "", [], {}):
        return {}, None

    if isinstance(value, str):
        parsed = parse_loose_object(value)
        if isinstance(parsed, dict):
            if not obj_fields:
                # A free-keyed map (v6.1's dental_arch_findings.findings, whose
                # keys are FDI numbers) declares no field names to filter by --
                # keeping only "declared" keys would empty it every time.
                return parsed, "parsed pseudo-JSON string into object"
            parsed, _ = repair_object_keys(parsed, obj_fields)
            kept = {k: v for k, v in parsed.items() if k in obj_fields}
            if kept:
                return kept, "parsed pseudo-JSON string into object"

        colonless = parse_colonless_pairs(value, obj_fields)
        if colonless:
            return colonless, "parsed colon-less key/value string into object"

        bare = value.strip().strip("\"'")
        owners = [k for k, spec_str in obj_fields.items() if bare in enum_values_of(spec_str)]
        if owners:
            note = (f"bare enum string assigned to {owners[0]}" if len(owners) == 1
                    else f"bare enum string assigned to {len(owners)} matching field(s)")
            return {k: bare for k in owners}, note
        return {}, "unrecognized object string dropped"

    return {}, f"unexpected {type(value).__name__} for object field dropped"


def _enum_separators(value: str) -> str:
    """
    'not included' / 'not-included' -> 'not_included'. The enum values are
    written with underscores and the VLM often answers with the same words
    separated the way prose separates them. Only the separators are touched,
    so this can turn a dropped answer into a recovered one and can never turn
    one allowed value into a different one.
    """
    return re.sub(r'[\s\-]+', '_', value.strip().lower())


def _enum_by_prefix(value: str, allowed: List[str]) -> Optional[str]:
    """
    'partial' -> 'partially_erupted': the VLM often answers with a truncated
    form of the enum value. Only accepted when exactly ONE allowed value
    matches, so an ambiguous stem (e.g. 'none' where both 'none' and
    'none_visible' existed) is still dropped rather than guessed at.

    Separators are normalized first (see _enum_separators), which matters most
    for maxilla_included: schema v6.6 made it a two-value enum where
    "not_included" empties the whole arch, so an answer of "not included"
    falling through to "unanswered" is a silently lost drop.
    """
    v = _enum_separators(value)
    if not v:
        return None
    matches = [a for a in allowed
               if _enum_separators(a).startswith(v) or v.startswith(_enum_separators(a))]
    return matches[0] if len(matches) == 1 else None


def _coerce_enum(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    allowed = spec.get("enum_values", []) or []
    if isinstance(value, str):
        v = value.strip()
        if v in allowed or not allowed:
            return v, None
        recovered = _enum_by_prefix(v, allowed)
        if recovered:
            return recovered, f"truncated enum value '{v}' read as '{recovered}'"
        return None, "value not in enum dropped"
    if value is None:
        return None, None
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, str) and v in allowed:
                return v, "enum recovered from object"
        bools = [v for v in value.values() if isinstance(v, bool)]
        if bools and not any(bools) and "none" in allowed:
            return "none", "all-negative object read as 'none'"
        return None, "object could not be reduced to an enum, dropped"
    return None, f"unexpected {type(value).__name__} for enum field dropped"


def _coerce_list_int(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    if value is None:
        return [], None
    if isinstance(value, str):
        value = _parse_value(value)
        if not isinstance(value, list):
            value = [value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list):
        return [], f"unexpected {type(value).__name__} for list[int] dropped"
    out, dropped = [], False
    for item in value:
        if isinstance(item, bool):
            dropped = True
        elif isinstance(item, int):
            out.append(item)
        elif isinstance(item, float) and item.is_integer():
            out.append(int(item))
        elif isinstance(item, str) and item.strip().isdigit():
            out.append(int(item.strip()))
        elif isinstance(item, dict):
            nums = [v for v in item.values()
                    if isinstance(v, int) and not isinstance(v, bool)]
            if len(nums) == 1:
                out.append(nums[0])
            else:
                dropped = True
        else:
            dropped = True
    return out, "non-integer list member(s) dropped" if dropped else None


def _coerce_list_object(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    if value is None:
        return [], None
    if isinstance(value, str):
        parsed = _parse_value(value)
        value = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])
        note = "parsed string into list[object]"
    else:
        note = None
    if isinstance(value, dict):
        return [value], "single object wrapped into list"
    if not isinstance(value, list):
        return [], f"unexpected {type(value).__name__} for list[object] dropped"
    out = [item for item in value if isinstance(item, dict)]
    if len(out) != len(value):
        note = "non-object list member(s) dropped"

    # Each member is itself an object described by object_fields -- give the
    # members the same key/type repair a plain object fact gets.
    obj_fields: Dict[str, str] = spec.get("object_fields", {}) or {}
    if obj_fields:
        repaired, item_notes = [], []
        for item in out:
            item, renamed = repair_object_keys(item, obj_fields)
            item, retyped = repair_sub_field_types(item, obj_fields)
            item_notes.extend([f"key(s) renamed: {', '.join(renamed)}"] if renamed else [])
            item_notes.extend(retyped)
            repaired.append(item)
        out = repaired
        if item_notes:
            member_note = "list member(s) repaired: " + "; ".join(sorted(set(item_notes)))
            note = f"{note}; {member_note}" if note else member_note
    return out, note


def _coerce_bool(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    if isinstance(value, bool) or value is None:
        return value, None
    if isinstance(value, str):
        parsed = _parse_scalar(value)
        if isinstance(parsed, bool) or parsed is None:
            return parsed, "string coerced to bool"
    if isinstance(value, (int, float)):
        return bool(value), "number coerced to bool"
    return None, f"unexpected {type(value).__name__} for bool field dropped"


def _coerce_string(value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    if isinstance(value, str) or value is None:
        return value, None
    if isinstance(value, dict):
        for key in ("findings", "description", "text", "impression"):
            if isinstance(value.get(key), str):
                return value[key], f"string recovered from object.{key}"
        return None, "object could not be reduced to a string, dropped"
    if isinstance(value, (int, float, bool)):
        return str(value), "scalar coerced to string"
    return None, f"unexpected {type(value).__name__} for string field dropped"


_COERCERS = {
    "object": _coerce_object,
    "enum": _coerce_enum,
    "list[int]": _coerce_list_int,
    "list[object]": _coerce_list_object,
    "bool": _coerce_bool,
    "string": _coerce_string,
    "int": lambda v, s: (v, None) if isinstance(v, int) or v is None else (None, "non-int dropped"),
    "list[string]": lambda v, s: (v, None) if isinstance(v, list) or v is None
                                 else ([v], "scalar wrapped into list") if isinstance(v, str)
                                 else ([], "unexpected value for list[string] dropped"),
}


def coerce_field(field: str, value: Any, spec: Dict) -> Tuple[Any, Optional[str]]:
    coercer = _COERCERS.get(spec.get("type", ""))
    if coercer is None:
        return value, None
    return coercer(value, spec)


# ── Whole-prediction normalization ────────────────────────────────────────

def normalize_prediction(pred: Dict,
                         schema_path: Optional[str] = None) -> Tuple[Dict, List[Dict]]:
    specs = load_field_specs(schema_path)
    out = copy.deepcopy(pred)
    repairs: List[Dict] = []

    def fix_container(container: Dict, where: str) -> None:
        for field, value in list(container.items()):
            spec = resolve_spec(field, specs)
            if spec is None:
                continue
            new_value, note = coerce_field(field, value, spec)
            if note is None:
                continue
            preview = json.dumps(value, default=str)
            repairs.append({
                "where": where,
                "field": field,
                "action": note,
                "type": spec.get("type"),
                "original": preview[:200],
            })
            container[field] = new_value

    g = out.get("global")
    if isinstance(g, dict):
        fix_container(g, "global")

    teeth = out.get("teeth")
    if isinstance(teeth, dict):
        for tooth_key, tooth in list(teeth.items()):
            if not isinstance(tooth, dict):
                repairs.append({
                    "where": "teeth",
                    "field": tooth_key,
                    "action": f"tooth entry was {type(tooth).__name__}, replaced with empty object",
                    "type": "object",
                    "original": json.dumps(tooth, default=str)[:200],
                })
                teeth[tooth_key] = {}
                continue
            fix_container(tooth, f"teeth.{tooth_key}")

    return out, repairs


def summarize_repairs(repairs: List[Dict]) -> str:
    counts: Dict[str, int] = {}
    for r in repairs:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    return "; ".join(f"{v}x {k}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


# ── CLI (audit / repair in place) ─────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="Directory of {case_id}_pred.json files")
    ap.add_argument("--schema", default=None, help=f"Default: {DEFAULT_SCHEMA}")
    ap.add_argument("--write", action="store_true",
                    help="Rewrite the prediction files in place (default: audit only)")
    ap.add_argument("--verbose", action="store_true", help="Print every repair")
    args = ap.parse_args()

    files = sorted(Path(args.pred_dir).glob("*_pred.json"))
    if not files:
        print(f"[FAIL] no *_pred.json in {args.pred_dir}", file=sys.stderr)
        return 1

    total, touched = 0, 0
    for path in files:
        with open(path) as f:
            pred = json.load(f)
        fixed, repairs = normalize_prediction(pred, args.schema)
        if repairs:
            touched += 1
            total += len(repairs)
            print(f"[{path.stem}] {len(repairs)} repair(s): {summarize_repairs(repairs)}")
            if args.verbose:
                for r in repairs:
                    print(f"    {r['where']}.{r['field']} ({r['type']}): {r['action']}\n"
                          f"        was: {r['original']}")
        if args.write and repairs:
            with open(path, "w") as f:
                json.dump(fixed, f, indent=2)

    print(f"\n[INFO] {touched}/{len(files)} case(s) needed repair, {total} field(s) total")
    if args.write:
        print("[INFO] Files rewritten in place")
    else:
        print("[INFO] Audit only -- pass --write to rewrite the files")
    return 0


if __name__ == "__main__":
    sys.exit(main())