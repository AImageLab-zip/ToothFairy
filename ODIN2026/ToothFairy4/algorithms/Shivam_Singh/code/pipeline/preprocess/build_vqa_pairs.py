#!/usr/bin/env python3
"""
build_vqa_pairs.py

Build BATCHED qa_pairs.jsonl directly from schema.json (single source of truth).

Updated for schema v6.1 and the current image-creation scripts. Each
script emits its images plus a caption sidecar JSON, written at generation
time; every sidecar is keyed by schema.json's own "images_needed" names, so
nothing here has to translate between filename and key format:

  create_3d_renders.py    -> {case}_3d_left.png, _3d_frontal.png, _3d_right.png
                              + {case}_3d_captions.json
  create_panoramic.py     -> {case}_panoramic.png (single image, skipped
                              entirely if <4 teeth detected)
                              + {case}_panoramic_caption.json
  create_tooth_detail.py  -> {case}_tooth{fdi}_composite.png (one multi-panel
                              composite per tooth)
                              + {case}_tooth_captions.json
  create_sinus_detail.py  -> {case}_sinus_{right,left}_detail.png
                              + {case}_sinus_captions.json

Each of those is ONE multi-panel image per fact target, so every fact needs
exactly one image today.

WHICH QUESTIONS RIDE ON ONE CALL is declared explicitly in CALL_PLAN below,
not derived from which facts happen to share an image. Nine calls per case:
one per 3D view, four across the panoramic, one per sinus side. The
panoramic split is the reason the plan is explicit -- all 12 of its facts
share one image, so any image-derived grouping would merge them, whereas
dental_arch_findings_{arch} (a tooth-by-tooth survey of up to 16 teeth)
needs a response of its own per arch.

resolve_call_plan() cross-checks that table against schema.json on every
run and hard-fails on any disagreement, so a fact added to the schema and
not placed in a call is an error rather than a question silently never asked.

Output: still one record per case, with the same overall split (global /
dental_elements) as before. Each call carries an "images" dict (image_key ->
relative path), a "captions" dict (image_key -> caption text, pulled from
the sidecar JSON files) so downstream inference can pass captions to the VLM
alongside each image, and -- new in v6.1 -- a "guidance" string holding the
shared _definitions vocabulary that call's facts refer to by name.
If you don't want captions threaded through qa_pairs.jsonl (e.g. if
run_vqa_inference.py will read the sidecar files itself instead), the
--no-captions flag drops the "captions" field entirely.

A call is DROPPED if NONE of its needed images exist for the case (partial
image sets are still sent -- more images generally helps the VLM, and no
single image in a set is strictly required by the questions themselves).
dental_elements keeps its old policy: all 32 FDI keys always present, with
individual images possibly missing (skipped, not dropping the whole tooth).
What each of those keys ASKS can differ, though: a dental_elements fact may
carry an "applies_to_fdi" list narrowing it to the positions where the
question exists (tooth_{fdi}_mandible_canal, the lower molars) -- see
fact_applies_to_fdi below.

Usage:
    python build_vqa_pairs.py \
        --schema schema/schema.json \
        --images-dir test_5/outputs/images \
        --out test_5/outputs/qa_pairs.jsonl \
        --project-dir $HOME/project_ToothFairy4
"""

import json
import re
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ALL_FDIS = [11,12,13,14,15,16,17,18,
            21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38,
            41,42,43,44,45,46,47,48]

# The name of the record section every arch-level call is filed under.
# run_vqa_inference.py must use this same name when building VLM requests --
# keep both in sync.
GLOBAL_GROUP = "global"

# schema.json sections holding arch-level facts, in the order CALL_PLAN's
# coverage check reports them. (v6.1 folded the old "dental_arch" section
# into these two.)
GLOBAL_SECTIONS = ["mandible", "maxilla"]

TOOTH_SECTION = "dental_elements"

# ── Per-fact FDI gate ───────────────────────────────────────────────────────
#
# A dental_elements fact normally applies to all 32 positions -- that is what
# the tooth_{fdi}_ template means. An optional "applies_to_fdi" list narrows
# one fact to the positions where the question exists at all:
# tooth_{fdi}_mandible_canal is asked of the lower molars (36-38, 46-48)
# only, because the mandibular canal does not run under the other 26.
#
# Gating it here rather than telling the model "answer null unless this is a
# lower molar" is the difference between 6 questions and 32: the other 26 are
# not cheap null answers, they are 26 more chances to invent a canal finding
# at a position that has no canal, and 26 fields the guided_json decoder is
# forced to emit. Every consumer that walks dental_elements per FDI asks this
# same function -- see its importers -- so a fact cannot be asked in one arm
# and scored in another.
FDI_SCOPE_KEY = "applies_to_fdi"


def fact_applies_to_fdi(fact: Dict, fdi: int) -> bool:
    """Is this dental_elements fact asked at this FDI? (No gate -> yes.)"""
    scope = fact.get(FDI_SCOPE_KEY)
    if not scope:
        return True
    return int(fdi) in {int(f) for f in scope}


def tooth_facts_for_fdi(facts: List[Dict], fdi: int) -> List[Dict]:
    """The dental_elements facts asked at `fdi`, in schema order."""
    return [f for f in facts if fact_applies_to_fdi(f, fdi)]

# ── THE CALL PLAN ───────────────────────────────────────────────────────────
#
# One entry per VLM call: (call_key, images, [fact_id, ...]). This is written
# out EXPLICITLY rather than derived by bucketing facts on their shared
# images, because which questions ride on one response is a judgment about
# the model's attention, not a property of the pictures.
#
# The panoramic is the case in point. All 12 of its facts share one image, so
# any image-derived grouping merges them; here it is deliberately cut FOUR
# ways, along what the questions are ABOUT:
#
#   pano_tooth_findings_{arch} -- everything answered POSITION BY POSITION:
#       the tooth-by-tooth survey plus implants and bridges, which are also
#       located by FDI ("position 35", "from 24 to 27") and are what occupies
#       a position the survey reads as absent. One pass over the arch answers
#       all three, and the survey's own absent positions are exactly where the
#       model should be looking for a fixture or a pontic.
#   pano_others_{arch} -- the three arch-WIDE judgments (primary dentition,
#       bone quality, periodontal resorption), which describe the arch as a
#       whole rather than any one position.
#
# resolve_call_plan() hard-fails if this table and schema.json disagree:
# every fact must appear exactly once, and every call's images must match
# what its facts' own images_needed asks for. Adding a fact to schema.json
# without placing it here is an error, not a silent omission.
CALL_PLAN: List[Tuple[str, Tuple[str, ...], List[str]]] = [
    # 3d_left FACES THE PATIENT'S LEFT SIDE, so it carries the LEFT-side
    # facts. It read the other way round until 2026-08-11, on the strength of
    # a comment in create_3d_renders.py claiming a camera at the patient's
    # left brings their right side into the foreground; it does not, and
    # every side-specific 3D fact was being asked on a picture of the other
    # side. resolve_call_plan() checks this table against schema.json, so the
    # two cannot drift apart again silently.
    ("3d_left", ("3d_left",), [
        "mandible_condyle_left", "mandible_canal_left",
        "lower_left_wisdom_tooth", "upper_left_wisdom_tooth"]),
    ("3d_right", ("3d_right",), [
        "mandible_condyle_right", "mandible_canal_right",
        "lower_right_wisdom_tooth", "upper_right_wisdom_tooth"]),
    ("3d_frontal", ("3d_frontal",), [
        "maxilla_scope",
        "alveolar_bone_atrophy_mandible", "alveolar_bone_atrophy_maxilla"]),
    ("pano_tooth_findings_mandible", ("panoramic",), [
        "dental_arch_findings_mandible", "implants_mandible",
        "fixed_bridges_mandible"]),
    ("pano_tooth_findings_maxilla", ("panoramic",), [
        "dental_arch_findings_maxilla", "implants_maxilla",
        "fixed_bridges_maxilla"]),
    ("pano_others_mandible", ("panoramic",), [
        "primary_teeth_mandible", "bone_quality_mandible",
        "periodontal_bone_resorption_mandible"]),
    ("pano_others_maxilla", ("panoramic",), [
        "primary_teeth_maxilla", "bone_quality_maxilla",
        "periodontal_bone_resorption_maxilla"]),
    ("sinus_right", ("sinus_right_detail",), ["maxilla_sinus_right"]),
    ("sinus_left", ("sinus_left_detail",), ["maxilla_sinus_left"]),
]

# Calls asking about maxillary TEETH off the panoramic. Skipped only when the
# mask has no maxillary tooth at all -- 98 of 622 cases -- because then there
# is nothing in the panoramic for them to read. They are deliberately KEPT for
# the 256 cases whose maxillary bone is out of the volume but whose upper
# teeth are in it: the crowns are still on the panoramic, and 11.7% of those
# reference reports describe them ("as far as can be visualized, prosthetic
# crowns are present on teeth 14, 13, 12..."). This is the cheap read that
# replaces the 16 upper composites create_tooth_detail.py no longer makes --
# a composite needs roots, an arch read needs only crowns.
#
# pano_others_maxilla is NOT here: its three facts are bone (bone_quality,
# periodontal resorption) plus primary_teeth, and an unimaged maxilla's bone
# facts are dropped by postprocess anyway.
MAXILLA_ARCH_CALLS = frozenset({"pano_tooth_findings_maxilla"})

# Written by create_tooth_detail.py from the mask. This script plans calls
# from the images on disk and never opens a mask itself, so the verdict has
# to arrive as data; absent sidecar means "no information", and every call is
# planned as before.
_COVERAGE_SUFFIX = "_coverage.json"


def discover_coverage(images_dir: str) -> Dict[str, Dict]:
    """{case_id: {"maxilla": {"bone": bool, "teeth": bool}, "mandible": {...}}}"""
    out: Dict[str, Dict] = {}
    for path in sorted(Path(images_dir).glob(f"*{_COVERAGE_SUFFIX}")):
        case_id = path.name[: -len(_COVERAGE_SUFFIX)]
        try:
            out[case_id] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [WARN] unreadable coverage sidecar {path.name}: {exc}")
    return out

# Soft cap: a call carrying more questions than this gets a warning, since
# answer quality degrades once one response has to hold too many. Not enforced
# -- CALL_PLAN is explicit, so an oversized call is a deliberate choice.
MAX_FACTS_PER_CALL = 10

# Transient key stamped on each fact copy recording which schema.json section
# it came from. Never written to qa_pairs.jsonl -- only build_questions_block()
# sees the fact dicts, and it reads named fields.
SECTION_KEY = "_section"

# ── Filename patterns for the image outputs ─────────────────────────────────
# Each pattern captures a case_id (greedy, so it correctly handles case IDs
# that vary in digit width, e.g. "A004" vs "S0030") plus whatever identifies
# the image. `key_fn` builds the key string used in schema.json's
# "images_needed" lists, which is NOT always the filename: the tooth
# composite is written as "{case}_tooth48_composite.png" but the schema
# calls it "tooth_48_composite", so the underscore is inserted here.
_FILENAME_PATTERNS = [
    (re.compile(r"^(?P<case_id>.+)_panoramic$"),
     lambda m: "panoramic"),
    (re.compile(r"^(?P<case_id>.+)_3d_(?P<view>left|frontal|right)$"),
     lambda m: f"3d_{m['view']}"),
    (re.compile(r"^(?P<case_id>.+)_sinus_(?P<side>right|left)_detail$"),
     lambda m: f"sinus_{m['side']}_detail"),
    (re.compile(r"^(?P<case_id>.+)_tooth(?P<fdi>\d{2})_composite$"),
     lambda m: f"tooth_{m['fdi']}_composite"),
]

# Caption sidecar JSON files, one per case, per image-creation script. Note
# create_panoramic.py's is "_caption" singular (it writes exactly one) --
# globbing "*_captions.json" silently misses it, which is why each suffix is
# globbed for by name rather than the set being matched after the fact.
_CAPTION_SIDECAR_SUFFIXES = ["_3d_captions.json", "_panoramic_caption.json",
                             "_tooth_captions.json", "_sinus_captions.json"]


def discover_images(images_dir: str) -> Dict[str, Dict[str, str]]:
    """Discover all generated images by case and schema image key."""

    images_path = Path(images_dir)
    if not images_path.exists():
        return {}

    images_by_case: Dict[str, Dict[str, str]] = {}

    for img_file in images_path.glob("*.png"):
        name = img_file.stem
        for pattern, key_fn in _FILENAME_PATTERNS:
            m = pattern.match(name)
            if not m:
                continue
            case_id = m["case_id"]
            key = key_fn(m)
            images_by_case.setdefault(case_id, {})[key] = str(img_file)
            break

    return images_by_case


def discover_captions(images_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover per-case captions from the four sidecar JSON files and merge
    them into one dict per case.

    Every generator writes its sidecar keyed by schema.json's
    "images_needed" names already ("3d_left", "panoramic",
    "tooth_48_composite", "sinus_right_detail"), so
    keys are used verbatim -- no rewriting. Two scripts writing the same key
    for one case would be a bug in those scripts, not something to reconcile
    here, so a later file simply wins.
    """
    captions_by_case: Dict[str, Dict[str, str]] = {}

    images_path = Path(images_dir)
    for suffix in _CAPTION_SIDECAR_SUFFIXES:
        for cap_file in images_path.glob(f"*{suffix}"):
            case_id = cap_file.name[: -len(suffix)]
            if not case_id:
                continue

            try:
                data = json.loads(cap_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Could not read {cap_file}: {e}")
                continue

            captions_by_case.setdefault(case_id, {}).update(data)

    return captions_by_case


def to_relative(path: str, base_dir: Path) -> str:
    """
    Convert an absolute image path to one relative to base_dir (e.g. PROJECT_DIR).

    Storing relative paths means qa_pairs.jsonl works unchanged whether it's
    read on the host (resolve against PROJECT_DIR) or inside a container
    (resolve against the mount point, e.g. /project) -- no path rewriting
    needed downstream, just join with whatever base is in scope.
    """
    try:
        return str(Path(path).resolve().relative_to(base_dir.resolve()))
    except ValueError:
        # Not under base_dir for some reason -- keep as-is rather than fail.
        print(f"[WARN] Path not under project dir, keeping absolute: {path}")
        return path


def resolve_call_plan(schema: Dict) -> "OrderedDict[str, Tuple[Tuple[str, ...], List[Dict]]]":
    """
    Turn CALL_PLAN into the concrete call list, validated against the schema.

    Returns OrderedDict[call_key, (images_tuple, facts)] in plan order.

    Every check below is a HARD FAILURE rather than a warning, because each
    one describes a call that would go out subtly wrong and score badly
    without ever looking broken:

      - unknown fact_id: a typo, or a fact the schema renamed. Skipping it
        would silently drop the question from the run.
      - fact in the schema but in no call: the commonest drift, since adding
        a fact to schema.json is exactly when you forget to place it. It
        would never be asked, and its output_field would be missing from
        every prediction.
      - fact placed in two calls: asked twice, and the second answer wins
        when the calls are merged in run_vqa_inference.merge_dicts.
      - image mismatch: the plan sends a fact an image its own images_needed
        doesn't name -- e.g. schema.json moves a fact from panoramic to
        3d_frontal and the plan still points it at the panoramic, so the
        model is asked to read a finding off an image that cannot show it.
    """
    facts_by_id: "OrderedDict[str, Dict]" = OrderedDict()
    for section in GLOBAL_SECTIONS:
        for fact in schema.get(section, []) or []:
            fact_id = fact["fact_id"]
            if fact_id in facts_by_id:
                raise ValueError(f"schema.json declares '{fact_id}' twice")
            facts_by_id[fact_id] = dict(fact, **{SECTION_KEY: section})

    calls: "OrderedDict[str, Tuple[Tuple[str, ...], List[Dict]]]" = OrderedDict()
    placed: Dict[str, str] = {}

    for call_key, images_tuple, fact_ids in CALL_PLAN:
        if call_key in calls:
            raise ValueError(f"CALL_PLAN uses the call key '{call_key}' twice")
        call_facts = []
        for fact_id in fact_ids:
            fact = facts_by_id.get(fact_id)
            if fact is None:
                raise ValueError(
                    f"CALL_PLAN call '{call_key}' names '{fact_id}', which is "
                    f"not a fact in schema.json's {list(GLOBAL_SECTIONS)} sections")
            if fact_id in placed:
                raise ValueError(
                    f"'{fact_id}' is in two calls ('{placed[fact_id]}' and "
                    f"'{call_key}') -- it would be asked twice and the later "
                    f"answer would overwrite the earlier one")
            needed = list(fact.get("images_needed", []))
            if sorted(needed) != sorted(images_tuple):
                raise ValueError(
                    f"CALL_PLAN call '{call_key}' sends {list(images_tuple)} to "
                    f"'{fact_id}', but that fact's images_needed is {needed}")
            placed[fact_id] = call_key
            call_facts.append(fact)
        calls[call_key] = (tuple(images_tuple), call_facts)

    unplaced = [f for f in facts_by_id if f not in placed]
    if unplaced:
        raise ValueError(
            f"{len(unplaced)} schema fact(s) are in no CALL_PLAN call and would "
            f"never be asked: {unplaced}")

    oversized = [(k, len(f)) for k, (_, f) in calls.items() if len(f) > MAX_FACTS_PER_CALL]
    for call_key, n in oversized:
        print(f"[WARN] call '{call_key}' carries {n} questions (soft cap "
              f"{MAX_FACTS_PER_CALL}) -- more than the model reliably holds in "
              f"one response")

    return calls


# ── object_fields spec parsing ───────────────────────────────────────────────
#
# An object_fields value is a TYPE HEAD followed by free prose, e.g.
#   "bool -- true if at least one edentulous region exists in this arch"
#   "enum:none|mild|moderate|severe, only when present == true"
#   "list[object] -- ... each: {fdi_number: int, with_crown: bool}"
#   "object -- keys are FDI numbers ..., values: enum normal|filling|impacted"
# Everything after the head is guidance for the model, NOT part of the type,
# so the head has to be matched with a regex rather than by str.startswith +
# str.split: "enum:radiopaque|radiolucent, only when present == true" split on
# "|" yields a bogus final enum value carrying the whole trailing clause, and
# that bogus value then goes into guided_json as a legal answer.
_SPEC_HEAD_RE = re.compile(
    r"^\s*(list\[object\]|list\[int\]|list\[string\]"
    r"|enum:[A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*|object|bool|string|int)")

# "... each: {fdi_number: int, location: string, with_crown: bool}" -- the
# per-member shape of a list[object] sub-field, written inline in the schema.
_EACH_RE = re.compile(r"each\s*:\s*\{(.+)\}", re.S)

# "... values: enum normal|filling|post_and_core" -- the value type of a
# per-tooth object sub-field (dental_arch_findings' FDI -> finding map).
#
# Case-insensitive on purpose. It used to require a lowercase "values:", and a
# schema edit that started the sentence with "Values:" made this silently miss
# -- the guided-decode value type degraded to a free string with no enum, so
# nothing constrained the model to the vocabulary. A prose capitalisation must
# not be able to switch off a decoding constraint.
_VALUES_ENUM_RE = re.compile(
    r"values\s*:\s*enum[:\s]+([A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)", re.I)


def spec_head(spec_str: str) -> Optional[str]:
    """'enum:a|b, only when ...' -> 'enum:a|b'; unrecognized head -> None."""
    if not isinstance(spec_str, str):
        return None
    m = _SPEC_HEAD_RE.match(spec_str)
    return m.group(1) if m else None


def enum_values_of(spec_str: str) -> List[str]:
    """Enum members of a spec string, stopping before any trailing prose."""
    head = spec_head(spec_str) or ""
    return head[len("enum:"):].split("|") if head.startswith("enum:") else []


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split on `sep`, ignoring separators inside {}, [] or quotes."""
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
    return [p for p in parts if p.strip()]


def parse_each_fields(spec_str: str) -> "OrderedDict[str, str]":
    """
    The inline "each: {...}" member shape of a list[object] sub-field, as
    {member_field: member_spec}. Empty when the spec has no such block (the
    caller then falls back to an unconstrained object, which is a weaker
    constraint but never a wrong one).
    """
    out: "OrderedDict[str, str]" = OrderedDict()
    m = _EACH_RE.search(spec_str or "")
    if not m:
        return out
    for part in _split_top_level(m.group(1)):
        key, sep, rest = part.partition(":")
        if not sep or not key.strip():
            continue
        out[key.strip()] = rest.strip()
    return out


# FDI ranges an arch-scoped fact may legitimately answer with, keyed by the
# suffix its output_field carries.
#
# Ordered as the ANATOMICAL SWEEP each fact's question_text already asks for
# ("Going tooth by tooth through FDI 18-11, 21-28"), not ascending: a
# constrained decode emits object properties in schema order, so this is the
# order the model actually answers in, and a sweep that walks the arch
# end-to-end keeps it in step with the picture. Ascending order made it jump
# from 18 back to 11 mid-arch. Membership is unaffected -- these are also used
# as the arch's legal key set.
_ARCH_FDIS = {
    "_mandible": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
    "_maxilla":  [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
}
assert all(sorted(v) == sorted(f for f in ALL_FDIS if (f >= 31) == (k == "_mandible"))
           for k, v in _ARCH_FDIS.items()), "arch FDI sweep must cover exactly its arch"


def arch_fdis_of(output_field: str) -> Optional[List[int]]:
    """The arch FDI range implied by a fact's own name, or None."""
    for suffix, fdis in _ARCH_FDIS.items():
        if output_field.endswith(suffix):
            return fdis
    return None


def json_schema_for_spec(spec_str: str, arch_fdis: Optional[List[int]] = None) -> Dict:
    """
    JSON Schema for ONE object_fields entry (a sub-field of a fact).

    arch_fdis, when given, is the FDI range the owning fact covers; a
    free-keyed per-tooth map is then constrained to exactly those FDI keys.
    """
    head = spec_head(spec_str)

    if head and head.startswith("enum:"):
        values = enum_values_of(spec_str)
        return {"type": ["string", "null"], "enum": values + [None]}
    if head == "list[int]":
        return {"type": "array", "items": {"type": "integer"}}
    if head == "list[string]":
        return {"type": "array", "items": {"type": "string"}}
    if head == "list[object]":
        members = parse_each_fields(spec_str)
        if not members:
            # No inline member shape to bind to -- constrain the container
            # only. A wrong-but-plausible member schema would be worse than
            # none: guided decoding would then force the model AWAY from the
            # keys the prose actually asked for.
            return {"type": "array", "items": {"type": "object"}}
        props = {k: json_schema_for_spec(v) for k, v in members.items()}
        return {"type": "array",
                "items": {"type": "object", "properties": props,
                          "required": list(props), "additionalProperties": False}}
    if head == "object":
        # Per-tooth map (dental_arch_findings: FDI string -> finding enum).
        #
        # Expressed as 16 properties, one per FDI in the arch, rather than with
        # propertyNames/additionalProperties alone: those keywords are the ones
        # guided-decoding backends most often reject or ignore, and a rejected
        # schema fails the whole call. Plain properties are Draft-7 that every
        # backend compiles.
        #
        # It also fixes a real failure mode the code comments elsewhere in this
        # repo already document: both arch calls see the SAME full-mouth
        # panoramic and are scoped by prompt alone, so the model regularly
        # answers with the other arch's teeth. Here it physically cannot.
        #
        # ALL SIXTEEN ARE REQUIRED (v6.2). They used to be optional, because
        # the map carried "present teeth only" and the KEY SET was the answer
        # to which teeth exist -- so requiring all 16 would have destroyed the
        # fact. v6.2 moved absence into the VALUES ("absent" is an enum member
        # now), which is what makes a fixed key set possible: a tooth the
        # reader never got to is no longer indistinguishable from one that is
        # not in the mouth, because every position must be answered.
        if arch_fdis:
            values = _VALUES_ENUM_RE.search(spec_str or "")
            value_schema = ({"type": "string", "enum": values.group(1).split("|")}
                            if values else {"type": "string"})
            props = {str(f): value_schema for f in arch_fdis}
            return {"type": "object", "properties": props,
                    "required": list(props), "additionalProperties": False}
        return {"type": "object"}
    if head == "bool":
        return {"type": ["boolean", "null"]}
    if head == "int":
        return {"type": ["integer", "null"]}
    return {"type": ["string", "null"]}


def describe_schema_entry(fact: Dict, fdi: Optional[int] = None):
    """
    The PROMPT-side rendering of one fact's shape.

    Object facts render as a nested {sub_field: spec} map carrying the
    schema's own prose verbatim -- that prose is where v6.1 keeps its
    recognition criteria ("only when present == true", "length must equal
    quadrant_3 + quadrant_4"), and paraphrasing it into "one of [...]" threw
    exactly that away.

    `fdi` fills the {fdi} placeholders inside that prose, the same way
    build_questions_block fills them in output_field / question_text /
    description. Without it a per-tooth call shipped the placeholder raw --
    the model read "gates tooth_{fdi}_endodontic_treatment" and had to guess
    which tooth that meant.
    """
    fact_type = fact["type"]

    if fact_type == "enum":
        return f"one of {fact.get('enum_values', [])}"
    if fact_type == "list[int]":
        return "list of integers (FDI numbers)"
    if fact_type == "list[string]":
        return "list of strings"
    if fact_type in ("object", "list[object]"):
        obj = {k: (v.replace("{fdi}", str(fdi)) if fdi is not None and isinstance(v, str) else v)
               for k, v in (fact.get("object_fields", {}) or {}).items()}
        return [obj] if fact_type == "list[object]" else obj
    if fact_type == "bool":
        return "true or false"
    if fact_type == "int":
        return "integer"
    return "string"


def json_schema_entry(fact: Dict) -> Dict:
    """
    Machine-readable JSON Schema for one fact, used for vLLM guided decoding.

    describe_schema_entry()'s prose is what goes in the PROMPT, and prose
    alone does not bind the model -- it sometimes echoes that notation back
    as a string value instead of emitting a real nested object, which used to
    crash postprocessing. This is the same information as a constraint the
    sampler must obey.

    Sub-field order is preserved from the schema, which is why every
    object_fields block leads with "visual_evidence": the constrained decode
    emits the fields in this order, so the model describes what it sees
    BEFORE it commits to the enum values that must agree with the description.
    """
    fact_type = fact["type"]

    if fact_type == "enum":
        values = fact.get("enum_values", [])
        return {"type": ["string", "null"], "enum": values + [None]} if values else {"type": "string"}
    if fact_type == "list[int]":
        return {"type": "array", "items": {"type": "integer"}}
    if fact_type == "list[string]":
        return {"type": "array", "items": {"type": "string"}}
    if fact_type in ("object", "list[object]"):
        arch_fdis = arch_fdis_of(fact.get("output_field", ""))
        props = {key: json_schema_for_spec(spec, arch_fdis)
                 for key, spec in (fact.get("object_fields", {}) or {}).items()}
        obj = {"type": "object", "properties": props,
               "required": list(props), "additionalProperties": False}
        return {"type": "array", "items": obj} if fact_type == "list[object]" else obj
    if fact_type == "bool":
        return {"type": ["boolean", "null"]}
    if fact_type == "int":
        return {"type": ["integer", "null"]}
    return {"type": ["string", "null"]}


# Facts reference the schema's shared vocabulary as "_definitions.<key>" in
# their description. Only the definitions a call actually references are
# shipped with it, so a 1-fact sinus call doesn't carry the impaction and
# restoration vocabulary too.
_DEFINITION_REF_RE = re.compile(r"_definitions\.([A-Za-z0-9_]+)")


def build_guidance_block(facts: List[Dict], definitions: Dict[str, str],
                         visual_evidence_rule: Optional[str] = None,
                         facts_by_id: Optional[Dict[str, Dict]] = None) -> str:
    """
    The shared-vocabulary preamble for one call: the visual-evidence rule
    plus every _definitions entry these facts' descriptions reference.

    Without this the v6.1 descriptions are dangling pointers -- "See
    _definitions.restoration_types" tells the model nothing if the
    definitions never reach the prompt.

    References are followed THROUGH the schema's "same criteria as X" idiom,
    which every mirrored fact uses: dental_arch_findings_maxilla's whole
    description is "Same priority and exclusions as
    dental_arch_findings_mandible", so it names no definition of its own and
    the maxillary tooth survey was being asked to tell a filling from a
    post-and-core with none of the vocabulary that defines them -- while the
    mandibular survey, which spells the references out, got all of it.

    A fact inherits from a fact_id it mentions ONLY IF it names no
    _definitions of its own. That condition is what keeps inheritance to real
    mirroring: a fact that cites its own vocabulary is self-sufficient, and a
    passing mention of another fact ("Distinct from the per-tooth value in
    dental_arch_findings_mandible", in the periodontal facts) is a
    cross-reference for the reader, not a claim to share its definitions.
    Without it, pano_others_mandible dragged in the whole restoration
    vocabulary it never uses -- and pano_others_maxilla, whose facts phrase
    the same relationship differently, did not, leaving the two arches asked
    the same questions with different prompts.
    """
    facts_by_id = facts_by_id or {}
    keys: List[str] = []

    def direct_refs(fact: Dict) -> List[str]:
        return [k for k in _DEFINITION_REF_RE.findall(fact.get("description", "") or "")
                if k in definitions]

    def add(found: List[str]) -> None:
        for key in found:
            if key not in keys:
                keys.append(key)

    for fact in facts:
        own = direct_refs(fact)
        if own:
            add(own)
            continue
        description = fact.get("description", "") or ""
        for other_id, other in facts_by_id.items():
            if other_id != fact.get("fact_id") and other_id in description:
                add(direct_refs(other))

    lines: List[str] = []
    if visual_evidence_rule:
        lines.append(visual_evidence_rule)
    for key in keys:
        lines.append(f"{key}: {definitions[key]}")
    return "\n\n".join(lines)


def build_questions_block(facts: List[Dict], fdi: Optional[int] = None,
                          definitions: Optional[Dict[str, str]] = None,
                          visual_evidence_rule: Optional[str] = None,
                          facts_by_id: Optional[Dict[str, Dict]] = None) -> Dict[str, str]:
    """
    Combine a list of facts into one numbered questions block + one output
    schema block. If `fdi` is given, "{fdi}" placeholders in output_field /
    question_text / description are filled in (used for dental_elements facts).

    Emits BOTH renderings of the same schema: "output_schema" (prose, goes in
    the prompt) and "json_schema" (a real JSON Schema, passed to vLLM as
    guided_json so the model physically cannot answer with the wrong shape),
    plus "guidance" (the shared _definitions vocabulary these facts point at).

    Each question carries its fact's own "description" -- that field is where
    v6.1 puts the visual-recognition criteria (what a post-and-core looks like
    versus a crown, what counts as a canal fill), and a question asked without
    it is a materially harder question.
    """

    questions_lines = []
    schema_obj = {}
    json_props = {}

    for i, fact in enumerate(facts, 1):
        field = fact["output_field"]
        question_text = fact["question_text"]
        description = fact.get("description", "") or ""
        if fdi is not None:
            field = field.format(fdi=fdi)
            question_text = question_text.format(fdi=fdi)
            description = description.replace("{fdi}", str(fdi))
        questions_lines.append(f"{i}. [{field}] {question_text}")
        if description:
            questions_lines.append(f"   HOW TO JUDGE: {description}")
        schema_obj[field] = describe_schema_entry(fact, fdi=fdi)
        json_props[field] = json_schema_entry(fact)

    return {
        "questions": "\n".join(questions_lines),
        "guidance": build_guidance_block(facts, definitions or {}, visual_evidence_rule,
                                         facts_by_id),
        "output_schema": json.dumps(schema_obj, indent=2),
        "json_schema": {
            "type": "object",
            "properties": json_props,
            "required": list(json_props),
            "additionalProperties": False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING -- the half that makes this the preprocess driver
#
# build_vqa_pairs.py plans from the FILES ON DISK: whatever images and caption
# sidecars exist under --images-dir. That means the payload is only as complete
# as the render, and for most of this project's life the render was a separate
# job you had to remember to run first -- one script per generator, plus a
# fifth that looped over cases.
#
# Given --dataset-dir it now renders what is missing before it plans, so
# "images -> qa_pairs" is one command and a partial render cannot silently
# produce a partial payload. Without --dataset-dir the behaviour is exactly as
# before: plan from disk, render nothing.
# ─────────────────────────────────────────────────────────────────────────────

#: Per-generator completion markers, in dependency order. Each generator writes
#: its own, so a run interrupted after the 3D renders resumes at the sinuses
#: rather than from the top.
#:
#: The composites key on {case}_coverage.json and NOT on {case}_tooth_captions
#: .json, which is the obvious choice and is wrong: create_tooth_detail.py
#: writes the caption sidecar only `if captions`, so an edentulous case
#: finishes perfectly and leaves none. 4 of the 40 validate cases are like
#: that, and keying on it re-renders each of them on every resumed run while
#: reporting nothing amiss. coverage.json is written unconditionally.
GENERATORS = [
    ("panoramic",  "create_panoramic.py",    "{case}_panoramic_caption.json"),
    ("3d",         "create_3d_renders.py",   "{case}_3d_captions.json"),
    ("sinus",      "create_sinus_detail.py", "{case}_sinus_captions.json"),
    ("composites", "create_tooth_detail.py", "{case}_coverage.json"),
]


def _repo_module(name: str) -> str:
    """Absolute path of a sibling generator, via code/_repo.py."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(next(p for p in _P(__file__).resolve().parents
                               if (p / "_repo.py").is_file())))
    from _repo import module_path
    return str(module_path(name))


def render_case(case: str, dataset_dir: Path, images_dir: Path,
                no_facts: bool, force: bool, threads: int,
                facts_dir: Optional[Path] = None) -> str:
    """Render one case's four image sets. Returns a one-line status.

    Serial within the case on purpose. The concurrency that matters for a whole
    split is BETWEEN cases (--workers), and nesting the two oversubscribes the
    allocation: three generators x N workers competing for the same cores is
    slower than running serially, not faster.
    """
    volume = dataset_dir / "images" / f"{case}_0000.nii.gz"
    mask = dataset_dir / "masks" / f"{case}.nii.gz"
    # WHICH facts is a real choice, not a path detail. dataset/<split>/facts
    # is what gen_images_cpu.sh and aksssr_pipeline.sh have always rendered
    # from, and is the default here so this driver reproduces them. infer.py
    # renders from an AUDITED copy instead -- audit_facts.py corrects facts
    # the mask contradicts, and create_panoramic.py filters its outlines by
    # facts.structured.teeth_present, so the two can differ in PIXELS and not
    # merely in caption text. Pass --facts-dir to render from the audited
    # copy; the default is deliberately the historical behaviour.
    facts = (facts_dir or dataset_dir / "facts") / f"{case}.json"

    # Preflight, so one absent file skips a case instead of killing a 582-case
    # pool partway through.
    if not volume.is_file():
        return f"[SKIP] {case} -- no volume: {volume}"
    if not mask.is_file():
        return f"[SKIP] {case} -- no mask: {mask}"
    if not no_facts and not facts.is_file():
        return f"[SKIP] {case} -- no facts: {facts}"

    # Per-worker thread caps. Without these every worker's BLAS grabs all the
    # allocated cores and --workers 4 oversubscribes 4x.
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))

    common = ["--case-id", case, "--out-dir", str(images_dir)]
    fct = ["--no-facts"] if no_facts else ["--facts-file", str(facts)]
    argv = {
        "create_panoramic.py":    common + ["--volume", str(volume), "--mask", str(mask)] + fct,
        "create_3d_renders.py":   common + ["--mask", str(mask)] + fct,
        "create_sinus_detail.py": common + ["--volume", str(volume), "--mask", str(mask)],
        "create_tooth_detail.py": common + ["--volume", str(volume), "--mask", str(mask)] + fct
                                  + ["--pano-dir", str(images_dir),
                                     "--render-dir", str(images_dir)],
    }

    did = []
    for label, script, marker in GENERATORS:
        if not force and (images_dir / marker.format(case=case)).exists():
            did.append(f"{label}:skip")
            continue
        proc = subprocess.run([sys.executable, _repo_module(script)] + argv[script],
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            return (f"[FAIL] {case} -- {script} exited {proc.returncode}\n"
                    f"{proc.stderr[-1200:]}")
        did.append(label)
    return f"[PASS] {case} -- {' '.join(did)}"


def render_missing(cases: List[str], dataset_dir: Path, images_dir: Path,
                   no_facts: bool, force: bool, workers: int,
                   facts_dir: Optional[Path] = None) -> int:
    """Render every case that needs it. Returns the number that failed.

    A ThreadPoolExecutor is enough: every generator is a subprocess, so the GIL
    is not in the way and threads only have to wait on them.

    DO NOT run two of these over the same --images-dir at once. The completion
    marker is written at the END of each generator, so two workers that start
    the same case both see "not done" and race on the same PNG paths. Within
    one process that cannot happen -- the case list is partitioned -- but two
    concurrent JOBS is the shape to avoid.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 8)
    threads = max(1, cpus // max(1, workers))
    todo = [c for c in cases
            if force or not all((images_dir / m.format(case=c)).exists()
                                for _, _, m in GENERATORS)]
    print(f"[INFO] render: {len(todo)} of {len(cases)} case(s) need work, "
          f"{workers} worker(s) x {threads} thread(s) of {cpus} cpu(s)")
    if not todo:
        return 0

    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_case, c, dataset_dir, images_dir,
                               no_facts, force, threads, facts_dir): c
                   for c in todo}
        for fut in cf.as_completed(futures):
            line = fut.result()
            print(line, flush=True)
            if line.startswith("[FAIL]"):
                failed += 1
    if failed:
        print(f"[WARN] {failed} case(s) failed to render; they will be absent "
              f"from the payload", file=sys.stderr)
    return failed


def cases_in(dataset_dir: Path) -> List[str]:
    """Every case the dataset dir has a volume for."""
    return sorted(p.name[:-len("_0000.nii.gz")]
                  for p in (dataset_dir / "images").glob("*_0000.nii.gz"))


def build_vqa_records(schema_path: str, images_dir: str, out_path: str,
                       project_dir: str,
                       cases: Optional[List[str]] = None,
                       limit: Optional[int] = None,
                       include_captions: bool = True) -> None:

    with open(schema_path) as f:
        schema = json.load(f)

    print(f"[INFO] Loaded schema: {schema_path} (version {schema.get('version', '?')})")

    # Shared vocabulary the facts' descriptions point at ("_definitions.x"),
    # shipped per call with only the entries that call references.
    definitions = schema.get("_definitions", {}) or {}
    visual_evidence_rule = schema.get("_visual_evidence_rule")
    print(f"[INFO] Shared definitions: {len(definitions)}"
          f"{' + visual-evidence rule' if visual_evidence_rule else ''}")

    # Index of every arch-level fact, so build_guidance_block can follow a
    # "same criteria as <fact_id>" description to the definitions that fact
    # references.
    facts_by_id = {f["fact_id"]: f
                   for section in GLOBAL_SECTIONS
                   for f in schema.get(section, []) or []}

    project_dir_path = Path(project_dir)
    print(f"[INFO] Storing image paths relative to: {project_dir_path.resolve()}")

    images_by_case = discover_images(images_dir)
    print(f"[INFO] Discovered images for {len(images_by_case)} case(s)")

    captions_by_case = discover_captions(images_dir) if include_captions else {}
    if include_captions:
        print(f"[INFO] Discovered captions for {len(captions_by_case)} case(s)")

    case_coverage = discover_coverage(images_dir)
    if case_coverage:
        n_no_max = sum(1 for c in case_coverage.values()
                       if not c.get("maxilla", {}).get("teeth", True))
        print(f"[INFO] Coverage sidecars for {len(case_coverage)} case(s); "
              f"{n_no_max} with no maxillary tooth in the mask")
    else:
        print("[INFO] No coverage sidecars found — every arch call planned as "
              "before (re-run create_tooth_detail.py to write them)")

    if cases:
        images_by_case = {k: v for k, v in images_by_case.items() if k in cases}
        print(f"[INFO] Filtered to {len(images_by_case)} case(s)")

    if limit:
        images_by_case = dict(list(images_by_case.items())[:limit])
        print(f"[INFO] Limited to {limit} case(s)")

    # ── Precompute the arch-level calls (same for every case) ───────────
    # call_blocks[call_key] = (images_tuple, questions_block)
    calls = resolve_call_plan(schema)
    call_blocks: "OrderedDict[str, Tuple[Tuple[str, ...], Dict[str, str]]]" = OrderedDict(
        (call_key, (images_tuple,
                    build_questions_block(call_facts, definitions=definitions,
                                          visual_evidence_rule=visual_evidence_rule,
                                          facts_by_id=facts_by_id)))
        for call_key, (images_tuple, call_facts) in calls.items()
    )
    summary = ", ".join(f"{k}={len(v[1])}f" for k, v in calls.items())
    print(f"[INFO] {GLOBAL_GROUP}: {len(calls)} call(s) -> {summary}")

    tooth_facts = schema.get(TOOTH_SECTION, [])
    # Which facts this FDI is even asked (see fact_applies_to_fdi): all of
    # them at most positions, minus the FDI-gated ones elsewhere.
    tooth_facts_by_fdi = {fdi: tooth_facts_for_fdi(tooth_facts, fdi)
                          for fdi in ALL_FDIS}
    # Every tooth fact is answered in ONE call per FDI, so that call needs the
    # UNION of what THAT FDI's facts ask for -- not fact[0]'s list. They all
    # point at tooth_{fdi}_composite today, but reading only the first fact
    # would silently drop an image the moment one of them diverged, and taking
    # the union over ALL facts would pull in an image only a gated fact needs.
    def _images_needed(facts: List[Dict]) -> List[str]:
        keys: List[str] = []
        for fact in facts:
            for key in fact.get("images_needed", []):
                if key not in keys:
                    keys.append(key)
        return keys

    tooth_images_needed = _images_needed(tooth_facts)
    gated = {f["fact_id"]: f[FDI_SCOPE_KEY] for f in tooth_facts
             if f.get(FDI_SCOPE_KEY)}
    print(f"[INFO] {TOOTH_SECTION}: {len(tooth_facts)} fact(s)/tooth -> 1 call/FDI "
          f"({len(tooth_images_needed)} image(s)/tooth: {'+'.join(tooth_images_needed)})")
    for fact_id, scope in gated.items():
        print(f"[INFO] {TOOTH_SECTION}: {fact_id} asked at "
              f"{len(scope)} FDI(s) only: {scope}")
    # Tooth block depends on {fdi}, so compute once per FDI (32 total), reused for every case.
    tooth_block_by_fdi = {
        fdi: build_questions_block(tooth_facts_by_fdi[fdi], fdi=fdi,
                                   definitions=definitions,
                                   visual_evidence_rule=visual_evidence_rule,
                                   facts_by_id={f["fact_id"]: f for f in tooth_facts})
        for fdi in ALL_FDIS
    }
    # Tooth image keys also depend on {fdi} -- precompute the formatted tuple per FDI.
    tooth_image_keys_by_fdi = {
        fdi: [k.format(fdi=fdi) for k in _images_needed(tooth_facts_by_fdi[fdi])]
        for fdi in ALL_FDIS
    }

    records = []

    for case_id in sorted(images_by_case.keys()):
        case_images = images_by_case[case_id]
        case_captions = captions_by_case.get(case_id, {})

        print(f"\n[{case_id}] Building VQA record...")

        record = {"case_id": case_id}

        # Carried into the prediction by run_vqa_inference.py and read by
        # postprocess_pred.py. It has to travel with the payload because
        # "this tooth has no composite" is ambiguous downstream once an arch
        # can be skipped wholesale: no label in the mask (absent) vs not asked
        # about (unknown). Without this, an excluded maxilla whose teeth ARE
        # segmented comes out as "Complete edentulism".
        if case_id in case_coverage:
            record["coverage"] = case_coverage[case_id]

        # ── Arch-level calls, per CALL_PLAN ─────────────────────────────
        group_calls = {}
        dropped = []
        skipped_no_maxilla = []

        for call_key, (images_tuple, questions_block) in call_blocks.items():
            present = [k for k in images_tuple if k in case_images]
            if not present:
                dropped.append(call_key)
                continue
            if call_key in MAXILLA_ARCH_CALLS and not case_coverage.get(
                    case_id, {}).get("maxilla", {}).get("teeth", True):
                skipped_no_maxilla.append(call_key)
                continue
            call = {
                "images": {
                    k: to_relative(case_images[k], project_dir_path)
                    for k in present
                },
                "questions": questions_block,
            }
            if include_captions:
                call["captions"] = {
                    k: case_captions[k] for k in present if k in case_captions
                }
            group_calls[call_key] = call

        record[GLOBAL_GROUP] = group_calls

        if group_calls:
            print(f"  ✓ {GLOBAL_GROUP}: {list(group_calls.keys())}")
        if dropped:
            print(f"  [WARN] {GLOBAL_GROUP}: dropped call(s), no images found: {dropped}")
        if skipped_no_maxilla:
            print(f"  ⊘ {GLOBAL_GROUP}: skipped {skipped_no_maxilla} — no maxillary "
                  f"bone AND no maxillary tooth in the mask")

        # ── dental_elements (all 32 FDIs; entry always present) ─────────
        record["dental_elements"] = {}
        n_with_any_image = 0
        for fdi in ALL_FDIS:
            needed_keys = tooth_image_keys_by_fdi[fdi]
            present = {k: case_images[k] for k in needed_keys if k in case_images}

            if present:
                n_with_any_image += 1

            entry = {
                "images": {k: to_relative(v, project_dir_path) for k, v in present.items()},
                "questions": tooth_block_by_fdi[fdi],
            }
            if include_captions:
                entry["captions"] = {
                    k: case_captions[k] for k in present if k in case_captions
                }
            record["dental_elements"][f"tooth_{fdi}"] = entry

        print(f"  ✓ dental_elements: teeth with >=1 image "
              f"{n_with_any_image}/{len(ALL_FDIS)}")

        if not record[GLOBAL_GROUP] and n_with_any_image == 0:
            print(f"  ✗ No usable images at all for {case_id}, skipping case")
            continue

        records.append(record)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n[INFO] ========== VQA Records Generated ==========")
    print(f"[INFO] Total cases: {len(records)}")
    print(f"[INFO] Output: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build batched qa_pairs.jsonl directly from schema.json (v6.1)")
    ap.add_argument("--schema", required=True,
                     help="Path to schema.json (single source of truth)")
    ap.add_argument("--images-dir", required=True,
                     help="Directory with generated images (and caption sidecar JSONs)")
    ap.add_argument("--out", required=True,
                     help="Output path for qa_pairs.jsonl (one record per case)")
    ap.add_argument("--project-dir", required=True,
                     help="Project root directory. Image paths are stored in "
                          "qa_pairs.jsonl RELATIVE to this directory, so the "
                          "file works unchanged whether read on the host or "
                          "inside a container (just resolve against whatever "
                          "base is in scope there).")
    ap.add_argument("--cases", nargs="+", default=None,
                     help="Filter case IDs")
    ap.add_argument("--limit", type=int, default=None,
                     help="Limit to N cases (for testing)")
    ap.add_argument("--no-captions", action="store_true",
                     help="Don't read/attach the caption sidecar JSON files")

    # ── rendering: given a dataset dir, make the images this then plans from ──
    ap.add_argument("--dataset-dir", type=Path, default=None,
                     help="dataset/<split>. Given this, any case missing images "
                          "is RENDERED before the payload is planned, so a "
                          "partial render cannot silently produce a partial "
                          "payload. Omitted, nothing is rendered and this plans "
                          "from whatever is already in --images-dir, exactly as "
                          "before.")
    ap.add_argument("--workers", type=int, default=1,
                     help="cases rendered in parallel. Generators run SERIALLY "
                          "within a case, so this is the only concurrency knob "
                          "and threads-per-worker is derived from it. 1 case is "
                          "fine at 1; a 582-case split is not.")
    ap.add_argument("--force-render", action="store_true",
                     help="re-render even where the completion markers exist")
    ap.add_argument("--facts-dir", type=Path, default=None,
                     help="where the facts files come from (default: "
                          "<dataset-dir>/facts, which is what every batch "
                          "render in this project has used). Point it at an "
                          "AUDITED facts dir to reproduce infer.py, which "
                          "renders from one -- the two can differ in pixels, "
                          "not just captions.")
    ap.add_argument("--no-facts", action="store_true",
                     help="render with no facts file at all -- the NO_FACTS "
                          "ablation. Changes the PIXELS, not just the captions: "
                          "create_panoramic.py filters its outlines by "
                          "facts.structured.teeth_present.")

    args = ap.parse_args()

    try:
        if args.dataset_dir:
            todo = args.cases or cases_in(args.dataset_dir)
            if args.limit:
                todo = todo[:args.limit]
            render_missing(todo, args.dataset_dir, Path(args.images_dir),
                            args.no_facts, args.force_render, args.workers,
                            facts_dir=args.facts_dir)
        build_vqa_records(args.schema, args.images_dir, args.out, args.project_dir,
                           cases=args.cases, limit=args.limit,
                           include_captions=not args.no_captions)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)