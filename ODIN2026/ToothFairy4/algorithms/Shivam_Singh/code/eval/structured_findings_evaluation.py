#!/usr/bin/env python3
"""
code/eval/structured_findings_evaluation.py

Fact-level survey: prediction and summary scored against the GENERATED ground
truth (`dataset/{split}/outputs/ground_truth/{case}*_gt.json`), not against the
hand-coded REPORT_GT table in survey_findings.py.

WHY A SECOND SURVEY RATHER THAN AN EDIT
───────────────────────────────────────
(The shared half is no longer duplicated OR imported from the other survey: the
claim extractors live in structured_findings_helper.py, which both read. What follows is
about the half that is genuinely different -- the ground truth.)
survey_findings.py scores against REPORT_GT, a hand-curated {case: {category:
[fdi]}} table covering five categories, plus the facts file for absent teeth.
That last row is why its overview says "vs segmentation mask": no report
enumerates every position, so absence had to be scored against the mask, which
is not always right.

parse_reports_to_gt.py now emits ground truth in the SAME SHAPE as a prediction
-- {case_id, global, teeth} -- which changes what is possible. Absence, atrophy,
implants, bridges and the per-tooth findings all live in that file as ordinary
fields, so they can be scored directly and no axis has to fall back to the mask.

It is a separate module because swapping the GT source underneath
survey_findings.py would silently invalidate its own diff history: every
survey/survey_<ts>.txt it has written is mask-scored on that row, and a
GT-scored number in the same column would compare against a different question.

WHAT IT REUSES, AND WHY THAT MATTERS
────────────────────────────────────
Nothing here re-derives how a claim is read out of a prediction. The extractors
in survey_findings.py (pred_claims, pred_absent, pred_prosthetics, pred_anatomy
and their summary_* counterparts) already encode the awkward parts -- the
pre-v6.4 fallbacks, the ARCH_VALUE folding, the composite's restoration
priority, the rule that an arch whose pattern is "unknown" is sixteen UNREAD
positions rather than sixteen present teeth. Re-implementing any of that would
drift the moment the schema moves.

The ground truth is read with the PREDICTION extractors, not with separate ones.
That is the whole payoff of parse_reports_to_gt.py emitting prediction shape: a
GT file and a prediction file are the same kind of object, so `pred_claims(gt)`
and `pred_claims(pred)` are comparable by construction, with zero case-specific
logic on either side.

WHAT evaluate_predictions.py DID THAT THIS DOES NOT
──────────────────────────────────────────────────
That module was deleted on 2026-08-19. This one covers 68 of its 74 fields and
adds two it lacked (`dental_arch_findings_*.uncertain_teeth`), plus the SUMMARY
column, which it could not have -- it only ever read predictions. Of the six it
scored and this does not, five are on axes that measure nothing here: bone
quality's `location`/`type` and periodontal resorption's `pattern` sit under
axes already scored at their trivial constants, `primary_teeth` has N=0 in this
dataset, and `dentition_type` is derived rather than asked.

The one real capability lost is PER-CLASS confusion: it reported precision and
recall for each enum VALUE ({"buccal": {...}, "lingual": {...}}), where this
reports {scored, right} -- accuracy only. If the question is ever "which way is
this enum wrong" rather than "how often", that breakdown is not in this file and
would have to be written. `git log -- code/eval/evaluate_predictions.py` has it.

MULTI-READER POLICY
───────────────────
A case may have several reader files (A008_gt.json, F067_2_gt.json, ...). The
FIRST ALPHABETICALLY is used, matching official_ranking.py and the challenge's
own tarball convention, so this survey and the leaderboard are looking at the
same label.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple



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

from structured_findings_helper import (  # noqa: E402
    ALL_TEETH,
    ARCH_TEETH,
    CATEGORIES,
    _as_dict,
    _resolve_stage_dir,
    as_list,
    pred_absent,
    pred_anatomy,
    pred_claims,
    pred_prosthetics,
    summary_absent,
    summary_anatomy,
    summary_claims,
    summary_prosthetics,
)

# ── Call groups ──────────────────────────────────────────────────────────
# Which image a fact is read off, hence which VLM call answered it. Derived
# from the schema's own images_needed rather than listed here, so a fact that
# moves between images moves between groups without touching this file.
# The three groups are the ones worth separating operationally: the 3D renders
# and sinus crops are whole-jaw/regional reads, the panoramic is the arch
# survey, and the composites are the per-tooth detail.
GROUP_OF_IMAGE = {
    "3d_left": "3d", "3d_right": "3d", "3d_frontal": "3d",
    "sinus_right_detail": "sinus", "sinus_left_detail": "sinus",
    "panoramic": "panoramic",
    "tooth_{fdi}_composite": "detail",
}
GROUPS = ("3d", "sinus", "panoramic", "detail")

# pred_claims/pred_anatomy tag each claim with the source that made it; those
# tags predate this module and are the join back to a call group.
GROUP_OF_SOURCE = {"composite": "detail", "panoramic": "panoramic",
                   "3d_render": "3d+sinus", "sinus": "3d+sinus"}


def schema_groups(schema_path: Path) -> Dict[str, List[str]]:
    """{group: [output_field]} straight from the schema."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {g: [] for g in GROUPS}
    for section in ("mandible", "maxilla", "dental_elements"):
        for fact in schema.get(section) or []:
            images = fact.get("images_needed") or []
            groups = {GROUP_OF_IMAGE.get(i) for i in images} - {None}
            if len(groups) == 1:
                out[groups.pop()].append(fact.get("output_field") or "?")
    return out


# ── Global (non-per-tooth) axes ──────────────────────────────────────────
# Everything the per-tooth extractors do not cover. Each entry is
# (row label, fact name template, field, kind); the arch is substituted in.
# kind "enum" scores right/scored over the cases both sides answered; "bool"
# scores prec/rec treating true as the positive class.
#
# Each axis needs TWO paths, because a summary is not prediction-shaped: the
# prediction/GT side is `global.<fact>_<arch>.<field>`, the summary side is
# `<arch>.<...>`. `default` is what a MISSING summary block means -- postprocess
# writes alveolar_bone_atrophy and periodontal_bone_resorption only when there
# is something to say (`if atrophy: result[...] = atrophy`), so their absence is
# a negative answer, not an unanswered one. Where there is no default (None) a
# missing block is treated as unanswered and drops out of the denominator.
GLOBAL_AXES = [
    # label,               pred fact template,                   field,
    #                      summary path,                         kind,  default
    ("atrophy",            "alveolar_bone_atrophy_{arch}",       "atrophy",
     ("alveolar_bone_atrophy", "atrophy"),                       "bool", False),
    ("periodontal resorpt.", "periodontal_bone_resorption_{arch}", "extent",
     ("periodontal_bone_resorption", "extent"),                  "enum", "none"),
    ("bone quality",       "bone_quality_{arch}",                "present",
     ("bone_quality", "present"),                                "bool", None),
    ("primary teeth",      "primary_teeth_{arch}",               "primary_teeth",
     ("arch_findings", "primary_teeth"),                         "set",  set()),
]


# ── Side / single axes ───────────────────────────────────────────────────
# The per-SIDE reads: condyles, canals and sinuses answer right and left
# separately, so their unit is a side rather than an arch. Each entry gives the
# prediction-shape fact (with {side} substituted) and the summary-shape PATH,
# which is nested differently per axis -- `sinuses.{side}.mucosa_state` but
# `sinuses.scope.{side}` -- hence a path walk rather than a fixed two-level
# lookup.
SIDES = ("right", "left")
SIDE_AXES = [
    # `scope2`, not `enum`: postprocess folds the condyle summary to
    # included|not_included while the PREDICTIONS and the GT files still carry
    # the schema's three-way vocabulary, and the sinus scope axis below still
    # USES that vocabulary for real. So the fold has to be per-axis -- see _cast.
    ("condyle scope",  "mandible_condyle_{side}", "scope",
     ("mandible", "condyles", "{side}", "scope"), "scope2"),
    ("canal position", "mandible_canal_{side}",   "location",
     ("mandible", "canals", "{side}", "location"), "enum"),
    ("canal-adjacent teeth", "mandible_canal_{side}", "adjacent_teeth",
     ("mandible", "canals", "{side}", "adjacent_teeth"), "set"),
    ("sinus mucosa",   "maxilla_sinus_{side}",    "mucosa_state",
     ("maxilla", "sinuses", "{side}", "mucosa_state"), "enum"),
    ("sinus content",  "maxilla_sinus_{side}",    "sinus_content",
     ("maxilla", "sinuses", "{side}", "sinus_content"), "enum"),
    ("sinus scope",    "maxilla_sinus_{side}",    "scope",
     ("maxilla", "sinuses", "scope", "{side}"), "enum"),
]

# Facts answered once for the whole volume rather than per side or per arch.
# Mandible scope is deliberately absent: no fact asks it, postprocess derives
# it from the condyles ("scope_source": "derived_from_condyles"), so there is
# no prediction-side value to score it against.
SINGLE_AXES = [
    ("maxilla scope", "maxilla_scope", "maxilla_included",
     ("maxilla", "scope", "maxilla_included"), "enum"),
]


def _dig(doc: Dict, path: Tuple[str, ...], side: str = ""):
    """Walk a summary path, substituting {side}; None if any step is missing."""
    node = doc
    for step in path:
        node = _as_dict(node).get(step.format(side=side))
        if node is None:
            return None
    return node


def _cast(value, kind):
    if kind == "set":
        return {v for v in as_list(value) if isinstance(v, int)}
    if kind == "bool":
        return str(value).strip().lower() == "true"
    value = str(value).strip().lower()
    if kind == "scope2":
        # Binary inclusion: anything that is not an outright exclusion is
        # `included`, whichever vocabulary said so. This is what lets a GT file
        # and a raw prediction written under the three-way enum score against a
        # folded summary -- `partially_included` and `included` are the SAME
        # claim, and the grading between them is what postprocess drops.
        return "not_included" if value == "not_included" else "included"
    return value


def global_axis(doc: Dict, fact_tmpl: str, field: str, kind: str) -> Dict:
    """One global axis out of a PREDICTION- or GT-shaped doc -> {arch: value}.

    A missing or null field is left out rather than defaulted, so "never
    answered" stays distinguishable from "answered no" -- the same rule the
    per-tooth extractors follow.
    """
    facts = _as_dict(doc.get("global")) or doc
    out: Dict[str, object] = {}
    for arch in ARCH_TEETH:
        block = _as_dict(facts.get(fact_tmpl.format(arch=arch)))
        if not block:
            continue
        value = block.get(field)
        if value is None:
            continue
        out[arch] = _cast(value, kind)
    return out


def global_axis_summary(summary: Dict, path: Tuple[str, str],
                        kind: str, default) -> Dict:
    """Same axis out of a SUMMARY-shaped doc -> {arch: value}.

    `default` applies when the arch exists but the block does not, which for
    the conditionally-written axes is a real negative rather than a silence.
    """
    outer, field = path
    out: Dict[str, object] = {}
    for arch in ARCH_TEETH:
        arch_block = _as_dict(summary.get(arch))
        if not arch_block:
            continue
        value = _as_dict(arch_block.get(outer)).get(field)
        if value is None:
            if default is None:
                continue
            out[arch] = default
            continue
        out[arch] = _cast(value, kind)
    return out


# ── Scoring ──────────────────────────────────────────────────────────────
def pr(tp: int, fp: int, fn: int) -> str:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return f"{p:.2f}/{r:.2f}"


def score_sets(claimed: Set[int], truth: Set[int]) -> Counter:
    return Counter(tp=len(claimed & truth), fp=len(claimed - truth),
                   fn=len(truth - claimed))


def load_gt(gt_dir: Path, case_id: str) -> Optional[Dict]:
    """First reader alphabetically, or None if this case has no GT file."""
    hits = sorted(gt_dir.glob(f"{case_id}_gt.json")) + \
        sorted(gt_dir.glob(f"{case_id}_*_gt.json"))
    if not hits:
        return None
    return json.loads(hits[0].read_text(encoding="utf-8"))


def load_cases(run_dir: Path, gt_dir: Path,
               case_ids: Optional[List[str]]) -> List[Dict]:
    pred_dir = _resolve_stage_dir(run_dir / "predictions", "predictions", "*_pred.json")
    sum_dir = _resolve_stage_dir(run_dir / "summaries", "summaries", "*_summary.json")
    cases = []
    for pred_path in sorted(pred_dir.glob("*_pred.json")):
        case_id = pred_path.name[: -len("_pred.json")]
        if case_ids and case_id not in case_ids:
            continue
        gt = load_gt(gt_dir, case_id)
        if gt is None:
            print(f"[{case_id}] [WARN] no *_gt.json -- not scored", file=sys.stderr)
            continue
        sum_path = sum_dir / f"{case_id}_summary.json"
        cases.append({
            "case_id": case_id,
            "pred": json.loads(pred_path.read_text(encoding="utf-8")),
            "summary": (json.loads(sum_path.read_text(encoding="utf-8"))
                        if sum_path.exists() else {}),
            "gt": gt,
        })
    return cases


def field_kind(spec: str) -> str:
    """What KIND of comparison a schema object_fields entry admits.

    The schema states the type inline -- "bool -- true if...", "enum:a|b",
    "list[int] -- ..." -- so the type comes from the schema rather than a table
    here, and a field that changes type changes how it is scored without an
    edit. Free-text fields score as "skip": visual_evidence is prose and two
    correct readings will never be string-equal.
    """
    spec = (spec or "").strip().lower()
    if spec.startswith("bool"):
        return "bool"
    if spec.startswith("enum"):
        return "enum"
    if spec.startswith("list"):
        return "list"
    if spec.startswith("object"):
        # The per-FDI findings map -- the arch survey's actual output, and the
        # single most important field in the panoramic group. Scored position
        # by position rather than as one all-or-nothing dict comparison.
        return "map"
    if spec.startswith("int"):
        return "enum"          # scalar equality, e.g. implant counts
    return "skip"


def schema_facts(schema_path: Path) -> Dict[str, List[Tuple[str, str, Dict]]]:
    """{group: [(output_field, section, object_fields)]} from the schema."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    out: Dict[str, List[Tuple[str, str, Dict]]] = {g: [] for g in GROUPS}
    for section in ("mandible", "maxilla", "dental_elements"):
        for fact in schema.get(section) or []:
            groups = {GROUP_OF_IMAGE.get(i)
                      for i in (fact.get("images_needed") or [])} - {None}
            if len(groups) != 1:
                continue
            out[groups.pop()].append((fact.get("output_field") or "?", section,
                                      fact.get("object_fields") or {}))
    return out


# The identity of a list item, tried in order. `fdi_number` is the name the
# schema actually gives an implant's position -- it was missing here until
# 2026-08-16, so every implant fell through to the whole-dict comparison this
# function exists to avoid, and the same implant failed to match itself as soon
# as `location` was worded differently or `osseointegration_status` was null.
# A bridge has no single position, so it is keyed on its ordered span.
_ITEM_KEYS = ("fdi", "fdi_number", "tooth", "position")


def _item_key(item: Dict):
    """The comparable identity of one list item, or None if it has none."""
    for name in _ITEM_KEYS:
        value = item.get(name)
        if value is not None:
            return value
    start, end = item.get("span_start"), item.get("span_end")
    if isinstance(start, int) and isinstance(end, int):
        # Ordered, so 17-14 and 14-17 are the same bridge. This is an EXACT
        # span match: a bridge the model finds but measures 15-17 against a
        # reference 14-17 still counts as missed. That is deliberate -- an
        # overlap rule would be a different question, not a stricter one.
        return (min(start, end), max(start, end))
    return None


def _hashable(value) -> Set:
    """A list field as a comparable set.

    Some list fields are bare FDIs (`adjacent_teeth`, `primary_teeth`) and some
    are objects (`implants`, `bridges` carry a dict per item). An object is
    reduced to its position where it has one, so an implant the model got right
    but described differently still matches; failing that, to its sorted scalar
    items. Comparing whole dicts would make every list row score zero on a
    wording difference.
    """
    out = set()
    for item in as_list(value):
        if isinstance(item, dict):
            key = _item_key(item)
            out.add(key if key is not None
                    else tuple(sorted((k, str(v)) for k, v in item.items())))
        elif isinstance(item, (int, str)):
            out.add(item)
    return out


def _score_field(stat: Counter, kind: str, claimed, truth) -> None:
    """Accumulate one (claimed, truth) pair under the metric its kind implies."""
    if kind == "bool":
        c, t = str(claimed).lower() == "true", str(truth).lower() == "true"
        if c and t:
            stat["tp"] += 1
        elif c:
            stat["fp"] += 1
        elif t:
            stat["fn"] += 1
    elif kind == "list":
        stat += score_sets(_hashable(claimed), _hashable(truth))
    elif kind == "map":
        c, t = _as_dict(claimed), _as_dict(truth)
        for key, truth_value in t.items():
            got = c.get(key, c.get(str(key)))
            if got is None:
                continue
            stat["scored"] += 1
            stat["right"] += int(str(got).strip().lower()
                                 == str(truth_value).strip().lower())
    else:
        stat["scored"] += 1
        stat["right"] += int(str(claimed).strip().lower()
                             == str(truth).strip().lower())


def survey_groups(cases: List[Dict],
                  facts_by_group: Dict[str, List[Tuple[str, str, Dict]]]
                  ) -> Dict[str, Dict[Tuple[str, str], Tuple[str, Counter]]]:
    """Per-FIELD prediction accuracy, inside each call group.

    The point of splitting by group is to say WHICH read is failing, so the
    unit here is the individual schema field, not a rolled-up per-group total.
    Aggregation is over cases -- and for the templated per-tooth facts, over
    every tooth as well, since tooth_11_morphology and tooth_47_morphology are
    the same question asked twice.

    Only positions BOTH sides answered are scored. A field the prediction left
    null is not a wrong answer, and the GT files carry nulls of their own
    wherever the report said nothing -- counting either as an error would score
    the reports' silence.
    """
    out: Dict[str, Dict[Tuple[str, str], Tuple[str, Counter]]] = {
        g: {} for g in GROUPS}
    for group, facts in facts_by_group.items():
        for output_field, section, object_fields in facts:
            for field, spec in object_fields.items():
                kind = field_kind(spec if isinstance(spec, str) else "")
                if kind == "skip":
                    continue
                stat = Counter()
                for case in cases:
                    for claimed, truth in _fact_pairs(case, output_field,
                                                      section):
                        c, t = claimed.get(field), truth.get(field)
                        if c is None or t is None:
                            continue
                        _score_field(stat, kind, c, t)
                if stat:
                    out[group][(output_field, field)] = (kind, stat)
    return out


def _fact_pairs(case: Dict, output_field: str, section: str):
    """(prediction block, GT block) pairs for one fact in one case.

    One pair for a global fact; one per present tooth for a templated
    dental_elements fact, which is why a per-tooth row aggregates over teeth.
    """
    pred, gt = case["pred"], case["gt"]
    if section != "dental_elements":
        p = _as_dict(_as_dict(pred.get("global")).get(output_field))
        g = _as_dict(_as_dict(gt.get("global")).get(output_field))
        if p and g:
            yield p, g
        return
    pred_teeth, gt_teeth = _as_dict(pred.get("teeth")), _as_dict(gt.get("teeth"))
    for tooth_key, gt_block in gt_teeth.items():
        fdi = tooth_key.split("_")[-1]
        name = output_field.replace("{fdi}", fdi)
        p = _as_dict(_as_dict(pred_teeth.get(tooth_key)).get(name))
        g = _as_dict(_as_dict(gt_block).get(name))
        if p and g:
            yield p, g


def survey_axes(cases: List[Dict]) -> List[Tuple]:
    """The overall table: one row per key fact, PRED and SUMMARY vs GT."""
    rows: List[Tuple] = []
    per_tooth = {c: {"pred": Counter(), "summary": Counter(), "n": 0}
                 for c in CATEGORIES}
    absent = {"pred": Counter(), "summary": Counter(), "n": 0}
    implants = {"pred": Counter(), "summary": Counter(), "n": 0}
    bridges = {"pred": Counter(), "summary": Counter(), "n": 0}
    side = {label: {"pred": Counter(), "summary": Counter(), "n": 0}
            for label, *_ in SIDE_AXES + SINGLE_AXES}
    glob = {label: {"pred": Counter(), "summary": Counter(), "n": 0}
            for label, *_ in GLOBAL_AXES}

    for case in cases:
        gt, pred, summ = case["gt"], case["pred"], case["summary"]

        gt_claims, p_claims = pred_claims(gt), pred_claims(pred)
        s_claims = summary_claims(summ) if summ else {}
        for category in CATEGORIES:
            truth = set(gt_claims.get(category, {}))
            per_tooth[category]["n"] += len(truth)
            per_tooth[category]["pred"] += score_sets(
                set(p_claims.get(category, {})), truth)
            per_tooth[category]["summary"] += score_sets(
                set(s_claims.get(category, {})), truth)

        gt_a, p_a = pred_absent(gt)["absent"], pred_absent(pred)["absent"]
        s_a = summary_absent(summ)["absent"] if summ else set()
        absent["n"] += len(gt_a)
        absent["pred"] += score_sets(p_a, gt_a)
        absent["summary"] += score_sets(s_a, gt_a)

        gt_i, gt_b = pred_prosthetics(gt)
        p_i, p_b = pred_prosthetics(pred)
        s_i, s_b = summary_prosthetics(summ) if summ else (set(), False)
        implants["n"] += len(gt_i)
        implants["pred"] += score_sets(p_i, gt_i)
        implants["summary"] += score_sets(s_i, gt_i)
        bridges["n"] += int(gt_b)
        for stage, claim in (("pred", p_b), ("summary", s_b)):
            bridges[stage]["tp" if (claim and gt_b) else
                           "fp" if claim else "fn" if gt_b else "tn"] += 1

        for label, tmpl, field, path, kind, default in GLOBAL_AXES:
            g = global_axis(gt, tmpl, field, kind)
            read = {"pred": global_axis(pred, tmpl, field, kind),
                    "summary": global_axis_summary(summ or {}, path, kind, default)}
            for stage, d in read.items():
                for arch, truth in g.items():
                    if arch not in d:
                        continue
                    if kind == "set":
                        glob[label][stage] += score_sets(d[arch], truth)
                    else:
                        glob[label][stage]["scored"] += 1
                        glob[label][stage]["right"] += int(d[arch] == truth)
            # For a set axis N counts what the label asserts (teeth), not how
            # many arches were answered -- otherwise an axis nobody has any of
            # reports N=80 and a rate of 0.00, which reads as a failure rather
            # than as an empty row.
            glob[label]["n"] += (sum(len(v) for v in g.values()) if kind == "set"
                                 else len(g))

    for case in cases:
        gt, pred, summ = case["gt"], case["pred"], case["summary"]
        for label, tmpl, field, path, kind in SIDE_AXES + SINGLE_AXES:
            units = SIDES if "{side}" in tmpl else ("",)
            for s_ in units:
                fact = tmpl.format(side=s_)
                truth = _as_dict(_as_dict(gt.get("global")).get(fact)).get(field)
                if truth is None:
                    continue
                side[label]["n"] += (len(_cast(truth, kind)) if kind == "set"
                                     else 1)
                claims = {
                    "pred": _as_dict(_as_dict(pred.get("global")).get(fact)).get(field),
                    "summary": _dig(summ or {}, path, s_),
                }
                for stage, got in claims.items():
                    if got is None:
                        continue
                    if kind == "set":
                        side[label][stage] += score_sets(_cast(got, kind),
                                                         _cast(truth, kind))
                    else:
                        side[label][stage]["scored"] += 1
                        side[label][stage]["right"] += int(
                            _cast(got, kind) == _cast(truth, kind))

    def add(section, label, unit, stat, metric="prec/rec"):
        if not stat["n"]:
            # No label to score against: an explicit blank, because 0.00/0.00
            # reads as "got everything wrong" rather than "nothing to get".
            rows.append((section, label, 0, unit, metric, "  -  ", "  -  "))
            return
        if metric == "acc":
            def fmt(s):
                return (f"{s['right'] / s['scored']:.2f}" if s.get("scored")
                        else "  -  ")
        else:
            def fmt(s):
                return pr(s["tp"], s["fp"], s["fn"])
        rows.append((section, label, stat["n"], unit,
                     metric, fmt(stat["pred"]), fmt(stat["summary"])))

    add("teeth", "absent teeth", "teeth", absent)
    for category in CATEGORIES:
        add("findings", category, "teeth", per_tooth[category])
    add("prosthetics", "implants", "slots", implants)
    add("prosthetics", "fixed bridges", "cases", bridges)
    for label, _t, _f, _p, kind, _d in GLOBAL_AXES:
        add("global", label, "teeth" if kind == "set" else "arches",
            glob[label], "prec/rec" if kind == "set" else "acc")
    for label, _t, _f, _p, kind in SIDE_AXES:
        add("anatomy", label, "teeth" if kind == "set" else "sides",
            side[label], "prec/rec" if kind == "set" else "acc")
    for label, _t, _f, _p, kind in SINGLE_AXES:
        add("anatomy", label, "cases", side[label], "acc")
    return rows


def render(cases: List[Dict], rows: List[Tuple], groups: Dict[str, Counter],
           run_dir: Path, fact_groups: Dict[str, List[str]]) -> str:
    out: List[str] = []
    out.append(f"Fact-level survey: {run_dir}")
    out.append("")
    out.append(f"{len(cases)} cases, ground truth = first *_gt.json per case "
               f"(generated from the reference reports)")
    out.append("")

    out.append("CALL GROUPS -- raw prediction accuracy per field, by the image read")
    out.append("")
    for group in GROUPS:
        fields = groups.get(group) or {}
        names = fact_groups.get(group, [])
        out.append(f"[{group}] {len(names)} schema fact(s)")
        if not fields:
            out.append("  (nothing scoreable -- no field both sides answered)")
            out.append("")
            continue
        head = (f"  {'fact':38} {'field':22} {'N':>5} {'metric':8} score")
        out += [head, "  " + "-" * (len(head) - 2)]
        last = None
        for (fact, field), (kind, stat) in fields.items():
            shown = "" if fact == last else fact
            last = fact
            if kind in ("enum", "map"):
                n, score, metric = stat["scored"], (
                    f"{stat['right'] / stat['scored']:.2f}"
                    if stat["scored"] else "  -  "), "acc"
            else:
                n = stat["tp"] + stat["fn"]
                score, metric = pr(stat["tp"], stat["fp"], stat["fn"]), "prec/rec"
            if not n:
                score = "  -  "
            out.append(f"  {shown:38} {field:22} {n:5d} {metric:8} {score}")
        out.append("")
    out.append("N       = positions BOTH sides answered (enum) or that the GT")
    out.append("          asserts (bool/list). A field the prediction left null")
    out.append("          is not scored -- neither is one the report never stated.")
    out.append("bool/list rows are prec/rec with true as the positive class;")
    out.append("enum rows are right/scored. Per-tooth facts aggregate over teeth.")
    out.append("")
    out.append("=" * 78)
    out.append("OVERALL -- every key fact, raw prediction vs cross-source vote")
    head = (f"{'section':12} {'axis':22} {'N':>5} {'unit':6} {'metric':9} "
            f"{'PRED':11} {'SUMMARY':11}")
    out += [head, "-" * len(head)]
    for section, label, n, unit, metric, p, s in rows:
        out.append(f"{section:12} {label:22} {n:5d} {unit:6} {metric:9} "
                   f"{p:11} {s:11}")
    out.append("")
    out.append("N       = what the ground truth asserts.")
    out.append("PRED    = raw VQA reads, unioned over every source that can assert it.")
    out.append("SUMMARY = what survived postprocess_pred.py's cross-source vote,")
    out.append("          i.e. what can still reach the synthesized report.")
    out.append("acc     = right/scored over the arches both sides answered.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="dataset/{split}/outputs/ground_truth")
    ap.add_argument("--schema", type=Path, default=Path("schema/schema.json"))
    ap.add_argument("--case-ids", nargs="*")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cases = load_cases(args.run_dir, args.gt_dir, args.case_ids)
    if not cases:
        sys.exit("[FAIL] no cases with both a prediction and a *_gt.json")

    fact_groups = schema_groups(args.schema)
    rows = survey_axes(cases)
    groups = survey_groups(cases, schema_facts(args.schema))
    text = render(cases, rows, groups, args.run_dir, fact_groups)
    print(text)

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "run_dir": str(args.run_dir), "cases": len(cases),
            "call_groups": {g: {f"{fact}.{field}": dict(stat)
                                for (fact, field), (_k, stat)
                                in (groups.get(g) or {}).items()}
                             for g in GROUPS},
            "axes": [{"section": s, "axis": a, "n": n, "unit": u,
                      "metric": m, "pred": p, "summary": su}
                     for s, a, n, u, m, p, su in rows],
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
