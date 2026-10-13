#!/usr/bin/env python3
"""
code/eval/structured_findings_helper.py -- reading a run into claims. Shared by both surveys.

WHAT THIS IS, AND WHY IT IS NOT A SURVEY
────────────────────────────────────────
Everything here answers one question: given a run directory, WHAT DID THE
PIPELINE CLAIM? `pred_*` reads the raw predictions, `summary_*` reads the
post-vote summaries, `facts_absent` reads the segmentation. None of it knows
what the right answer is -- no ground truth, no scoring, no output format.

That line is the reason this file exists. Two surveys score those claims
against two different authorities:

    survey_findings.py   the hand-read REPORT_GT tables, plus the mask for
                         absence -- the only source that enumerates all 32
                         positions
    structured_findings_evaluation.py      the generated {case}_gt.json, in prediction shape

They are not interchangeable and were never going to merge: a number scored
against a hand-read table and a number scored against a generated one answer
different questions, and every survey_<ts>.txt already written is mask-scored
on the absent row. What they DO share is this -- the awkward part.

AND IT IS THE AWKWARD PART. These extractors encode the pre-v6.4 fallbacks, the
ARCH_VALUE folding that lets ~20 behaviours keep reading `filling`/`caries`
after schema v7.1 renamed them, the composite's restoration priority, and the
rule that an arch whose pattern is "unknown" is sixteen UNREAD positions rather
than sixteen present teeth. Re-implementing any of it in a second place would
drift the moment the schema moves -- which is exactly why structured_findings_evaluation.py
imported it from survey_findings.py rather than copying it, back when this was
a 2,540-line survey script that seven modules imported to get `_as_dict`.

Extracted 2026-08-19. The move was verified by re-running both surveys and
diffing: byte-identical output, before and after.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

CATEGORIES = ("impaction", "endodontic", "post_and_core", "crown", "fillings",
              "restoration", "caries", "root_remnant")
# The four enum values of _definitions.impaction_direction that assert
# impaction. "none" denies it; anything else the model has emitted here
# ("absent", "not_included", "not_erupted", "not_impacted") is an answer to
# a different question and is not a claim either way.
# The direction an impaction is tipped toward. Schema v6.4 replaced the old
# single enum (none|mesial|distal|horizontal|vertical, where "none" meant not
# impacted) with a bool + a separate `orientation`, and swapped
# horizontal/vertical for the buccolingual pair -- so both vocabularies are
# accepted here, since a survey run is often pointed at an older run's
# predictions.
IMPACTION_VALUES = {"mesial", "distal", "horizontal", "vertical",
                    "lingual", "buccal", "unspecified"}
# The 32 permanent positions, per arch, in FDI order. Fixed by the schema's
# dental_arch_findings_{arch}.findings, which answers exactly these sixteen
# keys per arch -- so this is also the denominator of the absent-teeth read.
ARCH_TEETH = {
    "mandible": tuple(range(31, 39)) + tuple(range(41, 49)),
    "maxilla": tuple(range(11, 19)) + tuple(range(21, 29)),
}
ALL_TEETH = frozenset(f for positions in ARCH_TEETH.values() for f in positions)
# 3D-render fact name -> the FDI it reads. Fixed by schema.json.
WISDOM_SLOT = {
    "lower_right_wisdom_tooth": 48,
    "upper_right_wisdom_tooth": 18,
    "lower_left_wisdom_tooth": 38,
    "upper_left_wisdom_tooth": 28,
}
# dental_arch_findings_{arch}.findings value -> category it votes for.
#
# Both vocabularies, because a survey run is often pointed at an older run's
# predictions. v7.1 (top) collapsed the enum to five values; the v6.x words
# below it can no longer be emitted but are still on disk in every stored
# prediction.
#
# "defect" maps to nothing on purpose, exactly as its v6.x half "caries" did:
# neither caries nor a root remnant is one of the five findings surveyed here.
# And note what v7.1 costs the post_and_core row -- a posted tooth now reads
# root_canal_treatment on the panoramic, so that row loses its panoramic
# source entirely and becomes composite-only, which is the truthful account of
# what the image can support.
ARCH_VALUE = {
    "impacted": "impaction",
    "root_canal_treatment": "endodontic",
    "restoration": "restoration",
    # v7.1 spells caries `defect`, because at arch resolution a carious lesion
    # and a retained root are one dark irregular crown -- so this value votes
    # on caries and CANNOT vote on root_remnant, which is composite-only.
    "defect": "caries",
    # -- v6.x, read-only --
    "post_and_core": "post_and_core",
    "filling": "fillings",
    "caries": "caries",
}
# restoration_summary.groups key / tooth_{fdi}_restoration.restoration_type
# value -> category. The schema spells the filling value plural.
RESTORATION_VALUE = {"crown": "crown", "post_and_core": "post_and_core",
                     "fillings": "fillings"}
# ---------------------------------------------------------------------------
# Report sentence scan -- over-inclusive by design; it exists to prove no
# statement was missed, and is filtered by eye, not by regex.
# ---------------------------------------------------------------------------
KEYWORD_RE = {
    "impaction": re.compile(
        r"impact|inclus|included|retain|unerupt|erupt|disodont|dysodont"
        r"|verted|ectopic|semi", re.I),
    "endodontic": re.compile(r"endodont|root canal|endocanal|obturat", re.I),
    "post_and_core": re.compile(r"post[- ]and[- ]core|endocanal post|\bpost\b"
                                r"|stump|abutment", re.I),
    "crown": re.compile(r"crown|capsul|prosthe|bridge|splint|rehabilitat", re.I),
    "fillings": re.compile(r"filling|restorat|conservative|amalgam|composite"
                           r"|reconstruct", re.I),
}
# "included in the scan volume" is scope, not impaction; "post-extraction
# socket" is not a post; "the crown of tooth X" is anatomy, not a prosthesis.
# Dropped from the strong scan unless the sentence says something only the
# finding itself says.
NOISE_RE = {
    # Three senses of "retain-" appear in these reports and NONE of them is a
    # tooth held in the bone. Measured over all 622 reports: "retainer" x22 in
    # 15 cases is the orthodontic appliance ("lingual splint/retainer from 33
    # to 43", "bar-retained prosthesis"); "retention" x9 in 8 cases is a
    # maxillary sinus retention CYST; "retained" x8 is a root fragment in 3 of
    # its 4 cases. The one genuine impaction sense -- F011's "73 retained with
    # 33, which is impacted" -- says "impacted" in the same sentence and is
    # matched on that word instead, so nothing is lost by demoting these.
    #
    # This cost two fabricated GT entries before it was caught: S0014 and
    # S0022 both drafted impaction [31,32,33,41,42,43] out of "orthodontic
    # splint/retainer extending from tooth 33 to tooth 43", and an invented
    # impaction is worse than a missing one -- it scores correct model silence
    # as a false negative. The crown table above already learned the same
    # lesson about bare "splint"; impaction had not.
    "impaction": re.compile(r"(included|inclusion)\s+(in|within|only|of|at)?\s*"
                            r"(the|its|their)?\s*(scan|acquisition|available|most"
                            r"|volume|portion|caudal|field)"
                            r"|retainer|retention\s+cyst|retained\s+root"
                            r"|bar[- ]retained", re.I),
    "post_and_core": re.compile(r"post[- ]extraction|posterior|post[- ]operative", re.I),
}
# "(?<!in)verted" keeps mesioverted/distoverted/normoverted but not the
# "Inverted Panorex" / "right-left inverted" disclaimers, which are about
# LR_INVERTED_REPORTS, not about a tooth.
STRONG_RE = {
    # "retain" is NOT strong: the noise clause above can only fire on a
    # sentence that is not strong, and every "retain-" sense in this corpus is
    # noise. A real "retained tooth" sentence still matches KEYWORD_RE and is
    # printed as weak/filtered with its FDIs, so it is offered to the reader
    # rather than dropped.
    "impaction": re.compile(r"impact|(?<!in)verted|unerupt|ectopic|semi"
                            r"|disodont|dysodont", re.I),
    "endodontic": re.compile(r"endodont|root canal|endocanal", re.I),
    "post_and_core": re.compile(r"post[- ]and[- ]core|endocanal post|stump"
                                r"|\bpost\b(?![- ]extraction)", re.I),
    # "crown" alone is anatomy far more often than prosthetics -- "the crown
    # of tooth 3.6", "includes the crowns of the upper dental elements",
    # "caries of the buccal crown". A prosthetic crown is always said with
    # one of these.
    # Bare "splint" is usually a lingual orthodontic retainer (A092, P482,
    # S0017, S0048) -- prosthetic only when it is the crowns being splinted.
    "crown": re.compile(r"prosthe|capsul|bridge|rehabilitat|cantilever"
                        r"|splinted\s+crowns|crowns?\s+splinted"
                        r"|crowns?\s+(on|at|are|is|from|supported)", re.I),
    "fillings": re.compile(r"filling|restorat|conservative|amalgam", re.I),
}
def _as_dict(value) -> Dict:
    return value if isinstance(value, dict) else {}
def as_list(value) -> List:
    """Coerce a maybe-scalar, maybe-missing list field into a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]
def split_sentences(text: str) -> List[str]:
    """Split on sentence punctuation without cutting FDI numbers.

    Reports write teeth as both "48" and "4.8", and refer to them as
    "e.d. 34" (elemento dentale). A naive split on ". " turns
    "tooth 4.8 impacted" into "8 impacted" and strands "e.d." from its
    tooth list -- in both cases the tooth number is lost.
    """
    guarded = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    guarded = re.sub(r"\b([a-zA-Z])\.(?=[a-zA-Z]\.|\s*\d)", r"\1<DOT>", guarded)
    return [s.replace("<DOT>", ".").strip()
            for s in re.split(r"(?<=[.;:])\s+", guarded) if s.strip()]
def report_paths(reports_dir: Path, case_id: str) -> List[Path]:
    return sorted(reports_dir.glob(f"{case_id}_*.txt")) or \
        sorted(reports_dir.glob(f"{case_id}.txt"))
def finding_sentences(reports_dir: Path, case_id: str,
                      category: str) -> List[Tuple[str, str, bool]]:
    """[(report name, sentence, is_strong)] for one case and category."""
    hits = []
    noise = NOISE_RE.get(category)
    strong_re = STRONG_RE[category]
    for path in report_paths(reports_dir, case_id):
        for sentence in split_sentences(path.read_text(encoding="utf-8")):
            if not KEYWORD_RE[category].search(sentence):
                continue
            strong = bool(strong_re.search(sentence))
            if noise and noise.search(sentence) and not strong:
                continue
            hits.append((path.name, sentence, strong))
    return hits
def pred_claims(pred: Dict) -> Dict[str, Dict[int, Dict[str, str]]]:
    """{category: {fdi: {source: value}}} for every claim in one prediction."""
    claims: Dict[str, Dict[int, Dict[str, str]]] = {c: {} for c in CATEGORIES}

    def record(category: str, fdi: int, value: str, source: str) -> None:
        claims[category].setdefault(fdi, {})[source] = value

    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        tooth = _as_dict(tooth)

        # No composite impaction claim: tooth_{fdi}_impaction left the schema
        # in v6.3 and postprocess_pred stopped reading it, so surveying it
        # would score a source the pipeline does not use. A legacy prediction
        # that still carries the fact is ignored here for the same reason.

        # Schema v6.4 folded the composite's restorative and endodontic claims
        # into tooth_{fdi}_morphology as bools. Read them the way
        # postprocess_pred.composite_restoration_type does -- same priority,
        # so the survey scores what the pipeline actually consumes -- and fall
        # back to the pre-v6.4 shapes when surveying an older run.
        morphology = _as_dict(tooth.get(f"tooth_{fdi}_morphology"))
        endo = _as_dict(tooth.get(f"tooth_{fdi}_endodontic_treatment"))

        rct = morphology.get("with_endo")
        if rct is None:
            rct = endo.get("root_canal_treatment")     # pre-v6.4 predictions
        # The field is a bool, but the VLM has emitted the strings "true"/"false".
        if str(rct).lower() == "true":
            record("endodontic", fdi,
                   str(endo.get("filling_quality") or "unspecified"), "composite")

        restoration = None
        for flag, value in (("with_post_and_core", "post_and_core"),
                            ("with_full_crown", "crown"),
                            ("with_fillings", "fillings")):
            if str(morphology.get(flag)).lower() == "true":
                restoration = value
                break
        if restoration is None:                        # pre-v6.4 predictions
            restoration = _as_dict(tooth.get(f"tooth_{fdi}_restoration")).get("restoration_type")
        if restoration in RESTORATION_VALUE:
            record(RESTORATION_VALUE[restoration], fdi, restoration, "composite")
        # The composite's second vote, into the panoramic's coarser row: a
        # crown or a filling IS a "restoration" as the arch read means it. A
        # post is not -- v7.1 files a posted tooth under root_canal_treatment,
        # so counting it here would make the two sources disagree about a
        # tooth they actually agree on. The composite picks ONE value by
        # priority, so a post-and-core UNDER a crown votes post_and_core only
        # and is silent here; that lossiness is the composite's, and predates
        # this row.
        if restoration in ("crown", "fillings"):
            record("restoration", fdi, restoration, "composite")

        # Two more composite facts, independent of the restorative one above --
        # a tooth can be carious AND filled, and a remnant carries findings of
        # its own. Neither is part of the restoration priority chain.
        if str(morphology.get("with_caries")).lower() == "true":
            record("caries", fdi, "unspecified", "composite")
        if str(morphology.get("is_remnant")).lower() == "true":
            record("root_remnant", fdi, "unspecified", "composite")

    facts = _as_dict(pred.get("global"))
    for slot, fdi in WISDOM_SLOT.items():
        fact = _as_dict(facts.get(slot))
        impacted = fact.get("impacted")
        if impacted is True or str(impacted).lower() == "true":
            # v6.4: the bool says WHETHER, `orientation` says which way. A
            # tooth impacted with no orientation given still counts as a claim.
            direction = fact.get("orientation")
            record("impaction", fdi,
                   direction if direction in IMPACTION_VALUES else "unspecified",
                   "3d_render")
        elif impacted in IMPACTION_VALUES:
            # pre-v6.4: one enum carried both, with "none" meaning not impacted
            record("impaction", fdi, impacted, "3d_render")

    for arch in ("mandible", "maxilla"):
        findings = _as_dict(_as_dict(facts.get(f"dental_arch_findings_{arch}")).get("findings"))
        for fdi, value in findings.items():
            # This fact answers one enum per tooth and has no direction or
            # quality field -- it only names the finding.
            if value in ARCH_VALUE:
                record(ARCH_VALUE[value], int(fdi), "unspecified", "panoramic")
    return claims
def composite_coverage(pred: Dict) -> Dict[str, Set[int]]:
    """{category: teeth whose composite fact was actually answered}."""
    # No entry for impaction: tooth_{fdi}_impaction was dropped from the
    # schema (v6.3), so no composite fact backs that category any more and
    # "was this tooth even read" is not a question about it.
    #
    # Each category lists the facts that would carry it, NEWEST FIRST, and any
    # one of them being present means the tooth was read. v6.4 folded the
    # restorative claims out of tooth_{fdi}_restoration and into
    # tooth_{fdi}_morphology, so naming only the old fact reported every v6.4+
    # tooth as uncropped -- which is the opposite of what this function is for,
    # and made "got no composite crop at all" count teeth that had one. Same
    # fallback pattern as pred_claims, for the same reason.
    fact_for = {"endodontic": ("morphology", "endodontic_treatment"),
                "post_and_core": ("morphology", "restoration"),
                "crown": ("morphology", "restoration"),
                "fillings": ("morphology", "restoration"),
                "restoration": ("morphology", "restoration"),
                "caries": ("morphology",),
                "root_remnant": ("morphology",)}
    covered: Dict[str, Set[int]] = {c: set() for c in CATEGORIES}
    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        for category, facts in fact_for.items():
            if any(isinstance(_as_dict(tooth).get(f"tooth_{fdi}_{f}"), dict)
                   for f in facts):
                covered[category].add(fdi)
    return covered
def summary_claims(summary: Dict) -> Dict[str, Dict[int, str]]:
    """{category: {fdi: value}} from the post-vote summary blocks."""
    kept: Dict[str, Dict[int, str]] = {c: {} for c in CATEGORIES}
    for arch in ("mandible", "maxilla"):
        block = _as_dict(summary.get(arch))

        for entry in block.get("impacted_teeth") or []:
            kept["impaction"][entry["tooth"]] = entry.get("impacted", "unspecified")

        endo = _as_dict(block.get("endodontic_summary"))
        quality = {fdi: q for q, teeth in
                   _as_dict(endo.get("quality_groups")).items() for fdi in teeth}
        for fdi in endo.get("teeth") or []:
            kept["endodontic"][fdi] = quality.get(fdi, "unspecified")

        groups = _as_dict(_as_dict(block.get("restoration_summary")).get("groups"))
        for group, teeth in groups.items():
            if group in RESTORATION_VALUE:
                for fdi in teeth:
                    kept[RESTORATION_VALUE[group]][fdi] = group
            # The coarse row, same rule as the PRED side in pred_claims: a
            # kept crown or filling is a kept "restoration". Without this the
            # row would read SUM 0 on every run and look like a vote that
            # threw everything away, when in fact the summary has no
            # restoration group of its own to read.
            if group in ("crown", "fillings"):
                for fdi in teeth:
                    kept["restoration"][fdi] = group

        # Caries rides in the per-tooth morphology block and is written only
        # when true; root remnants have a list of their own, which is also
        # where their resorption clause lives.
        for entry in block.get("tooth_findings") or []:
            fdi = entry.get("fdi")
            if not isinstance(fdi, int):
                continue
            if str(_as_dict(entry.get("morphology")).get("caries")).lower() == "true":
                kept["caries"][fdi] = "unspecified"
        for entry in block.get("root_remnants") or []:
            fdi = _as_dict(entry).get("fdi")
            if isinstance(fdi, int):
                kept["root_remnant"][fdi] = "unspecified"
    return kept
def _version_key(name: str) -> List[int]:
    """Sort key for a 'predictions_v6.10'-style directory name.

    Numeric, component by component, so v6.10 sorts ABOVE v6.9 rather than
    below it the way a plain string compare would put it.
    """
    return [int(n) for n in re.findall(r"\d+", name)] or [-1]
def _resolve_stage_dir(parent: Path, prefix: str, pattern: str) -> Path:
    """Find the directory actually holding a stage's files.

    aksssr_pipeline.sh nests each stage under a schema-version tag --
    predictions/predictions_v6.9/ rather than predictions/ -- so that a v6.9
    run cannot land on top of a v6.4 one. Older arms, which are not being
    re-run, still use the flat layout. Accept both rather than pick one and
    fail on the other: flat first, then the newest-versioned subdirectory that
    is actually non-empty (an empty one is skipped, not chosen and then
    reported as having no cases). Falls back to `parent`, so the caller's own
    error message names the directory the user expected.
    """
    if parent.is_dir() and any(parent.glob(pattern)):
        return parent
    subs = sorted((d for d in parent.glob(f"{prefix}_*") if d.is_dir()),
                  key=lambda d: _version_key(d.name), reverse=True)
    for d in subs:
        if any(d.glob(pattern)):
            return d
    return parent
def facts_absent(facts: Dict) -> Tuple[Set[int], Set[int]]:
    """(absent positions, positions this facts file enumerates at all)."""
    structured = _as_dict(facts.get("structured"))
    present = {f for f in as_list(structured.get("teeth_present"))
               if isinstance(f, int)}
    absent = {f for f in as_list(structured.get("teeth_absent"))
              if isinstance(f, int)}
    # Deciduous FDIs (5x-8x) appear in these lists for the mixed-dentition
    # cases; the arch fact only answers the permanent positions.
    return absent & ALL_TEETH, (present | absent) & ALL_TEETH
def pred_absent(pred: Dict) -> Dict:
    """Absent-tooth claims in one raw prediction, per source.

    Two sources can assert absence, and neither is the composite: the
    panoramic arch read answers all sixteen positions per arch with `absent`
    as an explicit value, and the four wisdom-tooth 3D renders answer
    eruption_state `absent`. The composite per-tooth facts cannot say it at
    all -- build_vqa_pairs only expands them for teeth the segmentation
    already places, so their silence at a position is the pipeline's own
    doing rather than a read, and counting it as a claim would score the
    mask against itself.
    """
    facts = _as_dict(pred.get("global"))
    by_source: Dict[str, Set[int]] = {"panoramic": set(), "3d_render": set()}
    read: Set[int] = set()
    for arch, positions in ARCH_TEETH.items():
        findings = _as_dict(_as_dict(facts.get(f"dental_arch_findings_{arch}"))
                            .get("findings"))
        for fdi in positions:
            value = findings.get(str(fdi), findings.get(fdi))
            if not isinstance(value, str):
                continue                   # position never answered
            read.add(fdi)
            if value.strip().lower() == "absent":
                by_source["panoramic"].add(fdi)
    for fact_name, fdi in WISDOM_SLOT.items():
        state = _as_dict(facts.get(fact_name)).get("eruption_state")
        if isinstance(state, str):
            read.add(fdi)
            if state.strip().lower() == "absent":
                by_source["3d_render"].add(fdi)
    return {"absent": by_source["panoramic"] | by_source["3d_render"],
            "by_source": by_source, "read": read}
def summary_absent(summary: Dict) -> Dict:
    """What postprocess's absent_teeth block asserts, per arch.

    Derived from `present` wherever it exists rather than from `teeth`: the
    block omits `teeth` when the pattern is "none", so reading `teeth` alone
    would turn a full arch into an unanswered one. An arch whose pattern
    could not be settled ("unknown") carries neither key -- that is sixteen
    UNREAD positions, not sixteen teeth called present.
    """
    absent, read = set(), set()
    for arch, positions in ARCH_TEETH.items():
        block = _as_dict(_as_dict(_as_dict(summary.get(arch))
                                  .get("arch_findings")).get("absent_teeth"))
        listed = {f for f in as_list(block.get("teeth")) if isinstance(f, int)}
        if "present" in block:
            present = {f for f in as_list(block.get("present"))
                       if isinstance(f, int)}
            read |= set(positions)
            absent |= (set(positions) - present) | listed
        elif listed:
            read |= set(positions)
            absent |= listed
    return {"absent": absent & ALL_TEETH, "read": read}
def pred_prosthetics(pred: Dict) -> Tuple[Set[int], bool]:
    facts = _as_dict(pred.get("global"))
    implants, bridge = set(), False
    for arch in ("mandible", "maxilla"):
        for entry in as_list(_as_dict(facts.get(f"implants_{arch}")).get("implants")):
            fdi = _as_dict(entry).get("fdi_number")
            if isinstance(fdi, int):
                implants.add(fdi)
        bridges = _as_dict(facts.get(f"fixed_bridges_{arch}"))
        bridge = bridge or bool(bridges.get("present")) or bool(bridges.get("bridges"))
    return implants, bridge
def summary_prosthetics(summary: Dict) -> Tuple[Set[int], bool]:
    implants, bridge = set(), False
    for arch in ("mandible", "maxilla"):
        block = _as_dict(_as_dict(summary.get(arch)).get("prosthetics"))
        for entry in as_list(block.get("implants")):
            fdi = _as_dict(entry).get("fdi_number")
            if isinstance(fdi, int):
                implants.add(fdi)
        bridge = bridge or bool(as_list(block.get("bridges")))
    return implants, bridge
def pred_anatomy(pred: Dict) -> Dict:
    facts = _as_dict(pred.get("global"))
    out = {"canal": {}, "ian_close": set(), "sinus": {}, "intrasinusal": set()}
    for side in ("right", "left"):
        canal = _as_dict(facts.get(f"mandible_canal_{side}"))
        # schema v6.4: `location` (lingual|buccal) replaced `course`
        # (regular|lingual|buccal). Fall back to the old key so runs predicted
        # under the previous schema still score.
        out["canal"][side] = canal.get("location", canal.get("course"))
        out["ian_close"] |= {f for f in as_list(canal.get("adjacent_teeth"))
                             if isinstance(f, int)}
        sinus = _as_dict(facts.get(f"maxilla_sinus_{side}"))
        out["sinus"][side] = sinus.get("mucosa_state")
        out["intrasinusal"] |= {f for f in as_list(sinus.get("intrasinusal_teeth"))
                                if isinstance(f, int)}
    return out
def summary_anatomy(summary: Dict) -> Dict:
    out = {"canal": {}, "ian_close": set(), "sinus": {}, "intrasinusal": set()}
    canals = _as_dict(_as_dict(summary.get("mandible")).get("canals"))
    for side in ("right", "left"):
        entry = _as_dict(canals.get(side))
        out["canal"][side] = entry.get("location", entry.get("course"))
        out["ian_close"] |= {f for f in as_list(entry.get("adjacent_teeth"))
                             if isinstance(f, int)}
    maxilla = _as_dict(summary.get("maxilla"))
    sinuses = _as_dict(maxilla.get("sinuses"))
    for side in ("right", "left"):
        out["sinus"][side] = _as_dict(sinuses.get(side)).get("mucosa_state")
    intrasinusal = _as_dict(maxilla.get("sinus_intrasinusal_teeth"))
    out["intrasinusal"] = {f for f in as_list(intrasinusal.get("teeth"))
                           if isinstance(f, int)}
    return out


def uses_legacy_shape(pred: Dict) -> bool:
    """Was this prediction generated before v6.4 restructured the per-tooth facts?

    v6.4 folded the composite restorative claims into tooth_{fdi}_morphology
    as bools (with_full_crown / with_fillings / with_post_and_core) and
    retired tooth_{fdi}_restoration. pred_claims reads both, so the PRED
    column is unaffected -- but postprocess_pred.py reads ONLY the current
    shape, so rebuilding summaries from a legacy prediction silently yields
    empty restoration groups. That looks exactly like a quality collapse in
    the SUMMARY column and is not one, so it is detected and stated.
    """
    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        tooth = _as_dict(tooth)
        if tooth.get(f"tooth_{fdi}_restoration") is not None:
            return True
        morphology = _as_dict(tooth.get(f"tooth_{fdi}_morphology"))
        if morphology and "with_full_crown" not in morphology:
            return True
    return False
def facts_prosthetics(facts: Dict) -> Tuple[Set[int], bool]:
    """(implant positions, bridge present) from a masks-derived facts file."""
    structured = _as_dict(facts.get("structured"))
    # The list carries a stray None for at least one case (A018).
    implants = {f for f in as_list(structured.get("implants")) if isinstance(f, int)}
    return implants, bool(structured.get("bridge_present"))
