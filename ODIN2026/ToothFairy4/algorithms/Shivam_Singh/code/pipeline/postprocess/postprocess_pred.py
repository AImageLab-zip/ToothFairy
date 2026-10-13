#!/usr/bin/env python3
"""
code/pipeline/postprocess/postprocess_pred.py

Converts a raw {case_id}_pred.json (from run_vqa_inference.py) into a
compact, pre-classified {case_id}_summary.json meant to be handed to a
report-writing LLM or template renderer, per schema/schema.json (v6.1).

WHAT SCHEMA V6.1 CHANGED HERE, AND WHY
────────────────────────────────────────────────────────────────────────
v6.1 deleted eight facts outright and reshaped most of the rest. The facts
that went away were not simply renamed -- each one's content now has to be
DERIVED from what is left:

  - absent_teeth_{arch} is gone. Absence is derived (collect_arch_absent)
    from the per-tooth composite reads plus dental_arch_findings_{arch},
    whose "findings" map is keyed by the PRESENT teeth.
  - impacted_teeth_{arch} is gone, and as of v6.3 so is
    tooth_{fdi}_impaction. Impaction is unioned across the four dedicated
    wisdom-tooth 3D facts and the "impacted" value in
    dental_arch_findings_{arch}; see build_impacted_teeth for what the
    composite read's removal costs (non-wisdom impaction) and buys.
  - root_remnant_{arch} is gone -- tooth_{fdi}_morphology.is_remnant is
    the single remaining source, so root remnants no longer have two
    sources to reconcile.
  - mandible_scope is gone. The mandible's inclusion level is derived from
    the two condyle scopes (build_mandible_scope); the condyle facts
    themselves are inclusion-only now, with no morphology field.
  - nasal_cavity is gone, and as of v6.4 so is its image, so the maxilla
    section has no turbinate block.
  - alveolar_bone_atrophy_{arch} is one presence + one yes/no per arch (v6.4
    replaced the graded extent with the `atrophy` bool), not a per-region
    list; bone_quality_{arch} is one most-notable finding with a type and an
    FDI-anchored location, not two free-text prose fields, and it is now
    gated on the per-tooth tooth_{fdi}_bone_quality read.
  - tooth_{fdi}_restoration is gone as of v6.4. Its restoration_type enum
    became three bools on tooth_{fdi}_morphology (with_fillings /
    with_full_crown / with_post_and_core), and the endodontic fact's
    root_canal_treatment bool became that same block's with_endo. Both are
    read back through composite_restoration_type / composite_morphology, so
    the gates below still vote on one value per tooth.
  - implants_{arch}, fixed_bridges_{arch} and primary_teeth_{arch} all became
    OBJECT facts wrapping the value that used to be the fact itself
    (implants[], bridges[], primary_teeth[]) -- read as before, they silently
    yielded nothing.
  - Every object fact now leads with a free-text "visual_evidence" field.
    It is carried in the prediction for auditing and never rendered.

  - AN EXCLUDED MAXILLA IS EMPTY (2026-08-10). The maxilla's inclusion gates
    the whole arch: not_included nulls every other maxilla fact and drops
    every tooth_11..28 read before any builder runs, while
    partially_included and fully_included keep everything. This supersedes
    the per-tissue split described under build_maxilla_section, where a
    not_included arch still reported its teeth. See drop_excluded_maxilla;
    --keep-excluded-maxilla restores the old behaviour.
    WHERE THAT ANSWER COMES FROM (2026-08-17): the FACTS, not the model --
    facts.structured.fov.maxilla, measured off the mask, overwrites
    maxilla_scope.maxilla_included in both directions before the gate reads
    it. The model's own answer decides only when no facts file was given.
    See facts_maxilla_included.
  - dentition_type is NOT an asked VQA fact. It's derived here from
    primary_teeth_mandible/_maxilla + tooth_{fdi}_eruption, per the
    threshold rule agreed this session (see build_dentition_type).
  - Per EXPLICIT instruction: NO merging across arches (mandible and
    maxilla are reported fully independently, each with its own
    bone_quality closing statement) and NO merging mandible canal
    right/left into one "both normal" sentence -- unlike maxilla sinuses,
    which DO have a real "both normal" merged sentence and keep that
    Type B group logic.

THE GATES ARE OFF -- THIS IS A BRIDGE, NOT A FILTER (2026-08-07)
────────────────────────────────────────────────────────────────
This module used to adjudicate the model's claims: findings asked TWICE --
once of the whole-arch panoramic (dental_arch_findings_{arch}) and once of the
tooth's own composite crop (tooth_{fdi}_morphology's with_* bools) -- were
dropped when the two sources disagreed, and teeth the model called uncertain
were forced back to "normal".

Every one of those gates that was ever surveyed cost recall and left precision
flat (endodontic 0.16 -> 0.17, fillings 0.17 -> 0.17, impaction 0.21 -> 0.19),
so they are ALL OFF now. REQUIRE_CROSS_SOURCE_AGREEMENT is the master switch
and it is False; DEMOTE_UNCERTAIN_TO_NORMAL is False beside it. What reaches
the summary is the UNION of what the sources claimed.

The gates are kept, not deleted, and every vote still RUNS -- the losers are
recorded under arch_findings.cross_source_dropped / .uncertain_demoted but no
longer removed, so the old arm is one flag away (--cross-validate,
--demote-uncertain) and both arms are readable from a single run.

WHAT THIS MODULE STILL DOES, and why it is not a pass-through:
  - DERIVES the facts schema v6.1--v6.4 deleted (absence, impaction, dentition
    type, mandible scope -- see the section above);
  - GROUPS teeth that share a finding, so the renderer emits one sentence for
    a group rather than one per tooth;
  - CLASSIFIES into the Type A / B / C rules below, which decide what is
    stated when normal and what is silent;
  - REPAIRS impossible output -- per-tooth findings on teeth reported absent
    (toothless_fdis), canal adjacency on anterior or opposite-side teeth
    (CANAL_ADJACENT_FDIS), implants at the other arch's FDIs, bridges with no
    parseable span, enforce_scope_consistency. These are NOT precision gates:
    they remove what cannot exist, not what might be wrong, and they stay on.

ABSTAIN remains a distinct verdict from DENY throughout, because the arch read
carries one finding per tooth by priority and has no value at all for crowns,
so its silence is usually structural rather than a denial. That distinction
now only shapes the audit record, not what is reported.

THE CLASSIFICATION THIS SCRIPT STILL IMPLEMENTS
──────────────────────────────────────────────────
  Type A -- silent when normal, no substitute sentence. Most per-tooth
            facts and most single whole-arch facts. Omitted entirely
            when normal/empty.

  Type B -- GROUPED features, now ONLY maxilla sinuses right/left (the
            one pair that still has a real "both normal" merged sentence
            in the decision tree):
              - if BOTH sides normal -> merge into ONE summary flag
              - if EITHER side abnormal -> BOTH sides included in the
                summary, including the normal one (no silent dropping)
            Mandible canals are explicitly NOT Type B anymore -- always
            two independent entries, right and left, no merge.

  Type C -- ALWAYS stated regardless of normal/abnormal. Both arch scope
            statements (the mandible's derived from its condyles, the
            maxilla's from maxilla_scope) and BOTH condyles, in every case,
            never dropped for being normal or for missing a VQA answer.
            Condyles carry INCLUSION ONLY as of v6.1 -- morphology
            classification is out of the schema's scope.

OUTPUT SHAPE
─────────────
{
  "case_id": "...",
  "dentition_type": {"dentition_type": "mixed"|"primary"|"permanent"|"edentulous"},
  "mandible": {
    "scope": {...},                                 # Type C (derived from condyles)
    "condyles": {"right": {"scope": ...},           # Type C, both sides always;
                 "left":  {"scope": ...}},          # inclusion only, no morphology
      "canals": {"right": {...}, "left": {...}},      # always both, no merge;
                                                    # location + adjacency
                                                    # reconciled between the 3D
                                                    # read and the per-molar
                                                    # composite one (v6.9)
    "alveolar_bone_atrophy": {"atrophy": bool,           # one judgment per arch
                              "fully_edentulous": true} | absent,
    "periodontal_bone_resorption": {...} | absent,
    "arch_findings": {
        "absent_teeth": {"pattern": ..., ...},      # derived, see collect_arch_absent
        "primary_teeth": [...] | absent
    },
    "root_remnants": [...] | absent,                 # per-tooth morphology only
    "impacted_teeth": [...] | absent,                # composite + 3D + panoramic union
    "wisdom_teeth": [...] | absent,                  # eruption state, present teeth only
    "endodontic_summary": {...} | absent,            # arch read x per-tooth read
    "restoration_summary": {...} | absent,           # arch read x per-tooth read
    "tooth_findings": [...] | absent,
    "prosthetics": {"implants": [...], "bridges": [...]} | absent,
    "bone_quality": {"present": bool, "findings": [...]}   # closes this arch's section, standalone
  },
  "maxilla": { ...same shape, plus "sinuses" instead of "canals" }
}

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

from normalize_pred import (DEFAULT_SCHEMA, normalize_prediction,  # noqa: E402
                            summarize_repairs)
import source_rules  # noqa: E402  -- the source rules, applied as a post-pass
import rules_config  # noqa: E402  -- THE ARM IS THE CONFIG FILE; see its docstring

# Every module-level flag below is a DEFAULT, not a decision. Each one is
# bound to a key in configs/postprocess/*.yaml through rules_config.BINDINGS,
# and --config overwrites it before any case is read. The value written here
# is the arm-6 setting, so a run with no --config behaves exactly as this
# file did before the config layer existed. rules_config.verify_defaults()
# is the check that the two never drift apart.


# ── Constants ─────────────────────────────────────────────────────────────

def _as_dict(value) -> Dict:
    """Object-typed fact or empty dict -- never a str/list."""
    return value if isinstance(value, dict) else {}


ALL_FDIS = [11,12,13,14,15,16,17,18,
            21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38,
            41,42,43,44,45,46,47,48]

NORMAL_ENUM_VALUES = {"normal", "none", "regular", "not_applicable",
                      "fully_erupted", "complete", "well"}

MANDIBLE_QUADRANTS = [("III", 31, 38), ("IV", 41, 48)]
MAXILLA_QUADRANTS  = [("I", 11, 18), ("II", 21, 28)]

MANDIBLE_WISDOM_FDIS = {38, 48}
MAXILLA_WISDOM_FDIS  = {18, 28}

# Only the premolar/molar roots of the matching quadrant can sit against the
# mandibular canal -- anything else the model reports (anterior teeth, the
# opposite side, a maxillary FDI) is a hallucination, not a finding.
#
# The third molars (38/48) MUST stay in these sets: they are the teeth whose
# roots most often reach the canal, and the ground-truth reports bear that out
# -- across dataset/, canal-adjacency sentences name 38 eighteen times and 48
# sixteen times, against a single mention each of 37 and 47. An earlier version
# of this set stopped at the second molars and so silently deleted almost every
# real adjacency finding. (This is NOT the deferred wisdom-tooth content: the
# deferral covers the wisdom-tooth 3D facts and tooth_{fdi}_eruption, not a
# canal sentence
# that happens to name a third molar.)
CANAL_ADJACENT_FDIS = {
    "right": {44, 45, 46, 47, 48},
    "left":  {34, 35, 36, 37, 38},
}

# Where the SECOND canal source speaks. schema v6.9's
# tooth_{fdi}_mandible_canal is asked of the lower molars only (its
# applies_to_fdi list), read off the composite's coronal row -- so on those
# six teeth the canal has two independent readers, and on the premolars above
# it still has one. Keep this in step with the schema's applies_to_fdi: a tooth
# listed here that the schema does not ask about simply never votes, but a
# tooth the schema asks about and this set omits throws its answer away.
CANAL_TOOTH_FDIS = {
    "right": {46, 47, 48},
    "left":  {36, 37, 38},
}


# ── Generic helpers ───────────────────────────────────────────────────────

def is_meaningful(value) -> bool:
    """
    Not null, not empty string/list/dict, and -- for strings -- not one of
    the schema's "this describes a normal state" enum values. Booleans are
    NOT special-cased here (which value counts as "meaningful" is field-
    specific); callers handle bools explicitly.
    """
    if value is None:
        return False
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        return s.lower() not in NORMAL_ENUM_VALUES
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def fdis_in_range(lo: int, hi: int) -> List[int]:
    """Real FDI codes within [lo, hi] (skips the non-existent 19/20/29/30/39/40 gaps)."""
    return [f for f in ALL_FDIS if lo <= f <= hi]


def _is_consecutive(fdis: List[int]) -> bool:
    if len(fdis) < 2:
        return True
    return all(fdis[i + 1] - fdis[i] == 1 for i in range(len(fdis) - 1))


# ── The composite per-tooth read: schema v6.4's bools -> this module's shapes ─
#
# v6.4 consolidated the per-tooth composite facts. tooth_{fdi}_restoration is
# GONE and tooth_{fdi}_morphology's enums are gone with it; what used to be
# three facts is now one block of independent bools:
#
#   was                                          is now
#   ───────────────────────────────────────────  ──────────────────────────────
#   morphology.root_remnant                      morphology.is_remnant
#   morphology.crown_morphology == "carious"     morphology.with_caries
#   morphology.root_morphology  == "fractured"   morphology.with_root_fracture
#   restoration.restoration_type == "fillings"   morphology.with_fillings
#                                == "crown"      morphology.with_full_crown
#                                == "post_and_core"  morphology.with_post_and_core
#   endodontic.root_canal_treatment              morphology.with_endo
#
# The gates below are built around the OLD shapes -- a single restoration enum
# per tooth, voted against the arch read's single finding per tooth -- and that
# design still holds: the arch read has not changed, and it can still only name
# one thing per tooth. So the bools are collapsed back into that one value here
# rather than threaded through every vote, which keeps the gate logic, its
# measured tuning (CROSS_VALIDATE_*) and its audit trail intact.
#
# Collapsing needs a priority, and the schema states one -- _definitions
# .restoration_types: "A tooth with a post belongs under post_and_core even if
# a crown also sits on top -- never double-count as crown." Post outranks
# crown, and crown outranks a filling for the same reason (a crown replaces the
# whole crown surface a filling would sit in).
COMPOSITE_RESTORATION_FLAGS = (
    ("with_post_and_core", "post_and_core"),
    ("with_full_crown",    "crown"),
    ("with_fillings",      "fillings"),
)


def composite_morphology(teeth: Dict, fdi: int) -> Dict:
    """This tooth's tooth_{fdi}_morphology block, or {} if it was never read."""
    return _as_dict(_as_dict(teeth.get(f"tooth_{fdi}")).get(f"tooth_{fdi}_morphology"))


def composite_restoration_type(morphology: Dict) -> Optional[str]:
    """
    The morphology bools as the single restoration enum the gates expect:
    "post_and_core" | "crown" | "fillings" | "none" | None.

    None vs "none" is the ABSTAIN/DENY distinction the whole gate rests on, so
    it is drawn carefully: "none" (an actual denial, which can outvote an arch
    claim) is returned only when the block ANSWERED at least one of the three
    flags. A block where all three are missing means the call never came back,
    which must stay None -> ABSTAIN, not a denial of every restoration.
    """
    morphology = _as_dict(morphology)
    for flag, value in COMPOSITE_RESTORATION_FLAGS:
        if morphology.get(flag) is True:
            return value
    if any(morphology.get(flag) is not None for flag, _ in COMPOSITE_RESTORATION_FLAGS):
        return "none"
    return None


# ── The panoramic tooth-by-tooth read (dental_arch_findings_{arch}) ────────

# Its per-tooth enum, and how each value maps onto the per-tooth composite
# facts it is cross-checked against. "crown" is deliberately absent from the
# arch read (see the schema's own note: too easily confused with a large
# filling at panoramic resolution), which is why a tooth reading "crown" on
# the composite side is skipped rather than counted as a mismatch.
#
# schema v7.1 coarsened the PROMPT enum to five findings -- restoration covers
# filling/crown/bridge-abutment, defect covers caries/root-remnant, and a
# posted tooth is filed under root_canal_treatment rather than getting its own
# value. Handled the same way as the v7.0 "unremarkable" rename: the new words
# are folded onto the internal tokens by ARCH_VALUE_ALIASES, so post_and_core
# on "filling"/"caries" below is untouched.
#
# "post_and_core" stays in this tuple even though no v7.1 answer can contain
# it: stored v6.x predictions do, and an out-of-enum value is read as
# "unreadable", which would silently turn every posted tooth in those files
# into a tooth nobody looked at. Reading old artefacts is the whole reason this
# tuple is separate from the schema's own enum.
ARCH_FINDING_VALUES = ("absent", "normal", "caries", "filling", "post_and_core",
                       "root_canal_treatment", "impacted")

# schema v7.0 renamed the residual "normal" -> "unremarkable", because the enum
# has no not-assessable value and all sixteen positions must carry one, so the
# word had to stop reading as a claim that the tooth is healthy.
#
# The rename is deliberately confined to the PROMPT. Everything below this line
# keeps "normal" as its internal token, and arch_findings_map folds the new
# word onto it -- some twenty behaviours in this file are keyed on that string,
# a few of which are dict keys rather than the enum, and renaming them all
# would make the v7.0 arm differ from v6.9 in two ways at once (the wording AND
# the code). It has to differ in one, or the measurement says nothing about
# which change moved the number.
#
# It is also what keeps v6.9 artefacts readable: the 582 extracted GT files
# and every stored prediction of that era were written as "normal", and they
# are still scored by this file.
# v7.1 adds two more prompt-side words to fold. "restoration" -> "filling"
# keeps crowned teeth behaving exactly as they did, because the crown skip
# rule in build_restoration_summary is what decides those, not this token;
# "defect" -> "caries" is the same coarsening in the other direction, and a
# root remnant now arrives here as "caries" (both are lost tooth substance, and
# the composite read is what tells them apart).
ARCH_VALUE_ALIASES = {"unremarkable": "normal",
                      "restoration": "filling",
                      "defect": "caries"}

# v6.2 made the findings map FIXED-KEY: all 16 of the arch's FDIs are answered
# every time, and absence moved from "key omitted" into the value "absent".
# Both conventions are read here -- a v6.1 prediction still carries present
# teeth only -- so a tooth counts as absent when its value SAYS absent or when
# it is missing from the map entirely. On a v6.2 answer the second half never
# fires (nothing is missing); on a v6.1 one the first half never fires
# (nothing says "absent"). See collect_arch_absent.
ARCH_ABSENT_VALUE = "absent"

# dental_arch_findings uses "filling"; tooth_{fdi}_restoration uses "fillings".
#
# The post-and-core row changed with schema v7.1. The arch read no longer has a
# post_and_core value -- a posted tooth is filed under root_canal_treatment,
# because at panoramic resolution a post and a well-condensed canal filling are
# the same bright material in the same canal. So root_canal_treatment is now
# what corroborates a composite post_and_core, and the gate keeps working on
# the evidence the image can actually carry instead of on a distinction it
# cannot. A tooth the panoramic reads as "filling" still DENIES a composite
# post (filling is lower priority), exactly as it did before.
ARCH_TO_RESTORATION = {"filling": "fillings",
                       "root_canal_treatment": "post_and_core"}


def arch_findings_map(fact, fdi_list: List[int], arch_name: str = "",
                      notes: Optional[List[str]] = None) -> Dict[int, str]:
    """
    dental_arch_findings_{arch}.findings as {fdi: finding}, keyed by real ints
    and filtered to this arch's own FDI range.

    QUADRANT LEAK. Both arch calls are handed the SAME full-mouth panoramic
    and are scoped to one arch by the prompt text alone, so the model
    routinely answers with the other arch's teeth -- a maxilla read that lists
    41-48, which is anatomically impossible for that fact. An FDI names its
    arch unambiguously, so the range filter here (not the key the answer
    arrived under) decides, and the stray entries are dropped outright rather
    than being trusted or re-homed: the opposite arch has its own read of the
    same picture, and an answer the model filed under the wrong arch is
    evidence it lost track of which half it was looking at, not a finding to
    salvage.

    Measured on the 40 v5 validate predictions: 15 stray keys across 3 cases
    (14 maxilla, 1 mandible). Small, but worth a note rather than a silent
    drop -- a case that leaks is a case whose whole arch read is suspect, and
    that is invisible in the summary otherwise.

    Non-numeric keys are dropped by the same rule (_as_fdi returns None), and
    counted separately: those are a different failure -- a malformed key
    rather than a confused one.
    """
    findings = _as_dict(_as_dict(fact).get("findings"))
    arch = set(fdi_list)
    out: Dict[int, str] = {}
    stray: List[int] = []
    malformed: List[str] = []
    for raw_fdi, value in findings.items():
        fdi = _as_fdi(raw_fdi)
        if fdi is None:
            malformed.append(str(raw_fdi))
            continue
        if fdi not in arch:
            stray.append(fdi)
            continue
        if not isinstance(value, str):
            continue
        value = value.strip().lower()
        # Out-of-enum values are dropped, not carried: this map is free-keyed,
        # so normalize_pred.py has no per-value spec to repair it against, and
        # every consumer here matches the enum exactly. A tooth whose value is
        # dropped still counts as PRESENT (its key is what says so).
        value = ARCH_VALUE_ALIASES.get(value, value)
        out[fdi] = value if value in ARCH_FINDING_VALUES else "unreadable"

    if stray:
        _note(notes, f"{arch_name}: dental_arch_findings_{arch_name} answered for "
                     f"{sorted(stray)} -- teeth of the OTHER arch, anatomically "
                     f"impossible for this fact; dropped")
    if malformed:
        _note(notes, f"{arch_name}: dental_arch_findings_{arch_name} had "
                     f"non-FDI keys {sorted(malformed)}; dropped")
    return out


def arch_findings_teeth(findings: Dict[int, str], value: str) -> set:
    """Every FDI the arch read gave this particular finding."""
    return {fdi for fdi, v in findings.items() if v == value}


# ── The uncertainty gate ──────────────────────────────────────────────────
#
# THE FLAG. When True, every tooth the model listed in
# dental_arch_findings_{arch}.uncertain_teeth is forced back to "normal"
# before anything downstream reads the map.
#
# Why a deterministic gate rather than trusting the prompt's base-rate
# instruction alone: measured on the first 5 v6.1 predictions, the arch read
# called 43.4% of teeth abnormal, and the per-tooth composite read called 44%
# of teeth crowned and 44% root-filled -- against reference reports where
# post_and_core appears in 6% of REPORTS at all. The v4 evaluation shows the
# same asymmetry from the other side: logical_precision 0.187 against
# logical_recall 0.354, i.e. the reports over-claim. Every demoted tooth
# removes a sentence from the report, so this trades recall for precision on
# purpose, against the weaker of the two numbers.
#
# TURNED OFF (2026-08-07). The measured trade never materialised: across the
# v5/v6 validate runs the gates in this file cost recall without moving
# precision (see REQUIRE_CROSS_SOURCE_AGREEMENT below for the per-finding
# numbers), and this one is the same shape -- "the model said it was unsure"
# is an argument for weighting the claim, not for overwriting it with a
# confident "normal". Postprocess is a BRIDGE from prediction to report now:
# it derives, groups and reconciles shapes, and does not delete findings.
#
# The gate itself is kept and still RUNS: the summary records what it would
# have demoted under arch_findings.uncertain_demoted either way, so the arm is
# one flag (or --demote-uncertain) away and remains measurable.
DEMOTE_UNCERTAIN_TO_NORMAL = False


def demote_uncertain_findings(findings: Dict[int, str], fact, fdi_list: List[int],
                              arch: str, notes: Optional[List[str]] = None,
                              demote: Optional[bool] = None) -> Tuple[Dict[int, str], List[int]]:
    """
    Force every self-declared uncertain tooth back to "normal".

    Returns (findings, demoted_fdis). `demoted_fdis` is reported whether or
    not the gate is on, so a run with DEMOTE_UNCERTAIN_TO_NORMAL = False
    still records exactly which teeth the gate would have taken out.

    Scope of the demotion, and what it deliberately does NOT do:
      - only teeth already in the findings map are touched. An uncertain FDI
        the model never listed is absent, and "uncertain" is not a reason to
        resurrect it as a present, normal tooth;
      - only the ARCH read is demoted. tooth_{fdi}_restoration and
        tooth_{fdi}_endodontic_treatment are separate calls against the
        tooth's own composite crop, and the model's doubt about a 16-tooth
        panoramic survey says nothing about what it saw close up. A finding
        both reads agree on therefore survives this gate, which is the
        intended behaviour -- the gate removes findings resting on the weaker
        source alone.
    """
    demote = DEMOTE_UNCERTAIN_TO_NORMAL if demote is None else demote
    arch_fdis = set(fdi_list)

    raw = _as_dict(fact).get("uncertain_teeth")
    uncertain = {f for f in as_fdi_list(raw) if f in arch_fdis}
    demoted = sorted(f for f in uncertain
                     if findings.get(f) not in (None, "normal"))
    if not demoted:
        return findings, []

    if demote:
        findings = {**findings, **{f: "normal" for f in demoted}}
        _note(notes, f"{arch}: {demoted} demoted to normal "
                     f"(model listed them as uncertain)")
    else:
        _note(notes, f"{arch}: {demoted} flagged uncertain by the model but "
                     f"KEPT (DEMOTE_UNCERTAIN_TO_NORMAL is off)")
    return findings, demoted


# ── The cross-source agreement gate ───────────────────────────────────────
#
# Several findings are asked TWICE, of two different images: once of the
# whole-arch panoramic (dental_arch_findings_{arch}) and once of that tooth's
# own composite crop (tooth_{fdi}_restoration / _endodontic_treatment /
# _endodontic_treatment). The schema says so itself -- "Cross-checked in
# postprocessing against tooth_{fdi}_morphology and
# tooth_{fdi}_endodontic_treatment for the same tooth".
#
# Until now that cross-check only ANNOTATED: the builders reported the UNION
# of the two reads and listed the disagreements under "conflicting_sources",
# so a finding exactly one source claimed -- and the other explicitly
# denied -- still reached the report. THE GATE BELOW DROPS IT INSTEAD.
#
# Measured on outputs/aksssr_v5_validate/predictions (40 cases), counting only
# teeth where BOTH reads actually answered:
#
#     root_canal_treatment    3 agree   79 contradict
#     fillings                2 agree   52 contradict
#     post_and_core           1 agree   17 contradict
#
# The two reads are very nearly uncorrelated, so this is a much heavier gate
# than DEMOTE_UNCERTAIN_TO_NORMAL: it removes ~96% of the restorative and
# endodontic findings these sections used to carry. That is a deliberate
# precision-for-recall trade, in the same direction as the uncertainty gate
# and for the same reason (v4 scored logical_precision 0.187 against
# logical_recall 0.354 -- the reports over-claim), but it is large enough that
# it must be measurable both ways. Hence the flag: set it False (or pass
# --no-cross-validate) to restore the union behaviour. The dropped teeth are
# recorded under arch_findings.cross_source_dropped either way, so one run
# shows exactly what the gate took out.
#
# TURNED OFF (2026-08-07) -- THIS IS NOW THE MASTER SWITCH, AND IT IS OFF.
# The per-finding exemptions below were added one at a time, and each one came
# back with the same measurement: the gate halved or quartered recall and left
# precision flat (0.16 -> 0.17 endodontic, 0.17 -> 0.17 fillings, 0.21 -> 0.19
# impaction). Every finding that was ever surveyed answered the same way, so
# the exemption list had become the rule and the gate the exception.
# Postprocess is a BRIDGE between the predictions and the report: it derives
# the values the schema no longer asks for, groups teeth that share a finding,
# and shapes the summary the renderer consumes -- it does not adjudicate the
# model's claims.
#
# WHAT THIS ONE FLAG NOW TURNS OFF, because each is subordinate to it:
#   - the per-finding cross-source votes (CROSS_VALIDATE_IMPACTION,
#     CROSS_VALIDATE_ENDODONTIC, CROSS_VALIDATE_RESTORATIONS);
#   - the caries agreement gate (REQUIRE_CARIES_ARCH_AGREEMENT);
#   - the implant-on-present-tooth gate (DROP_IMPLANTS_ON_PRESENT_TEETH);
#   - the bone-quality per-tooth confirmation
#     (REQUIRE_BONE_QUALITY_TOOTH_CONFIRMATION).
# The subordinate flags KEEP their measured values, so the record of which
# finding each measurement was taken on survives; with the master off they are
# all inert, and --cross-validate restores the whole previous arm exactly.
#
# WHAT IS DELIBERATELY NOT TURNED OFF, because it is a consistency repair
# rather than a precision gate -- it removes output that CANNOT exist, not
# output that MIGHT be wrong: per-tooth findings on absent teeth
# (toothless_fdis), canal adjacency claimed for anterior or opposite-side teeth
# (CANAL_ADJACENT_FDIS), implants placed at the other arch's FDIs, bridges with
# no parseable span, and enforce_scope_consistency.
#
# Every vote still RUNS with the gate off and every loser is still recorded
# under arch_findings.cross_source_dropped, so this arm stays fully auditable
# and one flag away from the old behaviour.
REQUIRE_CROSS_SOURCE_AGREEMENT = False

# ── ...except for impaction, which is exempt ──────────────────────────────
#
# The gate above assumes the sources are interchangeable, so that a
# disagreement is evidence against the finding. For impaction they are not,
# and the gate inverts quality instead of filtering it. Measured on
# outputs/aksssr_v5_validate against the 32 impacted teeth the reference
# reports state (survey_findings.py --category impaction):
#
#     source      claims   correct   precision
#     3d_render       74        21        0.28
#     panoramic       36         9        0.25
#     composite       18         0        0.00
#
# The composite read never once identifies an impacted tooth correctly, and
# on a wisdom tooth it is the source that DENIES: its "none" outvotes the
# 3d_render read that is right nearly a third of the time. Meanwhile a lone
# composite claim on a NON-wisdom tooth passes unchallenged, because the
# other two sources are wisdom-only and abstain there. So the gate removes
# the good source's true positives and keeps the bad source's false ones:
#
#     stage             claims   correct   precision   recall
#     union (no gate)      112        24        0.21     0.75
#     majority vote         26         5        0.19     0.16
#
# It discarded 19 of the 24 true impactions the pipeline had actually found,
# for no precision gain at all (0.19 vs 0.21). Recall drops 4.7x to buy
# nothing, so impaction does not pass through the gate.
#
# This is a per-finding exemption, NOT a change to the gate: the restorative
# and endodontic findings still cross-validate, where the same survey shows
# the two reads really are of comparable quality (endodontic: composite 0.18
# vs panoramic 0.24) and a disagreement is genuinely uninformative.
# Set this True to restore the majority vote; --no-cross-validate still turns
# everything off at once. Either way, the teeth the vote WOULD have removed
# are still recorded under arch_findings.cross_source_dropped.impacted, so a
# single run shows exactly what the exemption let through.
CROSS_VALIDATE_IMPACTION = False

# ── ...and the same exemption, on the same evidence, for endodontic
#    treatment and fillings ────────────────────────────────────────────────
#
# Impaction was exempted above because the gate cost recall and bought no
# precision. Measuring the other gated findings the same way
# (survey_findings.py, outputs/aksssr_v5_validate, summary stage):
#
#     finding      gate ON                    gate OFF
#                  claims tp prec  rec        claims tp prec  rec
#     endodontic      69  11 0.16  0.16         146  24 0.17  0.34
#     fillings        18   3 0.17  0.03          70  12 0.17  0.12
#     post_and_core   12   0 0.00  0.00          29   0 0.00  0.00
#     crown           72  13 0.19  0.17          72  13 0.19  0.17
#
# Endodontic and fillings are the impaction story again: recall doubles or
# quadruples, precision does not move (0.16 -> 0.17 and 0.17 -> 0.17). The
# gate was discarding 13 true endodontic teeth and 9 true fillings to remove
# nothing it could show was wrong, so both are exempt.
#
# The other two stay gated, because for them the gate is NOT what makes the
# summary worse:
#   post_and_core -- 0 correct with the gate and 0 correct without it, so
#     there is no recall to recover; turning it off only adds 17 more claims
#     that are all false. Nothing to win, noise to lose.
#   crown -- the panoramic arch read has no crown value, so the composite is
#     the only voter and the gate never had anything to reconcile. Identical
#     numbers on and off; the flag would be decoration.
#   caries -- not surveyed. No ground truth was read for it, so there is no
#     evidence to act on and it keeps the project default.
CROSS_VALIDATE_ENDODONTIC = False
CROSS_VALIDATE_RESTORATIONS: Dict[str, bool] = {
    "fillings": False,
    "post_and_core": True,
    "crown": True,
}

AGREE, DENY, ABSTAIN = "agree", "deny", "abstain"

# How a split vote is resolved -- see reconcile_sources.
UNANIMOUS, MAJORITY = "unanimous", "majority"

# The arch read carries ONE finding per tooth, resolved by the schema's own
# priority order -- v7.1: "impacted > root_canal_treatment > restoration >
# defect > unremarkable", i.e. these tokens once the aliases are folded in.
# This is what makes ABSTAIN a distinct verdict from DENY: a tooth that is both
# root-filled and restored is filed under root_canal_treatment there, so the
# arch read's silence about the restoration is a consequence of the priority
# rule, not a denial of it. post_and_core keeps its old rank for the sake of
# stored v6.x predictions; no v7.1 answer reaches it, so for a v7.1 run this
# tuple behaves exactly as the four-value order above.
ARCH_FINDING_PRIORITY = ("impacted", "post_and_core", "root_canal_treatment",
                         "filling", "caries")


def arch_verdict(findings: Dict[int, str], fdi: int, target: str) -> str:
    """
    What dental_arch_findings_{arch} says about `target` for one tooth.

      AGREE   -- it read this tooth as exactly that finding;
      DENY    -- it read this tooth and gave an answer that rules the finding
                 out: "normal", or a LOWER-priority finding (had the target
                 been visible, the priority rule would have reported it
                 instead of what it did report);
      ABSTAIN -- it cannot speak. Five ways that happens: the tooth is not in
                 the map at all (never read); its value was out of enum
                 ("unreadable"); it was read as "absent", and a tooth that is
                 not there carries no finding; a HIGHER-priority finding masks
                 the target; or the arch enum has no value for the target in
                 the first place (crown, deliberately excluded by the schema as
                 indistinguishable from a large filling on a panoramic).
    """
    value = findings.get(fdi)
    if value is None or value == "unreadable" or value == ARCH_ABSENT_VALUE:
        return ABSTAIN
    if target not in ARCH_FINDING_PRIORITY:
        return ABSTAIN
    if value == target:
        return AGREE
    if (value in ARCH_FINDING_PRIORITY
            and ARCH_FINDING_PRIORITY.index(value) < ARCH_FINDING_PRIORITY.index(target)):
        return ABSTAIN
    return DENY


# Whether a composite "carious" needs the panoramic not to contradict it.
#
# WHY IT EXISTS. Caries used to be ungated while everything beside it was, and
# the asymmetry showed: on A008 tooth 31 the composite claimed BOTH
# post_and_core and "carious" against a panoramic reading of "normal". The
# post was gated away; the caries -- same composite's word, same panoramic
# contradiction, same tooth -- went through to the report untouched. It
# survived because "caries" was not in ARCH_FINDING_PRIORITY, so arch_verdict
# abstained on it by construction.
#
# WHY IT IS NOW HONEST. Schema v6.2 adds "caries" to the arch enum and to the
# priority order, and tells the arch reader to note caries alongside
# restorations. So a "normal" from a v6.2 arch read is a real denial: the
# reader was asked about decay and recorded none. arch_verdict handles the
# target directly and no caries-specific verdict function is needed --
# "filling"/"post_and_core"/etc. outrank caries and therefore ABSTAIN (a
# restored tooth is frequently restored because of decay, and the enum holds
# one value per tooth), while "normal" falls through to DENY.
#
# THE ONE CAVEAT, and the reason this stays a separate flag. Predictions
# generated BEFORE v6.2 -- everything in outputs/aksssr_v5_* -- come from an
# arch reader that was never asked about caries, so their "normal" denies
# restorations only. Gating those is a precision play, not a deduction, and it
# will drop true caries on teeth the panoramic simply was not looking for
# decay on. Set this False (or pass --no-cross-validate) to measure a run
# without it. Once inference is re-run under v6.2 the caveat lapses.
#
# Subordinate to REQUIRE_CROSS_SOURCE_AGREEMENT: --no-cross-validate turns
# this off too, so one flag still restores the full union behaviour.
REQUIRE_CARIES_ARCH_AGREEMENT = True


# ── Caries exclusivity ────────────────────────────────────────────────────
#
# THE CLINICAL RULE, as given: a crown, a post-and-core or a filling EXCLUDES
# caries on the same tooth -- the decay is what the restoration replaced, so
# the restored surface cannot also be reported carious. (Crown, post-and-core
# and root canal treatment do co-exist with EACH OTHER; that is handled by
# arch_verdict's priority masking and by the skip=("crown",) rule in
# build_restoration_summary, not here.)
#
# Root canal treatment and impaction are deliberately NOT on the denying list:
# neither replaces a crown surface, and an impacted tooth can certainly decay.
RESTORATIONS_EXCLUDING_CARIES = ("fillings", "crown", "post_and_core")
# v7.1: "restoration" folds to "filling", and it now covers crowns and bridge
# abutments too -- so this one token denies caries for all three, which is what
# the composite-side list beside it has always done. post_and_core dropped out
# with the enum; a posted tooth reaches here as root_canal_treatment, which is
# deliberately NOT a denial (see above).
ARCH_VALUES_EXCLUDING_CARIES = ("filling",)


def arch_caries_verdict(findings: Dict[int, str], fdi: int) -> str:
    """
    The panoramic's verdict on caries for one tooth.

    Not arch_verdict: that masks a lower-priority target behind any
    higher-priority value, so "filling" would ABSTAIN on caries. Under the
    exclusivity rule above the masking case does not exist -- a filled tooth
    is not also carious -- so a restoration read is a DENIAL, not a silence.

      AGREE   -- read as "caries" (v6.2 enum).
      DENY    -- read as "normal" (asked about decay, recorded none) or as a
                 restoration that excludes it.
      ABSTAIN -- never read, out of enum, or "root_canal_treatment"/"impacted",
                 neither of which rules decay out.
    """
    value = findings.get(fdi)
    if value is None or value == "unreadable":
        return ABSTAIN
    if value == "caries":
        return AGREE
    if value == "normal" or value in ARCH_VALUES_EXCLUDING_CARIES:
        return DENY
    return ABSTAIN


def restoration_caries_verdict(restoration_type) -> str:
    """
    The composite's OWN restoration read as a verdict on caries.

    This is the tooth's internal consistency check, and it needs no second
    image: on A008 tooth 31 the same composite call claimed a post-and-core
    replacing the crown AND a carious crown. Both cannot describe one surface.
    """
    if not isinstance(restoration_type, str) or not restoration_type:
        return ABSTAIN
    return DENY if restoration_type in RESTORATIONS_EXCLUDING_CARIES else ABSTAIN


def gate_caries(findings: Dict[int, str], teeth: Dict, fdi_list: List[int],
                absent: Optional[set] = None,
                cross_validate: Optional[bool] = None,
                arch: str = "", notes: Optional[List[str]] = None) -> List[int]:
    """
    Cross-validate each composite "carious" against the arch read.

    THREE sources, not two:
      composite   -- tooth_{fdi}_morphology.with_caries, the claim itself (a
                     plain bool since v6.4, where it was the enum value
                     crown_morphology == "carious");
      panoramic   -- dental_arch_findings_{arch}, see arch_caries_verdict;
      restoration -- the SAME composite block's restorative bools, collapsed by
                     composite_restoration_type, which excludes caries when
                     they name a crown, post-and-core or filling.

    Resolved UNANIMOUS: exclusivity is a hard rule, so one restoration DENY
    settles it -- it is not outvoted by the other two.

    Returns the dropped FDIs -- caries is not a summary block of its own, it
    only ever appears inside build_tooth_findings, so unlike the endo and
    restoration gates there is no summary to return alongside.
    """
    absent = absent or set()
    if not REQUIRE_CARIES_ARCH_AGREEMENT:
        return []

    votes: Dict[int, Dict[str, str]] = {}
    for fdi in fdi_list:
        if fdi in absent:
            continue
        morphology = composite_morphology(teeth, fdi)
        # A root remnant has no crown to be carious about -- the schema says
        # so itself ("false if is_remnant is true"), and build_tooth_findings
        # already skips those teeth.
        if morphology.get("is_remnant"):
            continue
        votes[fdi] = {
            "composite": bool_verdict(morphology.get("with_caries")),
            "panoramic": arch_caries_verdict(findings, fdi),
            "restoration": restoration_caries_verdict(
                composite_restoration_type(morphology)),
        }

    _, dropped = reconcile_sources(votes, cross_validate)
    note_dropped(notes, arch, "caries", dropped, votes, cross_validate)
    return dropped


def enum_verdict(value, target, skip=()) -> str:
    """
    A composite-read enum field's verdict on `target`.

    None/missing -> ABSTAIN (the call was never made, or came back empty);
    a value in `skip` -> ABSTAIN (a named "not comparable" case, e.g. a tooth
    the composite read as "crown" when the target is a filling);
    otherwise AGREE if it IS the target, DENY if it is any other real answer.
    """
    if value is None or value == "":
        return ABSTAIN
    if value in skip:
        return ABSTAIN
    return AGREE if value == target else DENY


def bool_verdict(value) -> str:
    """A composite-read boolean's verdict: True agrees, False denies, None abstains."""
    if value is None:
        return ABSTAIN
    return AGREE if value else DENY


def gate_is_on(cross_validate: Optional[bool] = None) -> bool:
    """The effective gate setting: an explicit argument wins over the module flag."""
    return REQUIRE_CROSS_SOURCE_AGREEMENT if cross_validate is None else cross_validate


def reconcile_sources(votes: Dict[int, Dict[str, str]],
                      gate: Optional[bool] = None,
                      rule: str = UNANIMOUS) -> Tuple[List[int], List[int]]:
    """
    Apply the gate to one finding's per-tooth votes.

    `votes` maps FDI -> {source_name: AGREE|DENY|ABSTAIN}.

    Two decision rules, differing only in how they treat a split vote:

      UNANIMOUS -- a single DENY kills the finding. Right for the TWO-source
                   findings (restorations, root fillings), where a split is
                   1-vs-1 and there is no majority to be had: with only two
                   opinions, "one of them says no" is the whole of the
                   evidence against.
      MAJORITY  -- the finding survives when more sources AGREE than DENY.
                   Right for impaction, the one finding with THREE independent
                   sources, where a 2-vs-1 split really is corroboration
                   against a single dissent rather than a coin flip. A tie
                   does not survive -- a majority has to be a majority.

    (On a two-source finding the two rules coincide, since 1-vs-1 is a tie
    either way. The distinction is kept explicit rather than implicit so that
    adding a third source to a finding is a deliberate decision about which
    rule it should then follow.)

    Returns (kept, dropped), both sorted:
      kept    -- the teeth to report;
      dropped -- the teeth where the vote went against the finding. Always
                 reported, whether or not the gate is on, so a run with the
                 flag off still records exactly what the gate would have taken.

    A finding only ONE source can speak to (every other source ABSTAINing) is
    NOT dropped under either rule: there is no second opinion to disagree with
    it, and requiring corroboration is a stricter rule than requiring
    agreement. The single remaining source is the schema's designated one in
    every such case (crown is composite-only, non-wisdom impaction is
    composite-only).
    """
    gate = gate_is_on(gate)
    kept, dropped = [], []
    for fdi in sorted(votes):
        verdicts = list(votes[fdi].values())
        agree, deny = verdicts.count(AGREE), verdicts.count(DENY)
        if not agree:
            continue
        survives = agree > deny if rule == MAJORITY else deny == 0
        if survives:
            kept.append(fdi)
        else:
            dropped.append(fdi)
            if not gate:
                kept.append(fdi)
    return kept, dropped


def note_dropped(notes: Optional[List[str]], arch: str, finding: str,
                 dropped: List[int], votes: Dict[int, Dict[str, str]],
                 gate: Optional[bool] = None,
                 off_reason: str = "REQUIRE_CROSS_SOURCE_AGREEMENT is off") -> None:
    """One traceable line per gated finding, naming who said what.

    `off_reason` names the switch that spared these teeth, since a finding
    can be exempt on its own (impaction) rather than because the whole gate
    is off.
    """
    if not dropped:
        return
    gate = gate_is_on(gate)
    detail = "; ".join(
        f"{fdi} ({', '.join(f'{src}={v}' for src, v in sorted(votes[fdi].items()) if v != ABSTAIN)})"
        for fdi in dropped)
    verb = "dropped" if gate else f"KEPT ({off_reason})"
    _note(notes, f"{arch}: {finding} {verb} on sources disagreeing -- {detail}")


# ── Where an arch's absent-teeth list comes from ──────────────────────────

# Per-tooth "detected" values that mean "this tooth is not in the mouth".
#   "no"       -- the panoramic read did not list it among the present teeth
#                 (run_vqa_inference writes the flag from model_absent_teeth).
#   "no_image" -- create_tooth_detail.py never produced a composite for it,
#                 i.e. the segmentation carries no label for that FDI.
# The second is the stronger of the two and has no equivalent anywhere in the
# global dict, which is why it has to be read from `teeth` here: on the v4
# validate run it is the only source that catches A008's tooth 16 (absent per
# the reference report, never mentioned by any global call).
ABSENT_DETECTED_VALUES = ("no", "no_image")

# Per-tooth eruption states that mean the tooth IS in the mouth. A tooth the
# composite call actually described is present even if the panoramic read
# forgot to list it -- see collect_arch_absent.
# Every eruption_state except "absent" means the tooth IS there. Schema v6.4
# renamed "not_erupted" -> "complete_bony_inclusion"; the legacy name is kept
# so an older prediction file still resolves to "present" rather than silently
# falling through to absent.
PRESENT_ERUPTION_STATES = ("complete_bony_inclusion", "partially_erupted",
                           "fully_erupted", "not_erupted")


def collect_arch_absent(teeth: Dict, fdi_list: List[int], arch: str,
                        findings: Optional[Dict[int, str]] = None,
                        notes: Optional[List[str]] = None) -> Tuple[List[int], bool]:
    """
    The set of absent FDIs for ONE arch, and whether absence was assessable
    at all.

    Schema v6.1 dropped absent_teeth_{arch} entirely, so absence is no longer
    asked anywhere -- it is DERIVED from three sources, in strength order:

      1. detected == "no_image": create_tooth_detail.py produced no composite,
         i.e. the segmentation carries no label at that position. Strongest,
         and independent of anything the model said.
      2. tooth_{fdi}_eruption.eruption_state: "absent" means absent, and any
         of PRESENT_ERUPTION_STATES means present. This is a read of the
         tooth's OWN composite crop, so it outranks (3) both ways.
      3. dental_arch_findings_{arch}.findings: a tooth whose value is "absent"
         (v6.2's fixed-key form), or -- for v6.1 predictions, whose map
         carried present teeth only -- an arch FDI missing from the map. This
         is a single panoramic read covering 16 teeth at once and it drops
         teeth it simply didn't get to, which is why it never overrides (2).

    detected == "yes" on its own is NOT counted as presence: it only says a
    composite image was generated and a call was made, and run_vqa_inference
    deliberately still calls teeth the panoramic declared absent
    (TRUST_MODEL_ABSENCE = False). So a tooth whose per-tooth call came back
    empty falls through to (3) rather than being asserted present on the
    strength of having been asked about.

    Returns (sorted absent FDIs, assessable). assessable is False only when
    NO source said anything about this arch -- no per-tooth entries at all AND
    no arch findings map -- which is a failed/missing call, not a finding of
    "all 16 teeth present". See classify_absent_teeth_pattern.
    """
    findings = findings or {}
    listed_present = {fdi for fdi, v in findings.items() if v != ARCH_ABSENT_VALUE}

    absent: set = set()
    seen = bool(findings)
    from_arch_read: List[int] = []

    for fdi in fdi_list:
        tooth = _as_dict(teeth.get(f"tooth_{fdi}"))
        detected = tooth.get("detected")
        eruption = _as_dict(tooth.get(f"tooth_{fdi}_eruption")).get("eruption_state")
        if detected is not None or eruption is not None:
            seen = True

        if detected in ABSENT_DETECTED_VALUES:
            absent.add(fdi)
            continue
        if eruption == "absent":
            absent.add(fdi)
            continue
        if eruption in PRESENT_ERUPTION_STATES:
            continue                      # the tooth's own crop wins
        if findings and fdi not in listed_present:
            absent.add(fdi)
            from_arch_read.append(fdi)

    if from_arch_read:
        _note(notes, f"{arch}: {sorted(from_arch_read)} absent per "
                     f"dental_arch_findings_{arch} alone (no per-tooth answer)")

    return sorted(absent), seen


def _note(notes: Optional[List[str]], message: str) -> None:
    if notes is not None:
        notes.append(message)


# ── Absent-teeth pattern classification (unchanged logic; simpler inputs) ──

def classify_absent_teeth_pattern(absent_fdis: List[int],
                                  quadrants: List[Tuple[str, int, int]],
                                  assessable: bool = True) -> Dict:
    """
    Classify the absent-teeth pattern within ONE arch (its 2 quadrants).
    absent_fdis is ALREADY arch-scoped by collect_arch_absent, which routes
    every FDI to its own arch by number -- no whole-mouth-list filtering
    needed before calling this, unlike the v2/v3 version.

    assessable=False means no source spoke about this arch at all (its global
    call failed AND no per-tooth flags exist). That is NOT the same fact as
    "no teeth are absent", and collapsing the two is how a case whose maxilla
    call returned nothing came to assert all 16 upper teeth present: pattern
    "none" renders as "Teeth 11 to 18 and 21 to 28 are present." Such an arch
    gets pattern "unknown" instead, which the renderer drops rather than
    asserting anything about teeth nobody looked at.

    NOTE / caveat: "consecutive run" is only evaluated WITHIN a single
    quadrant here -- adjacent FDI integers across a quadrant boundary
    (e.g. 38 and 41) are NOT anatomically adjacent teeth. Best-effort
    heuristic, worth spot-checking against real cases with unusual
    absence patterns.
    """
    all_fdis = [f for _, lo, hi in quadrants for f in fdis_in_range(lo, hi)]
    if not assessable:
        return {"pattern": "unknown"}
    absent = sorted(f for f in absent_fdis if f in all_fdis)
    if not absent:
        return {"pattern": "none", "present": all_fdis}

    present = sorted(set(all_fdis) - set(absent))

    qstats = []
    for label, lo, hi in quadrants:
        q_all = fdis_in_range(lo, hi)
        q_absent = [f for f in absent if lo <= f <= hi]
        qstats.append({
            "quadrant": label,
            "n_total": len(q_all),
            "n_absent": len(q_absent),
            "absent": q_absent,
        })

    full_quads = [q["quadrant"] for q in qstats if q["n_absent"] == q["n_total"] and q["n_total"] > 0]

    if len(full_quads) == len(quadrants):
        return {"pattern": "quadrant_edentulous_both", "quadrants": full_quads, "present": present}

    if len(full_quads) == 1:
        other = next(q for q in qstats if q["quadrant"] not in full_quads)
        if other["n_absent"] == 0:
            return {"pattern": "quadrant_edentulous_one", "quadrant": full_quads[0], "present": present}
        return {"pattern": "quadrant_edentulous_one_plus_partial",
                "quadrant": full_quads[0], "other_quadrant_absent": other["absent"], "present": present}

    if len(present) <= 3:
        return {"pattern": "nearly_edentulous", "remaining": present, "present": present}

    if _is_consecutive(absent):
        return {"pattern": "consecutive_run", "start": absent[0], "end": absent[-1],
                "teeth": absent, "present": present}

    return {"pattern": "scatter", "teeth": absent, "present": present}


# ── Canals: NO merge -- always two independent entries ────────────────────
#
# TWO SOURCES SINCE schema v6.9, on both of this block's fields:
#
#   3d_render  -- mandible_canal_{side}, one read per side: a location and
#                 the list of teeth whose roots reach the canal.
#   composite  -- tooth_{fdi}_mandible_canal, one read per lower molar, off
#                 the coronal row of that tooth's own composite.
#
# They are not equally placed for the two fields, which is why the two are
# reconciled differently below (see CROSS_VALIDATE_CANAL_ADJACENCY and
# PREFER_COMPOSITE_CANAL_LOCATION).

# Does the composite's per-tooth adjacency read get to VETO the 3D read's?
#
# Unmeasured: this source is new, so there is no survey to point at the way
# there is for impaction and the restorations, and an unmeasured gate keeps
# the project default (True) rather than being switched off on a hunch --
# exactly the position REQUIRE_CARIES_ARCH_AGREEMENT is in.
#
# Subordinate to REQUIRE_CROSS_SOURCE_AGREEMENT, like every other per-finding
# gate, so on this arm (master off) the two sources UNION and every tooth the
# vote would have removed is still recorded in the run notes. Once
# survey_findings can measure canal adjacency, set it from the numbers.
CROSS_VALIDATE_CANAL_ADJACENCY = True

# Does the composite's location read outrank the 3D read's?
#
# YES, and this one is not a hunch. Buccal-vs-lingual is a depth judgement,
# and the 3D render is the wrong picture for it: the canal is a tube inside a
# semi-transparent bone surface, seen from the side. synthesize_report's
# DEFAULT_CANAL_LOCATION note measures what that costs -- of the 5 buccal
# sides in the validate references the 3D read got 0 right, having already
# answered "lingual" on four of them. A coronal cut is the only view that puts
# the canal and BOTH cortical plates in one picture, which is precisely why
# the per-tooth fact was written to be answered from that row alone.
#
# So the molar reads decide the side's location when they answer, the 3D read
# fills in when they do not, and synthesize_report's measured prior still
# covers the case where neither did. NOT subordinate to the cross-source gate:
# this is a choice of which reader to believe on an axis only one of them can
# see, not a precision gate that removes a finding.
PREFER_COMPOSITE_CANAL_LOCATION = True


def composite_canal(teeth: Dict, fdi: int) -> Dict:
    """This tooth's tooth_{fdi}_mandible_canal block, or {} if never read."""
    return _as_dict(_as_dict(teeth.get(f"tooth_{fdi}")).get(f"tooth_{fdi}_mandible_canal"))


def arch_canal_verdict(fact: Dict, side: str, fdi: int) -> str:
    """
    The 3D read's verdict on "this tooth's root reaches the canal".

      AGREE   -- the tooth is in mandible_canal_{side}.adjacent_teeth;
      DENY    -- the fact WAS answered and this tooth is not in the list. An
                 empty list is an answer, not a silence: the schema tells the
                 reader outright that empty is the normal reply, so a canal
                 read that names nobody has denied every tooth on its side;
      ABSTAIN -- the fact was never answered (null, or junk of a type this
                 field cannot be).

    A bare scalar counts as answered, not as junk: valid_canal_adjacent_teeth
    wraps it, and a prediction that came through normalize_pred has usually
    had it coerced to a list already.
    """
    value = fact.get("adjacent_teeth")
    answered = (isinstance(value, list) or isinstance(value, int)
                or (isinstance(value, str) and value.strip()))
    if not answered:
        return ABSTAIN
    return AGREE if fdi in valid_canal_adjacent_teeth(value, side) else DENY


def composite_canal_verdict(teeth: Dict, fdi: int) -> str:
    """
    The composite's verdict on the same claim, from this tooth's own coronal
    panels. A bool: true AGREEs, false DENIes, anything else (never asked --
    every tooth that is not a lower molar -- or answered null) ABSTAINs.
    """
    value = composite_canal(teeth, fdi).get("adjacent_to_teeth")
    if not isinstance(value, bool):
        return ABSTAIN
    return AGREE if value else DENY


def composite_canal_location(teeth: Dict, side: str) -> Optional[str]:
    """
    This side's buccolingual position as the molar composites read it, or None.

    Majority over whichever of the side's molars answered; a tie is None, not
    a coin flip -- two readers of the same canal disagreeing is exactly the
    case where the caller should fall through to the other source.
    """
    values = []
    for fdi in sorted(CANAL_TOOTH_FDIS.get(side, ())):
        loc = composite_canal(teeth, fdi).get("location")
        if loc in ("lingual", "buccal"):
            values.append(loc)
    if not values:
        return None
    lingual = values.count("lingual")
    buccal = values.count("buccal")
    if lingual == buccal:
        return None
    return "lingual" if lingual > buccal else "buccal"


def valid_canal_adjacent_teeth(value, side: str) -> List[int]:
    """
    Keep only FDIs that can anatomically touch the canal on this side
    (44-47 right, 34-37 left). Out-of-range codes, non-numeric junk and
    duplicates are dropped; an empty result means the caller should omit
    the adjacent_teeth field entirely.
    """
    allowed = CANAL_ADJACENT_FDIS.get(side, set())
    if not isinstance(value, list):
        value = [value] if value is not None else []
    kept = []
    for item in value:
        try:
            fdi = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if fdi in allowed and fdi not in kept:
            kept.append(fdi)
    return sorted(kept)


def build_single_canal(fact: Dict, side: str, absent: Optional[set] = None,
                       teeth: Optional[Dict] = None,
                       cross_validate: Optional[bool] = None,
                       notes: Optional[List[str]] = None) -> Dict:
    """
    mandible_canal_right/left no longer carry a .findings string (dropped
    per the "hard facts, not narrative" principle) -- location +
    adjacent_teeth are the whole of it now.

    location (schema v6.4, enum lingual|buccal) replaced the old `course`
    enum (regular|lingual|buccal). It is a POSITION axis, not an anomaly
    axis: every canal runs to one side or the other, so there is no "normal"
    value to branch a report sentence on any more, and no `normal` key is
    emitted for canals. The course sentence is unconditionally the regular
    one now; location surfaces only next to the roots it relates to (see
    synthesize_report.render_canal). Anything outside the enum -- including
    a legacy "regular" from a prediction generated under the old schema --
    is dropped to None, which reads as "position not stated".

    `absent` (see toothless_fdis) removes teeth that are not in the mouth: a
    canal cannot be in close relationship with the roots of a tooth that
    isn't there. In a fully edentulous mandible this empties the clause
    outright, which is the correct report.

    `teeth` brings the SECOND source (schema v6.9's per-molar
    tooth_{fdi}_mandible_canal). Both of this entry's fields are reconciled
    against it, by different rules:

      adjacent_teeth -- a per-tooth vote between the two readers, resolved
                        UNANIMOUS like every other two-source finding, and
                        gated by CROSS_VALIDATE_CANAL_ADJACENCY. The premolars
                        have no composite read at all, so there the 3D read
                        stands alone and reconcile_sources keeps it (a lone
                        source is never outvoted by silence).
      location       -- not a vote: the molar reads win outright where they
                        answer, see PREFER_COMPOSITE_CANAL_LOCATION.

    Without `teeth` -- a caller that has no per-tooth block, or a prediction
    generated before v6.9 -- every composite verdict ABSTAINs and the entry is
    exactly what the 3D read alone made it.
    """
    fact = _as_dict(fact)
    teeth = _as_dict(teeth)

    arch_location = fact.get("location")
    if arch_location not in ("lingual", "buccal"):
        arch_location = None
    tooth_location = composite_canal_location(teeth, side)
    if PREFER_COMPOSITE_CANAL_LOCATION and tooth_location:
        location, location_source = tooth_location, "composite"
    else:
        location, location_source = arch_location, "3d_render"
        if location is None and tooth_location:
            location, location_source = tooth_location, "composite"

    entry = {"side": side, "location": location}
    # Carried for auditing, rendered by nothing: which reader placed the
    # canal, and what the other one said when it disagreed. A location that
    # flips between two runs is then traceable to the source that changed.
    if location is not None:
        entry["location_source"] = location_source
        if arch_location and tooth_location and arch_location != tooth_location:
            entry["location_disagreement"] = {"3d_render": arch_location,
                                              "composite": tooth_location}

    votes: Dict[int, Dict[str, str]] = {}
    for fdi in sorted(CANAL_ADJACENT_FDIS.get(side, ())):
        votes[fdi] = {"3d_render": arch_canal_verdict(fact, side, fdi),
                      "composite": composite_canal_verdict(teeth, fdi)}
    gate = gate_is_on(cross_validate) and CROSS_VALIDATE_CANAL_ADJACENCY
    kept, dropped = reconcile_sources(votes, gate)
    note_dropped(notes, "mandible", f"canal_adjacent_{side}", dropped, votes, gate,
                 off_reason="CROSS_VALIDATE_CANAL_ADJACENCY is off")

    if absent:
        kept = [f for f in kept if f not in absent]
    if kept:
        entry["adjacent_teeth"] = sorted(kept)
    return entry


def build_canals(right_fact: Dict, left_fact: Dict, absent: Optional[set] = None,
                 teeth: Optional[Dict] = None,
                 cross_validate: Optional[bool] = None,
                 notes: Optional[List[str]] = None) -> Dict:
    """
    Explicitly NOT a Type B group anymore -- the mandible decision tree
    never has a combined "right and left canals with a regular course"
    sentence; each side always gets its own statement regardless of
    whether both happen to be normal. Always returns both sides.

    Each side reads only its OWN molars: CANAL_TOOTH_FDIS keeps 36-38 out of
    the right canal's evidence and 46-48 out of the left's, the same
    separation CANAL_ADJACENT_FDIS already enforces on the 3D read's list.
    """
    return {
        "right": build_single_canal(right_fact, "right", absent, teeth,
                                    cross_validate, notes),
        "left": build_single_canal(left_fact, "left", absent, teeth,
                                   cross_validate, notes),
    }


# ── Sinuses: STILL a Type B group -- the decision tree keeps a real merged
#    "both normal" sentence for this pair ─────────────────────────────────

def build_single_sinus(fact: Dict, side: str) -> Dict:
    fact = _as_dict(fact)
    mucosa = fact.get("mucosa_state", "normal")
    content = fact.get("sinus_content", "air")
    normal = mucosa == "normal" and content == "air"
    return {"side": side, "mucosa_state": mucosa, "sinus_content": content, "normal": normal}


def sinus_scope(fact: Optional[Dict]) -> str:
    """
    This side's inclusion in the acquisition, or "unknown" when the fact was
    never answered. Same three states as the condyles, plus the same refusal
    to guess at a missing one.
    """
    scope = _as_dict(fact).get("scope") if fact is not None else None
    return scope.strip() if isinstance(scope, str) and scope.strip() else "unknown"


def build_sinus_group(right_fact: Optional[Dict], left_fact: Optional[Dict]) -> Dict:
    """
    Sinus SCOPE is Type C -- always carried, both sides, in every case (the
    real reports state it constantly: "Maxillary sinuses not included in the
    scan volume." is one of the most common sentences in the corpus). Sinus
    CONTENT stays Type B: merged into one "all_normal" flag only when both
    sides are in scope AND both normal, with a single side reported alone
    when only one is in scope.

    Never returns None now: an out-of-scope pair still has a scope statement
    to make, and group_status "none_included" tells the renderer to make it
    and then stop (there is no mucosa to describe in a sinus that wasn't
    scanned).

    "unknown" counts as in scope, so an unanswered fact behaves exactly as
    it did before this change -- see the caveat in render_sinus_scope about
    the content that is then defaulted.
    """
    scopes = {"right": sinus_scope(right_fact), "left": sinus_scope(left_fact)}

    # NEITHER side was asked at all (the fact is absent from the prediction,
    # not merely incomplete). That absence is INFORMATIVE rather than merely
    # missing: the sinus questions are only asked when create_sinus_detail.py
    # produced a sinus crop, and it only produces one when there is a sinus in
    # the volume to crop. Verified across all 50 v4 cases -- "no sinus fact"
    # and "no {case}_sinus_*_detail image" coincide exactly, 50/50, no
    # mismatches. So the sinuses were not in the acquisition, and that is
    # stated as such (per explicit instruction this round) rather than
    # defaulted by build_single_sinus into an asserted healthy sinus.
    #
    # scope_source records that these two values were inferred here and did
    # not come from a VQA answer.
    if not _as_dict(right_fact) and not _as_dict(left_fact):
        return {"scope": {"right": "not_included", "left": "not_included"},
                "scope_source": "inferred_from_missing_fact",
                "group_status": "none_included"}

    right_in_scope = scopes["right"] != "not_included"
    left_in_scope = scopes["left"] != "not_included"

    if right_in_scope and left_in_scope:
        r = build_single_sinus(right_fact, "right")
        l = build_single_sinus(left_fact, "left")
        if r["normal"] and l["normal"]:
            # group_status still says all_normal -- that is what
            # render_sinuses keys on to fold the content into the scope
            # sentence, and the report text is unchanged. But the per-side
            # blocks are carried through now instead of being dropped: the
            # merge used to delete mucosa_state and sinus_content for BOTH
            # sides whenever both read normal, so a consumer asking the
            # summary "what did the model say about the right sinus mucosa?"
            # got nothing back for exactly the cases where it had an answer.
            # That cost 10 of 26 report-stated sides in the survey
            # (survey_findings.py) -- read as unanswered when they were
            # answered. A summary is the structured record, so it carries
            # what was read; grouping is the renderer's job, not the
            # record's.
            return {"scope": scopes, "group_status": "all_normal",
                    "right": r, "left": l}
        return {"scope": scopes, "group_status": "mixed", "right": r, "left": l}
    if right_in_scope:
        return {"scope": scopes, "group_status": "right_only",
                "right": build_single_sinus(right_fact, "right")}
    if left_in_scope:
        return {"scope": scopes, "group_status": "left_only",
                "left": build_single_sinus(left_fact, "left")}
    return {"scope": scopes, "group_status": "none_included"}


def build_intrasinusal_teeth(right_fact: Dict, left_fact: Dict) -> Dict:
    """
    Always stated (not silent when empty) -- a genuinely negative finding
    ("no roots project into either sinus") is itself worth a sentence.
    """
    right_fact, left_fact = _as_dict(right_fact), _as_dict(left_fact)
    combined = sorted(set(right_fact.get("intrasinusal_teeth") or []) |
                      set(left_fact.get("intrasinusal_teeth") or []))
    return {"teeth": combined, "state_negative": len(combined) == 0}


# ── Scope (Type C) -- rewritten: no coronoid, condyle inclusion independent ─

def scope_level(included: Optional[str]) -> str:
    """
    THREE levels, shared by both arches, but the two arches no longer feed it
    the same vocabulary:

      mandible -- derived from the condyle facts, which still answer
        fully_included|partially_included|not_included, so "complete" is
        reachable there.
      maxilla  -- schema v6.6 collapsed maxilla_included to included|
        not_included. "included" means the tan maxillary bone is in the
        volume at all; the fully-included state was dropped as unreachable
        (it needs both sinuses closed superiorly, which a dental CBCT does
        not reach). So the maxilla only ever produces "none" or "partial",
        and the renderer's "completely included" sentence is mandible-only
        now.

    THREE levels and not two, because an earlier version collapsed this to
    complete/partial and silently turned "not_included" into "partial" -- the
    renderer reads `level` and nothing else, so an arch the model said was
    outside the volume was reported as "partially included in the scan
    volume", flatly contradicting the prediction it came from.

    An unanswered/unknown inclusion falls back to "partial", the weakest
    claim of the three. A v6.5-or-earlier prediction answering the retired
    "partially_included" lands there too, which is exactly right: it is
    the same claim under the old vocabulary. "fully_included" from such a
    prediction still maps to "complete" rather than being silently
    reinterpreted.
    """
    if included == "fully_included":
        return "complete"
    if included == "not_included":
        return "none"
    return "partial"


def build_mandible_scope(g: Dict) -> Dict:
    """
    DERIVED, not asked. Schema v6.1 dropped mandible_scope: the only mandible
    inclusion facts left are the two condyles, which is enough, because the
    condyles are the part of the mandible a CBCT volume actually cuts off.
    The mandibular BODY is inside the volume of every scan in this dataset --
    it is what the scan is of -- so this never derives "none", which would
    otherwise switch off every bone finding in the arch (see bone_imaged).

      both condyles fully_included  -> complete
      anything else, including no answer at all -> partial

    mandible_included is carried through as the value the level came from, so
    the summary still records WHY, the same way the maxilla's does.
    """
    scopes = [_as_dict(g.get(f)).get("scope")
              for f in ("mandible_condyle_right", "mandible_condyle_left")]
    known = [s for s in scopes if isinstance(s, str) and s.strip()]
    included = ("fully_included" if known and all(s == "fully_included" for s in known)
                else "partially_included")
    return {"level": scope_level(included), "mandible_included": included,
            "scope_source": "derived_from_condyles"}


def build_maxilla_scope(fact: Dict) -> Dict:
    fact = _as_dict(fact)
    included = fact.get("maxilla_included")
    scope = {"level": scope_level(included), "maxilla_included": included}
    # Carried through the same way build_mandible_scope records that its level
    # was derived rather than answered: force_single_arch stamps this fact, and
    # without it the summary cannot tell a model-answered not_included from one
    # the sub-dataset's FOV imposed.
    source = fact.get("scope_source")
    if isinstance(source, str) and source.strip():
        scope["scope_source"] = source.strip()
    return scope


SCOPE_SUBSTRUCTURES = [
    # (arch scope fact, its inclusion field, substructure facts, label for the warning)
    # The mandible has no row: its scope is DERIVED from the condyles rather
    # than answered, so there is no independent claim left to contradict.
    #
    # INERT UNDER SCHEMA v6.6 and kept deliberately. This row only fires on
    # maxilla_included == "fully_included", and v6.6 retired that value -- the
    # sinus condition it used to enforce after the fact ("an arch is not fully
    # included while its sinuses are cut off") is now stated in the question
    # itself as the reason there is no fully-included answer to give. The row
    # stays so that a v6.5-or-earlier prediction replayed against this code
    # still gets the downgrade instead of a fully_included maxilla sailing
    # through unchecked.
    ("maxilla_scope", "maxilla_included",
     ("maxilla_sinus_right", "maxilla_sinus_left"), "sinus"),
]


def enforce_scope_consistency(g: Dict) -> Tuple[Dict, List[str]]:
    """
    Arch inclusion and substructure inclusion are answered as independent
    facts, so the VLM happily claims "mandible fully_included" while both
    condyles came back not_included -- and the sinuses are routinely
    not_included because the label carries no volume, which likewise cannot
    coexist with a fully_included maxilla. The condyles ARE part of the
    mandible and the sinuses ARE part of the maxilla: unless every
    substructure whose scope we actually know is fully_included, the arch
    itself is at best partially_included.

    Only downgrades (fully_included -> partially_included); a
    partially_included or not_included arch is left alone, as is an arch
    whose substructures carry no scope answer at all (nothing to contradict).

    Returns a shallow copy of g with the offending scope facts replaced, plus
    human-readable notes for each downgrade (empty when g was consistent).
    """
    notes: List[str] = []
    out = g

    for arch_field, incl_field, sub_fields, label in SCOPE_SUBSTRUCTURES:
        fact = _as_dict(g.get(arch_field))
        if fact.get(incl_field) != "fully_included":
            continue

        known = []
        for sf in sub_fields:
            scope = _as_dict(g.get(sf)).get("scope")
            if isinstance(scope, str) and scope.strip():
                known.append((sf, scope.strip()))
        if not known or all(s == "fully_included" for _, s in known):
            continue

        if out is g:
            out = dict(g)
        out[arch_field] = {**fact, incl_field: "partially_included"}
        detail = ", ".join(f"{sf.rsplit('_', 1)[-1]} {label} {s}" for sf, s in known)
        notes.append(f"{arch_field}.{incl_field} fully_included -> "
                     f"partially_included ({detail})")

    return out, notes


def build_condyle_entry(scope: Optional[str]) -> Dict:
    """
    Type C: ALWAYS stated -- both sides, every case. Never returns None, so a
    side is never silently dropped from the report.

    INCLUSION ONLY as of schema v6.1: the condyle facts no longer carry a
    morphology field ("Morphology classification is out of scope; inclusion
    only"), so nothing here claims a shape. A missing/blank scope answer
    becomes "unknown" rather than being guessed at.
    """
    scope = scope.strip() if isinstance(scope, str) and scope.strip() else "unknown"
    return {"scope": scope}


# The SUMMARY vocabulary for condyle inclusion is binary. The SCHEMA is not
# touched: predictions keep answering fully_included|partially_included|
# not_included, and the fold happens here, on the way into the summary, exactly
# the way ARCH_VALUE_ALIASES folds the arch survey's prompt words onto the
# tokens this file keys on.
#
# WHY BINARY. The three-way read carries no signal over validate-40 arm 6: its
# per-class precision equals the class prior to three decimals (not_included
# 42/70 = 0.600 against a 0.600 base rate, partially_included 4/10 = 0.400
# against 0.400). And `fully_included` is not a real state in this corpus -- 3
# of 2734 GT sides, never once in validate-40 or training-582, while the model
# emitted it 6 times on training and every one was wrong. So the grading is
# dropped and only the in/out claim is kept.
CONDYLE_SCOPE_BINARY = {"fully_included": "included",
                        "partially_included": "included",
                        "included": "included",
                        "not_included": "not_included"}


def merge_condyle_scopes(right: Optional[str], left: Optional[str]) -> str:
    """
    ONE value for both condyles, folded to included|not_included.

    The two sides are answered off two separate renders (3d_right, 3d_left) but
    they are one statement about one acquisition, and the corpus treats them
    that way: 971 of the 978 reference reports that mention the condyles use no
    laterality word at all, only 3 in 1000 assert a different scope per side,
    and every case-level consensus GT in the corpus (622 cases) is symmetric.
    Against that prior, arm 6 contradicts itself between its two renders in 10
    of 40 validate cases -- noise, not laterality.

    NOT_INCLUDED WINS a disagreement, which is the measured direction:

        any side not_included -> not_included   0.600 validate / 0.591 heldout
        both sides must say so, else included   0.550          / 0.545
        no merge, each side its own answer      0.575          / 0.568

    `unknown` survives only when NEITHER side answered; one silent side takes
    the other's answer rather than voting against it.
    """
    known = [CONDYLE_SCOPE_BINARY.get(s.strip())
             for s in (right, left) if isinstance(s, str) and s.strip()]
    known = [s for s in known if s]
    if not known:
        return "unknown"
    return "not_included" if "not_included" in known else "included"


# ── Alveolar bone atrophy: one presence + one yes/no per arch ─────────────

def build_alveolar_bone_atrophy(fact: Dict) -> Optional[Dict]:
    """
    ONE judgment per arch, and as of v6.4 a BOOL rather than a graded extent:
    the fact now answers present / fully_edentulous / atrophy, where v6.1-6.3
    answered present + extent (none|mild|moderate|severe). Severity is not
    inferred from the bool -- the model was not asked for one, so the renderer
    states atrophy without a grade (see synthesize_report.render_alveolar_bone
    _atrophy). `atrophy_extent` is gone from the schema's _definitions with it.

    `present` is presence of an EDENTULOUS REGION, not of atrophy, so a present
    arch with atrophy False is a real answer ("there is an edentulous area and
    its ridge is intact") and is kept -- the renderer has a sentence for it.
    Returns None when there is no edentulous area at all, or when `atrophy` was
    not answered and there is nothing renderable left.

    fully_edentulous is carried through unchanged. Nothing renders it today,
    but it is the arch-wide qualifier the model was asked for and the summary
    JSON is also read by the report-writing LLM arm, which can use it.
    """
    fact = _as_dict(fact)
    if not fact.get("present"):
        return None
    atrophy = fact.get("atrophy")
    if not isinstance(atrophy, bool):
        return None
    result: Dict = {"atrophy": atrophy}
    if fact.get("fully_edentulous") is True:
        result["fully_edentulous"] = True
    return result


# ── Periodontal bone resorption: extent + pattern (list, both may apply) ──

def build_periodontal_bone_resorption(fact: Dict) -> Optional[Dict]:
    fact = _as_dict(fact)
    extent = fact.get("extent")
    if extent in (None, "none"):
        return None
    result = {"extent": extent}
    pattern = fact.get("pattern")
    if is_meaningful(pattern):
        result["pattern"] = pattern if isinstance(pattern, list) else [pattern]
    return result


# ── Root remnants: the per-tooth is_remnant bool, single source ───────────

def build_root_remnants(teeth: Dict, fdi_list: List[int]) -> Optional[List[Dict]]:
    """
    SINGLE source as of schema v6.1: root_remnant_{arch} is gone, so a
    retained root is whatever tooth_{fdi}_morphology flags on the tooth's own
    composite crop -- there is no second arch-level list left to reconcile it
    against (hence no conflicting_sources here any more). v6.4 renamed that
    flag root_remnant -> is_remnant; nothing else about this changed.

    The accompanying periodontal_bone_resorption is taken from that same
    tooth's periodontal fact, which is the only place bone loss around this
    root is now described.
    """
    out = []
    for fdi in fdi_list:
        t = _as_dict(teeth.get(f"tooth_{fdi}"))
        if not composite_morphology(teeth, fdi).get("is_remnant"):
            continue
        bone_loss = _as_dict(t.get(f"tooth_{fdi}_periodontal_status")).get("bone_loss")
        out.append({"fdi": fdi,
                    "periodontal_bone_resorption": bool(bone_loss and bone_loss != "none")})
    return out or None


# ── Absent teeth carry no findings ────────────────────────────────────────

def toothless_fdis(absent_fdis: List[int], remnants: Optional[List[Dict]],
                   fdi_list: List[int]) -> set:
    """
    FDIs in this arch that have nothing left to image, and so cannot carry a
    per-tooth radiological finding of any kind.

    The VQA facts are answered by independent calls that never see each
    other, so absent_teeth_{arch} routinely contradicts the per-tooth
    questions -- most starkly when an arch is fully edentulous (all 16 FDIs
    absent) yet the per-tooth calls still return caries, endodontic
    treatment and crowns for teeth that are not in the mouth. absent_teeth
    is the authority here: nothing can be restored, root-filled, carious,
    periodontally involved, or adjacent to the canal if it is not there.

    ONE exception, and it is a real clinical entity rather than a
    concession to the model: a RETAINED ROOT REMNANT. Its tooth is
    legitimately reported absent while a root fragment is still in the bone
    and still visible -- so an absent FDI that root_remnants flagged stays
    out of this set and keeps its findings.

    Note what this deliberately does NOT suppress, because each remains
    possible in an edentulous arch: root remnants themselves, impacted
    (unerupted, hence "absent") teeth, implants, bridge spans, alveolar
    bone atrophy, and bone quality.
    """
    remnant_fdis = {r.get("fdi") for r in (remnants or [])}
    arch = set(fdi_list)
    out = set()
    for value in as_fdi_list(absent_fdis):
        if value in arch and value not in remnant_fdis:
            out.add(value)
    return out


def as_fdi_list(values) -> List[int]:
    """Sanitize a list-valued FDI fact ('36' / 36.0 / junk) into real FDI ints."""
    if not isinstance(values, list):
        values = [values] if values is not None else []
    return [f for f in (_as_fdi(v) for v in values) if f is not None]


def note_suppressed(notes: Optional[List[str]], arch: str,
                    toothless: set, fdi_list: List[int]) -> None:
    """Record that an arch's absent teeth are suppressing per-tooth findings."""
    if notes is None or not toothless:
        return
    if len(toothless) == len(fdi_list):
        notes.append(f"{arch} is fully edentulous -- per-tooth findings, "
                     f"restorations, endodontics and canal adjacency suppressed "
                     f"for all {len(fdi_list)} FDIs")
    else:
        notes.append(f"{arch}: per-tooth findings suppressed for absent "
                     f"{', '.join(str(f) for f in sorted(toothless))}")


# ── Impacted teeth: three sources, no arch-level list any more ────────────

# The dedicated 3D-render wisdom-tooth facts and the FDI each one describes.
# Named by SIDE OF THE MOUTH, so "lower_left" is FDI 38 (quadrant III) -- the
# schema's own question text says so, and reading these by the arch/quadrant
# convention instead would mirror every wisdom-tooth finding onto the wrong
# side.
WISDOM_TOOTH_FACTS = {
    "lower_left_wisdom_tooth": 38,
    "lower_right_wisdom_tooth": 48,
    "upper_right_wisdom_tooth": 18,
    "upper_left_wisdom_tooth": 28,
}

# Whether a tooth the arch read called ABSENT may still be reported impacted.
#
# TURNED OFF (2026-08-07), and the last of the removals to go. It was filed as
# a consistency repair rather than a precision gate -- "a tooth that is not in
# the mouth cannot be impacted" -- and kept on that basis when the gates went
# off. Surveying it on outputs/aksssr_v6_validate (34 cases) settled it the
# other way, because it is not behaving like a repair:
#
#     stage                    claims   tp   precision   recall
#     PRED (union of sources)      98   14        0.14     0.58
#     SUMMARY, this rule ON        49    8        0.16     0.33
#     SUMMARY, this rule OFF       98   14        0.14     0.58
#
# It is the ENTIRE difference between 98 and 49 -- the cross-source vote takes
# nothing here, since impaction has been exempt from it for two schema
# versions. Halving the claims moved precision 0.14 -> 0.16 and cost 12 true
# impactions, which is the same empty trade every other gate in this file was
# making.
#
# WHAT WE LOSE, and it is a real cost rather than a rounding error: the report
# can now emit "Absence of 16, 25, and 28." and "Teeth 22 and 28 are impacted."
# in the same paragraph, saying two contradictory things about tooth 28. That
# self-contradiction is what put this rule in. The counter-argument that won is
# that the contradiction is REAL in the data, not manufactured here -- an
# unerupted impacted tooth genuinely reads "absent" to a panoramic call, so the
# two sources are both right about what they saw -- and the ranking metric
# penalises the missing impaction more reliably than it penalises the clash.
# If the contradiction proves expensive, the better fix is in
# synthesize_report.py (exclude impacted teeth from the absence sentence),
# where it costs no recall, rather than deleting the finding here.
#
# The suppression still RUNS and its teeth are still recorded under
# arch_findings.cross_source_dropped.impacted, so the arm stays measurable;
# set this True to restore it. It is deliberately NOT subordinate to
# REQUIRE_CROSS_SOURCE_AGREEMENT -- it is not a cross-source vote, and
# --cross-validate should not silently drag it back in.
DROP_IMPACTION_ON_ABSENT_TEETH = False


def wisdom_facts_for_arch(g: Dict, fdi_list: List[int]) -> Dict[int, Dict]:
    """The wisdom-tooth facts whose FDI belongs to this arch, keyed by FDI."""
    arch = set(fdi_list)
    return {fdi: _as_dict(g.get(field))
            for field, fdi in WISDOM_TOOTH_FACTS.items() if fdi in arch}


def build_impacted_teeth(teeth: Dict, fdi_list: List[int],
                         findings: Dict[int, str], wisdom: Dict[int, Dict],
                         notes: Optional[List[str]] = None,
                         cross_validate: Optional[bool] = None,
                         arch: str = "",
                         absent: Optional[set] = None) -> Tuple[Optional[List[Dict]], List[int]]:
    """
    Impaction, reconciled across the two places the schema still records it --
    there is no impacted_teeth_{arch} fact any more, and as of the v6.3 schema
    no tooth_{fdi}_impaction either:

      1. the four wisdom-tooth 3D-render facts (lower_left_wisdom_tooth etc.),
         an independent read of those four teeth and the only one carrying a
         direction.
      2. dental_arch_findings_{arch}, whose per-tooth enum includes
         "impacted" (wisdom teeth only, by that fact's own definition).

    THE COMPOSITE READ (tooth_{fdi}_impaction) IS GONE and is deliberately not
    consulted even when an older prediction file still carries it. It was
    dropped from the schema as unreliable, which the survey bears out: on
    outputs/aksssr_v5_validate it made 18 impaction claims and got 0 of them
    right (survey_findings.py --category impaction). Reading it from
    legacy predictions would put those 18 known-false claims straight into the
    report, since impaction no longer passes through the agreement gate.

    THE COST is non-wisdom impaction, which now has no dedicated source at
    all: 1 and 2 are both wisdom-only, so an impacted canine or premolar is
    caught only if the panoramic arch read volunteers "impacted" outside the
    third molars. That is a real gap, and a small one -- of the 32 impacted
    teeth in the validate reports exactly one is not a third molar (F006's
    tooth 13), against the 18 false non-wisdom claims the composite read was
    contributing.

    UNIONED, not gated (CROSS_VALIDATE_IMPACTION is off -- see that flag for
    the measurement that settled it). The majority vote below still runs and
    still returns its losers as `dropped`, so arch_findings.cross_source_dropped
    records exactly what a gated run would have removed, but they are reported
    rather than deleted. MAJORITY is now the same rule as UNANIMOUS here, since
    two sources cannot split 2-vs-1; it is left in place as the rule that would
    apply if a third source were ever restored.

    A tooth only one source can speak to is kept, which after the composite
    read's removal is the ordinary case rather than the exception: ABSTAIN
    stays distinct from DENY, so a wisdom tooth is judged by up to two
    opinions and any other tooth by the panoramic read alone.

    `direction` keeps the first non-"none" tipping value seen, in source
    order; the arch read has no direction field, so it can only ever
    contribute "unspecified".

    `absent` (see toothless_fdis) USED TO BE dropped last, after the vote, on
    the rule that a tooth which is not in the mouth cannot be impacted. That
    suppression is now OFF (DROP_IMPACTION_ON_ABSENT_TEETH) -- see that flag
    for the survey that reversed it, and for what the report gives up. The
    absent teeth are still identified and still recorded in `dropped`, so the
    audit trail is unchanged; they are simply no longer removed from `kept`.

    Implants and fixed bridges are deliberately NOT filtered this way (they
    are built elsewhere and never consult `absent`): an implant or a pontic at
    an absent tooth's position is the normal case, not a contradiction --
    those findings describe the SITE, not the tooth.

    Returns (impacted_entries, dropped_fdis).
    """
    arch_fdis = set(fdi_list)
    wisdom_fdis = set(wisdom)
    votes: Dict[int, Dict[str, str]] = {fdi: {} for fdi in fdi_list}
    directions: Dict[int, List[Tuple[str, str]]] = {}

    def record(fdi: Optional[int], direction, source: str) -> None:
        """Cast one source's vote, and remember the tipping value it gave."""
        if fdi is None or fdi not in arch_fdis:
            return
        if direction is None or direction == "":
            votes[fdi][source] = ABSTAIN
            return
        if not is_meaningful(direction):        # "none" -- an actual denial
            votes[fdi][source] = DENY
            return
        votes[fdi][source] = AGREE
        directions.setdefault(fdi, []).append(
            (source, direction if isinstance(direction, str) else "unspecified"))

    # No "composite" source: tooth_{fdi}_impaction is gone from the schema and
    # is not read back off legacy predictions either -- see the docstring.

    # Schema v6.4 split this fact's single "impacted" enum into a bool plus a
    # separate "orientation" (normal|mesial|distal|lingual|buccal, answered for
    # every tooth). So the vote comes from the bool and the tipping value from
    # orientation -- where it used to be one field carrying both. The legacy
    # shape (impacted as a direction string, "none" = not impacted) is still
    # read so older prediction files keep working.
    #
    # "normal" is NOT a denial here, which is the trap: `record` reads any
    # NORMAL_ENUM_VALUES string as DENY, and an impacted tooth may legitimately
    # answer orientation "normal" -- a third molar fully enclosed in bone but
    # sitting upright is impacted by eruption alone. The bool has already
    # settled WHETHER; orientation only says which way, so an upright one
    # contributes the impaction with no direction attached.
    for fdi, fact in wisdom.items():
        flag = fact.get("impacted")
        if isinstance(flag, bool):
            if not flag:
                record(fdi, "none", "3d_render")          # an actual denial
            else:
                direction = fact.get("orientation")
                if not isinstance(direction, str) or not is_meaningful(direction):
                    direction = "unspecified"
                record(fdi, direction, "3d_render")
        else:
            record(fdi, flag, "3d_render")                # legacy enum shape

    # The arch read is ASYMMETRIC here, and deliberately so. It captures
    # impaction for WISDOM TEETH ONLY ("impacted applies to wisdom teeth
    # (38/48) only ... an impacted premolar/canine elsewhere in the arch is
    # not currently captured anywhere"), so:
    #   - an "impacted" value is a positive claim wherever it appears. If the
    #     model volunteers it for a non-wisdom tooth, that is still a finding,
    #     and the old union behaviour kept it;
    #   - but "normal" on a NON-wisdom tooth is silence about impaction, not a
    #     denial of it -- the fact was never asking. Only on a wisdom tooth
    #     may it deny, and so vote against the 3D-render read.
    # With the composite read gone this fact is now the ONLY source that can
    # report a non-wisdom impaction at all, and it does so only when the model
    # volunteers the value unprompted.
    impacted_on_panoramic = arch_findings_teeth(findings, "impacted")
    for fdi in fdi_list:
        if fdi in impacted_on_panoramic:
            # The arch read has no direction field -- it only says "impacted".
            record(fdi, "unspecified", "panoramic")
        elif fdi in wisdom_fdis:
            votes[fdi]["panoramic"] = arch_verdict(findings, fdi, "impacted")

    # Impaction is exempt from the gate (CROSS_VALIDATE_IMPACTION) -- the vote
    # still runs, so `dropped` still records what it would have removed, but
    # the losers are kept. --no-cross-validate can only turn the gate further
    # off, never back on.
    gate = gate_is_on(cross_validate) and CROSS_VALIDATE_IMPACTION
    kept, dropped = reconcile_sources(votes, gate, rule=MAJORITY)
    note_dropped(notes, arch, "impacted", dropped, votes, gate,
                 off_reason="CROSS_VALIDATE_IMPACTION is off")

    # Absence used to outrank impaction -- see the docstring and
    # DROP_IMPACTION_ON_ABSENT_TEETH. Applied here, after the vote, so `votes`
    # still reflects what each source actually claimed either way.
    if absent:
        absent_kept = [fdi for fdi in kept if fdi in absent]
        if absent_kept:
            if DROP_IMPACTION_ON_ABSENT_TEETH:
                kept = [fdi for fdi in kept if fdi not in absent]
                verb = "dropped"
            else:
                verb = "KEPT (DROP_IMPACTION_ON_ABSENT_TEETH is off)"
            dropped = sorted(set(dropped) | set(absent_kept))
            _note(notes, f"{arch}: impaction {verb} for absent "
                         f"{', '.join(str(f) for f in sorted(absent_kept))}")

    for fdi in kept:
        agreeing = [s for s, v in votes[fdi].items() if v == AGREE]
        if len(agreeing) == 1:
            _note(notes, f"tooth {fdi} impaction seen only by the "
                         f"{agreeing[0]} read")

    entries = []
    for fdi in kept:
        seen = directions.get(fdi, [])
        sources = [s for s, _ in seen]
        specific = next((d for _, d in seen if d != "unspecified"), None)
        entries.append({"tooth": fdi,
                        "impacted": specific or "unspecified",
                        "sources": sources})
    return (entries or None), dropped


def build_wisdom_teeth(wisdom: Dict[int, Dict], absent: Optional[set] = None,
                       teeth: Optional[Dict] = None,
                       notes: Optional[List[str]] = None,
                       fallback: bool = False) -> Optional[List[Dict]]:
    """
    The wisdom-tooth eruption state for the teeth that ARE present. Impaction
    is reported via build_impacted_teeth; this carries only what impaction
    doesn't -- an erupted-but-unimpacted third molar has nothing to say here
    either, so only a partially erupted / unerupted one is kept.

    The source is the dedicated 3D-render facts (lower_left_wisdom_tooth etc.).

    `fallback` (OFF by default) additionally accepts that tooth's own composite
    read, tooth_{fdi}_eruption.eruption_state, whenever the 3D fact didn't
    answer. The two share the enum verbatim (_definitions.eruption_state), so
    this is a source swap, not a value mapping.

    It exists for schemas that don't HAVE the four 3D wisdom facts -- the
    deduplicated uniform-32 schema drops them as redundant, and without the
    fallback that arm loses every wisdom-eruption sentence. It is NOT on by
    default because it is not a no-op for the main aksssr arm: measured on
    outputs/aksssr_v4_validate/predictions, 11 of 40 cases have a 3D wisdom
    fact that came back null while the composite read says "partially_erupted",
    so switching it on there silently adds 13 new sentences to reports that
    have already been scored. Turning it on is a per-arm decision, not a
    library default.

    Each fallback use is recorded as a note, so an arch whose wisdom sentences
    came from the unusual source is traceable without re-running anything.
    """
    absent = absent or set()
    teeth = teeth or {}
    out = []
    for fdi in sorted(wisdom):
        state = wisdom[fdi].get("eruption_state")
        if state is None and fallback:
            tooth = _as_dict(teeth.get(f"tooth_{fdi}"))
            state = _as_dict(tooth.get(f"tooth_{fdi}_eruption")).get("eruption_state")
            if state is not None:
                _note(notes, f"tooth {fdi} eruption state taken from the "
                             f"per-tooth composite read (no arch-level "
                             f"wisdom-tooth fact answered)")
        if fdi in absent or state in (None, "absent", "fully_erupted"):
            continue
        out.append({"tooth": fdi, "eruption_state": state})
    return out or None


# ── Prosthetics: already arch-scoped, no FDI-range filtering needed ───────

def _as_fdi(value) -> Optional[int]:
    """int / '36' / 36.0 -> 36, if it is a real FDI code; otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    return value if isinstance(value, int) and value in ALL_FDIS else None


def build_implants(fact: Dict, notes: Optional[List[str]] = None,
                   arch: str = "", fdi_list: Optional[List[int]] = None
                   ) -> Optional[List[Dict]]:
    """
    implants_{arch} is an OBJECT in v6.1 ({visual_evidence, two quadrant
    counts, implants[]}), not the bare list it used to be, and each member
    carries a singular `fdi_number` rather than an `fdi_numbers` list.

    An implant with no position is not reportable -- the sentence names it --
    so an entry is kept only if it has a usable fdi_number or a location
    string. The schema requires len(implants) == the two quadrant counts; a
    mismatch is recorded as a note (the LIST is what gets reported either
    way, since it is the only source with positions in it).

    QUADRANT LEAK, same failure and same rule as arch_findings_map. Both arch
    implant calls see the SAME full-mouth panoramic and are scoped to one arch
    by the prompt alone, so the model answers with the other arch's teeth --
    on the v5 validate run, S0000's MAXILLA call placed an implant at FDI 48.
    An FDI names its arch unambiguously, so `fdi_list` (this arch's own range)
    decides and a stray entry is dropped outright rather than re-homed: the
    opposite arch has its own read of the same picture, and an implant filed
    under the wrong arch is evidence the model lost track of which half it was
    looking at, not a finding to salvage. An entry with no fdi_number is not
    filtered -- prose location alone names no arch to be wrong about.
    """
    fact = _as_dict(fact)
    raw = fact.get("implants")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    arch_fdis = set(fdi_list) if fdi_list else None
    out, stray = [], []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fdi = _as_fdi(item.get("fdi_number"))
        location = item.get("location")
        if fdi is None and not is_meaningful(location):
            continue
        if fdi is not None and arch_fdis is not None and fdi not in arch_fdis:
            stray.append(fdi)
            continue
        entry = {k: v for k, v in item.items() if k != "visual_evidence"}
        if fdi is not None:
            entry["fdi_number"] = fdi
        out.append(entry)

    if stray:
        _note(notes, f"{arch}: implants_{arch} placed implant(s) at "
                     f"{sorted(stray)} -- teeth of the OTHER arch, anatomically "
                     f"impossible for this fact; dropped")

    counts = [fact.get(k) for k in ("quadrant_1_implant_count", "quadrant_2_implant_count",
                                    "quadrant_3_implant_count", "quadrant_4_implant_count")]
    declared = sum(c for c in counts if isinstance(c, int) and not isinstance(c, bool))
    if declared and declared != len(out):
        _note(notes, f"{arch}: implant counts sum to {declared} but "
                     f"{len(out)} implant(s) were described")

    return out or None


# An implant and a natural tooth cannot occupy the same socket, so an implant
# claimed at an FDI whose tooth is still there is a logic error, not a finding.
# It is a real one: on A008 implants_mandible put an implant at 45 while the
# arch read called 45 "normal", the composite called it fully_erupted, and the
# segmentation carried a tooth label there. The model's own visual_evidence
# even argues itself out of it -- "No, 45 is present. There is a metal
# screw-like object ... between tooth 45 and 46" -- and then files it under 45
# regardless, because the fact demands one FDI per implant.
#
# THE IMPLANT LOSES. Presence has three sources, and the implant claim has one
# panoramic read; more importantly the strongest presence source is not a model
# read at all. This mirrors the policy already in run_vqa_inference, where a
# generated composite outranks a panoramic call of absence.
#
# MEASURED, and kept ON -- unlike the cross-source gates above, this one pays
# for itself. Across outputs/aksssr_v5_validate it drops 9 implants, and
# checking each against the reports (survey_findings.py):
#   correctly dropped (6) -- A008/45, A019/46, A019/25, A041/13, A041/21,
#     F041/26: no report places an implant at any of these.
#   wrongly dropped (3)  -- A037/17, S0000/42, S0000/44: the reports do state
#     these, and both S0000 entries are implants under a fixed bridge, where
#     the "tooth is present" evidence is most likely the model reading the
#     bridge's crowns as teeth.
# Net on the summary: precision 0.34 -> 0.41, recall 0.31 -> 0.22. It trades
# 3 true implants for 6 false ones, which is a real trade rather than the
# empty one the endodontic and filling gates were making, so it stays.
DROP_IMPLANTS_ON_PRESENT_TEETH = True


def present_tooth_evidence(teeth: Dict, findings: Dict[int, str], fdi: int) -> Optional[str]:
    """
    Why FDI `fdi` is believed to hold a natural tooth, or None if nothing says
    so. Strength order matches collect_arch_absent's:

      1. detected == "yes" -- create_tooth_detail.py built a composite for this
         FDI, which it only does when the MASK carries a tooth label there.
         Implants are a separate mask label, so this is segmentation evidence
         of a natural tooth and is independent of anything the model said.
      2. eruption_state -- the tooth's own composite crop was read as erupted.
      3. the arch read gave it a real finding (not "absent", not out of enum).

    Returns a short source name for the audit note, so a dropped implant can be
    traced to what contradicted it.
    """
    tooth = _as_dict(teeth.get(f"tooth_{fdi}"))
    if tooth.get("detected") == "yes":
        return "segmented_tooth"
    if _as_dict(tooth.get(f"tooth_{fdi}_eruption")).get("eruption_state") in PRESENT_ERUPTION_STATES:
        return "composite_erupted"
    value = findings.get(fdi)
    if value is not None and value not in (ARCH_ABSENT_VALUE, "unreadable"):
        return f"panoramic={value}"
    return None


def gate_implants(implants: Optional[List[Dict]], teeth: Dict,
                  findings: Dict[int, str], cross_validate: Optional[bool] = None,
                  arch: str = "", notes: Optional[List[str]] = None
                  ) -> Tuple[Optional[List[Dict]], List[int]]:
    """
    Drop implants claimed at positions where the tooth is still present.

    Only entries carrying a real fdi_number are checked -- an implant reported
    by location prose alone has no position to contradict, and is left alone
    rather than guessed at.

    Returns (kept implants or None, dropped FDIs). The dropped list is recorded
    for audit whether or not the gate is on, same convention as the
    cross-source gates.
    """
    if not implants:
        return implants, []

    kept, dropped, why = [], [], {}
    for entry in implants:
        fdi = entry.get("fdi_number")
        evidence = (present_tooth_evidence(teeth, findings, fdi)
                    if isinstance(fdi, int) else None)
        if evidence is None:
            kept.append(entry)
            continue
        dropped.append(fdi)
        why[fdi] = evidence
        if not (DROP_IMPLANTS_ON_PRESENT_TEETH and gate_is_on(cross_validate)):
            kept.append(entry)

    if dropped:
        verb = ("dropped" if DROP_IMPLANTS_ON_PRESENT_TEETH and gate_is_on(cross_validate)
                else "KEPT (implant gate is off)")
        detail = "; ".join(f"{fdi} ({why[fdi]})" for fdi in dropped)
        _note(notes, f"{arch}: implant {verb} at position(s) whose tooth is "
                     f"still present -- {detail}")

    return (kept or None), sorted(dropped)


def build_bridges(fact: Dict) -> Optional[List[Dict]]:
    """
    fixed_bridges_{arch} is an OBJECT in v6.1 ({visual_evidence, present,
    bridges[]}), not a bare list.

    A bridge sentence is built around "from {span_start} to {span_end}", so an
    entry whose span is missing or is not a pair of real FDI codes (the VLM
    sometimes answers this fact with prose, e.g. span='Teeth 31-43 missing')
    is dropped rather than rendered with a hole in it.
    """
    fact = _as_dict(fact)
    if not fact.get("present"):
        return None
    raw = fact.get("bridges")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out = []
    for b in raw:
        if not isinstance(b, dict):
            continue
        start, end = _as_fdi(b.get("span_start")), _as_fdi(b.get("span_end"))
        if start is None or end is None:
            continue
        out.append({**{k: v for k, v in b.items() if k != "visual_evidence"},
                    "span_start": start, "span_end": end})
    return out or None


# ── Bone quality: closes each arch's section independently, no cross-arch
#    merge at all (per this round's explicit instruction) ─────────────────

BONE_QUALITY_TYPES = ("radiopaque", "radiolucent")

# Whether an arch-level bone_quality claim must be corroborated by at least one
# tooth_{fdi}_bone_quality read before it may be reported.
#
# TURNED OFF (2026-08-07), with the rest of the precision gates -- see
# REQUIRE_CROSS_SOURCE_AGREEMENT, to which this is subordinate, so
# --cross-validate restores it along with everything else. It is the same
# cross-source shape as the gates there (an arch read checked against the
# per-tooth reads) and was never separately surveyed, so it goes off on the
# general finding rather than on evidence of its own.
#
# The value below stays True to record what the schema's own prompt text still
# promises the model ("postprocessing reports that arch-level lesion only for
# teeth this per-tooth read confirms"); the master switch is what decides
# whether that promise is kept. If the gate stays off for good, that sentence
# in schema/schema.json should be rewritten to match.
REQUIRE_BONE_QUALITY_TOOTH_CONFIRMATION = True


def per_tooth_bone_quality_confirms(teeth: Dict, fdi_list: List[int]) -> bool:
    """
    Does ANY tooth in this arch confirm a bone lesion on its own composite crop?

    tooth_{fdi}_bone_quality is the arch fact's second source, and the schema
    promises the model this gate in that fact's own text: "This is the SECOND
    source for bone_quality_{arch}: postprocessing reports that arch-level
    lesion only for teeth this per-tooth read confirms, so a false positive
    here manufactures a lesion in the report and a false negative merely
    withholds one." That promise was never implemented -- the arch claim went
    into the report unchallenged -- so this is the gate catching up with what
    the prompt already says.

    Deliberately a WHOLE-ARCH confirmation, not a per-tooth match: the arch
    fact reports one lesion for the arch and locates it in free text, not by
    FDI, so there is no tooth to line up the per-tooth reads against. A single
    confirming tooth anywhere in the arch is enough to let the arch claim
    through; none at all silences it.
    """
    for fdi in fdi_list:
        tooth = _as_dict(teeth.get(f"tooth_{fdi}"))
        if _as_dict(tooth.get(f"tooth_{fdi}_bone_quality")).get("present") is True:
            return True
    return False


def build_bone_quality(fact: Dict, teeth: Optional[Dict] = None,
                       fdi_list: Optional[List[int]] = None,
                       notes: Optional[List[str]] = None,
                       arch: str = "",
                       cross_validate: Optional[bool] = None) -> Dict:
    """
    v6.1 collapsed this fact to ONE most-notable finding per arch
    ({present, type, location}), replacing the old radiopaque_*/radiolucent_*
    pairs of booleans and free-text strings.

    `location` is a short FDI-anchored phrase, not the old prose findings
    field, so the sentence is TEMPLATED by the renderer now rather than the
    schema's own text being emitted verbatim.

    "present" here means THE MODEL CLAIMED A LESION, which is not the same as
    "we have a reportable finding" -- a claim whose type is out of enum keeps
    present=True with no findings. The two must stay distinct because the
    renderer's negative statement ("No definite osteolytic or osteocondensing
    lesions.") may only be asserted when the model actually saw nothing.
    Collapsing an unusable positive claim into present=False, as this used to,
    turned a bad answer into a confident denial. See render_bone_quality.

    An arch claim no per-tooth read confirms WOULD be silenced the same careful
    way -- findings emptied, present left True, so the report says nothing
    rather than denying what the model claimed. That gate is currently OFF
    (REQUIRE_BONE_QUALITY_TOOTH_CONFIRMATION, subordinate to
    REQUIRE_CROSS_SOURCE_AGREEMENT): the claim is reported and the unconfirmed
    ones are noted instead. Pass teeth + fdi_list to run the check at all;
    without them the fact is taken at face value, which is what a caller with
    no per-tooth reads should do.
    """
    fact = _as_dict(fact)
    if not fact.get("present"):
        return {"present": False, "findings": []}
    lesion_type = fact.get("type")
    if lesion_type not in BONE_QUALITY_TYPES:
        return {"present": True, "findings": []}
    if teeth is not None and fdi_list is not None and \
            not per_tooth_bone_quality_confirms(teeth, fdi_list):
        gate = gate_is_on(cross_validate) and REQUIRE_BONE_QUALITY_TOOTH_CONFIRMATION
        verb = "dropped" if gate else "KEPT (bone-quality confirmation gate is off)"
        _note(notes, f"{arch}: bone_quality claim ({lesion_type}) {verb} -- no "
                     f"tooth_{{fdi}}_bone_quality read in this arch confirms it")
        if gate:
            return {"present": True, "findings": []}
    finding = {"type": lesion_type}
    if is_meaningful(fact.get("location")):
        finding["location"] = fact["location"].strip()
    return {"present": True, "findings": [finding]}


# ── Root canal treatment / restorations: single source now, plain grouping ─

def build_endo_summary(findings: Dict[int, str], teeth: Dict, fdi_list: List[int],
                       absent: Optional[set] = None,
                       cross_validate: Optional[bool] = None,
                       arch: str = "", notes: Optional[List[str]] = None
                       ) -> Tuple[Optional[Dict], List[int]]:
    """
    Two independent sources for the same fact, cross-validated:
      - dental_arch_findings_{arch}: every tooth it read as
        "root_canal_treatment" on the panoramic (v6.1's replacement for the
        old whole-mouth root_canal_treatment list, which no longer exists);
      - each tooth's own tooth_{fdi}_morphology.with_endo, read from that
        tooth's composite crop.

    with_endo is where the composite's root-filling claim lives as of v6.4.
    tooth_{fdi}_endodontic_treatment lost its root_canal_treatment bool and now
    only CHARACTERIZES a filling whose existence with_endo already established
    -- the schema says so in that fact's own question text ("Only answer this
    fact when tooth_{fdi}_morphology.with_endo == true ... presence of root
    canal treatment is established by that gate, not re-asked here"). So the
    vote is cast from the gate flag, and filling_quality/periapical_lesion are
    still read from the endodontic fact below.

    A tooth is reported when at least one source claims the root filling and
    -- with REQUIRE_CROSS_SOURCE_AGREEMENT on -- the other does not deny it.
    A tooth both reads answered and disagreed about is DROPPED, not reported
    with a conflict annotation as it was before the gate; see the gate's own
    comment for the measured effect and the reasoning.

    The arch read being single-finding-per-tooth is handled by arch_verdict,
    not counted as a disagreement: a root-filled tooth that also carries a
    post is filed under post_and_core on the panoramic, which ABSTAINs here
    rather than denying the root filling. Before the gate this showed up as a
    spurious entry in "conflicting_sources".

    `absent` (see toothless_fdis) is dropped from BOTH sources before voting
    -- a tooth that is not in the mouth has no canal to fill, so a claim from
    either call is noise, not a finding to reconcile.

    filling_quality follows the tooth it describes: a root filling the gate
    removed takes its quality grading with it. periapical_lesion does NOT --
    it has only the one source, is a lesion rather than a treatment, and does
    not depend on the root filling being real.

    Returns (summary, dropped_fdis) -- the dropped list is for the arch-level
    audit block and for suppressing the same teeth in build_tooth_findings.
    """
    absent = absent or set()
    votes: Dict[int, Dict[str, str]] = {}
    quality_by_fdi = {}
    periapical = []
    for fdi in fdi_list:
        if fdi in absent:
            continue
        t = _as_dict(teeth.get(f"tooth_{fdi}"))
        endo = _as_dict(t.get(f"tooth_{fdi}_endodontic_treatment"))
        votes[fdi] = {
            "composite": bool_verdict(composite_morphology(teeth, fdi).get("with_endo")),
            "panoramic": arch_verdict(findings, fdi, "root_canal_treatment"),
        }
        fq = endo.get("filling_quality")
        if fq and fq != "none":
            quality_by_fdi[fdi] = fq
        if endo.get("periapical_lesion"):
            periapical.append(fdi)

    # Exempt from the gate (CROSS_VALIDATE_ENDODONTIC), like impaction: the
    # vote still runs so `dropped` still records what it would have removed,
    # but the losers are kept.
    gate = gate_is_on(cross_validate) and CROSS_VALIDATE_ENDODONTIC
    kept, dropped = reconcile_sources(votes, gate)
    note_dropped(notes, arch, "root_canal_treatment", dropped, votes, gate,
                 off_reason="CROSS_VALIDATE_ENDODONTIC is off")

    if not kept and not periapical:
        return None, dropped

    kept_set = set(kept)
    quality_groups: Dict[str, List[int]] = {}
    for fdi, q in quality_by_fdi.items():
        if fdi not in kept_set:
            continue
        quality_groups.setdefault(q, []).append(fdi)
    for q in quality_groups:
        quality_groups[q].sort()

    result: Dict = {}
    if kept:
        result["teeth"] = kept
    if quality_groups:
        result["quality_groups"] = quality_groups
    if periapical:
        result["periapical_teeth"] = sorted(periapical)
    if dropped:
        result["conflicting_sources"] = dropped
    return result, dropped


def build_restoration_summary(findings: Dict[int, str], teeth: Dict,
                              fdi_list: List[int],
                              absent: Optional[set] = None,
                              cross_validate: Optional[bool] = None,
                              arch: str = "", notes: Optional[List[str]] = None
                              ) -> Tuple[Optional[Dict], List[int]]:
    """
    Two sources per restoration type, cross-validated:
      - dental_arch_findings_{arch}'s per-tooth value, mapped through
        ARCH_TO_RESTORATION ("filling" -> "fillings"). This replaces the old
        full_crowns / dental_fillings / post_and_core_crowns whole-mouth
        lists, which v6.1 does not have;
      - the composite read's own restorative answer. tooth_{fdi}_restoration is
        gone as of v6.4; its enum now lives as three independent bools on
        tooth_{fdi}_morphology, collapsed back to one value per tooth by
        composite_restoration_type (see it for the priority and for why a
        never-answered block stays ABSTAIN instead of denying everything).

    With REQUIRE_CROSS_SOURCE_AGREEMENT on, a tooth one source restores and
    the other explicitly calls "none" (or calls a different restoration type)
    is DROPPED rather than reported with a conflict annotation.

    CROWNS ARE COMPARED AGAINST NOTHING, and so are never gated. The arch read
    has no "crown" value at all -- the schema drops it deliberately ("too
    easily confused with a large filling or dense enamel at panoramic
    resolution") and says so: skip the comparison, not a mismatch, for any
    tooth reading "crown" on the composite side. Crowns are reported from the
    composite read alone.

    `absent` (see toothless_fdis) is dropped from every source: there is no
    crown to restore on a tooth that is not in the mouth. An implant or a
    bridge pontic at that position is still reported -- by build_implants /
    build_bridges, which are not filtered here.

    Returns (summary, dropped_fdis) -- dropped_fdis is the union across
    restoration types, for the arch audit block and for suppressing the same
    teeth in build_tooth_findings.
    """
    absent = absent or set()
    per_tooth: Dict[int, Optional[str]] = {}
    for fdi in fdi_list:
        if fdi in absent:
            continue
        per_tooth[fdi] = composite_restoration_type(composite_morphology(teeth, fdi))

    crowned = {fdi for fdi, r in per_tooth.items() if r == "crown"}

    groups: Dict[str, List[int]] = {}
    conflicts: Dict[str, List[int]] = {}
    all_dropped: set = set()

    # Crown: composite-only, no second opinion to gate against.
    if crowned:
        groups["crown"] = sorted(crowned)

    # post_and_core before fillings: the crown/filling resolution below reads
    # more naturally in that order, and the groups dict is keyed, not ordered.
    for arch_value, rtype in sorted(ARCH_TO_RESTORATION.items(), reverse=True):
        votes: Dict[int, Dict[str, str]] = {}
        for fdi, r in per_tooth.items():
            # A crowned tooth read from the panoramic as "filling" is the
            # schema's own named failure mode, and the arch read has no crown
            # value to answer with instead. The composite's crown wins
            # outright -- not a disagreement, and not a filling either, so the
            # tooth is left out of the fillings vote entirely. Without this
            # the same tooth is reported twice, once crowned and once filled.
            # post_and_core is NOT resolved that way: a post under a crown is
            # a real finding the arch read can genuinely see down the canal,
            # so there the crown only makes the composite ABSTAIN.
            if rtype == "fillings" and fdi in crowned:
                continue
            votes[fdi] = {
                "composite": enum_verdict(r, rtype, skip=("crown",)),
                "panoramic": arch_verdict(findings, fdi, arch_value),
            }
        # Per restoration type -- fillings are exempt, post_and_core and
        # crown are not (CROSS_VALIDATE_RESTORATIONS). The vote runs either
        # way so `conflicts` still records what it would have removed.
        gate = gate_is_on(cross_validate) and CROSS_VALIDATE_RESTORATIONS.get(rtype, True)
        kept, dropped = reconcile_sources(votes, gate)
        note_dropped(notes, arch, rtype, dropped, votes, gate,
                     off_reason=f"CROSS_VALIDATE_RESTORATIONS[{rtype}] is off")
        if kept:
            groups[rtype] = kept
        if dropped:
            conflicts[rtype] = dropped
            all_dropped.update(dropped)

    # "A tooth with a post belongs under post_and_core even if a crown also
    # sits on top -- never double-count as crown" (schema _definitions).
    if groups.get("crown") and groups.get("post_and_core"):
        remaining = [f for f in groups["crown"] if f not in set(groups["post_and_core"])]
        if remaining:
            groups["crown"] = remaining
        else:
            del groups["crown"]

    if not groups:
        return None, sorted(all_dropped)
    result: Dict = {"groups": groups}
    if conflicts:
        result["conflicting_sources"] = conflicts
    return result, sorted(all_dropped)


# ── Per-tooth findings (Type A) ───────────────────────────────────────────

def build_tooth_findings(teeth: Dict, fdi_list: List[int],
                         absent: Optional[set] = None,
                         gated_endo: Optional[set] = None,
                         gated_restoration: Optional[set] = None,
                         gated_caries: Optional[set] = None) -> List[Dict]:
    """
    Abnormal-only per-tooth findings. A tooth flagged root_remnant==true
    defers entirely to the arch-level root_remnants summary (built above)
    -- per the decision tree, it is NOT also given its own morphology
    sentence here, to avoid double-reporting the same finding.

    Teeth in `absent` (see toothless_fdis) are skipped outright: caries,
    fractures, periodontal bone loss and periapical lesions all need a tooth
    to be attached to.

    `gated_endo` / `gated_restoration` carry the teeth the cross-source
    agreement gate removed from endodontic_summary / restoration_summary.
    They are suppressed here too. This block reads the SAME composite fields
    those summaries do, so without it a gated finding would survive by a
    second route: the template renderer takes only caries from here, but the
    summary JSON is also handed to a report-writing LLM, which sees
    everything. periapical_lesion is not suppressed -- it is a single-source
    lesion, not the treatment claim that was gated.
    """
    absent = absent or set()
    gated_endo = gated_endo or set()
    gated_restoration = gated_restoration or set()
    gated_caries = gated_caries or set()
    out = []
    for fdi in fdi_list:
        if fdi in absent:
            continue
        t = _as_dict(teeth.get(f"tooth_{fdi}"))
        if t.get("detected") in ("no", "no_image"):
            continue

        entry: Dict = {}

        # v6.4: two bools where v6.3 had two enums (crown_morphology /
        # root_morphology). The summary key names follow the bools, so a
        # consumer reads morphology.caries rather than testing an enum value
        # against the string "carious"; synthesize_report.render_morphology
        # _findings is the one that matters, and caries is still all it renders.
        morphology = composite_morphology(teeth, fdi)
        if not morphology.get("is_remnant"):
            caries = morphology.get("with_caries") is True
            fracture = morphology.get("with_root_fracture") is True
            # A caries claim the panoramic contradicted is dropped, but the
            # root reading beside it is a different finding from a different
            # part of the tooth and keeps its entry.
            if caries and fdi in gated_caries:
                caries = False
            if caries or fracture:
                entry["morphology"] = {"caries": caries, "root_fracture": fracture}

        endo = _as_dict(t.get(f"tooth_{fdi}_endodontic_treatment"))
        # The root filling's existence comes from the morphology gate flag now,
        # not from the endodontic fact -- see build_endo_summary.
        rct = morphology.get("with_endo")
        fq, peri = endo.get("filling_quality"), endo.get("periapical_lesion")
        if fdi in gated_endo:
            rct, fq = None, None        # the root filling did not survive the gate
        if rct or (fq and fq != "none") or peri:
            e = {}
            if rct is not None:
                e["root_canal_treatment"] = bool(rct)
            if fq and fq != "none":
                e["filling_quality"] = fq
            if peri:
                e["periapical_lesion"] = True
            entry["endodontic"] = e

        # The graded bone_loss enum and the furcation bool are the whole of
        # this fact now: its free-text "findings" field is gone from the
        # schema. That field used to WIN over the enum when present, so a
        # tooth's periodontal entry could carry a sentence of model prose
        # instead of a value anything downstream could group or compare --
        # and visual_evidence already says the same thing, for auditing.
        perio = _as_dict(t.get(f"tooth_{fdi}_periodontal_status"))
        bone_loss, furcation = perio.get("bone_loss"), perio.get("furcation_involvement")
        if (bone_loss and bone_loss != "none") or furcation:
            p = {"bone_loss": bone_loss}
            if furcation:
                p["furcation_involvement"] = True
            entry["periodontal"] = p

        # v6.4: the three restorative bools on the morphology fact, collapsed
        # to the one value per tooth this entry has always carried. The same
        # collapse the restoration summary votes on, so the two cannot drift.
        restoration = composite_restoration_type(morphology)
        if (isinstance(restoration, str) and restoration and restoration != "none"
                and fdi not in gated_restoration):
            entry["restoration"] = restoration

        if entry:
            entry["fdi"] = fdi
            out.append(entry)

    return out


# ── Dentition type: DERIVED, not asked -- see the mixed-dentition rule ────

def primary_teeth_of(fact) -> List[int]:
    """
    primary_teeth_{arch} is an OBJECT in v6.1 ({visual_evidence,
    primary_teeth}), where v4 had a bare list[int]. Both shapes are accepted:
    ground-truth files written before the shape change reach this function
    still in the old shape, and reading the object as a list produced
    an empty primary dentition (and so "permanent" for every mixed-dentition
    child).
    """
    if isinstance(fact, dict):
        fact = fact.get("primary_teeth")
    return as_fdi_list(fact)


def build_dentition_type(primary_mandible, primary_maxilla,
                         teeth: Dict) -> Dict:
    """
    dentition_type is no longer a VQA fact. Derived per the rule agreed
    this session:
      primary_tooth_count > 2 AND unerupted_permanent_count > 2 -> mixed
      primary_tooth_count > 0 AND permanent_present_count == 0  -> primary
      primary_tooth_count == 0 AND permanent_present_count == 0 -> edentulous
      default -> permanent

    unerupted_permanent_count counts 'not_erupted'/'partially_erupted'
    ONLY (not 'absent' -- a truly absent/undeveloped tooth is not evidence
    of active eruption). Wisdom teeth (18/28/38/48) are excluded, since
    their eruption timing is not evidence of mixed dentition the way a
    non-wisdom permanent tooth's is.
    """
    primary_count = len(primary_teeth_of(primary_mandible)) + len(primary_teeth_of(primary_maxilla))

    non_wisdom_permanent_fdis = [f for f in ALL_FDIS
                                 if f not in MANDIBLE_WISDOM_FDIS | MAXILLA_WISDOM_FDIS]
    unerupted_count, present_count = 0, 0
    for fdi in non_wisdom_permanent_fdis:
        t = _as_dict(teeth.get(f"tooth_{fdi}"))
        eruption = _as_dict(t.get(f"tooth_{fdi}_eruption")).get("eruption_state")
        if eruption in ("not_erupted", "partially_erupted"):
            unerupted_count += 1
            present_count += 1
        elif eruption == "fully_erupted":
            present_count += 1
        # 'absent' or missing data: counts toward neither

    if primary_count == 0 and present_count == 0:
        return {"dentition_type": "edentulous"}
    if primary_count > 0 and present_count == 0:
        return {"dentition_type": "primary"}
    if primary_count > 2 and unerupted_count > 2:
        return {"dentition_type": "mixed"}
    return {"dentition_type": "permanent"}


# ── Arch section assembly ─────────────────────────────────────────────────

def record_cross_source_dropped(result: Dict, endo_dropped: List[int],
                                restoration_dropped: List[int],
                                impacted_dropped: List[int],
                                caries_dropped: Optional[List[int]] = None) -> None:
    """
    Carried for auditing, alongside uncertain_demoted: which teeth the
    cross-source agreement gate acted on (or would have, with the flag off).
    Nothing renders it -- it exists so a missing finding can be traced to the
    gate rather than to the model, and it is written even when the gate
    emptied a summary block out of existence.
    """
    dropped = {k: v for k, v in (("root_canal_treatment", endo_dropped),
                                 ("restorations", restoration_dropped),
                                 ("impacted", impacted_dropped),
                                 ("caries", caries_dropped or [])) if v}
    if dropped:
        result["arch_findings"]["cross_source_dropped"] = dropped


def build_mandible_section(g: Dict, teeth: Dict, notes: Optional[List[str]] = None,
                           wisdom_eruption_fallback: bool = False,
                           cross_validate: Optional[bool] = None) -> Dict:
    """
    Key insertion order matches the confirmed report order:
    Scope -> Atrophy -> Canals -> Dental Arch -> Tooth Findings ->
    Prosthetics -> bone_quality_mandible -> Condyles.
    This isn't just cosmetic -- if the report renderer iterates this dict
    in insertion order, the order genuinely matters, not just for reading
    the JSON by eye.
    """
    fdi_list = fdis_in_range(31, 48)
    findings = arch_findings_map(g.get("dental_arch_findings_mandible"), fdi_list,
                                 "mandible", notes)
    findings, uncertain_demoted = demote_uncertain_findings(
        findings, g.get("dental_arch_findings_mandible"), fdi_list, "mandible", notes)
    wisdom = wisdom_facts_for_arch(g, fdi_list)
    absent, absent_known = collect_arch_absent(teeth, fdi_list, "mandible",
                                               findings, notes)

    # Root remnants are built FIRST (though written into `result` in report
    # order, further down) because they are the one exception to the
    # absent-teeth suppression -- see toothless_fdis.
    remnants = build_root_remnants(teeth, fdi_list)
    toothless = toothless_fdis(absent, remnants, fdi_list)
    note_suppressed(notes, "mandible", toothless, fdi_list)

    scope = build_mandible_scope(g)
    bone_imaged = scope["level"] != "none"
    result: Dict = {"scope": scope}

    # Every BONE fact below is gated on bone_imaged -- see build_maxilla_section
    # for the reasoning. The teeth are never gated: an arch can be out of the
    # volume while its teeth are still visible.
    if bone_imaged:
        atrophy = build_alveolar_bone_atrophy(g.get("alveolar_bone_atrophy_mandible", {}))
        if atrophy:
            result["alveolar_bone_atrophy"] = atrophy

        # NO merge -- always two independent entries (explicit instruction this session)
        # `teeth` is the second reader: tooth_{fdi}_mandible_canal on the six
        # lower molars (schema v6.9).
        result["canals"] = build_canals(g.get("mandible_canal_right", {}),
                                        g.get("mandible_canal_left", {}), toothless,
                                        teeth, cross_validate, notes)

        periodontal = build_periodontal_bone_resorption(g.get("periodontal_bone_resorption_mandible", {}))
        if periodontal:
            result["periodontal_bone_resorption"] = periodontal

    absent_pattern = classify_absent_teeth_pattern(absent, MANDIBLE_QUADRANTS, absent_known)
    primary = primary_teeth_of(g.get("primary_teeth_mandible"))
    result["arch_findings"] = {
        "absent_teeth": absent_pattern,
        "primary_teeth": sorted(primary) if primary else None,
    }
    # Carried for auditing: which teeth the uncertainty gate acted on (or
    # would have, with the flag off). Nothing renders it -- it exists so a
    # missing finding can be traced to the gate rather than to the model.
    if uncertain_demoted:
        result["arch_findings"]["uncertain_demoted"] = uncertain_demoted

    if remnants:
        result["root_remnants"] = remnants

    impacted, impacted_dropped = build_impacted_teeth(
        teeth, fdi_list, findings, wisdom, notes, cross_validate, "mandible",
        # The FULL absent set, not `toothless` -- toothless_fdis exempts
        # retained root remnants so their findings survive (endodontic filling
        # material in a retained root is real). That exemption must not extend
        # to impaction: a root fragment cannot be an impacted tooth. A022 hit
        # exactly this -- 24 reported absent AND a remnant AND "impacted".
        absent=set(as_fdi_list(absent)) & set(fdi_list))
    if impacted:
        result["impacted_teeth"] = impacted

    wisdom_teeth = build_wisdom_teeth(wisdom, toothless, teeth, notes,
                                      fallback=wisdom_eruption_fallback)
    if wisdom_teeth:
        result["wisdom_teeth"] = wisdom_teeth

    # The arch-level source is dental_arch_findings_mandible's per-tooth map
    # (v6.1's replacement for the old whole-mouth root_canal_treatment /
    # full_crowns / dental_fillings / post_and_core_crowns lists), already
    # scoped to this arch by arch_findings_map.
    endo, endo_dropped = build_endo_summary(findings, teeth, fdi_list, toothless,
                                            cross_validate, "mandible", notes)
    if endo:
        result["endodontic_summary"] = endo

    restorations, restoration_dropped = build_restoration_summary(
        findings, teeth, fdi_list, toothless, cross_validate, "mandible", notes)
    if restorations:
        result["restoration_summary"] = restorations

    caries_dropped = gate_caries(findings, teeth, fdi_list, toothless,
                                 cross_validate, "mandible", notes)

    record_cross_source_dropped(result, endo_dropped, restoration_dropped,
                                impacted_dropped, caries_dropped)

    # Only suppress when the gate actually removed them: with it off, the
    # dropped lists are recorded for audit but nothing is taken out anywhere.
    suppress = gate_is_on(cross_validate)
    tooth_findings = build_tooth_findings(
        teeth, fdi_list, toothless,
        set(endo_dropped) if suppress else None,
        set(restoration_dropped) if suppress else None,
        set(caries_dropped) if suppress else None)
    if tooth_findings:
        result["tooth_findings"] = tooth_findings

    prosthetics = {}
    implants = build_implants(g.get("implants_mandible"), notes, "mandible", fdi_list)
    implants, implants_dropped = gate_implants(implants, teeth, findings,
                                               cross_validate, "mandible", notes)
    if implants_dropped:
        result["arch_findings"].setdefault("cross_source_dropped", {})[
            "implants_on_present_teeth"] = implants_dropped
    if implants:
        prosthetics["implants"] = implants
    bridges = build_bridges(g.get("fixed_bridges_mandible"))
    if bridges:
        prosthetics["bridges"] = bridges
    if prosthetics:
        result["prosthetics"] = prosthetics

    if bone_imaged:
        result["bone_quality"] = build_bone_quality(
            g.get("bone_quality_mandible", {}), teeth, fdi_list, notes, "mandible",
            cross_validate)

    # Type C: both sides unconditionally -- no `if` guard, a missing VQA
    # answer must still produce a stated condyle. Inclusion only in v6.1;
    # the morphology field these entries used to carry no longer exists.
    # Both sides carry the SAME value by construction (merge_condyle_scopes),
    # folded to included|not_included. The two entries stay -- the renderer, the
    # surveys and the LIM arm all read `condyles.{right,left}.scope` -- while the
    # report states them once, together. With a facts file,
    # source_rules.apply_condyle_fov can still overrule this from the mask.
    merged = merge_condyle_scopes(
        _as_dict(g.get("mandible_condyle_right")).get("scope"),
        _as_dict(g.get("mandible_condyle_left")).get("scope"))
    result["condyles"] = {"right": build_condyle_entry(merged),
                          "left": build_condyle_entry(merged)}

    return result


def build_maxilla_section(g: Dict, teeth: Dict, notes: Optional[List[str]] = None,
                          wisdom_eruption_fallback: bool = False,
                          cross_validate: Optional[bool] = None) -> Dict:
    """
    Key insertion order matches the confirmed report order:
    Scope -> Atrophy -> Periodontal -> Dental Arch -> Tooth Findings ->
    Prosthetics -> bone_quality_maxilla -> Maxillary Sinuses.
    """
    fdi_list = fdis_in_range(11, 28)
    findings = arch_findings_map(g.get("dental_arch_findings_maxilla"), fdi_list,
                                 "maxilla", notes)
    findings, uncertain_demoted = demote_uncertain_findings(
        findings, g.get("dental_arch_findings_maxilla"), fdi_list, "maxilla", notes)
    wisdom = wisdom_facts_for_arch(g, fdi_list)
    absent, absent_known = collect_arch_absent(teeth, fdi_list, "maxilla",
                                               findings, notes)

    # Built first for the same reason as in build_mandible_section, and
    # written into `result` in report order further down.
    remnants = build_root_remnants(teeth, fdi_list)
    toothless = toothless_fdis(absent, remnants, fdi_list)
    note_suppressed(notes, "maxilla", toothless, fdi_list)

    scope = build_maxilla_scope(g.get("maxilla_scope", {}))
    bone_imaged = scope["level"] != "none"
    result: Dict = {"scope": scope}

    # Every BONE fact is dropped when the maxilla was not imaged, because a
    # fact about bone that was never imaged is an assertion about nothing (and
    # bone_quality's default is the assertive negative "No definite osteolytic
    # or osteocondensing lesions observed.").
    #
    # SUPERSEDED, 2026-08-10 -- this gate used to be the whole story, and the
    # split was by TISSUE rather than by section: a not_included arch dropped
    # its bone facts but still reported its teeth and prosthetics, on the
    # reasoning that a volume can catch the upper teeth while stopping short
    # of the maxillary bone. It cannot catch them while stopping short of the
    # maxilla itself, which is what not_included says, so the arch is now
    # emptied upstream instead (drop_excluded_maxilla) and this gate only ever
    # sees facts that are already None. It is kept as the belt to that
    # braces: --keep-excluded-maxilla turns the upstream drop off, and then
    # this is once again the only thing standing between an unimaged maxilla
    # and a confident sentence about its bone.
    if bone_imaged:
        atrophy = build_alveolar_bone_atrophy(g.get("alveolar_bone_atrophy_maxilla", {}))
        if atrophy:
            result["alveolar_bone_atrophy"] = atrophy

        periodontal = build_periodontal_bone_resorption(g.get("periodontal_bone_resorption_maxilla", {}))
        if periodontal:
            result["periodontal_bone_resorption"] = periodontal

    absent_pattern = classify_absent_teeth_pattern(absent, MAXILLA_QUADRANTS, absent_known)
    if not bone_imaged:
        # AN EXCLUDED MAXILLA HAS NO ABSENT TEETH, IT HAS UNASSESSABLE ONES
        # (2026-08-17). The sixteen upper positions of a volume that stops
        # below the maxilla were never imaged, and a position that was never
        # imaged answers neither question: it is not absent, and -- the bug
        # this replaces -- it is not present either. `classify_absent_teeth_
        # _pattern` has no state for that, so the block is rewritten here.
        #
        # Measured on the reports rather than argued (code/survey_upper_
        # mentions.py, all 1000 references): where fov.maxilla == "excluded",
        # 77% of reports say only that the maxilla is out of the volume and
        # 13% never mention the upper arch at all; 15% of those cases name an
        # upper tooth anywhere and 4.5% call one absent. The radiologist
        # declines the question, and so does this block.
        #
        # `unassessable` carries the positions rather than dropping them so
        # the state is auditable and distinct from pattern "unknown", which
        # means nobody looked. Neither `teeth` nor `present` is written: every
        # consumer reads those two, so their absence is what makes the sixteen
        # positions unclaimed rather than claimed-as-present.
        absent_pattern = {"pattern": "unassessable",
                          "unassessable": list(fdi_list),
                          "source": scope.get("scope_source")
                          or "maxilla_scope.maxilla_included == not_included"}
        if notes is not None:
            notes.append("maxilla: not in the volume -- its 16 positions are "
                         "unassessable, neither absent nor present")
    primary = primary_teeth_of(g.get("primary_teeth_maxilla"))
    result["arch_findings"] = {
        "absent_teeth": absent_pattern,
        "primary_teeth": sorted(primary) if primary else None,
    }
    # Carried for auditing: which teeth the uncertainty gate acted on (or
    # would have, with the flag off). Nothing renders it -- it exists so a
    # missing finding can be traced to the gate rather than to the model.
    if uncertain_demoted:
        result["arch_findings"]["uncertain_demoted"] = uncertain_demoted

    if remnants:
        result["root_remnants"] = remnants

    impacted, impacted_dropped = build_impacted_teeth(
        teeth, fdi_list, findings, wisdom, notes, cross_validate, "maxilla",
        # The FULL absent set, not `toothless` -- toothless_fdis exempts
        # retained root remnants so their findings survive (endodontic filling
        # material in a retained root is real). That exemption must not extend
        # to impaction: a root fragment cannot be an impacted tooth. A022 hit
        # exactly this -- 24 reported absent AND a remnant AND "impacted".
        absent=set(as_fdi_list(absent)) & set(fdi_list))
    if impacted:
        result["impacted_teeth"] = impacted

    wisdom_teeth = build_wisdom_teeth(wisdom, toothless, teeth, notes,
                                      fallback=wisdom_eruption_fallback)
    if wisdom_teeth:
        result["wisdom_teeth"] = wisdom_teeth

    endo, endo_dropped = build_endo_summary(findings, teeth, fdi_list, toothless,
                                            cross_validate, "maxilla", notes)
    if endo:
        result["endodontic_summary"] = endo

    restorations, restoration_dropped = build_restoration_summary(
        findings, teeth, fdi_list, toothless, cross_validate, "maxilla", notes)
    if restorations:
        result["restoration_summary"] = restorations

    caries_dropped = gate_caries(findings, teeth, fdi_list, toothless,
                                 cross_validate, "maxilla", notes)

    record_cross_source_dropped(result, endo_dropped, restoration_dropped,
                                impacted_dropped, caries_dropped)

    # Only suppress when the gate actually removed them: with it off, the
    # dropped lists are recorded for audit but nothing is taken out anywhere.
    suppress = gate_is_on(cross_validate)
    tooth_findings = build_tooth_findings(
        teeth, fdi_list, toothless,
        set(endo_dropped) if suppress else None,
        set(restoration_dropped) if suppress else None,
        set(caries_dropped) if suppress else None)
    if tooth_findings:
        result["tooth_findings"] = tooth_findings

    prosthetics = {}
    implants = build_implants(g.get("implants_maxilla"), notes, "maxilla", fdi_list)
    implants, implants_dropped = gate_implants(implants, teeth, findings,
                                               cross_validate, "maxilla", notes)
    if implants_dropped:
        result["arch_findings"].setdefault("cross_source_dropped", {})[
            "implants_on_present_teeth"] = implants_dropped
    if implants:
        prosthetics["implants"] = implants
    bridges = build_bridges(g.get("fixed_bridges_maxilla"))
    if bridges:
        prosthetics["bridges"] = bridges
    if prosthetics:
        result["prosthetics"] = prosthetics

    if bone_imaged:
        result["bone_quality"] = build_bone_quality(
            g.get("bone_quality_maxilla", {}), teeth, fdi_list, notes, "maxilla",
            cross_validate)

    # The maxillary sinuses are cavities INSIDE the maxilla, and the nasal
    # cavity sits with them at the same superior level -- if the maxilla was
    # not in the scan volume then neither were they, whatever the sinus facts
    # claim (they are frequently unanswered and default to a healthy sinus,
    # which would put a fabricated "normally pneumatized and clear" sentence
    # in a report about anatomy that was never imaged). The whole MAXILLA
    # SINUS block is therefore dropped with the rest of the maxillary bone.
    if bone_imaged:
        sr = g.get("maxilla_sinus_right", {})
        sl = g.get("maxilla_sinus_left", {})
        result["sinuses"] = build_sinus_group(sr, sl)   # Type C scope, always
        sinuses = result["sinuses"]

        # A root cannot be shown projecting into a sinus that wasn't scanned,
        # and the negative form of this statement ("no pathological
        # relationships are detected") would be an assertion about anatomy
        # nobody imaged.
        if sinuses["group_status"] != "none_included":
            result["sinus_intrasinusal_teeth"] = build_intrasinusal_teeth(sr, sl)

    # NO nasal cavity block: schema v6.1 dropped the nasal_cavity fact
    # entirely (create_sinus_detail.py still writes the image, but nothing
    # asks a question about it), so there is no turbinate finding to report.

    return result


# ── "not_included" means the whole arch, teeth and all ────────────────────
#
# THE RULE (2026-08-10, revised 2026-08-17): the maxilla's inclusion is the
# gate, and with a facts file it is read from the ACQUISITION -- facts
# .structured.fov.maxilla, which extract_facts.py measures off the mask and
# audit_facts.py corrects. Only when there is no facts file does the model's
# own maxilla_scope.maxilla_included decide. Either way the three states are:
#
#     fully_included / partially_included -> KEEP EVERYTHING, unchanged. A
#         partial maxilla is still an imaged maxilla; the report describes
#         what was caught, exactly as the reference reports do ("As far as
#         can be assessed, ...").
#     not_included                        -> DELETE THE ARCH. Every other
#         maxilla fact := None and every tooth_11..28 entry dropped from
#         `teeth`, so nothing maxillary survives to be rendered.
#     unanswered / anything else          -> KEEP. Deleting an arch on the
#         strength of a fact the model never answered would be a guess, and
#         it is the destructive direction of the two.
#
# This SUPERSEDES the earlier per-tissue split (an arch could be
# not_included while its teeth were still reported -- see build_maxilla_
# section). A volume that does not contain the maxilla does not contain the
# maxillary teeth either: they sit in it. Reporting "Maxilla is not included
# in the scan volume." and then listing upper teeth is self-contradictory in
# a way a reader notices immediately, and it is the shape the model actually
# produces -- it is handed the same full-mouth panoramic for both arch
# prompts and answers the maxilla ones whether or not the arch is there.
#
# WHY THE FACTS AND NOT THE MODEL (2026-08-17). "Is the maxilla in this
# volume" is a property of the acquisition, and the mask answers it exactly:
# extract_facts.py already measures `mask == MAXILLA` against the volume's
# superior edge, and audit_facts.py already rewrites fov.maxilla to "excluded"
# below MIN_ARCH_BONE_MM3. The model is answering the same question off one 3D
# render and gets it wrong in the destructive direction -- on validate-40 the
# two disagree four times, and the one disagreement the old gate ACTED on was
# A018: fov "partial", maxillary bone in the mask, a four-sentence maxilla
# paragraph in its reference report ("Completely edentulous maxilla. Severe
# atrophy ... maxillary sinuses ... no osteolytic lesions"), and the arch was
# emptied to a single false sentence because one render looked bare. The three
# disagreements in the other direction (model "included", fov "excluded") were
# already caught by the coverage sidecar, which is the same measurement under
# another name; routing them through the facts only renames the scope_source.
#
# SURVEYED against the consensus ground truth (docs/postprocess.md, THE
# RULE -- maxilla FOV scope): the gate goes 0.850 -> 0.875 on validate-40 and
# does so on ALL THREE model arms, each moving one case and a DIFFERENT case
# per arm (A018 arm 6, S0037 arm 5, S0017 AWQ base) -- every arm invents one
# not_included on a maxilla the mask can see. On the training split the same
# comparison is a tie (0.789 both ways, 24 cases moved, 12-12), against
# in-sample predictions.
#
# WHY NOT THE SUB-DATASET PREFIX, the blunter version of the same idea. The
# datasets do differ in FOV -- from the NIfTI headers of all 622 cases, mean Z
# extent is A 88 mm, F 81 mm, P 50.7 +/- 0.4, S 52.4 +/- 4.0, so P and S are
# ~5 cm single-arch slabs -- but 27% of S reports and 9.5% of P reports DO
# describe upper teeth ("coronal restorations on teeth 26, 14 and 15"), and
# S's modal scope word in the references is "partially included" (31/52), not
# "not included" (19/52). FORCE_MAXILLA_EXCLUDED_PREFIXES below restores the
# blunt version for the arm that wants to measure it; it is empty by default.
#
# NO FACTS FILE, NO CHANGE. facts_maxilla_included returns None then and the
# model's answer is the gate again, exactly as before -- which is also what
# keeps `postprocess_now.sh NO_SOURCE_RULES=1`-style runs comparable.
FORCE_MAXILLA_EXCLUDED_PREFIXES: set = set()

# Maxilla facts that are about TEETH, not about bone, and therefore survive an
# excluded maxilla. All four are answered off the panoramic, which still shows
# the upper crowns when the volume stopped below the maxillary bone; the bone
# facts (atrophy, periodontal resorption, bone_quality, both sinuses) do not
# appear here and are nulled. Listed explicitly rather than derived, because
# "is this fact about bone or about teeth" is a clinical judgment the schema
# does not encode -- see drop_excluded_maxilla for the measurement behind it.
MAXILLA_TOOTH_FACTS = frozenset({
    "dental_arch_findings_maxilla",
    "implants_maxilla",
    "fixed_bridges_maxilla",
    "primary_teeth_maxilla",
})

# Set True (--drop-excluded-maxilla-teeth) to null MAXILLA_TOOTH_FACTS as well,
# i.e. an excluded maxilla reports nothing at all -- the behaviour between the
# 2026-08-10 delete-everything pass and this one.
#
# UNRESOLVED, AND DELIBERATELY A FLAG. The case for keeping the teeth is the
# reference corpus: 30 of the 256 excluded-maxilla cases (11.7%) describe an
# upper tooth, always a crown finding the panoramic can still see. The case
# against is the only measurement available, and it is too small to settle
# anything -- the validate split has predictions for 40 cases, of which
# exactly TWO (P452, S0027) reach this code path, and neither one's reference
# mentions an upper tooth. On those two, keeping the teeth costs BLEU 0.1677
# -> 0.1657 and METEOR 0.3767 -> 0.3750. n=2 is noise, and BLEU is not the
# target metric; RadFact Logical F1 is, and it needs the judge server. Run
# both arms once there are predictions for a split where the group is more
# than two cases.
DROP_EXCLUDED_MAXILLA_TEETH = False

# Whether an excluded maxilla is dropped AT ALL. This was a CLI-only parameter
# (`--keep-excluded-maxilla`) with no module default, which made it the one arm
# knob a config file could not name. Given a module default it binds like the
# rest; `postprocess_prediction(drop_excluded=...)` still overrides it, exactly
# the way `cross_validate=` overrides REQUIRE_CROSS_SOURCE_AGREEMENT.
DROP_EXCLUDED_MAXILLA = True

# Same treatment for the wisdom-eruption fallback: take a 3D wisdom fact's
# missing eruption_state from that tooth's own composite read. Required by any
# schema that omits the four 3D wisdom facts -- schema_dedup.json does -- and
# NOT a no-op on the aksssr arm, where 11 of 40 validate cases gain sentences.
WISDOM_ERUPTION_FALLBACK = False

_MAXILLA_FIELD_CACHE: Dict[str, List[str]] = {}


def maxilla_fact_fields(schema_path: Optional[str] = None) -> List[str]:
    """
    The output_field of every fact in the schema's `maxilla` section, read
    from schema.json rather than listed here -- the schema is the single
    source of truth, so a fact added to that section is dropped along with the
    rest without this module being touched.
    """
    path = str(schema_path or DEFAULT_SCHEMA)
    if path not in _MAXILLA_FIELD_CACHE:
        with open(path) as f:
            schema = json.load(f)
        _MAXILLA_FIELD_CACHE[path] = [fact["output_field"]
                                      for fact in schema.get("maxilla", []) or []]
    return _MAXILLA_FIELD_CACHE[path]


def maxilla_included_answer(pred: Dict) -> Optional[str]:
    """The model's own maxilla_scope.maxilla_included, or None if unanswered."""
    scope = _as_dict(_as_dict(pred.get("global")).get("maxilla_scope"))
    included = scope.get("maxilla_included")
    return included.strip() if isinstance(included, str) and included.strip() else None


def mask_says_maxilla_excluded(pred: Dict) -> bool:
    """
    The MASK's verdict, carried in the prediction as `coverage` (written by
    create_tooth_detail.py, threaded through build_vqa_pairs.py and
    run_vqa_inference.py). False when no coverage travelled with the
    prediction -- an older file, or a run whose images predate the sidecar.

    This has to be consulted alongside the model's own maxilla_included,
    because the two gates sit at different ends of the pipeline and CAN
    disagree: the composites are skipped from the mask before any inference,
    while `not_included` is the model's answer afterwards. When the mask
    skipped the upper composites and the model then said "included", every
    upper tooth arrives as detected == "no_image" and ABSENT_DETECTED_VALUES
    reads that as absent -- which rendered a fully-dentate maxilla as
    "Complete edentulism". Gating the drop on the mask as well closes it.
    """
    return _as_dict(_as_dict(pred.get("coverage")).get("maxilla")).get("bone") is False


def facts_maxilla_included(facts: Optional[Dict]) -> Optional[str]:
    """
    THE ACQUISITION'S OWN ANSWER, from facts.structured.fov.maxilla:

        "excluded" -> "not_included"   (audit_facts.py wrote it: the mask holds
                                        less than MIN_ARCH_BONE_MM3 of maxilla)
        "partial"  -> "included"       (extract_facts.py: maxillary bone IS in
                                        the mask, cut off superiorly or carrying
                                        fewer than four upper teeth)
        no maxilla key in an fov block -> "included" (neither condition fired,
                                        so the maxilla is in the volume whole)
        no fov block / no facts        -> None, nothing is claimed

    None is the only value that leaves the model's own answer in charge -- see
    the rule above FORCE_MAXILLA_EXCLUDED_PREFIXES for why the facts outrank it
    when they exist.

    Note fov is derived from the MASK, not from the report (extract_facts.py
    measures `mask == MAXILLA` against the volume's superior edge), so this is
    available wherever a segmentation is -- it is not reference-report leakage.
    """
    structured = _as_dict(_as_dict(facts).get("structured"))
    if "fov" not in structured or not isinstance(structured["fov"], dict):
        return None
    value = structured["fov"].get("maxilla")
    if value is None:
        return "included"
    return "not_included" if str(value).strip().lower() == "excluded" else "included"


def is_forced_excluded_case(case_id: Optional[str]) -> bool:
    """
    Sub-dataset is the case-ID prefix letter (A004, S0030).
    Off unless --force-maxilla-excluded-prefixes asked for it.
    """
    return bool(case_id) and case_id[0].upper() in FORCE_MAXILLA_EXCLUDED_PREFIXES


def drop_excluded_maxilla(pred: Dict, schema_path: Optional[str] = None,
                          forced: bool = False,
                          scope_source: str = "forced_by_dataset_prefix"
                          ) -> Tuple[Dict, List[str]]:
    """
    Empty the maxilla of a case whose maxilla_scope says not_included. Returns
    a shallow copy of `pred` with new `global`/`teeth` dicts (the caller's
    prediction object is left alone) plus human-readable notes, empty when
    there was nothing to drop.

    maxilla_scope itself is the one maxilla fact KEPT: it carries the only
    true thing left to say about the arch, and build_maxilla_scope reads
    `maxilla_included` off it to set the section's scope level -- nulling it
    would leave level "unknown" -> "Maxilla is partially included in the scan
    volume.", the opposite of what the case just claimed. `forced` rewrites it
    instead of trusting it, for the prefix override, and stamps a scope_source
    so the summary can tell the two apart.

    WHAT SURVIVES, AND WHY (revised 2026-08-10). Every BONE fact goes: they
    are already gated on `bone_imaged` in build_maxilla_section, and nulling
    them here makes the summary say so explicitly. The ARCH-LEVEL TOOTH facts
    stay -- dental_arch_findings_maxilla and the prosthetics that ride with
    it -- because they are read off the PANORAMIC, which still shows the upper
    crowns of a volume that stopped below the maxillary bone. 11.7% of the
    reference reports for exactly those cases describe one ("as far as can be
    visualized, prosthetic crowns are present on teeth 14, 13, 12..."), and
    dropping the fact made that unreachable.

    The per-tooth COMPOSITE reads (tooth_11..28) do NOT survive, for the
    opposite reason: a composite is built around a tooth's own extent and its
    three root panels are empty when the crop has no root in it. Those images
    are no longer generated at all (create_tooth_detail.py), so for a current
    prediction there is nothing here to drop; the drop remains for replaying
    older prediction files that still carry them.

    Net effect: a not_included maxilla reports its teeth and nothing about its
    bone -- the split is by TISSUE, which is where this started before the
    2026-08-10 delete-everything pass. dentition_type still sees
    primary_teeth_maxilla, which is correct: it is a tooth fact.
    """
    g = dict(pred.get("global", {}) or {})
    teeth = dict(pred.get("teeth", {}) or {})
    notes: List[str] = []

    keep = ({"maxilla_scope"} if DROP_EXCLUDED_MAXILLA_TEETH
            else MAXILLA_TOOTH_FACTS | {"maxilla_scope"})
    nulled = [f for f in maxilla_fact_fields(schema_path)
              if f not in keep and is_meaningful(g.get(f))]
    for field in maxilla_fact_fields(schema_path):
        if field not in keep:
            g[field] = None
    if forced:
        g["maxilla_scope"] = {"maxilla_included": "not_included",
                              "scope_source": scope_source}

    dropped = [f"tooth_{fdi}" for fdi in fdis_in_range(11, 28) if f"tooth_{fdi}" in teeth]
    for key in dropped:
        del teeth[key]

    if nulled:
        notes.append(f"nulled {len(nulled)} answered maxilla fact(s): "
                     f"{', '.join(sorted(nulled))}")
    if dropped:
        notes.append(f"dropped {len(dropped)} upper tooth read(s): "
                     f"{', '.join(sorted(dropped))}")

    out = dict(pred)
    out["global"] = g
    out["teeth"] = teeth
    return out, notes


# ── Top-level assembly ────────────────────────────────────────────────────

def postprocess_prediction(pred: Dict, normalize: bool = True,
                           schema_path: Optional[str] = None,
                           quiet: bool = False,
                           wisdom_eruption_fallback: Optional[bool] = None,
                           cross_validate: Optional[bool] = None,
                           drop_excluded: Optional[bool] = None,
                           facts: Optional[Dict] = None) -> Dict:
    # None means "whatever the configuration says", which is the module global
    # -- the same contract `cross_validate` has always had, extended to the two
    # parameters that used to hardcode their own default here and so could not
    # be set from configs/postprocess/*.yaml. An explicit argument still wins.
    if wisdom_eruption_fallback is None:
        wisdom_eruption_fallback = WISDOM_ERUPTION_FALLBACK
    if drop_excluded is None:
        drop_excluded = DROP_EXCLUDED_MAXILLA

    if normalize:
        pred, repairs = normalize_prediction(pred, schema_path)
        if repairs and not quiet:
            print(f"    [WARN] {pred.get('case_id')}: repaired {len(repairs)} "
                  f"schema-violating field(s): {summarize_repairs(repairs)}",
                  file=sys.stderr)

    case_id = pred.get("case_id")

    # Before anything reads the facts: a maxilla the case itself calls
    # not_included has no teeth and no bone to report on. partially_included
    # and fully_included keep everything -- see the rule above the constant.
    forced = is_forced_excluded_case(case_id)
    by_facts = facts_maxilla_included(facts)
    # THE FACTS OUTRANK BOTH the model and the coverage sidecar when they say
    # the maxilla IS there: the model's not_included is then overwritten rather
    # than acted on, so the arch survives and the section opens with the
    # sentence the acquisition supports ("partially included"). Without this,
    # A018 -- maxilla in the mask, fov "partial", a four-sentence maxilla
    # paragraph in its reference -- was emptied down to "Maxilla is not
    # included in the scan volume." on the model's word alone.
    if by_facts == "included":
        g_scope = dict(_as_dict(_as_dict(pred.get("global")).get("maxilla_scope")))
        was = g_scope.get("maxilla_included")
        if was != "included":
            g_scope["maxilla_included"] = "included"
            g_scope["scope_source"] = "facts_fov_maxilla_in_volume"
            pred = dict(pred)
            pred["global"] = dict(pred.get("global", {}) or {},
                                  maxilla_scope=g_scope)
            if not quiet:
                print(f"    [INFO] {case_id}: maxilla scope {was!r} -> 'included' "
                      f"(facts fov says the maxilla is in the volume)",
                      file=sys.stderr)
    by_mask = mask_says_maxilla_excluded(pred) and by_facts != "included"
    if drop_excluded and (forced or by_facts == "not_included" or by_mask
                          or (by_facts is None
                              and maxilla_included_answer(pred) == "not_included")):
        # The mask overrules the model on its own subject. When the voxels
        # say there is no maxillary bone, the scope is rewritten too, not just
        # the facts nulled -- otherwise level stays "partial", bone_imaged
        # stays True, and bone_quality emits its assertive default ("No
        # definite osteolytic or osteocondensing lesions") about bone that was
        # never in the volume.
        excluded_by_facts = by_facts == "not_included"
        pred, arch_notes = drop_excluded_maxilla(
            pred, schema_path, forced=forced or excluded_by_facts or by_mask,
            scope_source=("forced_by_dataset_prefix" if forced
                          else "facts_fov_maxilla_excluded" if excluded_by_facts
                          else "mask_has_no_maxillary_bone"))
        if arch_notes and not quiet:
            why = ("forced by dataset prefix" if forced
                   else "facts fov says the maxilla is excluded" if excluded_by_facts
                   else "mask carries no maxillary bone" if by_mask
                   else "maxilla_scope says not_included")
            print(f"    [WARN] {case_id}: maxilla excluded ({why}): "
                  f"{'; '.join(arch_notes)}", file=sys.stderr)

    g = pred.get("global", {}) or {}
    teeth = pred.get("teeth", {}) or {}

    # An arch cannot be "completely included" while one of its own
    # substructures is cut off or absent from the volume.
    g, scope_notes = enforce_scope_consistency(g)
    if scope_notes and not quiet:
        print(f"    [WARN] {case_id}: scope inconsistency: {'; '.join(scope_notes)}",
              file=sys.stderr)

    out: Dict = {"case_id": case_id}

    out["dentition_type"] = build_dentition_type(
        g.get("primary_teeth_mandible"), g.get("primary_teeth_maxilla"), teeth)

    # No cross-arch merge of any kind (this round's explicit instruction) --
    # mandible and maxilla are fully independent sections, each closing
    # with its own bone_quality statement.
    absent_notes: List[str] = []
    out["mandible"] = build_mandible_section(g, teeth, absent_notes,
                                             wisdom_eruption_fallback,
                                             cross_validate)
    out["maxilla"] = build_maxilla_section(g, teeth, absent_notes,
                                           wisdom_eruption_fallback,
                                           cross_validate)
    if absent_notes and not quiet:
        print(f"    [WARN] {case_id}: {'; '.join(absent_notes)}", file=sys.stderr)

    # ── THE SOURCE RULES (docs/postprocess.md) ────────────────────────
    # A post-pass, not surgery in the builders above: each rule says a field
    # should come from a different SOURCE, and rewriting the finished dict
    # keeps every rule to one function and leaves this file's existing paths
    # untouched. With no facts file it is a no-op -- see code/pipeline/postprocess/source_rules.py.
    if facts:
        out, rule_notes = source_rules.apply(out, pred, facts)
        if rule_notes and not quiet:
            print(f"    [INFO] {case_id}: source rules -- "
                  f"{'; '.join(rule_notes)}", file=sys.stderr)

    # WHICH ARM WROTE THIS FILE. outputs/ is gitignored and job logs rotate, so
    # a summary that cannot name its own configuration is a measurement with no
    # provenance. Only the config path and the keys that DIFFER from the arm-6
    # defaults are stored -- a run of the defaults stamps an empty dict rather
    # than thirty lines of restated defaults into all 40 files.
    out["postprocess_config"] = rules_config.provenance()

    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Postprocess {case_id}_pred.json into a compact, "
                     "pre-classified {case_id}_summary.json for synthesize_report.py "
                     "(the deterministic template renderer -- Task A).")
    ap.add_argument("--pred-dir", required=True,
                     help="Directory of {case_id}_pred.json files, or a single such file")
    ap.add_argument("--out-dir", required=True,
                     help="Output directory for {case_id}_summary.json files")
    ap.add_argument("--facts-dir", default=None,
                    help="dataset/<split>/facts -- enables THE SOURCE RULES of "
                         "docs/postprocess.md, which re-source absence, "
                         "impaction, endodontic, fillings, crown, implants, "
                         "canal adjacency/location, atrophy and the bridge gate. "
                         "It ALSO hands the maxilla's FOV scope to the facts: "
                         "structured.fov.maxilla overrides the model's "
                         "maxilla_scope answer in both directions, so the arch "
                         "is emptied when the mask has no maxilla and KEPT when "
                         "it has one, whatever the render looked like. "
                         "Without it postprocess behaves exactly as before.")
    ap.add_argument("--case-ids", nargs="+", default=None,
                     help="Filter to specific case IDs")
    ap.add_argument("--schema", default=None,
                     help="Path to schema.json (passed through to normalize_prediction). "
                          "Explicit is safer than normalize_pred.py's relative-path default "
                          "when running from a SLURM job's working directory.")
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="THE ARM. A configs/postprocess/*.yaml naming which of "
                         "the source rules, cross-source gates and FOV policies "
                         "of docs/postprocess.md are on. Omitted, the arm-6 "
                         "defaults apply and behaviour is identical to before "
                         "the config layer existed. Every legacy flag below "
                         "still works and OVERRIDES the file, so an existing "
                         "job script keeps its meaning. See "
                         "configs/postprocess/README.md.")
    ap.add_argument("--print-config", action="store_true",
                    help="Resolve --config over the defaults, print every "
                         "settable key with the section of docs/postprocess.md "
                         "that measured it, and exit without reading a "
                         "prediction. This is how an arm is checked BEFORE a "
                         "run rather than reconstructed from its output.")
    ap.add_argument("--wisdom-eruption-fallback", action="store_true",
                     help="When a 3D-render wisdom-tooth fact (lower_left_wisdom_tooth "
                          "etc.) gave no eruption_state, take it from that tooth's own "
                          "composite read (tooth_{fdi}_eruption) instead. Required for "
                          "schemas that omit those four facts -- schema_dedup.json drops "
                          "them, and without this its arm loses every wisdom-eruption "
                          "sentence. OFF by default: it is not a no-op on the aksssr arm, "
                          "where 11 of 40 validate cases would gain new sentences.")
    ap.add_argument("--no-cross-validate", action="store_true",
                     help="Turn OFF the cross-source agreement gate "
                          "(REQUIRE_CROSS_SOURCE_AGREEMENT). THIS IS NOW THE DEFAULT, "
                          "so the flag is a no-op kept for the scripts and job headers "
                          "that still pass it. A finding asked of both the panoramic "
                          "arch read and the tooth's own composite crop is reported as "
                          "the UNION and the disagreements are only annotated.")
    ap.add_argument("--cross-validate", action="store_true",
                     help="Turn the precision gates back ON -- the pre-2026-08-07 arm. "
                          "Restores the cross-source agreement gate and everything "
                          "subordinate to it (the per-finding CROSS_VALIDATE_* votes at "
                          "their measured settings, the caries gate, the "
                          "implant-on-present-tooth gate and the bone-quality per-tooth "
                          "confirmation), so a finding two sources disagree about is "
                          "DROPPED instead of reported. Either way the affected teeth "
                          "are listed under arch_findings.cross_source_dropped, so both "
                          "arms are traceable from one run.")
    ap.add_argument("--demote-uncertain", action="store_true",
                     help="Turn the uncertainty gate back ON "
                          "(DEMOTE_UNCERTAIN_TO_NORMAL, off by default since "
                          "2026-08-07): force every tooth the model listed in "
                          "dental_arch_findings_{arch}.uncertain_teeth back to "
                          "'normal' before anything reads the map. Independent of "
                          "--cross-validate. The affected teeth are recorded under "
                          "arch_findings.uncertain_demoted either way.")
    ap.add_argument("--keep-excluded-maxilla", action="store_true",
                     help="Report the maxilla even in cases whose maxilla_scope says "
                          "not_included -- the pre-2026-08-10 per-tissue split, where a "
                          "not_included arch dropped its BONE facts but still listed its "
                          "teeth. OFF by default: an arch that is not in the volume has "
                          "no teeth in the volume either. For the arm that measures what "
                          "the drop is worth.")
    ap.add_argument("--drop-excluded-maxilla-teeth", action="store_true",
                     help="For a not_included maxilla, drop the ARCH-LEVEL TOOTH "
                          "facts too (dental_arch_findings_maxilla, implants, "
                          "bridges, primary teeth), leaving the section as the "
                          "scope sentence alone. OFF by default: those facts are "
                          "read off the panoramic, which still shows the upper "
                          "crowns, and 11.7%% of the reference reports for these "
                          "cases describe one. The A/B arm for that call -- see "
                          "DROP_EXCLUDED_MAXILLA_TEETH for what is and is not "
                          "measured.")
    ap.add_argument("--force-maxilla-excluded-prefixes", default=None, metavar="LETTERS",
                     help="Case-ID prefix letters whose maxilla is treated as "
                          "not_included WHATEVER the model answered -- 'PS' reproduces "
                          "the FOV-based arm (P and S are ~51 mm single-arch volumes). "
                          "Empty by default: the per-case maxilla_scope answer is the "
                          "gate, and forcing S costs the 27%% of its reference reports "
                          "that do describe upper teeth.")
    args = ap.parse_args()

    if args.cross_validate and args.no_cross_validate:
        ap.error("--cross-validate and --no-cross-validate are contradictory")

    if args.force_maxilla_excluded_prefixes and args.keep_excluded_maxilla:
        ap.error("--force-maxilla-excluded-prefixes and --keep-excluded-maxilla "
                 "are contradictory")

    # ── the arm ────────────────────────────────────────────────────────────
    # THREE LAYERS, in this order: the arm-6 defaults, the config file, then
    # whatever legacy flag was typed. The last layer is what keeps every job
    # script and doc example in the repo meaning what it says -- a flag on the
    # command line is an explicit instruction and outranks a file.
    #
    # Nothing below sets a module global by hand any more. rules_config.apply()
    # writes all of them through BINDINGS, which is also the only list of what
    # is settable at all, so a knob added there is settable from a config file
    # and printable by --print-config without this function changing.
    overrides: Dict[str, object] = {}
    if args.cross_validate:
        overrides["cross_source.require_agreement"] = True
    elif args.no_cross_validate:
        overrides["cross_source.require_agreement"] = False
    if args.demote_uncertain:
        overrides["gates.demote_uncertain_to_normal"] = True
    if args.keep_excluded_maxilla:
        overrides["maxilla_fov.drop_excluded"] = False
    if args.drop_excluded_maxilla_teeth:
        overrides["maxilla_fov.drop_excluded_teeth"] = True
    if args.force_maxilla_excluded_prefixes is not None:
        overrides["maxilla_fov.force_excluded_prefixes"] =             list(args.force_maxilla_excluded_prefixes)
    if args.wisdom_eruption_fallback:
        overrides["extraction.wisdom_eruption_fallback"] = True

    try:
        settings = rules_config.load_and_apply(args.config, overrides)
    except rules_config.ConfigError as exc:
        ap.error(str(exc))

    if args.print_config:
        if args.config:
            print(f"# {args.config}")
        print(rules_config.describe(settings))
        drift = rules_config.verify_defaults()
        if drift:
            print()
            print("[FAIL] rules_config.DEFAULTS have drifted from the module "
                  "constants -- a run with no --config and a run with "
                  "default.yaml would now differ:")
            for d in drift:
                print(f"  {d}")
            raise SystemExit(1)
        return

    pred_path = Path(args.pred_dir)
    if pred_path.is_file():
        pred_files = [pred_path]
    else:
        pred_files = sorted(pred_path.glob("*_pred.json"))

    if args.case_ids:
        pred_files = [p for p in pred_files if p.stem.replace("_pred", "") in args.case_ids]

    if not pred_files:
        print(f"[WARN] No prediction files found under {pred_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for pf in pred_files:
        pred = json.loads(pf.read_text(encoding="utf-8"))
        summary = postprocess_prediction(
            pred, schema_path=args.schema,
            # Every one of these is now None on purpose: the configuration has
            # already been applied to the module globals above, and None is
            # what tells postprocess_prediction to read them. Passing the
            # argparse values here as well would silently outrank --config.
            wisdom_eruption_fallback=None,
            cross_validate=None,
            drop_excluded=None,
            facts=source_rules.load_facts(
                args.facts_dir, pred.get("case_id", pf.stem.replace("_pred", ""))))
        case_id = pred.get("case_id", pf.stem.replace("_pred", ""))
        out_path = Path(args.out_dir) / f"{case_id}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{case_id}] -> {out_path}")


if __name__ == "__main__":
    main()

"""
Usage:
    python3 code/pipeline/postprocess/postprocess_pred.py \
        --pred-dir  outputs/aksssr_v5_validate/predictions \
        --out-dir   outputs/aksssr_v5_validate/summaries \
        --schema    schema/schema.json

    python code/pipeline/postprocess/postprocess_pred.py \
        --pred-dir dataset/training/outputs/predictions/A004_pred.json \
        --out-dir  dataset/training/outputs/summaries
"""