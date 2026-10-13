#!/usr/bin/env python3
"""
code/pipeline/postprocess/synthesize_report.py

Renders a {case_id}_summary.json (from postprocess_pred.py) into the final
free-text CBCT dental report -- Task A, the deterministic template path,
as opposed to Task B's LLM-generated free text (generate_report_llm.py).

WHY A TEMPLATE, NOT AN LLM, FOR THIS STEP
────────────────────────────────────────────
Per this session's own decision: RadFact can only be as good as the
underlying inference, and the captioning score (BLEU-4/METEOR) rewards
n-gram overlap with real radiologist reports, which are themselves
templatic. A deterministic renderer also can't introduce a NEW clinical
error on top of whatever the VQA predictions already got wrong -- it's a
guaranteed-faithful transcription of the structured summary, nothing more.

WHERE THE SENTENCES CAME FROM
────────────────────────────────
Every literal string template below is either:
  (a) a REAL text snippet captured from an actual report and given
      verbatim during this session's decision-tree review (marked with
      "REAL SNIPPET" in a comment), or
  (b) my own construction, built to match the pattern of (a) for cases
      the decision tree didn't give an explicit example for (marked
      "INFERRED" -- these are the ones most worth spot-checking against
      real reports before trusting the output).

Two things NOT done here, on purpose, matching earlier decisions:
  - No merging of mandible canal right/left into one "both normal"
    sentence -- always two independent statements.
  - No merging of mandible and maxilla into one report-level closing
    section -- each arch's bone_quality closes that arch's own section.

"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo bootstrap -- see code/_repo.py.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(next(
    p for p in _pathlib.Path(__file__).resolve().parents
    if (p / "_repo.py").is_file())))
from _repo import add_code_paths  # noqa: E402
add_code_paths()
import rules_config  # noqa: E402  -- THE ARM IS THE CONFIG FILE

# THIS RENDERER HAS TWO SETTABLE THINGS and they are bound in
# rules_config.BINDINGS: `report.render_fillings` (RENDER_FILLINGS below) and
# `priors.canal_location` (DEFAULT_CANAL_LOCATION below, which source_rules.py
# also reads -- the config is now the ONE place that prior is stated). Every
# other string in this file is a report template, which is the schema's and the
# radiologist's business, not an experiment's, and stays in code.


# ── Text-list joining helpers ──────────────────────────────────────────────

def as_list(value) -> List:
    """Coerce a summary field that the schema declares list-valued into a list.

    The summaries are produced upstream by the VQA model, which sometimes
    emits a bare scalar for a single-element list (e.g. "fdi_numbers": 28
    instead of [28]). Rather than trusting the schema at every call site,
    every list-consuming renderer normalizes through here.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def join_list(items: List, sep: str = ", ", final: str = " and ") -> str:
    """[34, 36, 47] -> '34, 36, and 47'; [34, 36] -> '34 and 36'; [34] -> '34'"""
    items = [str(i) for i in as_list(items)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}{final}{items[1]}"
    return sep.join(items[:-1]) + f",{final}{items[-1]}"


def teeth_word(items: List) -> str:
    """'Tooth' for one, 'Teeth' for several -- used throughout."""
    return "Tooth" if len(as_list(items)) == 1 else "Teeth"


# ── Tooth-set helpers ──────────────────────────────────────────────────────

# The third molars, the only teeth the schema reads impaction for with more
# than one source behind it -- see render_impacted_teeth.
WISDOM_TEETH = frozenset({18, 28, 38, 48})


def is_absent_tooth(fdi: int, arch_findings: Optional[Dict]) -> bool:
    """Is this tooth known to be missing from the arch being rendered?

    Reads arch_findings.absent_teeth, whose shape varies by pattern
    (see render_absent_teeth) but which always carries `present`. So:
      - a present list settles it either way;
      - without one -- pattern "unknown", nothing read about this arch --
        presence is UNKNOWN, not false, and only teeth named in an explicit
        absent list are treated as absent.
    """
    absent = (arch_findings or {}).get("absent_teeth") or {}
    present = as_list(absent.get("present"))
    if present:
        return fdi not in present
    if absent.get("pattern") == "quadrant_edentulous_both":
        return True
    return fdi in as_list(absent.get("teeth"))


# ATROPHY RIDES ON THE ABSENCE SENTENCE, 2026-08-17, and has no sentence of
# its own. An alveolar process resorbs because teeth were lost, so the finding
# belongs in the clause that says which ones -- "Absence of teeth 36, 37 and
# 38, with associated bone atrophy." -- rather than in a free-standing
# "Atrophy of the mandibular bone." that repeats the arch and states the
# consequence before the cause. Complete edentulism gets the same treatment in
# its own frame (edentulism_sentence).
#
# The corollary is that atrophy on an arch with NO absence sentence goes
# unstated: pattern "none" says every tooth is there, "unknown" says nothing
# was read, and neither offers a clause to attach to. render_alveolar_bone
# _atrophy is the sentence that used to cover those and is no longer routed.
#
# "with marked atrophy" is a GRADE, and the schema stopped carrying one at
# v6.4: alveolar_bone_atrophy is a bool (see render_alveolar_bone_atrophy),
# so the adjective is fixed text here, not a reading. It is defensible only
# because this sentence is reachable on one state -- an arch with no teeth
# left in it -- where advanced resorption is the norm rather than a claim
# about this patient. The partial-absence clause takes no adjective for the
# same reason, since there no such norm exists. If a grade ever returns to
# the schema, fill the adjective from it and delete this note.

def is_complete_edentulism(absent: Optional[Dict]) -> bool:
    """The arch has nothing in it -- the state that gets the merged sentence."""
    return (absent or {}).get("pattern") == "quadrant_edentulous_both"


def edentulism_sentence(arch_adj: str, atrophy: bool) -> str:
    """
    arch_adj = 'mandibular' or 'maxillary'.

    TWO SENTENCES, ADJACENT — NOT ONE (2026-08-17, measured). The pairing is
    the point: an alveolar process resorbs because the teeth went, so the two
    findings belong side by side, cause before consequence. But they must stay
    two SENTENCES, because RadFact scores claim by claim and a fused sentence
    yields fewer phrases to match. Fusing them cost, on validate-40 under the
    same judge, on the four cases it touched:

        S0009   recall 0.556 -> 0.222, precision 1.000 -> 1.000   -0.285 final
        P123    recall 1.000 -> 1.000, precision 1.000 -> 1.000   -0.008
        F043    recall 0.267 -> 0.333                             +0.041
        A018    (also carries the maxilla FOV rule)               +0.047

    Precision never moved on any of them, which is the signature: merging two
    supported claims cannot make either wrong, it can only make one
    unfindable. The partial-absence clause below ("..., with associated
    bone atrophy") is a DIFFERENT case and stays fused -- there the
    clause ADDS a finding that had no sentence at all before, and it scored
    +0.137 over the 12 cases it touched.

    "Marked atrophy of the alveolar processes." is a corpus form (A018's own
    reference, and the frame appears throughout the training reports); it also
    avoids repeating the arch word a second time in two sentences. On the fixed
    "marked", see the note above is_complete_edentulism: the grade is defensible
    only because this sentence is reachable on one state, an arch with nothing
    left in it, where advanced resorption is the norm.
    """
    if atrophy:
        return (f"Complete {arch_adj} edentulism. "
                f"Marked atrophy of the alveolar processes.")
    return f"Complete {arch_adj} edentulism."


# ── Preamble ────────────────────────────────────────────────────────────────

def render_preamble(dentition: Dict) -> Optional[str]:
    dtype = dentition.get("dentition_type")
    if dtype == "mixed":
        # REAL SNIPPET
        return "Pediatric patient with incomplete eruption of the permanent teeth."
    if dtype == "primary":
        # REAL SNIPPET
        return "Full primary dentition, permanent teeth unerupted."
    # permanent / edentulous: skip statement (REAL: decision tree explicitly
    # says skip for both)
    return None


# ── Scope (Type C) ──────────────────────────────────────────────────────────

def render_mandible_scope(scope: Dict) -> str:
    level = scope.get("level")
    if level == "complete":
        # REAL SNIPPET
        return "Mandible is completely included in the scan volume."
    if level == "none":
        # INFERRED: same sentence shape as the two REAL SNIPPETs, for the
        # third scope state. Saying "partially included" here would assert
        # the opposite of what the prediction said.
        return "Mandible is not included in the scan volume."
    # REAL SNIPPET
    return "Mandible is partially included in the scan volume."


def render_maxilla_scope(scope: Dict) -> str:
    """
    TWO reachable sentences as of schema v6.6, which collapsed
    maxilla_included to included|not_included: "included" carries no claim
    about how much, so it renders as the corpus's own hedge, "partially
    included" -- which is also its modal wording for this arch (56/63 F
    reports, 31/52 S). The "completely included" branch is now reachable only
    from a v6.5-or-earlier prediction that answered fully_included; it is kept
    for those rather than deleted, since dropping it would silently restate
    such a case as the weaker claim.
    """
    level = scope.get("level")
    if level == "complete":
        # REAL SNIPPET
        return "Maxilla is completely included in the scan volume."
    if level == "none":
        # REAL SNIPPET, and it closes the arch (2026-08-17). "...and not
        # assessable" is the corpus's own clause for this state -- A079's
        # "Maxilla: not included in the scan volume and not assessable." --
        # and it is the whole maxilla block in 77% of the reports whose
        # fov.maxilla is "excluded". The clause is what licenses saying
        # nothing further: see render_maxilla_main.
        return "Maxilla is not included in the scan volume and not assessable."
    # REAL SNIPPET
    return "Maxilla is partially included in the scan volume."


# ── Alveolar bone atrophy (one extent per arch) ─────────────────────────────

def render_alveolar_bone_atrophy(atrophy: Optional[Dict], arch_adj: str) -> List[str]:
    """
    NOT ROUTED as of 2026-08-17. Atrophy is stated in the absence sentence's
    own clause now -- "Absence of teeth 36, 37 and 38, with associated bone
    atrophy." -- so neither arch assembler calls this. It is kept
    for the one state that clause cannot cover, an atrophic arch with no
    absence sentence to hang from, and because the history below is the
    reason the finding is written the way it is. Re-routing it is one line in
    render_mandible_main / render_maxilla_main.

    ONE sentence for the arch. Schema v6.1 replaced the per-region list with a
    single overall judgment per arch, so there is no side/location left to
    build a locative phrase from -- and inventing one would name a site the
    fact never gave.

    NO SEVERITY WORD as of v6.4. The fact is a bool now (`atrophy`), where it
    used to be extent = none|mild|moderate|severe, so the corpus's
    "{Mild/Moderate/Severe} atrophy of the mandibular bone" cannot be filled in
    without grading a finding the model was never asked to grade. The bare
    "Atrophy of the ... bone." keeps the corpus frame minus the adjective;
    restoring the grade means restoring `extent` to the schema, not guessing
    here.

    arch_adj = 'mandibular' or 'maxillary'.

    THE NEGATIVE SENTENCE IS GONE, 2026-08-16. "No signs of bone atrophy." was
    carried here as a corpus form and it is not one: across the 40 validate
    references the phrase -- in any wording, negated any way -- occurs ZERO
    times, while 24 of the 40 mention atrophy positively. Radiologists state
    this finding when it is there and say nothing when it is not, so the
    negative could only ever have matched a reference by accident.

    It also asserted the one thing the pipeline cannot see: the model claims
    atrophy on 2 of 22 abnormal arches and 0 of 19 partially edentulous ones,
    so "no signs" was silence dressed as a reading. THE RULE -- alveolar bone
    atrophy states atrophy where the arch carries an edentulous region at all
    (2026-08-17; it was full edentulism only before that); everywhere else this
    renderer emits nothing.

    Kept here rather than deleted, so restoring it is a one-line change if a
    later corpus check finds it after all:
        "No signs of bone atrophy."
    """
    if not atrophy:
        return []
    present = atrophy.get("atrophy")
    if present is not True:
        return []
    # REAL SNIPPET pattern minus the severity adjective (see above).
    return [f"Atrophy of the {arch_adj} bone."]


# ── Canals: NO merge, always two independent statements ────────────────────

# The buccolingual position is stated on EVERY canal sentence, and falls back
# to lingual when NEITHER reader placed the canal.
#
# "Neither" is two readers as of schema v6.9: the 3D read (mandible_canal_
# {side}.location) and the per-molar composite read
# (tooth_{fdi}_mandible_canal.location, off the coronal row). postprocess_pred
# reconciles them into this one `location` -- the molar reads win where they
# answer, see PREFER_COMPOSITE_CANAL_LOCATION there -- so nothing changes in
# this renderer beyond the fallback firing less often. The measured prior
# below is still the right one for the cases where it does fire: it was taken
# against the references, not against any particular reader.
#
# Two axes share this sentence and the reports keep them apart: "regular
# course" is the anomaly judgement, "predominantly lingual" is where the canal
# sits buccolingually. schema v6.4's `location` covers only the second, so the
# regular-course frame stays unconditional and the position rides beside it
# rather than replacing it. That is the corpus form verbatim -- "The right
# mandibular canal has a regular course, predominantly lingual".
#
# Defaulting an unread location is a prior, not a guess. Of the 70 sides the
# validate references place buccolingually, 61 are lingual, 5 buccal and 4
# central (which the enum cannot express at all). The VLM has no discriminative
# power on this axis either -- it scored 0 of 5 on the buccal sides and had
# already answered "lingual" on four of them -- so the default costs nothing it
# was getting right and recovers the sides it left unanswered.
DEFAULT_CANAL_LOCATION = "lingual"


def _adjacent_teeth_clause(entry: Dict) -> str:
    """
    ', with close relationship with the roots of teeth 47 and 48' -- or '' when
    the canal has no adjacent teeth.

    The location used to ride in here, as 'running lingually in close
    relationship with ...', because a bare position read as an anomaly claim
    the enum does not make. render_canal now states it on every canal sentence
    inside the regular-course frame, which removes that reading, so repeating
    it here would only say the same thing twice.
    """
    teeth = entry.get("adjacent_teeth")
    if not teeth:
        return ""
    noun = "the root of tooth" if len(as_list(teeth)) == 1 else "the roots of teeth"
    return f", with close relationship with {noun} {join_list(teeth)}"


def render_canal(entry: Dict) -> str:
    # REAL SNIPPET. Singular "canal" -- this sentence describes ONE side.
    # "Right mandibular canal with a regular course." is the corpus form (45
    # occurrences, the most frequent canal sentence in the training set), and
    # the references that also place the canal extend that same frame with
    # ", predominantly lingual" rather than replacing it.
    side_word = entry["side"].capitalize()
    location = entry.get("location")
    if location not in ("lingual", "buccal"):
        location = DEFAULT_CANAL_LOCATION
    return (f"{side_word} mandibular canal with a regular course, "
            f"predominantly {location}{_adjacent_teeth_clause(entry)}.")


def render_canals(canals: Optional[Dict]) -> List[str]:
    # Absent key: the mandibular bone was not in the scan volume, so the canal
    # running through it wasn't either -- postprocess_pred omits the fact.
    if not canals:
        return []
    return [render_canal(canals[side]) for side in ("right", "left") if canals.get(side)]


# ── Periodontal bone resorption ─────────────────────────────────────────────

def render_periodontal_bone_resorption(pbr: Dict) -> str:
    """
    SILENCED -- returns "" so the caller emits nothing.

    Measured on outputs/aksssr_v6_facts_validate, per case, against the
    reference reports: 7 cases got this sentence, the references discuss
    periodontal status in 12, and only 3 of ours land in one of those --
    F015, F036, P112 and S0021 assert periodontal disease in cases whose
    reports never raise the subject. Within the 3 that do overlap the
    sentence still has to get `extent` (mild/moderate/severe) and the bone
    loss pattern (horizontal/vertical) right to be entailed, so 3 is the
    ceiling and not the score.

    Silencing costs little recall that was ever being earned: 9 of the 12
    cases where the references discuss periodontal status got no sentence
    from us anyway.

    Note the fact is built for BOTH arches but only the maxilla ever reached
    the text -- render_report calls this once, from the maxilla branch. The
    mandible's periodontal_bone_resorption has always been summary-only.

    The sentence, kept verbatim so re-enabling is restoring the body:
        "Findings consistent with {extent} periodontal disease[, with
         {horizontal|vertical} bone loss]."
    """
    return ""


# ── Dental arch: absent teeth, primary teeth ────────────────────────────────

def render_absent_teeth(absent: Dict, quadrant_lo_hi: tuple,
                        arch_adj: str = "", atrophy: bool = False) -> Optional[str]:
    """
    quadrant_lo_hi = ((lo1,hi1),(lo2,hi2)) e.g. mandible ((31,38),(41,48)).
    pattern=='none' -> REAL SNIPPET: "Teeth from 48 to 38 are present in the
    arch." -- ONE range sweeping right-to-left across the midline, not two
    quadrant ranges. The corpus never splits a full arch in two: it writes
    "Teeth from 18 to 28 are present in the arch." (8), "...17 to 27..." (6),
    "...48 to 38..." (5), and "Teeth from N to N are present in the arch" is
    the top present-teeth template overall (25 of 436 such sentences, in 299
    reports). The previous "Teeth 31 to 38 and 41 to 48 are present." matched
    no reference at all -- it lacked "from", lacked "in the arch", and split
    the arch by quadrant.
    Any absence -> REAL SNIPPET: "Absence of {}." filled with the teeth list
    carrying its own noun -- "Absence of tooth 38." for one, "Absence of teeth
    38 and 48." for several -- and, when the caller passes `atrophy`, closed
    with ", with associated bone atrophy" instead of the period. That clause
    is where the arch's atrophy finding is stated; there is no separate
    sentence for it (see the note above is_complete_edentulism).
    Nothing present at all -> "Complete {arch} edentulism with marked
    atrophy." -- the same merge in the frame complete edentulism uses; without
    `atrophy` the sentence is "Complete {arch} edentulism." See
    edentulism_sentence on why the grade is fixed text. arch_adj defaults from
    the FDI quadrants, so an older two-argument call still names the arch
    correctly.
    pattern=='unknown' -> NO SENTENCE. Nothing was ever read about this
    arch's dentition (see classify_absent_teeth_pattern), and the "present"
    sentence above is a positive assertion about all 16 teeth -- exactly the
    claim we cannot make. Silence is the only honest rendering.
    pattern=='unassessable' -> NO SENTENCE either, for the stronger version of
    the same reason: the arch was not in the volume, so neither the absence
    nor the presence sentence can be earned. The scope sentence says it
    instead (render_maxilla_scope), and render_maxilla_main returns before
    reaching this for such an arch -- this branch is the belt to that brace.
    """
    pattern = absent.get("pattern")
    (lo1, hi1), (lo2, hi2) = quadrant_lo_hi
    # FDI quadrants 3 and 4 are the mandible, 1 and 2 the maxilla.
    arch_word = arch_adj or ("mandibular" if str(lo1)[0] in "34" else "maxillary")

    if pattern in ("unknown", "unassessable"):
        return None

    if pattern == "none":
        # Sweep from the RIGHT quadrant's 8 to the LEFT quadrant's 8 -- FDI
        # quadrants 1 and 4 are the patient's right, 2 and 3 the left. So
        # maxilla ((11,18),(21,28)) -> "18 to 28" and mandible
        # ((31,38),(41,48)) -> "48 to 38", both matching the corpus.
        first_is_right = str(lo1)[0] in "14"
        right, left = (hi1, hi2) if first_is_right else (hi2, hi1)
        return f"Teeth from {right} to {left} are present in the arch."

    if is_complete_edentulism(absent):
        return edentulism_sentence(arch_word, atrophy)

    # Every other pattern (consecutive_run, scatter, nearly_edentulous,
    # quadrant_edentulous_one, quadrant_edentulous_one_plus_partial) --
    # only "Absence of {}." was given as the real captured snippet for the
    # general "some teeth absent" case. Fill it with the clearest available
    # list of what's actually absent for that pattern. INFERRED which list
    # to use per pattern; the sentence frame itself is the real snippet.
    if pattern == "consecutive_run":
        teeth = absent.get("teeth", [])
    elif pattern == "scatter":
        teeth = absent.get("teeth", [])
    elif pattern == "nearly_edentulous":
        # 'remaining' is what's PRESENT, not absent -- invert against the
        # quadrant's full FDI set for the absence statement.
        all_fdis = set(range(lo1, hi1 + 1)) | set(range(lo2, hi2 + 1))
        teeth = sorted(all_fdis - set(as_list(absent.get("remaining"))))
    elif pattern in ("quadrant_edentulous_one", "quadrant_edentulous_one_plus_partial"):
        quad = absent.get("quadrant")
        quad_lo, quad_hi = (lo1, hi1) if quad in ("I", "III") else (lo2, hi2)
        teeth = list(range(quad_lo, quad_hi + 1))
        if pattern == "quadrant_edentulous_one_plus_partial":
            teeth = sorted(set(teeth) | set(as_list(absent.get("other_quadrant_absent"))))
    else:
        teeth = absent.get("teeth", []) or absent.get("present", [])

    if not teeth:
        return None
    # REAL SNIPPET, plus the atrophy clause in place of the period.
    #
    # "bone atrophy", NOT "{mandibular|maxillary} atrophy" (2026-08-17). The
    # arch is already named by the section this sentence sits in -- the line
    # opens "Mandible: ..." or "Maxilla: ..." -- so the adjective repeated the
    # one thing the reader cannot be in doubt about, and it made a claim that
    # reads as arch-scoped where the corpus states the finding of the bone
    # itself ("Atrophy of the mandibular bone", "atrophy of the alveolar
    # processes"). `arch_word` is still used for the complete-edentulism frame
    # above, where there is no absence list to carry the arch.
    sentence = f"Absence of {teeth_word(teeth).lower()} {join_list(teeth)}"
    if atrophy:
        sentence += ", with associated bone atrophy"
    return sentence + "."


def render_primary_teeth(primary: Optional[List[int]]) -> Optional[str]:
    if not primary:
        return None
    # REAL SNIPPET
    return f"Still present are the deciduous teeth {join_list(primary)}."


# ── Root remnants ────────────────────────────────────────────────────────────

def render_root_remnants(remnants: List[Dict]) -> List[str]:
    """
    SILENCED -- root remnants never reach the report.

    Same treatment, and the same reasoning, as render_restoration_summary:
    a finding measured to be wrong more often than right is not worth
    asserting. The fact stays in the summary JSON (root_remnants) so
    survey_findings.py can keep measuring it and the sentence can come back
    if the read improves.

    Measured on outputs/aksssr_v6_facts_validate against the reference
    reports, per case (survey_findings.py does not cover this finding,
    so these came from a concept screen of report vs reference text):
        107 root-remnant teeth claimed, across 35 of 40 cases;
        the references mention a root remnant in 10 cases;
        82 of the 107 claims sit in cases whose references never mention one
        at all -- unentailed regardless of which tooth was named.

    It was the largest single sentence class in the report: 107 of 681
    sentences, 15.7%, each carrying a second claim in its resorption clause
    (88 "without", 19 "with"), so roughly 200 claims, most of them false.

    Why it is this bad: build_root_remnants reads morphology.is_remnant and
    nothing else. Schema v6.1 retired the arch-level root_remnant_{arch}
    fact, so unlike endodontic or fillings there is no second source and no
    cross-source vote -- one composite misread reaches the text unfiltered.
    Same structural weakness as crown.

    The sentence, kept verbatim so re-enabling is restoring the body:
        "A likely root remnant of {fdi} is present, {with|without}
         associated periodontal bone resorption."
    """
    return []


# ── Impacted teeth ───────────────────────────────────────────────────────────

def render_impacted_teeth(impacted: List[Dict],
                          arch_findings: Optional[Dict] = None) -> List[str]:
    """
    THE FACT OF IMPACTION ONLY, FOR WISDOM TEETH THAT ARE PRESENT -- one
    grouped sentence, no direction and no orientation:

        Tooth 38 is impacted.  /  Teeth 38 and 48 are impacted.

    Three restrictions, all per explicit instruction:

    1. NO ANGLE, NO ORIENTATION. The impaction angle (mesial/distal/
       horizontal/vertical) and the buccal/lingual orientation are not
       rendered. Both stay in the summary JSON; they just never reach the
       report, like endodontic filling quality and the non-carious
       morphology findings.

    2. WISDOM TEETH ONLY (18/28/38/48). Impaction is a third-molar finding
       in this corpus, and it is the only tooth group the pipeline reads it
       for with any corroboration: the dedicated 3D wisdom-tooth facts and
       the panoramic arch read both cover 18/28/38/48, while the composite
       per-tooth read that was the ONLY source for a canine or premolar was
       retired from the schema (see the wisdom-tooth facts' description).
       An impaction claim on a non-wisdom tooth therefore rests on a single
       uncorroborated read, and is dropped here rather than stated.

    3. PRESENT TEETH ONLY. An absent tooth cannot be impacted, and saying it
       is contradicts the absence sentence rendered a few parts earlier in
       the same line. postprocess_pred.py's build_impacted_teeth already
       applies "absence outranks impaction" against its own absence set;
       this is the same rule re-applied against the arch findings actually
       being rendered, so the two sentences in one report line can never
       disagree. When the arch's dentition was never read (pattern
       "unknown", no present list), presence is unknown rather than false --
       only teeth listed as explicitly absent are dropped.

    The wording is the corpus's own: "Tooth N is impacted." is the most
    frequent impaction statement in dataset/training/reports (8 occurrences),
    with "Teeth N and N are impacted." (4) as its plural. Teeth-first is
    correct here specifically -- impaction is the one finding category where
    the corpus puts the tooth before the finding (58.9%), the reverse of the
    74.4% finding-first default everywhere else.
    """
    teeth = [entry["tooth"] for entry in impacted
             if entry.get("tooth") is not None and entry["tooth"] in WISDOM_TEETH]
    teeth = [fdi for fdi in teeth if not is_absent_tooth(fdi, arch_findings)]
    if not teeth:
        return []
    # REAL SNIPPET: "Tooth 38 is impacted." / "Teeth 38 and 48 are impacted."
    teeth = sorted(set(teeth))
    verb = "is" if len(teeth) == 1 else "are"
    return [f"{teeth_word(teeth)} {join_list(teeth)} {verb} impacted."]


# ── Wisdom teeth (eruption state only) ───────────────────────────────────────

_ERUPTION_PHRASE = {
    "not_erupted": "unerupted",
    "partially_erupted": "partially erupted",
}


def render_wisdom_teeth(wisdom: List[Dict], dentition_type: Optional[str] = None) -> List[str]:
    """
    Schema v6.1 added four dedicated wisdom-tooth facts (an independent 3D
    read of 18/28/38/48), and eruption state is the part of them that
    impaction doesn't already cover. postprocess_pred.py only keeps the
    teeth that are present and NOT fully erupted, so every entry here is a
    finding worth a sentence.

    RENDERED FOR MIXED DENTITION ONLY, per explicit instruction this round.
    In a permanent dentition an unerupted third molar is an unremarkable
    incidental -- it is the wisdom teeth's normal state in a large share of
    adults, and stating it spends a sentence asserting something the
    reference reports do not bother to say. In a MIXED dentition it is part
    of the eruption picture the report is actually about, so it is kept there
    (and only there: the preamble already covers the primary case with "Full
    primary dentition, permanent teeth unerupted", and an edentulous arch has
    no third molar to describe).

    Impaction is NOT affected -- render_impacted_teeth is a separate finding
    and still fires in every dentition. A wisdom tooth that is both impacted
    and unerupted keeps its impaction sentence and loses only the eruption
    one, outside mixed dentition.

    REAL SNIPPET: "Tooth 48 is unerupted." -- the corpus's own wording, and
    teeth-first like the impaction sentence next to it. Teeth sharing a state
    are grouped into one sentence, same as impaction.
    """
    if dentition_type != "mixed":
        return []

    by_state: Dict[str, List] = {}
    for entry in wisdom:
        phrase = _ERUPTION_PHRASE.get(entry.get("eruption_state"))
        if phrase and entry.get("tooth") is not None:
            by_state.setdefault(phrase, []).append(entry["tooth"])

    out = []
    for phrase in ("unerupted", "partially erupted"):
        teeth = sorted(by_state.get(phrase, []))
        if not teeth:
            continue
        verb = "is" if len(teeth) == 1 else "are"
        out.append(f"{teeth_word(teeth)} {join_list(teeth)} {verb} {phrase}.")
    return out


# ── Tooth findings (morphology / endodontic / restoration groupings) ───────

def render_endodontic_summary(endo: Dict) -> List[str]:
    """
    The treated teeth, and nothing about HOW they were treated.

    endodontic_summary.quality_groups (adequate / inadequate / overfilled /
    discontinuous) is deliberately NOT rendered -- per explicit instruction
    this round, every filling-quality sentence is dropped, whether the
    quality is normal or abnormal. The fact is still carried in the summary
    JSON for anything downstream that wants it; it just never reaches the
    report text.
    """
    teeth = as_list(endo.get("teeth"))
    if not teeth:
        return []
    # REAL SNIPPET
    return [f"Endodontic treatment involving teeth {join_list(teeth)}."]


# Fillings reach the report again as of 2026-08-16; crown and post-and-core do
# not. The flag exists so the change is one line to undo, and because the
# sentence is only safe when THE RULE -- fillings has re-sourced the group:
# see render_restoration_summary.
RENDER_FILLINGS = True


def render_restoration_summary(restorations: Dict,
                               rules_applied: bool = False) -> List[str]:
    """
    FILLINGS ONLY -- crown and post-and-core never reach the report, and
    fillings reach it only when the source rules have run (`rules_applied`).

    All three groups were silenced together at first, the same treatment
    endodontic filling quality, the impaction angle and the non-carious
    morphology findings still get. They are all still carried in the summary
    JSON (restoration_summary.groups) for anything downstream that wants them.

    Measured on outputs/aksssr_v5_validate against the reference reports
    (survey_findings.py), which is why silence beats the sentence:
        crown          74 claims, 13 right (precision 0.18) -- and it is
                       the ONE finding with no second source, since the
                       panoramic arch read has no crown value, so nothing
                       filters it before it reaches the text;
        fillings       80 claims, 13 right (0.16);
        post_and_core  29 claims,  0 right -- not once correct in 40 cases.

    The three sentences this used to emit, kept verbatim so re-enabling is
    a matter of restoring the body (teeth_word() lowercased at each, since
    unlike its other call sites these are not sentence-initial):
        "Prosthetic crown(s) on tooth/teeth {FDIs}."
        "Post-and-core on tooth/teeth {FDIs}."
        "Conservative composite restoration(s) on tooth/teeth {FDIs}."

    FILLINGS ARE BACK, 2026-08-16, and the other two are not -- which is why
    this function no longer silences the group as a unit. The three were
    silenced together because they share a renderer, not because they share
    evidence: measured per source (docs/postprocess.md, THE RULE --
    fillings), the composite's own `with_fillings` scores 0.39 precision,
    which clears the >=0.35 bar every other reported finding meets, while
    crown (0.28/0.10) and post-and-core (0.07/0.24) do not.

    What made the shipped fillings number 0.17 was not the composite but the
    arch survey aliased onto it -- `ARCH_VALUE_ALIASES` maps `restoration` ->
    `filling`, turning 38 composite claims into 161. THE RULE drops that
    alias, so what reaches this function is the composite's list. RENDERING
    THIS WITHOUT THAT RULE PUTS 134 FALSE CLAIMS BACK IN THE TEXT: the
    sentence is therefore gated on `rules_applied` -- the caller's check that
    the summary carries `source_rules`, which is postprocess_pred.py's record
    that the rules ran. The gate FAILS CLOSED: a summary built without
    --facts-dir renders no filling sentence at all, which is the pre-rules
    behaviour and the safe one.
    """
    if not (RENDER_FILLINGS and rules_applied):
        return []
    groups = restorations.get("groups") or {}
    fillings = [f for f in as_list(groups.get("fillings")) if isinstance(f, int)]
    if not fillings:
        return []
    # REAL SNIPPET, restored verbatim.
    return [f"Conservative composite restoration(s) on "
            f"{teeth_word(fillings).lower()} {join_list(sorted(fillings))}."]


def render_morphology_findings(tooth_findings: List[Dict]) -> List[str]:
    """
    SILENCED -- morphology contributes nothing to the report.

    Caries was the last finding this rendered; root fracture had already been
    dropped, and v6.4 retired crown_morphology='defect' and root_morphology=
    'resorption' from the schema. With caries silenced the function emits
    nothing at all, and is kept only so the ordering in render_tooth_findings
    stays intact.

    Measured on outputs/aksssr_v6_facts_validate, per case, against the
    reference reports: 6 cases got a caries sentence, the references mention
    caries in 9, and exactly ONE of ours falls in one of those. A034, A041,
    A097, F067 and P112 assert destructive caries in cases whose reports
    never mention caries at all -- 5 of 6 unentailed before tooth numbers are
    even compared.

    Silencing costs almost no recall: 8 of the 9 cases whose references
    discuss caries got no sentence from us anyway.

    The sentence, kept verbatim so re-enabling is restoring the body
    (teeth_word() lowercased, since this is not sentence-initial):
        "Destructive caries involving tooth/teeth {FDIs}."

    The wording, if it does come back, was chosen against
    dataset/training/reports: "Destructive caries involving tooth N" is the
    corpus's most frequent caries template (4 occurrences, ahead of
    "Destructive caries of tooth N" (3), "Coronal caries of tooth N" (3) and
    "Caries involving tooth N" (3)); finding-first matches the corpus's
    dominant order for tooth findings (74.4% of 3967 tooth-referencing
    sentences, 93.4% for lesions); and no hedge, since with_caries is a plain
    bool with no confidence field. "Destructive" is a severity the fact does
    not distinguish -- it was chosen for n-gram overlap, not accuracy.
    """
    return []


def render_tooth_findings(tooth_findings: List[Dict], endo_summary: Optional[Dict],
                          restoration_summary: Optional[Dict],
                          rules_applied: bool = False) -> List[str]:
    """Assembles all tooth-level findings for one arch, in this order:
    morphology -> endodontic -> restoration.

    The restoration call is kept even though two of its three groups are
    silenced: the ordering is what the report's sentence sequence depends on,
    so re-enabling one stays a one-function change. `rules_applied` is
    threaded from synthesize_report and gates fillings -- see
    render_restoration_summary.
    """
    out = []
    out.extend(render_morphology_findings(tooth_findings))
    if endo_summary:
        out.extend(render_endodontic_summary(endo_summary))
    if restoration_summary:
        out.extend(render_restoration_summary(restoration_summary, rules_applied))
    return out


# ── Prosthetics ──────────────────────────────────────────────────────────────

def _implant_position(imp: Dict) -> str:
    """
    Where the implant is. v6.1's implant entries carry a single `fdi_number`
    plus a free-text `location` ("position 35"), replacing the v4 `fdi_numbers`
    list -- the FDI is preferred, the location string is the fallback for an
    implant the model could only place descriptively, and an entry with
    neither never gets here (build_implants drops it).
    """
    fdi = imp.get("fdi_number")
    if fdi is not None:
        return str(fdi)
    location = str(imp.get("location") or "").strip()
    # "position 35" already reads as a position -- don't say it twice.
    return re.sub(r"^\s*positions?\s+", "", location, flags=re.IGNORECASE)


def render_implants(implants: List[Dict]) -> List[str]:
    """
    ONE SENTENCE PER CLAUSE SET, not per implant. v6.1 gives one entry per
    implant, and rendering each one separately produced the repetition the
    reference reports never have -- "Endosseous implant in position 45.
    Endosseous implant in position 46." where a radiologist writes a single
    sentence naming both. Implants are therefore grouped by their trailing
    clauses (crown, osseointegration) and their positions listed together.

    Grouping is by clause set and NOT by arch or by position range: the
    clauses are what the sentence asserts about every position it names, so
    merging two implants that differ in them would attach a crown or an
    osseointegration verdict to an implant that was never given one. Arch
    needs no handling here -- render_prosthetics is already called once per
    arch section, so this list is single-arch by construction. Group order
    follows first appearance, and positions keep the order they arrived in
    (already FDI-ascending out of build_implants, and a prose-only location
    has no number to sort on anyway).
    """
    groups: List[Tuple[Tuple[str, ...], List[str]]] = []
    for imp in implants:
        positions = _implant_position(imp)
        if not positions:
            continue
        clauses = []
        if imp.get("with_crown"):
            # REAL SNIPPET clause
            clauses.append("restored with an implant-supported crown")
        status = imp.get("osseointegration_status")
        if status == "poor":
            # REAL SNIPPET clause
            clauses.append("with signs of peri-implant bone resorption")
        elif status == "well":
            # REAL SNIPPET clause
            clauses.append("with radiographically well osseointegration")
        key = tuple(clauses)
        for existing_key, existing_positions in groups:
            if existing_key == key:
                existing_positions.append(positions)
                break
        else:
            groups.append((key, [positions]))

    out = []
    for clauses, positions in groups:
        # REAL SNIPPET base, pluralised when the group carries several.
        noun = "Endosseous implant" if len(positions) == 1 else "Endosseous implants"
        sentence = f"{noun} in position {', '.join(positions)}"
        if clauses:
            sentence += ", " + ", ".join(clauses)
        out.append(sentence + ".")
    return out


def render_bridges(bridges: List[Dict]) -> List[str]:
    """
    A SPANLESS ENTRY IS THE MASK'S, and gets its own sentence. THE RULE --
    fixed bridges sources presence and arch from the segmentation's bridge
    label, which carries no span: the label marks the pontic, not the
    abutments, and closing the span by dilation lands +/-1 position with no
    consistent sign. The VLM's own spans are worse -- 0 of 16 exact against
    the reference, and 0 of 2 among those that survive a per-arch gate -- so
    there is nothing to fall back on either. Naming no span is the honest
    output, not a degraded one.
    """
    out = []
    for b in bridges:
        if b.get("span_start") is None or b.get("span_end") is None:
            # REAL SNIPPET shape, minus every clause the mask cannot fill.
            out.append("Prosthetic bridge exists.")
            continue
        clauses = []
        if b.get("abutment_teeth"):
            # REAL SNIPPET pattern
            clauses.append(f"anchor abutment teeth on {join_list(b['abutment_teeth'])}")
        if b.get("implant_supported_teeth"):
            fdis = as_list(b["implant_supported_teeth"])
            region_word = "region" if len(fdis) == 1 else "regions"
            noun = "an implant" if len(fdis) == 1 else "implants"
            clauses.append(f"{noun} in the {join_list(fdis)} {region_word}")
        pontics_clause = ""
        if b.get("pontic_teeth"):
            fdis = as_list(b["pontic_teeth"])
            region_word = "region" if len(fdis) == 1 else "regions"
            noun = "a pontic" if len(fdis) == 1 else "pontics"
            pontics_clause = f", and {noun} in the {join_list(fdis)} {region_word}"
        # REAL SNIPPET
        sentence = (f"Fixed prosthetic bridge from {b['span_start']} to {b['span_end']}, "
                   f"with {', and '.join(clauses)}{pontics_clause}.")
        out.append(sentence)
    return out


def render_prosthetics(prosthetics: Dict) -> List[str]:
    out = []
    out.extend(render_implants(prosthetics.get("implants", [])))
    out.extend(render_bridges(prosthetics.get("bridges", [])))
    return out


# ── Bone quality ─────────────────────────────────────────────────────────────

def render_bone_quality(bq: Optional[Dict]) -> List[str]:
    # Absent key (not merely present=false): the arch's bone was not in the
    # scan volume at all, so postprocess_pred omitted the fact rather than
    # letting it assert a negative about unimaged anatomy. No sentence.
    if bq is None:
        return []
    if not bq.get("present"):
        # REAL SNIPPET, verbatim -- and the single most frequent sentence of
        # any kind in dataset/training/reports: 70 occurrences, across 208 of
        # the 933 reports (22.3%).
        #
        # No trailing verb. 84.2% of the corpus's 215 bone-lesion negations
        # are a bare noun phrase like this; "...are observed" is 1.9% and the
        # participle form we used to emit ("lesions observed") occurs zero
        # times. Both slots match the majority too: "definite" (48.8%, vs
        # none 41.4% / "frank" 4.7% / "evident" 4.7%) and the pair
        # "osteolytic or osteocondensing" (54.0%, vs "radiolucent or
        # radiopaque" 20.9% / "osteolytic or osteosclerotic" 14.0%).
        return ["No definite osteolytic or osteocondensing lesions."]

    # A CLAIMED LESION SAYS NOTHING AT ALL -- neither the finding nor the
    # negation. The positive claims do not survive inspection: on A008 the
    # mandible's own visual_evidence hedges its "radiolucent area" as "likely
    # corresponding to the mandibular canal", i.e. normal anatomy, and files it
    # as a lesion regardless. Reporting that invents pathology.
    #
    # But the negative cannot be substituted either: "No definite osteolytic or
    # osteocondensing lesions." is an assertion, and asserting it over a read
    # that flagged something is not silence, it is a contradiction of the only
    # source consulted. The arch's bone-quality sentence is dropped instead,
    # which is the one option that claims nothing in either direction -- the
    # same treatment build_bone_quality's absent-key case already gets.
    #
    # This is deliberately not a precision/recall trade to be tuned: the
    # positives are unverifiable and the negatives would be unearned. If a
    # future schema gives bone quality a second source to cross-check against,
    # the positives become gateable and this can be revisited.
    #
    # THAT SECOND SOURCE NOW EXISTS: tooth_{fdi}_bone_quality, and
    # postprocess_pred.build_bone_quality gates the arch claim on it -- a claim
    # no tooth in the arch confirms arrives here with findings=[], so anything
    # still carrying a finding has been corroborated. Unlocking the positive
    # sentence is therefore now possible and is a deliberate report-content
    # decision, not a code fix; it stays silent until the per-tooth source's
    # precision has been measured the way the gated findings were
    # (survey_findings.py). Until then the gate only reaches the summary
    # JSON, which the LLM report arm reads.
    return []


# ── Condyles (Type C, always stated) ────────────────────────────────────────

# What the whole section collapses to when neither condyle made it into the
# acquisition -- wording set by explicit instruction this round (it replaces
# the earlier "Excluded from the acquisition and not visible." snippet, which
# never named what was excluded).
BOTH_CONDYLES_EXCLUDED = "Condyles not included in the scan volume."

# The other half of the pair, same shape. The summary's condyle vocabulary is
# binary (postprocess_pred.merge_condyle_scopes), so "included" carries no claim
# about HOW MUCH is in the volume and the sentence stops there.
BOTH_CONDYLES_INCLUDED = "Condyles included in the scan volume."


def render_condyles(condyles: Dict) -> List[str]:
    """
    ONE sentence for both sides, always. The per-side sentences this used to
    emit ('Right mandibular condyle is partially included in the scan volume.')
    are gone: postprocess forces the two sides to one value, and the corpus
    states them together anyway -- 971 of the 978 reference reports that mention
    the condyles use no laterality word, and only 3 in 1000 assert a different
    scope per side.

    Returning a LIST of at most one sentence keeps the caller
    (render_mandible_condyle, which joins it) and the empty case -- no answer on
    either side -- exactly as they were. A pre-fold summary written by an older
    run still renders: `fully_included`/`partially_included` are read as
    `included` here too, rather than falling through to "not".
    """
    scopes = {entry.get("scope") for side in ("right", "left")
              if isinstance(entry := condyles.get(side), dict)}
    scopes.discard("unknown")
    scopes.discard(None)
    if not scopes:
        # No inclusion answer came back on either side, and inclusion is the
        # ONLY thing the condyle fact carries -- there is no shape left to fall
        # back on, so the block has nothing to state.
        return []
    # not_included wins a residual disagreement, for the same reason it wins in
    # merge_condyle_scopes; a summary built by this pipeline can no longer hold
    # one, but a hand-edited or pre-merge summary still renders sensibly.
    if "not_included" in scopes:
        return [BOTH_CONDYLES_EXCLUDED]
    return [BOTH_CONDYLES_INCLUDED]


# ── Maxillary sinuses ────────────────────────────────────────────────────────

_SINUS_INCLUSION_WORD = {
    "fully_included": "included",
    "partially_included": "partially included",
    "not_included": "not included",
}


# REAL SNIPPET, and the single most common sinus sentence in the corpus (40
# occurrences). When nothing was in the acquisition this is the WHOLE section
# -- no content sentence, no intrasinusal-teeth sentence.
SINUS_NOT_ASSESSABLE = ("Maxillary sinuses: not included in the acquisition "
                        "volume and not assessable.")


def _pair_inclusion_word(scopes: Dict) -> Optional[str]:
    """
    One inclusion word for the pair: "partially included" if EITHER side is
    only partially in (the weaker claim governs), "included" when every side
    with a known scope is fully in. None when neither side has a scope, which
    is the one case with nothing truthful to say about inclusion.
    """
    known = [s for s in (scopes.get("right"), scopes.get("left"))
             if s in ("fully_included", "partially_included")]
    if not known:
        return None
    return "included" if all(s == "fully_included" for s in known) else "partially included"


def render_sinuses(sinuses: Optional[Dict]) -> List[str]:
    """
    Scope is ALWAYS stated (Type C). Per explicit instruction this round the
    normal-content case is folded INTO the scope sentence rather than getting
    its own -- "included in the acquisition volume, normal aerated" -- while
    an abnormal sinus keeps its own following sentence.
    """
    if not sinuses:
        return []
    status = sinuses.get("group_status")
    scopes = sinuses.get("scope", {})

    # Nothing in the volume: this sentence is the entire section.
    if status == "none_included":
        return [SINUS_NOT_ASSESSABLE]

    word = _pair_inclusion_word(scopes)
    scope_clause = f"Maxillary sinuses are {word} in the acquisition volume" if word else None

    if status == "all_normal":
        if scope_clause:
            return [f"{scope_clause}, normally aerated."]
        # No scope was answered -- fall back to describing the content alone
        # rather than inventing an inclusion status. (REAL SNIPPET)
        return ["The maxillary sinuses appear normally pneumatized and clear, "
                "with no evidence of mucosal thickening or signs of sinusopathy."]

    lead = [scope_clause + "."] if scope_clause else []

    if status == "mixed":
        right_abn = not sinuses["right"]["normal"]
        left_abn = not sinuses["left"]["normal"]
        if right_abn and left_abn:
            # REAL SNIPPET
            return lead + ["The maxillary sinuses show bilateral mucosal thickening."]
        if right_abn:
            # REAL SNIPPET
            return lead + ["The maxillary sinus shows mild mucosal thickening on the right side."]
        if left_abn:
            # REAL SNIPPET
            return lead + ["The maxillary sinus shows mild mucosal thickening on the left side."]

    # right_only / left_only -- one side in the volume, the other out. Both
    # sides are still accounted for: the in-scope one by name, the excluded
    # one by the not-assessable clause. INFERRED (no real snippet covers this
    # single-side edge case).
    for status_name, side, other in (("right_only", "right", "left"),
                                     ("left_only", "left", "right")):
        if status != status_name:
            continue
        side_word = _SINUS_INCLUSION_WORD.get(scopes.get(side))
        opening = (f"{side.capitalize()} maxillary sinus is {side_word} in the "
                   f"acquisition volume" if side_word
                   else f"{side.capitalize()} maxillary sinus")
        excluded = (f"the {other} maxillary sinus is not included in the "
                    f"acquisition volume and not assessable")
        if sinuses[side]["normal"]:
            return [f"{opening}, normally aerated; {excluded}."]
        return [f"{opening}; {excluded}.",
                f"The {side} maxillary sinus shows mild mucosal thickening."]
    return lead


def render_intrasinusal_teeth(entry: Dict) -> str:
    teeth = as_list(entry.get("teeth"))
    if not teeth:
        # No real snippet was given for the explicit negative -- INFERRED,
        # matching the style of the v2/v3 template's equivalent statement.
        return "No pathological relationships are detected between the dental apices and the floor of the maxillary sinuses."
    if len(teeth) == 1:
        # REAL SNIPPET
        return f"Root of {teeth[0]} intrasinusal."
    # REAL SNIPPET
    return f"Roots of {join_list(teeth)} intrasinusal."


# NO nasal-cavity renderer: schema v6.1 dropped the nasal_cavity fact, so
# postprocess_pred.py no longer emits a nasal_cavity block for it to render.


# ── Arch section assembly ─────────────────────────────────────────────────
#
# FOUR blocks: Mandible, Maxilla, Maxillary sinuses, Mandibular condyles --
# condyles and sinuses pulled out of their arch's main paragraph into their
# own blocks. Each arch's MAIN block still ends with bone_quality; condyle/
# sinus content no longer follows it in the same paragraph.
#
# NO blank lines anywhere in the output. Under the inline-label style a main
# block is ONE line -- its three parts are joined by a space, and the four
# blocks are joined by a single "\n" in synthesize_report():
#   Scope
#   Atrophy -> [Periodontal, maxilla only] -> Dental Arch -> Tooth Findings -> Prosthetics
#   bone_quality

def render_mandible_main(m: Dict, dentition_type: Optional[str] = None,
                        rules_applied: bool = False) -> str:
    scope_line = render_mandible_scope(m["scope"])

    absent = m["arch_findings"]["absent_teeth"]
    edentulous = is_complete_edentulism(absent)
    atrophy = (m.get("alveolar_bone_atrophy") or {}).get("atrophy") is True
    absent_sentence = render_absent_teeth(absent, ((31, 38), (41, 48)),
                                          "mandibular", atrophy)

    middle_parts: List[str] = []
    # COMPLETE EDENTULISM LEADS. It is the state of the whole arch, not one
    # finding among several: everything after it -- the canals, whatever tooth
    # findings survive -- is read against an arch with no teeth in it. Every
    # other absence sentence stays where it was, after the canals, because it
    # is a list of positions and belongs beside the other per-tooth
    # statements. Either way it now CARRIES the atrophy finding rather than
    # being followed by a sentence for it, so there is no atrophy line here.
    if edentulous:
        middle_parts.append(absent_sentence)
    middle_parts.extend(render_canals(m.get("canals")))

    if absent_sentence and not edentulous:
        middle_parts.append(absent_sentence)
    primary_sentence = render_primary_teeth(m["arch_findings"].get("primary_teeth"))
    if primary_sentence:
        middle_parts.append(primary_sentence)
    middle_parts.extend(render_root_remnants(m.get("root_remnants", [])))
    middle_parts.extend(render_impacted_teeth(m.get("impacted_teeth", []),
                                              m["arch_findings"]))
    middle_parts.extend(render_wisdom_teeth(m.get("wisdom_teeth", []), dentition_type))

    middle_parts.extend(render_tooth_findings(m.get("tooth_findings", []),
                                              m.get("endodontic_summary"),
                                              m.get("restoration_summary"),
                                              rules_applied))

    if m.get("prosthetics"):
        middle_parts.extend(render_prosthetics(m["prosthetics"]))

    middle_line = " ".join(p for p in middle_parts if p)
    bone_quality_line = " ".join(render_bone_quality(m.get("bone_quality")))

    # Inline-label style: the whole block sits on ONE line after "Mandible: ",
    # so join with a space rather than a newline.
    return " ".join(p for p in [scope_line, middle_line, bone_quality_line] if p)


def render_mandible_condyle(m: Dict) -> str:
    return " ".join(render_condyles(m.get("condyles", {})))


def render_maxilla_main(x: Dict, dentition_type: Optional[str] = None,
                       rules_applied: bool = False) -> str:
    scope_line = render_maxilla_scope(x["scope"])

    # AN EXCLUDED MAXILLA REPORTS ITS SCOPE AND NOTHING ELSE (2026-08-17).
    # Every upper position in such a volume is unassessable -- see
    # postprocess_pred.build_maxilla_section -- so there is no finding to
    # state, and the scope sentence already says why. This replaces the
    # "As far as can be assessed, ..." hedge below, which let arch-level
    # reads through on an arch nobody could see: P063's maxilla is called
    # "partially included in the acquisition" by both its reports and this
    # renderer had it asserting "As far as can be assessed, complete
    # edentulism."
    #
    # THE COST, STATED: 10% of the reference reports for these cases do name
    # an upper tooth -- restorations and crowns mostly, 4.5% an absence, and
    # the absences cluster on 18/28 (code/studies/survey_upper_mentions.py). Those
    # sentences are now unreachable. The trade is deliberate: 77% of the same
    # reports say only what this sentence says, and the arch-level read that
    # would fill the other 10% is the one measured at 0.34 precision on the
    # fourteen non-wisdom upper positions.
    if x["scope"].get("level") == "none":
        return scope_line

    absent = x["arch_findings"]["absent_teeth"]
    edentulous = is_complete_edentulism(absent)
    atrophy = (x.get("alveolar_bone_atrophy") or {}).get("atrophy") is True
    absent_sentence = render_absent_teeth(absent, ((11, 18), (21, 28)),
                                          "maxillary", atrophy)

    middle_parts: List[str] = []
    # Complete edentulism leads here too, atrophy folded into the absence
    # sentence -- see render_mandible_main.
    if edentulous:
        middle_parts.append(absent_sentence)

    periodontal = x.get("periodontal_bone_resorption")
    if periodontal:
        middle_parts.append(render_periodontal_bone_resorption(periodontal))

    if absent_sentence and not edentulous:
        middle_parts.append(absent_sentence)
    primary_sentence = render_primary_teeth(x["arch_findings"].get("primary_teeth"))
    if primary_sentence:
        middle_parts.append(primary_sentence)
    middle_parts.extend(render_root_remnants(x.get("root_remnants", [])))
    middle_parts.extend(render_impacted_teeth(x.get("impacted_teeth", []),
                                              x["arch_findings"]))
    middle_parts.extend(render_wisdom_teeth(x.get("wisdom_teeth", []), dentition_type))

    middle_parts.extend(render_tooth_findings(x.get("tooth_findings", []),
                                              x.get("endodontic_summary"),
                                              x.get("restoration_summary"),
                                              rules_applied))

    if x.get("prosthetics"):
        middle_parts.extend(render_prosthetics(x["prosthetics"]))

    middle_line = " ".join(p for p in middle_parts if p)
    # ONE BONE-QUALITY SENTENCE PER REPORT, AND IT CLOSES THE MANDIBLE.
    # 2026-08-16. It used to close each arch independently -- a deliberate
    # decision recorded at the top of this file ("each arch's bone_quality
    # closes that arch's own section"), now overridden. The sentence is a
    # whole-jaw negation in the corpus rather than an arch-scoped one, and
    # emitting it per arch put 64 copies into 40 reports: 13% of all generated
    # text restating a single claim. The mandible carries it because that arch
    # is in the volume in every case in this dataset and the maxilla is not.
    #
    # The maxilla's bone_quality fact is still built and still in the summary
    # JSON; it simply has no sentence. If a volume ever lacks the mandible the
    # report carries no bone-quality line at all, which is the honest outcome
    # -- see render_bone_quality on why the negation is an assertion and not a
    # safe default.
    bone_quality_line = ""

    # "Maxilla is not included in the scan volume. Absence of 11, 12, 13..."
    # contradicts itself in one breath. The volume can hold the upper CROWNS
    # while stopping below the maxillary bone, which is what a not_included
    # arch with tooth findings means, and the corpus has a set phrase for
    # precisely that -- REAL SNIPPET, and the opener of every reference report
    # in this situation: "As far as can be assessed, absence from the arch of
    # teeth 13, 17, 18..." (S0007), "As far as can be visualized, prosthetic
    # crowns are present on teeth 14, 13, 12..." (P251).
    if middle_line and x["scope"].get("level") == "none":
        middle_line = ("As far as can be assessed, "
                       + middle_line[0].lower() + middle_line[1:])

    # Inline-label style: one line after "Maxilla: " -- join with a space.
    return " ".join(p for p in [scope_line, middle_line, bone_quality_line] if p)


def render_maxilla_sinus(x: Dict) -> str:
    """
    Scope/aeration only. The intrasinusal-teeth sentence ("Roots of 26 and 27
    intrasinusal." / its negative form) is NOT rendered -- dropped per
    explicit instruction this round, and confirmed to stay dropped once it was
    measured: 1 correct against 10 false positives across the validate split,
    precision 0.09 (survey_findings.py). The fact is still built into the
    summary JSON (sinus_intrasinusal_teeth); it just never reaches the report,
    exactly like endodontic filling quality and the restoration groups. The
    nasal-cavity sentence is gone for a different reason: v6.1 has no
    nasal_cavity fact at all.
    """
    return " ".join(render_sinuses(x.get("sinuses")))


# ── Top-level assembly ────────────────────────────────────────────────────

# The renderers build standalone sentences that name their own structure
# ("Maxillary sinuses are partially included..."). Under an inline label that
# name is already said, so the echo is trimmed off the front of the body:
# "Maxillary sinuses: partially included in the acquisition volume, ...".
# Only the FIRST sentence is trimmed -- a later sentence in the same block
# still needs its subject.
_LABEL_ECHO = {
    "Mandible":            r'(?:The\s+)?Mandible\s+(?:is\s+)?',
    "Maxilla":             r'(?:The\s+)?Maxilla\s+(?:is\s+)?',
    "Maxillary sinuses":   r'(?:The\s+)?Maxillary\s+sinuses(?:\s*:|\s+are|\s+)\s*',
    "Mandibular condyles": r'(?:The\s+)?(?:Mandibular\s+)?[Cc]ondyles\s+(?:are\s+)?',
}


def _strip_label_echo(label: str, body: str) -> str:
    """Drop a leading restatement of the section label from the block body."""
    if not body:
        return body
    pattern = _LABEL_ECHO.get(label)
    if not pattern:
        return body
    stripped = re.sub(rf'^{pattern}', '', body, count=1)
    return stripped if stripped else body

def synthesize_report(summary: Dict) -> str:
    """
    Full report in the corpus's INLINE-LABEL style -- the most common layout in
    dataset/training/reports (363/933 = 38.9% of reports):

        Mandible: <findings on the same line as the label>
        Maxilla: <findings>
        Maxillary sinuses: <findings>
        Mandibular condyles: <findings>

    Three conventions, each measured on the training corpus:
      * label and text share ONE line, separated by ": " (not a heading line);
      * labels are Title Case, matching the corpus spelling for the three
        anatomical blocks -- "Mandible" (452 reports), "Maxilla" (259),
        "Mandibular condyles" (167). ALL-CAPS labels are a different, rarer
        style (52 reports, 5.6%);
      * blocks are one per line, separated by a single "\\n". Only 19% of
        inline-label reports put a blank line between them.

    Section order is Mandible -> Maxilla -> Maxillary sinuses -> Mandibular
    condyles. The corpus's canonical three-block order is Mandible -> Maxilla
    -> Mandibular condyles (157 reports use exactly that; only 8 deviate); the
    sinus block is inserted after Maxilla. "Maxillary sinuses" is the corpus's
    own spelling in the 5 reports that give the sinuses a label of their own.
    """
    lines: List[str] = []

    dentition = summary.get("dentition_type", {}) or {}
    preamble = render_preamble(dentition)
    if preamble:
        lines.append(preamble)

    # Threaded into both arch renderers because the wisdom-tooth eruption
    # sentence is rendered for mixed dentition only -- see render_wisdom_teeth.
    dtype = dentition.get("dentition_type")

    # postprocess_pred.py writes `source_rules` only when it ran them, so this
    # is the record that THE RULE -- fillings re-sourced the fillings group.
    # Without it the fillings sentence would carry the arch survey's aliased
    # `restoration` claims -- 161 against the composite's 38, at 0.17
    # precision. See render_restoration_summary.
    # THE FILLINGS RULE SPECIFICALLY, not "the source rules ran at all". The
    # key is written by source_rules.apply() whenever a facts file was present,
    # INCLUDING when every rule is switched off by a config -- so testing the
    # key's presence let an ablation arm render the aliased fillings group, the
    # exact 134-false-claims case render_restoration_summary is gated against.
    # The `applied` list is what says which rules actually ran.
    _rules = summary.get("source_rules")
    rules_applied = (isinstance(_rules, dict)
                     and "fillings" in (_rules.get("applied") or []))

    blocks = [
        ("Mandible",            render_mandible_main(summary["mandible"], dtype,
                                                     rules_applied)),
        ("Maxilla",             render_maxilla_main(summary["maxilla"], dtype,
                                                    rules_applied)),
        ("Maxillary sinuses",   render_maxilla_sinus(summary["maxilla"])),
        ("Mandibular condyles", render_mandible_condyle(summary["mandible"])),
    ]
    for label, body in blocks:
        body = _strip_label_echo(label, body)
        if body:
            lines.append(f"{label}: {body}")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Render {case_id}_summary.json into the final CBCT dental report.")
    ap.add_argument("--summary-dir", required=True,
                    help="Directory of {case_id}_summary.json files, or a single such file")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for {case_id}.txt report files")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="The same configs/postprocess/*.yaml postprocess_pred.py "
                         "was given. It settles report.render_fillings and the "
                         "canal-location prior. PASS THE SAME FILE TO BOTH: the "
                         "summaries and the report text are one arm, and a "
                         "renderer running defaults over summaries built by an "
                         "ablation is a mixed arm with nothing in the output "
                         "saying so. infer.py passes it to both for this reason.")
    args = ap.parse_args()

    try:
        rules_config.load_and_apply(args.config)
    except rules_config.ConfigError as exc:
        ap.error(str(exc))

    summary_path = Path(args.summary_dir)
    if summary_path.is_file():
        summary_files = [summary_path]
    else:
        summary_files = sorted(summary_path.glob("*_summary.json"))

    if args.case_ids:
        summary_files = [p for p in summary_files if p.stem.replace("_summary", "") in args.case_ids]

    if not summary_files:
        print(f"[WARN] No summary files found under {summary_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for sf in summary_files:
        summary = json.loads(sf.read_text(encoding="utf-8"))
        report_text = synthesize_report(summary)
        case_id = summary.get("case_id", sf.stem.replace("_summary", ""))
        out_path = Path(args.out_dir) / f"{case_id}.txt"
        out_path.write_text(report_text, encoding="utf-8")
        print(f"[{case_id}] -> {out_path}")


if __name__ == "__main__":
    main()

""""
Usage:
    python3 code/pipeline/postprocess/synthesize_report.py \
        --summary-dir outputs/aksssr_v4_validate/summaries \
        --out-dir     outputs/aksssr_v4_validate/synthesized_reports
"""