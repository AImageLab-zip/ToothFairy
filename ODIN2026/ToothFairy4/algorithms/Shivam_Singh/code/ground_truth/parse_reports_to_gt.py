#!/usr/bin/env python3
"""
code/ground_truth/parse_reports_to_gt.py

Extracts structured ground truth from a radiologist's REFERENCE report
(free text) into the exact same JSON shape as {case_id}_pred.json --
{"case_id": ..., "global": {...}, "teeth": {...}} -- so
structured_findings_evaluation.py can compare prediction vs. ground truth field by
field with zero case-specific logic.

TWO STAGES, NOT ONE. THIS IS THE WHOLE DESIGN.
────────────────────────────────────────────────────────────────────────
  stage 1  LLM  report text -> {case}[_{reader}]_report_facts.json
                a SMALL, report-shaped intermediate: the lists and enums a
                radiology report actually states ("teeth 32,33,... are
                present; 31,37,41,46,47 are absent", "endodontic treatment
                in 35 and 36, adequate", "canal runs lingually").
  stage 2  CODE report_facts -> {case}[_{reader}]_gt.json
                deterministic expansion into the full schema shape: all 16
                arch positions, all 32 per-tooth fact blocks, the wisdom
                tooth facts, the quadrant implant counts.


NULL MEANS "THE REPORT DID NOT SAY"
────────────────────────────────────
Some fields have no neutral value to default to -- mandible_canal_*.
location is enum lingual|buccal, and a report that never describes the
canal's course supports neither. Those stay null rather than being filled
with a guess, and structured_findings_evaluation.py drops null-GT pairs from
exact-match scoring instead of counting the prediction wrong for
answering something the reference never claimed. Fields that DO have a
neutral value (extent "none", mucosa "normal", every bool, every list)
get it, which is what the report's silence actually means for them.

dentition_type is not a field on either side -- it is DERIVED by
postprocess_pred.build_dentition_type from primary_teeth + eruption, and
postprocess runs on the prediction side too, so the two agree by
construction. Nothing scores the field directly any more.

RE-EXPANDING WITHOUT THE GPU
─────────────────────────────
Stage 2 is pure CPU, so tuning it does not need a vLLM server:

    python code/ground_truth/parse_reports_to_gt.py --from-report-facts \\
        --reports-dir dataset/training/reports \\
        --schema schema/schema.json \\
        --out-dir dataset/training/outputs/ground_truth

re-derives every {case}_gt.json from the report_facts already on disk.
Same relationship scripts/postprocess_now.sh has to the VQA pipeline.

Usage:
    python code/ground_truth/parse_reports_to_gt.py \\
        --reports-dir dataset/training/reports \\
        --schema      schema/schema.json \\
        --out-dir     dataset/training/outputs/ground_truth \\
        --vllm-url    http://localhost:8001/v1 \\
        --model       qwen3-text
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
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

from normalize_pred import normalize_prediction, summarize_repairs  # noqa: E402

try:
    import requests
except ImportError:
    sys.exit("pip install requests")


ALL_FDIS = [11,12,13,14,15,16,17,18,
            21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38,
            41,42,43,44,45,46,47,48]

# Arch membership and the FIXED key order dental_arch_findings_{arch}.findings
# declares (48->41, 31->38 / 18->11, 21->28). Both are needed: the first to
# reject an FDI the extractor put in the wrong arch, the second because the
# schema calls that order part of the answer.
ARCH_FDIS = {
    "mandible": [48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38],
    "maxilla":  [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28],
}
ARCH_PRIMARY_FDIS = {
    "mandible": list(range(71, 76)) + list(range(81, 86)),
    "maxilla":  list(range(51, 56)) + list(range(61, 66)),
}
WISDOM_FDI = {"mandible": {"right": 48, "left": 38}, "maxilla": {"right": 18, "left": 28}}
# lower_LEFT_wisdom_tooth is 38 and lower_RIGHT is 48 (schema's own wording);
# upper_right is 18, upper_left is 28.
WISDOM_FACT = {38: "lower_left_wisdom_tooth", 48: "lower_right_wisdom_tooth",
               18: "upper_right_wisdom_tooth", 28: "upper_left_wisdom_tooth"}

ERUPTION_STATES = ("complete_bony_inclusion", "partially_erupted", "fully_erupted", "absent")
ORIENTATIONS = ("normal", "mesial", "distal", "lingual", "buccal")
FILLING_QUALITIES = ("adequate", "inadequate", "overfilled", "discontinuous")
BONE_LOSS_EXTENTS = ("none", "mild", "moderate", "severe")
RESORPTION_PATTERNS = ("horizontal", "vertical")
SCOPES3 = ("fully_included", "partially_included", "not_included")
SCOPES2 = ("fully_included", "partially_included")
# maxilla_scope is BINARY on the ground-truth side, and only here.
#
# The prediction side keeps the three-way enum: the VLM answers it off the
# 3d_frontal render, where "the volume boundary cuts across the maxilla" is
# something you can actually see. A REPORT cannot support the distinction --
# surveying the 582 training cases, 51 came back "fully_included" and only 4
# of those reports state an extent at all; 35 state nothing about the scan's
# extent anywhere and the value was inferred from the mere existence of a
# "Maxilla" section (which is why 49 of the 51 are A-cases, the cohort whose
# reports carry that section header). Asking a report to distinguish fully
# from partially covered manufactures a claim it never made, and
# a survey then grades the model against it.
#
# Collapsing to included|not_included keeps the one thing a report DOES
# settle. structured_findings_evaluation.py folds the prediction the same way before
# comparing, so nothing on the prediction side has to change -- see
# fold_maxilla_scope there.
MAXILLA_SCOPES = ("included", "not_included")
_MAXILLA_SCOPE_LEGACY = {"fully_included": "included",
                         "partially_included": "included"}
# ── Scope default: partially_included ────────────────────────────────────
#
# The same defect the MAXILLA_SCOPES note above describes, fixed a different
# way for the fields that keep a three-way enum. Stage 1 is handed two or
# three values and must pick one even where the report picks none, so it
# reaches for a strong one: across the 1274 training drafts the sinus scope
# came back "fully_included" 53 times while only 11 reports contain any
# full-inclusion wording at all, and it came back null 2410 times.
#
# The rule, per the convention: a scope is partially_included UNLESS the
# report earns something stronger.
#
#   fully_included  the report SAYS so -- "completely/fully/entirely
#                   included", "included in its entirety". Anything weaker,
#                   including a bare "sinuses included in the scan volume",
#                   is not enough: a dental CBCT essentially never closes a
#                   sinus's superior border or contains a whole condyle.
#   not_included    the DRAFT says so, and the field allows it. This one
#                   trusts stage 1 rather than a regex on purpose -- reports
#                   phrase exclusion many ways ("the acquisition volume
#                   ... excludes the condylar processes bilaterally"), and
#                   the draft gets 2414 condyle sides right that a phrase
#                   list would drop to partially_included.
#   partially_included   everything else, including null.
#
# Applied post-hoc to the stored *_report_facts.json, so re-deriving GT
# under this rule costs no LLM calls -- parse_reports_to_gt.py re-reads the
# cached drafts and only re-normalises.
_FULL_INCLUSION = re.compile(
    r"(?<!not\s)(?:complete|full|entire)(?:ly)?\s+includ|"
    r"(?:complete|full|entire)\s+inclusion|"
    r"includ\w*\s+in\s+(?:its\s+)?(?:entirety|full)", re.I)


def _resolve_scope(draft_value: Optional[str], report_text: Optional[str],
                   allow_not_included: bool) -> str:
    """One scope field, under the partially_included default. See the note above."""
    if allow_not_included and draft_value == "not_included":
        return "not_included"
    if draft_value == "fully_included" and report_text and _FULL_INCLUSION.search(report_text):
        return "fully_included"
    return "partially_included"


MUCOSA_STATES = ("normal", "thickening")
SINUS_CONTENTS = ("air", "fluid", "mixed")
CANAL_LOCATIONS = ("lingual", "buccal")
# The lower molars schema.json asks tooth_{fdi}_mandible_canal about -- its
# applies_to_fdi list, mirrored here the way every other shape in this file
# mirrors the schema, because the GT is written to match the prediction field
# for field. A tooth the schema asks about and this set omits gets NO ground
# truth at all (a survey then scores the model's answer against
# nothing), so the two lists have to move together.
CANAL_TOOTH_FDIS = (36, 37, 38, 46, 47, 48)
LESION_TYPES = ("radiopaque", "radiolucent")
OSSEO_STATES = ("well", "poor")


# ══ STAGE 1: the report-shaped intermediate ════════════════════════════════
#
# One flat spec per arch, phrased in the vocabulary a report uses, NOT the
# schema's. Every field here answers a question a radiologist's prose can
# actually answer; nothing here asks for image evidence, and no fact is
# asked twice.

_SHARED_FIELDS = """\
  "teeth_present":      [int]   FDI numbers the report states are PRESENT
  "teeth_absent":       [int]   FDI numbers the report states are ABSENT/missing/extracted
  "presence_enumerated": "present" | "absent" | "both" | "neither"
                                WHICH of those two lists the report actually
                                ENUMERATES, as its own statement about the dentition.
                                "Absence of teeth 34, 35, 36 and 45" enumerates the
                                absent teeth -> "absent". "Teeth 32, 33 ... are
                                present; teeth 31, 37 ... are absent" -> "both".
                                A report that never states which teeth are there,
                                and only names teeth while describing findings
                                ("calculus on 41, 42 and 43"), enumerates NEITHER --
                                answer "neither" even though you listed those teeth
                                as present above. This decides what is assumed about
                                the teeth the report never names, so answer it from
                                the report's sentences, not from your lists.
  "primary_teeth":      [int]   deciduous teeth still in place ({primary_range})
  "unerupted": [ {{"fdi": int,
                  "state": "complete_bony_inclusion" | "partially_erupted",
                  "orientation": "normal"|"mesial"|"distal"|"lingual"|"buccal"}} ]
                                teeth the report calls included/impacted/unerupted/
                                partially erupted. state = fully covered by bone vs.
                                partly through. An OSTEO-MUCOSAL inclusion is
                                "partially_erupted", not "complete_bony_inclusion":
                                the word names bone over part of the crown and mucosa
                                over the rest, so part of the crown is already through
                                the bone. Reserve "complete_bony_inclusion" for a tooth
                                the report puts entirely inside bone ("included in
                                bone", "impacted in bone", "osseous inclusion").
                                orientation = the tipping direction
                                if the report names one, else "normal".
  "endodontic": [ {{"fdi": int,
                   "quality": "adequate"|"inadequate"|"overfilled"|"discontinuous"|null,
                   "periapical_lesion": bool}} ]
                                root-canal-treated teeth ("endodontic treatment",
                                "canal filling", "sequelae of endodontic treatment").
                                quality: "within limits of adequacy"/congruous ->
                                "adequate"; short of the apex -> "inadequate"; beyond
                                the apex -> "overfilled"; gaps/voids -> "discontinuous";
                                report does not judge it -> null.
                                periapical_lesion: true only if the report describes an
                                apical radiolucency/granuloma/lesion at that tooth.
  "post_and_core":      [int]   teeth with a post (and core) in the canal
  "crowns":             [int]   teeth carrying a full crown / prosthetic cap
  "fillings":           [int]   teeth with a direct restoration / filling / inlay
  "caries":             [int]   teeth the report calls carious / decayed / with a
                                carious lesion. NOT a tooth whose CANAL FILLING is
                                called discontinuous or short of the apex -- that is
                                an endodontic quality, and it belongs only in
                                "endodontic" above.
  "root_remnants":      [int]   residual roots / radicular remnants (no crown left)
  "root_fractures":     [int]   teeth with a root fracture
  "periapical_lesions": [int]   teeth with an apical lesion but NO endodontic entry above
  "implants": [ {{"fdi": int|null, "location": string|null, "quadrant": 1-4|null,
                 "osseointegration_status": "well"|"poor"|null,
                 "with_crown": bool}} ]
                                implant positions by the FDI they replace. An implant
                                located only by region gets fdi null, the report's own
                                words in "location", and the quadrant those words name
                                ("posterior right mandible" -> 4) -- one entry per
                                implant, so "two implants in the posterior mandible"
                                is TWO entries. Fill "location" whenever the report
                                gives one, even when you could also resolve the FDI.
  "bridges": [ {{"span_start": int|null, "span_end": int|null, "location": string|null,
                "quadrant": 1-4|null, "abutment_teeth": [int], "pontic_teeth": [int],
                "implant_supported_teeth": [int]}} ]
                                fixed prosthetic bridges. "a bridge between 13 and 15"
                                means span_start 13, span_end 15, abutment_teeth
                                [13, 15]. pontic_teeth are ONLY the positions inside
                                that span with no tooth of their own (here [14]) --
                                never the teeth outside the span, and empty when the
                                report does not say which positions are replaced.
                                A bridge given only by region ("a bridge in the left
                                maxillary sector") takes null spans and a "location".
  "unlocated": [ {{"finding": "crown"|"filling"|"caries"|"endodontic"|"post_and_core"
                              |"root_remnant"|"root_fracture"|"periapical_lesion"
                              |"impacted"|"implant"|"bridge",
                  "location": string, "quadrant": 1-4|null}} ]
                                findings the report states but does not pin to a tooth
                                ("conservative restorations in the posterior sector").
                                location = the report's own words, quadrant = the one
                                they name if they name one. This is where a finding
                                goes INSTEAD of a guessed FDI -- never both.
  "alveolar_atrophy":   "none" | "present" | "fully_edentulous"
                                "present" when the report describes atrophy/resorption
                                of the ALVEOLAR PROCESS or RIDGE -- "mandible with
                                moderate bone atrophy", "atrophy in the edentulous
                                areas" -- whatever degree word it uses;
                                "fully_edentulous" when it states this jaw has no
                                teeth at all
  "periodontal_extent": "none"|"mild"|"moderate"|"severe"
                                PERIODONTAL bone loss around the REMAINING TEETH
                                (marginal/crestal bone loss, periodontitis). This is
                                a different finding from alveolar_atrophy above: a
                                report describing atrophy of the ridge has said
                                nothing about periodontal support, so its degree word
                                does NOT come here. "Mandible with moderate bone
                                atrophy" is alveolar_atrophy "present" and
                                periodontal_extent "none" -- the word "moderate"
                                belongs to the atrophy and must not be copied down
                                here. Periradicular or periapical osteorarefaction is
                                a bone lesion around the root TIPS -- it goes in
                                "bone_lesions" below and not here either. This field
                                is only for loss of the crestal bone BETWEEN the
                                teeth: periodontal disease, marginal bone loss,
                                recession of the crest along the roots. Leave "none"
                                otherwise, and never invent a degree the report did
                                not give.
  "periodontal_pattern": [ "horizontal" | "vertical" ]      only when extent != "none"
  "bone_lesions": [ {{"type": "radiopaque"|"radiolucent", "location": string,
                     "teeth": [int]}} ]
                                osteolytic/radiolucent or osteocondensing/radiopaque
                                areas of PATHOLOGICAL significance. A sentence saying
                                none are identified -> empty list.
  "uncertain_teeth":    [int]   teeth the report explicitly says cannot be assessed
                                (artefact, outside the volume). A general remark that
                                is not tied to specific tooth numbers -> empty list.
"""

_MANDIBLE_EXTRA = """\
  "condyle_right_scope": "fully_included"|"partially_included"|"not_included"|null
  "condyle_left_scope":  "fully_included"|"partially_included"|"not_included"|null
                                the report's own statement about the mandibular
                                condyles. This is almost always its OWN short
                                section, usually the last line ("Mandibular
                                condyles: not included in the scan volume", "Excluded
                                from the acquisition and not visible") -- read it and
                                record it here, for BOTH sides unless the report
                                distinguishes them. null only if the report never
                                mentions the condyles at all.
  "canal_right": {{"location": "lingual"|"buccal"|null, "adjacent_teeth": [int]}}
  "canal_left":  {{"location": "lingual"|"buccal"|null, "adjacent_teeth": [int]}}
                                the mandibular canal, ONE SIDE PER FIELD: canal_right
                                is the right canal (quadrant 4, teeth 41-48) and
                                canal_left the left one (quadrant 3, teeth 31-38). A
                                tooth the report names for one side never appears
                                under the other.
                                location = which side of the roots the canal runs on,
                                and ONLY when the report uses one of those two words
                                about its COURSE ("lingual course", "predominantly
                                buccal development"). If the report's only mention of
                                the canal is where it sits relative to a tooth -- "in
                                close relationship with impacted tooth 48", "without
                                contiguity to the roots" -- then it never described a
                                course and location is null. Do not pick the more
                                likely side; null is the answer.
                                adjacent_teeth = teeth whose roots the report says
                                contact / are close to / have a relationship with the
                                canal -- empty when the report says there is no
                                pathological relationship.
"""

_MAXILLA_EXTRA = """\
  "maxilla_scope": "included"|"not_included"|null
                                whether the maxilla is in the scan volume AT ALL.
                                Only two values, on purpose: a report can say the
                                maxilla is there or is missing, but it does not
                                reliably say whether it is FULLY or only PARTLY
                                covered, so that distinction is not asked for here
                                -- "included" covers both. null when the report
                                never states the scan's extent over the maxilla;
                                a section that merely DESCRIBES the maxilla
                                (atrophy, sinuses, teeth) is not a statement about
                                the volume boundary and must stay null.
  "sinus_right": {{"scope": "fully_included"|"partially_included"|null,
                  "mucosa_state": "normal"|"thickening",
                  "sinus_content": "air"|"fluid"|"mixed",
                  "intrasinusal_teeth": [int]}}
  "sinus_left":  {{same shape}}
                                the maxillary sinuses. A report saying they are
                                "normally pneumatized with no mucosal thickening"
                                means mucosa_state "normal" and sinus_content "air".
                                intrasinusal_teeth = roots the report says project
                                into the sinus.
"""


def build_extraction_prompt(report_text: str, arch: str) -> str:
    """One arch's report-facts spec + the FULL report text.

    The whole report is given (not just its {arch} paragraph): section
    headings vary between readers, and findings for one jaw are routinely
    stated in a sentence that sits under another heading -- the condyle
    paragraph in particular is its own section.
    """
    quadrants = ("31-38 and 41-48 (lower jaw)" if arch == "mandible"
                 else "11-18 and 21-28 (upper jaw)")
    fields = _SHARED_FIELDS.format(
        primary_range="71-75, 81-85" if arch == "mandible" else "51-55, 61-65")
    fields += (_MANDIBLE_EXTRA if arch == "mandible" else _MAXILLA_EXTRA).format()

    return f"""You are reading a dental radiology report and recording, as JSON, ONLY what
the report actually says about the {arch.upper()}: FDI {quadrants}.

RULES
1. Record statements, do not make judgements. If the report does not mention
   something, leave the field at its default: [] for every list, "none" for
   alveolar_atrophy and periodontal_extent, null for a nullable enum, false
   for a bool. Silence is NOT a finding -- a report describes what is
   notable and does not list every normal tooth. In particular, a tooth the
   report never names is NOT absent: leave it out of both teeth_present and
   teeth_absent.
2. Copy tooth-number enumerations COMPLETELY. "Teeth 32, 33, 34, 35, 36, 38,
   42, 43, 44, 45 and 48 are present" is eleven numbers in teeth_present --
   every one of them, none dropped, none invented.
2a. NO TOOTH BELONGS IN BOTH PRESENCE LISTS, and an explicit absence beats
   an inferred presence. A005's report says "Complete mandibular dentition;
   18 and 28 absent in the maxilla" -- a completeness claim about the LOWER
   jaw and an absence about the upper. Read wholesale, it produced a maxilla
   with 11-17 and 21-28 present AND 18, 28 absent: the completeness claim
   applied to the wrong arch, then 18 subtracted from it and 28 forgotten.
   A completeness claim ("complete dentition", "presence of all dental
   elements") counts only for the arch the sentence names, and whatever it
   licenses, every tooth the report calls absent comes out of the presence
   list.
3. Only FDI numbers in {quadrants} belong in this answer -- ignore what the
   report says about the other jaw's TEETH. The non-tooth fields at the end
   are the exception: the sentence they answer often sits under its own
   heading rather than under "{arch}", and it still belongs here.
4. A tooth belongs in as many finding lists as the report supports -- a
   root-canal-treated tooth carrying a crown appears in BOTH "endodontic"
   and "crowns". Do not pick one.
5. Do not copy prose into the JSON: no free-text keys, no extra keys, no
   explanations. Only the field names listed below.
6. AN IMPACTED TOOTH AND THE TOOTH IT PRESSES ON ARE DIFFERENT TEETH.
   "38 mesioverted impacting on the crown of 37, 48 impacting on the distal
   root of 47" reports TWO impacted teeth, 38 and 48. 37 and 47 are what they
   press against and are NOT impacted. After any of
       impacting on / impinging on / pressing against / encroaching on /
       in (close) relationship with / in contact with / in contiguity with
   the tooth named is the one being pressed upon -- unless that tooth carries
   the impaction word itself: "the mandibular canal is in close relationship
   with IMPACTED tooth 48" reports 48 as impacted, because the subject there
   is the canal, not another tooth.
7. A CROWN IS A PROSTHESIS; A RESTORATION ON A CROWN IS A FILLING. The word
   "crown" alone is anatomy -- the top of the tooth.
       crowns:   prosthetic crown, capsule, bridge, prosthetic rehabilitation,
                 cantilever, splinted crowns
       fillings: conservative restoration, restoration involving/on the crown,
                 filling, composite, amalgam, Black class restoration,
                 coronal restoration
   "Conservative restorations are present on the crowns of teeth 23, 24, 25,
   26, 27 and a probable prosthetic crown on tooth 16" is FIVE fillings
   (23-27) and ONE crown (16). Do not give all six teeth to both lists: when
   one sentence names two findings, split the teeth between them by which
   clause each number sits in. Rule 4 still holds for a tooth the report
   really does describe twice ("prosthetic crown on 36, which has undergone
   endodontic therapy" is 36 in BOTH crowns and endodontic).
8. A FINDING ON AN ABSENT TOOTH IS NOT A FINDING ON A TOOTH. If the report
   says the tooth is missing, what stands in its place belongs to the
   prosthesis or the socket, not to the tooth:
       pontic, prosthetic replacement element, cantilevered crown,
       crown on the implant in position X, residual/remnant material,
       post-extraction socket, root remnant
   "22 absent with a prosthetic replacement element" puts 22 in teeth_absent
   and in NO finding list. "Prosthetic crown on endosseous implants in
   positions 46 and 47" is an implant finding, not a crowns finding.
9. A TRAILING CLAUSE BELONGS TO ITS OWN SUBJECT, not to the whole sentence.
   "Multiple edentulism with absence of teeth 45, 46, 47, 48, 35, 36 and 37,
   and presence of a root remnant of 32, previously treated endodontically"
   reports endodontic treatment on 32 ALONE. The seven absent teeth are not
   endodontically treated -- they are not there.
10. An orthodontic SPLINT or RETAINER is an appliance, not a finding. "Lingual
   retainer from 33 to 43" is neither impaction nor a crown; those teeth get
   no finding from that sentence.
11a. A ROOT REMNANT IS A TOOTH THAT IS STILL THERE. A retained root is tooth
   substance at that position, so "15 is a remnant" or "a root remnant of
   tooth 22 can be appreciated" puts that FDI in root_remnants AND in
   teeth_present -- never in teeth_absent. Absence is a separate statement
   the report has to make itself ("absent", "missing", "extracted"); it does
   not follow from the remnant, and inferring it is the commonest way this
   field goes wrong.
11. CARIES AND A FILLING ARE OPPOSITE STATES OF ONE SURFACE. Decay is damage;
   a restoration is what replaced the damage. A tooth the report calls
   carious goes in "caries" and NOT in "fillings", and the reverse. Read the
   clause the number sits in: "Conservative dental treatment is present at
   16-14. Carious process involving 16-14-24" gives fillings [16, 14] and
   caries [16, 14, 24] -- 24 is carious only, because the first sentence
   never names it. "N with restoration of a carious lesion" is the one case
   that reads BOTH ways round: the caries was treated, so it is a filling.
12. A POSITION CAN BE NAMED WITHOUT A NUMBER, AND THE TOOTH TYPE IS PART OF
   IT. "In the canine region of the 4th quadrant" is FDI 43 -- quadrant 4 is
   lower right, and its canine is 43. Resolve these, but resolve them
   exactly:
       central incisor 1  lateral incisor 2  canine 3  first premolar 4
       second premolar 5  first molar 6  second molar 7  third molar 8
   The same digits are what an ORDINAL names, counting from the midline:
   "III quadrant: the fourth tooth shows destructive caries" is the
   quadrant as the first digit and the ordinal as the second, so 34. "IV
   quadrant: the mesial surface of the fifth tooth is in contact with the
   distal surface of the fourth tooth" is 45 and 44.
   so the canine of quadrant 2 is 23, not 24. A GROUP name expands only to
   the teeth that group contains: "the upper canine-incisor group" is
   13-11 and 21-23, and never a molar.

   BUT ONLY A CLAIM ABOUT ALL OF THE GROUP EXPANDS TO ALL OF IT. Compare:
       "THE DISTAL TEETH of the 1st quadrant have been endodontically
        treated"      -> a claim about every one of them: 16, 17, 18.
       "Presence OF ROOT REMNANTS IN the molar region of the 1st quadrant"
                      -> a claim that some exist there. The region is known,
                         the teeth are not, and 16, 17, 18 asserts three
                         remnants where the report may describe one.
   The first names its teeth; the second names a place. Give the second to
   "unlocated" with its quadrant -- rule 14 -- and never to a finding list.

   NEVER OVER-ASSUME. A plural on its own is not a count: "root remnants in
   the molar region" licenses no particular tooth, and answering 16, 17 and
   18 asserts three findings the report never made. Record only what is
   named -- if another sentence says "root remnants of tooth 17", that is
   the one tooth you have. Where the report itself quantifies ("two root
   remnants", "multiple remnants") and still does not say which, keep it in
   "unlocated" and let a reader decide.
   When the phrase does not pin the tooth down at all ("the posterior
   sector", "the edentulous areas"), the same applies.
13. A SIDE-SCOPED FIELD TAKES ONLY TEETH OF ITS OWN SIDE. canal_right and
   sinus_right hold right-side FDIs, canal_left and sinus_left left-side
   ones, and a tooth's own number says which side it is on. A sentence that
   covers both sides at once still assigns each tooth by its number: "The
   mandibular canal has a predominantly lingual course bilaterally, in close
   relationship with the roots of tooth 37 on the left" puts 37 in
   canal_left ALONE -- "bilaterally" describes the course, not the contact,
   and 37 is a left-side tooth either way. Never list one tooth on both
   sides.
14. WHEN THE TOOTH CANNOT BE DETERMINED, GIVE THE QUADRANT.
   An implant or a bridge may be located by region rather than by tooth
   ("two implants in the posterior mandible", "a bridge in the left
   maxillary sector"): give "fdi"/"span_start"/"span_end" as null, put the
   words the report used in "location", and give "quadrant" as 1, 2, 3 or 4
   whenever the region names one -- "the left maxillary sector" is quadrant
   2, "the posterior right mandible" is 4. Leave "quadrant" null only when
   the report does not narrow it that far ("the posterior sectors",
   "bilaterally"). For any OTHER finding whose tooth you cannot pin down,
   add one entry to "unlocated", with its quadrant on the same rule, rather
   than putting a guessed number in a finding list -- a wrong tooth is a
   false statement about a real tooth, while an unlocated finding is merely
   one nobody can score. Neither is invented and neither is dropped.
14a. A RANGE IN A FINDING SENTENCE SWEEPS THE ARCH TOO. It is not only
   presence that is stated this way: "the prosthetic crowns of the maxillary
   teeth from 17 to 26" is thirteen crowned teeth, 17-16-15-14-13-12-11 then
   21-22-23-24-25-26, and "Teeth from 13 to 23 endodontically treated" is
   six. Counting 13, 14, 15 ... 18, 21, 22, 23 instead puts 18 in a run it
   is not in and drops 11 and 12 out of one they are in. The rule is the
   same wherever the range appears: follow the arch, never the numbers.
15. A PERIODONTAL POCKET IS NOT A PERIAPICAL LESION, AND IT IS NOT A CBCT
   FINDING AT ALL. A pocket is a soft-tissue measurement; what a CBCT shows
   is the bone that receded around it, which this schema already records as
   periodontal_extent and periodontal_pattern. "Severe periodontal pocket
   due to horizontal bone resorption involving teeth 28, 27 and 26" is
   resorption, and belongs there and nowhere else -- it is not
   periapical_lesions, which is a radiolucency AT THE APEX of a root, and
   not caries. A080's report says exactly that sentence and its extraction
   answered periapical lesions on nine teeth.
16. A HEDGE ABOUT WHICH FINDING IT IS MEANS NO FINDING. The report is
   allowed to decline, and when it does, record nothing rather than picking
   the likelier half:
       "not well distinguishable between post-and-core or endodontic
        treatment"          -> neither "post_and_core" nor "endodontic"
       "the presence of a post-and-core cannot be excluded"
                            -> not an assertion that one is there
   Put the tooth in "uncertain_teeth" instead, which records the doubt
   without claiming the finding. This is NOT the same as the corpus's
   ordinary register: "probable conservative treatment involving 34",
   "likely rebuilt with a post-and-core" and "apparently adequate" all name
   a finding and estimate its confidence -- record those normally.

FIELDS ({arch}):
{fields}
REPORT:
\"\"\"
{report_text}
\"\"\"

Respond with ONLY the JSON object. No preamble, no markdown fences, no commentary."""


# ── LLM call ────────────────────────────────────────────────────────────────

def call_extraction_llm(prompt: str, vllm_url: str, model: str,
                        temperature: float = 0.0, max_tokens: int = 2000,
                        timeout: int = 120, max_retries: int = 3) -> str:
    """
    temperature=0.0 (not just low) -- this is text extraction against a
    single source document, not synthesis; there's no reason for any
    sampling variance here at all.

    THINKING IS DISABLED, AND reasoning_content IS A FALLBACK. The judge
    server runs with --reasoning-parser qwen3, which splits the model's
    output in two: the chain of thought goes to `reasoning_content` and the
    answer to `content`. A Qwen3 that spends its whole max_tokens budget
    thinking therefore returns content=None -- and calling .strip() on that
    raised "'NoneType' object has no attribute 'strip'" on every retry, so
    the case died with "no successful extractions, no output written"
    (seen on A018). run_vqa_inference.py hit this first and handles it the
    same two ways: ask the template not to think, and read reasoning_content
    when content comes back empty anyway.
    """
    url = vllm_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            text = message.get("content") or message.get("reasoning_content")
            if not text:
                raise ValueError("model returned neither content nor reasoning_content")
            return text.strip()
        except Exception as e:
            last_err = e
            print(f"  [warn] extraction call failed (attempt {attempt}/{max_retries}): {e}",
                  file=sys.stderr)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"extraction call failed after {max_retries} attempts: {last_err}")


def _extract_json(text: str) -> Dict:
    """Strip markdown fences if the model added them anyway, then parse.

    Falls back to the outermost {...} span: a model that prefixed one
    sentence of commentary despite being told not to still produced a
    usable object, and losing a whole arch over it is not worth it.
    """
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) > 1:
            t = parts[1]
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ══ STAGE 1b: sanitize the intermediate ════════════════════════════════════
#
# Cheap, total, and schema-free: the intermediate is small enough that every
# field can be range-checked here, so stage 2 never has to defend itself
# against a stray string where an FDI belonged.

def _as_fdi_list(value: Any, allowed: List[int]) -> List[int]:
    """Ints (or int-like strings) that are real FDIs in THIS arch, deduped,
    in the arch's declared order. An FDI from the other jaw is dropped, not
    salvaged: it means the extractor answered about the wrong arch."""
    allowed_set = set(allowed)
    out = []
    for v in value if isinstance(value, list) else []:
        if isinstance(v, bool):
            continue
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v)
        if isinstance(v, int) and v in allowed_set and v not in out:
            out.append(v)
    return [f for f in allowed if f in out]


def _as_enum(value: Any, allowed: Tuple[str, ...], default: Optional[str]) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _as_entries(value: Any, allowed: List[int]) -> List[Dict]:
    """list[object] entries keyed by a valid in-arch FDI, first one wins."""
    allowed_set, seen, out = set(allowed), set(), []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        fdi = item.get("fdi", item.get("fdi_number"))
        if isinstance(fdi, str) and fdi.strip().isdigit():
            fdi = int(fdi)
        if not isinstance(fdi, int) or isinstance(fdi, bool) or fdi not in allowed_set:
            continue
        if fdi in seen:
            continue
        seen.add(fdi)
        out.append({**item, "fdi": fdi})
    return sorted(out, key=lambda e: e["fdi"])


def _as_text(value: Any) -> Optional[str]:
    """A non-empty string, or None. Used for the free-text `location`."""
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _as_quadrant(value: Any, arch: str) -> Optional[int]:
    """1-4, and only a quadrant this arch actually has.

    A maxilla block claiming quadrant 3 is the same cross-arch leak the
    laterality screen catches for teeth, caught here before it is written.
    """
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    allowed = (1, 2) if arch == "maxilla" else (3, 4)
    return value if isinstance(value, int) and value in allowed else None


def _unlocated_entries(value: Any) -> List[Dict]:
    """The entries _as_entries drops: no usable FDI, but a stated location.

    An entry with neither is not a finding at all and stays dropped.
    """
    out = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        fdi = item.get("fdi", item.get("fdi_number"))
        if isinstance(fdi, str) and fdi.strip().isdigit():
            fdi = int(fdi)
        if isinstance(fdi, int) and not isinstance(fdi, bool):
            continue
        if _as_text(item.get("location")):
            out.append(item)
    return out


_ABSENCE_WORD = re.compile(r"absen|missing|edentul|extract|agenes|avuls", re.I)


def _absence_stated(report_text: Optional[str], fdi: int) -> bool:
    """Does a sentence that names this tooth also call the position empty?

    The guard for the one inference this corpus invites most: "A root remnant
    of tooth 22 can be appreciated" says a fragment of 22 is THERE, and the
    extractor reads it as 22 being gone. A remnant and an absence can coexist
    -- A012's "Absence of teeth 36, 37, 38, 46 and 48. At the 46 region, the
    presence of a likely root remnant" is exactly that -- but one does not
    follow from the other, so the absence has to be said.
    """
    if not report_text:
        return True
    flat = re.sub(r"(\d)\.(\d)", r"\1\2", report_text)
    for sentence in re.split(r"(?<=[.;:])\s+", flat):
        if re.search(rf"\b{fdi}\b", sentence) and _ABSENCE_WORD.search(sentence):
            return True
    return False


def _word_in(report_text: Optional[str], *words: str) -> bool:
    """Is any of these words actually in the source report?

    Used to gate the two fields whose enum values ARE the words a report
    would have to use -- a canal running lingually or buccally, resorption
    described as horizontal or vertical. If neither word occurs anywhere in
    the report, no answer was read off it, so the field cannot be filled.
    A guard, not an extractor: it only ever removes a value the source does
    not carry, and it is skipped entirely when the report text is not to
    hand. F008's report says the canal is "in close relationship with
    impacted tooth 48" and never describes its course; the extractor
    answered "buccal", then "lingual" once told not to, on two successive
    runs. The word is simply not there.
    """
    if not report_text:
        return True
    low = report_text.lower()
    return any(w in low for w in words)


_ERUPTION_CLAIM = re.compile(r"impact|inclus|disodon|unerupt|germ", re.I)

# Depth words, in the vocabulary these reports actually use. Kept narrow on
# purpose: an earlier draft matched bare "partially"/"semi-", which fires on
# "Semi-total edentulism of the 2nd quadrant, with tooth 28 completely
# impacted" -- a sentence that says nothing about how deep 28 sits.
_PARTIAL_DEPTH = ("osteomucosal", "osteo-mucosal", "osteomucosa", "osteo-mucosa",
                  "osteomucous", "osteo-mucous",
                  "partially impacted", "partially included", "partially erupted",
                  "partial impaction", "partial inclusion", "partial bony",
                  "semi-impacted", "semi impacted", "semi-included", "semi included",
                  "semi-inclusion", "semi-erupted", "semi-bony", "semibony")
_BONY_DEPTH = ("in bone", "in the bone", "within bone", "within the bone",
               "bony inclusion", "bone inclusion", "osseous inclusion",
               "intraosseous", "intra-osseous", "endosseous inclusion",
               "completely included", "complete inclusion", "entirely included")


def _eruption_depth_guard(fdi: int, report_text: Optional[str],
                          state: str) -> str:
    """complete_bony_inclusion is the strong claim; make the report earn it.

    Stage 1 offers two values and describes them as "fully covered by bone vs.
    partly through", so the model has to pick one even where the report picks
    neither -- and it reaches for the first. Two ways that goes wrong, and the
    render contradicts both:

      * OSTEO-mucosal inclusion. The word names bone over part of the crown
        and mucosa over the rest, so part of the crown is already through the
        bone and is drawn standing clear of it.
      * a bare "impacted", with no depth word at all -- "the right mandibular
        canal is in close relationship with impacted tooth 48". That states
        the tooth cannot erupt normally, which is what `impacted` records; it
        does not state that the tooth is entirely enclosed in bone.

    So complete_bony_inclusion survives only where a sentence naming this tooth
    AND making an eruption claim also says the tooth is in bone. Anything else
    that sentence can support falls to partially_erupted -- which is already
    this field's default for an unreadable value, so an explicit-but-unearned
    answer now behaves like a missing one.

    Per tooth, because one report routinely describes several third molars in
    different states, and only over sentences that make an eruption claim,
    because "impacted in bone" three sentences away is about another tooth.
    When no such sentence names this FDI the state is LEFT ALONE: reports write
    ranges ("teeth 48-37") and periphrases ("the tooth germ"), so a failure to
    match text is not evidence about depth. Like the other stage-2 source
    guards it is skipped when the report text is not to hand, and it can only
    ever move a state toward partially_erupted.
    """
    if state != "complete_bony_inclusion" or not report_text:
        return state
    claims = [s for s in re.split(r"(?<=[.;])\s+", report_text)
              if str(fdi) in s and _ERUPTION_CLAIM.search(s)]
    if not claims:
        return state
    low = " ".join(claims).lower()
    if any(w in low for w in _PARTIAL_DEPTH):
        return "partially_erupted"
    if any(w in low for w in _BONY_DEPTH):
        return state
    return "partially_erupted"


_QUADRANT_ARCH = {"1": "maxilla", "2": "maxilla", "3": "mandible", "4": "mandible",
                  "i": "maxilla", "ii": "maxilla", "iii": "mandible", "iv": "mandible"}
_MAXILLA_WORDS = ("maxill", "sinus", "palat", "upper", "antrum")
_MANDIBLE_WORDS = ("mandib", "lower", "ramus", "condyl", "symphys", "chin")
_QUAD_RE = re.compile(r"\b(?:([1-4])(?:st|nd|rd|th)?|(i{1,3}v?|iv))\s+quadrant", re.I)


def _lesion_arch(lesion: dict):
    """Which jaw a bone lesion belongs to, or None when the report never says.

    The extractor lists every lesion in the report without assigning it to an
    arch, so this is where placement happens. Three sources, most specific
    first -- FDI numbers the lesion names, then FDI numbers and quadrant
    references in its free-text location, then arch words. Quadrants are worth
    parsing: "Osteorarefaction in the 3rd quadrant" and "canine region of the
    III quadrant" name a jaw exactly, in both arabic and roman, and they are a
    large share of what arch words alone cannot place.

    Returns None rather than guessing when a location is genuinely
    arch-agnostic ("post-extraction socket", "periradicular bone") or names
    both jaws. The caller turns that into a null, not a false.
    """
    teeth = lesion.get("teeth") or []
    up = any(11 <= t <= 28 for t in teeth)
    lo = any(31 <= t <= 48 for t in teeth)
    if up != lo:
        return "maxilla" if up else "mandible"
    if up and lo:
        return None

    text = (lesion.get("location") or "").lower()
    for m in _QUAD_RE.finditer(text):
        q = (m.group(1) or m.group(2) or "").lower()
        if q in _QUADRANT_ARCH:
            up = up or _QUADRANT_ARCH[q] == "maxilla"
            lo = lo or _QUADRANT_ARCH[q] == "mandible"
    for t in re.findall(r"\b([1-4][1-8])\b", text):
        n = int(t)
        up = up or 11 <= n <= 28
        lo = lo or 31 <= n <= 48
    up = up or any(w in text for w in _MAXILLA_WORDS)
    lo = lo or any(w in text for w in _MANDIBLE_WORDS)
    if up != lo:
        return "maxilla" if up else "mandible"
    return None


def sanitize_report_facts(raw: Dict, arch: str,
                          report_text: Optional[str] = None) -> Dict:
    """Raw stage-1 JSON -> a fully typed, fully defaulted report-facts dict.

    Everything downstream reads THIS, never the raw model output, so an
    omitted key, a string "35" where an int belonged, or an invented key is
    absorbed here once instead of in every consumer.
    """
    raw = raw if isinstance(raw, dict) else {}
    fdis = ARCH_FDIS[arch]
    primary = ARCH_PRIMARY_FDIS[arch]
    f: Dict[str, Any] = {"arch": arch}

    f["teeth_present"] = _as_fdi_list(raw.get("teeth_present"), fdis)
    f["teeth_absent"] = _as_fdi_list(raw.get("teeth_absent"), fdis)
    f["presence_enumerated"] = _as_enum(raw.get("presence_enumerated"),
                                        ("present", "absent", "both", "neither"), None)
    f["primary_teeth"] = _as_fdi_list(raw.get("primary_teeth"), primary)

    f["unerupted"] = [
        {"fdi": e["fdi"],
         "state": _eruption_depth_guard(
             e["fdi"], report_text,
             _as_enum(e.get("state"), ERUPTION_STATES[:2], "partially_erupted")),
         "orientation": _as_enum(e.get("orientation"), ORIENTATIONS, "normal")}
        for e in _as_entries(raw.get("unerupted"), fdis)]

    f["endodontic"] = [
        {"fdi": e["fdi"],
         "quality": _as_enum(e.get("quality"), FILLING_QUALITIES, None),
         "periapical_lesion": _as_bool(e.get("periapical_lesion"))}
        for e in _as_entries(raw.get("endodontic"), fdis)]

    for key in ("post_and_core", "crowns", "fillings", "caries",
                "root_remnants", "root_fractures", "periapical_lesions",
                "uncertain_teeth"):
        f[key] = _as_fdi_list(raw.get(key), fdis)

    f["implants"] = [
        {"fdi": e["fdi"], "location": _as_text(e.get("location")),
         "osseointegration_status": _as_enum(e.get("osseointegration_status"),
                                             OSSEO_STATES, None),
         "with_crown": _as_bool(e.get("with_crown"))}
        for e in _as_entries(raw.get("implants"), fdis)]

    # An implant the report locates only by region ("two implants in the
    # posterior mandible") has no FDI to be keyed by, and _as_entries drops it.
    # Dropping it silently is the wrong half of the trade -- the finding is
    # real, it is only unscoreable -- so it is kept here, out of the FDI-keyed
    # structures, and surfaces in the GT under "_unlocated". Same for a bridge
    # given without a span, and for any other finding stage 1 could not pin to
    # a tooth (prompt rule 14).
    f["implants_unlocated"] = [
        {"location": _as_text(e.get("location")),
         "quadrant": _as_quadrant(e.get("quadrant"), arch),
         "osseointegration_status": _as_enum(e.get("osseointegration_status"),
                                             OSSEO_STATES, None),
         "with_crown": _as_bool(e.get("with_crown"))}
        for e in _unlocated_entries(raw.get("implants"))]
    f["unlocated"] = [
        {"finding": _as_text(e.get("finding")), "location": _as_text(e.get("location")),
         "quadrant": _as_quadrant(e.get("quadrant"), arch)}
        for e in (raw.get("unlocated") if isinstance(raw.get("unlocated"), list) else [])
        if isinstance(e, dict) and _as_text(e.get("location"))]

    bridges = []
    for b in raw.get("bridges") if isinstance(raw.get("bridges"), list) else []:
        if not isinstance(b, dict):
            continue
        span = {k: b.get(k) for k in ("span_start", "span_end")}
        span = {k: (int(v) if isinstance(v, (int, str)) and str(v).isdigit() else None)
                for k, v in span.items()}
        bridges.append({
            "span_start": span["span_start"], "span_end": span["span_end"],
            "abutment_teeth": _as_fdi_list(b.get("abutment_teeth"), fdis),
            "pontic_teeth": _as_fdi_list(b.get("pontic_teeth"), fdis),
            "implant_supported_teeth": _as_fdi_list(b.get("implant_supported_teeth"), fdis),
        })
    f["bridges"] = bridges

    f["alveolar_atrophy"] = _as_enum(raw.get("alveolar_atrophy"),
                                     ("none", "present", "fully_edentulous"), "none")
    # "fully_edentulous" is the one atrophy value that empties a whole arch, so
    # it must have been said ABOUT THIS ARCH. Prompt rule 3 deliberately lets
    # the non-tooth fields be filled from a sentence under another heading --
    # right for the condyles and the canal, wrong here: P123_2 is a mandibular
    # scan whose report reads "complete edentulism" and never mentions the
    # upper jaw at all, and the maxilla block came back fully_edentulous, i.e.
    # sixteen absent teeth claimed for a jaw nobody imaged. When the block
    # names no tooth of its own AND the report never names this arch, the
    # claim has no source and is dropped to "present" -- the atrophy is real,
    # the jaw it belongs to is the other one.
    if (f["alveolar_atrophy"] == "fully_edentulous"
            and not f["teeth_present"] and not f["teeth_absent"]
            and report_text is not None
            and not _word_in(report_text,
                             *(("maxill", "upper") if arch == "maxilla"
                               else ("mandib", "lower")))):
        f["alveolar_atrophy"] = "present"
    f["periodontal_extent"] = _as_enum(raw.get("periodontal_extent"),
                                       BONE_LOSS_EXTENTS, "none")
    pattern = [p for p in (raw.get("periodontal_pattern") or [])
               if isinstance(p, str) and p.strip().lower() in RESORPTION_PATTERNS]
    # "only when extent != none" is the schema's own gate; honouring it here
    # keeps a pattern the report gave for a resorption it also called absent
    # from reaching the GT as an unexplainable pair.
    f["periodontal_pattern"] = ([p.strip().lower() for p in pattern]
                                if f["periodontal_extent"] != "none"
                                and _word_in(report_text, "horizontal", "vertical")
                                else [])

    lesions = []
    for l in raw.get("bone_lesions") if isinstance(raw.get("bone_lesions"), list) else []:
        if not isinstance(l, dict):
            continue
        ltype = _as_enum(l.get("type"), LESION_TYPES, None)
        if ltype is None:
            continue
        lesions.append({"type": ltype,
                        "location": l.get("location") if isinstance(l.get("location"), str) else "",
                        "teeth": _as_fdi_list(l.get("teeth"), fdis)})
    f["bone_lesions"] = lesions

    if arch == "mandible":
        for side in ("right", "left"):
            f[f"condyle_{side}_scope"] = _resolve_scope(
                _as_enum(raw.get(f"condyle_{side}_scope"), SCOPES3, None),
                report_text, allow_not_included=True)
            canal = raw.get(f"canal_{side}")
            canal = canal if isinstance(canal, dict) else {}
            f[f"canal_{side}"] = {
                "location": (_as_enum(canal.get("location"), CANAL_LOCATIONS, None)
                             if _word_in(report_text, "lingual", "buccal") else None),
                "adjacent_teeth": _as_fdi_list(canal.get("adjacent_teeth"), fdis)}
    else:
        # Fold first, then validate: an intermediate extracted before the
        # enum was narrowed still carries fully_/partially_included, and a
        # re-expansion must read it rather than throw it away as unparseable.
        scope = raw.get("maxilla_scope")
        f["maxilla_scope"] = _as_enum(_MAXILLA_SCOPE_LEGACY.get(scope, scope),
                                      MAXILLA_SCOPES, None)
        for side in ("right", "left"):
            s = raw.get(f"sinus_{side}")
            s = s if isinstance(s, dict) else {}
            mucosa = _as_enum(s.get("mucosa_state"), MUCOSA_STATES, "normal")
            # No not_included branch: this fact is only asked when the sinus
            # has voxels in the scan, so SCOPES2 has nowhere to put it.
            scope = _resolve_scope(_as_enum(s.get("scope"), SCOPES2, None),
                                   report_text, allow_not_included=False)
            intrasinusal = _as_fdi_list(s.get("intrasinusal_teeth"), fdis)
            f[f"sinus_{side}"] = {
                "scope": scope,
                "mucosa_state": mucosa,
                "sinus_content": _as_enum(s.get("sinus_content"), SINUS_CONTENTS, "air"),
                # schema v7.1 removed the gate this used to mirror. It kept
                # intrasinusal_teeth only for a fully-included sinus WITH
                # thickening, which threw the finding away exactly where the
                # evidence is best: the floor is the INFERIOR border, the part
                # a partially included sinus still shows, and a root in the
                # sinus is a finding whether or not the lining is thickened.
                # Gating the GT the old way while the prediction answers
                # ungated would score every honest hit as a false positive, so
                # the two sides move together.
                "intrasinusal_teeth": intrasinusal}
    return f


def extract_report_facts(report_text: str, vllm_url: str, model: str) -> Dict:
    """Stage 1 for one report: two calls, mandible then maxilla."""
    out = {}
    for arch in ("mandible", "maxilla"):
        raw = _extract_json(call_extraction_llm(
            build_extraction_prompt(report_text, arch), vllm_url, model))
        out[arch] = sanitize_report_facts(raw, arch, report_text)
    return out


# ══ STAGE 2: deterministic expansion into the schema shape ═════════════════

def _mentioned_fdis(f: Dict) -> set:
    """Every FDI the report attached a per-tooth CLINICAL finding to.

    These outrank both presence lists, but only the findings that CANNOT
    EXIST WITHOUT A NATURAL TOOTH ROOT qualify: a canal filling, a post and
    core, a filling, caries, a root fracture, a periapical lesion. The
    report cannot describe the endodontic treatment of a tooth it also says
    is missing, so when the two collide the finding is the more specific
    claim and wins.

    A ROOT REMNANT COUNTS AS A TOOTH. It is the project's own convention and
    it is the physical reading: a retained root IS tooth substance at that
    position, "15 is a remnant" is enough to call 15 present, and the v7.1
    arch enum has a value for exactly that state -- "defect". So
    root_remnants joins the findings above rather than the exclusion below.
    Measured over both splits, 30 of the 123 teeth the extraction had placed
    in BOTH root_remnants and teeth_absent had no sentence calling them
    absent at all ("A root remnant of tooth 22 can be appreciated",
    "Presence of root remnants of tooth 36"): the absence was inferred from
    the remnant itself.

    A CROWN IS STILL EXCLUDED -- it does not imply a tooth. A crown sits
    equally well on a bridge pontic: A028's "Absence of the molar group of
    the IV quadrant and 36 ... Bridge 35-37 with crown on 36" puts a crown at
    a position the same sentence calls edentulous, and that crown is the
    pontic's. It stays absent, and is still carried on that tooth's own
    morphology flags, which is where the schema keeps it.
    """
    m = set(f["post_and_core"]) | set(f["fillings"]) \
        | set(f["caries"]) | set(f["root_fractures"]) \
        | set(f["periapical_lesions"]) | set(f["root_remnants"])
    m |= {e["fdi"] for e in f["endodontic"]}
    m |= {e["fdi"] for e in f["unerupted"]}
    return m


def valid_implants(f: Dict) -> List[Dict]:
    """Implants at positions where the report also describes a tooth's own
    findings are dropped.

    An implant is a replacement for a tooth, not a tooth, so it makes its
    position absent -- which is why a spurious one is expensive. A010's
    report puts implants in the mandible (36, 37, 45) and endodontic
    treatments in the maxilla; one extraction run also placed maxillary
    implants at 15 and 16, the very teeth it had just recorded canal
    fillings for. Both cannot be true of one position, and the canal filling
    is the claim the report actually makes.
    """
    mentioned = _mentioned_fdis(f)
    return [e for e in f["implants"] if e["fdi"] not in mentioned]


def resolve_presence(f: Dict) -> Tuple[List[int], List[int], List[int]]:
    """(present, absent, unstated) over this arch's sixteen positions.

    The one place presence is decided. Every other derived field reads this
    result, which is what makes "43 is absent in the arch map but erupted in
    tooth_43_eruption" unrepresentable rather than merely unlikely.

    A position the report never mentions is resolved by `presence_enumerated`
    -- which list the report itself ENUMERATED. An enumeration is exhaustive
    by construction, so "Absence of teeth 36, 37, 45, 46 and 48" makes every
    unnamed position present, and "teeth 32, 33 ... are present" makes every
    unnamed position absent. Enumerating both, or neither, leaves the
    remainder unstated: guessing all sixteen from silence is exactly the
    failure this rewrite removes.

    It has to be asked rather than inferred from which list came back
    non-empty. A013's report enumerates only absences; its teeth_present
    holds 41, 42, 43, 44, 31 and 33 because those are the teeth a sentence
    about calculus deposits happens to name. Reading that as an exhaustive
    presence list makes tooth 32 -- which the report never mentions and its
    absence list never claims -- absent.

    Three claims can collide at one position, and they are ranked. A
    per-tooth CLINICAL FINDING is the most specific and wins outright (see
    _mentioned_fdis). Next is an explicit ABSENCE statement: "Absence of
    teeth 16, 17, 18, 21, ..." beats the presence list, because A013's 21
    reached teeth_present only via the bridge that spans the gap it leaves,
    and a prosthesis spanning a position is not a tooth at it. The presence
    list is last.
    """
    fdis = ARCH_FDIS[f["arch"]]
    mentioned = _mentioned_fdis(f)
    implant_fdis = {e["fdi"] for e in valid_implants(f)}
    absent = (set(f["teeth_absent"]) | implant_fdis) - mentioned
    present = (set(f["teeth_present"]) | mentioned) - absent
    # "fully edentulous" empties the arch only when nothing contradicts it.
    # F043's report describes a partly dentate mandible in detail -- "43 and 42
    # both severely periodontally compromised ... covered by prosthetic crowns
    # splinted as a bridge" -- and also calls the crest edentulous, which the
    # extraction recorded as alveolar_atrophy fully_edentulous. Taken as an
    # override that produced an arch which was simultaneously "43 crowned" and
    # "all sixteen absent": the exact self-contradiction the two-stage design
    # exists to make unrepresentable. A named tooth outranks a summary adjective.
    if f["alveolar_atrophy"] == "fully_edentulous" and not present:
        absent, present = set(fdis), set()

    stated = present | absent
    unstated = [x for x in fdis if x not in stated]
    enumerated = f.get("presence_enumerated")
    if enumerated is None:
        # Pre-`presence_enumerated` report_facts on disk, and any extraction
        # that dropped the field: fall back to whichever list came back,
        # which is right whenever only one of them did.
        enumerated = ("both" if f["teeth_present"] and f["teeth_absent"]
                      else "present" if f["teeth_present"]
                      else "absent" if f["teeth_absent"] else "neither")
    if unstated and enumerated == "present":
        absent |= set(unstated)
        unstated = []
    elif unstated and enumerated == "absent":
        present |= set(unstated)
        unstated = []
    return ([x for x in fdis if x in present],
            [x for x in fdis if x in absent],
            unstated)


def bridge_abutments(f: Dict) -> set:
    """The teeth a bridge is anchored ON -- they carry a retainer crown, which
    is "restoration" in the v7.1 arch vocabulary. Pontics are NOT here: a
    pontic replaces a missing tooth, so its position is absent and is settled
    before this ever runs."""
    out = set()
    for b in f.get("bridges") or []:
        out |= set(b.get("abutment_teeth") or [])
    return out


def _arch_finding(fdi: int, f: Dict, present: List[int], absent: List[int],
                  impacted: set) -> Optional[str]:
    """One position's value for dental_arch_findings_{arch}.findings.

    Schema v7.1's five values, in its own priority order: impacted >
    root_canal_treatment > restoration > defect > unremarkable, with "absent"
    for a position that has no tooth. None = the report never placed a tooth
    here at all, which is not the same claim as "absent".

    THE VOCABULARY CHANGE FIXED A GROUND TRUTH THAT UNDERSTATED THE REPORT.
    The old enum had no crown value, so a tooth the report described only as
    crowned came out "normal" -- a reference asserting no finding at a
    position whose report says "prosthetic crown on 15". Same for a root
    remnant. Both are now expressible: crown, filling and bridge-abutment
    retainer are all "restoration", caries and remnant are both "defect".
    A posted tooth reads "root_canal_treatment" rather than carrying its own
    value, because a panoramic cannot separate a post from a canal filling --
    the composite fact tooth_{fdi}_morphology.with_post_and_core still can,
    and still does.

    The GT has to speak the PREDICTION's vocabulary: structured_findings_evaluation.py
    compares these strings directly, so a GT still saying "filling" against a
    v7.1 prediction saying "restoration" scores every restored tooth wrong.
    """
    if fdi in absent:
        return "absent"
    if fdi not in present:
        return None
    if fdi in impacted:
        return "impacted"
    if fdi in set(f["post_and_core"]) or fdi in {e["fdi"] for e in f["endodontic"]}:
        return "root_canal_treatment"
    if fdi in set(f["crowns"]) | set(f["fillings"]) | bridge_abutments(f):
        return "restoration"
    if fdi in set(f["caries"]) | set(f["root_remnants"]):
        return "defect"
    return "unremarkable"


def _wisdom_fact(fdi: int, present: List[int], absent: List[int],
                 unerupted: Dict[int, Dict]) -> Dict:
    """impacted is DERIVED, never asked -- the schema says so explicitly:
    true when the tooth is not fully erupted OR sits at a non-normal
    orientation.

    An ABSENT position answers null for both impacted and orientation (schema
    v6.7), not false/normal. There is no tooth there, so neither field has
    anything to be true OR false about, and the distinction has teeth
    downstream: postprocess_pred.reconcile reads false as a vote AGAINST an
    impaction another image claimed at that position, and null as an
    abstention. Absence beating impaction is handled after the vote, by
    DROP_IMPACTION_ON_ABSENT_TEETH, so the denial was doing that job twice."""
    entry = unerupted.get(fdi)
    if fdi in absent:
        state, orientation = "absent", None
    elif entry:
        state, orientation = entry["state"], entry["orientation"]
    elif fdi in present:
        state, orientation = "fully_erupted", "normal"
    else:
        state, orientation = None, None
    impacted = bool(state in ("complete_bony_inclusion", "partially_erupted")
                    or (orientation and orientation != "normal"))
    return {"visual_evidence": "", "eruption_state": state,
            "orientation": orientation,
            "impacted": None if state in (None, "absent") else impacted}


def expand_arch(f: Dict, out: Dict) -> Dict:
    """One arch's report-facts -> its schema-shaped facts in `out["global"]`,
    returning the per-tooth material the caller needs for `out["teeth"]`."""
    arch = f["arch"]
    present, absent, unstated = resolve_presence(f)
    unerupted = {e["fdi"]: e for e in f["unerupted"]}
    endo = {e["fdi"]: e for e in f["endodontic"]}
    impacted = {fdi for fdi, e in unerupted.items()
                if e["state"] != "fully_erupted" or e["orientation"] != "normal"}
    # Only the wisdom positions can read "impacted" in the arch map (the
    # schema has no non-wisdom impaction value for this field).
    arch_impacted = impacted & set(WISDOM_FACT)
    g = out["global"]

    # ── whole-arch facts ────────────────────────────────────────────────
    findings = {}
    for fdi in ARCH_FDIS[arch]:
        value = _arch_finding(fdi, f, present, absent, arch_impacted)
        if value is not None:
            findings[str(fdi)] = value
    g[f"dental_arch_findings_{arch}"] = {
        "visual_evidence": "", "findings": findings,
        "uncertain_teeth": list(f["uncertain_teeth"])}

    # present == true is the schema's own criterion (a run of 3+ missing
    # teeth, or an edentulous jaw), computed from the resolved absences
    # rather than taken from the report's adjective -- and OR'd with the
    # report when it states atrophy outright.
    longest_gap, run = 0, 0
    for fdi in ARCH_FDIS[arch]:
        run = run + 1 if fdi in absent else 0
        longest_gap = max(longest_gap, run)
    fully_edentulous = bool(present == [] and absent)
    atrophy_present = bool(longest_gap >= 3 or fully_edentulous
                           or f["alveolar_atrophy"] != "none")
    g[f"alveolar_bone_atrophy_{arch}"] = {
        "visual_evidence": "", "present": atrophy_present,
        "fully_edentulous": fully_edentulous if atrophy_present else False,
        # The report's word for it; the gap count says a gap exists, not
        # that the ridge under it lost height.
        "atrophy": (f["alveolar_atrophy"] != "none") if atrophy_present else False}

    g[f"primary_teeth_{arch}"] = {"visual_evidence": "",
                                  "primary_teeth": list(f["primary_teeth"])}

    q_lo, q_hi = ((3, 4) if arch == "mandible" else (1, 2))
    implants = [{"fdi_number": e["fdi"], "location": f"position {e['fdi']}",
                 "osseointegration_status": e["osseointegration_status"],
                 "with_crown": e["with_crown"]} for e in valid_implants(f)]
    g[f"implants_{arch}"] = {
        "visual_evidence": "",
        # Counts are computed from the list, not extracted -- the schema
        # requires len(implants) == the two counts summed, and a computed
        # count cannot disagree with the list it was computed from.
        f"quadrant_{q_lo}_implant_count": sum(1 for i in implants
                                              if i["fdi_number"] // 10 == q_lo),
        f"quadrant_{q_hi}_implant_count": sum(1 for i in implants
                                              if i["fdi_number"] // 10 == q_hi),
        "implants": implants}

    # A pontic is the replacement tooth a bridge carries, so it can only sit
    # at a position INSIDE the span that has no tooth of its own. Both are
    # checkable here and neither survives prompting: A013's "bridge between
    # teeth 11 and 22" came back listing eleven pontics, most of them outside
    # the span and several of them teeth the same report calls present.
    # Clamping beats asking again -- the constraint is geometric, not
    # clinical, so code can enforce it exactly.
    order = ARCH_FDIS[arch]
    bridges = []
    for b in f["bridges"]:
        span = [b["span_start"], b["span_end"]]
        if all(s in order for s in span):
            lo, hi = sorted(order.index(s) for s in span)
            inside = set(order[lo:hi + 1])
        else:
            inside = set(order)
        abutments = [t for t in b["abutment_teeth"] if t in inside] \
            or [s for s in span if s in inside]
        # DERIVED, not filtered. A pontic replaces a missing tooth, so the
        # pontics of a span ARE its absent positions -- and reading them off
        # presence recovers what the extractor cannot. A023's report says
        # "a bridge extending from 33 to 43, with pontic elements between
        # teeth 32 and 42"; the extraction answered pontic_teeth [32, 42],
        # taking the two teeth NAMED as the pontics when the sentence puts
        # the pontics BETWEEN them, at 31 and 41. Those two are exactly the
        # absent positions inside the span. The extractor's own list is kept
        # only where presence is unstated, so it still contributes where
        # nothing can be derived.
        candidates = (set(inside) - set(abutments))
        pontics = (candidates & set(absent)) | (
            {t for t in b["pontic_teeth"] if t in candidates} - set(present))
        bridges.append({
            "span_start": b["span_start"], "span_end": b["span_end"],
            "abutment_teeth": sorted(abutments),
            "pontic_teeth": sorted(pontics),
            "implant_supported_teeth": sorted(
                t for t in b["implant_supported_teeth"] if t in inside),
            "details": ""})
    g[f"fixed_bridges_{arch}"] = {
        "visual_evidence": "", "present": bool(bridges), "bridges": bridges}

    # BONE LESIONS ARE NOT ARCH-SCOPED BY THE EXTRACTOR, AND USED NOT TO BE HERE.
    # The extractor reads the whole report and lists every lesion in it, so
    # f["bone_lesions"] is identical for both arches. This line used to take
    # [0] unconditionally, which filed the report's FIRST lesion under BOTH
    # jaws: bone_quality_mandible and bone_quality_maxilla came out identical
    # in 968 of 1,274 cases, and 301 of the 379 non-empty maxillary locations
    # actually described the mandible ("post-extraction socket of tooth 48",
    # "symphyseal level"). bone_quality_maxilla.present then scored 0.000 in
    # every arm -- against a label describing the other jaw.
    #
    # Placement, in order: the FDI numbers the lesion names, then the FDI
    # numbers and quadrants in its location text, then arch words. A lesion
    # that names neither jaw is left UNPLACED, and an arch that has no lesion
    # of its own while an unplaced one exists gets `null`, not `false` -- the
    # report stated a lesion and did not say where, which is exactly the
    # "reference did not answer" case the loss mask and the metric both drop.
    here = [l for l in f["bone_lesions"] if _lesion_arch(l) == arch]
    unplaced = any(_lesion_arch(l) is None for l in f["bone_lesions"])
    lesion = here[0] if here else None
    g[f"bone_quality_{arch}"] = {
        "visual_evidence": "",
        "present": True if lesion else (None if unplaced else False),
        "type": lesion["type"] if lesion else None,
        "location": lesion["location"] if lesion else None}

    g[f"periodontal_bone_resorption_{arch}"] = {
        "visual_evidence": "", "extent": f["periodontal_extent"],
        "pattern": list(f["periodontal_pattern"])}

    if arch == "mandible":
        for side in ("right", "left"):
            g[f"mandible_condyle_{side}"] = {
                "visual_evidence": "", "scope": f[f"condyle_{side}_scope"]}
            g[f"mandible_canal_{side}"] = {
                "visual_evidence": "",
                "location": f[f"canal_{side}"]["location"],
                "adjacent_teeth": list(f[f"canal_{side}"]["adjacent_teeth"])}
    else:
        g["maxilla_scope"] = {"visual_evidence": "",
                              "maxilla_included": f["maxilla_scope"]}
        for side in ("right", "left"):
            s = f[f"sinus_{side}"]
            g[f"maxilla_sinus_{side}"] = {
                "visual_evidence": "", "scope": s["scope"],
                "mucosa_state": s["mucosa_state"],
                "sinus_content": s["sinus_content"],
                "intrasinusal_teeth": list(s["intrasinusal_teeth"])}

    for fdi in WISDOM_FDI[arch].values():
        g[WISDOM_FACT[fdi]] = _wisdom_fact(fdi, present, absent, unerupted)

    # ── per-tooth facts ─────────────────────────────────────────────────
    lesion_teeth = {t for l in f["bone_lesions"] for t in l["teeth"]}
    lesion_type_by_tooth = {t: l["type"] for l in f["bone_lesions"] for t in l["teeth"]}
    periapical = set(f["periapical_lesions"]) | {e["fdi"] for e in f["endodontic"]
                                                 if e["periapical_lesion"]}
    teeth: Dict[str, Dict] = {}
    endo_from_post: List[int] = []
    for fdi in sorted(ARCH_FDIS[arch]):
        is_absent = fdi in absent
        is_present = fdi in present
        if not is_absent and not is_present:
            # Report never placed a tooth here. Emitting a block would
            # assert something; omitting it says only that the reference is
            # silent, which is the truth.
            continue
        entry = unerupted.get(fdi)
        state = ("absent" if is_absent else
                 entry["state"] if entry else "fully_erupted")
        endo_entry = endo.get(fdi)

        # A POST IMPLIES THE ROOT CANAL TREATMENT IT WAS CEMENTED INTO.
        #
        # A post and core is placed in a canal that has already been root
        # treated -- clinically there is no other way to arrive at one, and the
        # remnant exception the schema notes ("a residual post in a fractured
        # root") is still a post that was placed after endodontic treatment. So
        # `with_post_and_core` true and `with_endo` false is not a finding, it
        # is a report that mentioned the conspicuous thing and left the routine
        # one unsaid -- the same report-by-exception habit that makes silence
        # mean "normal" everywhere else in this file.
        #
        # Where it actually bites is narrower than it first looks, and the
        # reason is the union: over ALL gt files, including per-reader ones,
        # 37 of 142 training and 17 of 31 validate post-and-core teeth had no
        # endo. But consensus_report_facts() unions the readers first, so one
        # reader's "endodontically treated" already answers another's silence.
        # On the CONSENSUS files -- the ones training reads and structured_findings_evaluation.py
        # scores -- validate was already 17/17 consistent, and re-surveying both
        # baseline arms after this change reproduced their numbers exactly.
        #
        # So this repairs the per-reader files, the single-reader cases, and
        # the handful the union missed: in the 120-case SFT pool it moved 6
        # teeth and added 5 with_endo positives (178 -> 183). Worth doing --
        # a contradiction in a training label is worth removing at any rate --
        # but it is not a scoring correction, and it does not explain the
        # baseline's 0.03/0.18 on post_and_core.
        #
        # Derived, not extracted, exactly like bone_loss from the arch extent
        # and the canal facts from the side: the implication runs one way only
        # (post -> endo, never endo -> post), and it adds nothing where the
        # report already said so. filling_quality stays null -- the report gave
        # no quality, and inventing "adequate" is the error this file avoids
        # everywhere else.
        has_post = fdi in set(f["post_and_core"])
        endo_implied = has_post and endo_entry is None
        if endo_implied:
            endo_from_post.append(fdi)

        # A DEFAULT NEEDS A TOOTH TO BE ABOUT.
        #
        # Everywhere a tooth is there, silence is a negative and the dense
        # default is right: radiology reports state findings by exception, so
        # a present tooth the report never calls carious is a tooth without
        # caries, and `with_caries: false` is a real answer worth scoring and
        # worth training on.
        #
        # Where the report says the tooth is GONE, the same silence says
        # nothing at all. `with_caries: false` on an extracted 16 is not a
        # negative finding, it is a category error -- there is no crown to
        # be sound. Emitting it invented ~7 supervised negatives per absent
        # position, and `bone_loss` was worse than defaulted, it was forced
        # ("none" if is_absent) over the arch extent that would otherwise
        # apply. `null` is this file's established word for "the reference
        # did not answer here": structured_findings_evaluation.py
        # both drop null-GT pairs, so the metric and docs/vision_sft_plan.md's
        # -100 loss mask end up reading one definition instead of two.
        #
        # What survives absence is what the report ASSERTED at that position
        # rather than what silence implied: a lesion named at an edentulous
        # site is a real finding and stays. So the rule is per value, not per
        # field -- keep a positive, null a default -- and no anatomical
        # judgement about which fact "belongs to" a site is baked in here.
        def stated(value):
            """The value, or None where only silence would have produced it.

            Every caller passes a boolean built as `fdi in <finding set>`, so
            False means "no sentence put this finding here" and True means one
            did. bone_loss does NOT go through this -- see below, its value
            comes from an arch-level assertion rather than from silence, and
            it needs a different answer.
            """
            return None if (is_absent and not value) else value

        teeth[f"tooth_{fdi}"] = {
            f"tooth_{fdi}_morphology": {
                "visual_evidence": "",
                "is_remnant": stated(fdi in set(f["root_remnants"])),
                "with_caries": stated(fdi in set(f["caries"])),
                "with_root_fracture": stated(fdi in set(f["root_fractures"])),
                "with_endo": stated(endo_entry is not None or has_post),
                "with_fillings": stated(fdi in set(f["fillings"])),
                "with_full_crown": stated(fdi in set(f["crowns"])),
                "with_post_and_core": stated(fdi in set(f["post_and_core"])),
            },
            # The absence statement itself -- the one thing an absent
            # position does answer, and the field that carries it.
            f"tooth_{fdi}_eruption": {"visual_evidence": "", "eruption_state": state},
            f"tooth_{fdi}_endodontic_treatment": {
                "visual_evidence": "",
                # The schema gates this fact on morphology.with_endo. A
                # tooth with no canal filling has no filling quality to
                # report -- null, not a guessed "adequate".
                "filling_quality": endo_entry["quality"] if endo_entry else None,
                "periapical_lesion": stated(fdi in periapical)},
            f"tooth_{fdi}_periodontal_status": {
                "visual_evidence": "",
                # Reports state periodontal status for the arch, not per
                # tooth; the arch-level extent is that statement applied to
                # each tooth it covers -- and it covers TEETH. So an absent
                # position gets null, not the extent and not the "none" this
                # used to force: periodontal bone loss is measured around a
                # root, and "moderate generalized bone loss" is a claim about
                # the dentition, not about the ridge where 46 used to be.
                # Free-text `findings` stays empty -- no survey reads
                # prose fields either way.
                "bone_loss": None if is_absent else f["periodontal_extent"],
                "furcation_involvement": stated(False), "findings": ""},
            f"tooth_{fdi}_bone_quality": {
                "visual_evidence": "", "present": stated(fdi in lesion_teeth),
                "type": lesion_type_by_tooth.get(fdi)},
        }

        # tooth_{fdi}_mandible_canal (schema v6.9), lower molars only.
        #
        # DERIVED from this side's canal_{side} facts, not extracted again:
        # the report describes the canal once per side, and the per-tooth
        # fact asks the same two things about the tooth the canal passes.
        # Same move as tooth_{fdi}_periodontal_status.bone_loss above, which
        # applies the arch-level extent to each tooth it covers.
        #
        #   location         -- the side's course, or null where the report
        #                       never described one. NOT defaulted: the
        #                       prediction is graded per-tooth on an axis the
        #                       reference genuinely left open, and
        #                       survey_facts drops null-GT pairs.
        #   adjacent_to_teeth -- whether THIS tooth is one the report names as
        #                       reaching the canal. False is the neutral value
        #                       and the report's silence really does mean it:
        #                       an adjacency is a finding a report states.
        if arch == "mandible" and fdi in CANAL_TOOTH_FDIS:
            canal = f[f"canal_{'left' if fdi <= 38 else 'right'}"]
            teeth[f"tooth_{fdi}"][f"tooth_{fdi}_mandible_canal"] = {
                "visual_evidence": "",
                # `location` is the canal's own course and survives an absent
                # tooth; the adjacency is about the tooth, so at an absent
                # position it is null unless the report named it there.
                "location": canal["location"],
                "adjacent_to_teeth": stated(fdi in canal["adjacent_teeth"])}

    # What stage 2 had to overrule in stage 1. Written per case so the rate
    # is measurable across a whole split instead of eyeballed on a handful:
    # every one of these is a contradiction the extraction produced and the
    # expansion silently absorbed, and a rate that climbs is the signal that
    # the stage-1 prompt needs work again.
    conflicts = {}
    # Not a contradiction to resolve but a derivation to declare, so the
    # rate stays visible per case rather than being folded in silently.
    if endo_from_post:
        conflicts["endo_implied_by_post_and_core"] = sorted(endo_from_post)
    dropped = [e["fdi"] for e in f["implants"] if e not in valid_implants(f)]
    if dropped:
        conflicts["implants_at_teeth_with_findings"] = dropped
    both = sorted(set(f["teeth_present"]) & set(f["teeth_absent"]))
    if both:
        conflicts["in_both_presence_lists"] = both
    absent_but_found = sorted(set(f["teeth_absent"]) & _mentioned_fdis(f))
    if absent_but_found:
        conflicts["absent_but_has_findings"] = absent_but_found
    clamped = sorted({t for b in f["bridges"] for t in b["pontic_teeth"]}
                     - {t for b in bridges for t in b["pontic_teeth"]})
    if clamped:
        conflicts["pontics_outside_span_or_on_teeth"] = clamped

    # Findings the report states without pinning them to a tooth. They cannot
    # be scored -- nothing downstream is keyed by "the posterior sector" -- but
    # a GT that silently drops them reads as a report that never mentioned
    # them, which is the same false silence rule 1 of the prompt is about.
    # Kept beside the scored fields, not inside them.
    unlocated = ([{**e, "kind": "implant"} for e in f.get("implants_unlocated") or []]
                 + [{**e, "kind": "bridge"} for e in f.get("bridges_unlocated") or []]
                 + [dict(e) for e in f.get("unlocated") or []])

    return {"teeth": teeth, "present": present, "absent": absent,
            "unstated": unstated, "conflicts": conflicts,
            "unlocated": unlocated}


def expand_to_schema(case_id: str, report_facts: Dict) -> Dict:
    """Both arches' report-facts -> one {case_id, global, teeth} GT."""
    out: Dict[str, Any] = {"case_id": case_id, "global": {}, "teeth": {}}
    derivation = {}
    for arch in ("mandible", "maxilla"):
        f = report_facts.get(arch) or sanitize_report_facts({}, arch)
        result = expand_arch(f, out)
        out["teeth"].update(result["teeth"])
        derivation[arch] = {"present": result["present"], "absent": result["absent"],
                            "unstated": result["unstated"],
                            "conflicts": result["conflicts"],
                            "unlocated": result["unlocated"]}
    # Not part of the compared shape (structured_findings_evaluation.py walks the
    # schema's own fields), but the record of what the report actually
    # supported -- an arch with a long "unstated" list is a report the
    # extraction could not resolve, and that must be visible, not silent.
    out["_derivation"] = derivation
    return out


# ── Multi-report grouping (case IDs follow the established A/F/P/S + digits
#    convention used throughout this project, e.g. A007, S0030 --
#    split_dataset.py already parses case IDs this same way) ────────────────

_CASE_ID_RE = re.compile(r"^([A-Z]+\d+)(?:_(.+))?$")


def group_reports_by_case(report_files: List[Path]) -> Dict[str, List[Tuple[Optional[str], Path]]]:
    """
    'A007.txt' -> case 'A007', single report, radiologist=None.
    'A007_1.txt' / 'A007_doctor1.txt' -> case 'A007', radiologist='1'/'doctor1'.
    A filename that doesn't match the case-ID pattern at all falls back to
    using its whole stem as the case_id (single report, no radiologist
    suffix) rather than being silently dropped.
    """
    groups: Dict[str, List[Tuple[Optional[str], Path]]] = {}
    for f in report_files:
        m = _CASE_ID_RE.match(f.stem)
        if m:
            case_id, radiologist = m.group(1), m.group(2)
        else:
            case_id, radiologist = f.stem, None
        groups.setdefault(case_id, []).append((radiologist, f))
    return groups


# ══ Consensus across radiologists -- ON THE INTERMEDIATE ═══════════════════
#
# Consensus used to run over the schema-shaped extractions, which meant
# merging two already-inconsistent structures field by field and hoping the
# result stayed internally consistent (it could not: a union of presence
# lists and a majority vote of the arch map are computed independently).
# Merging the SMALL representation and expanding ONCE afterwards gives a
# consensus that is consistent by the same construction every single
# extraction is.

def _union_ints(lists: List[List[int]]) -> List[int]:
    seen = []
    for lst in lists:
        for v in lst:
            if v not in seen:
                seen.append(v)
    return sorted(seen)


def _majority(values: List, default=None):
    values = [v for v in values if v is not None]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _agreement(values: List) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return Counter(values).most_common(1)[0][1] / len(values)


def consensus_report_facts(per_reader: List[Dict], arch: str) -> Tuple[Dict, Dict]:
    """Merge N readers' report-facts for one arch. Lists union (a finding
    only one reader mentioned is still a finding the reference reports
    support -- same rule postprocess_pred.py uses across image sources),
    scalars take the majority. Presence is the one place a union is wrong:
    "present" and "absent" are contradictory claims about the same position,
    so each position is voted on separately."""
    facts = [p[arch] for p in per_reader if arch in p]
    if not facts:
        return sanitize_report_facts({}, arch), {}
    if len(facts) == 1:
        return facts[0], {}

    merged: Dict[str, Any] = {"arch": arch}
    agreement: Dict[str, Any] = {}

    present_votes, absent_votes = [], []
    for fdi in ARCH_FDIS[arch]:
        p = sum(1 for f in facts if fdi in f["teeth_present"])
        a = sum(1 for f in facts if fdi in f["teeth_absent"])
        if p or a:
            (present_votes if p >= a else absent_votes).append(fdi)
            agreement.setdefault("presence", {})[str(fdi)] = max(p, a) / (p + a)
    merged["teeth_present"] = present_votes
    merged["teeth_absent"] = absent_votes
    merged["presence_enumerated"] = _majority(
        [f.get("presence_enumerated") for f in facts])
    agreement["presence_enumerated"] = _agreement(
        [f.get("presence_enumerated") for f in facts])

    for key in ("primary_teeth", "post_and_core", "crowns", "fillings", "caries",
                "root_remnants", "root_fractures", "periapical_lesions",
                "uncertain_teeth"):
        merged[key] = _union_ints([f[key] for f in facts])
        agreement[key] = {str(fdi): sum(1 for f in facts if fdi in f[key]) / len(facts)
                          for fdi in merged[key]}

    for key, vote_fields in (("unerupted", ("state", "orientation")),
                             ("endodontic", ("quality", "periapical_lesion")),
                             ("implants", ("osseointegration_status", "with_crown"))):
        by_fdi: Dict[int, List[Dict]] = {}
        for f in facts:
            for e in f[key]:
                by_fdi.setdefault(e["fdi"], []).append(e)
        merged[key] = [
            {"fdi": fdi,
             **{vf: _majority([e.get(vf) for e in entries]) for vf in vote_fields}}
            for fdi, entries in sorted(by_fdi.items())]
        agreement[key] = {str(fdi): len(entries) / len(facts)
                          for fdi, entries in sorted(by_fdi.items())}

    merged["bridges"] = [b for f in facts for b in f["bridges"]][:1] if any(
        f["bridges"] for f in facts) else []
    # NOT [:1]. bone_quality carries one lesion per ARCH, and which one that
    # is cannot be chosen before the lesions are placed -- truncating here
    # threw away the maxillary lesion whenever a mandibular one was listed
    # first. Keep them all; the arch builder takes the first of its own.
    merged["bone_lesions"] = [l for f in facts for l in f["bone_lesions"]]

    for key, allowed, default in (("alveolar_atrophy", ("none", "present", "fully_edentulous"), "none"),
                                  ("periodontal_extent", BONE_LOSS_EXTENTS, "none")):
        merged[key] = _majority([f[key] for f in facts], default)
        agreement[key] = _agreement([f[key] for f in facts])
    merged["periodontal_pattern"] = sorted({p for f in facts for p in f["periodontal_pattern"]}) \
        if merged["periodontal_extent"] != "none" else []

    if arch == "mandible":
        for side in ("right", "left"):
            merged[f"condyle_{side}_scope"] = _majority([f[f"condyle_{side}_scope"] for f in facts])
            agreement[f"condyle_{side}_scope"] = _agreement([f[f"condyle_{side}_scope"] for f in facts])
            merged[f"canal_{side}"] = {
                "location": _majority([f[f"canal_{side}"]["location"] for f in facts]),
                "adjacent_teeth": _union_ints([f[f"canal_{side}"]["adjacent_teeth"] for f in facts])}
            agreement[f"canal_{side}"] = _agreement([f[f"canal_{side}"]["location"] for f in facts])
    else:
        merged["maxilla_scope"] = _majority([f["maxilla_scope"] for f in facts])
        agreement["maxilla_scope"] = _agreement([f["maxilla_scope"] for f in facts])
        for side in ("right", "left"):
            sides = [f[f"sinus_{side}"] for f in facts]
            merged[f"sinus_{side}"] = {
                "scope": _majority([s["scope"] for s in sides]),
                "mucosa_state": _majority([s["mucosa_state"] for s in sides], "normal"),
                "sinus_content": _majority([s["sinus_content"] for s in sides], "air"),
                "intrasinusal_teeth": _union_ints([s["intrasinusal_teeth"] for s in sides])}
            agreement[f"sinus_{side}"] = _agreement([s["mucosa_state"] for s in sides])

    # Back through the sanitizer: the merge builds a raw-shaped dict, and its
    # remaining gate (pattern only when extent != none) must hold for the
    # consensus too. intrasinusal_teeth is no longer gated -- see v7.1 above.
    return sanitize_report_facts(merged, arch), agreement


# ── Writing one case ───────────────────────────────────────────────────────

def finalize_gt(case_id: str, report_facts: Dict, schema_path: str,
                quiet: bool = False) -> Dict:
    """Expand, then run normalize_pred.py's coercion as a final assertion.

    Stage 2 builds the objects from the schema's own field names, so
    normalization should now be a no-op -- which is exactly why it is still
    run: any repair it reports is a bug in the expansion, not a malformed
    model answer, and it says so in the warning."""
    gt = expand_to_schema(case_id, report_facts)
    derivation = gt.pop("_derivation", None)
    gt, repairs = normalize_prediction(gt, schema_path)
    if repairs and not quiet:
        print(f"    [WARN] {case_id}: expansion produced {len(repairs)} field(s) the "
              f"schema had to repair -- this is an expand_to_schema bug: "
              f"{summarize_repairs(repairs)}", file=sys.stderr)
    if derivation is not None:
        gt["_derivation"] = derivation
    return gt


# ── Claims the audit could not settle ─────────────────────────────────────
#
# audit_report_facts.py --mask-unresolved takes every claim that still raises
# an ERROR out of its reader's file and lists it under "_unresolved". This is
# its other half: expand the case once WITH the claim and once WITHOUT, and
# null every label the two disagree on. Those are exactly the labels the claim decides -- a
# crown claim reaches with_full_crown and the arch map's entry, a presence
# claim reaches the whole tooth block -- found by running stage 2 rather than
# by a second, hand-kept map from report fields to schema fields that would
# drift from expand_arch. null is already this file's word for "the reference
# did not answer here": build_sft_targets.py masks it and
# structured_findings_evaluation.py drops it.

def _claim_variants(block: Dict, item: Dict, arch: str,
                    report_text: Optional[str]) -> Tuple[Dict, Dict]:
    """(one reading, the other) of one arch block for one unresolved claim."""
    import copy
    a, b = copy.deepcopy(block), copy.deepcopy(block)
    fdi = item["fdi"]
    if item["field"] == "presence":
        a["teeth_present"] = a["teeth_present"] + [fdi]
        b["teeth_absent"] = b["teeth_absent"] + [fdi]
    else:
        a[item["field"].split(".")[0]] = copy.deepcopy(item.get("original"))
    return (sanitize_report_facts(a, arch, report_text),
            sanitize_report_facts(b, arch, report_text))


def _null_disagreement(target: Dict, a: Dict, b: Dict, fdi_map: bool = False) -> int:
    """Null in `target` every value on which `a` and `b` disagree; count them.

    A tooth block or fact that exists in only one reading, and an entry of the
    arch map ({"46": "restoration"}), is REMOVED rather than nulled: a missing
    position is how those containers already say "unstated", and a null
    inside the map would be supervised as part of the map.
    """
    n = 0
    for k in set(a) | set(b):
        av, bv = a.get(k), b.get(k)
        if av == bv or k not in target:
            continue
        if fdi_map or k not in a or k not in b:
            target.pop(k)
        elif isinstance(av, dict) and isinstance(bv, dict) and isinstance(target[k], dict):
            n += _null_disagreement(target[k], av, bv, fdi_map=(k == "findings"))
            continue
        else:
            target[k] = None
        n += 1
    return n


def apply_unresolved(case_id: str, gt: Dict, per_reader: List[Dict],
                     unresolved: List[List[Dict]], texts: List[Optional[str]],
                     schema_path: str) -> Dict:
    """Null in `gt` what each reader's unresolved claims decide.

    `gt` is the expansion of `per_reader` merged (or of the one reader). Each
    claim is replayed in the SAME merge it belongs to, so a claim another
    reader states outright changes nothing and nulls nothing -- the union
    already carries it.
    """
    record = []
    for r, items in enumerate(unresolved):
        for item in items:
            arch = item["arch"]
            readings = []
            for variant in _claim_variants(per_reader[r][arch], item, arch, texts[r]):
                readers = [dict(f) for f in per_reader]
                readers[r] = {**readers[r], arch: variant}
                merged = {a: consensus_report_facts(readers, a)[0]
                          for a in ("mandible", "maxilla")}
                readings.append(finalize_gt(case_id, merged, schema_path, quiet=True))
            n = sum(_null_disagreement(gt[k], readings[0][k], readings[1][k])
                    for k in ("global", "teeth"))
            record.append({k: item[k] for k in ("arch", "field", "fdi", "screen")}
                          | {"reader": r, "labels_nulled": n})
    if record:
        gt.setdefault("_derivation", {})["unresolved"] = record
    return gt


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports-dir", required=True,
                    help="Directory of reference report .txt files, or a single such file. "
                        "Multiple reports for one case are grouped by the A/F/P/S+digits case-ID "
                        "prefix (e.g. A007_1.txt and A007_2.txt both belong to case A007).")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-facts-dir", default=None,
                    help="Where the stage-1 intermediate {case}[_{reader}]_report_facts.json "
                        "is written/read. Defaults to --out-dir.")
    ap.add_argument("--vllm-url", default="http://localhost:8001/v1",
                    help="Qwen3 TEXT model endpoint (same one generate_report_llm.py uses)")
    ap.add_argument("--model", default="qwen3-text")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of CASES (not report files) to process")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cases whose consensus (or single-report) output already exists")
    ap.add_argument("--consensus", action="store_true",
                    help="When a case has multiple radiologist reports, also compute a "
                        "consensus {case_id}_gt.json (per-position presence vote, union of "
                        "findings) on top of each individual {case_id}_{radiologist}_gt.json")
    ap.add_argument("--first-report-only", action="store_true",
                    help="Use only the FIRST radiologist's report per case and write it "
                        "straight to {case_id}_gt.json. Without this (and without "
                        "--consensus) a multi-report case writes only per-radiologist "
                        "files, so structured_findings_evaluation.py -- which looks up "
                        "{case_id}_gt.json -- silently skips it. Cheapest way to get "
                        "one comparable GT file for every case; no consensus is computed.")
    ap.add_argument("--from-report-facts", action="store_true",
                    help="Skip stage 1 entirely: re-expand the {case}_report_facts.json "
                        "already on disk into {case}_gt.json. Pure CPU, no vLLM server, "
                        "so the deterministic half can be tuned without a GPU.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate reports/schema and print planned extractions, no LLM calls")
    args = ap.parse_args()

    facts_dir = Path(args.report_facts_dir or args.out_dir)

    reports_path = Path(args.reports_dir)
    if reports_path.is_file():
        report_files = [reports_path]
    else:
        report_files = sorted(reports_path.glob("*.txt"))

    groups = group_reports_by_case(report_files)
    if args.case_ids:
        groups = {cid: files for cid, files in groups.items() if cid in args.case_ids}
    case_ids = sorted(groups)
    if args.limit:
        case_ids = case_ids[:args.limit]

    if not case_ids:
        print(f"[WARN] No report files found under {reports_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(facts_dir, exist_ok=True)

    if args.first_report_only and args.consensus:
        print("[FAIL] --first-report-only and --consensus are mutually exclusive: "
              "consensus needs every radiologist's report.", file=sys.stderr)
        sys.exit(1)

    n_written = 0
    for case_id in case_ids:
        radiologist_files = sorted(groups[case_id], key=lambda rf: (rf[0] or ""))
        n_available = len(radiologist_files)

        # One report per case: keep the lowest-numbered radiologist and treat
        # the case as single-report, so the extraction lands in
        # {case_id}_gt.json -- the only name structured_findings_evaluation.py looks up.
        if args.first_report_only:
            radiologist_files = radiologist_files[:1]

        n_radiologists = len(radiologist_files)
        is_multi = n_radiologists > 1

        consensus_path = Path(args.out_dir) / f"{case_id}_gt.json"
        if args.resume and consensus_path.exists():
            print(f"[{case_id}] [SKIP] already exists: {consensus_path}")
            continue

        if args.dry_run:
            labels = [r or "(single report)" for r, _ in radiologist_files]
            dropped = (f" -- first-report-only: using {labels[0]}, "
                       f"skipping {n_available - 1} other report(s)"
                       if args.first_report_only and n_available > 1 else "")
            print(f"[{case_id}] DRY-RUN: {n_radiologists} report(s): {labels}"
                 f"{' -- consensus would be computed' if is_multi and args.consensus else ''}"
                 f"{dropped} -> {consensus_path.name} "
                 f"({2 * n_radiologists} LLM call(s))")
            continue

        per_reader: List[Dict] = []
        per_reader_unresolved: List[List[Dict]] = []
        per_reader_text: List[Optional[str]] = []
        for radiologist, rf in radiologist_files:
            suffix = f"_{radiologist}" if is_multi else ""
            facts_path = facts_dir / f"{case_id}{suffix}_report_facts.json"

            if args.from_report_facts:
                if not facts_path.exists():
                    print(f"[{case_id}] [SKIP] no report facts to expand: {facts_path}",
                          file=sys.stderr)
                    continue
                stored = json.loads(facts_path.read_text(encoding="utf-8"))
                # Re-sanitize against the report too, not just the stored
                # JSON: the source-text guards are part of stage 2, so a
                # re-expansion applies them to intermediates extracted
                # before they existed.
                report_text = rf.read_text(encoding="utf-8", errors="replace").strip()
                facts = {arch: sanitize_report_facts(stored.get(arch) or {}, arch,
                                                     report_text)
                         for arch in ("mandible", "maxilla")}
                unresolved = stored.get("_unresolved") or []
            else:
                report_text = rf.read_text(encoding="utf-8", errors="replace").strip()
                if not report_text:
                    print(f"[{case_id}] [SKIP] empty report file: {rf.name}")
                    continue
                print(f"[{case_id}] extracting from {rf.name}"
                     f"{f' (radiologist={radiologist})' if radiologist else ''}...")
                try:
                    facts = extract_report_facts(report_text, args.vllm_url, args.model)
                except Exception as e:
                    print(f"[{case_id}] [ERROR] extraction failed for {rf.name}: {e}",
                          file=sys.stderr)
                    continue
                facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
                unresolved = []

            per_reader.append(facts)
            per_reader_unresolved.append(unresolved)
            per_reader_text.append(report_text)

            if is_multi:
                gt = finalize_gt(case_id, facts, args.schema)
                gt = apply_unresolved(case_id, gt, [facts], [unresolved],
                                      [report_text], args.schema)
                out_path = Path(args.out_dir) / f"{case_id}_{radiologist}_gt.json"
                out_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
                print(f"[{case_id}] -> {out_path}")
                n_written += 1

        if not per_reader:
            print(f"[{case_id}] [ERROR] no successful extractions, no output written",
                  file=sys.stderr)
            continue

        if is_multi and args.consensus:
            merged, agreement = {}, {}
            for arch in ("mandible", "maxilla"):
                merged[arch], agreement[arch] = consensus_report_facts(per_reader, arch)
            # The readers' unresolved claims travel with the merge, so the
            # audit of this file knows them as settled rather than missing.
            merged_out = dict(merged)
            if any(per_reader_unresolved):
                merged_out["_unresolved"] = [
                    {k: v for k, v in u.items() if k != "original"}
                    for items in per_reader_unresolved for u in items]
            (facts_dir / f"{case_id}_report_facts.json").write_text(
                json.dumps(merged_out, indent=2), encoding="utf-8")
            gt = finalize_gt(case_id, merged, args.schema)
            gt = apply_unresolved(case_id, gt, per_reader, per_reader_unresolved,
                                  per_reader_text, args.schema)
            gt["_agreement"] = {**agreement, "n_radiologists": len(per_reader)}
            consensus_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
            print(f"[{case_id}] -> {consensus_path} (consensus of {len(per_reader)} radiologists)")
            n_written += 1
        elif not is_multi:
            # Single report -- write directly as {case_id}_gt.json, no
            # separate per-radiologist file needed.
            gt = finalize_gt(case_id, per_reader[0], args.schema)
            gt = apply_unresolved(case_id, gt, per_reader, per_reader_unresolved,
                                  per_reader_text, args.schema)
            consensus_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
            print(f"[{case_id}] -> {consensus_path}")
            n_written += 1
        # is_multi and NOT --consensus: individual files already written
        # above, no combined file -- matches the old script's behavior of
        # --consensus being opt-in, not automatic.

    if not args.dry_run:
        print(f"[INFO] wrote {n_written} ground-truth file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
