#!/usr/bin/env python3
"""
code/pipeline/postprocess/source_rules.py

The ten source rules of docs/postprocess.md, applied to a finished summary.

WHY A POST-PASS AND NOT SURGERY IN postprocess_pred.py
──────────────────────────────────────────────────────
Each rule says "this field should come from a different source". Implementing
that inside the nine builders that currently produce those fields would touch
`collect_arch_absent`, `build_impacted_teeth`, `build_implants`,
`build_single_canal`, `build_alveolar_bone_atrophy` and the restoration and
endodontic summaries -- six hundred lines of interlocking logic, in a file with
no test suite, where the only verification available is a survey diff.

So the rules run afterwards, over the finished dict, exactly as
`validate_summary_with_facts.py` already does. Three consequences, all of them
wanted:

  * With no facts file the module is a no-op and postprocess behaves as before.
  * Every rule is one function, named for its section in the document, and can
    be turned off on its own -- from a config file, without touching this file:
    see RULES below and configs/postprocess/README.md.
  * What each rule changed is recorded in `source_rules` inside the summary,
    so a rewrite is never silent -- the same contract audit_facts.py keeps.

WHAT IS DELIBERATELY NOT HERE
─────────────────────────────
`fixed bridges` does not DERIVE anything: the mask's per-arch bridge finding is
written into the facts file by `audit_facts.py --derive-bridge-arches`, and this
module only USES it. It uses it as a SOURCE, not a gate -- an arch the mask
marks gets a bridge whether the model found one or not, because gating alone
left 4 of the 6 mask-confirmed cases reporting no bridge at all. See
apply_bridges. And `crown` is applied because the summary JSON is an instrument
worth keeping accurate, even though the crown renderer is silenced and no crown
reaches the report either way.

THE MEASUREMENTS ARE IN THE DOCUMENT, NOT HERE
──────────────────────────────────────────────
Every rule below is one paragraph in docs/postprocess.md with the table
that chose it. This module states what it does and points there; it does not
restate the evidence, which would then have two homes and drift.

ALL OF IT IS VALIDATE-40 EVIDENCE. See the TODO in that document: nine of the
ten rules are unconfirmed on the training split, and two of them (crown,
impaction C-vs-E) rest on margins inside the noise of 40 cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── the arch ranges, and the wisdom slots ───────────────────────────────────
ARCH_FDIS = {
    "mandible": [f for f in range(31, 39)] + [f for f in range(41, 49)],
    "maxilla":  [f for f in range(11, 19)] + [f for f in range(21, 29)],
}
UPPER = set(ARCH_FDIS["maxilla"])
WISDOM_SLOT = {"lower_right_wisdom_tooth": 48, "upper_right_wisdom_tooth": 18,
               "lower_left_wisdom_tooth": 38, "upper_left_wisdom_tooth": 28}
SIDE_OF_FDI = {48: "right", 18: "right", 38: "left", 28: "left"}
UNERUPTED = ("complete_bony_inclusion", "partially_erupted")

# The composite's canal fact is asked here only, so these are the teeth whose
# `location` votes on a side -- see THE RULE -- canal location.
CANAL_TEETH = {"left": (36, 37, 38), "right": (46, 47, 48)}

# The prior for a canal side no read settled. synthesize_report.py carries the
# same name and both are bound to `priors.canal_location` in
# rules_config.BINDINGS, so the config file is the one place it is stated and
# the two cannot drift apart. The value here is the default.
DEFAULT_CANAL_LOCATION = "lingual"

# ── THE SWITCHES ARE THE ARM ────────────────────────────────────────────────
# One per rule, named for its section in docs/postprocess.md so a survey diff
# traces back to a paragraph. THESE ARE DEFAULTS, NOT DECISIONS: each is bound
# to `source_rules.<name>` in rules_config.BINDINGS, and a
# configs/postprocess/*.yaml passed as --config rewrites this dict in place
# before any case is read. Edit the config for an experiment; edit here only
# when adding a rule, and add its binding in the same commit.
RULES = {
    "absent_teeth": True,
    "impaction": True,
    "endodontic": True,
    "fillings": True,
    "crown": True,
    "implants": True,
    "canal_adjacent": True,
    "canal_location": True,
    "atrophy": True,
    "bridges": True,
    "condyle_fov": True,
}


def _d(value) -> Dict:
    return value if isinstance(value, dict) else {}


def _ints(value) -> List[int]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]


def _true(value) -> bool:
    return value is True or str(value).lower() == "true"


def _morph(pred: Dict, fdi: int) -> Dict:
    return _d(_d(_d(pred.get("teeth")).get(f"tooth_{fdi}")).get(f"tooth_{fdi}_morphology"))


def _eruption(pred: Dict, fdi: int) -> Optional[str]:
    v = _d(_d(_d(pred.get("teeth")).get(f"tooth_{fdi}")).get(f"tooth_{fdi}_eruption")).get("eruption_state")
    return v.strip().lower() if isinstance(v, str) else None


# ── THE RULE — absent teeth ─────────────────────────────────────────────────

def absent_list(facts: Dict) -> Set[int]:
    """
    facts.structured.teeth_absent, minus the maxillary positions when the mask
    says the maxilla was not in the volume.

    What the gate filters is UNASSESSABLE positions rather than wrong ones:
    `teeth_absent` is the complement of `teeth_present`, so a mask silence at a
    position means either "no tooth" or "no image", and `fov.maxilla` is the
    field that says which.

    THE EVIDENCE IS THE REPORTS, NOT THE GENERATED GT (survey_upper_mentions.py
    over all 1000 reference reports). Where fov.maxilla == "excluded", 77% of
    reports make only an arch statement -- "the maxilla is partially included
    in the acquisition" and nothing further -- and 13% never refer to the upper
    arch at all; 15% of those cases name an upper tooth anywhere and only 5%
    call one absent, against 26% where the maxilla is imaged. Sixteen upper
    absences there answer a question the radiologist declined to answer.

    AT FACT LEVEL THE GATE IS A WASH, and the docstring should not claim
    otherwise: scored on positions a report actually settles, F1 0.858 with the
    gate against 0.860 without, trading 56 true positives for 66 false ones.
    (The much larger margin the {case}_gt.json answered set reports -- 0.824 vs
    0.728 -- is presence_enumerated fill-in: 2072 of its 2491 answered upper
    positions in excluded cases were never stated by a report.) The gate stays
    because 1904 of the mask's 2026 upper absence claims in those cases land on
    positions no report settles at all: fact-level scoring drops them, RadFact
    charges every sentence built from them.

    Gate on "excluded" only, and gate the whole upper arch:

      - "partial" is a maxilla that WAS imaged -- gating it collapses recall
        from 0.860 to 0.655.
      - firing only where the mask carries no upper tooth is not supportable:
        those 7 cases settle ONE upper position between them, and 101 of the
        354 excluded cases carry the whole upper arch in the mask anyway.
      - an empty upper arch with the maxilla IMAGED is edentulism, not an FOV
        artefact: gating that too costs F1 0.858 -> 0.850 (0.870 -> 0.828 on
        validate) and would delete A018's "Completely edentulous maxilla".

    See docs/postprocess.md, THE RULE -- absent teeth, and regenerate with
    code/studies/absent_fov_gate_evidence.py.
    """
    structured = _d(facts.get("structured"))
    absent = {f for f in _ints(structured.get("teeth_absent"))
              if f in ARCH_FDIS["mandible"] or f in UPPER}
    fov = _d(structured.get("fov")).get("maxilla")
    if isinstance(fov, str) and fov.strip().lower() == "excluded":
        absent -= UPPER
    return absent


# Findings a mask-absent tooth may still legitimately carry. Every one of these
# is a real entity AT a position with no tooth, and suppressing them is how a
# gate becomes a bug -- an implant sits exactly where a tooth is missing, a
# pontic spans by definition, a retained root remains after the crown is gone,
# and an unerupted tooth is present without being visible as one.
GATE_EXEMPT = ("implants", "bridges", "root_remnants", "impacted_teeth",
               "wisdom_teeth", "alveolar_bone_atrophy", "bone_quality")


def maxilla_excluded(facts: Dict) -> bool:
    """The acquisition did not contain the maxilla. See absent_list."""
    fov = _d(_d(facts.get("structured")).get("fov")).get("maxilla")
    return isinstance(fov, str) and fov.strip().lower() == "excluded"


def apply_absent(summary: Dict, facts: Dict, notes: List[str]) -> None:
    """Rewrite each arch's absent-teeth block, then gate the per-tooth findings."""
    absent = absent_list(facts)
    excluded = maxilla_excluded(facts)
    for arch, fdis in ARCH_FDIS.items():
        arch_block = _d(summary.get(arch))
        findings = arch_block.get("arch_findings")
        if not isinstance(findings, dict):
            continue
        block = findings.get("absent_teeth")
        if not isinstance(block, dict):
            block = findings["absent_teeth"] = {}
        if arch == "maxilla" and excluded:
            # UNASSESSABLE, NOT ABSENT -- and not present either, which is what
            # writing `present` here would have asserted. Same rewrite
            # build_maxilla_section makes when the scope is not_included; done
            # again because this rule runs after it and owns the block. The
            # `teeth`/`present` keys are cleared rather than left stale.
            block.clear()
            block.update({"pattern": "unassessable",
                          "unassessable": sorted(fdis),
                          "source": "facts.fov.maxilla == excluded"})
            notes.append("maxilla: fov excluded -- 16 positions unassessable, "
                         "no absence or presence claimed")
            continue
        arch_absent = sorted(f for f in absent if f in fdis)
        before = sorted(_ints(block.get("teeth")))
        block["teeth"] = arch_absent
        block["present"] = [f for f in fdis if f not in absent]
        block["source"] = "facts.teeth_absent, FOV-gated"
        if arch_absent != before:
            notes.append(f"{arch}: absent_teeth {len(before)} -> {len(arch_absent)} "
                         f"(from the mask list)")


def _gate_on_absent(summary: Dict, absent: Set[int], notes: List[str]) -> None:
    """Drop per-tooth findings at a mask-absent FDI. See GATE_EXEMPT."""
    for arch, fdis in ARCH_FDIS.items():
        block = _d(summary.get(arch))
        gone = {f for f in absent if f in fdis}
        if not gone:
            continue
        dropped = 0

        endo = _d(block.get("endodontic_summary"))
        if endo.get("teeth"):
            keep = [f for f in _ints(endo["teeth"]) if f not in gone]
            dropped += len(_ints(endo["teeth"])) - len(keep)
            endo["teeth"] = keep

        groups = _d(_d(block.get("restoration_summary")).get("groups"))
        for name, teeth in list(groups.items()):
            keep = [f for f in _ints(teeth) if f not in gone]
            dropped += len(_ints(teeth)) - len(keep)
            groups[name] = keep

        findings = block.get("tooth_findings")
        if isinstance(findings, list):
            keep = [t for t in findings
                    if not (isinstance(t, dict) and t.get("tooth") in gone)]
            dropped += len(findings) - len(keep)
            block["tooth_findings"] = keep

        if dropped:
            notes.append(f"{arch}: {dropped} per-tooth claim(s) dropped at "
                         f"mask-absent positions")


# ── THE RULE — impaction ────────────────────────────────────────────────────

def apply_impaction(summary: Dict, pred: Dict, notes: List[str]) -> None:
    """
    Re-derive `impacted` from the schema's own _definitions.impacted rule,
    preferring the COMPOSITE's eruption_state and taking `orientation` from the
    3D wisdom fact, which is the only source that carries one.

    The model writes an `impacted` bool of its own and the pipeline takes it;
    re-deriving from the two fields it wrote alongside scores better, and
    preferring the composite's eruption better again. 0.52 -> 0.75 F1.
    """
    glob = _d(pred.get("global"))
    for arch, fdis in ARCH_FDIS.items():
        block = summary.get(arch)
        if not isinstance(block, dict):
            continue
        # The third instance of the same trap: 55 of 80 arch blocks carry no
        # `impacted_teeth` key, because the builder omits it when the model
        # claimed nothing. Guarding on the key's presence made this rule able
        # to REMOVE an impaction and never to add one -- recall 0.52 against
        # the 0.76 the rule was chosen on. A rule that re-sources a field must
        # be able to write it where the old source was silent.
        out: List[Dict] = []
        for slot, fdi in WISDOM_SLOT.items():
            if fdi not in fdis:
                continue
            fact = _d(glob.get(slot))
            es_3d = fact.get("eruption_state")
            es_3d = es_3d.strip().lower() if isinstance(es_3d, str) else None
            es_det = _eruption(pred, fdi)
            state = es_det or es_3d                    # composite first
            orient = fact.get("orientation")
            orient = orient.strip().lower() if isinstance(orient, str) else None
            if state == "absent":
                continue                               # absent is not a claim
            impacted = None
            if state in UNERUPTED:
                impacted = True
            elif orient and orient != "normal":
                impacted = True
            elif state == "fully_erupted":
                impacted = False
            if impacted:
                out.append({"tooth": fdi,
                            "impacted": orient if orient and orient != "normal"
                                        else "unspecified",
                            "sources": ["composite_eruption"] if es_det else ["3d_render"],
                            "derived": "schema _definitions.impacted"})
        before = len(block.get("impacted_teeth") or [])
        block["impacted_teeth"] = sorted(out, key=lambda e: e["tooth"])
        if before != len(out):
            notes.append(f"{arch}: impacted_teeth {before} -> {len(out)} (re-derived)")


# ── THE RULE — endodontic / fillings / crown ────────────────────────────────

def apply_endodontic(summary: Dict, pred: Dict, notes: List[str]) -> None:
    """The composite's `with_endo` alone. The arch read adds 2 true positives
    for 20 false ones and never reaches a tooth the composite could not see."""
    for arch, fdis in ARCH_FDIS.items():
        arch_block = _d(summary.get(arch))
        endo = arch_block.get("endodontic_summary")
        if not isinstance(endo, dict):
            continue
        keep = sorted(f for f in fdis if _true(_morph(pred, f).get("with_endo")))
        before = len(_ints(endo.get("teeth")))
        endo["teeth"] = keep
        endo["source"] = "composite with_endo"
        if before != len(keep):
            notes.append(f"{arch}: endodontic {before} -> {len(keep)} (composite only)")


def apply_restorations(summary: Dict, pred: Dict, facts: Dict,
                       notes: List[str]) -> None:
    """
    fillings from the composite alone; crown from the mask.

    The arch survey's `restoration` value is aliased onto `filling` upstream,
    which turns 38 composite claims into 161 at 0.17 precision -- v7.1's arch
    vocabulary has no filling value precisely because a crown and a large
    filling are one bright capped tooth at that resolution.
    """
    mask_crowns = {f for f in _ints(_d(facts.get("structured")).get("crowns"))}
    for arch, fdis in ARCH_FDIS.items():
        rs = _d(summary.get(arch)).get("restoration_summary")
        if not isinstance(rs, dict):
            continue
        groups = rs.get("groups")
        if not isinstance(groups, dict):
            groups = rs["groups"] = {}
        if RULES["fillings"]:
            keep = sorted(f for f in fdis if _true(_morph(pred, f).get("with_fillings")))
            before = len(_ints(groups.get("fillings")))
            groups["fillings"] = keep
            if before != len(keep):
                notes.append(f"{arch}: fillings {before} -> {len(keep)} (composite only)")
        if RULES["crown"] and mask_crowns:
            keep = sorted(f for f in fdis if f in mask_crowns)
            before = len(_ints(groups.get("crown")))
            groups["crown"] = keep
            if before != len(keep):
                notes.append(f"{arch}: crown {before} -> {len(keep)} (mask crown list)")


# ── THE RULE — implants, and the bridge gate ────────────────────────────────

def apply_implants(summary: Dict, facts: Dict, notes: List[str]) -> None:
    """
    Positions from the mask; `with_crown` and `osseointegration_status` carried
    over from the VLM entry at the same FDI, because no segmentation can say
    either. The VLM's own positions score 0.00/0.00 -- it places implants in
    the wrong region, which is a failure no drop-rule can repair.
    """
    mask = {f for f in _ints(_d(facts.get("structured")).get("implants"))}
    if not mask:
        return
    for arch, fdis in ARCH_FDIS.items():
        arch_block = summary.get(arch)
        if not isinstance(arch_block, dict):
            continue
        # NEITHER `if "implants" not in pros` NOR `if not pros`. The VLM finding
        # nothing in an arch leaves the key -- or the whole prosthetics block --
        # absent, and skipping there makes this rule additive only where the
        # model already claimed something, which is the one place the mask is
        # least needed. Both guards were written and both were wrong: they cost
        # 11 of 25 mask implants, recall 0.31 against the 0.53 the rule was
        # chosen on. Creating the block is safe because the renderers read fixed
        # keys (`if m.get("prosthetics")`), not dict order.
        pros = arch_block.get("prosthetics")
        if not isinstance(pros, dict):
            if not any(f in fdis for f in mask):
                continue
            pros = arch_block["prosthetics"] = {}
        by_fdi = {e.get("fdi_number"): e for e in (pros.get("implants") or [])
                  if isinstance(e, dict)}
        out = []
        for fdi in sorted(f for f in mask if f in fdis):
            src = _d(by_fdi.get(fdi))
            entry = {"fdi_number": fdi, "location": f"position {fdi}",
                     "source": "facts.implants"}
            for key in ("osseointegration_status", "with_crown"):
                if src.get(key) is not None:
                    entry[key] = src[key]
            out.append(entry)
        before = len(pros.get("implants") or [])
        pros["implants"] = out
        if before != len(out):
            notes.append(f"{arch}: implants {before} -> {len(out)} (mask positions)")


def apply_bridges(summary: Dict, facts: Dict, notes: List[str]) -> None:
    """
    Presence and arch from the mask's bridge label. NO SPAN IS CLAIMED.

    This was a gate first, and a gate is not enough: dropping the VLM's
    wrong-arch bridges left 4 of the 6 mask-confirmed cases reporting no bridge
    at all, because the model had put them in the other arch or missed them.
    So the field is sourced -- an arch the mask marks gets a bridge whether the
    model found one or not, which is the same asymmetry break as absence and
    implants.

    And no span, on measurement rather than caution: of the VLM's 5 spans only
    2 survive the per-arch gate, and those two score 0 exact -- A041 says 22-23
    against a reference 21-23, S0000 says 33-37 against 31-42 and 43-45.
    Carrying the surviving spans would import two known-wrong ones to gain
    nothing. `bridge_arches` has no span in it either, so the entry written
    here is a bare presence marker and synthesize_report renders it as
    "Prosthetic bridge exists."

    A facts file with NO `bridge_arches` key was never audited, and that is not
    evidence of no bridge -- the rule does not fire.
    """
    structured = _d(facts.get("structured"))
    if "bridge_arches" not in structured:
        return
    arches = {a for a in (structured.get("bridge_arches") or []) if a in ARCH_FDIS}
    for arch in ARCH_FDIS:
        arch_block = summary.get(arch)
        if not isinstance(arch_block, dict):
            continue
        pros = arch_block.get("prosthetics")
        if not isinstance(pros, dict):
            if arch not in arches:
                continue
            pros = arch_block["prosthetics"] = {}
        before = len(pros.get("bridges") or [])
        pros["bridges"] = ([{"present": True, "source": "facts.bridge_arches"}]
                           if arch in arches else [])
        after = len(pros["bridges"])
        if before != after:
            notes.append(f"{arch}: bridges {before} -> {after} (mask bridge label)")


# ── THE RULE — canal-adjacent teeth, and canal location ─────────────────────

def apply_canals(summary: Dict, pred: Dict, facts: Dict, notes: List[str]) -> None:
    """
    adjacent_teeth from the mask's `ian_close_teeth`; location from the
    composite when its per-tooth votes agree, else lingual.

    Adjacency is a distance between two segmented structures, so the mask
    computes what the VLM estimates by eye -- 0.69/0.69 against 0.21/0.54 for
    the two reads unioned. Location is a prior: 55 lingual / 8 buccal here and
    640/154 in training, and neither read has ever identified a buccal canal.
    """
    canals = _d(_d(summary.get("mandible")).get("canals"))
    if not canals:
        return
    close = {f for f in _ints(_d(facts.get("structured")).get("ian_close_teeth"))}
    for side in ("right", "left"):
        entry = _d(canals.get(side))
        if not entry:
            continue
        if RULES["canal_adjacent"]:
            # A canal's own side only: 4x is right, 3x is left.
            mine = sorted(f for f in close
                          if (f in range(41, 49)) == (side == "right")
                          and f in ARCH_FDIS["mandible"])
            before = len(_ints(entry.get("adjacent_teeth")))
            entry["adjacent_teeth"] = mine
            entry["adjacent_source"] = "facts.ian_close_teeth"
            if before != len(mine):
                notes.append(f"canal {side}: adjacent {before} -> {len(mine)} (mask)")
        if RULES["canal_location"]:
            votes = []
            for fdi in CANAL_TEETH[side]:
                v = _d(_d(_d(pred.get("teeth")).get(f"tooth_{fdi}"))
                       .get(f"tooth_{fdi}_mandible_canal")).get("location")
                if isinstance(v, str) and v.strip().lower() in ("lingual", "buccal"):
                    votes.append(v.strip().lower())
            unanimous = votes and len(set(votes)) == 1
            entry["location"] = votes[0] if unanimous else DEFAULT_CANAL_LOCATION
            entry["location_source"] = ("composite_unanimous" if unanimous
                                        else "prior_lingual")


# ── THE RULE — alveolar bone atrophy ────────────────────────────────────────

def apply_atrophy(summary: Dict, facts: Dict, notes: List[str]) -> None:
    """
    State atrophy when the arch carries an EDENTULOUS REGION at all -- either
    because the prediction says so (`present`, which is what puts the block in
    the summary in the first place) or because the mask says the arch has no
    teeth left.

    THE GATE MOVED, 2026-08-17, from full edentulism to `present`. Full
    edentulism is the safe end of the same prior and it scored like one: 4 TP,
    0 FP, 18 FN over the 80 validate arches -- precision 1.00 at recall 0.18,
    every error a miss. Atrophy is not confined to jaws that lost every tooth,
    and the mask gate could not say so. Gating on `present` instead reads
    12 TP / 9 FP / 10 FN, recall 0.18 -> 0.55 for precision 1.00 -> 0.57.

    The mask term is KEPT as a second trigger rather than replaced, because it
    fires where `present` cannot: one of the 5 fully-edentulous arches has no
    summary block at all (the model answered present=false on a jaw with no
    teeth in it). Or-ing the two is a strict gain on this run -- +1 TP, no new
    FP -- and it keeps the anatomy prior for exactly the arches the model
    declines to see.

    `present` alone is close to trusting the model outright: on all 21 arches
    where the block exists, its own `atrophy` was already true, so this rule
    now filters almost nothing and earns its keep through the mask term and
    through fully_edentulous. Dropping the `or edentulous` below is the pure
    present-gate; dropping the whole rule is the raw prediction (13/10/9).
    """
    absent = absent_list(facts)
    for arch, fdis in ARCH_FDIS.items():
        if arch not in summary:
            continue
        block = _d(summary.get(arch))
        edentulous = set(fdis).issubset(absent)
        # `present` is not stored on the block -- build_alveolar_bone_atrophy
        # returns None unless the fact answered present=true, so the block's
        # EXISTENCE is the present flag.
        if "alveolar_bone_atrophy" not in block and not edentulous:
            continue
        before = _d(block.get("alveolar_bone_atrophy")).get("atrophy")
        entry: Dict = {"atrophy": True,
                       "source": ("mask: arch fully edentulous" if edentulous
                                  else "pred: edentulous region present")}
        if edentulous:
            entry["fully_edentulous"] = True
        block["alveolar_bone_atrophy"] = entry
        if before is not True:
            notes.append(f"{arch}: atrophy {before!r} -> True")


# ── THE RULE — condyle scope from the mask's field of view ──────────────────

def condyles_excluded(facts: Dict) -> bool:
    """The acquisition cut the condyles off. Companion to maxilla_excluded."""
    fov = _d(_d(facts.get("structured")).get("fov")).get("condyles")
    return isinstance(fov, str) and fov.strip().lower() == "excluded"


def apply_condyle_fov(summary: Dict, facts: Dict, notes: List[str]) -> None:
    """
    The MASK decides condyle inclusion when it speaks, and it speaks in one
    direction only:

        fov.condyles == "excluded"  -> not_included   (the direct measurement)
        fov.maxilla  == "excluded"  -> not_included   (a volume without the
                                                       maxilla has no condyles)
        neither key                 -> leave the merged read alone

    WHY ONLY WHEN IT SPEAKS. `fov.condyles` is written by extract_facts.py when
    the mandible mask touches the volume's superior boundary; when the condition
    does not fire the key is simply absent (33 vs 7 of the 40 validate cases,
    519 vs 63 of 582 training). Treating that silence as `included` -- i.e.
    letting the mask answer the whole axis -- is the WORST variant measured:
    0.525 validate / 0.545 heldout / 0.699 training, below both the reads and a
    constant. Used only where it actually claims something, it is the best:

        pre-rule, per side, no merge          0.575 / 0.568 / 0.716
        merged reads alone                    0.600 / 0.591 / 0.716
        facts alone, silence = included       0.525 / 0.545 / 0.699
        THIS RULE (facts win, else merged)    0.600 / 0.636 / 0.720

    THE MAXILLA CLAUSE ADDS NOTHING TODAY and is kept as a guard: `fov.maxilla
    == excluded` implies `fov.condyles == excluded` in 14/14 validate cases,
    4/4 heldout and 331 of 340 training. It earns 0 sides on every split. In the
    9 training cases where the two disagree it overrides the more direct
    measurement, which is the cost of keeping it -- accepted because a volume
    that stops below the maxilla cannot hold a condyle, and because a facts file
    without `fov.condyles` would otherwise leave the axis to the reads alone.

    WHAT NONE OF THIS BUYS. Every winning variant lands on `not_included` for
    all 40 validate cases: 0.600 is exactly the majority-class score, and on
    training the constant (0.722) still edges this rule (0.720). It buys a
    coherent source hierarchy and a bilaterally consistent statement, not signal.
    """
    by_condyles = condyles_excluded(facts)
    if not (by_condyles or maxilla_excluded(facts)):
        return
    source = ("mask: fov.condyles excluded" if by_condyles
              else "mask: fov.maxilla excluded")
    condyles = _d(_d(summary.get("mandible")).get("condyles"))
    for side in ("right", "left"):
        entry = condyles.get(side)
        if not isinstance(entry, dict):
            continue
        before = entry.get("scope")
        entry["scope"] = "not_included"
        entry["source"] = source
        if before != "not_included":
            notes.append(f"condyle {side}: {before!r} -> 'not_included' ({source})")


# ── entry point ─────────────────────────────────────────────────────────────

def apply(summary: Dict, pred: Dict, facts: Dict) -> Tuple[Dict, List[str]]:
    """Apply every enabled rule. Returns (summary, notes); `summary` is mutated."""
    notes: List[str] = []
    if not facts:
        return summary, notes
    if RULES["absent_teeth"]:
        apply_absent(summary, facts, notes)
    if RULES["impaction"]:
        apply_impaction(summary, pred, notes)
    if RULES["endodontic"]:
        apply_endodontic(summary, pred, notes)
    if RULES["fillings"] or RULES["crown"]:
        apply_restorations(summary, pred, facts, notes)
    if RULES["implants"]:
        apply_implants(summary, facts, notes)
    if RULES["bridges"]:
        apply_bridges(summary, facts, notes)
    if RULES["canal_adjacent"] or RULES["canal_location"]:
        apply_canals(summary, pred, facts, notes)
    if RULES["atrophy"]:
        apply_atrophy(summary, facts, notes)
    if RULES["condyle_fov"]:
        apply_condyle_fov(summary, facts, notes)
    # THE GATE RUNS LAST, and that ordering is load-bearing. It used to run
    # inside apply_absent, i.e. before the rules that RE-SOURCE the per-tooth
    # findings -- so a filling the gate had just dropped at an absent position
    # was written straight back by apply_restorations, and A019's maxilla
    # reported "Absence of 14, 15, ..." and "composite restoration(s) on teeth
    # 15 and 27" in the same paragraph. A gate is only a gate if nothing writes
    # after it.
    if RULES["absent_teeth"]:
        _gate_on_absent(summary, absent_list(facts), notes)
    summary["source_rules"] = {
        "applied": sorted(k for k, v in RULES.items() if v),
        "changes": notes,
    }
    return summary, notes


def load_facts(facts_dir: Optional[str], case_id: str) -> Dict:
    """The case's facts file, or {} -- a missing file disables every rule."""
    if not facts_dir:
        return {}
    path = Path(facts_dir) / f"{case_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
