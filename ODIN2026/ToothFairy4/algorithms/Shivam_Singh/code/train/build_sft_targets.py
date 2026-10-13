#!/usr/bin/env python3
"""
build_sft_targets.py -- tooth-composite calls + generated GT -> SFT rows.

docs/vision_sft_plan.md §3. One row per tooth call: the exact prompt the pipeline
would send, the answer the reference reports support, and a per-field decision
about which of those fields may contribute loss.

WHAT THIS FILE DOES NOT DO
──────────────────────────
It does not tokenize, and it does not store the rendered prompt. A row carries
the qa_pairs.jsonl call entry verbatim (`images`, `captions`, `questions`) and
trainer.py's collator renders it at load time through the SAME
imported function this file validates it with. Two reasons, and the second is
the important one:

  * size -- build_captioned_image_blocks() inlines each PNG as a base64 data
    URI, ~0.7 MB per call, ~2 GB over the pool;
  * parity -- §3.4 requires the training prompt to be byte-identical to the
    serving prompt. Storing a rendered copy makes that a fact about the day
    this file ran. Storing the INPUTS and calling
    run_vqa_inference.build_call_prompt() on both sides makes it true by
    construction, and the token-id diff against vLLM /tokenize then has one
    thing left to check rather than two.

THE MASK -- the reasons a field is not supervised
─────────────────────────────────────────────────
Field-level, not token-level: this file says WHICH fields may contribute loss,
the collator turns that into -100 spans. Every exclusion has a number behind it
in the plan, and they are counted separately in the audit so a mask that
swallows a field can be seen rather than inferred.

  unstated    the tooth is in `_derivation[arch].unstated` -- the report never
              placed a tooth at that position at all (~45% of training slots).
  null        the GT value is null. After 51154c1 this is where absent-tooth
              positions land: a report that says 16 is gone says nothing about
              whether 16 is carious, so those fields are null, not false.
              Same word structured_findings_evaluation.py already
              score by, so loss and metric read one definition (§3.2).
  gated       filling_quality where morphology.with_endo is not true. The
              schema gates the fact; supervising it anyway teaches the model to
              answer a question it was told not to ask.
  refused     with_root_fracture (4 positives in 4 cases) and
              furcation_involvement (0 positives in 1,785 slots). Refused on
              COUNT, not on inter-reader agreement -- see the note below
              DEFAULT_REFUSED for why a low Jaccard is a coverage statistic
              and not a reason to drop a field. They stay in the OUTPUT SHAPE
              so the decoded object is still schema-valid; they just carry no
              gradient.
  evidence    every empty visual_evidence string, plus any the STUDENT could
              not confirm. Prose arrives only via --evidence-from -- a teacher
              shown the image and the already-established answer -- and code/train/visual_evidence/README.md
              supervises it only where check_evidence_perceivable.py kept it,
              since training the model to name cues it cannot resolve would
              manufacture the over-calling the experiment exists to remove.

              THE FIELD IS FIRST IN EVERY FACT, by the schema's own ordering,
              so whatever stands here conditions every supervised decision
              after it whether or not it carries gradient. That makes this a
              structural choice, not a loss one, and it has to be the SAME for
              the tooth calls and the nine `global` ones. It was not: arm 6
              supervised 20,878 tooth strings, left 9,572 more empty, and gave
              all 13,346 arch strings "" because the teacher pass was written
              for teeth alone -- three conditioning regimes reported as one
              number. The rows built here now carry "" throughout, all 47,763
              of them, which is also why --evidence-weight has nothing left to
              weight.

              One thing the flag does not do, and that went unnoticed through
              arm 6: pointing --evidence-from at a SCREENED directory does not
              keep failed prose in the target and merely unsupervise it. The
              screen's filtered copy has already dropped those strings, so
              they arrive here empty and are masked for that reason instead.
              Conditioning on prose the screen rejected means passing the raw
              draft dir.
  capped      §3.5's 6:1 rule. A supervised boolean whose negatives outnumber
              its positives by more than --neg-cap has the surplus negatives
              masked, seeded, down to that ratio -- because a ~92%-negative
              boolean under token CE in a 248,320-vocab will learn to emit
              `false`, which is arm 0 wearing a costume (R1). Masked rather
              than dropped: see apply_negative_cap() for why dropping calls
              cannot do this.

SPLIT SAFETY
────────────
Plan §0: nothing from dataset/validate may enter training. Enforced here at
load time, by reading the validate directory rather than trusting the case
list, and the count refused is printed. A case list is a convention; this is
a check.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path


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

ROOT = REPO_ROOT

import run_vqa_inference as RVI          # noqa: E402  -- prompt parity, §3.4

# A LOW INTER-READER JACCARD IS A COVERAGE NUMBER, NOT A CONTRADICTION NUMBER.
#
# This is the correction that rebuilt this list (2026-08-13), and it matters
# because the whole refusal set was derived from the wrong statistic.
#
# Radiology reports state findings by exception. Reader A writes "post and core
# on 16"; reader B writes about the sinus and never mentions 16 at all. The
# pairwise Jaccard of their post_and_core sets is 0 -- and nothing is in
# dispute. Silence is not denial, which is the same rule this project already
# applies in two other places: parse_reports_to_gt.py nulls an absent tooth's
# fields rather than defaulting them, and verify_report_facts.combine() scores
# "one claims, one silent" as positive.
#
# So the union is the label: a tooth is positive if ANY reader claims it. The
# GT already does this -- consensus_report_facts() unions through _union_ints()
# -- and it was verified rather than assumed: over 341 multi-reader training
# cases the consensus file equals the union of its readers on 341/341 for
# with_post_and_core (43 teeth) and with_endo (418).
#
# Re-derived on POSITIVE COUNTS in the 120-case wide pool, which is what a
# rare-label problem actually turns on:
#
#     field                positives   cases   neg:pos
#     with_fillings              218      48      9:1
#     with_endo                  185      58     11:1
#     with_full_crown            120      43     17:1
#     with_caries                 54      27     38:1
#     periapical_lesion           42       -     50:1
#     is_remnant                  33      18     64:1
#     with_post_and_core          21       8    101:1
#     with_root_fracture           4       4    532:1
#
# What stays refused is refused on COUNT, an argument the union reading does
# not touch:
DEFAULT_REFUSED = {
    # 4 positives in 4 cases. With whole-case held-out splits, a single case
    # moves the entire field; there is nothing to fit and nothing to measure.
    "with_root_fracture",
    # 0 positives against 1,785 negatives -- every example says false, so the
    # only thing it can teach is "emit false", which is R1's operating-point
    # shift produced on purpose. Not an agreement question at all.
    "furcation_involvement",
}

# Supervised despite being sparse, by explicit decision. Kept as a named set
# because the caveat is real even though the refusal argument was not: 21
# positives spread over 8 CASES, and cases are the unit the held-out split
# works in, so this field's variance is a case-count problem rather than a
# label-quality one. Score it separately, and do not let it into an aggregate
# that a conclusion about H1 rests on.
SUPERVISED_BY_DECISION = {"with_post_and_core"}
EVIDENCE_FIELD = "visual_evidence"
# Free text, scored by nobody: no survey reads prose fields.
PROSE_FIELDS = {"findings"}

MASK_REASONS = ("unstated", "null", "gated", "refused", "evidence",
                "prose", "capped")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_case_ids() -> set[str]:
    import re
    out = set()
    for p in (ROOT / "dataset/validate/images").glob("*_0000.nii.gz"):
        m = re.match(r"([A-Z]+\d+)", p.name)
        if m:
            out.add(m.group(1))
    return out


def schema_field_order(schema: dict) -> dict:
    """{fact_id_template: [field, ...]} in the schema's own order.

    Order is load-bearing -- vLLM decodes with a grammar that follows it, and a
    field derived from another must come after it -- so the target object is
    emitted in this order and never in dict-insertion order from the GT.
    """
    out = {}
    for fact in schema["dental_elements"]:
        out[fact["fact_id"]] = list((fact.get("object_fields") or {}).keys())
    return out


def arch_of(fdi: int) -> str:
    return "maxilla" if fdi < 30 else "mandible"


# ── Arch-level (global) calls ────────────────────────────────────────────────
# The panoramic / 3D / sinus calls. Structurally the same job as a tooth call --
# one call, several facts, one constrained object -- but three things differ and
# each of them silently corrupts the target if reused from the tooth path.
#
# 1. FIELD NAMES ARE NOT UNIQUE ACROSS FACTS. `location` is free prose in
#    bone_quality_* ("from which tooth to which tooth") and an ENUM in
#    mandible_canal_* (lingual|buccal). PROSE_FIELDS matches bare names, so
#    reusing it would mask a scored enum. Arch keys are fact-qualified.
#
# 2. THERE IS NO `unstated` SIGNAL. For teeth, _derivation carries
#    present/absent/unstated per arch, so a tooth the report never mentions is
#    masked rather than trained as a negative. Nothing equivalent exists for
#    arch facts: parse_reports_to_gt.py's doctrine is "null means the report did
#    not say", and every field that HAS a neutral value is defaulted in code
#    ([] for lists, "none" for extent, partially_included for scope). So a
#    defaulted negative is indistinguishable from an observed one, and the only
#    honest defences are the two below.
#
# 3. SOME FIELDS ARE CONSTANT. Measured over the 144-case pool: sinus `scope` is
#    partially_included in 100% of cases -- not because sinuses are, but because
#    that is the literal fallback in _resolve_scope. A field where every example
#    says the same thing can only teach "emit that value", which is R1's
#    operating-point shift produced on purpose. Refused outright, on the same
#    grounds as DEFAULT_REFUSED.
#
#    THAT REASONING IS HALF RIGHT, AND ARM 5 MEASURED THE OTHER HALF (plan log, 'Arm 5 -> arm 6').
#    Refusing a constant does not leave it alone; it leaves it WITHOUT A
#    GRADIENT. Arm 5 trained with both sinus `scope` fields refused and the
#    baseline's 15/38 `fully_included` became 38/38 -- the one answer that is
#    always wrong, on a field whose truth is constant and therefore free to
#    anchor. "Not worth supervising" and "safe to leave unsupervised" are
#    different claims, and this file conflated them.
#
#    So ARCH_REFUSED is now a DEFAULT rather than a law: `--supervise` takes
#    dotted `fact_id.field` keys and removes them from it. Teaching the model
#    to emit the constant is exactly what is wanted here -- it is the correct
#    answer 100% of the time, and it costs one enum token per sinus call.
ARCH_SECTIONS = ("mandible", "maxilla")

ARCH_PROSE = {
    # Free text, scored by nobody -- the bone_quality `location` twin of
    # PROSE_FIELDS, kept fact-qualified so mandible_canal_*.location (an enum)
    # is NOT caught by it.
    "bone_quality_mandible.location",
    "bone_quality_maxilla.location",
}

ARCH_REFUSED = {
    # 100% partially_included across the pool -- a code default, not an
    # observation. See note 3 above.
    "maxilla_sinus_right.scope",
    "maxilla_sinus_left.scope",
}

# {fact_id.field: (gate_field, gate_is_satisfied)} -- the schema's own
# "only when ..." clauses, made executable. Supervising a gated field while its
# gate is false trains a value the schema says should not exist there.
ARCH_GATES = {
    "bone_quality_mandible.type": ("present", lambda v: v is True),
    "bone_quality_mandible.location": ("present", lambda v: v is True),
    "bone_quality_maxilla.type": ("present", lambda v: v is True),
    "bone_quality_maxilla.location": ("present", lambda v: v is True),
    "alveolar_bone_atrophy_mandible.atrophy": ("present", lambda v: v is True),
    "alveolar_bone_atrophy_mandible.fully_edentulous": ("present", lambda v: v is True),
    "alveolar_bone_atrophy_maxilla.atrophy": ("present", lambda v: v is True),
    "alveolar_bone_atrophy_maxilla.fully_edentulous": ("present", lambda v: v is True),
    "periodontal_bone_resorption_mandible.pattern": ("extent", lambda v: v not in (None, "none")),
    "periodontal_bone_resorption_maxilla.pattern": ("extent", lambda v: v not in (None, "none")),
}


def arch_field_order(schema: dict) -> dict:
    """{fact_id: [field, ...]} for mandible+maxilla, in the schema's own order.

    Same contract as schema_field_order(): the grammar decodes in this order and
    derived fields must follow what they derive from, so the target object is
    emitted in it rather than in GT dict order.
    """
    out = {}
    for sec in ARCH_SECTIONS:
        for fact in schema.get(sec) or []:
            out[fact["fact_id"]] = list((fact.get("object_fields") or {}).keys())
    return out


def facts_in_call(call: dict) -> list:
    """Which facts this call actually asks for, read off its own json_schema.

    build_vqa_pairs.py groups facts by their exact `images_needed` set, and the
    group NAMES ('pano_others_mandible') are not derivable from the schema. The
    grammar the server decodes against is, though: its top-level properties are
    exactly the fact ids in the call. Reading them here means the target can
    never disagree with what was asked -- re-deriving the grouping could.
    """
    js = ((call.get("questions") or {}).get("json_schema") or {})
    return list((js.get("properties") or {}).keys())


def build_arch_row(case_id: str, call_key: str, call: dict, gt_global: dict,
                   order: dict, evidence: dict | None, supervise_evidence: bool = False,
                   arch_refused: set | None = None):
    """One SFT row for a panoramic / 3D / sinus call, or (None, reason)."""
    if arch_refused is None:
        arch_refused = ARCH_REFUSED
    images = {k: v for k, v in (call.get("images") or {}).items()
              if v and (ROOT / v).exists()}
    if not images:
        return None, "no_image"

    wanted = [f for f in facts_in_call(call) if f in order]
    if not wanted:
        return None, "no_fact_in_schema"

    target, mask, reasons = {}, {}, collections.Counter()
    for fact_id in wanted:
        src = gt_global.get(fact_id)
        if src is None:
            reasons["no_gt_block"] += 1
            continue
        target[fact_id], mask[fact_id] = {}, {}
        for field in order[fact_id]:
            value = src.get(field)
            key = f"{fact_id}.{field}"
            target[fact_id][field] = (
                (evidence or {}).get(fact_id, "") if field == EVIDENCE_FIELD
                else value)

            if field == EVIDENCE_FIELD:
                text = target[fact_id][field]
                why = (None if (supervise_evidence and isinstance(text, str)
                                and text.strip()) else "evidence")
            elif key in ARCH_PROSE or field in PROSE_FIELDS:
                why = "prose"
            elif key in arch_refused:
                why = "refused"
            elif key in ARCH_GATES:
                gate_field, ok = ARCH_GATES[key]
                why = None if ok(src.get(gate_field)) else "gated"
                if why is None and value is None:
                    why = "null"
            elif value is None:
                why = "null"
            else:
                why = None
            mask[fact_id][field] = why
            if why:
                reasons[why] += 1
    if not target:
        return None, "no_fact"
    return {
        "case_id": case_id, "call_key": call_key, "kind": "arch",
        "prefix": case_id[0],
        "call": {"images": call.get("images"), "captions": call.get("captions"),
                 "questions": call.get("questions")},
        "target": target, "mask": mask,
        "supervised": sum(1 for f in mask.values() for w in f.values() if not w),
        "masked": dict(reasons),
    }, None


def build_row(case_id: str, fdi: int, call: dict, gt_teeth: dict,
              derivation: dict, order: dict, evidence: dict | None,
              refused: set, supervise_evidence: bool = False):
    """One SFT row, or (None, reason) if the call cannot be used."""
    tooth_key = f"tooth_{fdi}"
    block = gt_teeth.get(tooth_key)
    if not block:
        return None, "no_gt_block"

    images = {k: v for k, v in (call.get("images") or {}).items()
              if v and (ROOT / v).exists()}
    if not images:
        return None, "no_image"

    arch = derivation.get(arch_of(fdi)) or {}
    unstated = fdi in set(arch.get("unstated") or [])

    with_endo = ((block.get(f"{tooth_key}_morphology") or {}).get("with_endo")
                 is True)

    target, mask, reasons = {}, {}, collections.Counter()
    for fact_tmpl, fields in order.items():
        fact_id = fact_tmpl.replace("{fdi}", str(fdi))
        src = block.get(fact_id)
        if src is None:
            continue                      # e.g. mandible_canal on an upper tooth
        target[fact_id], mask[fact_id] = {}, {}
        for field in fields:
            value = src.get(field)
            target[fact_id][field] = (
                (evidence or {}).get(fact_id, "") if field == EVIDENCE_FIELD
                else value)

            if field == EVIDENCE_FIELD:
                # Supervised where the STUDENT confirmed the teacher's prose,
                # masked where it did not. The point of training on the
                # rationale is that the model learns to look before it answers,
                # and the point of NOT training on unconfirmed prose is that
                # teaching it to name cues it cannot resolve is manufacturing
                # the over-calling the experiment exists to remove. The screen
                # decides per row; nothing here decides it globally.
                #
                # An empty string is never supervised: there is nothing to
                # learn from "", and training on it would teach the model to
                # skip the field entirely.
                text = target[fact_id][field]
                why = (None if (supervise_evidence and isinstance(text, str)
                                and text.strip())
                       else "evidence")
            elif field in PROSE_FIELDS:
                why = "prose"
            elif field in refused:
                why = "refused"
            elif unstated:
                why = "unstated"
            elif value is None:
                why = "null"
            elif field == "filling_quality" and not with_endo:
                why = "gated"
            else:
                why = None
            mask[fact_id][field] = why
            if why:
                reasons[why] += 1
    if not target:
        return None, "no_fact"
    return {
        "case_id": case_id, "fdi": fdi, "arch": arch_of(fdi),
        "prefix": case_id[0],
        "call": {"images": call.get("images"), "captions": call.get("captions"),
                 "questions": call.get("questions")},
        "target": target, "mask": mask,
        "supervised": sum(1 for f in mask.values() for w in f.values() if not w),
        "masked": dict(reasons),
    }, None


def short_fact(fact_id: str) -> str:
    """Grouping key for per-field statistics and the negative cap.

    `tooth_33_morphology` -> `morphology`, so every tooth pools into one field.
    Arch ids are returned WHOLE: `fact_id.split("_", 2)[2]` on
    `bone_quality_mandible` yields `mandible`, which silently pools it with
    `implants_mandible` and `fixed_bridges_mandible` -- three unrelated fields
    sharing one cap and one audit row.
    """
    return fact_id.split("_", 2)[2] if fact_id.startswith("tooth_") else fact_id


def apply_negative_cap(rows: list, cap: float, seed: int) -> None:
    """§3.5's 6:1 rule, applied per field by MASKING negatives, not by dropping calls.

    The plan states it as "cap negative-only calls at a 6:1 per-field ratio",
    and the reason is a threshold on token CE: a ~92%-negative boolean in a
    248,320-vocab will happily learn to emit `false`, which is arm 0 wearing a
    costume (R1).

    Dropping calls cannot deliver that. The ratios differ wildly per field --
    is_remnant 73:1 against with_fillings 7.7:1 -- while a call carries every
    field at once, so dropping enough calls to fix is_remnant would delete most
    of with_fillings' positives with them. And dropping the all-negative calls
    specifically is worse than useless: a tooth with no findings is exactly what
    the model must learn to call normal, and removing those IS rebalancing
    toward positives, which the same paragraph forbids.

    So the negatives are masked instead, per field, down to cap x positives.
    Every call stays, every positive stays, the output shape is untouched, and
    only the loss changes -- which is the thing the 6:1 number was ever about.

    Seeded, because which negatives get masked must be reproducible across arms:
    arm 2 and arm 3 have to see the identical supervision or the contrast is not
    the contrast.

    Enums are left alone. The rule is stated for booleans, and "negative" is not
    defined for eruption_state.
    """
    if not cap or cap <= 0:
        return
    rng = random.Random(seed)

    slots = collections.defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        for fact_id, fields in row["mask"].items():
            short = short_fact(fact_id)
            for field, why in fields.items():
                if why:                                   # already not supervised
                    continue
                v = row["target"][fact_id][field]
                if not isinstance(v, bool):
                    continue
                slots[f"{short}.{field}"]["pos" if v else "neg"].append(
                    (row, fact_id, field))

    print(f"\n[INFO] §3.5 negative cap at {cap:g}:1 (seeded {seed})")
    for name in sorted(slots):
        pos, neg = slots[name]["pos"], slots[name]["neg"]
        if not pos:
            # No positive anywhere: every example says false, so the only thing
            # it can teach is "emit false". Mask the field entirely and say so
            # -- this is what caught furcation_involvement.
            for row, fact_id, field in neg:
                row["mask"][fact_id][field] = "capped"
            if neg:
                print(f"  {name:38} 0 positives -> all {len(neg)} masked")
            continue
        keep = int(cap * len(pos))
        if len(neg) <= keep:
            continue
        rng.shuffle(neg)
        for row, fact_id, field in neg[keep:]:
            row["mask"][fact_id][field] = "capped"
        print(f"  {name:38} {len(pos):>4} pos, {len(neg):>5} neg -> "
              f"keep {keep:>4} ({len(neg) - keep} masked)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-jsonl", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--schema", type=Path, default=ROOT / "schema/schema.json")
    ap.add_argument("--case-list", type=Path,
                    help="one case id per line; default = every case in the payload")
    ap.add_argument("--out", type=Path, help="output .jsonl (omit for audit only)")
    ap.add_argument("--include-arch", action="store_true",
                    help="also emit rows for the panoramic / 3D / sinus calls. "
                         "They are ~19%% of calls but carry ~47%% of the "
                         "asserted GT facts, so tooth-only training addresses "
                         "about half the scored surface. Off by default because "
                         "every arm measured so far was tooth-only and the "
                         "comparison has to stay honest.")
    ap.add_argument("--drop-fdis", type=Path, metavar="JSON",
                    help="{case_id: [fdi, ...]} to exclude, from "
                         "audit_report_facts.py's true contradictions. NOT the "
                         "whole audit: 150 of its findings are `crowns names X "
                         "which the same block calls absent`, which is a bridge "
                         "PONTIC or an implant crown -- a real finding, and "
                         "dropping it would delete exactly the evidence the "
                         "restoration fields are trained on. Only "
                         "present-and-absent-at-once is unusable, and after the "
                         "2026-08-14 manual verification that is 36 teeth in 24 "
                         "cases, 0.3%% of the corpus.")
    ap.add_argument("--evidence-from", type=Path,
                    help="a predictions/ dir whose visual_evidence strings become "
                         "the target text (see code/train/visual_evidence/README.md). Without it every evidence "
                         "string is empty, which is a DIFFERENT experiment -- "
                         "see --allow-empty-evidence.")
    ap.add_argument("--allow-empty-evidence", action="store_true",
                    help="emit rows with empty visual_evidence anyway. The field "
                         "carries no loss either way, but its tokens condition "
                         "the fields after it, so this changes what is learned.")
    ap.add_argument("--refuse", nargs="*", default=None, metavar="FIELD",
                    help="fields to leave unsupervised, replacing the default "
                         f"set ({', '.join(sorted(DEFAULT_REFUSED))})")
    ap.add_argument("--supervise", nargs="*", default=(), metavar="FIELD",
                    help="fields to supervise even if refused by default. A "
                         "bare name frees a tooth field from DEFAULT_REFUSED; "
                         "a dotted fact_id.field frees an arch field from "
                         "ARCH_REFUSED (e.g. maxilla_sinus_left.scope, which "
                         "arm 5 left unsupervised and which then drifted to "
                         "38/38 wrong -- see note 3 above)")
    ap.add_argument("--neg-cap", type=float, default=6.0, metavar="R",
                    help="§3.5: mask supervised boolean negatives down to R x "
                         "positives, per field. 0 disables.")
    ap.add_argument("--seed", type=int, default=42,
                    help="which negatives get masked -- must match across arms")
    # Evidence README. 0 keeps the prose in the target for conditioning but out of the
    # loss. Above 0 it is supervised WHERE THE SCREEN KEPT IT -- pass
    # --evidence-from a screened dir, or this trains on unverified prose.
    #
    # The default is not a taste: on the 2026-08-14 pool the surviving evidence
    # is ~3.4 strings x ~68 tokens = ~230 prose tokens per row against ~10
    # tokens of supervised decision content, so ~23:1. At full weight ~96% of
    # the gradient would teach radiology prose -- a LANGUAGE task, which hands
    # arm 3 an advantage that has nothing to do with H1b and corrupts the one
    # contrast the experiment exists for. 0.04 makes the two contribute
    # comparable loss mass.
    ap.add_argument("--evidence-weight", type=float, default=0.04,
                    help="loss weight for visual_evidence tokens; 0 = masked")
    args = ap.parse_args()

    refused = set(args.refuse) if args.refuse is not None else set(DEFAULT_REFUSED)
    refused -= set(args.supervise)
    print(f"[INFO] refused fields: {sorted(refused) or 'none'}")
    # Arch refusals live in their own namespace: DEFAULT_REFUSED holds bare
    # field names, ARCH_REFUSED holds `fact_id.field`, so one --supervise list
    # can serve both without a dotted key ever colliding with a bare one.
    arch_refused = set(ARCH_REFUSED) - set(args.supervise)
    if args.include_arch:
        freed = set(ARCH_REFUSED) & set(args.supervise)
        print(f"[INFO] refused arch fields: {sorted(arch_refused) or 'none'}"
              + (f" -- SUPERVISING {sorted(freed)} (plan log: a refused constant "
                 f"drifts, it does not stay put)" if freed else ""))
    print(f"[INFO] visual_evidence weight {args.evidence_weight:g}"
          + (" (masked)" if args.evidence_weight <= 0 else
             " -- supervised only where the screen kept the prose"))
    if SUPERVISED_BY_DECISION - refused:
        print(f"[INFO] supervised by explicit decision despite a measured "
              f"ceiling: {sorted(SUPERVISED_BY_DECISION - refused)} "
              f"-- see the note in this file; score it separately")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    order = schema_field_order(schema)
    arch_order = arch_field_order(schema) if args.include_arch else {}

    drop_fdis = {}
    if args.drop_fdis:
        drop_fdis = {k: set(v) for k, v in
                     json.loads(args.drop_fdis.read_text(encoding="utf-8")).items()}
        print(f"[INFO] --drop-fdis: {sum(len(v) for v in drop_fdis.values())} "
              f"tooth/case pair(s) across {len(drop_fdis)} case(s)")

    # The payload must have been built from THIS schema. Content hash, not
    # mtime: every file under code/ and schema/ carries an mtime from a
    # checkout with no content change (plan §6).
    pin = args.qa_jsonl.parent / "SCHEMA.sha256"
    if pin.exists():
        want = pin.read_text(encoding="utf-8").split()[0]
        got = sha256(args.schema)
        if want != got:
            sys.exit(f"[FAIL] payload was built from a different schema.\n"
                     f"       {pin}: {want}\n       {args.schema}: {got}")
        print(f"[PASS] schema hash matches the payload's pin ({got[:12]}…)")
    else:
        print(f"[WARN] no SCHEMA.sha256 beside {args.qa_jsonl} -- payload/schema "
              f"agreement is unverified")

    if not args.evidence_from and not args.allow_empty_evidence:
        sys.exit("[FAIL] --evidence-from is required (see the evidence README): the target's "
                 "visual_evidence must be TEACHER prose -- a model shown the "
                 "image and the already-established answer, told not to "
                 "re-judge it, as draft_evidence.py does. The generated GT "
                 "has it empty, "
                 "and training on empty teaches the model to skip the field "
                 "that makes it look before it answers.\n"
                 "       That needs a teacher pass over this pool, followed by "
                 "--check-perceivable screening.\n"
                 "       Pass --allow-empty-evidence to build the audit anyway.")

    forbidden = validate_case_ids()

    wanted = None
    if args.case_list:
        wanted = {l.strip() for l in args.case_list.read_text().splitlines()
                  if l.strip() and not l.startswith("#")}

    rows, dropped, refused_cases = [], collections.Counter(), []
    per_field = collections.defaultdict(lambda: collections.Counter())
    n_cases = 0

    for line in args.qa_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        case_id = rec["case_id"]

        if case_id in forbidden:
            refused_cases.append(case_id)          # plan §0
            continue
        if wanted is not None and case_id not in wanted:
            continue

        gt_path = args.gt_dir / f"{case_id}_gt.json"
        if not gt_path.exists():
            dropped["no_gt_file"] += 1
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        n_cases += 1

        ev = None
        if args.evidence_from:
            p = args.evidence_from / f"{case_id}_pred.json"
            if p.exists():
                pred = json.loads(p.read_text(encoding="utf-8"))
                ev = pred.get("teeth") or {}

        for tooth_key, call in (rec.get("dental_elements") or {}).items():
            fdi = int(tooth_key.split("_")[1])
            if fdi in drop_fdis.get(case_id, ()):
                dropped["audit_contradiction"] += 1
                continue
            row, why = build_row(case_id, fdi, call, gt.get("teeth") or {},
                                 gt.get("_derivation") or {}, order,
                                 (ev or {}).get(tooth_key) if ev else None,
                                 refused, args.evidence_weight > 0)
            if row is None:
                dropped[why] += 1
                continue
            rows.append(row)

        if args.include_arch:
            # Evidence is deliberately NOT passed: the teacher/screen chain of
            # the teacher/screen chain was run for dental_elements only, and an unscreened teacher
            # string is the "confident prose about something it cannot see"
            # failure the screen exists to catch. Arch rows carry
            # visual_evidence masked; at weight 0.04 that costs almost nothing.
            for call_key, call in (rec.get("global") or {}).items():
                row, why = build_arch_row(case_id, call_key, call,
                                          gt.get("global") or {}, arch_order,
                                          None, False, arch_refused)
                if row is None:
                    dropped[f"arch:{why}"] += 1
                    continue
                rows.append(row)

    apply_negative_cap(rows, args.neg_cap, args.seed)

    for row in rows:
        for fact_id, fields in row["mask"].items():
            short = short_fact(fact_id)
            for field, w in fields.items():
                per_field[f"{short}.{field}"][w or "SUPERVISED"] += 1

    if refused_cases:
        print(f"[INFO] §0: refused {len(refused_cases)} validate case(s): "
              f"{sorted(set(refused_cases))[:8]}")
    else:
        print(f"[PASS] §0: no validate case id appeared in the payload "
              f"({len(forbidden)} checked)")

    print(f"\n[INFO] {n_cases} case(s), {len(rows)} tooth call(s)")
    print(f"[INFO] dropped calls: {dict(dropped) or 'none'}")

    # Prompt parity: render one row through the pipeline's own builder. Doing
    # it here means a payload the collator cannot render fails now, at zero
    # GPU cost, rather than in the first training step.
    if rows:
        probe = rows[0]
        blocks, js = RVI.build_call_prompt(
            probe["call"], RVI.TOOTH_USER_TEMPLATE,
            extra_fmt={"fdi": probe["fdi"],
                       "tooth_reading": "\n" + RVI.HOW_TO_READ_A_TOOTH})
        if blocks is None:
            sys.exit("[FAIL] build_call_prompt returned no blocks for "
                     f"{probe['case_id']} tooth {probe['fdi']}")
        n_img = sum(1 for b in blocks if b.get("type") == "image_url")
        print(f"[PASS] prompt parity: {probe['case_id']} tooth {probe['fdi']} "
              f"renders {len(blocks)} block(s), {n_img} image(s), "
              f"json_schema {'present' if js else 'ABSENT'}")

    print(f"\n{'field':40}{'SUPERVISED':>11}" +
          "".join(f"{r:>10}" for r in MASK_REASONS))
    sup_tot = collections.Counter()
    for field in sorted(per_field):
        c = per_field[field]
        sup_tot["SUPERVISED"] += c["SUPERVISED"]
        for r in MASK_REASONS:
            sup_tot[r] += c[r]
        print(f"{field:40}{c['SUPERVISED']:>11}" +
              "".join(f"{c[r] or '':>10}" for r in MASK_REASONS))
    total = sum(sup_tot.values())
    print(f"{'TOTAL':40}{sup_tot['SUPERVISED']:>11}" +
          "".join(f"{sup_tot[r]:>10}" for r in MASK_REASONS))
    if total:
        print(f"\n[INFO] {sup_tot['SUPERVISED']:,} of {total:,} field slots "
              f"supervised ({100 * sup_tot['SUPERVISED'] / total:.1f}%)")

    # Positive/negative balance on the supervised booleans -- §3.5's 6:1 cap
    # is about this number, and "emit false" is the degenerate solution it
    # guards against.
    print(f"\n{'supervised boolean':40}{'true':>8}{'false':>8}{'ratio':>10}")
    for field in sorted(per_field):
        vals = collections.Counter()
        for row in rows:
            for fact_id, fields in row["mask"].items():
                short = short_fact(fact_id)
                for f, w in fields.items():
                    if w or f"{short}.{f}" != field:
                        continue
                    v = row["target"][fact_id][f]
                    if isinstance(v, bool):
                        vals[v] += 1
        if vals:
            t, f = vals[True], vals[False]
            ratio = f"{f / t:.1f}:1" if t else "all false"
            flag = "  <- past 6:1" if t and f / t > 6 else ""
            print(f"{field:40}{t:>8}{f:>8}{ratio:>10}{flag}")

    print(f"\n{'prefix':10}{'cases':>7}{'calls':>8}{'share':>8}")
    by = collections.Counter(r["prefix"] for r in rows)
    cs = collections.defaultdict(set)
    for r in rows:
        cs[r["prefix"]].add(r["case_id"])
    for p in sorted(by):
        print(f"{p:10}{len(cs[p]):>7}{by[p]:>8}{100 * by[p] / len(rows):>7.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[INFO] wrote {len(rows)} row(s) -> {args.out}")
    else:
        print("\n[INFO] audit only (no --out); nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
