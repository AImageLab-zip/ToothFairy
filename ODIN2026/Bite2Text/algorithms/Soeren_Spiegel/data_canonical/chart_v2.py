"""Per-annotator records -> one all-teeth MULTILABEL chart per case.

Each EN report is represented by one structured record (schema: `validate.py`); this
module merges them into the 32-slot target the round trains on.

**Why multilabel.** The v1 chart kept one token per slot via a fixed priority
(`absent > impacted > root remnant > implant > lesion > crown > endo > filling > present`), and
measured on train that collapse hid **46% of endo positives** (674 true -> 363 charted), **61%
of implant**, 34% crown, 15% filling. The hidden slots were not merely unsupervised — they
carried a *different* class, so a crowned root-treated tooth trained as evidence *against* endo.
A slot here is a SET, so `36 = {absent, implant}` is representable and nothing is destroyed.

**Why `healthy` replaces `present`.** 265/1000 reports explicitly attest health ("Teeth from 18
to 27 are present in the arch"), and the v1 pipeline discarded 753 such slots. The empty set
*is* healthy, so the common case costs one short token and every positive is a real finding.

**The merge, and what it refuses to do.**

  * **Attributes union across annotators.** r1 "34 absent" + r2 "34 implant" -> `{absent,
    implant}`. Both survive; that pair is clinically ordinary (a fixture in a healed socket).
  * **The status axis is exclusive** — `absent` / `impacted` / `root_remnant`, or none. When two
    annotators disagree there, this module does **not** pick a winner: it masks the status axis
    of that slot, keeps the attribute union, and records the disagreement in `conflicts`. The
    fallback rule in (prefer the more exhaustive annotator) is deliberately NOT implemented
    — `--report` prints the conflict count so the decision is made against data, and if the
    count is small the rule is never needed.

**Two supervision masks, and the difference is the same one `tooth_state` drew.**

  * `mask_status` — a slot is supervised when some annotator charted it, or when it is
    unmentioned in an arch that is in the scan (the weak `healthy` label, the dense negative
    supervision). Masked on a status conflict.
  * `mask_attr` — **only** slots some annotator actually charted. An unmentioned tooth is weak
    evidence of "there", never of "no filling": annotators omit restorations they did not think
    worth reporting, so a zero there is a false negative, not supervision. What changed since v1
    is that the *explicit* set is now huge (8422 slots/case-union vs 5358), because attested-
    present teeth are charted.

**The gates carry over from `tooth_state` unchanged, and one is new.** FOV, third molars,
arch statements and mixed dentition all still mask the *weak* label only. New, from the chart-v2
audit: **positive attestation overrides the FOV gate** — F054 says "Maxilla partially included"
and then names teeth 18-27 present, and an annotator naming a tooth is proof they saw it. That
is `tooth_state`'s `keep_explicit_out_of_fov=True` principle, promoted to a rule.

Usage:
    python data_canonical/chart_v2.py --report        # label + conflict distribution
    python data_canonical/chart_v2.py --test          # invariants + mirror equivariance
    python data_canonical/chart_v2.py --show A019     # one case
    python data_canonical/chart_v2.py --nontooth      # are the non-tooth enums
                                                      #   harmonised, or do they hold synonyms?
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mirror import mirror_record  # noqa: E402
from tooth_state import (  # noqa: E402
    FDI_INDEX, FDI_ORDER, MIRROR_PERM, N_SLOTS, THIRD_MOLARS, arch_statement_gate,
)
# The format lives on the container side and this module imports it, never the reverse —
# `data_canonical/` is not vendored into the submission image.
from toothfairy.chart_fmt_v2 import (  # noqa: E402
    BONE_FLAGS, FOV_REGIONS, chart_block, parse_chart,
)

RAW_DIR = os.path.join(_HERE, "raw_reports")

# ── the label space ─────────────────────────────────────────────────────────────
# The empty set is `healthy`. It is not a member of LABELS: making it one would let a slot be
# both healthy and absent, which is the contradiction the whole design removes.
STATUS_LABELS: tuple[str, ...] = ("absent", "impacted", "root_remnant")
ATTR_LABELS: tuple[str, ...] = ("implant", "crown", "endo", "lesion", "filling")
LABELS: tuple[str, ...] = STATUS_LABELS + ATTR_LABELS
LABEL_INDEX: dict[str, int] = {c: i for i, c in enumerate(LABELS)}
N_LABELS = len(LABELS)
HEALTHY = "healthy"

# Restoration -> attribute. Identical to `tooth_state._RESTOR_ATTRS`: no new labelling logic, so
# the classes stay comparable to the v1 numbers. implant_crown is BOTH (it is the crown screwed
# onto the fixture, and the corpus records it instead of, not alongside, `implant`).
# bridge_abutment / prosthetic_pontic / post_and_core carry no attribute of their own.
_RESTOR_ATTRS: dict[str, tuple[str, ...]] = {
    "implant": ("implant",),
    "implant_crown": ("implant", "crown"),
    "prosthetic_crown": ("crown",),
    "filling": ("filling",),
}
# Lesion -> the single `lesion` attribute. root_fracture / peri_implant_resorption / other /
# root_remnant map to nothing, exactly as in `tooth_state` (too few, and not what the judge
# trace flags).
_LESION_KINDS = frozenset({"caries", "periapical_lesion", "periapical_granuloma",
                           "periapical_cyst"})
_STATUS_MAP = {"absent": "absent", "impacted": "impacted", "root_remnant": "root_remnant"}
# A status the annotator declined to give. The slot is unsupervised FOR THAT ANNOTATOR — it is
# not evidence of health.
_NO_CLAIM = frozenset({"not_assessable", "not_mentioned"})

# Attribute/status pairs that cannot both be true of one tooth. `absent` + implant / crown /
# lesion are all clinically ordinary (fixture in a healed socket, its crown, a residual
# periapical lucency) and are NOT listed — resolving those would destroy the very combination
# showed the v1 chart was losing.
_INCOMPATIBLE_WITH_ABSENT = frozenset({"endo", "filling"})


def _slot_labels(entry: dict) -> tuple[set[str], bool]:
    """One per-report tooth entry -> ``(labels, has_status_claim)``.

    ``has_status_claim`` is False when the annotator charted the tooth but declined its status
    (`not_assessable`), which supervises the attributes without supervising the status axis."""
    out: set[str] = set()
    status = entry.get("status")
    claim = status not in _NO_CLAIM and status is not None
    if status in _STATUS_MAP:
        out.add(_STATUS_MAP[status])
    for r in (entry.get("restorations") or []):
        out.update(_RESTOR_ATTRS.get(r, ()))
    if entry.get("endo"):
        out.add("endo")
    if (entry.get("lesion") or "") in _LESION_KINDS:
        out.add("lesion")
    return out, claim


def load_case_records(case_id: str, raw_dir: str = RAW_DIR) -> list[dict]:
    """Every per-report record of one case, ordered by rid so the merge is deterministic."""
    out = []
    for fn in sorted(os.listdir(raw_dir)):
        if fn.startswith(f"{case_id}__") and fn.endswith(".json"):
            with open(os.path.join(raw_dir, fn)) as fh:
                out.append(json.load(fh))
    return out


def all_case_ids(raw_dir: str = RAW_DIR) -> list[str]:
    return sorted({fn.split("__", 1)[0] for fn in os.listdir(raw_dir) if fn.endswith(".json")})


def _arch_of(fdi: str) -> str:
    return "maxilla" if fdi[0] in "12" else "mandible"


def _has_deciduous(records: list[dict]) -> bool:
    return any(k not in FDI_INDEX for r in records for k in (r.get("teeth") or {}))


def merge_case(records: list[dict], superior_truncated: bool = False) -> dict:
    """Per-annotator records -> the merged 32-slot multilabel chart.

    Returns ``{"labels": {fdi: set[str]}, "mask_status": {fdi: bool},
    "mask_attr": {fdi: bool}, "conflicts": [...]}``. A slot whose label set is empty and whose
    status mask is True is ``healthy``."""
    per_annotator: dict[str, list[tuple[set[str], bool]]] = defaultdict(list)
    for rec in records:
        for fdi, entry in (rec.get("teeth") or {}).items():
            if fdi not in FDI_INDEX:            # deciduous: handled by the case-level gate
                continue
            per_annotator[fdi].append(_slot_labels(entry))

    # ── The arch gate. Two levels, and the difference is what the chart is FOR.
    #
    # This chart is a perception target: it teaches the model to read teeth and pathology off
    # the volume. So the governing question is not "did an annotator say something" but "is it
    # in the image at all". An arch the scan does not contain is **fully masked** — every slot
    # `n/a` — even where an annotator charted it, because that claim cannot be learned from
    # pixels that do not exist and supervising it teaches exactly the hallucination the FOV
    # gate was built to stop.
    #
    # `partial` is different and keeps the chart-v2 F054 rule: the arch IS in the volume, just
    # clipped, so a tooth an annotator named is a tooth they saw. Those stay supervised; only
    # the weak `healthy` label is withheld, since silence about a clipped arch is not evidence
    # of health.
    fov = merged_fov(records)
    arch_ok, arch_masked = {}, {}
    for arch in ("maxilla", "mandible"):
        status = fov[arch]
        masked = status in _ARCH_ABSENT_FROM_SCAN
        ok = status == "included"
        if arch == "maxilla" and superior_truncated:
            masked, ok = True, False        # the pixels' opinion overrides the annotator's
        arch_masked[arch] = masked
        arch_ok[arch] = ok and not masked

    weak_ok = not (any(arch_statement_gate(r) for r in records) or _has_deciduous(records))

    labels: dict[str, set[str]] = {}
    mask_status: dict[str, bool] = {}
    mask_attr: dict[str, bool] = {}
    conflicts: list[dict] = []

    for fdi in FDI_ORDER:
        if arch_masked[_arch_of(fdi)]:
            # the arch is not in the volume: nothing here is learnable, so nothing is charted
            labels[fdi] = set()
            mask_status[fdi] = False
            mask_attr[fdi] = False
            continue
        votes = per_annotator.get(fdi)
        if votes:
            # ── explicitly charted by at least one annotator
            union: set[str] = set()
            status_votes: set[frozenset[str]] = set()
            for lab, claim in votes:
                union |= lab
                if claim:
                    status_votes.add(frozenset(lab & set(STATUS_LABELS)))
            attrs = union & set(ATTR_LABELS)
            statuses = union & set(STATUS_LABELS)

            status_ok = bool(status_votes)
            if len(status_votes) > 1:
                # annotators disagree about whether/how the tooth is there -> do not guess
                conflicts.append({"fdi": fdi, "kind": "status_disagreement",
                                  "votes": sorted(sorted(v) for v in status_votes)})
                statuses = set()
                status_ok = False
            elif len(statuses) > 1:
                # one annotator asserting two exclusive statuses: an extraction artifact
                conflicts.append({"fdi": fdi, "kind": "status_multiple_in_one_record",
                                  "votes": [sorted(statuses)]})
                statuses = set()
                status_ok = False
            elif "absent" in statuses and (attrs & _INCOMPATIBLE_WITH_ABSENT):
                conflicts.append({"fdi": fdi, "kind": "absent_with_incompatible_attr",
                                  "attrs": sorted(attrs & _INCOMPATIBLE_WITH_ABSENT)})
                statuses = set()
                status_ok = False

            labels[fdi] = statuses | attrs
            mask_status[fdi] = status_ok
            mask_attr[fdi] = True
        else:
            # ── unmentioned by every annotator: the weak `healthy` label, or nothing
            weak = (arch_ok[_arch_of(fdi)] and weak_ok and fdi not in THIRD_MOLARS)
            labels[fdi] = set()
            mask_status[fdi] = weak
            mask_attr[fdi] = False

    return {"labels": labels, "mask_status": mask_status, "mask_attr": mask_attr,
            "conflicts": conflicts}


def chart_v2(case_id: str, superior_truncated: bool = False,
             raw_dir: str = RAW_DIR) -> dict:
    return merge_case(load_case_records(case_id, raw_dir), superior_truncated)


def _pick(values, order):
    """Merge one single-valued field across annotators.

    Unlike a tooth slot these fields hold ONE value, so a disagreement cannot be unioned. Rule:
    ignore `not_mentioned` (silence is not a claim), take the most frequent remaining value, and
    break ties toward the **more restrictive** reading — over-claiming that an arch was in the
    scan is what manufactures false health claims, so the tie-break errs the safe way."""
    from collections import Counter

    vals = [v for v in values if v and v != "not_mentioned"]
    if not vals:
        return "not_mentioned"
    c = Counter(vals)
    best = max(c.values())
    tied = [v for v in vals if c[v] == best]
    return min(tied, key=lambda v: order.index(v) if v in order else len(order))


# most restrictive first — the tie-break direction
_FOV_ORDER = ("excluded", "not_assessable", "partial", "included")
_SINUS_ORDER = ("excluded", "not_assessable", "opacified", "mucosal_thickening", "clear")
_CANAL_ORDER = ("not_assessable", "apico_lingual", "buccal", "lingual", "regular")
# An arch with one of these statuses is NOT IN THE VOLUME, so nothing about its teeth is
# learnable from the image and every slot is masked — see `merge_case`.
_ARCH_ABSENT_FROM_SCAN = frozenset({"excluded", "not_assessable"})


def merged_fov(records: list[dict]) -> dict[str, str]:
    """``{region: status}`` merged across annotators. Used by BOTH the tooth gate and the
    reported FOV line, so the chart cannot say an arch is out of the scan and then chart it."""
    return {reg: _pick([((r.get("fov") or {}).get(reg) or {}).get("status") for r in records],
                       _FOV_ORDER)
            for reg in FOV_REGIONS}


def merge_nontooth(records: list[dict]) -> dict:
    """The non-tooth half of the chart, merged across annotators."""
    sinus_order, canal_order = _SINUS_ORDER, _CANAL_ORDER
    fov = merged_fov(records)
    ian = {}
    for side in ("left", "right"):
        ian[side] = _pick([((r.get("ian_canal") or {}).get(side) or {}).get("course")
                           for r in records], canal_order)
    sinuses = _pick([(r.get("sinuses") or {}).get("status") for r in records], sinus_order)

    bone = []
    for r in records:
        bp = r.get("bone_and_perio") or {}
        a = bp.get("atrophy")
        if isinstance(a, dict) and a.get("present") and "atrophy" not in bone:
            bone.append("atrophy")
        elif a is True and "atrophy" not in bone:
            bone.append("atrophy")
        if "periodontitis" in bp and "periodontitis" not in bone:
            bone.append("periodontitis")
    return {"fov": fov, "ian": ian, "sinuses": sinuses,
            "bone": [b for b in BONE_FLAGS if b in bone]}


def chart_text_from_records(records: list[dict], superior_truncated: bool = False,
                            collapse_healthy: bool = True) -> str:
    """The frozen block for records already in hand.

    The dataset needs this rather than ``chart_text``: under the augmentation mirror augmentation the
    target must be built from *mirrored* records, and `superior_truncated` is only known after
    the render. Loading from disk per sample would also put 1000 file reads inside the training
    loop.

    ``collapse_healthy=True`` (the training target) spells the empty set ``n/a``, the same as an
    unsupervised slot. Pass ``False`` for the **gold** chart — `eval/eval_chart_v2.py` skips
    unsupervised slots and projects the rest onto `present`, and both need the distinction the
    target throws away."""
    m = merge_case(records, superior_truncated)
    nt = merge_nontooth(records)
    slots = {fdi: (sorted(m["labels"][fdi], key=LABEL_INDEX.__getitem__)
                   if m["mask_status"][fdi] or m["mask_attr"][fdi] else None)
             for fdi in FDI_ORDER}
    return chart_block(slots, fov=nt["fov"], ian=nt["ian"], sinuses=nt["sinuses"],
                       bone=nt["bone"], collapse_healthy=collapse_healthy)


def chart_text(case_id: str, superior_truncated: bool = False, raw_dir: str = RAW_DIR,
               collapse_healthy: bool = True) -> str:
    """The full frozen block for one case — this is the training target."""
    return chart_text_from_records(load_case_records(case_id, raw_dir), superior_truncated,
                                   collapse_healthy=collapse_healthy)


def load_all_records(raw_dir: str = RAW_DIR) -> dict[str, list[dict]]:
    """``{case_id: [record, ...]}`` for the whole corpus, read once. The dataset holds this so
    a training step costs no disk I/O and the mirror path can re-merge in memory."""
    out: dict[str, list[dict]] = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json") or "__" not in fn:
            continue
        with open(os.path.join(raw_dir, fn)) as fh:
            out.setdefault(fn.split("__", 1)[0], []).append(json.load(fh))
    return out


def chart_slots(case_id: str, superior_truncated: bool = False,
                raw_dir: str = RAW_DIR) -> dict:
    """``{fdi: sorted list | None}`` — the same content ``parse_chart`` returns, so round-trip can
    compare them directly."""
    m = merge_case(load_case_records(case_id, raw_dir), superior_truncated)
    return {fdi: (sorted(m["labels"][fdi], key=LABEL_INDEX.__getitem__)
                  if m["mask_status"][fdi] or m["mask_attr"][fdi] else None)
            for fdi in FDI_ORDER}


def slot_text(label_set: set[str]) -> str:
    """A slot's labels in canonical order, or ``healthy`` for the empty set. The order is
    ``LABELS`` order so the string is a function of the set, not of iteration order."""
    if not label_set:
        return HEALTHY
    return "+".join(sorted(label_set, key=LABEL_INDEX.__getitem__))


# ── report / self-test ──────────────────────────────────────────────────────────

def _report() -> None:
    import collections
    ids = all_case_ids()
    lab_count = collections.Counter()
    per_slot = collections.Counter()
    conf = collections.Counter()
    n_sup_status = n_sup_attr = n_healthy = 0
    n_multi = 0
    cases_with_conflict = 0
    combo = collections.Counter()
    for cid in ids:
        m = chart_v2(cid)
        if m["conflicts"]:
            cases_with_conflict += 1
        for c in m["conflicts"]:
            conf[c["kind"]] += 1
        for fdi in FDI_ORDER:
            s = m["labels"][fdi]
            if m["mask_status"][fdi]:
                n_sup_status += 1
                if not s:
                    n_healthy += 1
            if m["mask_attr"][fdi]:
                n_sup_attr += 1
            for x in s:
                lab_count[x] += 1
            if s:
                per_slot[len(s)] += 1
            if len(s) > 1:
                n_multi += 1
                combo[slot_text(s)] += 1

    tot = len(ids) * N_SLOTS
    print(f"=== chart v2 over {len(ids)} cases ({tot} slots) ===\n")
    print(f"status-axis supervised : {n_sup_status:>6} ({n_sup_status / tot:.0%})")
    print(f"   of which healthy    : {n_healthy:>6} ({n_healthy / max(n_sup_status, 1):.0%})")
    print(f"attribute supervised   : {n_sup_attr:>6} ({n_sup_attr / tot:.0%})")
    print(f"\n{'label':<14}{'slots':>8}")
    for c in LABELS:
        print(f"   {c:<11}{lab_count[c]:>8}")
    print(f"\nslots carrying >1 label: {n_multi}  "
          f"(the v1 chart could represent none of these)")
    print(f"label-count per non-healthy slot: {dict(sorted(per_slot.items()))}")
    print(f"\ncommonest multilabel combinations:")
    for k, v in combo.most_common(12):
        print(f"   {k:<28}{v:>6}")
    n_conf = sum(conf.values())
    print(f"\nconflicts: {n_conf} over {cases_with_conflict} cases  {dict(conf)}")
    rate = n_conf / max(n_sup_status + n_conf, 1)
    print(f"   = {rate:.2%} of slots that would otherwise carry a supervised status")
    if rate < 0.02:
        print("   -> masking the status axis is the whole answer; the 'more exhaustive "
              "annotator' fallback is NOT needed and should be deleted rather than built.")

    # ── the point of the round, stated against the v1 chart it replaces
    # (`python data_canonical/chart_render.py --report`, same 622 cases)
    v1 = {"absent": 1915, "impacted": 392, "root_remnant": 98, "implant": 122,
          "crown": 392, "endo": 469, "lesion": 425, "filling": 960}
    print(f"\n{'label':<14}{'v1 chart':>10}{'v2 chart':>10}{'recovered':>11}")
    for c in LABELS:
        d = lab_count[c] - v1[c]
        print(f"   {c:<11}{v1[c]:>10}{lab_count[c]:>10}{d:>+11}")
    print(f"   {'TOTAL':<11}{sum(v1.values()):>10}{sum(lab_count[c] for c in LABELS):>10}"
          f"{sum(lab_count[c] for c in LABELS) - sum(v1.values()):>+11}")
    print("   v1 lost these to the one-token-per-slot collapse; nothing here is new labelling.")


def _nontooth_report() -> None:
    """Are the four non-tooth lines HARMONISED, or do their vocabularies carry synonyms?

    Asked before non-tooth weighting wired a class weight to every (field, value) pair:
    weighting a distinction the annotators do not themselves make spends the round's lever on
    label noise. The test is empirical — the 367 multi-report cases give two independent reads
    of the same volume, so the pairwise agreement matrix says which values are interchangeable
    in practice rather than in the schema.

    Read the output in this order:

      1. the merged distribution, which is the prior the loss inverts;
      2. **agreement %**, which is the ceiling on what any model can score;
      3. the off-diagonal cells, which say *what* is being confused with what.
    """
    import collections
    ids = all_case_ids()
    names = ([f"FOV.{r}" for r in FOV_REGIONS]
             + ["IAN.left", "IAN.right", "Sinuses", "Bone.atrophy", "Bone.periodontitis"])
    dist = collections.defaultdict(collections.Counter)
    pair = collections.defaultdict(collections.Counter)
    n_multi = 0

    for cid in ids:
        recs = load_case_records(cid)
        nt = merge_nontooth(recs)
        for r in FOV_REGIONS:
            dist[f"FOV.{r}"][nt["fov"][r]] += 1
        for s in ("left", "right"):
            dist[f"IAN.{s}"][nt["ian"][s]] += 1
        dist["Sinuses"][nt["sinuses"]] += 1
        for b in BONE_FLAGS:
            dist[f"Bone.{b}"]["present" if b in nt["bone"] else "absent"] += 1

        if len(recs) < 2:
            continue
        n_multi += 1
        per: dict[str, list[str]] = {}
        for r in FOV_REGIONS:
            per[f"FOV.{r}"] = [((rec.get("fov") or {}).get(r) or {}).get("status")
                               or "not_mentioned" for rec in recs]
        for s in ("left", "right"):
            per[f"IAN.{s}"] = [((rec.get("ian_canal") or {}).get(s) or {}).get("course")
                               or "not_mentioned" for rec in recs]
        per["Sinuses"] = [(rec.get("sinuses") or {}).get("status") or "not_mentioned"
                          for rec in recs]
        for rec in recs:
            bp = rec.get("bone_and_perio") or {}
            a = bp.get("atrophy")
            a = a.get("present") if isinstance(a, dict) else a
            per.setdefault("Bone.atrophy", []).append("present" if a else "absent")
            per.setdefault("Bone.periodontitis", []).append(
                "present" if "periodontitis" in bp else "absent")
        for f, vv in per.items():
            for i in range(len(vv)):
                for j in range(i + 1, len(vv)):
                    pair[f][tuple(sorted((vv[i], vv[j])))] += 1

    print(f"=== non-tooth fields over {len(ids)} cases "
          f"({n_multi} of them multi-report) ===\n")
    print(f"{'field':<28}merged distribution (all 622 cases)")
    for f in names:
        tot = sum(dist[f].values())
        print(f"   {f:<25}" + "  ".join(f"{v}:{c}({c / tot:.0%})"
                                        for v, c in dist[f].most_common()))

    print(f"\n{'field':<28}{'agree':>7}   top disagreements (unordered annotator pairs)")
    for f in names:
        p = pair[f]
        tot = sum(p.values())
        if not tot:
            continue
        agree = sum(c for k, c in p.items() if k[0] == k[1])
        dis = sorted(((k, c) for k, c in p.items() if k[0] != k[1]), key=lambda x: -x[1])
        print(f"   {f:<25}{agree / tot:>6.0%}   "
              + ", ".join(f"{a}|{b}:{c}" for (a, b), c in dis[:4]))

    print("""
READING (measured 2026-08-14, and the reason chart_loss_v2 carries a warning):

 * ⚠ **IAN is NOT a harmonised enum.** 37%/39% agreement against 65-93% everywhere else, and
   the largest cell in its matrix is `lingual|regular` (180 pairs pooled over both sides) —
   bigger than `regular|regular` (140) or `lingual|lingual` (105). The cause is in the
   vocabulary: `CANAL_COURSE` mixes two AXES. `regular` describes the canal's course,
   `lingual`/`buccal`/`apico_lingual` its bucco-lingual position, and they are not exclusive —
   the notes under `course=lingual` read "regular course, predominantly lingual". Whichever
   fact the extractor caught first became the label.
 * ⚠ **`apico_lingual` is a synonym-in-use of `lingual`** — a refinement, 39 cross-pairs against
   8 `apico_lingual|apico_lingual` self-pairs. `_CANAL_ORDER` already resolves the pair toward
   `apico_lingual`, so the target asserts the finer reading on ~30 cases where the other
   annotator said plain `lingual`.
 * **`not_mentioned` is a reporting-behaviour class, not a perception class**, on the two
   fields where it dominates: `FOV.coronoid_processes` disagrees `excluded|not_mentioned` on 93
   pairs against **3** `excluded|excluded` self-pairs; `FOV.maxillary_sinuses` on 81. What the
   pixels show there is presumably as stable as `FOV.mandibular_condyles` (97% excluded, 91%
   agreement); what varies is whether an annotator bothered to say so.
 * **`FOV.maxilla excluded|partial` (50 pairs) is a genuine boundary disagreement**, not a
   synonym — and it is load-bearing, because `excluded` masks the whole upper arch's teeth.
 * The rest hold up: Sinuses 93%, condyles 91%, Bone.atrophy 85%, periodontitis 89%.

None of this is fixed by a loss weight. Harmonising `CANAL_COURSE` means changing the format,
the parser, the prose template and the metric together — a vocabulary round, with new labelling
logic, which is exactly what the chart-v2 round forbade itself. What non-tooth weighting does instead is state
the ceiling: **read any IAN movement against 37% annotator agreement**, and prefer
`FOV.maxilla` / `Bone.atrophy` (65% / 85%, both near-balanced priors) as the fields where a
real signal is legible.""")


def _show(case_id: str) -> None:
    recs = load_case_records(case_id)
    m = chart_v2(case_id)
    print(f"=== {case_id} ({len(recs)} report(s)) ===")
    for q in "1234":
        row = []
        for fdi in FDI_ORDER:
            if fdi[0] != q:
                continue
            if not m["mask_status"][fdi] and not m["mask_attr"][fdi]:
                row.append(f"{fdi} -")
            else:
                row.append(f"{fdi} {slot_text(m['labels'][fdi])}")
        print(f"  Q{q}: " + ", ".join(row))
    print(f"  conflicts: {m['conflicts'] or 'none'}")
    print(f"  ('-' = unsupervised; 'healthy' = supervised, no finding)")


def _test() -> None:
    ids = all_case_ids()
    assert len(ids) == 622, f"expected 622 cases, found {len(ids)}"

    n_conf = 0
    for cid in ids:
        recs = load_case_records(cid)
        m = merge_case(recs)
        assert set(m["labels"]) == set(FDI_ORDER), f"{cid}: not all 32 slots present"
        for fdi in FDI_ORDER:
            s = m["labels"][fdi]
            assert s <= set(LABELS), f"{cid}.{fdi}: unknown label in {s}"
            # the status axis is exclusive
            assert len(s & set(STATUS_LABELS)) <= 1, f"{cid}.{fdi}: two statuses {s}"
            # a masked-status slot never carries a status label
            if not m["mask_status"][fdi]:
                assert not (s & set(STATUS_LABELS)), f"{cid}.{fdi}: status label under a mask"
            # attributes only where an annotator charted the tooth
            if not m["mask_attr"][fdi]:
                assert not s, f"{cid}.{fdi}: labels on an uncharted slot"
            # the resolved contradiction never survives
            if "absent" in s:
                assert not (s & _INCOMPATIBLE_WITH_ABSENT), f"{cid}.{fdi}: absent+{s}"
        n_conf += len(m["conflicts"])
    print(f"partition + exclusivity: OK over {len(ids)} cases ({n_conf} conflicts)")

    # ── truncation only ever removes upper-arch supervision, never adds any
    for cid in ids[:150]:
        recs = load_case_records(cid)
        a = merge_case(recs, superior_truncated=False)
        b = merge_case(recs, superior_truncated=True)
        for fdi in FDI_ORDER:
            if fdi[0] in "34":
                assert a["labels"][fdi] == b["labels"][fdi], f"{cid}: truncation hit the mandible"
                assert a["mask_status"][fdi] == b["mask_status"][fdi], cid
            else:
                assert b["mask_status"][fdi] <= a["mask_status"][fdi], \
                    f"{cid}.{fdi}: truncation ADDED upper supervision"
    print("superior_truncated: OK (upper arch only, monotone)")

    # ── mirroring the records permutes the chart by quadrant swap and nothing else
    qswap = {"1": "2", "2": "1", "3": "4", "4": "3"}
    n_mir = 0
    for cid in ids:
        recs = load_case_records(cid)
        a = merge_case(recs)
        b = merge_case([mirror_record(r) for r in recs])
        for fdi in FDI_ORDER:
            f2 = qswap[fdi[0]] + fdi[1]
            assert b["labels"][f2] == a["labels"][fdi], \
                f"{cid}: {fdi}->{f2} not mirror-equivariant ({a['labels'][fdi]} vs {b['labels'][f2]})"
            assert b["mask_status"][f2] == a["mask_status"][fdi], f"{cid}: {fdi} status mask"
            assert b["mask_attr"][f2] == a["mask_attr"][fdi], f"{cid}: {fdi} attr mask"
        n_mir += 1
    print(f"mirror equivariance: OK over {n_mir} cases")

    # the slot permutation itself, against the shipped constant the dataset relies on
    for i, fdi in enumerate(FDI_ORDER):
        assert MIRROR_PERM[i] == FDI_INDEX[qswap[fdi[0]] + fdi[1]]
    print("MIRROR_PERM: OK (matches the quadrant swap)")

    # ── parse(render(case)) == chart_slots(case), exhaustive, both truncations.
    # Two modes since 2026-08-13. GOLD (uncollapsed) is still exact and is the one the metric
    # depends on. TARGET (collapsed) is exact only modulo `None -> []`, because `healthy` and
    # `n/a` now share a surface — asserted here so the lossy step is a checked property of the
    # format and not something a future reader has to infer from a failing test.
    n_rt = n_collapsed_slots = 0
    for cid in ids:
        for trunc in (False, True):
            want = chart_slots(cid, superior_truncated=trunc)

            gold_txt = chart_text(cid, superior_truncated=trunc, collapse_healthy=False)
            got = parse_chart(gold_txt, collapsed=False)
            assert got is not None, f"{cid}: gold chart failed to parse (trunc={trunc})"
            assert got["slots"] == want, (
                f"{cid} gold round-trip mismatch (trunc={trunc}): "
                f"{ {k: v for k, v in want.items() if got['slots'].get(k) != v} }")

            txt = chart_text(cid, superior_truncated=trunc)
            assert HEALTHY not in txt, f"{cid}: collapsed target still spells `healthy`"
            got_c = parse_chart(txt)
            assert got_c is not None, f"{cid}: target chart failed to parse (trunc={trunc})"
            assert got_c["slots"] == {f: (v or []) for f, v in want.items()}, (
                f"{cid} collapsed round-trip mismatch (trunc={trunc})")
            n_collapsed_slots += sum(1 for v in want.values() if v is None)

            recs = load_case_records(cid)
            nt = merge_nontooth(recs)
            assert got["fov"] == nt["fov"] and got["ian"] == nt["ian"], f"{cid}: non-tooth"
            assert got["sinuses"] == nt["sinuses"] and got["bone"] == nt["bone"], f"{cid}: bone"
            n_rt += 1
    print(f"round-trip: OK over {len(ids)} cases x 2 truncations ({n_rt} charts); "
          f"gold exact, target exact modulo {n_collapsed_slots} masked slots folded into n/a")

    # ── a malformed block must return None, never a partial guess
    good = chart_text(ids[0])
    for broken, why in (
        (good.replace("Q2:", "Q1:", 1), "duplicated quadrant"),
        (good.replace("11 ", "19 ", 1), "invalid FDI"),
        (good.split("\n", 2)[0] + "\n" + good.split("\n")[1], "missing quadrants"),
        (good.replace("n/a", "sparkling", 1), "unknown label"),
        (good + "\nJunk: whatever", "unrecognised line"),
        (good.replace("Sinuses: ", "Sinuses: bright ", 1), "invalid enum"),
    ):
        assert parse_chart(broken) is None, f"parser accepted a broken block ({why})"
    print("parser rejection: OK (6 malformed blocks refused)")

    # ── slot_text is a function of the SET, not of insertion order
    import random
    rng = random.Random(2026)
    for _ in range(500):
        k = rng.randint(1, 4)
        s = set(rng.sample(LABELS, k))
        assert slot_text(s) == slot_text(set(reversed(sorted(s)))), s
    assert slot_text(set()) == HEALTHY
    print("slot_text: OK (order-independent, empty == healthy)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--show", default=None)
    ap.add_argument("--nontooth", action="store_true",
                    help=" are the FOV/IAN/Sinuses/Bone vocabularies harmonised, or do "
                         "they carry synonyms? Distributions + inter-annotator agreement.")
    a = ap.parse_args()
    if a.test:
        _test()
    if a.show:
        _show(a.show)
    if a.nontooth:
        _nontooth_report()
    if a.report or not (a.test or a.show or a.nontooth):
        _report()
