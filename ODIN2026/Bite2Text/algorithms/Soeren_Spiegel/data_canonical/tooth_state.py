"""Per-tooth state labels from a canonical record — dense-negative supervision
for the auxiliary head.

The canonical record holds only the teeth an
annotator chose to mention (median 8 of 32), so a healthy tooth has no entry at all. Trained on
that alone the model sees "tooth 36 absent" in the 13 val cases where it is and *nothing* in the
other 49; it never receives a gradient for "36 is present and unremarkable", and it over-claims
absence by 3.5x. This module turns the record into a 32-slot label vector in
which an unmentioned tooth, in an arch that is in the scan, is a POSITIVE example of ``PRESENT``.

Two label groups, with different masks, and the difference is load-bearing:

  * **status** (4-way) is trained on every in-FOV tooth, mentioned or not. "Unmentioned => the
    tooth is there" is a safe weak label, which is where the dense negative supervision comes from.
  * **attributes** (6 independent binaries) are trained ONLY on teeth that carry an entry.
    Annotators omit fillings they did not think worth reporting, so "unmentioned => no filling"
    is a *false* negative and would be dense noise rather than dense supervision.

What is masked out of the weak PRESENT label, and why (all measured on train, see ``--report``):

  1. **Arches that are not in the scan.** ``record["fov"][arch].status`` must be ``included``.
     Only 61/498 train cases have an ``included`` maxilla (183 ``excluded``, 164 ``partial``,
     90 ``not_assessable``), so nearly all the dense supervision is mandibular — the mandible is
     ``included`` in 494/498. ``superior_truncated`` (image-derived, ``RenderedCase``) is a second,
     independent kill switch for the upper arch: the FOV record is the annotator's opinion of the
     scan, the truncation flag is the pixels'.
  2. **Third molars 18/28/38/48.** They are the most-omitted tooth *and* the most-abnormal tooth,
     which is exactly the combination the weak label cannot survive. Among the 38 train cases whose
     annotator charted >=20 teeth — i.e. was demonstrably exhaustive — tooth 36 is charted in 95%
     and 46 in 92%, but 38 in only 76% and 48 in 87%; and of the third molars that ARE charted,
     86%/79% are absent, impacted or a root remnant (36: 61%). So an exhaustive annotator skips a
     third molar four times more often than a first molar while three quarters of the ones they do
     record are abnormal: "unmentioned" there is silence, not health. Explicit third-molar entries
     are still trained on (they are 337 of the 410 impactions in the corpus).
  3. **Edentulous / arch-completeness cases.** ``P462``'s ``teeth`` dict is *empty* and its complete
     edentulism lives in ``other_findings`` free text; ``P506`` charts two teeth and says "Only 43
     and 33 present in arch" in prose. Both renders show a bare, implant-borne ridge — the weak
     label would have called all 16 mandibular teeth PRESENT. ``arch_statement_gate`` catches both
     (and all 17 empty-``teeth`` train cases) off the free text; it fires on 131/498 train cases,
     which lose their weak labels but keep every explicit entry.
  4. **Mixed dentition.** 6 train cases chart a deciduous code (5x/6x/7x/8x); their permanent
     successors are unerupted, not "present and unremarkable".

The FDI slot order is defined ONCE here (``FDI_ORDER``) and imported everywhere else. Under a
left-right image mirror the order permutes by ``MIRROR_PERM`` (= ``i ^ 8``: Q1<->Q2, Q3<->Q4, tooth
position unchanged), and every gate above is mirror-invariant by construction, so
``tooth_state(mirror_record(r))[i] == tooth_state(r)[i ^ 8]``. ``--test`` asserts exactly that over
all 622 records; ``dataset_fiveviews`` relies on it and derives labels from the mirrored record.

Usage:
    python data_canonical/tooth_state.py --report            # train label distribution
    python data_canonical/tooth_state.py --test              # mirror equivariance + invariants
"""
from __future__ import annotations

import re

import numpy as np

# ── The canonical slot order. Defined here, never re-derived. ────────────────────
FDI_ORDER: tuple[str, ...] = tuple(f"{q}{i}" for q in "1234" for i in range(1, 9))
FDI_INDEX: dict[str, int] = {c: i for i, c in enumerate(FDI_ORDER)}
N_SLOTS = len(FDI_ORDER)                      # 32
# Left-right mirror permutation on the slot axis. FDI_ORDER is four contiguous blocks of eight
# (Q1,Q2,Q3,Q4) and the mirror swaps Q1<->Q2 and Q3<->Q4 without moving tooth position, which is
# a flip of the block's low bit — i.e. XOR 8. Asserted in _test_mirror.
MIRROR_PERM: tuple[int, ...] = tuple(i ^ 8 for i in range(N_SLOTS))

STATUS_CLASSES: tuple[str, ...] = ("present", "absent", "impacted", "root_remnant")
STATUS_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATUS_CLASSES)}
N_STATUS = len(STATUS_CLASSES)
IGNORE_INDEX = -100                            # matches torch.nn.functional.cross_entropy default

ATTRIBUTES: tuple[str, ...] = (
    "implant", "crown", "filling", "endo", "caries", "periapical_lesion")
N_ATTR = len(ATTRIBUTES)

# restorations -> attribute. implant_crown is BOTH an implant and a crown (it is the crown
# screwed onto the fixture, and the corpus records it instead of, not alongside, `implant` on
# 86 teeth). bridge_abutment / prosthetic_pontic / post_and_core have no attribute of their own.
_RESTOR_ATTRS = {
    "implant": ("implant",),
    "implant_crown": ("implant", "crown"),
    "prosthetic_crown": ("crown",),
    "filling": ("filling",),
}
# lesion -> attribute. The three periapical variants collapse to one head; root_fracture,
# peri_implant_resorption and other have no head (too few, and not what the judge trace flags).
_LESION_ATTRS = {
    "caries": ("caries",),
    "periapical_lesion": ("periapical_lesion",),
    "periapical_granuloma": ("periapical_lesion",),
    "periapical_cyst": ("periapical_lesion",),
}

THIRD_MOLARS: frozenset[str] = frozenset({"18", "28", "38", "48"})

# Free text that says something about how much of the arch is *there*, in a form the per-tooth
# `teeth` dict does not carry. Matched against every string in the record (see `_iter_strings`).
# Mirror-invariant: `mirror_record` only permutes FDI digits and swaps side words, so a record
# that matches before the flip matches after it.
_ARCH_STATEMENT_RE = re.compile(
    r"edentul|adentul|anodont"          # "complete edentulism", "edentulous saddles 33-37"
    r"|cawood"                          # Cawood-Howell classifies an EDENTULOUS ridge (44 cases)
    r"|knife[-\s]edge"                  # knife-edge ridge => long-standing tooth loss
    r"|\bonly\b[^.;]{0,60}\b(?:present|remaining|remains|in the arch)",   # "Only 43 and 33 present"
    re.IGNORECASE,
)


def _arch_of(fdi: str) -> str:
    """FDI quadrant -> the ``fov`` key that gates it. 1,2 = upper; 3,4 = lower."""
    return "maxilla" if fdi[0] in "12" else "mandible"


def _iter_strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _iter_strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _iter_strings(x)


def arch_statement_gate(record) -> bool:
    """True when the record says something about arch completeness that the ``teeth`` dict does
    not encode per tooth — edentulism, a resorbed ridge, "only X remaining", a denture. Such a
    case gets NO weak PRESENT labels (its explicit entries are still supervised).

    This is not a nicety. P462 records complete edentulism only in ``other_findings`` and leaves
    ``teeth`` empty; P506 charts 2 teeth and puts "Only 43 and 33 present in arch" in prose. The
    renders show bare ridges. Without this gate both cases would teach 16 false PRESENT labels."""
    for p in (record.get("prostheses") or []):
        if (p.get("type") or "") == "denture":
            return True
    for s in _iter_strings({k: v for k, v in record.items() if k != "case_id"}):
        if _ARCH_STATEMENT_RE.search(s):
            return True
    return False


def has_deciduous(record) -> bool:
    """True when the record charts a deciduous code (quadrants 5-8). Mixed dentition: the
    permanent successor is unerupted, so it is neither 'present and unremarkable' nor charted."""
    return any(k not in FDI_INDEX for k in (record.get("teeth") or {}))


def _attrs_of(entry) -> np.ndarray:
    a = np.zeros(N_ATTR, dtype=np.float32)
    for r in (entry.get("restorations") or []):
        for name in _RESTOR_ATTRS.get(r, ()):
            a[ATTRIBUTES.index(name)] = 1.0
    if entry.get("endo"):
        a[ATTRIBUTES.index("endo")] = 1.0
    for name in _LESION_ATTRS.get(entry.get("lesion") or "", ()):
        a[ATTRIBUTES.index(name)] = 1.0
    return a


def tooth_state(record, superior_truncated: bool = False,
                mask_third_molars: bool = True,
                keep_explicit_out_of_fov: bool = True):
    """Canonical record -> ``(labels, mask)`` over ``FDI_ORDER``.

    ``labels["status"]``  int64[32]   class index, or ``IGNORE_INDEX`` where unsupervised.
    ``labels["attr"]``    float32[32,6] 0/1 targets, meaningless where masked.
    ``mask["status"]``    bool[32]     True = supervised (== labels["status"] != IGNORE_INDEX).
    ``mask["attr"]``      bool[32,6]   True = supervised (uniform across the 6 for a given tooth).

    ``superior_truncated`` is the image-derived flag from ``RenderedCase``; when True the whole
    upper arch is unsupervised regardless of what the annotator said about the maxilla.

    ``mask_third_molars`` drops only the WEAK label on 18/28/38/48 — an explicit entry there is
    still supervised. ``keep_explicit_out_of_fov`` keeps explicitly-charted teeth supervised even
    when their arch failed the FOV gate: the annotator naming a tooth is proof it was visible, and
    the gate exists to protect the *unmentioned* slots, not to throw away real statements. Set it
    False for the literal FOV-first reading (``--report`` prints both)."""
    teeth = record.get("teeth") or {}
    fov = record.get("fov") or {}

    arch_ok = {}
    for arch in ("maxilla", "mandible"):
        ok = (fov.get(arch) or {}).get("status") == "included"
        if arch == "maxilla" and superior_truncated:
            ok = False
        arch_ok[arch] = ok

    # Case-level kill switches for the WEAK label only.
    weak_ok = not (arch_statement_gate(record) or has_deciduous(record))

    status = np.full(N_SLOTS, IGNORE_INDEX, dtype=np.int64)
    attr = np.zeros((N_SLOTS, N_ATTR), dtype=np.float32)
    attr_mask = np.zeros((N_SLOTS, N_ATTR), dtype=bool)

    for fdi, i in FDI_INDEX.items():
        entry = teeth.get(fdi)
        in_fov = arch_ok[_arch_of(fdi)]
        if entry is not None:
            if not (in_fov or keep_explicit_out_of_fov):
                continue
            cls = STATUS_INDEX.get(entry.get("status"))
            if cls is not None:                      # unknown status -> leave unsupervised
                status[i] = cls
            attr[i] = _attrs_of(entry)
            attr_mask[i] = True
        elif in_fov and weak_ok and not (mask_third_molars and fdi in THIRD_MOLARS):
            status[i] = STATUS_INDEX["present"]      # the weak label — the point of this module

    labels = {"status": status, "attr": attr}
    mask = {"status": status != IGNORE_INDEX, "attr": attr_mask}
    return labels, mask


# ── report / self-test ──────────────────────────────────────────────────────────

def _load(split: str):
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "splits" / "canonical" / f"{split}.jsonl"
    return [json.loads(l) for l in p.open()]


def _report(split: str = "train") -> None:
    import collections
    rows = _load(split)
    for keep in (True, False):
        n_weak = collections.Counter()
        n_expl = collections.Counter()
        n_attr_pos = collections.Counter()
        n_attr_sup = 0
        sup_per_case, attr_per_case = [], []
        for r in rows:
            rec = r["record"]
            labels, mask = tooth_state(rec, keep_explicit_out_of_fov=keep)
            teeth = rec.get("teeth") or {}
            for i, fdi in enumerate(FDI_ORDER):
                if labels["status"][i] == IGNORE_INDEX:
                    continue
                cls = STATUS_CLASSES[labels["status"][i]]
                (n_expl if fdi in teeth else n_weak)[cls] += 1
            sup_per_case.append(int(mask["status"].sum()))
            attr_per_case.append(int(mask["attr"][:, 0].sum()))
            n_attr_sup += int(mask["attr"][:, 0].sum())
            for j, a in enumerate(ATTRIBUTES):
                n_attr_pos[a] += int((labels["attr"][:, j] * mask["attr"][:, j]).sum())

        tot = sum(n_weak.values()) + sum(n_expl.values())
        print(f"\n=== {split}: {len(rows)} cases, keep_explicit_out_of_fov={keep} ===")
        print(f"status slots supervised: {tot} of {len(rows) * N_SLOTS} "
              f"({tot / (len(rows) * N_SLOTS):.1%}); "
              f"median {int(np.median(sup_per_case))}/case, "
              f"masked {N_SLOTS - int(np.median(sup_per_case))}/case")
        print(f"{'class':<14}{'weak':>8}{'explicit':>10}{'total':>8}")
        for c in STATUS_CLASSES:
            print(f"{c:<14}{n_weak[c]:>8}{n_expl[c]:>10}{n_weak[c] + n_expl[c]:>8}")
        others = max(n_weak[c] + n_expl[c] for c in STATUS_CLASSES if c != "present")
        print(f"weak present {n_weak['present']} vs largest other class {others} "
              f"-> {n_weak['present'] / max(others, 1):.1f}x")
        print(f"attribute slots supervised: {n_attr_sup} "
              f"(median {int(np.median(attr_per_case))}/case)")
        for a in ATTRIBUTES:
            print(f"   {a:<18}{n_attr_pos[a]:>6} pos "
                  f"({n_attr_pos[a] / max(n_attr_sup, 1):.1%} of supervised)")

    # the gates, quantified
    n_arch = sum(1 for r in rows if arch_statement_gate(r["record"]))
    n_dec = sum(1 for r in rows if has_deciduous(r["record"]))
    n_empty = sum(1 for r in rows if not r["record"].get("teeth"))
    n_empty_gated = sum(1 for r in rows
                        if not r["record"].get("teeth") and arch_statement_gate(r["record"]))
    print(f"\ngates: arch-statement {n_arch}/{len(rows)}, deciduous {n_dec}/{len(rows)}; "
          f"empty-teeth cases {n_empty}, of which gated {n_empty_gated}")

    # ── Is the FOV gate right? The weak-PRESENT count is NOT free to be large: it is bounded
    # by the in-FOV, uncharted slots, and the maxilla is `included` in almost no case. Compare
    # the shipped count against that arithmetic ceiling, and against the ceiling with each gate
    # switched off, so a low weak/other ratio can be attributed rather than guessed at.
    ceil_fov = ceil_no3m = 0
    for r in rows:
        rec = r["record"]
        teeth = rec.get("teeth") or {}
        for fdi in FDI_ORDER:
            if fdi in teeth:
                continue
            if (rec.get("fov") or {}).get(_arch_of(fdi), {}).get("status") != "included":
                continue
            ceil_no3m += 1
            if fdi not in THIRD_MOLARS:
                ceil_fov += 1
    def n_weak_labels(rec, **kw):
        teeth = rec.get("teeth") or {}
        lb, _ = tooth_state(rec, **kw)
        return sum(1 for i, f in enumerate(FDI_ORDER)
                   if f not in teeth and lb["status"][i] != IGNORE_INDEX)

    n_shipped = sum(n_weak_labels(r["record"]) for r in rows)
    n_lift3m = sum(n_weak_labels(r["record"], mask_third_molars=False) for r in rows)
    n_incl_max = sum(1 for r in rows
                     if (r["record"].get("fov") or {}).get("maxilla", {}).get("status") == "included")
    print(f"\nFOV-gate audit (this is why weak PRESENT is not 10x the other classes):")
    print(f"   maxilla 'included' in {n_incl_max}/{len(rows)} cases -> the upper arch is almost "
          f"entirely unsupervised by design; supervision is essentially mandible-only")
    print(f"   uncharted in-FOV slots, incl. third molars : {ceil_no3m}")
    print(f"   ... excluding third molars (the ceiling)   : {ceil_fov}")
    print(f"   weak PRESENT actually emitted              : {n_shipped}  "
          f"({n_shipped / max(ceil_fov, 1):.0%} of the ceiling; the shortfall is the "
          f"{n_arch + n_dec} arch-statement/deciduous cases)")
    print(f"   weak PRESENT if third molars were kept     : {n_lift3m}")
    print(f"   -> the FOV gate is not suppressing labels; the in-FOV ceiling itself is "
          f"{ceil_fov / (len(rows) * N_SLOTS):.0%} of all slots.")

    # the third-molar decision, with the numbers behind it
    import collections
    exhaustive = [r for r in rows if len(r["record"].get("teeth") or {}) >= 20]
    print(f"\nthird-molar decision (mask_third_molars=True), measured on the {len(exhaustive)} "
          f"cases whose annotator charted >=20 teeth:")
    print(f"{'tooth':<8}{'charted':>10}{'abnormal|charted':>19}")
    for c in ("38", "48", "36", "46", "18", "28", "16", "26"):
        ch = [r["record"]["teeth"][c] for r in exhaustive if c in r["record"]["teeth"]]
        abn = sum(1 for t in ch if t["status"] != "present")
        print(f"{c:<8}{len(ch)}/{len(exhaustive)} ({len(ch) / len(exhaustive):>4.0%}){'':>2}"
              f"{abn / max(len(ch), 1):>15.0%}")
    print("   -> third molars are the most-omitted AND the most-abnormal slot; masked.")

    # the "no perception" floor the head must beat (acceptance criterion 4)
    prior = np.zeros((N_SLOTS, N_STATUS))
    for r in rows:
        labels, mask = tooth_state(r["record"])
        for i in range(N_SLOTS):
            if mask["status"][i]:
                prior[i, labels["status"][i]] += 1
    with np.errstate(invalid="ignore"):
        acc = np.nansum(prior.max(1)) / max(prior.sum(), 1)
    print(f"\nper-tooth training prior ('no perception' floor): "
          f"argmax accuracy {acc:.3f} on {split}")


def _test_mirror() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mirror import mirror_record

    # the permutation itself
    for i, fdi in enumerate(FDI_ORDER):
        q, pos = fdi[0], fdi[1]
        swapped = {"1": "2", "2": "1", "3": "4", "4": "3"}[q] + pos
        assert MIRROR_PERM[i] == FDI_INDEX[swapped], (fdi, swapped)

    n = 0
    for split in ("train", "val", "test"):
        for r in _load(split):
            rec = r["record"]
            for trunc in (False, True):
                la, ma = tooth_state(rec, superior_truncated=trunc)
                lb, mb = tooth_state(mirror_record(rec), superior_truncated=trunc)
                perm = list(MIRROR_PERM)
                assert np.array_equal(lb["status"], la["status"][perm]), \
                    f"{r['case_id']} status not mirror-equivariant (truncated={trunc})"
                assert np.array_equal(lb["attr"], la["attr"][perm]), \
                    f"{r['case_id']} attr not mirror-equivariant (truncated={trunc})"
                assert np.array_equal(mb["attr"], ma["attr"][perm]), \
                    f"{r['case_id']} attr mask not mirror-equivariant (truncated={trunc})"
                # mask/label consistency, which the trainer relies on for ignore_index
                assert np.array_equal(ma["status"], la["status"] != IGNORE_INDEX)
            n += 1
    print(f"mirror equivariance: OK over {n} records x 2 truncation settings")

    # truncation must only ever remove upper-arch supervision, never add any
    for r in _load("train")[:120]:
        l0, m0 = tooth_state(r["record"], superior_truncated=False)
        l1, m1 = tooth_state(r["record"], superior_truncated=True)
        assert np.array_equal(l0["status"][16:], l1["status"][16:]), "truncation touched the mandible"
        assert (m1["status"][:16] <= m0["status"][:16]).all(), "truncation ADDED upper supervision"
    print("superior_truncated: OK (upper arch only, monotone)")

    # attributes are never supervised for an unmentioned tooth
    for split in ("train", "val", "test"):
        for r in _load(split):
            _, mask = tooth_state(r["record"])
            charted = {FDI_INDEX[k] for k in r["record"].get("teeth", {}) if k in FDI_INDEX}
            assert set(np.nonzero(mask["attr"][:, 0])[0].tolist()) <= charted, r["case_id"]
    print("attribute mask: OK (charted teeth only)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the label distribution")
    ap.add_argument("--split", default="train")
    ap.add_argument("--test", action="store_true", help="mirror equivariance + invariants")
    a = ap.parse_args()
    if a.test:
        _test_mirror()
    if a.report or not a.test:
        _report(a.split)
