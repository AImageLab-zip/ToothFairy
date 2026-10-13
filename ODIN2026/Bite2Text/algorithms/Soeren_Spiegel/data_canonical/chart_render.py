"""Chart target renderer — canonical record -> the frozen chart block.

The chart-QA target is ``chart block + delimiter + report``. This module builds the first half
from the **existing** state-vector machinery in ``tooth_state.py``: the same labels, the same
``arch_statement_gate``, the same third-molar gate, the same mixed-dentition gate, the same
per-slot FOV / ``superior_truncated`` mask. No new labelling logic — the gates are where the
real work went and they stay the single source of truth. Everything here is a *projection* of
``tooth_state`` onto one token per slot, plus string assembly that lives in
``toothfairy/chart_parse.py`` so the parser is the exact inverse of one function.

**The flattening.** ``tooth_state`` returns two label groups per slot — a 4-way status and 6
independent attribute binaries — and the frozen format has room for one token. ``CHART_PRIORITY``
resolves it, **status first**:

    absent > impacted > root remnant > implant > lesion > crown > endo > filling > present

Two consequences worth stating, because both look wrong at a glance:

  * **Status wins over attributes**, so the 120 corpus slots that are ``absent`` *and* carry an
    implant chart as ``absent``, not ``implant``. That is not information thrown away: the prose
    half of the same target says exactly the same thing — ``render_prose._tooth_clause`` returns
    a bare "tooth NN is absent" for an absent tooth and drops its restorations. A chart that
    disagreed with the report it is glued to would poison the chart<->report consistency metric
    it exists to enable. It also makes the projection back to the 4 status classes
    (``chart_parse.CHART_TO_STATUS``) lossless, which is what makes chart macro-F1 comparable to
    the auxiliary head's 0.514 rather than a different number with the same name.
  * **Attributes are ordered rarest-first**, so a crowned, root-treated tooth with a periapical
    lesion charts as ``lesion``. The rare, clinically decisive finding survives the collapse; the
    routine restoration is the one that loses.

``filling`` is charted even though the original 7-class sketch
omitted it. It is one of the six shipped ``ATTRIBUTES``, it is the single most common one
(1100+ slots), and dropping it would be exactly the "new labelling logic" chart forbids.

Usage:
    python data_canonical/chart_render.py --test          # round-trip + mirror equivariance
    python data_canonical/chart_render.py --report        # class distribution + token cost
    python data_canonical/chart_render.py --show P462     # one case's target
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toothfairy.chart_parse import (  # noqa: E402
    CHART_CLASSES, CHART_INDEX, CHART_TO_STATUS, DELIMITER, chart_block, parse_chart,
)
from tooth_state import (  # noqa: E402  (same directory; callers put it on sys.path)
    ATTRIBUTES, FDI_INDEX, FDI_ORDER, IGNORE_INDEX, N_SLOTS, STATUS_CLASSES, has_deciduous,
    tooth_state,
)

# (label group, name) in resolution order. The first hit wins. See the module docstring for why
# status comes first and why the attributes are ordered the way they are.
CHART_PRIORITY: tuple[tuple[str, str], ...] = (
    ("status", "absent"), ("status", "impacted"), ("status", "root_remnant"),
    ("attr", "implant"), ("attr", "lesion"), ("attr", "crown"), ("attr", "endo"),
    ("attr", "filling"),
)
# tooth_state has no single 'lesion' attribute: caries and periapical_lesion are separate heads
# and both collapse to it (the corpus's other lesion kinds have no attribute at all).
_LESION_ATTRS = ("caries", "periapical_lesion")
_STATUS_TOKEN = {"absent": "absent", "impacted": "impacted", "root_remnant": "root remnant"}


def _slot_class(status_idx: int, attr_row: np.ndarray) -> str:
    """One (status, attributes) pair -> one chart class token."""
    status = STATUS_CLASSES[status_idx]
    for group, name in CHART_PRIORITY:
        if group == "status":
            if status == name:
                return _STATUS_TOKEN[name]
        elif name == "lesion":
            if any(attr_row[ATTRIBUTES.index(a)] for a in _LESION_ATTRS):
                return "lesion"
        elif attr_row[ATTRIBUTES.index(name)]:
            return name
    return "present"


def chart_labels(record, superior_truncated: bool = False) -> np.ndarray:
    """int64[32] of ``CHART_CLASSES`` indices over ``FDI_ORDER``, ``IGNORE_INDEX`` where the
    ``tooth_state`` gates leave the slot unsupervised. This is the reference the chart is
    scored against and the right-hand side of the round-trip law."""
    labels, mask = tooth_state(record, superior_truncated=superior_truncated)
    out = np.full(N_SLOTS, IGNORE_INDEX, dtype=np.int64)
    for i in range(N_SLOTS):
        if mask["status"][i]:
            out[i] = CHART_INDEX[_slot_class(int(labels["status"][i]), labels["attr"][i])]
    return out


def chart_state(record, superior_truncated: bool = False) -> dict[str, str]:
    """``{fdi: class}`` over the charted slots — the same content as ``chart_labels``, in the
    shape ``chart_parse.parse_chart`` returns, so round-trip can compare them directly.

    Mixed dentition returns ``{}``: those 6 records chart a deciduous code, their permanent
    successors are unerupted, and the whole block collapses to the sentinel."""
    if has_deciduous(record):
        return {}
    lab = chart_labels(record, superior_truncated=superior_truncated)
    return {FDI_ORDER[i]: CHART_CLASSES[lab[i]]
            for i in range(N_SLOTS) if lab[i] != IGNORE_INDEX}


def chart_text(record, superior_truncated: bool = False) -> str:
    """The chart block for one record, in the frozen format."""
    if has_deciduous(record):
        return chart_block({}, mixed_dentition=True)
    state = chart_state(record, superior_truncated=superior_truncated)
    quadrants: dict[str, list[tuple[str, str]]] = {q: [] for q in "1234"}
    for fdi in FDI_ORDER:                       # FDI_ORDER is already Q1->Q4, ascending
        if fdi in state:
            quadrants[fdi[0]].append((fdi, state[fdi]))
    return chart_block(quadrants)


def concat_target(record, report: str, superior_truncated: bool = False) -> str:
    """The full chart-QA training target: chart block, frozen delimiter, canonical prose.

    ``report`` must already be the target the run would otherwise use — under the augmentation mirror
    augmentation that is ``render_prose.render(mirror_record(rec))`` and ``record`` must be the
    same mirrored record, or the chart and the prose disagree on laterality."""
    return chart_text(record, superior_truncated=superior_truncated) + DELIMITER + report


def status_projection(chart: dict[str, str]) -> dict[str, str]:
    """``{fdi: chart class}`` -> ``{fdi: tooth_state status class}``. Lossless (status-first
    priority), so a 4-class macro-F1 computed on this is the head's metric, not a lookalike."""
    return {k: CHART_TO_STATUS[v] for k, v in chart.items()}


# ── report / self-test ──────────────────────────────────────────────────────────

def _load(split: str):
    import json
    p = _REPO_ROOT / "splits" / "canonical" / f"{split}.jsonl"
    return [json.loads(l) for l in p.open()]


def _all_rows():
    return [r for split in ("train", "val", "test") for r in _load(split)]


def _test() -> None:
    """round-trip (round-trip) + mirror-equivariance (mirror equivariance), both exhaustive over all 622 records."""
    from mirror import mirror_is_safe, mirror_record

    rows = _all_rows()

    # ── parse(render(rec)) == chart_state(rec), over every record and both truncations
    n = 0
    for r in rows:
        for trunc in (False, True):
            want = chart_state(r["record"], superior_truncated=trunc)
            got = parse_chart(chart_text(r["record"], superior_truncated=trunc))
            assert got is not None, f"{r['case_id']}: chart failed to parse (trunc={trunc})"
            assert got == want, (f"{r['case_id']} round-trip mismatch (trunc={trunc}): "
                                 f"{ {k: v for k, v in want.items() if got.get(k) != v} }")
            n += 1
    print(f"round-trip: OK over {len(rows)} records x 2 truncation settings ({n} charts)")

    # ── chart(mirror(rec)) is chart(rec) with the quadrants swapped
    qswap = {"1": "2", "2": "1", "3": "4", "4": "3"}
    unsafe = set()
    for r in rows:
        rec = r["record"]
        if not mirror_is_safe(rec):
            unsafe.add(r["case_id"])
            continue
        for trunc in (False, True):
            a = parse_chart(chart_text(rec, superior_truncated=trunc))
            b = parse_chart(chart_text(mirror_record(rec), superior_truncated=trunc))
            want = {qswap[k[0]] + k[1]: v for k, v in a.items()}
            assert b == want, f"{r['case_id']}: chart not mirror-equivariant (trunc={trunc})"
    expected_unsafe = {"P208", "F029", "F056", "S0051", "S0020", "P119", "S0007"}
    print(f"mirror equivariance: OK over {len(rows) - len(unsafe)} safe records; "
          f"unsafe set = {sorted(unsafe)}")
    assert unsafe == expected_unsafe, (f"mirror-unsafe set changed: {sorted(unsafe)} != "
                                       f"{sorted(expected_unsafe)}")

    # ── the status projection is lossless: it must reproduce tooth_state's own status labels
    for r in rows:
        for trunc in (False, True):
            labels, mask = tooth_state(r["record"], superior_truncated=trunc)
            if has_deciduous(r["record"]):
                continue
            proj = status_projection(chart_state(r["record"], superior_truncated=trunc))
            for i, fdi in enumerate(FDI_ORDER):
                if mask["status"][i]:
                    assert proj[fdi] == STATUS_CLASSES[labels["status"][i]], \
                        f"{r['case_id']} {fdi}: status projection lost information"
    print("status projection: OK (lossless back to tooth_state.STATUS_CLASSES)")

    # ── the sentinel path
    dec = [r for r in rows if has_deciduous(r["record"])]
    for r in dec:
        assert parse_chart(chart_text(r["record"])) == {}, r["case_id"]
    print(f"mixed-dentition sentinel: OK on {len(dec)} records ({[r['case_id'] for r in dec]})")


def _report() -> None:
    import collections
    rows = _all_rows()
    cls = collections.Counter()
    per_case_slots, chart_chars = [], []
    n_max_empty = n_mand_empty = 0
    for r in rows:
        st = chart_state(r["record"])
        cls.update(st.values())
        per_case_slots.append(len(st))
        txt = chart_text(r["record"])
        chart_chars.append(len(txt))
        n_max_empty += "Maxilla: not charted" in txt
        n_mand_empty += "Mandible: not charted" in txt
    tot = sum(cls.values())
    print(f"\n=== chart classes over all {len(rows)} records ({tot} charted slots) ===")
    for c in CHART_CLASSES:
        print(f"   {c:<14}{cls[c]:>7}  ({cls[c] / max(tot, 1):>6.2%})")
    print(f"\ncharted slots/case: min {min(per_case_slots)} "
          f"median {int(np.median(per_case_slots))} max {max(per_case_slots)}")
    print(f"chart block chars : median {int(np.median(chart_chars))} max {max(chart_chars)}")
    print(f"'Maxilla: not charted'  fires on {n_max_empty}/{len(rows)} records")
    print(f"'Mandible: not charted' fires on {n_mand_empty}/{len(rows)} records")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="round-trip + mirror equivariance")
    ap.add_argument("--report", action="store_true", help="class distribution + size")
    ap.add_argument("--show", default=None, help="print one case's full concatenated target")
    a = ap.parse_args()
    if a.test:
        _test()
    if a.show:
        row = next(r for r in _all_rows() if r["case_id"] == a.show)
        print(concat_target(row["record"], row["prose"]))
    if a.report or not (a.test or a.show):
        _report()
