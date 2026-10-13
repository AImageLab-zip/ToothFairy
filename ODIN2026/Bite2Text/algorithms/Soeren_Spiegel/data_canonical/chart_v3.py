"""Chart-v3 merge — the VOCABULARY round. Splits `CANAL_COURSE` into two axes and turns
annotator silence into a mask.

The TOOTH half is `chart_v2`'s, imported unchanged: this round touches the non-tooth
vocabulary and nothing else, so a tooth-metric movement between a v2 and a v3 run would be a
bug, not a result. Only `merge_nontooth_v3` and the format are new.

**The measurement that justifies the round** (`python data_canonical/chart_v3.py --harmonise`):

    current mixed enum                              38.0% agreement  (n=786 annotator pairs)
    split -> `course`,   where both annotators claim 97.8%           (n=178)
    split -> `position`, where both annotators claim 87.2%           (n=203)

`regular` is the canal's *course*; `lingual`/`buccal` its bucco-lingual *position*. They are not
exclusive — the notes under `course=lingual` read "regular course, predominantly lingual" — so
forcing one enum made the two annotators look like they disagreed about anatomy when they had
merely written down different halves of the same finding. `lingual|regular` was the LARGEST cell
in the matrix (180 pairs). **The 37% ceiling was a schema artifact.** That is why non-tooth loss
weights moved every other field and left IAN flat, and why evaluation
found the IAN line scoring 15% entailment against combined_v1's 64%.

Derivation, per side, from v2's `(course, note)` — no new labelling, no re-extraction:

    value               -> course                              position
    regular             -> regular                             None (masked)
    lingual/apico_ling. -> regular if note says "regular" else None    lingual
    buccal              -> regular if note says "regular" else None    buccal
    not_assessable      -> not_assessable                       not_assessable
    not_mentioned/absent-> None (masked)                        None (masked)

⚠ **A position is NOT evidence of an irregular course.** Saying "predominantly lingual" says
nothing about whether the course is regular, so the course axis is left MASKED unless the note
says so explicitly. Asserting `irregular` there would invent a label the corpus never carries —
`irregular` is deliberately absent from `chart_fmt_v3.CANAL_COURSE`.

⚠ **`apico_lingual` folds into `lingual`** — a synonym-in-use (39 cross-pairs vs 8 self-pairs),
and v2's `_CANAL_ORDER` used to resolve the pair toward the finer reading, asserting it on ~30
cases where the other annotator said plain `lingual`.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toothfairy.chart_fmt_v3 import (  # noqa: E402
    BONE_FLAGS, CANAL_COURSE, CANAL_POSITION, FDI_ORDER, FOV_REGIONS, FOV_STATUS,
    SINUS_STATUS, chart_block, parse_chart,
)
from data_canonical.chart_v2 import (  # noqa: E402
    LABEL_INDEX, RAW_DIR, all_case_ids, load_case_records, load_all_records, merge_case, _pick,
)

# Most-restrictive-first, the tie-break direction — same convention as v2's orders.
# `not_mentioned` never appears: silence is not a value here, it is the absence of one, and it
# is dropped from the vote entirely (see `_pick_claimed`).
_FOV_ORDER = ("excluded", "not_assessable", "partial", "included")
_SINUS_ORDER = ("excluded", "not_assessable", "opacified", "mucosal_thickening", "clear")
_COURSE_ORDER = ("not_assessable", "regular")
_POSITION_ORDER = ("not_assessable", "buccal", "lingual")

_SILENT = frozenset({None, "not_mentioned", ""})


def split_axes(course: str | None, note: str | None) -> tuple[str | None, str | None]:
    """v2's single mixed value + its free-text note -> ``(course, position)``.

    ``None`` on either axis means the annotator made no claim about it.
    """
    n = (note or "").lower()
    if course in _SILENT:
        return None, None
    if course == "not_assessable":
        return "not_assessable", "not_assessable"
    if course == "regular":
        return "regular", None
    if course in ("lingual", "apico_lingual", "buccal"):
        pos = "buccal" if course == "buccal" else "lingual"
        # only an explicit "regular course" in the note licenses the course axis
        return ("regular" if "regular" in n else None), pos
    return None, None                      # an unknown value claims nothing


def _pick_claimed(values, order):
    """Merge across annotators, ignoring silence. ``None`` iff nobody claimed the axis.

    This is the second half of the round: under v2 a silent annotator voted `not_mentioned` and
    could win the merge, so one annotator's silence erased the other's finding. Silence is now
    simply not a vote.
    """
    claimed = [v for v in values if v not in _SILENT]
    return _pick(claimed, order) if claimed else None


def merged_fov_v3(records: list[dict]) -> dict[str, str | None]:
    return {reg: _pick_claimed([((r.get("fov") or {}).get(reg) or {}).get("status")
                                for r in records], _FOV_ORDER)
            for reg in FOV_REGIONS}


def merge_nontooth_v3(records: list[dict]) -> dict:
    """The non-tooth half under the v3 vocabulary."""
    course, position = {}, {}
    for side in ("left", "right"):
        pairs = [split_axes(((r.get("ian_canal") or {}).get(side) or {}).get("course"),
                            ((r.get("ian_canal") or {}).get(side) or {}).get("note"))
                 for r in records]
        course[side] = _pick_claimed([c for c, _ in pairs], _COURSE_ORDER)
        position[side] = _pick_claimed([p for _, p in pairs], _POSITION_ORDER)

    sinuses = _pick_claimed([(r.get("sinuses") or {}).get("status") for r in records],
                            _SINUS_ORDER)
    bone = []
    for r in records:
        bp = r.get("bone_and_perio") or {}
        a = bp.get("atrophy")
        if ((isinstance(a, dict) and a.get("present")) or a is True) and "atrophy" not in bone:
            bone.append("atrophy")
        if "periodontitis" in bp and "periodontitis" not in bone:
            bone.append("periodontitis")
    return {"fov": merged_fov_v3(records), "ian_course": course, "ian_position": position,
            "sinuses": sinuses, "bone": [b for b in BONE_FLAGS if b in bone]}


def chart_text_from_records(records: list[dict], superior_truncated: bool = False,
                            collapse_healthy: bool = True) -> str:
    """Same signature as `chart_v2.chart_text_from_records` so the dataset can swap them."""
    m = merge_case(records, superior_truncated)          # TOOTH HALF: v2's, unchanged
    nt = merge_nontooth_v3(records)
    slots = {fdi: (sorted(m["labels"][fdi], key=LABEL_INDEX.__getitem__)
                   if m["mask_status"][fdi] or m["mask_attr"][fdi] else None)
             for fdi in FDI_ORDER}
    return chart_block(slots, fov=nt["fov"], ian_course=nt["ian_course"],
                       ian_position=nt["ian_position"], sinuses=nt["sinuses"],
                       bone=nt["bone"], collapse_healthy=collapse_healthy)


def chart_text(case_id: str, superior_truncated: bool = False, raw_dir: str = RAW_DIR,
               collapse_healthy: bool = True) -> str:
    return chart_text_from_records(load_case_records(case_id, raw_dir), superior_truncated,
                                   collapse_healthy=collapse_healthy)


# ── the measurement that licenses the round ────────────────────────────────────────────────
def _harmonise() -> None:
    recs = collections.defaultdict(list)
    for cid in all_case_ids():
        recs[cid] = load_case_records(cid)

    cur = collections.Counter()
    ax = collections.Counter()
    for rs in recs.values():
        if len(rs) < 2:
            continue
        for a, b in itertools.combinations(rs, 2):
            for side in ("left", "right"):
                va = ((a.get("ian_canal") or {}).get(side) or {})
                vb = ((b.get("ian_canal") or {}).get(side) or {})
                ca, cb = va.get("course"), vb.get("course")
                if ca is None or cb is None:
                    continue
                cur["n"] += 1
                cur["agree"] += (ca == cb)
                pa = split_axes(ca, va.get("note"))
                pb = split_axes(cb, vb.get("note"))
                for i, name in ((0, "course"), (1, "position")):
                    x, y = pa[i], pb[i]
                    ax[name + "_all_n"] += 1
                    ax[name + "_all_ok"] += (x == y)
                    if x is not None and y is not None:
                        ax[name + "_claim_n"] += 1
                        ax[name + "_claim_ok"] += (x == y)

    print(f"\nIAN vocabulary harmonisation  ({cur['n']} annotator pairs, both sides pooled)\n")
    print(f"  v2 mixed enum                       {cur['agree'] / cur['n']:>6.1%}")
    for f in ("course", "position"):
        n_all, n_cl = ax[f + "_all_n"], ax[f + "_claim_n"]
        print(f"  v3 {f:<9} all pairs            {ax[f + '_all_ok'] / n_all:>6.1%}   (n={n_all})")
        print(f"  v3 {f:<9} BOTH annotators claim {ax[f + '_claim_ok'] / n_cl:>6.1%}   "
              f"(n={n_cl}, {n_cl / n_all:.0%} of pairs)")
    print("""
READING. The split recovers agreement from 38% to 88-98% on the pairs where both annotators
actually made the claim — in line with every other field (65-93%). The remaining ~75% of pairs
are one-sided silence, which is why `not_mentioned` is a MASK in v3 and not a class: there is
nothing there to agree or disagree about, and training on it teaches the model to predict that
an annotator will say nothing.

⚠ Read `position` against 87%, not 100%: `lingual|buccal` is a genuine disagreement and it
survives the split. That is the honest ceiling for the position axis.""")


def _test() -> None:
    """Gates. Every one of these has a failure mode that would silently corrupt the target."""
    ids = all_case_ids()
    assert ids, "no raw records"

    # 1. derivation table, including the trap: a position must NOT assert a course
    assert split_axes("regular", None) == ("regular", None)
    assert split_axes("lingual", None) == (None, "lingual")
    assert split_axes("lingual", "regular course, predominantly lingual") == ("regular", "lingual")
    assert split_axes("apico_lingual", None) == (None, "lingual"), "apico_lingual must fold"
    assert split_axes("buccal", None) == (None, "buccal")
    assert split_axes("not_assessable", None) == ("not_assessable", "not_assessable")
    assert split_axes("not_mentioned", None) == (None, None)
    assert split_axes(None, None) == (None, None)

    # 2. silence never wins a merge (the v2 bug this round fixes)
    loud = {"ian_canal": {"left": {"course": "lingual", "note": ""}}}
    mute = {"ian_canal": {"left": {"course": "not_mentioned", "note": ""}}}
    got = merge_nontooth_v3([loud, mute])
    assert got["ian_position"]["left"] == "lingual", got
    assert merge_nontooth_v3([mute, mute])["ian_position"]["left"] is None

    # 3. round-trip over the whole corpus, both truncation states, uncollapsed
    n_bad = 0
    for cid in ids:
        for trunc in (False, True):
            txt = chart_text(cid, superior_truncated=trunc, collapse_healthy=False)
            got = parse_chart(txt, collapsed=False)
            if got is None:
                n_bad += 1
                print(f"   !! unparseable: {cid} trunc={trunc}")
                continue
            again = chart_block(got["slots"], fov=got["fov"], ian_course=got["ian_course"],
                                ian_position=got["ian_position"], sinuses=got["sinuses"],
                                bone=got["bone"], collapse_healthy=False)
            assert again == txt, f"round-trip not exact: {cid}"
    assert n_bad == 0, f"{n_bad} unparseable charts"
    print(f"round-trip exact over {len(ids)} cases x 2 truncations")

    # 4. THE TOOTH HALF MUST BE IDENTICAL TO v2's — this round changes the non-tooth vocabulary
    #    only, so a tooth difference here would make any v2-vs-v3 comparison unreadable.
    from data_canonical.chart_v2 import chart_text as v2_text
    from toothfairy.chart_fmt_v2 import parse_chart as v2_parse
    for cid in ids:
        a = parse_chart(chart_text(cid, collapse_healthy=False), collapsed=False)["slots"]
        b = v2_parse(v2_text(cid, collapse_healthy=False), collapsed=False)["slots"]
        assert a == b, f"tooth half drifted from v2 on {cid}"
    print(f"tooth half identical to chart_v2 on all {len(ids)} cases")

    # 5. no v3 value is a substring of another on the same line (the loss scans by char span)
    for vocab in (FOV_STATUS, CANAL_COURSE, CANAL_POSITION, SINUS_STATUS):
        for x, y in itertools.permutations(vocab, 2):
            assert x not in y, f"{x!r} is a substring of {y!r} — span scanning would mislabel"
    print("vocabulary substring-safe")

    # 6. corpus-level: how much supervision survives masking silence?
    tot = collections.Counter()
    for cid in ids:
        nt = merge_nontooth_v3(load_case_records(cid))
        for side in ("left", "right"):
            tot["course_claimed"] += nt["ian_course"][side] is not None
            tot["position_claimed"] += nt["ian_position"][side] is not None
            tot["sides"] += 1
        tot["sinus_claimed"] += nt["sinuses"] is not None
        tot["cases"] += 1
    print(f"claimed after merge: course {tot['course_claimed']}/{tot['sides']} sides "
          f"({tot['course_claimed'] / tot['sides']:.0%}), "
          f"position {tot['position_claimed']}/{tot['sides']} "
          f"({tot['position_claimed'] / tot['sides']:.0%}), "
          f"sinuses {tot['sinus_claimed']}/{tot['cases']} "
          f"({tot['sinus_claimed'] / tot['cases']:.0%})")
    print("\nchart_v3 gates: ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harmonise", action="store_true",
                    help="the agreement measurement that licenses the round")
    ap.add_argument("--test", action="store_true", help="gates (round-trip, tooth parity)")
    ap.add_argument("--show", metavar="CASE", help="print one case's v3 chart")
    a = ap.parse_args()
    if a.harmonise:
        _harmonise()
    if a.test:
        _test()
    if a.show:
        print(chart_text(a.show, collapse_healthy=False))
    if not (a.harmonise or a.test or a.show):
        ap.print_help()


if __name__ == "__main__":
    main()
