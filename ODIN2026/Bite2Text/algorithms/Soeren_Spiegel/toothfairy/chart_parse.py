"""Frozen chart format + container-safe splitter/parser.

The chart-QA target is ``chart block + delimiter + report``. This module owns

  1. the **frozen format** — class vocabulary, line templates, delimiter — so the renderer
     (``data_canonical/chart_render.py``, train side) and the parser (this file, container +
     eval side) cannot drift apart; and
  2. the **inverse**: ``split_output`` peels the report off a decoded generation and
     ``parse_chart`` turns the chart block back into ``{fdi: class}``.

**Stdlib only, and zero ``data_canonical`` imports.** ``data_canonical/`` is not vendored into
the submission image and a module-scope import of it has already crashed the container once;
the dependency therefore points the other way — ``chart_render`` imports the format from here.

The container calls exactly one function, ``split_output``, and ships ``report_text``. The
chart must never reach the judge or the leaderboard, so a formatting regression has to be
*loud*: ``split_output`` returns an explicit ``split_failed`` flag rather than guessing
silently, and every caller counts it.

The format::

    Chart:
    Q1: 11 present, 12 crown, 13 present
    Q2: 21 present
    Mandible: not charted
    Third molars: not charted

    Report:
    <canonical prose>

Rules, all of them load-bearing:

  * one class token per slot, ascending FDI within a quadrant, ``", "`` between entries,
    quadrant order Q1 -> Q4. Nothing else is free — every degree of freedom here is
    cross-entropy spent on wording rather than on teeth.
  * a quadrant line is **omitted when empty**; when BOTH quadrants of an arch are empty the
    arch collapses to one fixed ``<Arch>: not charted`` line. "Not charted", not "not in the
    scan": the slots are dropped by the ``tooth_state`` gates, and 81/622 cases reach an empty
    maxilla with ``fov.maxilla.status`` of ``partial`` or ``included`` — asserting an FOV fact
    there would be false.
  * ``Third molars: not charted`` is a constant line (18/28/38/48 never carry the weak label).
    An explicitly charted third molar still appears in its quadrant line.
  * mixed dentition collapses the whole block to ``Chart: not charted (mixed dentition)``.
"""
from __future__ import annotations

import re

# ── Class vocabulary ────────────────────────────────────────────────────────────
# One token per slot, flattened from the two label groups of ``data_canonical/tooth_state.py``
# (4-way status + 6 independent attributes) by the fixed priority in ``CHART_PRIORITY`` there.
# STATUS FIRST: the status axis is the one the auxiliary head was scored on, the one the judge
# trace flags (absence over-claiming), and the one the prose target itself keeps — an absent
# tooth's restorations are dropped by ``render_prose._tooth_clause``, so charting an
# implant-borne absent site as ``absent`` matches the report it is concatenated with.
CHART_CLASSES: tuple[str, ...] = (
    "present", "absent", "impacted", "root remnant",     # status (root remnant is 2 words)
    "implant", "crown", "endo", "lesion", "filling",     # attributes, rarest-first
)
CHART_INDEX: dict[str, int] = {c: i for i, c in enumerate(CHART_CLASSES)}

# Back-projection to the 4 status classes of ``tooth_state.STATUS_CLASSES``. Lossless by
# construction (status wins every tie), which is what makes the chart's macro-F1 directly
# comparable to the auxiliary head's 0.514.
CHART_TO_STATUS: dict[str, str] = {
    "present": "present", "absent": "absent", "impacted": "impacted",
    "root remnant": "root_remnant",
    "implant": "present", "crown": "present", "endo": "present",
    "lesion": "present", "filling": "present",
}

# ── Frozen format ───────────────────────────────────────────────────────────────
CHART_HEADER = "Chart:"
REPORT_HEADER = "Report:"
DELIMITER = f"\n\n{REPORT_HEADER}\n"
MIXED_DENTITION_LINE = f"{CHART_HEADER} not charted (mixed dentition)"
THIRD_MOLAR_LINE = "Third molars: not charted"
ARCH_EMPTY_LINE = "{arch}: not charted"
QUADRANT_LABEL = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
ARCH_LABEL = {"12": "Maxilla", "34": "Mandible"}

_ENTRY_SEP = ", "
_CLASS_ALT = "|".join(re.escape(c) for c in sorted(CHART_CLASSES, key=len, reverse=True))
_ENTRY_RE = re.compile(rf"^([1-4][1-8]) ({_CLASS_ALT})$")
_QUAD_LINE_RE = re.compile(r"^Q([1-4]): (.+)$")
_ARCH_LINE_RE = re.compile(r"^(Maxilla|Mandible): not charted$")


def chart_block(quadrants: dict[str, list[tuple[str, str]]],
                mixed_dentition: bool = False) -> str:
    """Frozen rendering of ``{quadrant_digit: [(fdi, class), ...]}`` -> the chart block.

    The single place the format is *written*. ``chart_render.chart_text`` supplies the entries;
    keeping the string assembly here means the parser below is the exact inverse of one
    function, not of a function in a module the container cannot import."""
    if mixed_dentition:
        return MIXED_DENTITION_LINE
    lines = [CHART_HEADER]
    for arch_digits, arch_name in ARCH_LABEL.items():
        arch_lines = []
        for q in arch_digits:
            entries = quadrants.get(q) or []
            if entries:
                body = _ENTRY_SEP.join(f"{fdi} {cls}" for fdi, cls in entries)
                arch_lines.append(f"{QUADRANT_LABEL[q]}: {body}")
        lines += arch_lines or [ARCH_EMPTY_LINE.format(arch=arch_name)]
    lines.append(THIRD_MOLAR_LINE)
    return "\n".join(lines)


# ── Splitting a decoded generation ──────────────────────────────────────────────

def _is_chart_line(line: str) -> bool:
    """True for any line the frozen format can emit inside the chart block."""
    return (line == CHART_HEADER or line == MIXED_DENTITION_LINE
            or line == THIRD_MOLAR_LINE
            or bool(_QUAD_LINE_RE.match(line)) or bool(_ARCH_LINE_RE.match(line)))


def split_output(text: str) -> tuple[str | None, str, bool]:
    """``(chart_text | None, report_text, split_failed)``.

    The container ships ``report_text`` and nothing else. ``split_failed`` is True whenever the
    frozen delimiter was not found — the model wrote something off-format and the fallback had
    to guess, which is a fact the caller must count and log rather than absorb.

    Fallback ladder:

      1. the frozen delimiter. A duplicated delimiter is not a failure — the chart is the first
         block, so the FIRST delimiter is the true one and the rest belongs to the report.
      2. a line that is just ``Report:``. The LAST one wins, so a report that happens to use
         the word cannot steal the split back.
      3. **a leading run of chart-shaped lines, peeled off even with no delimiter at all.**
         This rung is not decoration: without it a generation truncated inside the chart has no
         delimiter, falls through to "the whole text is the report", and publishes the chart to
         the leaderboard. garbage-input check caught exactly that. An empty report is a visible failure the
         caller's own fallback handles; a chart on the leaderboard is not visible at all.
      4. the whole text as the report."""
    if not text:
        return None, "", True
    idx = text.find(DELIMITER)
    if idx != -1:
        return text[:idx].strip(), text[idx + len(DELIMITER):].strip(), False
    hits = list(re.finditer(rf"^{re.escape(REPORT_HEADER)}\s*$", text, re.MULTILINE))
    if hits:
        m = hits[-1]
        head = text[:m.start()].strip()
        return (head or None), text[m.end():].strip(), True
    lines = text.split("\n")
    n = 0
    while n < len(lines) and _is_chart_line(lines[n].strip()):
        n += 1
    if n:
        return "\n".join(lines[:n]).strip(), "\n".join(lines[n:]).strip(), True
    return None, text.strip(), True


def n_delimiters(text: str) -> int:
    """How many frozen delimiters the generation contains. 1 is correct; 0 means the split
    fell back; >1 means the model restarted the format mid-report. Logged, not acted on."""
    return (text or "").count(DELIMITER)


# ── Parsing a chart block ───────────────────────────────────────────────────────

def parse_chart(chart_text: str | None) -> dict[str, str] | None:
    """``{fdi: class}`` for every charted slot, or ``None`` on a malformed block.

    Strict: an unknown class token, a non-FDI code, a repeated slot, a quadrant digit that
    disagrees with its line, or a line that is none of the four legal shapes fails the whole
    block. A guess here would be scored as if the model had produced it.

    The mixed-dentition sentinel parses to ``{}`` — validly charted nothing, which is a
    different thing from a parse failure and must stay distinguishable."""
    if chart_text is None:
        return None
    lines = [l.strip() for l in chart_text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    if lines[0] == MIXED_DENTITION_LINE:
        return {} if len(lines) == 1 else None
    if lines[0] != CHART_HEADER:
        return None

    out: dict[str, str] = {}
    seen_third_molar_line = False
    for line in lines[1:]:
        if line == THIRD_MOLAR_LINE:
            if seen_third_molar_line:
                return None
            seen_third_molar_line = True
            continue
        if _ARCH_LINE_RE.match(line):
            continue
        m = _QUAD_LINE_RE.match(line)
        if not m:
            return None
        quad, body = m.group(1), m.group(2)
        for chunk in body.split(_ENTRY_SEP):
            e = _ENTRY_RE.match(chunk.strip())
            if not e:
                return None
            fdi, cls = e.group(1), e.group(2)
            if fdi[0] != quad or fdi in out:
                return None
            out[fdi] = cls
    return out


def parse_output(text: str) -> tuple[dict[str, str] | None, str, dict]:
    """One call for the inference path: ``(chart | None, report_text, flags)``.

    ``flags`` = ``{"split_failed": bool, "parse_failed": bool, "n_delimiters": int}``.
    ``parse_failed`` is only meaningful when a chart block was found at all — a generation
    with no chart is a split failure, not a parse failure, and conflating them hides which
    half of the format the model dropped."""
    chart_text, report, split_failed = split_output(text)
    chart = parse_chart(chart_text)
    return chart, report, {
        "split_failed": split_failed,
        "parse_failed": chart_text is not None and chart is None,
        "n_delimiters": n_delimiters(text),
    }
