"""Presence-statement neutralisation — take bare "it is there" out of the RFT reward.

Why
---
Measured on the ground truth, **the same normal tooth is written up differently from report to
report**: only 45.5% of the slots of a state=normal tooth state its presence (`Tooth 31 present`)
and 54.5% are empty, depending on whether the radiologist enumerated the dentition in that
report. The reward, however, uses "is the ground truth empty" to split normal regions
(silence = 1.0) from finding regions (silence = 0.0), so **the same input carried opposite
rules**. On top of that `Tooth 31 present` holds no claim at all, so claim F1 degenerates to
empty vs empty = 1.0 there (27.1% of the tooth finding slots on the validation union).

Decision: **stating that a tooth is present is optional** → neutralise it when scoring.
**Arch-level presence statements are not neutralised.**

What is removed and what is kept
--------------------------------
Line by line, a line that is **only a tooth-presence statement** is removed. Lines that carry
more content are kept — the ground truth really contains all of the following.

  remove "Tooth 31 present" · "- 46 present" · "tooth 12 is present"
  keep   "Tooth 18 present and extruded"        (extruded = supraeruption)
         "tooth 45 present only as root remnant" (root_rest)
         "tooth 36 present with periodontitis"   (periodontal)
         "tooth 23 present in the 22 region"     (displacement)
         "complete mandibular dentition"         (arch level — excluded by decision)
         "complete edentulism of the mandible"   (edentulous = missing, not a presence statement)
         "No osteolytic or osteocondensing lesions present"  (negated finding: 'present' differs)
"""
from __future__ import annotations

import re

# --- tooth presence statement: the whole line is "(tooth) NN (is) present" ------ #
# Only **valid FDI numbers** are accepted. With `\d{1,2}` a hallucinated "Tooth 99 present" would
# have its whole line removed and slip past the invalid-FDI gates
# (`reward._off_region_rate` · `terms.fdi_report`).
_FDI = r"(?:1[1-8]|2[1-8]|3[1-8]|4[1-8])"
_TOOTH_PRESENT = re.compile(
    rf"^(?:tooth|teeth|element)?\s*{_FDI}\s*(?:is\s+|are\s+)?present(?:\s+in\s+the\s+arch)?$",
    re.I)

# ⚠️ **Arch-level presence statements are not neutralised.**
#   Lines such as `complete mandibular dentition` or `All other maxillary elements present in the
#   arch` are left alone. They hold no claim, so `reward_drop_claimless` already excludes them
#   from scoring (they do not turn into normal regions). Only tooth presence lines are targeted.

_BULLET = " \t-•*·"


def is_presence_line(line: str) -> bool:
    """Is this line **only** a tooth-presence statement? False (= keep it) if a finding is on it."""
    s = line.strip(_BULLET).strip().rstrip(".;,").strip()
    if not s:
        return False
    return bool(_TOOTH_PRESENT.match(s))


def strip_presence(text: str | None) -> str:
    """Text with presence-only lines removed (whitespace tidied); everything else is verbatim."""
    if not text:
        return ""
    keep = [ln for ln in text.split("\n") if not is_presence_line(ln)]
    return "\n".join(keep).strip()
