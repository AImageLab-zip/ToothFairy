"""The findings record: the single structured representation the whole pipeline
routes through.

Upstream (masks, geometry) fills slots; the renderer reads slots. Nothing may be
written into a report that is not a slot here -- that is what keeps RadFact
precision a property of the architecture rather than something we hope for.

Slots are deliberately limited to findings that are BOTH extractable from the
reference reports (for calibration) AND derivable from a TF3-label segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
ALL_FDI = UPPER + LOWER
THIRD_MOLARS = [18, 28, 38, 48]

# TF3 label ids
LAB_LOWER_JAW, LAB_UPPER_JAW = 1, 2
LAB_CANAL_L, LAB_CANAL_R = 3, 4
LAB_SINUS_L, LAB_SINUS_R = 5, 6
LAB_PHARYNX = 7
LAB_BRIDGE, LAB_CROWN, LAB_IMPLANT = 8, 9, 10

# TF3 also labels the pulp chamber of every tooth, as 100 + the tooth's own id
# (111-148). Nothing used them until the endodontic derivation: root-canal filling
# material is the densest thing that can occupy that space, so a bright pulp is a
# direct readout of endodontic treatment. The segmentation predicts them at
# per-tooth Dice 0.77-0.84, which is ample for an intensity statistic.
def pulp_label(fdi: int) -> int:
    return 100 + fdi

UNKNOWN = None  # a slot we could not determine -- never rendered


def to_clinical_fdi(fdi: int) -> int:
    """ToothFairy3 label id -> the FDI number a clinician actually writes.

    The TF3 labels are left/right mirrored with respect to the reports: what the
    label set calls "Lower Right First Premolar" (44) is the tooth the surgeons
    call 34. Measured over 1182 tooth statements in 475 cases, agreement with the
    reports is 78.0% as-is versus 95.2% mirrored; restricted to absence
    statements (the asymmetric ones, where a mirror actually shows up) it is
    65.1% versus 93.8%.

    Nothing internal to the pipeline exposes this, because the model, the TF3
    labels and the canal labels are all self-consistent -- the reports are the
    only external evidence of laterality.
    """
    quadrant, tooth = divmod(fdi, 10)
    return {1: 2, 2: 1, 3: 4, 4: 3}[quadrant] * 10 + tooth


def mirror_side(side: str) -> str:
    """Same correction for laterality words: TF3's canal 3 ("Left") is the
    clinical right canal."""
    return {"left": "right", "right": "left"}[side]


@dataclass
class Findings:
    """One case. `None` everywhere means 'unknown', which the renderer skips."""

    case_id: str = ""

    # --- coverage / field of view (~21% of report content, near-deterministic) ---
    mandible_included: bool | None = None
    maxilla_coverage: str | None = None       # "none" | "partial" | "full"
    condyles_included: bool | None = None
    sinuses_included: bool | None = None

    # --- dentition ---
    teeth_present: dict[int, bool] = field(default_factory=dict)   # FDI -> present
    impacted: set[int] = field(default_factory=set)
    edentulism: str | None = None             # "none" | "partial" | "complete"

    # --- inferior alveolar canal ---
    canal_course: str | None = None           # "regular" | "lingual" | "buccal"
    canal_contact: dict[str, int] = field(default_factory=dict)  # "left"/"right" -> FDI

    # --- restorations ---
    implants: set[int] = field(default_factory=set)   # FDI sites
    crowns: set[int] = field(default_factory=set)
    endodontic: set[int] = field(default_factory=set)  # FDI sites, from pulp density
    restorations: set[int] = field(default_factory=set)  # FDI sites, from crown density
    has_bridge: bool | None = None
    # Site-less variants: an implant occupies a MISSING tooth site, so there is no
    # tooth label to attach it to and naming a site would be a precision risk.
    has_implant: bool | None = None
    has_crown: bool | None = None

    # --- bone / pathology (report-only for now; TF3 has no labels for these) ---
    atrophy: str | None = None                # "none" | "mild" | "moderate" | "severe"
    osteolytic_lesion: bool | None = None

    def slots(self) -> dict[str, object]:
        """Flattened view used by the fast proxy metric: one key per assertable fact."""
        out: dict[str, object] = {}
        for name in (
            "mandible_included", "maxilla_coverage", "condyles_included",
            "sinuses_included", "edentulism", "canal_course", "atrophy",
            "osteolytic_lesion", "has_bridge", "has_implant", "has_crown",
        ):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
        for fdi, present in self.teeth_present.items():
            out[f"tooth_{fdi}"] = present
        for fdi in self.impacted:
            out[f"impacted_{fdi}"] = True
        for side, fdi in self.canal_contact.items():
            out[f"canal_contact_{side}"] = fdi
        for fdi in self.implants:
            out[f"implant_{fdi}"] = True
        for fdi in self.crowns:
            out[f"crown_{fdi}"] = True
        for fdi in self.endodontic:
            out[f"endodontic_{fdi}"] = True
        for fdi in self.restorations:
            out[f"restoration_{fdi}"] = True
        return out


def proxy_prf(candidate: Findings, reference: Findings) -> tuple[float, float, float]:
    """Fast stand-in for RadFact: agreement over schema slots.

    RadFact splits reports into per-tooth/per-site phrases and scores
    precision = entailed_candidate / candidate, recall = entailed_reference / reference.
    Because our reports are RENDERED FROM this record, our phrase set is known
    exactly, so entailment reduces to slot agreement -- no LLM in the loop.
    """
    cand, ref = candidate.slots(), reference.slots()
    if not cand:
        return 0.0, 0.0, 0.0
    if not ref:
        return 0.0, 0.0, 0.0
    entailed_c = sum(1 for k, v in cand.items() if k in ref and ref[k] == v)
    entailed_r = sum(1 for k, v in ref.items() if k in cand and cand[k] == v)
    p = entailed_c / len(cand)
    r = entailed_r / len(ref)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1
