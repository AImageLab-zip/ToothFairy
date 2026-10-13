"""Per-tooth 10-axis categorical claim schema (canonical definition).

The findings head output width, the per-axis logit boundaries and the soft-injection width are
baked into the model code, so the axis definitions live here as code constants; agreement with
the canonical schema JSON. The findings
classification pipeline and the report generation pipeline share the **same output space**.

This module is the lowest-level canonical definition of the claim space. The FDI order constants
(FDI_UP/LO/ALL) live here as well, so that the data and encoder modules import them from here.

Axis order (fixed):
  [state 8 | orientation 3 | endo 2 | periapical 5 | restoration 4 | fracture 2 | atrophy 2
   | caries 1 | post_core 1 | periodontal 1]
  -> N_AXES = 10, 7 softmax axes (state, orientation, endo, periapical, restoration, fracture,
     atrophy) + 3 sigmoid axes (caries, post_core, periodontal)
  -> findings head logits (AUX_DIM) = 29 (softmax contributes n_cat, sigmoid contributes 1)
  -> soft injection (SOFT_DIM) = 22 (softmax contributes the abnormal probabilities, i.e. every
     category except idx0; sigmoid contributes the positive probability)

GT representation: 10 integer labels per tooth (softmax = class index 0..n_cat-1, sigmoid = 0/1
with positive = categories[1]). idx0 = negative (normal/none), which is the densify default.

**The training label space is deliberately coarser than the annotation vocabulary.** The GT files
and the extraction spec keep the fine-grained original labels; they are folded at read time by
`Axis.merge`. So in this file `categories` holds the **training labels** and `raw_categories`
holds the **original labels as written in the GT files** (`raw` is omitted when the two agree).
Five things were folded:
  - endo   5->2 : complete/overfill/underfill/incomplete -> `present` (counts 879/40/95/184)
  - atrophy 4->2: mild/moderate/severe -> `present` (344/703/1329)
  - restoration 5->4: `denture_abutment` (20 teeth, 8 patients) -> `prosthetic_crown`. For 6 of
    those 8 patients the report text also records a crown on the same tooth (e.g. `Prosthetic
    crown with clasp on tooth 43`), so folding this way preserves the fact that a prosthesis
    is present.
  - fracture 5->2: crown/root/endo_material/post_material -> `present` (17/12/6/**0**)
  - the maxilla `fracture` axis is **dropped**: all 1000 reports are `none`, i.e. zero signal.
The rationale is that the fine-grained distinctions are rare (mostly <0.6%) and the annotations
themselves are not consistent about them. To undo it, delete `merge`/`raw` and restore the full
`categories` — the GT files are lossless.
⚠️ This narrows the findings head from 38 to 29 logits, so checkpoints trained before the merge
have an incompatible shape (revert this file to read them).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


# FDI query order (must match encoder tooth-query cross-attention output: up16 then lo16).
# This order is the canonical reference for findings head outputs, GT labels and the token stack.
FDI_UP = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LO = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
FDI_ALL = FDI_UP + FDI_LO                                     # 32


@dataclass(frozen=True)
class Axis:
    name: str
    head: str            # "softmax" | "sigmoid"
    categories: tuple    # **training labels**. index order = label; categories[0] = negative
    raw: tuple = ()      # **original labels** of the GT files / extraction spec (before merging).
    merge: tuple = ()    # ((original label, training label), ...) - folded when the GT is read.

    @property
    def raw_categories(self) -> tuple:
        """Every label that actually occurs in the ground-truth files."""
        return self.raw or self.categories

    def resolve(self, v):
        """Original GT label -> training label. Labels that are not merged pass through."""
        for src, dst in self.merge:
            if v == src:
                return dst
        return v

    @property
    def n_cat(self) -> int:
        return len(self.categories)

    @property
    def logit_dim(self) -> int:
        """Logit width this axis occupies in the findings head output (softmax=n_cat,
        sigmoid=1)."""
        return self.n_cat if self.head == "softmax" else 1

    @property
    def soft_dim(self) -> int:
        """Width this axis occupies in the soft injection, i.e. abnormal probabilities
        (softmax=n_cat-1, sigmoid=1)."""
        return self.n_cat - 1 if self.head == "softmax" else 1


# Canonical source: per_tooth.axes of the claim schema JSON (order as fixed below).
AXES: tuple[Axis, ...] = (
    Axis("state", "softmax",
         ("normal", "partial_impacted", "full_impacted", "supraeruption",
          "root_rest", "implant", "missing", "not_available")),
    # not_available: outside the field of view, i.e. not assessable. mesioverted/distoverted
    # were split out into their own `orientation` axis.
    Axis("orientation", "softmax",
         ("normoverted", "mesioverted", "distoverted")),  # tooth angulation, split out of state
    # endo/fracture/atrophy only learn presence; restoration folds denture_abutment into the
    # crown category (see the module docstring). raw = GT vocabulary, categories = training labels.
    Axis("endo", "softmax", ("none", "present"),
         raw=("none", "complete", "overfill", "underfill", "incomplete"),
         merge=(("complete", "present"), ("overfill", "present"),
                ("underfill", "present"), ("incomplete", "present"))),
    Axis("periapical", "softmax",
         ("normal", "apical_osteorarefaction", "nonapical_osteorarefaction",
          "osteosclerosis", "cementoma")),
    Axis("restoration", "softmax",
         ("none", "filling", "prosthetic_crown", "pontic"),
         raw=("none", "filling", "prosthetic_crown", "pontic", "denture_abutment"),
         merge=(("denture_abutment", "prosthetic_crown"),)),
    Axis("fracture", "softmax", ("none", "present"),
         raw=("none", "crown_fracture", "root_fracture",
              "fractured_endo_material", "fractured_post_material"),
         merge=(("crown_fracture", "present"), ("root_fracture", "present"),
                ("fractured_endo_material", "present"),
                ("fractured_post_material", "present"))),
    Axis("atrophy", "softmax", ("none", "present"),
         raw=("none", "mild", "moderate", "severe"),
         merge=(("mild", "present"), ("moderate", "present"), ("severe", "present"))),
    Axis("caries", "sigmoid", ("none", "caries")),
    Axis("post_core", "sigmoid", ("none", "post_and_core")),
    Axis("periodontal", "sigmoid", ("normal", "marginal_bone_loss")),
)

N_AXES = len(AXES)                                          # 10
AUX_DIM = sum(a.logit_dim for a in AXES)                    # 29 (state 8 + orientation 3 + ...)
SOFT_DIM = sum(a.soft_dim for a in AXES)                    # 22
SOFTMAX_AXES = tuple(i for i, a in enumerate(AXES) if a.head == "softmax")  # (0..6)
SIGMOID_AXES = tuple(i for i, a in enumerate(AXES) if a.head == "sigmoid")  # (7,8,9)
AXIS_NAMES = tuple(a.name for a in AXES)


def logit_slices() -> list[slice]:
    """10 slices that cut the findings head output (.., AUX_DIM) into per-axis logit ranges."""
    out, off = [], 0
    for a in AXES:
        out.append(slice(off, off + a.logit_dim))
        off += a.logit_dim
    return out




def build_axis_labels(dense: dict) -> np.ndarray:
    """Densified per-report claim dict -> (32, 10) int64 labels in FDI_ALL order.

    Densification fills all 32 teeth and all axes with the negative default, so a missing entry
    is a schema violation and raises. FDI codes outside the permanent dentition (deciduous or
    supernumerary teeth) are ignored; only the 32 slots of FDI_ALL are used.
    """
    axes = AXES
    teeth = dense.get("teeth", {})
    y = np.zeros((len(FDI_ALL), len(axes)), dtype=np.int64)
    for i, fdi in enumerate(FDI_ALL):
        td = teeth.get(str(fdi))
        if td is None:
            raise KeyError(f"dense GT missing FDI {fdi} (densification must fill all 32 teeth)")
        for j, ax in enumerate(axes):
            raw = td.get(ax.name)
            v = ax.resolve(raw)                               # original -> training label
            if v not in ax.categories:
                raise ValueError(
                    f"FDI {fdi} axis {ax.name!r}: {raw!r} not in {ax.raw_categories}")
            y[i, j] = ax.categories.index(v)
    return y


def compute_class_weights(dense_paths, beta: float = 0.99) -> dict[str, torch.Tensor]:
    """Training-set claim frequencies -> effective-number (Cui et al. 2019) class weights.

    Settings: **effective number with beta=0.99, no cap** (beta already bounds the ratio at
    1/(1-beta)), weights expressed as a **multiple of normal/none (idx0) = 1.0**, and
    **categories with n=0 are left unweighted (=1.0)**. The raw effective-number weight
    w_c=(1-beta)/(1-beta^n_c) is divided by the one of the negative class so that normal=1.0.

    Returns dict[axis_name -> tensor]:
      - softmax axis: CE `weight` vector (n_cat,) - multiples of normal=1.0, n=0 category = 1.0.
      - sigmoid axis: BCE `pos_weight` scalar - the positive class (idx1) relative to normal.
    For softmax axes CE(reduction=mean) cancels the weight sum, so only the relative ratios
    matter; for sigmoid axes the absolute pos_weight acts directly. Both are expressed the same
    way (multiples of normal=1.0).
    """
    axes = AXES
    cnt = [defaultdict(int) for _ in axes]
    for p in dense_paths:
        y = build_axis_labels(json.load(open(p)))              # (32, 10)
        for j in range(len(axes)):
            for v in y[:, j]:
                cnt[j][int(v)] += 1

    def en(n: int):                                            # raw effective-number weight
        return (1.0 - beta) / (1.0 - beta ** n) if n > 0 else None

    weights: dict[str, torch.Tensor] = {}
    for j, ax in enumerate(axes):
        w0 = en(cnt[j].get(0, 0))                              # normal/none(idx0)
        if ax.head == "softmax":
            vec = []
            for ci in range(ax.n_cat):
                wc = en(cnt[j].get(ci, 0))
                vec.append(1.0 if (wc is None or w0 is None) else wc / w0)  # n=0→1.0, normal=1.0
            weights[ax.name] = torch.tensor(vec, dtype=torch.float32)
        else:                                                 # sigmoid: positive=idx1
            wp = en(cnt[j].get(1, 0))
            pw = 1.0 if (wp is None or w0 is None) else wp / w0
            weights[ax.name] = torch.tensor(pw, dtype=torch.float32)
    return weights


# --------------------------------------------------------------------------- #
# non-teeth region claim axes (maxilla / mandible / mandibular canal vectors of the schema)
# --------------------------------------------------------------------------- #
# Categorical claim supervision covers the non-teeth regions as well, so the findings head is the
# complete per-element claim classifier. Slot mapping: maxilla=slot0, mandible=slot1,
# nerve_R=slot2, nerve_L=slot3. **Soft injection is kept for teeth only** (non-teeth regions
# contribute the findings loss but no injected probabilities). Each region has its own axes, hence
# its own head (unlike teeth, which share one head).
# The separate arch axes were removed and perio_stage moved into the maxilla/mandible axes: the
# two arch summary tokens are excluded from generation and from the findings head (their text GT
# would only have been regex-derived), which left the arch-level perio_stage claim without a home.
# The maxilla/mandible tokens attend over all tooth patches, so they have the evidence to predict
# the arch-level periodontal state.
# The arch queries themselves **remain** in the tooth-query cross-attention, so that the other
# anatomical tokens can still see the arch context through the cross-label self-attention; they
# are only excluded from generation and from claim supervision.
_PERIO_STAGE = Axis("perio_stage", "softmax", ("normal", "stage_I", "stage_II", "stage_III"))

MAXILLA_AXES: tuple[Axis, ...] = (
    _PERIO_STAGE,                                    # formerly the upper arch axis
    Axis("sinus_inclusion", "softmax",
         ("not_included", "pneumatization_increased", "pneumatization_normal")),
    Axis("sinus_mucosa", "softmax",
         ("normal", "right_thickening", "left_thickening", "both_thickening",
          "right_retention_cyst", "left_retention_cyst", "both_retention_cyst")),
    # The fracture axis was dropped: all 1000 reports are `none` (6 categories, 0 positives), so
    # its 6 logits took up head capacity with no learning signal. The maxilla.fracture key stays
    # in the GT files (build_nonteeth_labels only walks the axis list, so it is ignored) and is
    # only removed from what is trained.
    Axis("cyst", "softmax",
         ("none", "right_wisdom", "left_wisdom", "both_wisdom",
          "right_posterior", "left_posterior", "both_posterior", "anterior")),
)
MANDIBLE_AXES: tuple[Axis, ...] = (
    _PERIO_STAGE,                                    # formerly the lower arch axis
    Axis("condyle_included", "sigmoid", (False, True)),
    Axis("coronoid_included", "sigmoid", (False, True)),
    Axis("fracture", "softmax",
         ("none", "right_angle", "right_body", "right_parasymphysis",
          "left_parasymphysis", "left_body", "left_angle",
          "right_ramus", "left_ramus", "right_condyle", "left_condyle")),  # incl. ramus/condyle
    Axis("cyst", "softmax",
         ("none", "right_wisdom", "left_wisdom", "both_wisdom", "right_premolar",
          "left_premolar", "right_posterior", "left_posterior", "both_posterior", "anterior")),
)
NERVE_AXES: tuple[Axis, ...] = (
    Axis("position", "softmax", ("lingual", "buccal", "central")),
    Axis("superficialized", "sigmoid", (False, True)),
    Axis("canal_near_7", "sigmoid", (False, True)),
    Axis("canal_near_8", "sigmoid", (False, True)),
    Axis("not_evaluable", "sigmoid", (False, True)),
)

# region_name -> axes (the right and left canal slots share one set of axes). The separate arch
# entry was removed; perio_stage now lives on maxilla/mandible.
NONTEETH_AXES: dict[str, tuple[Axis, ...]] = {
    "maxilla": MAXILLA_AXES, "mandible": MANDIBLE_AXES, "nerve": NERVE_AXES,
}


def region_aux_dim(axes: tuple[Axis, ...]) -> int:
    return sum(a.logit_dim for a in axes)


def region_logit_slices(axes: tuple[Axis, ...]) -> list[slice]:
    out, off = [], 0
    for a in axes:
        out.append(slice(off, off + a.logit_dim))
        off += a.logit_dim
    return out


def _axis_index(ax: Axis, v) -> int:
    resolved = ax.resolve(v)                                  # original -> training label
    if resolved not in ax.categories:
        raise ValueError(f"axis {ax.name!r}: {v!r} not in {ax.raw_categories}")
    return ax.categories.index(resolved)


def build_nonteeth_labels(dense: dict) -> dict[str, np.ndarray]:
    """Densified per-report claim dict -> non-teeth int64 labels.
      maxilla:  (|MAXILLA_AXES|=4,)                   (dense['maxilla']={axis: cat})
      mandible: (|MANDIBLE_AXES|=5,)
      nerve:    (2, |NERVE_AXES|=5)  [right, left]  (dense['nerve']={right:{axis:cat}, left:{...}})
    Densification fills the non-teeth regions too, so a missing entry is a schema violation.

    perio_stage is part of the maxilla/mandible axes; a fallback keeps older GT files that still
    carry a separate dense['arch']={upper,lower} entry readable."""
    out: dict[str, np.ndarray] = {}
    legacy_arch = dense.get("arch") or {}
    for key, axes, legacy_side in (("maxilla", MAXILLA_AXES, "upper"),
                                   ("mandible", MANDIBLE_AXES, "lower")):
        d = dict(dense[key])
        if "perio_stage" not in d and legacy_side in legacy_arch:
            d["perio_stage"] = legacy_arch[legacy_side]      # older GT layout
        out[key] = np.array([_axis_index(a, d[a.name]) for a in axes], dtype=np.int64)
    # nerve: one axis dict per side (right/left)
    nv = dense["nerve"]
    yn = np.zeros((2, len(NERVE_AXES)), dtype=np.int64)
    for si, side in enumerate(("right", "left")):
        for j, a in enumerate(NERVE_AXES):
            yn[si, j] = _axis_index(a, nv[side][a.name])
    out["nerve"] = yn
    return out






# --------------------------------------------------------------------------- #
# **Textual rendering** of the axis probabilities.
#
# Why text: the information in the findings head already reaches the LLM input as a continuous
# vector (a linear probe reads it out of the visual prefix as well as out of the encoder context,
# mean AUC 0.907 -> 0.909 over 21 classes), yet greedy decoding still does not pick the finding
# up. The bottleneck is therefore the decision rule rather than the representation: if the model
# simply has no notion of the threshold, the probabilities have to be given **as text** for it to
# learn that rule.
#
# Rendering rules: **the same for every slot, all axes, probability values**.
#   - softmax axes list the per-category probability in schema order. Fixing the order lets the
#     model read a value by position (sorting by probability would move the positions around from
#     slot to slot and make it harder to learn).
#   - sigmoid axes render the single positive probability.
#   - Two decimals. A third decimal only adds tokens and does not help learn the threshold.
# --------------------------------------------------------------------------- #
def render_probs(axes: tuple, logits) -> str:
    """Per-axis logits of one region (AUX_DIM,) -> human-readable probabilities. Takes a
    torch tensor."""
    import torch

    lines = []
    off = 0
    for a in axes:
        n = a.logit_dim
        chunk = logits[off:off + n].float()
        off += n
        if a.head == "softmax":
            p = torch.softmax(chunk, dim=-1)
            body = " ".join(f"{_cat_name(c)} {v:.2f}"
                            for c, v in zip(a.categories, p.tolist()))
        else:                                     # sigmoid = a single positive probability
            body = f"{torch.sigmoid(chunk)[0].item():.2f}"
        lines.append(f"{a.name}: {body}")
    return "\n".join(lines)


def _cat_name(c) -> str:
    """Category label as a display string; boolean axes (condyle_included etc.) render
    True/False."""
    return c if isinstance(c, str) else str(c)
