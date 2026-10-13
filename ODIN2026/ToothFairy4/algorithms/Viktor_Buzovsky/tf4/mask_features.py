"""Segmentation mask -> Findings.

Input masks must be in the canonical RAS frame produced by
`trinity_infer.segment_case`:
    axis 0 -> patient RIGHT, axis 1 -> ANTERIOR, axis 2 -> SUPERIOR.

Every threshold here is a calibration target -- see `calibrate.py`, which scores
these derivations against the report-parsed findings. A derivation that does not
beat the corpus prior gets dropped from the renderer rather than shipped.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    LAB_BRIDGE,
    LAB_CANAL_L,
    LAB_CANAL_R,
    LAB_CROWN,
    LAB_IMPLANT,
    LAB_LOWER_JAW,
    LAB_PHARYNX,
    LAB_SINUS_L,
    LAB_SINUS_R,
    LAB_UPPER_JAW,
    LOWER,
    UPPER,
    Findings,
    mirror_side,
    pulp_label,
    to_clinical_fdi,
)

# All thresholds below come from sweep_thresholds.py against 475 cases.
MIN_TOOTH_VOXELS = 5000     # best balanced point: presence 95.6%, absence 65.1%
MIN_JAW_VOXELS = 5000
MIN_UPPER_JAW_VOXELS = 20000  # maxilla none-vs-included: 76.2% (prior: 54.6%)
MIN_RESTORATION_VOXELS = 150
CONTACT_MM = 2.0
IMPACTION_MM = 3.0
# Signed canal position below which the course is called lingual. The sweep put
# the best cutoff (0.0) at 64.0% against a 61.4% always-majority prior -- a 2.6pt
# gain, i.e. barely better than guessing, and it emits "lingual" 87% of the time
# regardless. Kept because it costs nothing, but it is NOT a real signal.
CANAL_LINGUAL_CUTOFF = 0.0

# Endodontic filling material is denser than anything else that can sit inside a
# pulp chamber. The threshold is expressed as a RATIO to the case's own jaw-bone
# p95 rather than as an absolute value, because the volumes are not on a shared
# intensity scale: several F-set volumes are clipped at a maximum of 2400, below
# the absolute cutoff that works elsewhere, so an absolute rule can never fire on
# them at all. Measured over 8723 teeth in 506 cases against the ground-truth
# reference reports:
#     absolute pp90 >= 2800      0.68 claims/case  precision 0.79  recall 0.70
#     pp90 >= 1.4 x bone p95     0.78 claims/case  precision 0.78  recall 0.78
# and the ratio form is also the more even across sets (F 0.69 / P 0.82 / S 0.64
# versus F 0.63 / P 0.84 / S 0.66).
ENDO_BONE_RATIO = 1.4
MIN_PULP_VOXELS = 40

# A restoration is dense material inside the tooth crown rather than the canal,
# so the statistic is the tooth's own p99 against the same bone reference. It is
# a far weaker signal than the pulp one -- enamel and beam-hardening live in the
# same intensity range -- and it is only worth stating on teeth we are not
# already saying something about. Measured on those teeth only:
#     p99 >= 2.4 x bone p95   0.45 claims/case  entailed 0.391
#     p99 >= 2.2 x bone p95   0.69 claims/case  entailed 0.361
#     p99 >= 2.0 x bone p95   1.74 claims/case  entailed 0.251
# The claims between 2.0 and 2.2 are entailed at a marginal 0.18, below the
# ~0.24 break-even, so 2.2 is the cut.
RESTORATION_BONE_RATIO = 2.2

# Upper-arch absence is only claimed when this many upper teeth are visible, on
# top of maxilla_coverage == "full". See the derivation for the measurements.
MIN_UPPER_TEETH_FOR_ABSENCE = 4


def _count(mask: np.ndarray, label: int) -> int:
    return int((mask == label).sum())


def _safely(fallback, fn, *args):
    """Run an optional derivation; on any failure leave that slot empty.

    An unfilled slot costs one finding. An exception costs the entire report,
    because build_report falls back to the prior-only text -- which is how a
    missing scipy in the container image turned every case into the same 64
    words. Optional findings are individually expendable; the report is not.
    """
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: {fn.__name__} failed, slot left empty: {exc!r}", flush=True)
        return fallback


def infer_axes(mask: np.ndarray) -> dict[str, tuple[int, int]] | None:
    """Recover the anatomical axes from the segmentation itself.

    TF3 and TF4 affines contradict each other (TF3 declares LPS, TF4 declares
    LAS, yet their arrays differ by a flip of axes 1 and 2), so at least one
    header is wrong and neither can be trusted for left/right or up/down. The
    labels, however, are unambiguous: the upper jaw is superior to the lower jaw,
    the pharynx is posterior to the mandible, and labels 3/4 are the LEFT and
    RIGHT canals by definition.

    Returns {"sup": (axis, sign), "ant": (axis, sign), "left": (axis, sign)},
    where sign +1 means increasing index goes that way. None if undeterminable.
    """
    def centroid(label: int):
        idx = np.argwhere(mask == label)
        return idx.mean(axis=0) if len(idx) >= 200 else None

    lower = centroid(LAB_LOWER_JAW)
    if lower is None:
        return None
    upper, pharynx = centroid(LAB_UPPER_JAW), centroid(LAB_PHARYNX)
    canal_l, canal_r = centroid(LAB_CANAL_L), centroid(LAB_CANAL_R)

    axes: dict[str, tuple[int, int]] = {}
    used: set[int] = set()

    def claim(name: str, delta) -> None:
        """Assign `name` to the strongest axis not already taken."""
        order = np.argsort(-np.abs(delta))
        for ax in order:
            ax = int(ax)
            if ax not in used:
                axes[name] = (ax, 1 if delta[ax] > 0 else -1)
                used.add(ax)
                return

    # Most reliable first: labels 3 and 4 ARE the left and right canals.
    if canal_l is not None and canal_r is not None:
        claim("left", canal_l - canal_r)
    # The pharynx sits behind the mandible.
    if pharynx is not None:
        claim("ant", lower - pharynx)
    # Superior last, from whatever axis remains.
    ref = upper if upper is not None else centroid(LAB_CROWN)
    if ref is not None:
        claim("sup", ref - lower)

    missing = [k for k in ("sup", "ant", "left") if k not in axes]
    free = [a for a in (0, 1, 2) if a not in used]
    if len(missing) == 1 and len(free) == 1:
        axes[missing[0]] = (free[0], 1)
    if len(axes) != 3 or len({v[0] for v in axes.values()}) != 3:
        return None
    return axes


def _superior_border_clipped(mask: np.ndarray, label: int, axes: dict, slices: int = 3) -> bool:
    """Is `label` cut off by the superior edge of the field of view?"""
    ax, sign = axes["sup"]
    sl: list[object] = [slice(None)] * 3
    sl[ax] = slice(-slices, None) if sign > 0 else slice(0, slices)
    return bool((mask[tuple(sl)] == label).any())


def _touches_superior_border(mask: np.ndarray, label: int, slices: int = 3) -> bool:
    """Anatomy clipped by the top of the field of view."""
    return bool((mask[:, :, -slices:] == label).any())


def _bbox(mask: np.ndarray, label: int):
    idx = np.argwhere(mask == label)
    if idx.size == 0:
        return None
    return idx.min(axis=0), idx.max(axis=0), idx


def derive_findings(
    mask: np.ndarray,
    spacing: float = 0.3,
    case_id: str = "",
    image: np.ndarray | None = None,
) -> Findings:
    """Mask (and optionally the CBCT itself) -> Findings.

    `image` must be in the same frame and shape as `mask`. It is optional so that
    mask-only callers keep working; without it the intensity-derived slots
    (currently `endodontic`) are simply left empty.
    """
    f = Findings(case_id=case_id)
    axes = infer_axes(mask)

    # --- coverage ---------------------------------------------------------
    lower = _count(mask, LAB_LOWER_JAW)
    upper = _count(mask, LAB_UPPER_JAW)
    f.mandible_included = lower > MIN_JAW_VOXELS

    if upper < MIN_UPPER_JAW_VOXELS:
        f.maxilla_coverage = "none"
    elif axes is not None:
        # A maxilla clipped by the top of the volume is "partially included";
        # the corpus says that or "not included" 87% of the time, so "full" is
        # only claimed when the upper jaw is clearly inside the FOV.
        f.maxilla_coverage = (
            "partial" if _superior_border_clipped(mask, LAB_UPPER_JAW, axes) else "full"
        )

    # If the maxilla is outside the field of view the sinuses necessarily are
    # too -- that is geometry, not a guess. The reverse is NOT claimed: TF3
    # annotates the sinuses in only ~10% of cases, so a present sinus label is
    # not evidence and its absence certainly is not.
    if f.maxilla_coverage == "none":
        f.sinuses_included = False
    elif _count(mask, LAB_SINUS_L) + _count(mask, LAB_SINUS_R) >= MIN_JAW_VOXELS:
        f.sinuses_included = True

    # condyles_included is deliberately left unset. TF3 has no condyle label, so
    # the only handle is "is the mandible clipped at the top of the FOV" -- and
    # calibration put that at 76.7% agreement, while the corpus prior ("not
    # included") is right in 342 of 343 reports. The derivation is worse than the
    # prior, so the renderer falls through to the prior instead.

    # --- dentition --------------------------------------------------------
    # Presence is trustworthy (95.6%); absence is not (65.1% at best, because
    # TF3 labels retained roots that a surgeon calls an absent tooth). Absence is
    # therefore only asserted for sites INSIDE the span of teeth we can actually
    # see -- otherwise every tooth outside a small field of view would be
    # reported missing.
    # Label ids are converted to CLINICAL FDI here (see to_clinical_fdi): every
    # tooth number leaving this function is what a surgeon would write.
    for fdi in LOWER + UPPER:
        if _count(mask, fdi) >= MIN_TOOTH_VOXELS:
            f.teeth_present[to_clinical_fdi(fdi)] = True

    # Absence is asserted across the WHOLE lower arch, not just the span between
    # the teeth we can see. The span rule was protecting precision that did not
    # need protecting: measured over 527 cases against the ground-truth reference,
    #     span only        1.04 claims/case  entailed 0.541  recall of stated 0.38
    #     whole lower arch 2.91 claims/case  entailed 0.450  recall of stated 0.84
    # so the 1.87 extra claims are entailed at a marginal rate of 0.40, well above
    # the ~0.24 break-even, and they more than double absence recall. Half of all
    # missed reference absence statements were for teeth just outside the span.
    #
    # The UPPER arch gets no absence claims at all. Measured the same way it is a
    # disaster -- 3.93 claims/case entailed 0.147, and restricting to 18/28 makes
    # it worse at 0.054 -- because a partially included maxilla clips its teeth
    # below MIN_TOOTH_VOXELS, so they read as missing when they are merely out of
    # frame. This is the failure the old span rule existed to contain; removing
    # the upper arch removes it at the source.
    lower_clinical = [to_clinical_fdi(fdi) for fdi in LOWER]
    if sum(1 for fdi in lower_clinical if f.teeth_present.get(fdi)) >= 2:
        for fdi in lower_clinical:
            f.teeth_present.setdefault(fdi, False)

    # ...with one exception, added after set A was scored for the first time.
    # When the WHOLE maxilla is in the field of view -- not merely clipped -- and
    # we can actually see teeth in it, upper-arch absence becomes a good claim.
    # Both conditions are needed. Measured over the 622 cases:
    #     maxilla "full" alone        47 cases fire, 6.87 claims  entailed 0.433
    #     ...and >= 4 upper present   41 cases fire, 5.68 claims  entailed 0.549
    # The guard matters because "full" coverage with no visible upper teeth is
    # self-contradictory -- it fired on one S case and claimed all 16 upper teeth
    # absent, none of them correctly. With the guard, all 41 firing cases are in
    # set A and none in F, P or S.
    #
    # This is the slot removed earlier as a landmine. The removal was right for
    # the data available then: with only F/P/S derived, "full" occurred once in
    # 527 cases and the claims were entailed 0.147. Set A is where the maxilla is
    # genuinely imaged, and there the same claim is entailed 0.549 against a
    # break-even of 0.178 -- A's references are so dense (29.3 phrases/case) that
    # recall is scarce and weak claims still pay.
    upper_clinical = [to_clinical_fdi(fdi) for fdi in UPPER]
    if (f.maxilla_coverage == "full"
            and sum(1 for fdi in upper_clinical if f.teeth_present.get(fdi))
            >= MIN_UPPER_TEETH_FOR_ABSENCE):
        for fdi in upper_clinical:
            f.teeth_present.setdefault(fdi, False)

    lower_present = [t for t in LOWER if f.teeth_present.get(t)]
    upper_present = [t for t in UPPER if f.teeth_present.get(t)]
    if not lower_present and not upper_present:
        f.edentulism = "complete"
    elif len(lower_present) < len(LOWER):
        f.edentulism = "partial"
    else:
        f.edentulism = "none"

    # --- canal course -----------------------------------------------------
    if axes is not None:
        course = _canal_course(mask, axes)
        if course:
            f.canal_course = course

    # --- canal / third-molar relations ------------------------------------
    # Label sides are mirrored too: TF3's "Left" canal (3) is the clinical right.
    for label_side, canal_label, label_molar in (
        ("right", LAB_CANAL_R, 48), ("left", LAB_CANAL_L, 38)
    ):
        molar = to_clinical_fdi(label_molar)
        if not f.teeth_present.get(molar):
            continue
        d = _min_distance_mm(mask, canal_label, label_molar, spacing)
        if d is not None and d <= CONTACT_MM:
            f.canal_contact[mirror_side(label_side)] = molar

    if axes is not None:
        for molar, neighbour in ((48, 47), (38, 37), (18, 17), (28, 27)):
            if _is_impacted(mask, molar, neighbour, spacing, axes):
                f.impacted.add(to_clinical_fdi(molar))

    # --- restorations -----------------------------------------------------
    f.has_bridge = _count(mask, LAB_BRIDGE) >= MIN_RESTORATION_VOXELS
    f.has_implant = _count(mask, LAB_IMPLANT) >= MIN_RESTORATION_VOXELS
    f.has_crown = _count(mask, LAB_CROWN) >= MIN_RESTORATION_VOXELS

    # Per-site crown claims. My regex calibration scored these at 17.6% site
    # precision and I nearly dropped them, but that number is measured against a
    # parser that only catches "crown on tooth NN" and misses the far commoner
    # "conservative restoration on the occlusal surface of 36" -- so it badly
    # undercounts the references. The canal_contact ablation proved this failure
    # mode is real: emission rates said we over-claimed 3x, yet removing those
    # claims dropped BOTH precision and recall. RadFact is the arbiter, not the
    # regex.
    # Per-slot isolation from here down. `_restoration_sites` needs scipy, and
    # scipy was missing from the container image: the ImportError propagated out
    # of derive_findings, out of build_report, and every case fell back to the
    # 64-word prior report. One optional finding must never be able to do that,
    # so each of the remaining slots degrades on its own.
    f.crowns = _safely(set(), _restoration_sites, mask, LAB_CROWN)

    # Implants stay site-less: an implant occupies a MISSING site, so
    # nearest-tooth returns a neighbour and would name the wrong position.

    # --- radiopaque material ----------------------------------------------
    if image is not None:
        bone = _safely(None, _bone_reference, mask, image)
        if bone is not None:
            f.endodontic = _safely(set(), _endodontic_teeth, mask, image, bone)
            # Only on teeth we are not already describing: saying "conservative
            # restoration of tooth 46" about a tooth we have just called
            # endodontically treated or crowned is a second phrase competing for
            # the same reference statement, and the judge rejects a restoration
            # claim when the matching reference phrase is about a crown.
            restored = _safely(set(), _restored_teeth, mask, image, bone)
            f.restorations = restored - f.endodontic - f.crowns

    # --- consistency ------------------------------------------------------
    # A tooth we have made a positive finding about cannot also be absent. This
    # fires for real: a tooth whose crown is clipped can fall under
    # MIN_TOOTH_VOXELS while still carrying a pulp label full of root-canal
    # filling, which produced "Absence of teeth 38, 45, 46, 47 and 48." next to
    # "Endodontic treatment of tooth 45." on P019. A contradiction costs twice --
    # the wrong claim is unentailed and it can contradict the reference phrase
    # that the right claim would have matched.
    evidence = (
        f.endodontic | f.crowns | f.restorations | f.impacted | set(f.canal_contact.values())
    )
    for fdi in evidence:
        f.teeth_present[fdi] = True

    return f


def _bone_reference(mask: np.ndarray, image: np.ndarray) -> float | None:
    """The case's own intensity scale, taken from its jaw bone.

    The volumes are not on a shared scale -- several F-set volumes clip at a
    maximum of 2400 while P-set volumes reach 3400 -- so every density rule here
    is a ratio to this value rather than an absolute threshold. Jaw bone is the
    natural reference because it is present, large, and physically the same
    tissue in every case.
    """
    if image.shape != mask.shape:
        return None
    jaw = (mask == LAB_LOWER_JAW) | (mask == LAB_UPPER_JAW)
    if int(jaw.sum()) < MIN_JAW_VOXELS:
        return None
    p95 = float(np.percentile(image[jaw], 95))
    return p95 if np.isfinite(p95) and p95 > 0 else None


def _endodontic_teeth(mask: np.ndarray, image: np.ndarray, bone_p95: float) -> set[int]:
    """Teeth whose pulp chamber contains radiopaque filling material.

    The reference reports state an endodontic finding on 0.90 teeth per case and
    two surgeons agree with each other about it more often than about any other
    finding type (77%), which makes it the most reproducible thing in the corpus
    and therefore the most worthwhile to detect.
    """
    cutoff = ENDO_BONE_RATIO * bone_p95
    treated: set[int] = set()
    for fdi in LOWER + UPPER:
        pulp = mask == pulp_label(fdi)
        if int(pulp.sum()) < MIN_PULP_VOXELS:
            continue
        # p90 rather than the max: a single bright voxel is noise or a partial
        # volume with the neighbouring dentin, whereas a filled canal is bright
        # along its whole length.
        if float(np.percentile(image[pulp], 90)) >= cutoff:
            treated.add(to_clinical_fdi(fdi))
    return treated


def _restored_teeth(mask: np.ndarray, image: np.ndarray, bone_p95: float) -> set[int]:
    """Teeth carrying dense restorative material somewhere in the crown.

    Lower arch only. Split by arch the claim behaves completely differently:
        lower  0.29 claims/case  entailed 0.583
        upper  0.33 claims/case  entailed 0.244
    The upper arch sits exactly on the break-even rate, so it contributes
    nothing, and it is measured here on GROUND-TRUTH masks -- on predicted masks
    a clipped maxilla is the least reliable thing the segmentation produces, so
    the real upper number can only be worse. The endodontic rule keeps both
    arches because its upper claims are 0.292, comfortably positive.
    """
    cutoff = RESTORATION_BONE_RATIO * bone_p95
    restored: set[int] = set()
    for fdi in LOWER:
        tooth = mask == fdi
        if int(tooth.sum()) < MIN_TOOTH_VOXELS:
            continue
        if float(np.percentile(image[tooth], 99)) >= cutoff:
            restored.add(to_clinical_fdi(fdi))
    return restored


def _restoration_sites(mask: np.ndarray, label: int, max_mm: float = 8.0,
                       spacing: float = 0.3) -> set[int]:
    """Assign each restoration blob to the tooth site it sits on.

    A crown physically caps its tooth, so nearest-tooth is the right rule. An
    implant occupies a MISSING site, so its nearest tooth is a neighbour -- those
    are returned too, but the renderer only states crowns per-site until the
    implant rule is calibrated.
    """
    from scipy import ndimage

    blobs = mask == label
    if blobs.sum() < MIN_RESTORATION_VOXELS:
        return set()
    labelled, n = ndimage.label(blobs)
    if n == 0:
        return set()

    tooth_voxels = {}
    for fdi in LOWER + UPPER:
        idx = np.argwhere(mask == fdi)
        if len(idx) >= MIN_TOOTH_VOXELS:
            tooth_voxels[fdi] = idx[:: max(1, len(idx) // 300)].astype(np.float32)
    if not tooth_voxels:
        return set()

    sites: set[int] = set()
    for blob_id in range(1, n + 1):
        pts = np.argwhere(labelled == blob_id)
        if len(pts) < MIN_RESTORATION_VOXELS:
            continue
        pts = pts[:: max(1, len(pts) // 200)].astype(np.float32)
        best, best_d = None, None
        for fdi, tv in tooth_voxels.items():
            d = float(np.sqrt(((pts[:, None, :] - tv[None, :, :]) ** 2).sum(-1).min()))
            if best_d is None or d < best_d:
                best, best_d = fdi, d
        if best is not None and best_d * spacing <= max_mm:
            sites.add(to_clinical_fdi(best))
    return sites


def local_bone_segment(row: np.ndarray, x: int) -> tuple[int, int] | None:
    """The contiguous run of mandible along `row` that contains (or is nearest to) x.

    A left-right line through the mandible crosses BOTH hemi-arches, so taking
    min/max of the whole row puts the "centre" on the patient midline and makes
    every canal look lateral. Only the local hemi-arch cross-section is meaningful
    for buccal vs lingual.
    """
    if row.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(row) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [row.size - 1]))
    best = None
    for s, e in zip(starts, ends):
        lo, hi = int(row[s]), int(row[e])
        if hi - lo < 3:
            continue
        dist = 0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
        if best is None or dist < best[0]:
            best = (dist, lo, hi)
    return (best[1], best[2]) if best is not None else None


def _canal_course(mask: np.ndarray, axes: dict) -> str | None:
    """Lingual vs buccal: is the canal medial or lateral within the mandible?

    For each sampled canal voxel, find the mandible's medial and lateral extent
    along the left-right axis at that same position, and record where the canal
    sits between them (0 = midline, 1 = outer cortex). Mean below 0.5 means the
    canal runs predominantly lingual.
    """
    # The canal labels carve a hole out of the jawbone label, which would split
    # the local bone run in two and leave the canal sitting just outside it.
    # Filling the canals back in keeps the cross-section solid.
    jaw = (mask == LAB_LOWER_JAW) | (mask == LAB_CANAL_L) | (mask == LAB_CANAL_R)
    if jaw.sum() < MIN_JAW_VOXELS:
        return None
    lr_axis = axes["left"][0]
    other = [a for a in (0, 1, 2) if a != lr_axis]
    jaw_midline = float(np.argwhere(jaw)[:, lr_axis].mean())

    rel: list[float] = []
    for label in (LAB_CANAL_L, LAB_CANAL_R):
        idx = np.argwhere(mask == label)
        if len(idx) < 200:
            continue
        side = 1.0 if float(idx[:, lr_axis].mean()) > jaw_midline else -1.0
        for voxel in idx[:: max(1, len(idx) // 400)]:
            sl: list[object] = [0, 0, 0]
            sl[lr_axis] = slice(None)
            sl[other[0]] = int(voxel[other[0]])
            sl[other[1]] = int(voxel[other[1]])
            x = int(voxel[lr_axis])
            seg = local_bone_segment(np.flatnonzero(jaw[tuple(sl)]), x)
            if seg is None:
                continue
            lo, hi = seg
            half = (hi - lo) / 2.0
            if half < 1.5:
                continue
            rel.append(side * (x - (hi + lo) / 2.0) / half)

    if len(rel) < 50:
        return None
    # Threshold set by sweep_thresholds.py against the report-parsed references.
    # The corpus vocabulary is "lingual" vs "regular", never "buccal".
    return "lingual" if float(np.mean(rel)) < CANAL_LINGUAL_CUTOFF else "regular"


def _min_distance_mm(mask: np.ndarray, label_a: int, label_b: int, spacing: float) -> float | None:
    a = np.argwhere(mask == label_a)
    b = np.argwhere(mask == label_b)
    if len(a) == 0 or len(b) == 0:
        return None
    a = a[:: max(1, len(a) // 1500)].astype(np.float32)
    b = b[:: max(1, len(b) // 1500)].astype(np.float32)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).min()
    return float(np.sqrt(d2) * spacing)


def _is_impacted(mask: np.ndarray, molar: int, neighbour: int, spacing: float, axes: dict) -> bool:
    """A third molar whose occlusal tip sits well short of its neighbour's."""
    m = _bbox(mask, molar)
    n = _bbox(mask, neighbour)
    if m is None or n is None:
        return False
    if len(m[2]) < MIN_TOOTH_VOXELS or len(n[2]) < MIN_TOOTH_VOXELS:
        return False

    ax, sign = axes["sup"]
    # How far each tooth reaches toward its own occlusal plane: lower teeth erupt
    # upward, upper teeth downward.
    if molar in (38, 48):
        reach_m = m[1][ax] if sign > 0 else -m[0][ax]
        reach_n = n[1][ax] if sign > 0 else -n[0][ax]
    else:
        reach_m = -m[0][ax] if sign > 0 else m[1][ax]
        reach_n = -n[0][ax] if sign > 0 else n[1][ax]
    return float(reach_n - reach_m) * spacing >= IMPACTION_MM
