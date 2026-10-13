"""Cross-section video + five-view panoramic stills dataset (finetune_qwen_v1).

This is the wiring deliberately left to the training run: it puts the
two halves of the toothfairy representation into one Qwen message for the first time.

  - **video**  = ``render.render_case`` cross-section stack — buccolingual slices through
                 the dental arch (the "slicing" half).                    -> ``rc.frames``
  - **stills** = ``views.five_views`` DEFAULT_VIEWS (the panoramic half):
                 ``pano_union, pano_mandible, pano_rgb, occlusal_rgb, mpr_rgb``.

Layout: the five stills sit AFTER the video in the user turn, matching HEADLINE's
video-then-image order and ``toothfairy.dataset.assemble_messages``. Each still is named
in the layout hint (built from the views actually returned, so a dropped view never
dangles), and a single legend explains the RGB channels + laterality once.

All the Qwen plumbing (SYSTEM_PROMPT, ``video_content``, ``processor_inputs``,
``extract_vision``, ``load_split_records``, completion-only masking in the trainer) is
reused unchanged from ``toothfairy.dataset``, so train/infer preprocessing is identical.

- ``regime`` and the view/frame counts are logged per sample, and a template-regime
    or zero-still sample is DROPPED from training (it would teach a full report from two
    duplicated axial slices — the hallucination the system prompt forbids).
  - online augmentation — with prob ``mirror_p`` the volume is flipped left-right AND the
    target is the mirrored canonical prose (``data_canonical.mirror``); ``z_jitter_mm`` shifts
    the occlusal band. Mirror is skipped when ``mirror_is_safe`` is False (measurement-shaped
    FDI token) or when the structured record is absent (non-canonical target).
  - the FOV note fires on measured superior truncation, not on the 99%-firing z<100 rule.
  - ``video_axis`` is accepted for launcher compatibility; only "buccolingual" is implemented.

aux-head: with ``tooth_labels=True`` each
sample also carries the 32-slot per-tooth state vector from ``data_canonical/tooth_state.py``, for
the auxiliary head in ``train.py``. Two things make this non-obvious:

  - the labels are derived from the **mirrored** record when augmentation mirroring fires, never from the
    original — the quadrants must swap in lockstep with the pixels;
  - the upper-arch mask needs ``RenderedCase.superior_truncated``, which only exists AFTER the
    render, so the labels are built at the end of ``__getitem__``, not at the top.

``data_canonical`` is not vendored into the submission container, so — like ``_load_mirror`` — the
import is lazy and never happens unless ``tooth_labels`` is set.

with ``chart_concat=True`` the training target becomes
``chart block + "\\n\\nReport:\\n" + canonical prose`` (``data_canonical/chart_render.py``), and the
existing completion-only mask supervises both segments at weight 1.0. Two ordering constraints,
both silent if broken:

  - the chart is rendered from the **mirrored** record when augmentation fires, so the FDI quadrant swap
    propagates into the chart for free (mirror-equivariance proves the renderer is equivariant);
  - the chart's upper-arch gate needs ``RenderedCase.superior_truncated``, which only exists
    after the render — so the case is rendered with ``report=None`` and the assistant turn is
    appended afterwards, rather than passing the target into the render call.
"""
from __future__ import annotations

import _env  # noqa: F401  (blocks broken transformer_engine; must precede heavy imports)

import logging
import random
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from toothfairy.config import FOV_LIMITED_MM, FRAME_TILE, N_CROSS_SECTIONS
from toothfairy.dataset import (  # reuse plumbing
    SYSTEM_PROMPT,               # noqa: F401  (re-exported for the trainer/infer)
    video_content,
    extract_vision,              # noqa: F401
    processor_inputs,            # noqa: F401
    load_split_records,          # noqa: F401
    _local_volume_path,
    _zip,
    _VIDEO_HINT, _FOV_NOTE, _TRUNC_NOTE, _TASK,
)
from toothfairy.render import CROSS_STEP_MM, render_case
from toothfairy.views import (  # noqa: F401
    DEFAULT_VIEWS, MAX_MPX, TILE_V2, VIEWS_V2, five_views,
)
from toothfairy.volume_io import load_canonical_volume

log = logging.getLogger("dataset_fiveviews")


def _data_canonical_on_path():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_canonical"))


def _load_mirror():
    """Lazily import the canonical mirror + prose renderer (data_canonical/, repo root, not a
    package). MF4: imported ONLY when augmentation is used, so the submission container — which
    vendors this file but not data_canonical/ and never augments — does not crash at import."""
    _data_canonical_on_path()
    from mirror import mirror_record, mirror_is_safe
    import render_prose
    return mirror_record, mirror_is_safe, render_prose


def _load_tooth_state():
    """Lazily import the aux-head per-tooth label builder. Same containment rule as _load_mirror:
    data_canonical/ is not in the submission image, so this must never run at import time."""
    _data_canonical_on_path()
    from tooth_state import tooth_state, FDI_ORDER, N_ATTR, N_SLOTS, IGNORE_INDEX
    return tooth_state, FDI_ORDER, N_SLOTS, N_ATTR, IGNORE_INDEX


def _load_chart_v2(vocab: str = "v2"):
    """Lazily import the multilabel chart renderer. Same rule as ``_load_chart_render``:
    ``data_canonical/`` is not vendored into the submission image, so this must never run at
    import time.

    ``vocab="v3"`` selects the VOCABULARY round's merge (`data_canonical/chart_v3.py`): IAN split
    into `course` + `position`, and annotator silence spelled `n/a` so the loss can mask it. The
    TOOTH half is identical — `chart_v3._test` asserts that on all 622 cases — so a v2-vs-v3
    comparison is a one-change A/B on the non-tooth vocabulary.
    """
    _data_canonical_on_path()
    if vocab == "v3":
        from chart_v3 import chart_text_from_records
        from chart_v2 import load_all_records          # records are vocabulary-independent
        return chart_text_from_records, load_all_records
    from chart_v2 import chart_text_from_records, load_all_records
    return chart_text_from_records, load_all_records


def _load_chart_render():
    """Lazily import the chart renderer. Same containment rule again — the container splits
    charts with ``toothfairy.chart_parse`` (stdlib) and never renders one."""
    _data_canonical_on_path()
    from chart_render import chart_text, concat_target
    return chart_text, concat_target

# ── Layout hint: name each still and explain the RGB channels + laterality once ──
_VIEW_LABEL = {
    "pano_union":    "a panoramic reconstruction of the dental arch (grayscale pseudo-OPG)",
    "pano_maxilla":  "a panoramic reconstruction fitted to the maxilla",
    "pano_mandible": "a panoramic reconstruction fitted to the mandible",
    "pano_rgb":      "a color-windowed panoramic of the same arch",
    "pano_mand_rgb": "a color-windowed panoramic fitted to the mandible",
    "occlusal":      "an occlusal (top-down) projection of the maxilla and mandible",
    "occlusal_rgb":  "a color-windowed occlusal (top-down) projection of the maxilla and mandible",
    "mpr":           "a mid-sagittal and coronal cross-section",
    "mpr_rgb":       "a color-windowed mid-sagittal and coronal cross-section",
    "hemiarch":      "magnified right and left hemi-arch panoramics",
}
# `views.split_wide` names an arc tile `<view>_<i>of<k>`. The label has to say WHICH segment,
# or k magnified panoramics arrive as k identical descriptions and the only thing telling the
# model where a tile sits on the arch is its order in the message. Segment names are anatomical
# and follow the same convention as the rest of the prompt: image-left is the patient's right,
# so tile 1 is always the patient's RIGHT end of the arc.
_SEGMENTS = {
    2: ("the patient's right half", "the patient's left half"),
    3: ("the patient's right posterior segment", "the anterior segment",
        "the patient's left posterior segment"),
    4: ("the patient's right posterior segment", "the patient's right anterior segment",
        "the patient's left anterior segment", "the patient's left posterior segment"),
    # 6 tiles ~= 28 mm of arc each, i.e. 3-4 teeth per image — a periapical-radiograph field.
    # Named by dentition rather than by index so the phrase carries FDI-relevant information.
    6: ("the patient's right molar region", "the patient's right premolar region",
        "the patient's right canine and incisor region",
        "the patient's left canine and incisor region",
        "the patient's left premolar region", "the patient's left molar region"),
}


def _view_label(key: str) -> str:
    """Prompt phrase for a still, including the `<view>_<i>of<k>` arc tiles."""
    if key in _VIEW_LABEL:
        return _VIEW_LABEL[key]
    base, _, pos = key.rpartition("_")
    if base in _VIEW_LABEL and "of" in pos:
        i, _, k = pos.partition("of")
        if i.isdigit() and k.isdigit():
            i, k = int(i), int(k)
            seg = _SEGMENTS.get(k, ())
            where = seg[i - 1] if 0 < i <= len(seg) else f"segment {i} of {k}"
            return f"{_VIEW_LABEL[base]}, magnified — {where}"
    return key
_RGB_LEGEND = (
    " In the color-windowed views, red = soft tissue / lucent lesions, "
    "green = dentin / alveolar bone / inferior alveolar nerve canal, and "
    "blue = enamel / cortical bone / restorations and implants."
)
# ⚠ The video's colour map is NOT the stills'. The stills use hu.RGB_WINDOWS — cumulative ramps
# over ray *reductions*, where cortical bone reads blue. The video uses hu.TISSUE_BANDS —
# band-pass classes over single voxels, where cortical bone reads GREEN and only enamel and
# denser reads blue. Emitting the stills' legend for both would tell the model that cortical
# bone is blue in a frame where it is green, which is worse than no legend at all.
_VIDEO_RGB_LEGEND = (
    " In the cross-section video the colors are tissue classes: red = soft tissue and lucent "
    "areas, green = bone (cancellous through cortical, including the mandibular canal wall), "
    "and blue = enamel, restorations and implants, brightest for metal."
)
# Load-bearing for FDI quadrant assignment: the arch is unrolled
# low-x -> high-x and x is patient-left in LAS, so image-left is the patient's right. This
# holds for the panoramic stills and the per-jaw occlusal projection; the video's horizontal
# axis is lingual->buccal, so laterality there is carried by frame ORDER (right side first).
_LATERALITY = (
    " In the panoramic and occlusal views the left of the image is the patient's right side; "
    "the cross-section video sweeps from the patient's right side to the left."
)
# with no video there is no frame order to carry laterality, and the second clause above
# would describe an input that is not there. Same first clause, nothing promised beyond it.
_LATERALITY_STILLS = (
    " In these views the left of the image is the patient's right side."
)
_STILLS_ONLY_HINT = "This CBCT of the jaws is shown as"


def _user_text(n_frames: int, view_keys: list[str], z_mm: float | None,
               superior_truncated: bool = False, video_rgb: bool = False) -> str:
    # the video is a buccolingual arch sweep, not an inferior->superior stack. _VIDEO_HINT
    # still says "cross-sectional slices through the dental arch"; we do not append the false
    # "inferior to superior" claim (it was dropped from the hint below).
    labels = [_view_label(k) for k in view_keys]
    if n_frames:
        txt = _VIDEO_HINT.format(n_frames=n_frames)
        if labels:
            txt += ", followed by " + ", ".join(labels)
        txt += "."
        txt += _LATERALITY
    else:
        # panoramic-only. Never announce a video that was not sent.
        txt = _STILLS_ONLY_HINT + " " + (", ".join(labels) if labels else "a reconstruction")
        txt += "."
        txt += _LATERALITY_STILLS
    # `_rgb` is no longer always the suffix: an arc tile is `pano_rgb_2of3` (views.split_wide).
    # Matching on endswith would drop the colour legend from a tiled RGB run — the model would
    # be sent three colour-windowed panoramics and never told what the channels mean.
    if any("_rgb" in k for k in view_keys):
        txt += _RGB_LEGEND
    # gate on n_frames too. --video-rgb with no video would otherwise describe the tissue
    # colour map of a video the model was never sent — the same class of bug as _FOV_NOTE.
    if video_rgb and n_frames:
        txt += _VIDEO_RGB_LEGEND
    # fire the FOV note on MEASURED superior truncation, not on the z<100 rule that fired
    # on 99% of training prompts (and would never fire on the hidden complete scans).
    if superior_truncated:
        txt += _TRUNC_NOTE
    elif z_mm is not None and z_mm < FOV_LIMITED_MM:
        txt += _FOV_NOTE.format(z_mm=z_mm)
    return txt + _TASK


def assemble_fiveviews_messages(frames: list, stills: list, z_mm: float | None,
                                report: str | None,
                                superior_truncated: bool = False,
                                video_rgb: bool = False) -> list[dict]:
    """Wrap the cross-section video + five stills into Qwen chat messages.

    ``report=None`` -> inference (messages end at the user turn)."""
    view_keys = [k for k, _ in stills]
    vision = video_content(frames, stills)          # video item, then one image per still
    user_text = _user_text(len(frames), view_keys, z_mm, superior_truncated, video_rgb)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": vision + [{"type": "text", "text": user_text}]},
    ]
    if report is not None:
        msgs.append({"role": "assistant", "content": report})
    return msgs


def render_fiveviews_messages(volume_member: str, zf=None, local_path=None,
                              report: str | None = None,
                              n_cross: int = N_CROSS_SECTIONS,
                              views: tuple[str, ...] = DEFAULT_VIEWS,
                              mirror: bool = False,
                              z_jitter_mm: float = 0.0,
                              render_v2: bool = False,
                              video_rgb: bool = False,
                              tile: int = FRAME_TILE,
                              video_tile: int = FRAME_TILE,
                              still_token_budget: int = 0,
                              pano_split: int = 1,
                              max_mpx: float = MAX_MPX,
                              still_per_view: bool = False):
    """Render one case into Qwen messages.

    Returns ``(messages, regime, n_frames, n_stills, superior_truncated)`` — the counts so the
    caller can log/drop degenerate renders, the truncation flag so it can gate the aux-head
    upper-arch labels on the pixels rather than on the annotator's FOV opinion. Callers that
    only want the messages index ``[0]`` (submission/entrypoint.py), so the tuple may grow."""
    arr, spacing = load_canonical_volume(volume_member, zf=zf, local_path=local_path)
    arr = arr.astype(np.float32, copy=False)
    if mirror:                                   # LR flip on x (patient left<->right, LAS)
        arr = np.ascontiguousarray(arr[::-1, :, :])
    z_jit_vox = int(round(z_jitter_mm / float(spacing[2]))) if z_jitter_mm else 0
    # Video half: cross-section stack + FOV/truncation flags. render_case's own legacy stills
    # are discarded; the panoramic half comes from five_views. with_mpr=False skips the unused
    # sinus/condyle MPR stills.
    # render_v2 (2026-08-10 audit): hybrid ray reduction, cropped stills, validated crown
    # bands, and a video that is slab-reduced + HU-windowed + stripped of out-of-FOV frames.
    # Every lever defaults off — the submission container calls this function.
    # n_cross <= 0 means panoramic-only — the arch fit still runs (z_mm, regime and the
    # truncation flag come from it) but no frames are produced and no video item is emitted.
    rc = render_case(arr, spacing, n_cross=max(n_cross, 0), with_mpr=False,
                     z_jitter_vox=z_jit_vox, with_video=n_cross > 0,
                     video_slab_mm=(CROSS_STEP_MM if render_v2 else 0.0),
                     video_rgb=video_rgb, video_drop_empty=render_v2, tile=video_tile)
    stills = five_views(arr, spacing, views=views, z_jitter_vox=z_jit_vox, tile=tile,
                        reduce=("hybrid" if render_v2 else "mean"),
                        crop=render_v2, validate_rows=render_v2,
                        # The budget applies under --render-v2 OR --still-per-view. It used to
                        # be gated on render_v2 alone, which made --still-per-view a SILENT
                        # NO-OP: fit_token_budget early-returns on budget <= 0, so the flag
                        # that exists to stop one view starving never got consulted. Measured
                        # 2026-08-13 on F001, five views, still_mpx=1.59 — passing
                        # --still-per-view alone produced byte-identical renders, with
                        # pano_union at 126 tokens and pano_mandible at 171 next to mpr_rgb's
                        # 1008. Both flags still default off, so the container path
                        # (submission/entrypoint.py, which passes neither) is unchanged.
                        still_token_budget=(still_token_budget
                                            if (render_v2 or still_per_view) else 0),
                        pano_split=pano_split, max_mpx=max_mpx,
                        still_per_view=still_per_view)
    msgs = assemble_fiveviews_messages(
        rc.frames, stills, rc.z_mm, report, superior_truncated=rc.superior_truncated,
        video_rgb=video_rgb)
    return msgs, rc.regime, len(rc.frames), len(stills), rc.superior_truncated


class ToothFairyFiveViewsDataset(Dataset):
    """__getitem__ -> {messages, case_id, report} (or None on a render failure / dropped
    degenerate render; the collate drops None). Volumes are read local-cache-first, zip
    fallback (same contract as ToothFairyQwenDataset / ToothFairyAxialDataset)."""

    def __init__(self, records: list[dict], training: bool = True,
                 n_cross: int = N_CROSS_SECTIONS,
                 views: tuple[str, ...] = DEFAULT_VIEWS,
                 render_v2: bool = False, video_rgb: bool = False, tile: int = FRAME_TILE,
                 video_tile: int = FRAME_TILE, still_token_budget: int = 0,
                 pano_split: int = 1, max_mpx: float = MAX_MPX,
                 still_per_view: bool = False, video_axis: str = "buccolingual",
                 augment: bool = False, mirror_p: float = 0.5,
                 z_jitter_mm: float = 0.0, canonical: bool = False,
                 seed: int = 0, tooth_labels: bool = False,
                 chart_concat: bool = False, chart_v2: bool = False,
                 chart_vocab: str = "v2"):
        if video_axis != "buccolingual":
            raise NotImplementedError(
                f"video_axis={video_axis!r}: only 'buccolingual' is implemented "
                "(the axial path was not ported)")
        self.records = records
        self.training = training
        self.n_cross = n_cross
        self.views = views
        self.render_v2 = render_v2
        self.video_rgb = video_rgb
        self.tile = tile
        self.video_tile = video_tile
        self.still_token_budget = still_token_budget
        self.pano_split = pano_split
        self.max_mpx = max_mpx
        self.still_per_view = still_per_view
        self.augment = augment
        self.mirror_p = mirror_p
        self.z_jitter_mm = z_jitter_mm
        self.canonical = canonical
        self.tooth_labels = tooth_labels
        self.chart_concat = chart_concat
        # chart-v2: the target is the multilabel chart ALONE (no prose). The per-annotator
        # records are read once here rather than per sample — 1000 small files inside the
        # training loop would be pure I/O, and the mirror path needs them in memory anyway.
        self.chart_v2 = chart_v2
        self.chart_vocab = chart_vocab
        self._chart_v2_fn = None
        self._v2_records = None
        if chart_v2:
            if chart_concat:
                raise ValueError("--chart-v2 and --chart-concat are different targets; pick one")
            if chart_vocab not in ("v2", "v3"):
                raise ValueError(f"chart_vocab must be 'v2' or 'v3', got {chart_vocab!r}")
            self._chart_v2_fn, _load_all = _load_chart_v2(chart_vocab)
            self._v2_records = _load_all()
        if tooth_labels and not canonical:
            raise ValueError("tooth_labels needs --canonical: the 32-slot state vector is built "
                             "from the structured record, which only the canonical splits carry")
        if chart_concat and not canonical:
            raise ValueError("chart_concat needs --canonical: the chart is rendered from the "
                             "structured record, which only the canonical splits carry")
        self._seed = seed
        self._epoch = 0
        self._drops = 0     # degenerate renders dropped from training
        self._chartless = 0  # training samples that carry no structured record to chart
        self._mirror = None  # lazily loaded (mirror_record, mirror_is_safe, render_prose)
        self._tooth = None   # lazily loaded (tooth_state, FDI_ORDER, N_SLOTS, N_ATTR, IGNORE)
        self._chart = None   # lazily loaded (chart_text, concat_target)

    def set_epoch(self, epoch: int):
        """MF1: the trainer calls this each epoch so the per-sample RNG (mirror decision +
        z-jitter) varies epoch to epoch. Requires the loader's ``persistent_workers=False`` so
        respawned workers pick up the new epoch."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        report = rec["report"] if self.training else None
        # MF1: mix idx, epoch and the run seed so a case is augmented DIFFERENTLY each epoch
        # (and a WeightedRandomSampler re-draw of the same idx is not byte-identical).
        rng = random.Random(((idx + 1) * 0x9E3779B1)
                            ^ (self._epoch * 0x85EBCA6B)
                            ^ (self._seed & 0xFFFFFFFF))

        mirror = False
        z_jit = 0.0
        struct = rec.get("record")
        v2_records = self._v2_records.get(rec["case_id"]) if self.chart_v2 else None
        if self.training and self.augment:
            if self._mirror is None:
                self._mirror = _load_mirror()
            mirror_record, mirror_is_safe, render_prose = self._mirror
            if self.chart_v2:
                # mirror-equivariance holds on ALL 622 cases for the v2 chart, because it is derived from
                # structured enums and quadrant-swapped FDI keys rather than from prose. So
                # `mirror_is_safe` — a guard against measurement-shaped tokens being remapped
                # inside free text — does not gate this path, and the 7 records it excludes in
                # v1 are augmentable here.
                if v2_records and rng.random() < self.mirror_p:
                    mirror = True
                    v2_records = [mirror_record(r) for r in v2_records]
            elif struct is not None and mirror_is_safe(struct) and rng.random() < self.mirror_p:
                mirror = True
                # the record used for the TARGET is also the record used for the tooth
                # labels. Deriving labels from the un-mirrored record would train the head on
                # the wrong quadrants — silently, since the loss would still fall.
                struct = mirror_record(struct)
                report = render_prose.render(struct)   # target must match the flip
            if self.z_jitter_mm:
                z_jit = rng.uniform(-self.z_jitter_mm, self.z_jitter_mm)

        local_path = _local_volume_path(rec["case_id"])
        # with --chart-concat the assistant turn cannot be built before the render, because
        # the chart's upper-arch gate needs the render's `superior_truncated`. Render with no
        # target and append the turn below; `assemble_fiveviews_messages` does exactly that.
        deferred = (self.chart_concat or self.chart_v2) and report is not None
        try:
            msgs, regime, n_frames, n_stills, truncated = render_fiveviews_messages(
                rec["volume_member"],
                zf=None if local_path else _zip(),
                local_path=local_path,
                report=None if deferred else report,
                n_cross=self.n_cross,
                views=self.views,
                render_v2=self.render_v2, video_rgb=self.video_rgb,
                tile=self.tile, video_tile=self.video_tile,
                still_token_budget=self.still_token_budget,
                pano_split=self.pano_split, max_mpx=self.max_mpx,
                still_per_view=self.still_per_view,
                mirror=mirror,
                z_jitter_mm=z_jit,
            )
        except Exception:  # noqa: BLE001 — a bad volume shouldn't kill training
            return None
        # a template-regime render is two duplicated axial slices, and a zero-still render
        # is a video-only sample with a different prompt. Both are degenerate; drop them in
        # training (keep them at inference so every case still gets an output).
        if self.training and (regime == "template" or n_stills == 0):
            self._drops += 1
            log.warning("drop %s: regime=%s frames=%d stills=%d (total drops=%d)",
                        rec["case_id"], regime, n_frames, n_stills, self._drops)
            return None
        chart = None
        if deferred and self.chart_v2:
            # The v2 target is the chart ALONE — no delimiter, no prose. The report is rendered
            # deterministically from the chart at inference (`toothfairy/chart_prose_v2.py`),
            # so nothing is lost by not training on it, and every supervised token is spent on
            # teeth instead of on wording.
            if not v2_records:
                self._chartless += 1
                log.warning("drop %s: --chart-v2 but no per-annotator records (total=%d)",
                            rec["case_id"], self._chartless)
                return None
            report = self._chart_v2_fn(v2_records, superior_truncated=truncated)
            chart = report
            msgs.append({"role": "assistant", "content": report})
        elif deferred:
            if struct is None:
                # Every canonical row carries a record, so this is a corpus regression, not a
                # tolerable sample: a report-only target inside a chart-concat run teaches the
                # model the delimiter is optional. Drop it and say so.
                self._chartless += 1
                log.warning("drop %s: --chart-concat but no structured record "
                            "(total=%d)", rec["case_id"], self._chartless)
                return None
            if self._chart is None:
                self._chart = _load_chart_render()
            chart_text, concat_target = self._chart
            chart = chart_text(struct, superior_truncated=truncated)
            report = concat_target(struct, report, superior_truncated=truncated)
            msgs.append({"role": "assistant", "content": report})
        item = {"messages": msgs, "case_id": rec["case_id"], "report": report,
                "superior_truncated": bool(truncated)}
        if chart is not None:
            item["chart"] = chart
        if self.tooth_labels:
            item.update(self._tooth_item(struct, truncated))
        return item

    def _tooth_item(self, struct, superior_truncated: bool) -> dict:
        """aux-head labels for one sample. ``struct`` is the MIRRORED record when the flip fired.

        A record-less sample still returns full-size tensors (all-ignore) so the collate can
        stack a batch without special-casing; the head then contributes no gradient for it."""
        import torch
        if self._tooth is None:
            self._tooth = _load_tooth_state()
        tooth_state, _fdi_order, n_slots, n_attr, ignore = self._tooth
        if struct is None:
            return {"tooth_status": torch.full((n_slots,), ignore, dtype=torch.long),
                    "tooth_attr": torch.zeros(n_slots, n_attr),
                    "tooth_attr_mask": torch.zeros(n_slots, n_attr, dtype=torch.bool)}
        labels, mask = tooth_state(struct, superior_truncated=superior_truncated)
        return {"tooth_status": torch.from_numpy(labels["status"]),
                "tooth_attr": torch.from_numpy(labels["attr"]),
                "tooth_attr_mask": torch.from_numpy(mask["attr"])}
