"""ReportDataset — region-feature cache + per-region ground-truth targets.

Builds the targets used by per-structure decode, where every region is generated on its own.

Features are cached per patient (region_features/<pid>.npz, one pass of the frozen
segmentation network); the ground truth (categorical claim + region text) depends on the GT
variant:
  reconciled — one anat_dir/<pid>.json + dense_dir/<pid>.json per patient.
  sampled    — one file per report for multi-report patients, re-drawn at random every epoch
               (train), fixed by seed (val).
Splits are per patient (data/splits/{train,val}.txt), so a multi-report patient never straddles
a split.

Region target text = the anatomical buckets distributed over regions (the region-type embedding
carries the identity, so no header is prepended). The categorical claim ground truth (findings
head) comes from the dense labels via build_axis_labels/build_nonteeth_labels.
"""
from __future__ import annotations

import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ReportModelConfig
from ..schema.claims import FDI_ALL, AXES, build_axis_labels, build_nonteeth_labels
from ..perception import RegionBag

# EOS gate: an imaged tooth that is normal is always fed with an empty target
# (immediate EOS), so the model learns to gate itself.
_STATE_NOT_AVAILABLE = AXES[0].categories.index("not_available")   # out-of-field index (=7)
_EOS_NONTEETH = ("maxilla", "mandible", "nerve_right", "nerve_left",
                 "cls_upper", "cls_lower")  # 4 non-tooth + 2 arch slots: always fed, no leak
# The teeth_arch:upper/lower slots were dropped: the anatomical ground truth has no such bucket.
# perio_stage moved to the maxilla/mandible claims; the arch CLS queries remain in the model
# structure only.


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {ln}" for ln in lines)


# Switch that drops the two arch CLS slots (config `encoder.drop_arch_cls` -> `set_drop_arch_cls`).
_DROP_ARCH_CLS = False


def set_drop_arch_cls(on: bool) -> None:
    """Call once with the config value at model build time — the single point that keeps
    training and evaluation feeding the same set of slots."""
    global _DROP_ARCH_CLS
    _DROP_ARCH_CLS = bool(on)


_TOKEN_REGIONS = tuple(f"fdi:{f}" for f in FDI_ALL) + ("maxilla", "mandible",
                                                       "nerve_right", "nerve_left",
                                                       "cls_upper", "cls_lower")


def build_region_targets(anat: dict) -> dict[str, str]:
    """anatomical GT -> {region_id: target_text}. Empty regions are omitted (lossless skip).

    region_id convention (the report model maps these onto anatomical tokens):
      'maxilla'|'mandible'        : single non-tooth token (slots 0, 1)
      'nerve_right'|'nerve_left'  : right/left mandibular canal (slots 2, 3)
      'fdi:<FDI>'                 : that tooth's token

    The keys map 1:1 onto anatomical tokens, so this is a lookup and not a routing decision.
    A file carrying none of them is rejected rather than guessed at: inferring the slot from the
    wording of a sentence misroutes (quadrant notation in particular), and a target that silently
    lands on the wrong tooth is invisible in the loss curve.
    """
    if not any(k in anat for k in _TOKEN_REGIONS):
        raise KeyError(
            "anatomical ground truth carries none of the 38 token keys "
            "(fdi:<FDI>, maxilla, mandible, nerve_right, nerve_left, cls_upper, cls_lower); "
            "see DATA.md for the expected format")
    out: dict[str, str] = {}
    for key in _TOKEN_REGIONS:
        lines = [s.strip() for s in anat.get(key, []) if s and s.strip()]
        if lines:
            out[key] = _bullets(lines)
    return out


def build_region_targets_eos(anat: dict, fdi_state: np.ndarray,
                             include_not_available: bool = False) -> dict[str, str]:
    """Extended targets for the EOS gate: finding regions (text) + normal regions ('' -> EOS).

    On top of build_region_targets (findings only) this (1) always adds the 4 non-tooth slots and
    (2) adds every tooth, with an empty target for those without a finding. An empty target makes
    assemble_chat_sample train on [ctx=-100, EOS] only, so the model learns to emit EOS (say
    nothing) for a normal tooth. fdi_state (32,) = build_axis_labels(dense)[:,0].

    include_not_available:
      False (legacy) — out-of-field (not_available) teeth are skipped. ⚠️ The set of fed regions
        then **depends on the state axis of the ground truth**, so evaluation leaks the answer
        (the hidden test set carries no such information), and the submission pipeline already
        feeds all 32 teeth, so training (28.5 on average) and submission (38) differ.
      True — out-of-field teeth are fed too, with an empty target (EOS). The teeth buckets of the
        ground truth genuinely hold no sentence for an out-of-field tooth, so EOS is faithful to
        the reference, and the fed set is fixed at 36 slots regardless of patient, which makes
        the leak zero. Field-of-view facts are carried by the maxilla token's sentences. On the
        564-patient training split this adds 5,383 regions (+33%) on the normal side, so
        **false_eos_lambda has to go up** (lambda=1.0 lowers the share of findings from 46% to
        35%; ~1.6 restores the balance).
    """
    out = dict(build_region_targets(anat))                # regions that have findings (text)
    # With drop_arch_cls=True the two arch CLS slots are **not fed at all**.
    # It is a process-global switch because this function is called from five places (trainer,
    # evaluation, rollouts); threading an argument through all of them means one missed call
    # site silently splits the fed set between training and evaluation.
    nonteeth = tuple(r for r in _EOS_NONTEETH
                     if not (_DROP_ARCH_CLS and r in ("cls_upper", "cls_lower")))
    for rid in nonteeth:
        out.setdefault(rid, "")                           # non-tooth: always fed (normal=EOS)
    for i, f in enumerate(FDI_ALL):
        if not include_not_available and int(fdi_state[i]) == _STATE_NOT_AVAILABLE:
            continue                                      # legacy: skip out-of-field teeth
        out.setdefault(f"fdi:{f}", "")                    # normal tooth -> EOS
    return out


def merge_anat_union(anats: list[dict]) -> dict:
    """Several per-report anatomical ground truths -> **line union** per region (deduplicated,
    order preserved).

    Used so that reward scoring compares against the **union of all of a patient's reports**
    instead of the single report drawn for that epoch (`grpo.reward_gt=union`). The reference of
    the final evaluation is already such a union, so a reward that sees only one report
    **penalises a real finding recorded in another report as "spoken about a normal region"**
    (13.7% of the normal slots in the training split are such positions).

    Deduplication **compares** on whitespace-normalised, case-folded text but stores the original
    (stripped only), so sentences are never rewritten. Non-list keys (metadata such as
    `report_id` or `_route`) are dropped. With a single report the original dict is returned
    unchanged (identical to sampled).
    """
    if not anats:
        raise ValueError("merge_anat_union: nothing to merge — the reference must not "
                         "silently become all-normal")
    if len(anats) == 1:
        return anats[0]
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for anat in anats:
        for key, val in anat.items():
            if not isinstance(val, list):
                continue
            for raw in val:
                if not isinstance(raw, str):
                    continue
                s = raw.strip()
                if not s:
                    continue
                k = " ".join(s.split()).lower()
                bucket = seen.setdefault(key, set())
                if k in bucket:
                    continue
                bucket.add(k)
                out.setdefault(key, []).append(s)
    return out


def union_fdi_state(states: list[np.ndarray]) -> np.ndarray:
    """Per-report state axes (32,) -> union state. A tooth stays not_available only if **every**
    report called it that; if any report imaged it, it counts as imaged.

    Only used on the `build_region_targets_eos(include_not_available=False)` path. The current
    rollout config sets `eos_include_not_available: true`, so the value is not consumed there,
    but the union convention is defined so that both paths agree.
    """
    out = np.array(states[0], copy=True)
    for st in states[1:]:
        na = out == _STATE_NOT_AVAILABLE
        out[na] = np.asarray(st)[na]
    return out


def _index_reports(anat_dir: Path) -> dict[str, list[str]]:
    """patient -> [anat_path, ...] (sampled arm; per-report anatomical files with report_id)."""
    by_pid: dict[str, list] = defaultdict(list)
    for f in sorted(glob.glob(str(anat_dir / "*.json"))):
        d = json.load(open(f))
        rid = d.get("report_id")
        if rid:
            by_pid[rid].append(f)
    return by_pid


def read_split(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


class ReportDataset(Dataset):
    """Region cache + per-region ground truth. A patient read more than once has one file per
    report, and one of them is drawn per epoch."""

    def __init__(self, split_pids: list[str], cfg: ReportModelConfig,
                 repo_root: str | Path = ".", tokenizer=None,
                 resample: bool = False, seed: int | None = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.max_len = cfg.realizer.max_text_len
        self.root = Path(repo_root)
        self.feat_dir = self.root / cfg.perception.cache_dir
        self.use_tf2 = cfg.perception.backbone == "hybrid"           # hybrid-dental-tf2
        self.tf2_dir = self.root / cfg.perception.tf2_teeth_cache_dir
        self.use_tf2_density = getattr(cfg.perception, "tf2_density", False)  # lo/hi/sat sub-pools
        self.anat_dir = self.root / cfg.train.anat_dir
        self.dense_dir = self.root / cfg.train.dense_dir
        self.resample = resample                          # re-draw the report per epoch
        self.seed = cfg.train.seed if seed is None else seed
        self.epoch = 0
        split = set(split_pids)

        self.samples: list[tuple[str, str]] = []          # (pid, anat_path)
        self.patients: list[tuple[str, list[str]]] = []   # (pid, [anat_paths])
        for pid, paths in _index_reports(self.anat_dir).items():
            if pid not in split or not self._feat_ok(pid):
                continue
            if self.use_tf2 and not self._tf2_ok(pid):
                continue
            cands = sorted(p for p in paths
                           if (self.dense_dir / Path(p).name).exists())
            if not cands:
                continue
            self.patients.append((pid, cands))
            for p in cands:
                self.samples.append((pid, p))
        self.patients.sort()
        self.samples.sort()

    def __len__(self):
        return len(self.patients)

    def set_epoch(self, epoch: int) -> None:
        """Train: re-pick every multi-report patient's report each epoch; val is a no-op."""
        self.epoch = int(epoch)

    def _resolve(self, i: int) -> tuple[str, str]:
        """index -> (pid, anat_path). The report is picked deterministically from
        (seed, epoch, pid): train (resample=True) advances epoch, val (False) pins epoch=0."""
        pid, cands = self.patients[i]
        if len(cands) == 1:
            return pid, cands[0]
        e = self.epoch if self.resample else 0
        h = hashlib.sha256(f"{self.seed}:{e}:{pid}".encode()).hexdigest()
        return pid, cands[int(h[:16], 16) % len(cands)]

    def region_type_counts(self) -> tuple[int, int]:
        """(n_normal, n_finding) — region-type counts for the EOS gate (false-EOS class weight).
        Loads no features (anat/dense json only) and follows exactly the target-building path of
        __getitem__."""
        n_normal = n_finding = 0
        for i in range(len(self)):
            _pid, ap = self._resolve(i)
            ap = Path(ap)
            anat = json.load(open(ap))
            dense = json.load(open(self.dense_dir / ap.name))
            state = build_axis_labels(dense)[:, 0]
            for txt in build_region_targets_eos(
                    anat, state, self.cfg.train.eos_include_not_available).values():
                if txt:
                    n_finding += 1
                else:
                    n_normal += 1
        return n_normal, n_finding

    def report_paths(self) -> list[tuple[str, list[str]]]:
        """(pid, [anat_path, ...]) — **all** reports of a patient."""
        return [(pid, list(paths)) for pid, paths in self.patients]

    def _targets_for(self, anat_paths: list[str]) -> dict[str, str]:
        """List of anat files -> region target dict.

        A single file takes **exactly the same path** as `__getitem__`; two or more are merged
        line-wise per region (`merge_anat_union`).
        """
        anats: list[dict] = []
        states: list[np.ndarray] = []
        for p in anat_paths:
            ap = Path(p)
            dense = json.load(open(self.dense_dir / ap.name))
            states.append(build_axis_labels(dense)[:, 0])
            anats.append(json.load(open(ap)))
        anat = merge_anat_union(anats)
        return build_region_targets_eos(anat, union_fdi_state(states),
                                        self.cfg.train.eos_include_not_available)

    def union_targets_by_pid(self) -> dict[str, dict[str, str]]:
        """pid -> region targets built from the **union of all reports** (reward scoring only).

        `__getitem__` and `target_ids` (the teacher-forcing ground truth) do not use this — the
        result is only the reward reference of a rollout and the finding/normal split of regions
        (`grpo.reward_gt=union`). It reads no features (anat/dense json only), so the whole split
        can be computed quickly on CPU.
        """
        return {pid: self._targets_for(paths) for pid, paths in self.report_paths()}

    def _feat_file(self, pid: str):
        return self.feat_dir / f"{pid}.npz"

    def _tf2_file(self, pid: str):
        return self.tf2_dir / f"{pid}.npz"

    def _feat_ok(self, pid: str) -> bool:
        return self._feat_file(pid).exists()

    def _tf2_ok(self, pid: str) -> bool:
        return self._tf2_file(pid).exists()

    def _load_regions(self, pid: str):
        z = np.load(self._feat_file(pid))
        bag = RegionBag.from_npz(z)

        def pack(c):
            return {"pooled": bag.pooled[c], "mass": bag.mass[c], "center": bag.center[c]}
        regions = {c: pack(c) for c in (1, 2, 5)}
        teeth = {"upper": pack(3), "lower": pack(4)}
        return regions, teeth

    def _load_tf2(self, pid: str):
        """tf2 per-tooth density bag -> 16 teeth per arch. Row i = FDI_ALL[i] (identity):
        upper = rows 0-15 (FDI_UP), lower = rows 16-31 (FDI_LO). Only whole_pool is loaded by
        default; with use_tf2_density the lo/hi/sat sub-pools (pooled, mass, center) are loaded
        too and injected in tooth_query as diagonal K/V tokens just like whole. A tooth tf2 did
        not detect gets present=False -> learnable absent embedding."""
        z = np.load(self._tf2_file(pid))
        assert z["fdi_all"].tolist() == list(FDI_ALL), \
            f"tf2 fdi_all mismatch for {pid}: {z['fdi_all'].tolist()}"
        pool = z["whole_pool"].astype(np.float32)                 # (32,64)
        mass = z["whole_mass"].astype(np.float32)                 # (32,)
        center = z["center"].astype(np.float32)                   # (32,3)
        present = z["present"].astype(bool)                       # (32,)
        dens = None
        if self.use_tf2_density:                                  # lo/hi/sat density sub-pools
            dens = {b: {"pooled": z[f"{b}_pool"].astype(np.float32),
                        "mass": z[f"{b}_mass"].astype(np.float32),
                        "center": z[f"{b}_center"].astype(np.float32)}
                    for b in ("lo", "hi", "sat")}

        def arch(sl):
            d = {"pooled": pool[sl], "mass": mass[sl], "center": center[sl],
                 "present": present[sl]}
            if dens is not None:
                for b in ("lo", "hi", "sat"):
                    d[f"{b}_pooled"] = dens[b]["pooled"][sl]
                    d[f"{b}_mass"] = dens[b]["mass"][sl]
                    d[f"{b}_center"] = dens[b]["center"][sl]
            return d
        return {"upper": arch(slice(0, 16)), "lower": arch(slice(16, 32))}

    def __getitem__(self, i):
        pid, anat_path = self._resolve(i)
        anat_path = Path(anat_path)
        dense = json.load(open(self.dense_dir / anat_path.name))
        fdi_labels = build_axis_labels(dense)                  # (32, 10) int64
        nonteeth = build_nonteeth_labels(dense)                # maxilla/mandible/nerve/arch
        item = {
            "pid": pid,
            "fdi_labels": torch.from_numpy(fdi_labels),
            "nonteeth_labels": {k: torch.from_numpy(v) for k, v in nonteeth.items()},
        }
        anat = json.load(open(anat_path))
        # Every imaged slot is fed; a normal one carries an empty target and is trained to emit
        # an immediate EOS, so nothing about which slots are abnormal reaches the input.
        targets = build_region_targets_eos(
            anat, fdi_labels[:, 0], self.cfg.train.eos_include_not_available)
        item["region_targets_text"] = targets
        if self.tok is not None:
            item["target_ids"] = {
                rid: (torch.empty(0, dtype=torch.long) if not txt   # '' -> EOS only (normal)
                      else self.tok(txt, truncation=True, max_length=self.max_len,
                                    add_special_tokens=False,  # chat assembly adds one EOS
                                    return_tensors="pt")["input_ids"][0])
                for rid, txt in targets.items()
            }
        regions, teeth = self._load_regions(pid)
        item["regions"] = regions
        item["teeth"] = teeth
        if self.use_tf2:                                        # hybrid-dental-tf2: per-tooth
            item["teeth_tf2"] = self._load_tf2(pid)
        return item


def _pad_bag(bags: list[dict], feat_dim: int):
    """list of {pooled(Pi,F),mass(Pi),center(Pi,3)} -> padded tensors + mask."""
    B = len(bags)
    P = max((b["pooled"].shape[0] for b in bags), default=0)
    P = max(P, 1)
    pooled = np.zeros((B, P, feat_dim), np.float32)
    mass = np.zeros((B, P), np.float32)
    center = np.zeros((B, P, 3), np.float32)
    mask = np.zeros((B, P), bool)
    for i, b in enumerate(bags):
        n = b["pooled"].shape[0]
        if n:
            pooled[i, :n] = b["pooled"]
            mass[i, :n] = b["mass"]
            center[i, :n] = b["center"]
            mask[i, :n] = True
    return {"pooled": torch.from_numpy(pooled), "mass": torch.from_numpy(mass),
            "center": torch.from_numpy(center), "mask": torch.from_numpy(mask)}


def make_collate(feat_dim: int, with_bags: bool = True):
    """Collate the cached bags and pass the (variable) per-region targets as one dict per sample.
    Decoding is independent per region, so the report model flattens and batches the region
    targets at (sample, region) granularity.

    with_bags=False leaves out the cached bags (regions/teeth/teeth_tf2) — in-loop perception
    builds the features from the volume instead (the ground-truth and target path is the same)."""
    def collate(items):
        out = {
            "fdi_labels": torch.stack([it["fdi_labels"] for it in items]),
            "nonteeth_labels": {
                k: torch.stack([it["nonteeth_labels"][k] for it in items])
                for k in ("maxilla", "mandible", "nerve")},
            "pid": [it["pid"] for it in items],
        }
        if with_bags:
            out["regions"] = {c: _pad_bag([it["regions"][c] for it in items], feat_dim)
                              for c in (1, 2, 5)}
            out["teeth"] = {arch: _pad_bag([it["teeth"][arch] for it in items], feat_dim)
                            for arch in ("upper", "lower")}
        if "report_text" in items[0]:                             # whole-report target
            out["report_text"] = [it["report_text"] for it in items]
            out["report_ids"] = [it.get("report_ids") for it in items]
        else:                                                     # per-region: (sample,region) target
            out["region_targets"] = [it.get("target_ids", {}) for it in items]
            out["region_targets_text"] = [it["region_targets_text"] for it in items]
        if items and "teeth_tf2" in items[0]:                     # hybrid-dental-tf2: 16 teeth/arch
            out["teeth_tf2"] = {                                   # stack all keys present
                arch: {
                    key: torch.from_numpy(
                        np.stack([it["teeth_tf2"][arch][key] for it in items]))
                    for key in items[0]["teeth_tf2"][arch]}
                for arch in ("upper", "lower")}
        return out
    return collate
