"""Per-tooth density-split feature extraction from the frozen ToothFairy2 teeth segmentor.

Per CBCT volume, run the frozen ToothFairy2_Teeth segmentor once and, for every
individual tooth (tf2 label 1..32 = FDI, dataset.json order), **full per-label soft
mask-weighted pool** the tf2 [encoder-stem; decoder-penultimate] feature four ways
(weight = the tooth's own softmax prob P[L,x] from BEFORE argmax, exactly like dental
region_pooling — NOT a hard argmax mean):
  - whole   : P[L]-weighted mean over the whole tooth mask               (baseline whole-tooth)
  - lo      : P[L]-weighted mean over voxels with raw < k_lo * D  (caries / pulp, sub-dentin)
  - hi      : P[L]-weighted mean over voxels with k_hi * D < raw < sat_lo * vmax  (high-density,
              sub-ceiling: enamel + resolved metal on un-clipped scans)
  - sat     : P[L]-weighted mean over voxels with raw >= sat_lo * vmax   (ceiling pile = clipped-metal proxy)
mass = Σ_x P[L,x] over the (band) voxels (soft mass, consistent with dental region_pooling),
center = the centroid under the same weights (i.e. where inside the tooth that band sits).

**Anchor design**: the threshold anchor used to be the tooth's **own p50** (`> k_high·p50_self`).
In a fully crowned tooth the p50 is itself metal, so that relative cut collapses and `hi` came out
all-zero for **424 of 627 patients (67.6%)** — a self-reference defect. The anchor is therefore the
**per-patient dentin mode D** (the mode of the tooth-voxel histogram of that volume, robust).

  - **The intensity scale varies by a factor of 3.7 across scans** (dentin 963~3602, CV 0.24), so
    an absolute HU threshold is impossible -> **per-patient normalisation by D is required**.
    D is a single scalar per scan (no self-reference).
  - **The clipping regimes are separated**: in 65% of the sample enamel and metal saturate together
    at the ceiling vmax (max/dentin < 2.5), so metal cannot be separated by intensity. There metal
    lands in `sat` (the ceiling pile), while on scans where metal is resolved (max/dentin >= 3) it
    lands in `hi` (high density, below the ceiling). Keeping the two regimes in two separate pools
    lets the deep features disambiguate them by morphology. On scans where the metal intensity
    itself was clipped the value cannot be recovered at all (an honest limitation).
  - **Deep-feature pooling** (not raw statistics): endodontic material, crowns and restorations
    differ by location (a coronal cap vs. a linear root-canal filling) and an intensity histogram
    throws location away. The tf2 [E0;D0] features carry shape context through their receptive
    field, so pooling them per band can preserve that distinction.

Design decisions:
  - **full per-label soft mask-weighted pooling**: exactly as in dental region_pooling, pool with
    the **per-label softmax P[L,x] of the last tf2 layer, taken before the argmax** —
    pooled[L] = Σ_x P[L,x]·feat[x] / Σ_x P[L,x], mass[L] = Σ_x P[L,x]. A hard argmax mean smears
    the structure of boundary and low-probability voxels; here a boundary voxel **contributes to
    both neighbouring teeth in proportion to its probability**. Only voxels with
    P[L,x] > eps_soft (=0.02) are stored (boundaries included; low memory, single pass). `present`
    follows the argmax winner (as in the earlier baseline); a tooth only grazed by soft
    probability stays present=False (absent vector).
  - teeth feature source = tf2's OWN [E0;D0], so features / argmax
    labels / raw intensity are all derived from the SAME sliding-window patch → voxel
    aligned by construction.
  - density threshold is applied on the resampled-RAW intensity, NOT the
    CTNormalized network input — CTNormalization clips at p99.5, which would destroy
    the metal/endo signal (raw p90≈9000 vs dentin≈1600). We obtain resampled-raw by a
    second PreprocessAdapter pass with `normalization_schemes=['NoNormalization']`
    (same crop/resample geometry → voxel-identical grid to the feature pass).
  - **LPS→RAS flip** applied before inference (nnU-Net ignores affine / processes array
    order → LPS is L-R reversed vs RAS; earlier tf2 masks skipped this → 95 patients
    had L/R-swapped FDI labels). Flipping the input array unifies labels/feature/raw AND
    fixes the FDI↔label identity. See `prepare_tf2_volume`.
  - empty sub-pool (no metal in a normal tooth / no low voxels) → zero-vector + mass 0.
  - tf2 label L ↔ FDI_ALL[L-1] ↔ GT claim row (L-1): identity, no remap table.

The constants k_lo=0.6 / k_hi=1.5 / sat_lo=0.99 are defaults; they are exposed as arguments so the
thresholds can be retuned against the hi/sat coverage and the metal load recorded in the GT.

Row index in every (32, ·) array = tf2 label - 1 = FDI_ALL index = GT claim row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from .backbone import (build_predictor, encoder_stem_module,
                       decoder_penultimate_module)

# tf2 label L (1..32) -> FDI. dataset.json quadrant order Q1(UR)->Q2(UL)->Q3(LL)->Q4(LR),
# each quadrant central->3rd-molar. Identical to schema.claims.FDI_ALL.
FDI_UP = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LO = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
FDI_ALL = FDI_UP + FDI_LO
N_TEETH = 32
TF2_LABELS = list(range(1, N_TEETH + 1))

_LPS = ("L", "P", "S")  # volumes already in this order pass through unmodified (see below)


def prepare_tf2_volume(vol_path: str, tmpdir: Path):
    """Return (path, flipped): the volume path to run tf2 inference on. **Only RAS originals are
    L-R flipped**; LPS volumes pass through unmodified.

    WARNING: the purpose of this rule is **not anatomical correctness but agreement with the GT
    convention**. It is not "make tf2 output anatomically correct FDI numbers" but "make the FDI
    numbers of the tf2 output agree with the FDI numbers our GT uses". Training only needs the
    features and the labels to share one left/right convention, so the dataset convention is
    followed as it stands.

    Background: nnU-Net does not use the direction / origin of the NIfTI (see the SimpleITKIO
    comment "this part is NOT used by nnU-Net!") and feeds the array to the network in its stored
    order. Changing the handedness of the input array therefore flips the left/right of every
    output FDI number.

    Evidence — for each batch, the option that maximised agreement between the tf2 output and the
    GT was chosen:
      · RAS batch (F/P/S prefixes, 532 patients): the unmodified input disagrees with the GT
        (of 314 decidable patients, 244 mirrored vs 4 matching; accuracy 0.060 on informative
        slots vs 0.936 on mirror-invariant ones) -> **flip**.
      · LPS batch (A prefix, 95 patients): the flipped prediction disagrees with the GT for 18 of
        30 patients -> **leave unmodified**.

    WARNING: this measurement does **not** assume the GT is anatomically correct. On visual
    inspection a large part of the RAS-batch GT looks left-right mirrored (14 of 15 patients
    checked), but the decision was to follow the convention rather than edit the GT. So **do not
    read this code as evidence of which handedness tf2 was trained on** — that is a separate
    question, and reading it that way has flipped this condition twice in the past.

    WARNING: the dental segmentor (Dataset112) is a **separate path**. Do not sync this with
    pipeline/cache_dental.py::_prepare_ras_volume — the verdict differs per model / GT combination.
    """
    ax = nib.aff2axcodes(nib.load(vol_path).affine)
    if tuple(ax) == _LPS:
        return vol_path, False            # already LPS: pass through unmodified
    lr = next(i for i, c in enumerate(ax) if c in ("L", "R"))
    img = nib.load(vol_path)
    data = np.flip(np.asanyarray(img.dataobj), axis=lr).copy()
    fp = tmpdir / (Path(vol_path).parts[-3] + ".nii.gz")   # pid.nii.gz
    # The affine is left as in the original: nnU-Net does not look at the direction, so this is
    # harmless, and the output mask is restored in the geometry stored there. Note that the array
    # and the header of this temp file disagree, so any affine-aware tool reading it will be
    # misled — it is for tf2 inference only and must stay inside tmpdir.
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(fp))
    return str(fp), True


def build_tf2_predictor(model_dir: str, fold: int = 5,
                        checkpoint_name: str = "checkpoint_final.pth",
                        device: str = "cuda"):
    """Frozen ToothFairy2_Teeth predictor. tile_step_size=1.0 (non-overlapping tiles) so
    per-tooth pooling does not double-count overlap; a `visited` buffer still guards the
    boundary-clamped last tiles."""
    return build_predictor(model_dir, fold=fold, checkpoint_name=checkpoint_name,
                           tile_step_size=1.0, device=device)


@dataclass
class ToothDensityBag:
    """Per-volume per-tooth density-split features. Row i (0..31) = tf2 label i+1 = FDI_ALL[i].
    The anchor is the per-patient dentin mode `dentin` (a scalar); see the module docstring for
    the definition of the lo / hi / sat bands."""
    whole_pool: np.ndarray   # (32, 64) P[L]-weighted mean [stem;penult] over tooth mask
    whole_mass: np.ndarray   # (32,)   soft mass = Σ_x P[L,x] over tooth voxels
    lo_pool: np.ndarray      # (32, 64) P[L]-weighted mean over raw < k_lo*dentin (caries/pulp; zeros if empty)
    lo_mass: np.ndarray      # (32,)   soft mass Σ P[L] in lo band
    hi_pool: np.ndarray      # (32, 64) P[L]-weighted mean over k_hi*dentin < raw < sat_lo*vmax (zeros if empty)
    hi_mass: np.ndarray      # (32,)   soft mass Σ P[L] in hi band (high-density, sub-ceiling)
    sat_pool: np.ndarray     # (32, 64) P[L]-weighted mean over raw >= sat_lo*vmax (ceiling pile; zeros if empty)
    sat_mass: np.ndarray     # (32,)   soft mass Σ P[L] at ceiling (clipped-metal proxy)
    center: np.ndarray       # (32, 3) whole-tooth soft centroid, normalised [0,1] (z,y,x)
    lo_center: np.ndarray    # (32, 3) lo-band soft centroid (location separates crown from endo)
    hi_center: np.ndarray    # (32, 3) hi-band soft centroid
    sat_center: np.ndarray   # (32, 3) sat-band soft centroid
    p50: np.ndarray          # (32,)   per-tooth resampled-raw median (diagnostic; not the anchor)
    present: np.ndarray      # (32,)   bool: tooth mask non-empty
    dentin: float            # per-patient dentin mode (anchor; scan-specific, fixes self-reference)
    vmax: float              # per-patient max tooth-voxel raw intensity (ceiling)
    n_feat: int
    k_hi: float
    k_lo: float
    sat_lo: float

    def to_npz_dict(self) -> dict:
        return {
            "whole_pool": self.whole_pool, "whole_mass": self.whole_mass,
            "lo_pool": self.lo_pool, "lo_mass": self.lo_mass,
            "hi_pool": self.hi_pool, "hi_mass": self.hi_mass,
            "sat_pool": self.sat_pool, "sat_mass": self.sat_mass,
            "center": self.center, "lo_center": self.lo_center,
            "hi_center": self.hi_center, "sat_center": self.sat_center,
            "p50": self.p50, "present": self.present,
            "dentin": np.float32(self.dentin), "vmax": np.float32(self.vmax),
            "n_feat": np.int64(self.n_feat),
            "k_hi": np.float32(self.k_hi), "k_lo": np.float32(self.k_lo),
            "sat_lo": np.float32(self.sat_lo),
            "fdi_all": np.asarray(FDI_ALL, dtype=np.int64),
        }

    @staticmethod
    def from_npz(z) -> "ToothDensityBag":
        return ToothDensityBag(
            whole_pool=z["whole_pool"], whole_mass=z["whole_mass"],
            lo_pool=z["lo_pool"], lo_mass=z["lo_mass"],
            hi_pool=z["hi_pool"], hi_mass=z["hi_mass"],
            sat_pool=z["sat_pool"], sat_mass=z["sat_mass"],
            center=z["center"], lo_center=z["lo_center"],
            hi_center=z["hi_center"], sat_center=z["sat_center"],
            p50=z["p50"], present=z["present"],
            dentin=float(z["dentin"]), vmax=float(z["vmax"]),
            n_feat=int(z["n_feat"]), k_hi=float(z["k_hi"]), k_lo=float(z["k_lo"]),
            sat_lo=float(z["sat_lo"]))


def _preprocess(pred, nii_path: str, normalize: bool):
    """One PreprocessAdapter pass, reading the volume fresh (independent of the other pass —
    the adapter mutates props with crop bbox keys). normalize=False toggles NoNormalization
    to recover the resampled-RAW intensity on the SAME crop/resample grid as the feature
    pass (B3). Both passes share crop bbox / new_shape / resampling_fn → voxel-identical."""
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    image, props = SimpleITKIO().read_images([nii_path])
    cm = pred.configuration_manager
    saved = list(cm.configuration["normalization_schemes"])
    if not normalize:
        cm.configuration["normalization_schemes"] = ["NoNormalization"] * len(saved)
    try:
        ppa = PreprocessAdapterFromNpy([image], [None], [props], [None],
                                       pred.plans_manager, pred.dataset_json, cm,
                                       num_threads_in_multithreaded=1, verbose=False)
        data = next(ppa)["data"]
    finally:
        cm.configuration["normalization_schemes"] = saved
    return data


def _dentin_mode(raw_all: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Per-patient dentin anchor = robust mode of the pooled tooth-voxel raw histogram.

    Restricts to the [p5, p60] window so the saturation spike (clipped enamel+metal piled
    at the ceiling) cannot be mistaken for dentin: on clipped scans a global mode lands on that
    ceiling spike. numpy only (no scipy dependency).
    """
    if raw_all.size == 0:
        return 0.0
    lo, hi = np.percentile(raw_all, [5, 60])
    if not (hi > lo):
        return float(np.median(raw_all))
    bins = np.linspace(lo, hi, 200)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    cnt, _ = np.histogram(raw_all, bins=bins, weights=weights)   # weights = per-label softmax P[L] (soft)
    k = np.ones(7, dtype=float) / 7.0                 # moving-average smoothing
    sm = np.convolve(cnt.astype(float), k, mode="same")
    return float(ctr[int(np.argmax(sm))])


@torch.inference_mode()
def extract_tooth_density_bag(pred, nii_path: str,
                              k_hi: float = 1.5, k_lo: float = 0.6,
                              sat_lo: float = 0.99, eps_soft: float = 0.02,
                              pooling: str = "soft",
                              low_mem: bool = False) -> ToothDensityBag:
    # low_mem=True: pool the per-patch feat/probs on the CPU (only the net and the logits stay on
    #   the GPU) -> lower VRAM peak, independent of the volume size. The arithmetic is the same
    #   fp32 computation, so results match the GPU path to floating-point error. Meant for the
    #   inference on a small card (16GB); off by default and without effect on training.
    """Single frozen sliding-window pass → per-tooth {whole, lo, hi, sat} density-split pools.

    pooling="soft" (default): full per-label soft pooling — every voxel contributes to tooth t_
      with weight P[t_,x] (the softmax taken before the argmax).
    pooling="argmax": the earlier scheme — every voxel is assigned to its argmax tooth with weight
      1.0 (plain mean, voxel-count mass, plain centroid). Kept to reproduce models that were
      trained on argmax-pooled features: in that mode whole / mass / center / present match the
      earlier version of this extractor. The bands (lo/hi/sat) still use the dentin anchor and so
      differ from the earlier p50 bands, which is irrelevant because argmax models only use
      `whole`.

    The anchor is the per-patient dentin mode D (the mode of the tooth-voxel histogram of the whole
    volume), NOT the per-tooth p50 (which was the self-reference defect). Thresholds:
    lo = raw < k_lo·D / hi = k_hi·D < raw < sat_lo·vmax / sat = raw >= sat_lo·vmax, where vmax is
    the maximum over the confident tooth voxels of that volume, i.e. the per-scan ceiling.

    `nii_path` must already be LPS-oriented (call prepare_tf2_volume first). Features,
    per-label softmax and resampled-raw all come from the same padded plans-space grid.
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    net = pred.network
    if hasattr(net, "decoder"):
        net.decoder.deep_supervision = False
    dev = pred.device
    patch_size = pred.configuration_manager.patch_size

    # feature pass (CTNormalized input the network was trained on)
    data_norm = _preprocess(pred, nii_path, normalize=True)
    data_norm, _ = pad_nd_image(data_norm, patch_size, "constant", {"value": 0}, True, None)
    # raw pass (identical geometry, un-normalised → density thresholds live here, B3)
    data_raw = _preprocess(pred, nii_path, normalize=False)
    data_raw, _ = pad_nd_image(data_raw, patch_size, "constant", {"value": 0}, True, None)
    assert data_norm.shape == data_raw.shape, (data_norm.shape, data_raw.shape)

    spatial = np.asarray(data_norm.shape[1:], dtype=np.float64)
    slicers = pred._internal_get_sliding_window_slicers(data_norm.shape[1:])
    data_norm = data_norm.to(dev)
    data_raw = data_raw.to(dev)

    captured: dict[str, torch.Tensor] = {}
    h1 = encoder_stem_module(net).register_forward_hook(
        lambda m, i, o: captured.__setitem__("stem", o))
    h2 = decoder_penultimate_module(net).register_forward_pre_hook(
        lambda m, i: captured.__setitem__("penult", i[0]))

    visited = torch.zeros(tuple(int(s) for s in data_norm.shape[1:]), dtype=torch.bool, device=dev)
    # per-tooth accumulators (CPU, small: only tooth voxels)
    raw_acc: dict[int, list] = {t: [] for t in TF2_LABELS}
    feat_acc: dict[int, list] = {t: [] for t in TF2_LABELS}   # each (64, n) fp16
    w_acc: dict[int, list] = {t: [] for t in TF2_LABELS}      # each (n,) per-label softmax P[t_] = pooling weight
    coord_acc: dict[int, list] = {t: [] for t in TF2_LABELS}  # each (n,3) global coord, for centroids
    argmax_seen = np.zeros(N_TEETH, dtype=bool)              # argmax winner somewhere -> present
    c_feat = None
    use_amp = dev.type == "cuda"
    pool_dev = torch.device("cpu") if low_mem else dev   # low_mem: pool the heavy feat/probs on CPU
    try:
        for sl in slicers:
            sp = sl[1:]                                  # spatial slices
            patch = data_norm[sl][None]                  # (1, 1, *patch)
            with torch.autocast(dev.type, enabled=use_amp):
                logits = net(patch)[0]                   # (n_seg, *patch)
                stem = captured["stem"][0]
                penult = captured["penult"][0]
            feat = torch.cat([stem, penult], dim=0).float().to(pool_dev)   # (64,*patch) fp32 (low_mem:CPU)
            if c_feat is None:
                c_feat = feat.shape[0]
            probs = torch.softmax(logits.float(), dim=0).to(pool_dev)  # (n_seg,*patch) pre-argmax
            raw = data_raw[(0, *sp)].to(pool_dev)        # (*patch,)
            new = (~visited[sp]).to(pool_dev)            # exactly-once dedup (overlap-safe)
            feat_flat = feat.reshape(c_feat, -1)
            base = torch.tensor([sp[a].start for a in range(3)], device=pool_dev)
            am = probs.argmax(0)                          # (*patch,) hard label
            am_new = am[new]
            for _t in torch.unique(am_new).tolist():      # argmax-present = genuine detection
                if 1 <= _t <= N_TEETH:
                    argmax_seen[_t - 1] = True
            # per-tooth (mask, weight), depending on the pooling mode.
            #   soft (default): a voxel contributes to tooth t_ with weight P[t_,x] (the pre-argmax
            #     softmax); only voxels with P > eps are stored.
            #   argmax: the earlier scheme — assigned to the argmax tooth with weight 1.0
            #     (plain mean, voxel-count mass, plain centroid).
            entries = []                                  # list of (t_, m_bool(*patch), wv_tensor(n,))
            if pooling == "argmax":
                for _t in torch.unique(am_new).tolist():
                    if 1 <= _t <= N_TEETH:
                        m = (am == _t) & new
                        entries.append((_t, m, torch.ones(int(m.sum().item()), device=pool_dev)))
            else:
                teeth_p = probs[1:N_TEETH + 1]            # (32, *patch) tooth L at idx L-1
                tp_mask = (teeth_p > eps_soft) & new[None]  # (32, *patch) bool
                for li in tp_mask.reshape(N_TEETH, -1).any(dim=1).nonzero().flatten().tolist():
                    entries.append((li + 1, tp_mask[li], teeth_p[li][tp_mask[li]]))
            for t_, m, wv in entries:
                mf = m.reshape(-1)
                raw_acc[t_].append(raw[m].detach().cpu().numpy().astype(np.float32))
                feat_acc[t_].append(feat_flat[:, mf].detach().to(torch.float16).cpu().numpy())
                w_acc[t_].append(wv.detach().cpu().numpy().astype(np.float32))
                idx = torch.nonzero(m, as_tuple=False)    # (n, 3) local coords
                coord_acc[t_].append((idx + base).to(torch.int16).cpu().numpy())  # (n,3) global → centroid
            visited[sp] = True
    finally:
        h1.remove()
        h2.remove()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cf = c_feat or 64
    # per-patient anchor: pool ALL tooth raw voxels (raw only, so memory stays low) -> dentin
    # mode D and ceiling vmax. Every pool below is full per-label soft (weight = that tooth's
    # softmax P[L,x]), never a plain argmax mean.
    raw_by_tooth = {t_: np.concatenate(raw_acc[t_]) for t_ in TF2_LABELS if raw_acc[t_]}
    w_by_tooth = {t_: np.concatenate(w_acc[t_]) for t_ in TF2_LABELS if w_acc[t_]}
    coord_by_tooth = {t_: np.concatenate(coord_acc[t_]).astype(np.float32) for t_ in TF2_LABELS if coord_acc[t_]}
    if raw_by_tooth:
        all_raw = np.concatenate(list(raw_by_tooth.values()))
        all_w = np.concatenate(list(w_by_tooth.values()))
        dentin = _dentin_mode(all_raw, all_w)                      # weighted dentin mode
        conf_vox = all_raw[all_w > 0.5]                            # ceiling: confident voxels
        vmax = float(conf_vox.max()) if conf_vox.size else float(all_raw.max())
        del all_raw, all_w
    else:
        dentin = 0.0
        vmax = 0.0
    lo_thr = k_lo * dentin
    hi_thr = k_hi * dentin
    sat_thr = sat_lo * vmax

    whole_pool = np.zeros((N_TEETH, cf), np.float32); whole_mass = np.zeros(N_TEETH, np.float32)
    lo_pool = np.zeros((N_TEETH, cf), np.float32);    lo_mass = np.zeros(N_TEETH, np.float32)
    hi_pool = np.zeros((N_TEETH, cf), np.float32);    hi_mass = np.zeros(N_TEETH, np.float32)
    sat_pool = np.zeros((N_TEETH, cf), np.float32);   sat_mass = np.zeros(N_TEETH, np.float32)
    p50 = np.zeros(N_TEETH, np.float32);              present = np.zeros(N_TEETH, bool)
    center = np.zeros((N_TEETH, 3), np.float32)                   # whole-tooth soft centroid
    lo_center = np.zeros((N_TEETH, 3), np.float32)               # per-band soft centroid
    hi_center = np.zeros((N_TEETH, 3), np.float32)
    sat_center = np.zeros((N_TEETH, 3), np.float32)

    def _wpool(feat, w):                                          # soft-weighted mean + soft mass(Σw)
        s = float(w.sum())
        if s <= 0:
            return np.zeros(cf, np.float32), 0.0
        return ((feat * w).sum(axis=1) / s).astype(np.float32), s

    def _wcenter(coords, w):                                      # soft-weighted centroid, normalised [0,1]
        s = float(w.sum())
        if s <= 0:
            return np.zeros(3, np.float32)
        return (((coords * w[:, None]).sum(axis=0) / s) / spatial).astype(np.float32)

    for t_ in TF2_LABELS:
        i = t_ - 1
        if t_ not in raw_by_tooth or not argmax_seen[i]:           # soft spill-over only -> absent
            continue
        raw_all = raw_by_tooth[t_]                                  # (N,)
        w = w_by_tooth[t_]                                          # (N,) soft weights = P(tooth t_)
        coords = coord_by_tooth[t_]                                 # (N,3) global voxel coords
        feat_all = np.concatenate(feat_acc[t_], axis=1).astype(np.float32)  # (64, N)
        present[i] = True                                          # detection = argmax winner
        whole_pool[i], whole_mass[i] = _wpool(feat_all, w)         # soft mass = Σ P(tooth)
        center[i] = _wcenter(coords, w)                            # whole-tooth soft centroid
        p50[i] = float(np.median(raw_all))                         # diagnostic only (not the anchor)
        if dentin > 0:                                             # anchor needs positive dentin
            sat = raw_all >= sat_thr                               # ceiling pile (clipped-metal proxy)
            lo = raw_all < lo_thr                                  # caries/pulp (sub-dentin)
            hi = (raw_all > hi_thr) & (~sat)                       # high-density, sub-ceiling
            lo_pool[i], lo_mass[i] = _wpool(feat_all[:, lo], w[lo])
            hi_pool[i], hi_mass[i] = _wpool(feat_all[:, hi], w[hi])
            sat_pool[i], sat_mass[i] = _wpool(feat_all[:, sat], w[sat])
            lo_center[i] = _wcenter(coords[lo], w[lo])            # band location: crown vs root
            hi_center[i] = _wcenter(coords[hi], w[hi])
            sat_center[i] = _wcenter(coords[sat], w[sat])

    return ToothDensityBag(whole_pool, whole_mass, lo_pool, lo_mass, hi_pool, hi_mass,
                           sat_pool, sat_mass, center, lo_center, hi_center, sat_center,
                           p50, present, float(dentin), float(vmax), cf,
                           float(k_hi), float(k_lo), float(sat_lo))
