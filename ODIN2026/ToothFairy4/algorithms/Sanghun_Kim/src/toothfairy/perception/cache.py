"""Offline region-feature extraction — one frozen sliding-window pass per volume.

For every CBCT volume, run the frozen segmentation network once and pool the per-patch
[encoder-stem; decoder-penultimate] feature for all segmentation classes into a RegionBag
(the offline feature cache the encoder consumes). The expensive inference runs exactly once
per volume and is then reused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .backbone import encoder_stem_module, decoder_penultimate_module
from .region_pooling import pool_patch, RegionBag, POOL_CLASSES


def check_cache_provenance(out_dir, params: dict) -> None:
    """Abort when an existing **completed** cache was built with parameters that differ from
    the current run (guards against reusing a stale cache under a false provenance).

    `_meta.json` is written only when caching finished normally (an interrupted partial cache
    has none), so this guard does not get in the way of a legitimate resume with identical
    parameters; it only fails loudly when a directory already filled under *different*
    parameters (normalisation, pooling, ...) is about to be re-cached — a stale-cache trap
    that has bitten this pipeline before. `_meta.json` must also record `cache_params` (the
    dict of fields to compare) for this check to mean anything."""
    meta_f = Path(out_dir) / "_meta.json"
    if not meta_f.exists():
        return                                         # fresh dir (or interrupted partial cache)
    try:
        prev = json.loads(meta_f.read_text()).get("cache_params", {})
    except Exception:                                  # noqa: BLE001 — unreadable meta: let it pass
        return
    mism = {k: {"prev": prev.get(k), "now": v} for k, v in params.items()
            if k in prev and prev.get(k) != v}
    if mism:
        raise SystemExit(
            f"[abort] cache parameters in {meta_f} differ from the current run: {mism}. "
            "Stopping so that a stale cache is not reused — pass a different --out-dir, or "
            "empty that directory and re-run.")


@torch.inference_mode()
def extract_region_bag(pred, nii_path: str) -> RegionBag:
    """Run the frozen segmentation network once and pool [stem;penult] per patch for all classes.

    `pred` is an nnUNetPredictor from `toothfairy.perception.backbone.build_predictor`.
    Every patch enters every class bag (there is no mass threshold).
    """
    # heavy nnU-Net imports kept local so RegionBag/pool_patch import without them.
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    net = pred.network
    if hasattr(net, "decoder"):
        net.decoder.deep_supervision = False

    image, props = SimpleITKIO().read_images([nii_path])
    ppa = PreprocessAdapterFromNpy([image], [None], [props], [None],
                                   pred.plans_manager, pred.dataset_json,
                                   pred.configuration_manager,
                                   num_threads_in_multithreaded=1, verbose=False)
    data = next(ppa)["data"]
    data, _revert = pad_nd_image(data, pred.configuration_manager.patch_size,
                                 "constant", {"value": 0}, True, None)
    spatial = np.asarray(data.shape[1:], dtype=np.float64)
    slicers = pred._internal_get_sliding_window_slicers(data.shape[1:])
    data = data.to(pred.device)

    captured: dict[str, torch.Tensor] = {}
    h1 = encoder_stem_module(net).register_forward_hook(
        lambda m, i, o: captured.__setitem__("stem", o))
    h2 = decoder_penultimate_module(net).register_forward_pre_hook(
        lambda m, i: captured.__setitem__("penult", i[0]))

    bags = {c: {"pooled": [], "mass": [], "center": []} for c in POOL_CLASSES}
    c_feat = None
    use_amp = pred.device.type == "cuda"
    try:
        for sl in slicers:
            patch = data[sl][None]
            with torch.autocast(pred.device.type, enabled=use_amp):
                logits = net(patch)[0]                            # (n_seg, *patch)
                stem = captured["stem"][0]
                penult = captured["penult"][0]
            feat = torch.cat([stem, penult], dim=0).float()       # (64, *patch) fp32
            if c_feat is None:
                c_feat = feat.shape[0]
            sp = sl[1:]
            center = np.array(
                [((s.start + s.stop) / 2.0) / spatial[ax] for ax, s in enumerate(sp)],
                dtype=np.float32)
            per_c = pool_patch(logits, feat, POOL_CLASSES)
            for c in POOL_CLASSES:
                pooled, m = per_c[c]
                bags[c]["pooled"].append(pooled)
                bags[c]["mass"].append(np.float32(m))
                bags[c]["center"].append(center)
    finally:
        h1.remove()
        h2.remove()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cf = c_feat or 64

    def stack(c: int, key: str, dim: int):
        v = bags[c][key]
        if not v:
            return np.zeros((0, dim) if dim else (0,), dtype=np.float32)
        return np.stack(v) if dim else np.asarray(v, dtype=np.float32)

    return RegionBag(
        pooled={c: stack(c, "pooled", cf) for c in POOL_CLASSES},
        mass={c: stack(c, "mass", 0) for c in POOL_CLASSES},
        center={c: stack(c, "center", 3) for c in POOL_CLASSES},
        n_feat=cf)
