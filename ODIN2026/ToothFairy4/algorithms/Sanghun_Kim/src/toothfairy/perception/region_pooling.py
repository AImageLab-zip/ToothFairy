"""Anatomical region pooling — mask-weighted pooling of the
[encoder-stem; decoder-penultimate] feature by segmentation-class probability.

For each sliding-window patch p and class c:
    pooled_{p,c} = Σ_{x∈p} P[c,x]·[stem;penult][:,x] / Σ_{x∈p} P[c,x]      (64,)
    mass_{p,c}   = Σ_{x∈p} P[c,x]                                            (scalar)
The pooled vector divides by the patch-local mass only so it stays a unit-scale feature;
the deterministic, non-learned whole-volume normalisation Σ_p n / Σ_p m (mass-mean) is the
aggregation stage's job (encoder), not done here.

`pool_patch` is pure numpy/torch (no nnU-Net import) so it can be unit-tested on synthetic
logits/features. There is no mass threshold: every patch enters every class bag, so no
patch is silently dropped by a noise floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# class indices to pool (background 0 excluded); order is canonical for caching.
POOL_CLASSES = [1, 2, 3, 4, 5]


def pool_patch(logits: torch.Tensor, feat: torch.Tensor,
               classes: list[int]) -> dict[int, tuple[np.ndarray, float]]:
    """Mask-weighted pool one patch's [stem;penult] feature by class probability.

    logits (n_seg, *patch), feat (64, *patch) — BOTH must be fp32 here: the pooling
    numerator Σ_x P·feat reaches ~1e5–1e6 for a large class and overflows half's
    65504 max -> inf. Returns {c: (pooled(64,), mass)} for EVERY class — there is no
    mass threshold (no noise floor), so every patch is included.
    """
    P = torch.softmax(logits.float(), dim=0)             # (n_seg, *patch)
    c_feat = feat.shape[0]
    feat_flat = feat.float().reshape(c_feat, -1)         # (64, V)
    out: dict[int, tuple[np.ndarray, float]] = {}
    for c in classes:
        Pc = P[c].reshape(-1)                            # (V,)
        m = float(Pc.sum().item())
        pooled = (feat_flat @ Pc) / max(m, 1e-6)         # (64,) fp32
        out[c] = (pooled.detach().cpu().numpy().astype(np.float32), m)
    return out


@dataclass
class RegionBag:
    """Per-volume bag of per-patch tokens for every pooled segmentation class."""
    pooled: dict[int, np.ndarray]      # c -> (Nc, 64)  [stem;penult]
    mass: dict[int, np.ndarray]        # c -> (Nc,)
    center: dict[int, np.ndarray]      # c -> (Nc, 3) normalised [0,1] patch centre
    n_feat: int

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"n_feat": np.int64(self.n_feat)}
        for c in self.pooled:
            d[f"c{c}_pooled"] = self.pooled[c]
            d[f"c{c}_mass"] = self.mass[c]
            d[f"c{c}_center"] = self.center[c]
        return d

    @staticmethod
    def from_npz(z) -> "RegionBag":
        pooled, mass, center = {}, {}, {}
        for c in POOL_CLASSES:
            pooled[c] = z[f"c{c}_pooled"]
            mass[c] = z[f"c{c}_mass"]
            center[c] = z[f"c{c}_center"]
        return RegionBag(pooled, mass, center, int(z["n_feat"]))
