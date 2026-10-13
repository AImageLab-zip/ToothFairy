"""Soft-clip CT normalization — only the upper hard clip is relaxed (keeps metal/endo signal).

Standard CTNormalization is clip(x, p0.5, p99.5) -> (x-mean)/std. The upper hard clip
(p99.5 = 3070) flattens metal and endodontic material (>3070, ~0.5% of all voxels) onto one
value and therefore **discards information at the input**, which is a root cause of the
restoration/endo axes not being picked up. This module replaces **only that upper clip** with a
soft clip:

    for x > upper:  x' = upper + s*tanh((x - upper)/s)

The lower clip (air), the mean/std z-score (the scale the encoder expects), LPS->RAS and the
sliding-window path that follows are all unchanged. A segmentation probe confirmed that the
change does not break the frozen segmentation network (far-Dice 0.90~0.99).
`install_softclip_ctnorm(s)` swaps the CTNormalization.run used by the nnU-Net preprocessing
process-wide; preprocessing always goes through that class method, so this single hook covers
the whole pipeline.
"""
from __future__ import annotations

import numpy as np

_ORIG_RUN = None


def _softclip_run_factory(s: float):
    def run(self, image, seg=None):
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean = self.intensityproperties["mean"]
        std = self.intensityproperties["std"]
        lower = self.intensityproperties["percentile_00_5"]
        upper = self.intensityproperties["percentile_99_5"]
        image = image.astype(np.float32, copy=True)             # fp32 soft-clip math (no overflow)
        np.clip(image, lower, None, out=image)                  # hard clip on the low side (air)
        if s > 0:
            hi = image > upper
            image[hi] = upper + s * np.tanh((image[hi] - upper) / s)  # keeps metal/endo apart
        else:                                                   # s<=0 = standard upper hard clip
            np.clip(image, None, upper, out=image)
        image -= mean
        image /= max(std, 1e-8)
        return image.astype(self.target_dtype, copy=False)
    return run


def install_softclip_ctnorm(s: float) -> None:
    """Replace CTNormalization.run with the soft-clip version (scale s), process-wide, idempotent.

    With s<=0 the upper side keeps the standard hard clip, but the installed run still carries
    the lower clip and the z-score are unchanged."""
    import nnunetv2.preprocessing.normalization.default_normalization_schemes as m
    global _ORIG_RUN
    if _ORIG_RUN is None:
        _ORIG_RUN = m.CTNormalization.run
    m.CTNormalization.run = _softclip_run_factory(s)


