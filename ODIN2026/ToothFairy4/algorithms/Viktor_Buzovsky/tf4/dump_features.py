"""Cache raw per-case geometry from the native TF3 labels, once.

Thresholds (tooth voxel counts, canal lingual/buccal cut-off) then get swept
instantly against the report-parsed references instead of costing a full mask
reload per attempt.
"""

from __future__ import annotations

import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf4.calibrate import TF4_ROOT, tf3_label_path  # noqa: E402
from tf4.mask_features import (  # noqa: E402
    LAB_BRIDGE, LAB_CANAL_L, LAB_CANAL_R, LAB_CROWN, LAB_IMPLANT,
    LAB_LOWER_JAW, LAB_UPPER_JAW, infer_axes, local_bone_segment,
)
from tf4.schema import ALL_FDI  # noqa: E402
from tf4.trinity_infer import to_training_frame  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "features.json")


def canal_rel_positions(mask: np.ndarray, axes: dict) -> list[float]:
    """SIGNED position of the canal within the mandible cross-section.

    Negative = medial (lingual), positive = lateral (buccal). An unsigned
    distance from the jaw centre cannot tell the two apart -- the canal hugs a
    cortex either way -- which is why the first version returned ~0.82 for every
    case and scored 0% agreement.
    """
    # canals are holes in the jawbone label; fill them so the run stays solid
    jaw = (mask == LAB_LOWER_JAW) | (mask == LAB_CANAL_L) | (mask == LAB_CANAL_R)
    lr = axes["left"][0]
    other = [a for a in (0, 1, 2) if a != lr]
    jaw_midline = float(np.argwhere(jaw)[:, lr].mean())

    rel: list[float] = []
    for label in (LAB_CANAL_L, LAB_CANAL_R):
        idx = np.argwhere(mask == label)
        if len(idx) < 200:
            continue
        # which side of the midline this canal lies on
        side = 1.0 if float(idx[:, lr].mean()) > jaw_midline else -1.0
        for v in idx[:: max(1, len(idx) // 400)]:
            sl: list[object] = [0, 0, 0]
            sl[lr] = slice(None)
            sl[other[0]] = int(v[other[0]])
            sl[other[1]] = int(v[other[1]])
            x = int(v[lr])
            seg = local_bone_segment(np.flatnonzero(jaw[tuple(sl)]), x)
            if seg is None:
                continue
            lo, hi = seg
            half = (hi - lo) / 2.0
            if half < 1.5:
                continue
            rel.append(side * (x - (hi + lo) / 2.0) / half)
    return rel


def main() -> None:
    out = {}
    for case in sorted(os.listdir(TF4_ROOT)):
        lp = tf3_label_path(case)
        if lp is None:
            continue
        mask = to_training_frame(np.asarray(nib.load(lp).dataobj).astype(np.uint8))
        axes = infer_axes(mask)
        rec: dict[str, object] = {
            "teeth": {str(t): int((mask == t).sum()) for t in ALL_FDI},
            "lower_jaw": int((mask == LAB_LOWER_JAW).sum()),
            "upper_jaw": int((mask == LAB_UPPER_JAW).sum()),
            "bridge": int((mask == LAB_BRIDGE).sum()),
            "implant": int((mask == LAB_IMPLANT).sum()),
            "crown": int((mask == LAB_CROWN).sum()),
            "axes": {k: [int(v[0]), int(v[1])] for k, v in axes.items()} if axes else None,
        }
        if axes:
            rel = canal_rel_positions(mask, axes)
            rec["canal_rel_mean"] = float(np.mean(rel)) if len(rel) >= 50 else None
        else:
            rec["canal_rel_mean"] = None
        out[case] = rec
        print(f"{case} teeth={sum(1 for v in rec['teeth'].values() if v>300)} "
              f"canal_rel={rec['canal_rel_mean']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {len(out)} cases -> {OUT}")


if __name__ == "__main__":
    main()
