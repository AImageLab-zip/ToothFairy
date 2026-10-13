"""
Offline segmentation via the repo's own challenge predictor (task1_inference.py).

ORIENTATION (verified empirically, Aug 2026)
--------------------------------------------
The model was trained on R-first (RPI) array content. Every ToothFairy4 v2
volume is stored L-first (LAS 527, LPS 95), so the left/right array axis is
inverted relative to the training convention. Verified on three cases with
unambiguous single-implant ground truth in the reports:

    case   stored   report implant   flip 2 -> facts
    F030   LAS      25               25   correct   (no flip gives 15)
    P061   LAS      36               36   correct   (no flip gives 46)
    A046   LPS      35               35   correct   (no flip gives 45)

Axis 2 is the L/R axis (flipping axis 0 alone does NOT fix the mirror).
The earlier --auto-orient rule flipped axes (0,1) only and therefore could
never correct this, producing systematically L/R-mirrored masks.

Rule: flip array axis 2 when the volume is L-first; leave R-first volumes
alone. The mask is flipped back and returned in the ORIGINAL input space.

predict_semseg + BasePredictor already do TTA, LR-mirror, postprocessing, and
the FDI remap, so the returned mask is in the challenge FDI scheme.
"""

import gc
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from config import NNUNET_MODEL_FOLDER, NNUNET_CHECKPOINT, NNUNET_FOLDS, USE_MIRRORING
from task1_inference import BasePredictor, predict_semseg

# LR label-swap pairs, copied verbatim from run_inference.py / the trainer.
_LR_MAPPING = [(3, 4), (5, 6), (43, 44)]
_LR_MAPPING += [(l, r) for l, r in zip(range(19, 27), range(11, 19))]   # upper
_LR_MAPPING += [(l, r) for l, r in zip(range(27, 35), range(35, 43))]   # lower

# L/R axis, as a NEGATIVE index so a leading channel dim doesn't matter:
# for (C,Z,Y,X) and (Z,Y,X) alike, -1 is the last spatial axis = array axis 2.
_LR_FLIP = (-1,)

_PREDICTOR = None


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        p = BasePredictor(
            tile_step_size=0.95,
            use_mirroring=USE_MIRRORING,
            use_gaussian=True,
            perform_everything_on_device=False,
            allow_tqdm=False,
            tta_batch_size=1,
            lr_mapping=_LR_MAPPING,
            n_class=47,
            verbose=False,
        )
        p.initialize_from_trained_model_folder(
            str(NNUNET_MODEL_FOLDER),
            use_folds=NNUNET_FOLDS,
            checkpoint_name=NNUNET_CHECKPOINT,
        )
        _PREDICTOR = p
    return _PREDICTOR


def _axcodes(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def run_segmentation(cbct: sitk.Image) -> sitk.Image:
    """
    Predict on the raw input volume (no affine reorientation), correcting only
    the left/right array axis, and return an FDI-scheme mask in the ORIGINAL
    input space.
    """
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    predictor = _get_predictor()
    rw = SimpleITKIO()

    ax = _axcodes(cbct)
    # L-first volumes (LAS/LPS/...) are mirrored w.r.t. the R-first training
    # convention -> flip the L/R array axis. R-first volumes pass through.
    fax = _LR_FLIP if ax[:1].upper() == "L" else ()

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "CASE_0000.nii.gz"
        seg_path = Path(td) / "CASE.nii.gz"
        sitk.WriteImage(cbct, str(in_path), True)

        im, prop = rw.read_images([str(in_path)])          # (C, Z, Y, X)
        if fax:
            im = np.ascontiguousarray(np.flip(im, axis=fax))       # -> training L/R

        seg = predict_semseg(im, prop, predictor)
        if fax:
            seg = np.ascontiguousarray(np.flip(seg, axis=fax))     # -> back to input

        rw.write_seg(seg, str(seg_path), prop)
        mask = sitk.ReadImage(str(seg_path))

    print(f"=+= segmentation: axcodes={ax}  lr_flip={'yes' if fax else 'no'}")

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return mask
