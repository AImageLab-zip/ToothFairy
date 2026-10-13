"""
Orientation helpers.

`reorient` changes only the array/axis order + direction matrix; the physical
geometry (where each voxel sits in world space) is preserved. So reorienting a
CBCT and its mask to the same axcodes keeps them mutually aligned.
"""

import numpy as np
import SimpleITK as sitk


def reorient(image: sitk.Image, target_axcodes: str) -> sitk.Image:
    """Reorient to anatomical axcodes like 'RPI', 'RAS', 'LPS'."""
    return sitk.DICOMOrient(image, target_axcodes)


def current_axcodes(image: sitk.Image) -> str:
    """Axcodes of the image as it currently stands (from its direction)."""
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def maxilla_above_mandible(
    mask: sitk.Image,
    upper_labels: tuple[int, ...],
    lower_labels: tuple[int, ...],
    max_points: int = 3000,
) -> bool:
    """
    True if the maxillary teeth are physically superior to the mandibular
    teeth. Works in SimpleITK world space (LPS: +Z = superior), so it does not
    depend on array orientation and cannot be fooled by an L/R mirror-safe view.
    Returns True (i.e. does not block) if either label set is absent.
    """
    if not upper_labels or not lower_labels:
        return True

    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)

    def mean_world_superior(labels: tuple[int, ...]):
        zz, yy, xx = np.where(np.isin(arr, labels))
        if zz.size == 0:
            return None
        if zz.size > max_points:  # subsample for speed
            sel = np.linspace(0, zz.size - 1, max_points).astype(int)
            zz, yy, xx = zz[sel], yy[sel], xx[sel]
        zs = [
            mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))[2]
            for x, y, z in zip(xx, yy, zz)
        ]
        return float(np.mean(zs))

    up = mean_world_superior(upper_labels)
    lo = mean_world_superior(lower_labels)
    if up is None or lo is None:
        return True
    return up > lo
