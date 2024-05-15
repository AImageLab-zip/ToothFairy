#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import numpy
import nibabel

import numpy as np
import SimpleITK as sitk


__all__ = ['reorient_image_to_RAS']

# Global parameters
POSSIBLE_AXES_ORIENTATIONS = [
    "LAI", "LIA", "ALI", "AIL", "ILA", "IAL",
    "LAS", "LSA", "ALS", "ASL", "SLA", "SAL",
    "LPI", "LIP", "PLI", "PIL", "ILP", "IPL",
    "LPS", "LSP", "PLS", "PSL", "SLP", "SPL",
    "RAI", "RIA", "ARI", "AIR", "IRA", "IAR",
    "RAS", "RSA", "ARS", "ASR", "SRA", "SAR",
    "RPI", "RIP", "PRI", "PIR", "IRP", "IPR",
    "RPS", "RSP", "PRS", "PSR", "SRP", "SPR"
]
CORRECTION_MATRIX_COLUMNS = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "A": (0, 1, 0),
    "P": (0, -1, 0),
    "S": (0, 0, 1),
    "I": (0, 0, -1)
}


def swap_affine(axes):
    """ Build a correction matrix, from the given orientation of axes to RAS.
    Parameters
    ----------
    axes: str (manadtory)
        the given orientation of the axes.
    Returns
    -------
    rotation: array (4, 4)
        the correction matrix.
    """
    rotation = numpy.eye(4)
    rotation[:3, 0] = CORRECTION_MATRIX_COLUMNS[axes[0]]
    rotation[:3, 1] = CORRECTION_MATRIX_COLUMNS[axes[1]]
    rotation[:3, 2] = CORRECTION_MATRIX_COLUMNS[axes[2]]
    return rotation


def reorient_image(in_file, axes="RAS", prefix="swap", output_directory=None):
    """ Rectify the orientation of an image.
    Parameters
    ----------
    in_file: str (mandatory)
        the input image.
    axes: str (optional, default 'RAS')
        orientation of the original axes X, Y, and Z
        specified with the following convention: L=Left, R=Right,
        A=Anterion, P=Posterior, I=Inferior, S=Superior.
    prefix: str (optional, default 'swap')
        prefix of the output image.
    output_directory: str (optional, default None)
        the output directory where the rectified image is saved.
        If None use the same directory as the input image.
    Returns
    -------
    out_file: str
        the rectified image.
    Examples
    --------
     from pclinfmri.utils.reorientation import reorient_image
     rectified_image = reorient_image('image.nii', 'RAS', 's', None)
    <process>
        <return name="out_file" type="File" desc="the rectified image."/>
        <input name="in_file" type="File" desc="the input image."/>
        <input name="axes" type="String" desc="orientation of the original
            axes X, Y, and Z specified with the following convention:
            L=Left, R=Right, A=Anterion, P=Posterior, I=Inferior, S=Superior."/>
        <input name="prefix" type="String" desc="the prefix of the output
            image."/>
        <input name="output_directory" type="Directory" desc="the output
            directory where the rectified image is saved."/>
    </process>
    """
    # Check the input image exists on the file system
    if not os.path.isfile(in_file):
        raise ValueError("'{0}' is not a valid filename.".format(in_file))

    # Check that the outdir is valid
    if output_directory is not None:
        if not os.path.isdir(output_directory):
            raise ValueError("'{0}' is not a valid directory.".format(
                output_directory))
    else:
        output_directory = os.path.dirname(in_file)

    # Check that a valid input axes is specified
    if axes not in POSSIBLE_AXES_ORIENTATIONS:
        raise ValueError("Wrong coordinate system: {0}.".format(axes))

    # Get the transformation to the RAS space
    rotation = swap_affine(axes)
    det = numpy.linalg.det(rotation)
    if det != 1:
        raise Exception("Rotation matrix determinant must be one "
                        "not '{0}'.".format(det))

    # Load the image to rectify
    image = nibabel.load(in_file)

    # Get the input image affine transform
    affine = image.get_affine()

    # Apply the rotation to set the image in the RAS coordiante system
    transformation = numpy.dot(rotation, affine)
    image.set_qform(transformation)
    image.set_sform(transformation)

    # Save the rectified image
    fsplit = os.path.split(in_file)
    out_file = os.path.join(output_directory, prefix + fsplit[1])
    nibabel.save(image, out_file)

    return out_file


def reorient_image_to_RAS(image, interpolate_mode='nearest'):
    """https://github.com/jfm15/SpineFinder/blob/master/utility_functions/processing.py
    Reorients an image to standard radiology view.
    """
    direction = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(direction), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_direction = direction[:, ind]

    flip = np.diag(new_direction) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_direction, (new_extent * flip))
    new_direction = np.matmul(new_direction, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_direction.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    # resample.SetDefaultPixelValue(image.GetPixelIDValue())
    if interpolate_mode == 'bspline':
        resample.SetInterpolator(sitk.sitkBSpline)
    elif interpolate_mode == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)