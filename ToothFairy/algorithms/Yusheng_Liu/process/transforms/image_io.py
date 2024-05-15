
import os
from typing import Optional

import numpy as np
import SimpleITK as sitk
from pydicom import dicomio


__all__ = ['load_sitk_image', 'save_sitk_from_npy', 'save_sitk_image', 'dcm_2_mha']


def load_sitk_image(file_path: str, sort_by_distance: bool = True) -> dict:
    try:
        if os.path.isdir(file_path):
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(file_path)
            dcm_series = reader.GetGDCMSeriesFileNames(file_path, series_ids[0])
            reader.SetFileNames(dcm_series)
            sitk_image = _load_ct_from_dicom(file_path, sort_by_distance)
        else:
            if os.path.exists(file_path):
                sitk_image = sitk.ReadImage(file_path)
            else:
                raise ValueError(f'{file_path} is not exist!')
    except Exception as err:
        raise TypeError(f'load ct throws exception {err}, with file {file_path}!')

    origin = list(reversed(sitk_image.GetOrigin()))
    spacing = list(reversed(sitk_image.GetSpacing()))
    direction = list(sitk_image.GetDirection())
    res = {"sitk_image": sitk_image,
           "npy_image": sitk.GetArrayFromImage(sitk_image),
           "origin": origin,
           "spacing": spacing,
           "direction": direction}

    return res


def _load_ct_from_dicom(dcm_path: str, sort_by_distance: bool = True) -> sitk.Image:
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
            super(DcmInfo, self).__init__()

            self.dcm_path = dcm_path
            self.series_instance_uid = series_instance_uid
            self.acquisition_number = acquisition_number
            self.sop_instance_uid = sop_instance_uid
            self.instance_number = instance_number
            self.image_orientation_patient = image_orientation_patient
            self.image_position_patient = image_position_patient

            self.slice_distance = self._cal_distance()

        def _cal_distance(self):
            normal = [self.image_orientation_patient[1] * self.image_orientation_patient[5] -
                      self.image_orientation_patient[2] * self.image_orientation_patient[4],
                      self.image_orientation_patient[2] * self.image_orientation_patient[3] -
                      self.image_orientation_patient[0] * self.image_orientation_patient[5],
                      self.image_orientation_patient[0] * self.image_orientation_patient[4] -
                      self.image_orientation_patient[1] * self.image_orientation_patient[3]]

            distance = 0
            for i in range(3):
                distance += normal[i] * self.image_position_patient[i]
            return distance

    def is_sop_instance_uid_exist(dcm_info, dcm_infos):
        for item in dcm_infos:
            if dcm_info.sop_instance_uid == item.sop_instance_uid:
                return True
        return False

    def get_dcm_path(dcm_info):
        return dcm_info.dcm_path

    reader = sitk.ImageSeriesReader()
    if sort_by_distance:
        dcm_infos = []

        files = os.listdir(dcm_path)
        for file in files:
            file_path = os.path.join(dcm_path, file)

            dcm = dicomio.read_file(file_path, force=True)
            _series_instance_uid = dcm.SeriesInstanceUID
            _acquisition_number = dcm.AcquisitionNumber
            _sop_instance_uid = dcm.SOPInstanceUID
            _instance_number = dcm.InstanceNumber
            _image_orientation_patient = dcm.ImageOrientationPatient
            _image_position_patient = dcm.ImagePositionPatient

            dcm_info = DcmInfo(file_path, _series_instance_uid, _acquisition_number, _sop_instance_uid,
                               _instance_number, _image_orientation_patient, _image_position_patient)

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_path)

    reader.SetFileNames(dcm_series)
    sitk_image = reader.Execute()
    return sitk_image


def save_sitk_from_npy(npy_image: np.ndarray, save_path: str, origin=None, spacing=None,
                       direction=None, sitk_type=None, use_compression: bool = False):
    sitk_image = sitk.GetImageFromArray(npy_image)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    save_sitk_image(sitk_image, save_path, sitk_type, use_compression)


def dcm_2_mha(dcm_path: str, mha_path: str, use_compress: Optional[bool] = False):
    res = load_sitk_image(dcm_path)
    sitk.WriteImage(res['sitk_image'], mha_path, use_compress)


def save_sitk_image(sitk_image: sitk.Image, save_path: str, sitk_type=None, use_compression: bool = False):
    if sitk_type is not None:
        sitk_image = sitk.Cast(sitk_image, sitk_type)
    sitk.WriteImage(sitk_image, save_path, use_compression)

