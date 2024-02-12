import SimpleITK as sitk
import numpy as np
import sys
import os
import matplotlib.pyplot as pl
from PIL import Image as Img
import random
from skimage.measure import label,regionprops
from skimage import morphology

def Resample(Image, NewSpacing, Label, Size = None):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)
    if not Size:
        NewSize = [int(Array.shape[2] * Spacing[0] / NewSpacing[0]), int(Array.shape[1] * Spacing[1] / NewSpacing[1]),
               int(Array.shape[0] * Spacing[2] / NewSpacing[2])]
    else:
        NewSize = Size
    # print(NewSize)
    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    if Label:
        Resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        Resample.SetInterpolator(sitk.sitkLinear)
    Resample.SetOutputSpacing(NewSpacing)

    NewImage = Resample.Execute(Image)

    return NewImage

def Normalize(Image, LowerBound, UpperBound):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)

    Array[Array < LowerBound] = LowerBound
    Array[Array > UpperBound] = UpperBound

    Array = (Array.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)
    Array = (Array * 255).astype(np.uint8)
    Image = sitk.GetImageFromArray(Array)
    Image.SetSpacing(Spacing)
    Image.SetOrigin(Origin)
    Image.SetDirection(Direction)
    return Image

def max_connected_domain(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    print('111', num_connected_label)
    area_max = 0
    max_label = 0
    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_max:
            area_max = area
            max_label = i
    res_mask[np_output_mask == max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())
    return res_itk

def connected_domain_filter(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    # print('111', num_connected_label)
    area_thresh = 50
    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_thresh:
            res_mask[np_output_mask == i] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())
    return res_itk

def connected_domain_location(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask_copy = np.copy(itk_mask)
    itk_mask_copy[itk_mask_copy != 11] = 0
    itk_mask = sitk.GetImageFromArray(itk_mask_copy)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    print('111', num_connected_label)
    area_thresh = 0
    np_output_mask = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output_mask)
    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        center = lss_filter.GetCentroid(i)

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())
    center_point = [0,0,0]
    for i in range(3):
        center_point[i] = center[2-i]
    return center_point

def connected_domain_check(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    itk_mask = sitk.GetImageFromArray(itk_mask)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数
    print('Num:', num_connected_label)

    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        # print(i,area)


    return num_connected_label


def ROI_Region_Slice_Info(mask_array):
    mask_copy = np.copy(mask_array)
    mask_copy[mask_copy > 0] = 1
    mask_voxel_coords = np.where(mask_copy == 1)
    # print(mask_voxel_coords)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    center_z = int((minzidx + maxzidx)/2)
    center_x = int((minxidx + maxxidx)/2)
    center_y = int((minyidx + maxyidx)/2)
    return center_z,maxzidx,minzidx