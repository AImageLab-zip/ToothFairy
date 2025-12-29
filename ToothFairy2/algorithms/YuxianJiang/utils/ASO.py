import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.optimize import minimize
import argparse


def get_ConnectComponent_size(label_array):
    itk_mask = sitk.GetImageFromArray(label_array)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)
    num_connected_label = cc_filter.GetObjectCount()

    areas = []
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)
        areas.append(area)
    return areas

def ASO(gt_root, pred_root):

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
              34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

    def get_bad_num(thres, gt_area_list, pred_area_list):
        bad_num = 0
        for gt_areas, pred_areas in zip(gt_area_list, pred_area_list):
            pred_num = np.sum(np.array(pred_areas) > thres)
            gt_num = len(gt_areas)
            if (pred_num > 0 and gt_num == 0) or (pred_num == 0 and gt_num > 0):
                bad_num += 1
        return bad_num

    for label in labels:
        gt_area_list = []
        pred_area_list = []
        for case in tqdm(os.listdir(gt_root)):
            if not case.endswith(".mha"):
                continue
            gt = sitk.ReadImage(os.path.join(gt_root, case))
            pred = sitk.ReadImage(os.path.join(pred_root, case))
            gt_array = sitk.GetArrayFromImage(gt)
            pred_array = sitk.GetArrayFromImage(pred)
            gt_array_label = np.zeros_like(gt_array)
            pred_array_label = np.zeros_like(pred_array)
            gt_array_label[gt_array == label] = 1
            pred_array_label[pred_array == label] = 1
            gt_areas = get_ConnectComponent_size(gt_array_label)
            pred_areas = get_ConnectComponent_size(pred_array_label)
            gt_area_list.append(gt_areas)
            pred_area_list.append(pred_areas)
        all_threshold = np.arange(1, 201, 1) * 100
        min_thres = 0
        max_thres = 0
        min_bad_num = 1000
        init_bad_num = get_bad_num(0, gt_area_list, pred_area_list)
        for thres in tqdm(all_threshold):
            bad_num = get_bad_num(thres, gt_area_list, pred_area_list)
            if bad_num < min_bad_num:
                min_bad_num = bad_num
                min_thres = thres
                max_thres = thres
            elif bad_num == min_bad_num:
                max_thres = thres
        print(f"{label}: init bad num {init_bad_num}, threshold [{min_thres}, {max_thres}], min num {min_bad_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    args = parser.parse_args()
    ASO(args.gt_root, args.pred_root)
