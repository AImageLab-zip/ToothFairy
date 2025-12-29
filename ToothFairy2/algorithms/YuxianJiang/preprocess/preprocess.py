
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import shutil
import argparse

def prepare_dataset(input_root):
    image_read_root = os.path.join(input_root, "imagesTr")
    gt_read_root = os.path.join(input_root, "labelsTr")

    label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
    bound = [-1000, 5000]

    for case_name in os.listdir(gt_read_root):
        image_path = os.path.join(image_read_root, case_name.replace('.mha', '_0000.mha'))
        gt_path = os.path.join(gt_read_root, case_name)
        gt = sitk.ReadImage(gt_path)
        image = sitk.ReadImage(image_path)
        gt_array = sitk.GetArrayFromImage(gt)
        image_array = sitk.GetArrayFromImage(image)
        gt_unique = np.unique(gt_array)

        new_gt_array = np.zeros_like(gt_array)
        new_gt_array = np.uint8(new_gt_array)
        for j, v in enumerate(label_list):
            new_gt_array[gt_array == v] = int(j + 1)
        new_gt = sitk.GetImageFromArray(new_gt_array.astype(np.uint8))

        new_image_array = image_array.copy()
        new_image_array[new_image_array < bound[0]] = bound[0]
        new_image_array[new_image_array > bound[1]] = bound[1]
        new_image_array = (new_image_array.astype(np.float32) - bound[0]) / (bound[1] - bound[0])
        new_image = sitk.GetImageFromArray(new_image_array.astype(np.float32))

        new_image.SetSpacing(image.GetSpacing())
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetDirection(image.GetDirection())
        new_gt.SetSpacing(image.GetSpacing())
        new_gt.SetOrigin(image.GetOrigin())
        new_gt.SetDirection(image.GetDirection())
        sitk.WriteImage(new_image, image_path)
        sitk.WriteImage(new_gt, gt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    args = parser.parse_args()
    prepare_dataset(args.input_root)

