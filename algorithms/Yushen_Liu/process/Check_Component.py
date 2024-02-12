import SimpleITK as sitk
import numpy as np
import sys
import os
import shutil
from utils import connected_domain_check
dataset_dir = r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\dataset\DenseMask'
badcase_dir = r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\Dense_Mask\bad_case'
finecase_dir = r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\Dense_Mask\fine_case'
bad_case = []
for case_name in os.listdir(dataset_dir):
    print('*' * 25)
    print(case_name)
    mask = sitk.ReadImage(os.path.join(dataset_dir,case_name))
    mask_array = sitk.GetArrayFromImage(mask)

    num_con = connected_domain_check(mask_array)

    if num_con != 2:
        bad_case.append(case_name)
        shutil.copy(os.path.join(dataset_dir,case_name),os.path.join(badcase_dir,case_name))
    else:
        shutil.copy(os.path.join(dataset_dir, case_name), os.path.join(finecase_dir, case_name))

print('Bad Case Num:',len(bad_case))
print('Bad Case:',bad_case)