import SimpleITK as sitk
import numpy as np
import sys
import os
import matplotlib.pyplot as pl
from PIL import Image as Img
import random
from glob import glob

imgList = sorted(glob(r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\ToothFairy_Dataset\Dataset\*\data.npy'))
DenseMaskList = sorted(glob(r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\ToothFairy_Dataset\Dataset\*\gt_alpha.npy'))
SparseMaskList = sorted(glob(r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\ToothFairy_Dataset\Dataset\*\gt_sparse.npy'))
save_path = r'E:\CBCT_Project\IAN\Tooth_Fairy\ToothFairy_Dataset_new\dataset'
List = dict()
List['img'] = imgList
List['DenseMask'] = DenseMaskList
List['SparseMask'] = SparseMaskList
for phase in ['img','SparseMask','DenseMask']:
    for case in List[phase]:
        data = np.load(case,allow_pickle=True)
        data = np.flip(data,axis=0)

        data = sitk.GetImageFromArray(data)
        data.SetSpacing((0.3,0.3,0.3))
        path = os.path.join(save_path,phase)
        if not os.path.exists(path):
            os.mkdir(path)
        sitk.WriteImage(data,os.path.join(path,case.split('\\')[-2]+'.nii.gz'))
