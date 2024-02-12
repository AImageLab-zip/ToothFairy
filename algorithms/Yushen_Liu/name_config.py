import SimpleITK as sitk
import numpy as np
import sys
import os
import random
from multiprocessing import Pool, cpu_count

def Normalize(Image, LowerBound, UpperBound):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)

    Array[Array < LowerBound] = LowerBound
    Array[Array > UpperBound] = UpperBound
    # Array = (Array  - np.mean(Array )) / np.std(Array )
    Array = (Array.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)
    Array = (Array * 255).astype(np.uint8)
    Image = sitk.GetImageFromArray(Array)
    Image.SetSpacing(Spacing)
    Image.SetOrigin(Origin)
    Image.SetDirection(Direction)
    return Image

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

def main(img_path,case):
    Image = sitk.ReadImage(os.path.join(img_path,case))
    if max(np.unique(sitk.GetArrayFromImage(Image)) )> 256:
        print('Doning Preprocessing')
        Image = Normalize(Image, -750, 3000)
        Image.SetSpacing((0.3,0.3,0.3))
    sitk.WriteImage(Image,os.path.join(img_path,case))
    # os.rename(os.path.join(img_path,case),os.path.join(img_path,case))

if __name__ == "__main__":
    img_path = '/input/images/cbct'
    pool = Pool(int(cpu_count() / 2))
    print('pool count',int(cpu_count() / 2))
    for case in os.listdir(img_path):
        print('***************')
        print(case)
        if '_0000' in case:
            continue
        try:
            pool.apply_async(main, (img_path, case))
        except Exception as err:
            print('Outer single copy throws exception %s, with case name %s!' % (err, case))

    pool.close()
    pool.join()


