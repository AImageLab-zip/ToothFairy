import SimpleITK as sitk
import numpy as np
import sys
import os
import random
from multiprocessing import Pool, cpu_count
import json
from batchgenerators.utilities.file_and_folder_operations import save_json
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

def process_case(case_name):
    # Load and test the image for this case


    # Write segmentation file path to result.json for this case
    return {
            "outputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "error_messages": [],
        }

def main(img_path,case,_case_results):
    Image = sitk.ReadImage(os.path.join(img_path,case))
    # Image = Normalize(Image, -750, 3000)
    Image.SetSpacing((1.0,1.0,1.0))
    sitk.WriteImage(Image,os.path.join(img_path,case))

def save(_output_file,_case_results):
    with open(str(_output_file), "w") as f:
        json.dump(_case_results, f)

if __name__ == "__main__":
    img_path = '/output/images/inferior-alveolar-canal'
    json_path = '/output/results.json'
    pool = Pool(int(cpu_count() / 2))
    print('pool count',int(cpu_count() / 2))
    _case_results = []
    for case in os.listdir(img_path):
        print('***************')
        print(case)
        try:
            pool.apply_async(main, (img_path, case, _case_results))
        except Exception as err:
            print('Outer single copy throws exception %s, with case name %s!' % (err, case))

    pool.close()
    pool.join()



