### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
import torchio as tio
import skimage.measure as measure
import scipy.ndimage as nd

### Internal Imports ###
from paths import paths as p
from inference import inference_toothfairy
from evaluation import evaluation_functions as ev

########################



def run_evaluation(
    input_data_path : Union[str, pathlib.Path], 
    input_csv_path : Union[str, pathlib.Path],
    inference_method,
    inference_method_params,
    echo : bool=False,
    output_save_path : Union[str, pathlib.Path] = None,
    output_csv_path : Union[str, pathlib.Path] = None,
    transforms=None) -> None:
    """
    Documentation here.
    """
    dices = []
    hds95 = []
    input_dataframe = pd.read_csv(input_csv_path)

    dataframe = []
    for idx in range(len(input_dataframe)):
        with tc.set_grad_enabled(False):
            if echo:
                print(f"Case: {idx + 1} / {len(input_dataframe)}")
            current_case = input_dataframe.loc[idx]
            dense_available = current_case['Dense Available']
            if dense_available:
                input_path = input_data_path / current_case['Input Path']
                ground_truth_path = input_data_path / current_case['Dense Path']
                volume = sitk.ReadImage(input_path)
                volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
                ground_truth = sitk.ReadImage(ground_truth_path)
                spacing = ground_truth.GetSpacing()
                direction = ground_truth.GetDirection()
                origin = ground_truth.GetOrigin()
                ground_truth = sitk.GetArrayFromImage(ground_truth).swapaxes(0, 1).swapaxes(1, 2)
                
                if transforms is not None:
                    subject = tio.Subject(
                    input = tio.ScalarImage(tensor=tc.from_numpy(volume).unsqueeze(0)),
                    label = tio.LabelMap(tensor=tc.from_numpy(ground_truth).unsqueeze(0)))
                    result = transforms(subject)
                    transformed_input = result['input'].data.numpy()[0]
                    ground_truth = result['label'].data.numpy()[0]
                    output = inference_toothfairy.run_inference_direct(transformed_input, inference_method, inference_method_params)
                    volume = transformed_input
                else:
                    output = inference_toothfairy.run_inference(input_path, inference_method, inference_method_params)

                ### Calculate the evaluation metrics ###
                dice = ev.dice_coefficient(output, ground_truth)
                dices.append(dice)
                if echo:
                    print(f"Dice: {dice}")
                    
                hd95 = ev.hausdorff_distance_95(output, ground_truth, voxelspacing=spacing)
                if echo:
                    print(f"HD95: {hd95}")
                hds95.append(hd95)

                labels = measure.label(output)
                components = len(np.unique(labels)) - 1
                unique, counts = np.unique(labels, return_counts=True)

                gt_labels = measure.label(ground_truth)
                gt_components = len(np.unique(gt_labels)) - 1
                gt_unique, gt_counts = np.unique(gt_labels, return_counts=True)

                if echo:
                    print(f"Components: {components}")
                    print(f"GT Components: {gt_components}")
                    print(f"Counts: {counts}")
                    print(f"GT Counts: {gt_counts}")
                        
                if output_csv_path is not None:
                    path = current_case['Input Path']
                    to_append = (path, dice, hd95, components, gt_components)
                    dataframe.append(to_append)

                if output_save_path is not None:
                    case_path = output_save_path / current_case['Input Path']
                    if not os.path.isdir(case_path):
                        os.makedirs(case_path)
                    
                    to_save = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0))
                    to_save.SetSpacing(spacing)
                    to_save.SetDirection(direction)
                    to_save.SetOrigin(origin)
                    sitk.WriteImage(to_save, case_path / "input.nii.gz", useCompression=True)
                    
                    to_save = sitk.GetImageFromArray(ground_truth.swapaxes(2, 1).swapaxes(1, 0).astype(np.uint8))
                    to_save.SetSpacing(spacing)
                    to_save.SetDirection(direction)
                    to_save.SetOrigin(origin)
                    sitk.WriteImage(to_save, case_path / "ground_truth.nii.gz", useCompression=True)
                    
                    to_save = sitk.GetImageFromArray(output.swapaxes(2, 1).swapaxes(1, 0).astype(np.uint8))
                    to_save.SetSpacing(spacing)
                    to_save.SetDirection(direction)
                    to_save.SetOrigin(origin)
                    sitk.WriteImage(to_save, case_path / "output.nii.gz", useCompression=True)        
                print()
            
    if output_csv_path is not None:
        dataframe = pd.DataFrame(dataframe, columns=['Case', 'DC', 'HD95', 'Components', "GT_Components"])
        if not os.path.isdir(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))
        dataframe.to_csv(output_csv_path, index=False)    
        
################################################