### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
import torchio

### Internal Imports ###
from paths import hpc_paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol




def augment_by_random_elastic(input_data_path, output_data_path, input_csv_path, output_csv_path):
    ### Params ###
    loading_params = io.default_volumetric_pytorch_load_params
    saving_params = io.default_volumetric_save_params
    loader = io.VolumetricLoader(**loading_params)
    image_saver = io.VolumetricSaver(**saving_params | {'use_compression' : False})
    gt_saver = io.VolumetricSaver(**saving_params)

    ###################

    ### Params ###

    cases_to_generate = 10000
    min_control_points = 5
    max_control_points = 15

    ### Augmentation ###
    input_dataframe = pd.read_csv(input_csv_path)
    num_cases = len(input_dataframe)
    output_data = []
    for idx in range(cases_to_generate):
        case_id = random.randint(0, num_cases - 1)
        row = input_dataframe.loc[case_id]
        dense_available = row['Dense Available']
        if dense_available:
            input_path, sparse_path, dense_path = row['Input Path'], row['Sparse Path'], row['Dense Path']
        else:
            input_path, sparse_path = row['Input Path'], row['Sparse Path']

        loader.load(input_data_path / input_path)
        input = loader.volume
        spacing = loader.spacing
        loader.load(input_data_path / sparse_path)
        sparse = loader.volume
        if dense_available:
            loader.load(input_data_path / dense_path)
            dense = loader.volume

        control_points = random.randint(min_control_points, max_control_points)
        displacement = 40 * 5 / control_points
        transform = torchio.RandomElasticDeformation(num_control_points=control_points, max_displacement=displacement)
        if dense_available:
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=input), sparse=torchio.LabelMap(tensor=sparse), dense=torchio.LabelMap(tensor=dense))
        else:
            subject = torchio.Subject(tensor=torchio.ScalarImage(tensor=input), sparse=torchio.LabelMap(tensor=sparse))
            
        result = transform(subject)
        warped_input = result['tensor'].data
        warped_sparse = result['sparse'].data
        if dense_available:
            warped_dense = result['dense'].data

        save_input_path = pathlib.Path(f"{idx}") / "image.nii"
        save_sparse_path = pathlib.Path(f"{idx}") / "sparse.nrrd"
        save_dense_path = pathlib.Path(f"{idx}") / "dense.nrrd"

        create_folder(output_data_path / save_input_path)
        create_folder(output_data_path / save_sparse_path)
        create_folder(output_data_path / save_dense_path)
            
        save_image(warped_input[0].detach().cpu().numpy(), list(spacing.numpy().astype(np.float64)), output_data_path / save_input_path)
        gt_saver.save(warped_sparse, spacing, output_data_path / save_sparse_path)
        if dense_available:
            gt_saver.save(warped_dense, spacing, output_data_path / save_dense_path)

        to_append = (save_input_path, save_dense_path, save_sparse_path, dense_available)
        output_data.append(to_append)

    ### Copy Original ###
    for idx, row in input_dataframe.iterrows():
        dense_available = row['Dense Available']
        if dense_available:
            input_path, sparse_path, dense_path = row['Input Path'], row['Sparse Path'], row['Dense Path']
        else:
            input_path, sparse_path = row['Input Path'], row['Sparse Path']
            
        copy_file(input_data_path / input_path, output_data_path / input_path)
        copy_file(input_data_path / sparse_path, output_data_path / sparse_path)
        if dense_available:
            copy_file(input_data_path / dense_path, output_data_path / dense_path)
        to_append = (input_path, dense_path, sparse_path, dense_available)
        output_data.append(to_append)

    ### Create Dataframe ###
    output_dataframe = pd.DataFrame(output_data, columns=['Input Path', 'Dense Path', 'Sparse Path', 'Dense Available'])
    output_dataframe.to_csv(output_csv_path, index=False)

def save_image(volume, spacing, save_path, origin=None, direction=None):
    image  = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(save_path))

def copy_file(input_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    shutil.copy(input_path, output_path)

def create_folder(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))    


def run():
    input_data_path = p.parsed_toothfairy_path / "Shape_256_256_256_D"
    output_data_path = p.parsed_toothfairy_path / "ElasticShape_256_256_256_D2"
    input_csv_path = p.parsed_toothfairy_path / "training_dataset.csv"
    output_csv_path = p.parsed_toothfairy_path / "training_elastic_256_d2_dataset.csv"
    augment_by_random_elastic(input_data_path, output_data_path, input_csv_path, output_csv_path)
    

if __name__ == "__main__":
    run()