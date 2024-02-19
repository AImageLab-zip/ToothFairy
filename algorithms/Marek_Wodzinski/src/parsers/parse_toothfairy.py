### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as nd

### Internal Imports ###
from paths import hpc_paths as p
# from paths import paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol




def parse_toothfairy():
    input_path = p.raw_toothfairy_path
    output_path = p.parsed_toothfairy_path
    output_shape_path_1 = output_path / "Shape_256_256_256_D"
    output_shape_path_2 = output_path / "Shape_350_350_350_D"
    output_csv_path = output_path / "dataset.csv"
    if not os.path.exists(output_shape_path_1):
        os.makedirs(output_shape_path_1)
    if not os.path.exists(output_shape_path_2):
        os.makedirs(output_shape_path_2)

    ### Parsing Params ###
    output_size_1 = (256, 256, 256)
    output_size_2 = (350, 350, 350)
    device = "cuda:0"
    

    cases = os.listdir(input_path / "Dataset")
    print(f"Num cases: {len(cases)}")
    ### Parsing ###
    dataframe = []
    for idx, case in enumerate(cases):
        case_path = os.path.join(input_path, "Dataset", case)
        print()
        print(f"Current case: {case_path}")
        volume_path = os.path.join(case_path, "data.npy")
        dense_path = os.path.join(case_path, "gt_alpha.npy")
        sparse_path = os.path.join(case_path, "gt_sparse.npy")
        volume_1, dense_1, sparse_1, dense_available = parse_case(volume_path, dense_path, sparse_path, output_size_1, device)
        volume_2, dense_2, sparse_2, dense_available = parse_case(volume_path, dense_path, sparse_path, output_size_2, device)

        output_volume_path = pathlib.Path(case) / "input.mha"
        output_dense_path = pathlib.Path(case) / "dense.mha"
        output_sparse_path = pathlib.Path(case) / "sparse.mha"
        dataframe.append((output_volume_path, output_dense_path, output_sparse_path, dense_available))

        output_volume_path_1 = output_shape_path_1 / output_volume_path
        output_dense_path_1 = output_shape_path_1 / output_dense_path
        output_sparse_path_1 = output_shape_path_1 / output_sparse_path
        
        output_volume_path_2 = output_shape_path_2 / output_volume_path
        output_dense_path_2 = output_shape_path_2 / output_dense_path
        output_sparse_path_2 = output_shape_path_2 / output_sparse_path

        if not os.path.exists(output_shape_path_1 / case):
            os.makedirs(output_shape_path_1 / case)
            
        if not os.path.exists(output_shape_path_2 / case):
            os.makedirs(output_shape_path_2 / case)

        to_save = sitk.GetImageFromArray(volume_1.swapaxes(2, 1).swapaxes(1, 0))
        sitk.WriteImage(to_save, str(output_volume_path_1))
        
        to_save = sitk.GetImageFromArray(volume_2.swapaxes(2, 1).swapaxes(1, 0))
        sitk.WriteImage(to_save, str(output_volume_path_2))

        if dense_available:
            to_save = sitk.GetImageFromArray(dense_1.swapaxes(2, 1).swapaxes(1, 0))
            sitk.WriteImage(to_save, str(output_dense_path_1), useCompression=True)
            
            to_save = sitk.GetImageFromArray(dense_2.swapaxes(2, 1).swapaxes(1, 0))
            sitk.WriteImage(to_save, str(output_dense_path_2), useCompression=True)
            
        to_save = sitk.GetImageFromArray(sparse_1.swapaxes(2, 1).swapaxes(1, 0))
        sitk.WriteImage(to_save, str(output_sparse_path_1))
        
        to_save = sitk.GetImageFromArray(sparse_2.swapaxes(2, 1).swapaxes(1, 0))
        sitk.WriteImage(to_save, str(output_sparse_path_2))

    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Dense Path', 'Sparse Path', 'Dense Available'])
    dataframe.to_csv(output_csv_path, index=False)


def parse_case(volume_path, dense_path, sparse_path, output_size, device):
    volume = np.load(volume_path).swapaxes(0, 1).swapaxes(1, 2)
    try:
        dense = np.load(dense_path).swapaxes(0, 1).swapaxes(1, 2)
        dense_available = True
    except:
        dense = None
        dense_tc = None
        resampled_dense_tc = None
        dense_available = False
    sparse = np.load(sparse_path).swapaxes(0, 1).swapaxes(1, 2)
    sparse = nd.binary_dilation(sparse, structure=np.ones((3, 3, 3)), iterations=3)
    print(f"Volume shape: {volume.shape}")
    print(f"Dense available: {dense_available}")
    if dense_available:
        print(f"Dense shape: {dense.shape}")
    print(f"Sparse shape: {sparse.shape}")
    
    volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    sparse_tc = tc.from_numpy(sparse.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if dense_available:
        dense_tc = tc.from_numpy(dense.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Volume TC shape: {volume_tc.shape}")
    if dense_available:
        print(f"Dense TC shape: {dense_tc.shape}")
    print(f"Sparse TC shape: {sparse_tc.shape}")

    resampled_volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
    if dense_available:
        resampled_dense_tc = pre_vol.resample_tensor(dense_tc, (1, 1, *output_size), mode='bilinear')
    resampled_sparse_tc = pre_vol.resample_tensor(sparse_tc, (1, 1, *output_size), mode='bilinear')

    print(f"Resampled Volume TC shape: {resampled_volume_tc.shape}")
    if dense_available:
        print(f"Resampled Dense TC shape: {resampled_dense_tc.shape}")
    print(f"Resampled Sparse TC shape: {resampled_sparse_tc.shape}")

    volume_tc = volume_tc[0, 0, :, :, :].detach().cpu().numpy()
    resampled_volume_tc = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy()

    sparse_tc = sparse_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)
    resampled_sparse_tc = (resampled_sparse_tc[0, 0, :, :, :].detach().cpu().numpy() > 0.5).astype(np.uint8)
    
    if dense_available:
        dense_tc = dense_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)
        resampled_dense_tc = (resampled_dense_tc[0, 0, :, :, :].detach().cpu().numpy() > 0.5).astype(np.uint8)

    return resampled_volume_tc, resampled_dense_tc, resampled_sparse_tc, dense_available


def split_dataframe(input_csv_path, training_csv_path, validation_csv_path, split_ratio = 0.9, seed=1234):
    dataframe = pd.read_csv(input_csv_path)
    dataframe = dataframe.sample(frac=1, random_state=seed)
    training_dataframe = dataframe[:int(split_ratio*len(dataframe))]
    validation_dataframe = dataframe[int(split_ratio*len(dataframe)):]
    print(f"Dataset size: {len(dataframe)}")
    print(f"Training dataset size: {len(training_dataframe)}")
    print(f"Validation dataset size: {len(validation_dataframe)}")
    if not os.path.isdir(os.path.dirname(training_csv_path)):
        os.makedirs(os.path.dirname(training_csv_path))
    if not os.path.isdir(os.path.dirname(validation_csv_path)):
        os.makedirs(os.path.dirname(validation_csv_path))
    training_dataframe.to_csv(training_csv_path)
    validation_dataframe.to_csv(validation_csv_path)



def run():
    parse_toothfairy()
    split_dataframe(p.parsed_toothfairy_path / "dataset.csv", p.parsed_toothfairy_path / "training_dataset.csv", p.parsed_toothfairy_path / "validation_dataset.csv", split_ratio = 0.9, seed=1234)
    pass

if __name__ == "__main__":
    run()