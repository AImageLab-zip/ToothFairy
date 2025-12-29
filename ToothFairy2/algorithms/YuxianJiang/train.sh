#!/usr/bin/env bash


nnUNetv2_train 112 3d_fullres 0 -tr nnUNetTrainerNoDA

nnUNetv2_train 112 3d_fullres 1 -tr nnUNetTrainerNoDA

nnUNetv2_train 112 3d_fullres 2 -tr nnUNetTrainerNoDA

nnUNetv2_train 112 3d_fullres 3 -tr nnUNetTrainerNoDA

nnUNetv2_train 112 3d_fullres 4 -tr nnUNetTrainerNoDA
