#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from torch import nn
import torch
import numpy as np
from networks.neural_network import SegmentationNetwork
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
import torch.nn.functional

default_dict = {
    "base_num_features": 32,
    "conv_per_stage": 2,
    "initial_lr": 0.01,
    "lr_scheduler": None,
    "lr_scheduler_eps": 0.001,
    "lr_scheduler_patience": 30,
    "lr_threshold": 1e-06,
    "max_num_epochs": 1000,
    "net_conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "net_num_pool_op_kernel_sizes": [[1,1,1],[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    "net_pool_per_axis": [4, 5, 5],
    "num_batches_per_epoch": 250,
    "num_classes": 1,
    "num_input_channels": 1,
    "transpose_backward": [0, 1, 2],
    "transpose_forward": [0, 1, 2],
}
def initialize_nnunetv2(threeD=True, num_classes=2):
    """
    This is specific to the U-Net and must be adapted for other network architectures
    :return:
    """
    # self.print_to_log_file(self.net_num_pool_op_kernel_sizes)
    # self.print_to_log_file(self.net_conv_kernel_sizes)

    if threeD:
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d
    default_dict["num_classes"] = num_classes
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    network_class = PlainConvUNet

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2]
    }

    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    network = network_class(input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=default_dict['net_conv_kernel_sizes'],
        strides=default_dict['net_num_pool_op_kernel_sizes'],
        num_classes=2,
        deep_supervision=True,
        **conv_or_blocks_per_stage,
        **kwargs['PlainConvUNet'])
    print("nnUNetv2 have {} paramerters in total".format(
        sum(x.numel() for x in network.parameters())))
    return network.cuda()