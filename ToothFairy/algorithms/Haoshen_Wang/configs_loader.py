import configargparse
import numpy as np
import os

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--exp_name" , type = str, default = '.', help = 'Experiment name')
    parser.add_argument("--image_dir", type = str, default = './dataset/imagesTr', help = 'The dir of images')
    parser.add_argument("--label_dir" , type = str, default = './dataset/labelsTr', help = 'The dir of labels')
    parser.add_argument("--test_dir", type = str, default = 'test_dir', help = 'The dir of test images')
    parser.add_argument('--plan_path' , type = str, default = 'plans.json' , help = 'The path of plan')
    parser.add_argument('--network', type=str, default='UNet',help = 'The network architecture' )
    parser.add_argument('--checkpoint' , type = str, default='checkpoint-105.pth'  , help =' checkpoints')
    return parser

def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    assert cfg.exp_name is not None
    assert cfg.image_dir  is not None
    assert cfg.label_dir is not None
    
    return cfg

