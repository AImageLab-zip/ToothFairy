import argparse
import os
import shutil
from glob import glob
from networks.net_factory_3d import net_factory_3d
import torch
import time
import numpy as np
# from networks.unet_3D import unet_3D
# from networks.vnet import VNet
# from test_3D_util import test_all_case
# from test_3D_util import test_all_case_without_score
from test_3D_util_mirror import test_all_case_without_score
# from test_3D_util_rt import test_all_case_without_score

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/input/images/cbct', help='Name of Experiment')  # Path 1/4
# parser.add_argument('--exp', type=str,
#                     default='BraTS2019/Interpolation_Consistency_Training_25', help='experiment_name'ValidationSe)
parser.add_argument('--model', type=str,
                    default='nnUNetv2', help='model_name')
parser.add_argument('--TTA', type=bool,
                    default=False, help='TTA')

def Inference(FLAGS):
    num_classes = 43 ############ 2 ##################
    test_save_path = "/output/images/oral-pharyngeal-segmentation"      # Path 2/4
    json_path = r'/output/results.json'                                 # Path 3/4
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    print(f"Files in {FLAGS.root_path}: {os.listdir(FLAGS.root_path)}")
    # net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=43).cuda()
    
    ################### 1 ###############
    # /workspace

    save_model_list = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']        # Path 4/4
    model_weights = [1, 1, 1, 1, 1]

    model_weights = np.array(model_weights) / np.sum(model_weights)
    save_model_list = [os.path.join('/workspace/weight', fold, 'checkpoint_best.pth') for fold in save_model_list]
    # net.load_state_dict(torch.load(save_mode_path_0)['network_weights'])
    # net.eval()
    print("init weight from {}".format(save_model_list))
    print("init weight by weight {}".format(model_weights))
    ################# 2 #################
    avg_metric = test_all_case_without_score(save_model_list, model_weights, model_name=FLAGS.model,base_dir=FLAGS.root_path, num_classes=num_classes,
                               patch_size=(80, 160, 160), json_path=json_path,test_save_path=test_save_path, TTA_flag=FLAGS.TTA)  ############ 3 preprocess json #########
    print(f"Files in {test_save_path}: {os.path.exists(test_save_path)}, {os.listdir(test_save_path)}")
    return avg_metric


if __name__ == '__main__':
    T1 = time.time()
    FLAGS = parser.parse_args()
    torch.set_num_threads(8)

    print('Whether GPU?',torch.cuda.is_available(),torch.cuda.get_device_name(0))

    metric = Inference(FLAGS)
    T2 = time.time()
    print('Inference Time: %s s' % (T2 - T1))
    print(metric)
