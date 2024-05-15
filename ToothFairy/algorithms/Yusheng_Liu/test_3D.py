import argparse
import os
import shutil
from glob import glob
from networks.net_factory_3d import net_factory_3d
import torch
import time
# from networks.unet_3D import unet_3D
# from networks.vnet import VNet
# from test_3D_util import test_all_case
# from test_3D_util import test_all_case_without_score
from test_3D_util_mirror import test_all_case_without_score
# from test_3D_util_rt import test_all_case_without_score

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/input/images/cbct', help='Name of Experiment')
# parser.add_argument('--exp', type=str,
#                     default='BraTS2019/Interpolation_Consistency_Training_25', help='experiment_name'ValidationSe)
parser.add_argument('--model', type=str,
                    default='nnUNet_small', help='model_name')
parser.add_argument('--TTA', type=bool,
                    default=True, help='TTA')

def Inference(FLAGS):
    num_classes = 2
    test_save_path = "/output/images/inferior-alveolar-canal"
    json_path = r'/output/results.json'
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    # net = VNet(n_classes=num_classes, in_channels=1).cuda()
    # save_mode_path = os.path.join(
    #     snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    if FLAGS.model == 'nnUNet':
        # save_mode_path = '/workspace/weight/model_109_iter2_flip_raw_data.model'
        # save_mode_path = r'/workspace/weight/model_107_iter2_flip.model'
        save_mode_path = r'/workspace/weight/model_110_iter3_flip_raw_data.model'
        net.load_state_dict(torch.load(save_mode_path)['state_dict'])
    elif FLAGS.model == 'nnUNet_small':
        save_mode_path = '/workspace/weight/model_109_small_stage5_version2.model'
        net.load_state_dict(torch.load(save_mode_path)['state_dict'])
    else:
        save_mode_path = r'/media/ps/lys_ssd/nnunetv2_data/nnUNet_trained_models/Dataset106_IAN_Dense_Val_iter1_data_filter/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth'
    # save_mode_path = '/home/ps/disk12t/xinrui/nnFormer-miccai/nnformer/DATASET/nnFormer_raw/RESULTS_FOLDER/nnFormer/3d_fullres/Task002_Synapse/nnFormerTrainerV2_1__nnFormerPlansv2.1/fold_3/model_best.model'
    # net.load_state_dict(torch.load(save_mode_path)['state_dict'])
    # print(torch.load(save_mode_path))
        net.load_state_dict(torch.load(save_mode_path)['network_weights'])
    print("init weight from {}".format(save_mode_path))
    net.eval()
    # avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
    #                            patch_size=(64, 128, 128), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    avg_metric = test_all_case_without_score(net, model_name=FLAGS.model,base_dir=FLAGS.root_path, num_classes=num_classes,
                               patch_size=(80, 160, 192), stride_xy=64, stride_z=64, json_path=json_path,test_save_path=test_save_path, TTA_flag=FLAGS.TTA)
    # avg_metric = test_all_case_without_score(net, base_dir=FLAGS.root_path, num_classes=num_classes,
    #                            patch_size=(40, 224, 192), stride_xy=64, stride_z=64, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    T1 = time.time()
    FLAGS = parser.parse_args()
    torch.set_num_threads(8)

    print('Whether GPU?',torch.cuda.is_available(),torch.cuda.get_device_name(0))

    metric = Inference(FLAGS)
    T2 = time.time()
    print('Inference Time: %s ms' % ((T2 - T1) * 1000))
    print(metric)
