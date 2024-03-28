import argparse
import os.path
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import numpy as np
import time
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.transforms import Rand3DElasticd, RandAdjustContrastd, RandScaleIntensityd, \
    RandCropByPosNegLabeld, RandZoomd, RandFlipd, ToTensord, RandSpatialCrop, RandZoom, \
    RandRotate, RandFlip, Rand3DElastic, RandAdjustContrast, RandScaleIntensity, ToTensor, RandRotated, SpatialPadd, RandSpatialCropd, SpatialPad
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, AdamW

from dataloader.dataloader import Labeled_Dataset, Unlabeled_Dataset, pop_from_list
from Validation_functions import validation_model
from networks.AVGUNet import AVGUNet


ProjectDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPOCH = 500
# FROZEN_EPOCH = 50
ITERLENGTH = 250
VALIDATION_SPACE = 25
KEYS = ['image', 'label']
TRAIN_DIR = 'TrainingSet_ToothFairy'
NAME_FILE = 'train_samples_plus'

def main():
    # pre_patch = (128, 128, 128)
    # patch_size = (80, 80, 80)
    pre_patch = (128, 128, 128)
    patch_size = (96, 96, 96)

    dice_ce = DiceCELoss(sigmoid=False, softmax=True, squared_pred=False, include_background=True, to_onehot_y=True)
    # gdf_losser = GeneralizedDiceFocalLoss(sigmoid=False, softmax=False, include_background=True, to_onehot_y=True)
    # dicefocal_losser = DiceFocalLoss(sigmoid=False, softmax=False, include_background=True, to_onehot_y=True, gamma=5.0)
    labeled_transform = transforms.Compose([
        SpatialPadd(KEYS,  pre_patch, mode='reflect'),
        RandCropByPosNegLabeld(keys=KEYS, spatial_size=pre_patch, label_key='label', pos=1.0, neg=0.0),
        pop_from_list(),
        Rand3DElasticd(keys=KEYS, mode=('bilinear', 'nearest'),
                       prob=0.3, sigma_range=(0.005, 0.01), magnitude_range=(0.005, 0.01)),
        RandZoomd(keys=KEYS, min_zoom=(1, 0.8, 0.8), max_zoom=(1, 1.2, 1.2),
                  mode=['trilinear', 'nearest'], align_corners=[True, None], prob=0.3),
        RandRotated(keys=['image', 'label'], range_x=3.15, range_y=3.15, range_z=3.15, mode=['bilinear', 'nearest'],
                    prob=0.3, padding_mode="reflection"),
        # SpatialPadd(KEYS, (96, 96, 96), mode='reflect'),
        RandSpatialCropd(keys=KEYS, roi_size=patch_size, random_size=False),
        RandFlipd(keys=KEYS, spatial_axis=0, prob=0.5),
        RandFlipd(keys=KEYS, spatial_axis=1, prob=0.5),
        RandFlipd(keys=KEYS, spatial_axis=2, prob=0.5),
        RandAdjustContrastd(['image'], prob=0.2, gamma=(0.7, 1.5)),
        RandScaleIntensityd(['image'], 0.1, prob=0.2),
        ToTensord(KEYS),
    ])

    ListFiles_labeled = f"{ProjectDir}/Data/NPY_Datasets/{TRAIN_DIR}/{NAME_FILE}.txt"
    labeled_data_root = os.path.join(args.data_root, 'Labeled')
    with open(ListFiles_labeled, 'r') as f:
        SAMPLE_NUM = len(f.readlines())
    print(f'sample num is {SAMPLE_NUM}')
    repeat_num = math.ceil((ITERLENGTH * args.batch_size) / SAMPLE_NUM)
    datasets_labeled = Labeled_Dataset(ListFiles_labeled, labeled_data_root, transforms=labeled_transform, repeat_num=repeat_num, cache=False)
    dataloader_labeled = DataLoader(datasets_labeled, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, drop_last=True, num_workers=8)
    print(f'dataset lenth is {len(datasets_labeled)}')
    labeled_iter = iter(dataloader_labeled)

    # model = TrResUNet_ECA(in_ch=1, out_ch=2, heads=4, depths=6, activate='softmax')
    # model = UNet(in_ch=1, out_ch=2, activate='softmax')
    # model = TrUNet_512TransDim_DeepS(in_ch=1, out_ch=2, heads=16, depths=6, activate='softmax', image_size=(64, 64, 64), patch_size=(2, 2, 2),
    #                            out_patch=(4, 4, 4), dim_expansion_rate=1)
    model = AVGUNet(in_ch=1, out_ch=2, sliding_window=False)
    # model = UNet_SkipDCTAttn(in_ch=1, out_ch=2, img_size=(64, 64, 64), activate='softmax')
    model = model.cuda()
    # optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optim = AdamW(model.parameters(), lr=args.lr)

    loss_record = []
    model.train()
    # ALL_EPOCH = EPOCH + FROZEN_EPOCH
    ORIGIN_FLAG = False
    for epoch in range(EPOCH):
        if abs(EPOCH - epoch) <=10 and ORIGIN_FLAG == False:
            labeled_transform = transforms.Compose([
                SpatialPadd(KEYS, pre_patch, mode='reflect'),
                RandCropByPosNegLabeld(keys=KEYS, spatial_size=patch_size, label_key='label', pos=1.0, neg=0.0),
                pop_from_list(),
                ToTensord(KEYS),
            ])
            datasets_labeled = Labeled_Dataset(ListFiles_labeled, labeled_data_root, transforms=labeled_transform,
                                               repeat_num=repeat_num, cache=False)
            dataloader_labeled = DataLoader(datasets_labeled, batch_size=args.batch_size, shuffle=True,
                                            pin_memory=True, drop_last=True, num_workers=8)
            labeled_iter = iter(dataloader_labeled)
            print('pure origin images training')
            ORIGIN_FLAG = True
        time_epoch_start = time.time()
        for iteration in range(ITERLENGTH):
            torch.cuda.synchronize()
            time_iter_start = time.time()
            for pair in range(50):
                try:
                    x, target = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(dataloader_labeled)
                    x, target = next(labeled_iter)

                label_zero = False
                for item in range(args.batch_size):
                    if target[item].max() < 0.1:
                        label_zero = True
                if pair == 49:
                    logging.info('label is all zero matrix')
                if label_zero:
                    print('label is all zero matrix, get the sample again!!!')
                    continue
                else:
                    break

            # see_x = x[0][0]
            # see_x = sitk.GetImageFromArray(see_x)
            # sitk.WriteImage(see_x, '/home/yangdong/IAN_Project/grocery/x.nii.gz')
            # see_y = target[0][0]
            # see_y = sitk.GetImageFromArray(see_y)
            # sitk.WriteImage(see_y, '/home/yangdong/IAN_Project/grocery/y.nii.gz')
            # continue

            x = x.cuda()
            target = target.cuda()

            output = model(x)
            loss1 = dice_ce(output[0], target)
            loss2 = dice_ce(output[1], target)
            loss3 = dice_ce(output[2], target)
            loss4 = dice_ce(output[3], target)
            loss5 = dice_ce(output[4], target)

            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_record.append(loss.item())

            torch.cuda.synchronize()
            time_iter_end = time.time()

            if loss.item() > 100000000:
                print('model grad vanish, exit system!')
                sys.exit()
            else:
                pass

            # if epoch < FROZEN_EPOCH:
            #     logging.info(f"frozen epoch {epoch}: {iteration + 1}/{ITERLENGTH}, loss:{loss.item():.3f}, "
            #                  f"time:{(time_iter_end - time_iter_start): .3f}")
            # else:
            #     logging.info(f"train epoch {epoch-FROZEN_EPOCH}: {iteration + 1}/{ITERLENGTH}, loss:{loss.item():.3f}, "
            #                  f"time:{(time_iter_end - time_iter_start): .3f}")
            logging.info(f"epoch {epoch}: {iteration + 1}/{ITERLENGTH}, loss:{loss.item():.3f}, "
                         f"detail:{loss1.item():.3f}_{loss2.item():.3f}_{loss3.item():.3f}_{loss4.item():.3f}_{loss5.item():.3f}, "
                         f"time:{(time_iter_end - time_iter_start): .3f}")

        time_epoch_end = time.time()
        # if epoch < FROZEN_EPOCH:
        #     logging.info(f"FROZEN EPOCH {epoch}, lr: {optim.state_dict()['param_groups'][0]['lr']}, "
        #                  f"time: {(time_epoch_end - time_epoch_start):.3f}")
        # else:
        #     logging.info(f"TRAIN EPOCH {epoch-FROZEN_EPOCH}, lr: {optim.state_dict()['param_groups'][0]['lr']}, "
        #                  f"time: {(time_epoch_end - time_epoch_start):.3f}")
        logging.info(f"TRAIN EPOCH {epoch}, lr: {optim.state_dict()['param_groups'][0]['lr']}, "
                     f"time: {(time_epoch_end - time_epoch_start):.3f}")

        plt_losses(loss_record, epoch)

        # if (epoch + 1) == FROZEN_EPOCH:
        #     for param in model.parameters():
        #         param.requires_grad = True
        #     optim = AdamW(model.parameters(), lr=args.lr)

        # if epoch < FROZEN_EPOCH:
        #     pass
        # else:
        #     adjust_lr(optim, epoch-FROZEN_EPOCH, EPOCH, 0.9, args.lr)
        adjust_lr(optim, epoch, EPOCH, 0.9, args.lr)

        # if ((epoch + 1) % 50 == 0) or (((epoch + 1) % 10 == 0) and (epoch+1 >= EPOCH - 30)):
        if ((epoch + 1) % 50 == 0):
            model_save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
            torch.save({'state_dict': model.state_dict(),
                        'optim_dict': optim.state_dict()}, model_save_path)

        '''
        validation part
        '''
        Validation_flag = False
        if epoch < 50:
            if ((epoch + 1) % 10 == 0) or (epoch == 4):
                Validation_flag = True
        else:
            if ((epoch + 1) % VALIDATION_SPACE == 0):
                Validation_flag = True
        if Validation_flag:
            logging.info('********* validation now *********')
            model.eval()
            model.set_sw(True)
            # # train 0.45 spacing image
            validation_dir = f'{ProjectDir}/Data/NII_Datasets/ToothFairy_Datasets/Origin_imagesLa_WinCut_norm/val'
            name_list_file = f'/{ProjectDir}/Data/NPY_Datasets/TrainingSet_ToothFairy/val_samples.txt'
            val_save_root = f'{ProjectDir}/Validation_View/{args.version_name}/E{epoch}'
            refer_label_root = f'{ProjectDir}/Data/NII_Datasets/ToothFairy_Datasets/Origin_Labels/val'
            # # end

            # train 0.3 spacing image
            # validation_dir = f'{ProjectDir}/Data/NII_Datasets/Labeled_Datasets_OriginSize/TestingSet/images'
            # name_list_file = f'/{ProjectDir}/Data/NII_Datasets/Labeled_Datasets_OriginSize/TestingSet/validation_samples_ori.txt'
            # val_save_root = f'{ProjectDir}/Validation_View/{args.version_name}/E{epoch}'
            # refer_label_root = '/home/yangdong/IAN_Project/Data/NII_Datasets/Labeled_Datasets_OriginSize/TestingSet/labels'
            # end
            if os.path.exists(val_save_root) == False:
                os.makedirs(val_save_root)
            # model.set_inference(True)
            validation_model(validation_dir=validation_dir, names_path=name_list_file, model=model,
                             patch_size=patch_size, save_dir=val_save_root, refer_label_root=refer_label_root)
            del validation_dir, name_list_file, val_save_root

            # model.set_inference(False)
            model.set_sw(False)
            model.train()
            logging.info('********* validation end *********')

    model_save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
    torch.save({'state_dict': model.state_dict(),
                'optim_dict': optim.state_dict()}, model_save_path)


def check_and_backup():
    if args.resume != 'not resume':
        # 确认一下是否继续训练，并确认继续训练的模型地址
        resume_path = os.path.join(ProjectDir, 'models', args.model_name, args.resume)
        print(f'******resume path is: {resume_path}******\n')
        resume_flag = input('do you want to continue resume action?')
        if resume_flag not in {'Y', 'y'} or resume_flag.lower() != 'yes':
            sys.exit()
    else:
        # 如果为训练新的模型，为了防止忘记修改信息而将训练好的文件覆盖，此处会检测文件夹名称，若存在则自动退出
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
            print(f"创建{snapshot_path}")
        else:
            print(f"{snapshot_path} is exist, the program will exit!!!")
            sys.exit()

        # 将训练时的代码备份到保存model的文件夹下
        if os.path.exists(snapshot_path + f'/code'):
            if_delete = input(f"是否移除{snapshot_path + f'{os.sep}code'}?")
            if if_delete == "y" or if_delete == "Y":
                print(f"移除{snapshot_path + f'{os.sep}code'}")
                shutil.rmtree(snapshot_path + f'{os.sep}code')
            else:
                sys.exit()
        print("---copy code---")
        shutil.copytree(ProjectDir + f"/code", snapshot_path + f"/code")
        # 单独一个文件夹用来备份训练各个模型时的代码
        if os.path.isdir(ProjectDir + f"/previous_code/{args.version_name}_code"):
            print(f"移除{snapshot_path + f'{os.sep}code'}")
            shutil.rmtree(ProjectDir + f"/previous_code/{args.version_name}_code")

        shutil.copytree(ProjectDir + f"/code", ProjectDir + f"/previous_code/{args.version_name}_code")


def adjust_lr(optimizer, epoch, num_epoch, power, init_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power((1 - epoch/num_epoch), power), 8)


def plt_losses(loss_list, epoch):
    plt.cla()
    plt.plot(range(len(loss_list)), loss_list, color='blue', label='Generator')
    if os.path.isdir(snapshot_path + f"/loss_version"):
        plt.savefig(snapshot_path + f"/loss_version/epoch_{epoch}.jpg")
    else:
        os.makedirs(snapshot_path + f"/loss_version")
        plt.savefig(snapshot_path + f"/loss_version/epoch_{epoch}.jpg")


if __name__=="__main__":
    print(f'Number of available GPUs: {torch.cuda.device_count()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=ProjectDir + f'/Data/NPY_Datasets/{TRAIN_DIR}/TrainingSet',
                        help='directory name which reserves the CT images')
    parser.add_argument('--version_name', '-version_name', type=str, help='version_name')
    parser.add_argument('--batch_size', '-batch_size', type=int, default=6, help='batch_size per gpu')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='the initial learning rate of model')
    # parser.add_argument('--consistency_rampup', type=float, default=300.0, help='consistency_rampup')
    parser.add_argument('--gpu', '-gpu', type=str, default="1", help='which GPU to use')
    parser.add_argument('--resume', '-resume', default='not resume', help='the name of model which need to resume and '
                                                              'be trained more, default is not resume')

    args = parser.parse_args(
                             "-version_name AVGUNet_ToothFairy_plus_E250_FinalOrigin "
                             "-gpu 1 ".split())

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # torch.cuda.set_device(1)

    snapshot_path = ProjectDir + f"/models/" + args.version_name

    check_and_backup()
    if args.resume != 'not resume':
        resume_path = os.path.join(ProjectDir, 'models', args.model_name, args.resume)

    # log模块的设定
    logging.basicConfig(filename=snapshot_path + f"/log.txt", level=logging.INFO,
                        filemode='a', format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(os.path.basename(os.path.abspath(__file__)))
    logging.info('\n' + str(args))

    # wait_count = 28000
    # for i in range(wait_count):
    #     time.sleep(1)
    #     print(f'program will start after {wait_count - i} seconds')
    device_id = 1  # 指定显卡的ID
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f'Number of available GPUs: {torch.cuda.device_count()}')
    with torch.cuda.device(device):
        print(f'current device is {torch.cuda.current_device()}')
        main()
    # main()