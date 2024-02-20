import os
import shutil
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from file_utils import read_csv, write_txt,write_csv
from evaluations.dice import compute_meandice
from transforms.mask_one_hot import one_hot
import torch
from utils import connected_domain_filter,connected_domain_check
from torchvision import transforms
from transforms.transform import ColorJitter, NoiseJitter, PaintingJitter

def models_reliable_compute(out_mask_dir,files_name,save_reliable_path,model_list,num_class = 1):
    if not os.path.exists(save_reliable_path):
        write_csv(save_reliable_path, ['series_uid', 'mDice', 'num of components'], mul=False, mod='w')
    else:
        os.remove(save_reliable_path)
        write_csv(save_reliable_path, ['series_uid', 'mDice', 'num of components'], mul=False, mod='w')

    for case_name in files_name:
        print("Calculating...",case_name)
        out_masks = []
        for i in model_list:
            model_dir = os.path.join(out_mask_dir,i,case_name)
            mask = sitk.ReadImage(model_dir)
            raw_image_shape = mask.GetSize()
            mask_array = sitk.GetArrayFromImage(mask)
            out_masks.append(mask_array)
        if len(out_masks) > 2:
            # compute consistency loss.
            all_dice = []
            gt_mask = torch.from_numpy(out_masks[-1].astype(np.float16)).reshape(
                [1, 1, raw_image_shape[0], raw_image_shape[1], raw_image_shape[2]])
            # gt_mask = one_hot(gt_mask, num_class + 1, dim=1)
            for i in range(0, len(out_masks) - 1):
                predict_mask = torch.from_numpy(out_masks[i]). \
                    reshape([1, 1, raw_image_shape[0], raw_image_shape[1], raw_image_shape[2]])
                # predict_mask = one_hot(predict_mask, num_class, dim=1)
                dice = compute_meandice(predict_mask, gt_mask, include_background=False)
                dice = dice.numpy().squeeze()
                print(model_list[-1],'vs',model_list[i],dice)
                sum_dice = 0.
                sun_num = 0.
                if not np.isnan(dice):
                    sum_dice += dice
                    sun_num += 1
                all_dice.append(sum_dice / sun_num)
            mDice = np.mean(np.array(all_dice))
        else:
            mDice = 1.0

        mask = out_masks[-1]
        # is_complete = 1
        # for i in range(1, num_class + 1):
        #     temp_mask = mask.copy()
        #     temp_mask = np.where(temp_mask == i, 1, 0)
        #     if np.sum(temp_mask) == 0:
        #         is_complete = 0
        #         break
        print(mask.shape,np.unique(mask))
        num_con = connected_domain_check(mask)
        write_csv(save_reliable_path, [case_name, mDice, num_con], mul=False, mod='a')
        print(f'reliable mDice: {mDice}')

def select_reliable_series(out_csv_path,out_csv_path_2, out_txt_path, score_thres=0.9, num_selected=1000):
    all_df_1 = read_csv(out_csv_path)[1:]
    all_df_2 = read_csv(out_csv_path_2)[1:]
    all_df = []
    for index in range(len(all_df_1)):
        all_df.append(all_df_1[index]+all_df_2[index][1:])
    # print(all_df)
    # print(all_df_1,all_df_2)
    all_score = {}
    unselected_series_ = []
    bottleneck_thresh = 100
    for score in all_df:
        if (float(score[1]) + float(score[3]))/2 > score_thres and (float(score[2]) + float(score[4]))/2 == 2:
            all_score[score[0]] = (float(score[1]) + float(score[3]))/2
        else:
            unselected_series_.append(score[0])

    candidates = sorted(all_score.items(), key=lambda item: item[1], reverse=True)
    num_selected = min(num_selected, len(candidates))
    selected_candidates = candidates[:num_selected]
    unselected_candidates = candidates[num_selected:]
    selected_series = [item[0] for item in selected_candidates]
    selected_dice = [item[1] for item in selected_candidates]
    print('bottleneck_thresh:',np.min(selected_dice))
    unselected_series = [item[0] for item in unselected_candidates]
    unselected_series += unselected_series_

    write_txt(os.path.join(out_txt_path,'reliable_series_100.txt'), selected_series)
    write_txt(os.path.join(out_txt_path,'unreliable_series_100.txt'), unselected_series)

def selected_img_save(img_path,iteration_save_path,txt_file):
    if not os.path.exists(os.path.join(iteration_save_path,'image_selected')):
        os.mkdir(os.path.join(iteration_save_path,'image_selected'))

    save_path = os.path.join(iteration_save_path,'image_selected')

    pseudo_cases = []
    for line in open(txt_file):
        pseudo_cases.append(line.split('\n')[0])
    print(pseudo_cases)


    for c in pseudo_cases:
        shutil.copy(os.path.join(img_path, c.replace('.nii.gz','_0000.nii.gz')), os.path.join(save_path, c.replace('.nii.gz', '_0000.nii.gz')))


def strong_data_augment(image_path,label_path,process_path):
    if not os.path.exists(process_path):
        os.mkdir(process_path)

    save_image_path = os.path.join(process_path,'image')
    save_label_path = os.path.join(process_path,'label')

    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)

    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)

    for case in [ i for i in os.listdir(image_path) if i not in os.listdir(save_image_path)]:
        print('Case Name',case)
        image = sitk.ReadImage(os.path.join(image_path, case))
        image_array = sitk.GetArrayFromImage(image)

        mask = sitk.ReadImage(os.path.join(label_path,case.replace('_0000','')))
        mask_array = sitk.GetArrayFromImage(mask)

        sample = {}
        sample['image'] = image_array.astype(np.uint8)
        sample['label'] = mask_array.astype(np.uint8)


        dict_strong_transform = transforms.Compose([
            ColorJitter(),
            NoiseJitter(),
            PaintingJitter()])

        sample_trans = dict_strong_transform(sample)
        image_process_array = sample_trans['image']
        label_process_array = sample_trans['label']

        image_process = sitk.GetImageFromArray(image_process_array)
        image_process.SetOrigin(mask.GetOrigin())
        image_process.SetSpacing(mask.GetSpacing())
        image_process.SetDirection(mask.GetDirection())
        sitk.WriteImage(image_process,os.path.join(save_image_path,case))

        label_process = sitk.GetImageFromArray(label_process_array)
        label_process.SetOrigin(mask.GetOrigin())
        label_process.SetSpacing(mask.GetSpacing())
        label_process.SetDirection(mask.GetDirection())
        sitk.WriteImage(label_process, os.path.join(save_label_path, case.replace('_0000','')))


if __name__ == "__main__":
    # file_dir = '../raw_data/pseudo_mask/iters0_reliable_score/'
    # out_csv_path = '../raw_data/pseudo_mask/iters0_reliable_score/reliable_score.csv'
    # out_txt_path = '../raw_data/pseudo_mask/iters0_reliable_score/'
    # file_names = os.listdir(file_dir)
    # select_reliable_series(file_dir, file_names, out_csv_path, out_txt_path, score_thres=0.9, num_selected=1000)
    model_list = ['nnunet','nnunetv2','fine_label']

    model_dir_1 = r'/media/ps/lys/CBCT_IAN/Fairy_Tooth/experiment_newdata_val/iter1/fine_label_img_filter'

    files_name = [i for i in os.listdir(os.path.join(model_dir_1, model_list[-1])) if i.endswith('.gz')]

    models_reliable_compute(model_dir_1,files_name,os.path.join(model_dir_1,'reliable_score.csv'),model_list,num_class=1)


