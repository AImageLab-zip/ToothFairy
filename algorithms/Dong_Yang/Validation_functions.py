import logging
import os.path
from medpy.metric import dc
import SimpleITK as sitk
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from dataloader.dataloader_pretrain import mask_data


def validation_model(validation_dir, names_path, model, patch_size, save_dir, metrics=True, mode='segmentation',
                     pretrain_inform=None, device='cuda', refer_label_root=None):
    assert mode in ('segmentation', 'reconstruction')
    with open(names_path, 'r') as f:
        names_list = f.readlines()
        names_list = [name.replace('\n', '') for name in names_list]

    infer = SlidingWindowInferer(roi_size=patch_size, overlap=0.5)
    dice_list = []
    for name in names_list:
        # 获取文件名并排序
        print(f'{name} is inferenceing')
        validation_nii = os.path.join(validation_dir, name + '.nii.gz')
        validation_image = sitk.ReadImage(validation_nii)
        x = sitk.GetArrayFromImage(validation_image)
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        x = x.to(device)

        if mode == 'segmentation':
            with torch.no_grad():
                output = infer(x, model)
                output = output.cpu().numpy()
            output = np.argmax(output, axis=1)[0]
            try:
                refer_label_path = os.path.join(refer_label_root, name + '.nii.gz')
                refer_label = sitk.ReadImage(refer_label_path)
                refer_label_arr = sitk.GetArrayFromImage(refer_label)

                dice = dc(output, refer_label_arr)
                logging.info(f'{name} dice is {dice}')
                dice_list.append(dice)
            except:
                logging.info('there is some wrong when calculate dice!!!')
        else:
            x, _, _ = mask_data(x, pretrain_inform['ratio'], pretrain_inform['base_stride'], pretrain_inform['mode'])
            masked_img = sitk.GetImageFromArray(x.cpu().numpy()[0][0])
            masked_img.SetSpacing(validation_image.GetSpacing())
            masked_img.SetOrigin(validation_image.GetOrigin())
            masked_img.SetDirection(validation_image.GetDirection())
            masked_save_path = os.path.join(save_dir, name + '_masked.nii.gz')
            sitk.WriteImage(masked_img, masked_save_path)
            with torch.no_grad():
                output = infer(x, model)
                output = output.cpu().numpy()
            output = output[0][0]

        out_img = sitk.GetImageFromArray(output)
        out_img.SetSpacing(validation_image.GetSpacing())
        out_img.SetOrigin(validation_image.GetOrigin())
        out_img.SetDirection(validation_image.GetDirection())

        save_path = os.path.join(save_dir, name + '.nii.gz')
        sitk.WriteImage(out_img, save_path)
    try:
        dice_list = np.array(dice_list)
        mean_dice = dice_list.mean()
        print(f'mean dice is {mean_dice}')
    except:
        pass