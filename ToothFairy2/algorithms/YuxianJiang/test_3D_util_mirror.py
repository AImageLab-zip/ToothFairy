import math
import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
from torch.cuda.amp import autocast
from skimage import filters,measure
from skimage import morphology
import json
from scipy.ndimage import gaussian_filter
from networks.net_factory_3d import net_factory_3d
# import cupy as np

# def _compute_stats(voxels):
#     # if len(voxels) == 0:
#     #     return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
#     # median = np.median(voxels)
#     mean = np.mean(voxels)
#     sd = np.std(voxels)
#     # mn = np.min(voxels)
#     # mx = np.max(voxels)
#     percentile_99_5 = np.percentile(voxels, 99.5)
#     percentile_00_5 = np.percentile(voxels, 00.5)
#     return mean, sd, percentile_99_5, percentile_00_5

def save(_output_file,_case_results):
    with open(str(_output_file), "w") as f:
        json.dump(_case_results, f)

def process_case(case_name):
    # Load and test the image for this case


    # Write segmentation file path to result.json for this case
    return {
            "outputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=case_name)
            ],
            "error_messages": [],
        }

def compute_gaussian(tile_size, sigma_scale = 1. / 8, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def normalized(image):
    # mask = all_data[-1] > 0
    # voxels = list(modality[mask][::10])  # no need to take every voxel
    ################################################################
    # 107
    # mean = 73.87152
    # sd = 17.486288
    # percentile_99_5 = 139.99219848632777
    # percentile_00_5 = 35.504154205322266

    ################################################################
    # 109
    # mean = 811.5441284179688       ############## 6 ##############
    # std = 998.568603515625
    # percentile_99_5 = 3476.0
    # percentile_00_5 = -991.9979858398438

    LowerBound,UpperBound = -1000,5000
    image[image < LowerBound] = LowerBound
    image[image > UpperBound] = UpperBound
    # # Array = (Array  - np.mean(Array )) / np.std(Array )
    image = (image.astype(np.float32) - LowerBound) / (UpperBound - LowerBound)
    # image = (image * 255).astype(np.uint8)

    # mean, sd, percentile_99_5, percentile_00_5 = _compute_stats(voxels)
    # image = np.clip(image, percentile_00_5, percentile_99_5)
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image

def normalized_v2(image):
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image.astype(np.float32)

def test_single_case_v2(save_model_list,model_weights, image, patch_size, num_classes=1, do_mirroring = False, use_gaussian = True):
    print(f"using TTA: {do_mirroring}")
    print("Accelerated version.")
    w, h, d = image.shape
    # image = normalized_v2(image)
    with autocast():
        with torch.no_grad():
            # if the size of image is less than patch_size, then padding it
            add_pad = False
            if w < patch_size[0]:
                w_pad = patch_size[0]-w
                add_pad = True
            else:
                w_pad = 0
            if h < patch_size[1]:
                h_pad = patch_size[1]-h
                add_pad = True
            else:
                h_pad = 0
            if d < patch_size[2]:
                d_pad = patch_size[2]-d
                add_pad = True
            else:
                d_pad = 0
            wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
            hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
            dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
            if add_pad:
                image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                       (dl_pad, dr_pad)], mode='constant', constant_values=0)
            ww, hh, dd = image.shape

            step_size = 0.5
            target_step_sizes_in_voxels = [i * step_size for i in patch_size]
            sx = math.ceil((ww - patch_size[0]) / target_step_sizes_in_voxels[0]) + 1
            sy = math.ceil((hh - patch_size[1]) / target_step_sizes_in_voxels[1]) + 1
            sz = math.ceil((dd - patch_size[2]) / target_step_sizes_in_voxels[2]) + 1
            print("{}, {}, {}".format(sx, sy, sz))

            num_steps = [sx,sy,sz]
            steps = []
            for dim in range(len(patch_size)):
                # the highest step value for this dimension is
                max_step_value = image.shape[dim] - patch_size[dim]
                if num_steps[dim] > 1:
                    actual_step_size = max_step_value / (num_steps[dim] - 1)
                else:
                    actual_step_size = 99999999999  # does not matter because there is only one step at 0

                steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
                steps.append(steps_here)
            # score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
            # cnt = np.zeros(image.shape).astype(np.float32)
            score_map_torch = torch.zeros((num_classes, ) + image.shape, dtype=torch.float16).cuda()
            cnt_torch = torch.zeros(image.shape, dtype=torch.float16).cuda()
            # print(score_map_torch.shape, score_map_torch.dtype)

            if use_gaussian:
                gaussian = compute_gaussian(patch_size)
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip(min=mn)
                gaussian_torch = torch.from_numpy(gaussian).cuda()

            image_torch = torch.from_numpy(image.astype(np.float16)).cuda()

            for j, save_model_path in enumerate(save_model_list):
                net = net_factory_3d(net_type='nnUNetv2', in_chns=1, class_num=43).cuda()
                net.load_state_dict(torch.load(save_model_path)['network_weights'])
                net.eval()
                # print(f"load from {save_model_path}")
                for x in range(0, sx):
                    xs = steps[0][x]
                    for y in range(0, sy):
                        ys = steps[1][y]
                        for z in range(0, sz):
                            zs = steps[2][z]
                            # test_patch = image[xs:xs+patch_size[0],
                            #                 ys:ys+patch_size[1], zs:zs+patch_size[2]]
                            # test_patch = np.expand_dims(np.expand_dims(
                            #     test_patch, axis=0), axis=0).astype(np.float32)

                            # # y = torch.zeros([1, num_classes] + list(test_patch.shape[2:]),
                            # #                             dtype=torch.float16).cuda()
                            # test_patch = torch.from_numpy(test_patch).cuda()

                            test_patch = image_torch[xs:xs+patch_size[0],
                                            ys:ys+patch_size[1], zs:zs+patch_size[2]]
                            test_patch = test_patch[None, None, :, :, :]

                            y = net(test_patch)[0]
                            # y = torch.softmax(y, dim=1, dtype=torch.float16)
                            y = y[0, :, :, :, :]

                            if use_gaussian:
                                score_map_torch[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y * gaussian_torch
                                cnt_torch[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += gaussian_torch
                            else:
                                score_map_torch[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                                cnt_torch[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1


            score_map = score_map_torch.cpu().data.numpy()
            cnt = cnt_torch.cpu().data.numpy()
            score_map = score_map/np.expand_dims(cnt, axis=0)
            label_map = np.argmax(score_map, axis=0)
            print(score_map.shape, score_map.dtype)
            # print(label_map.shape, label_map.dtype)

            if add_pad:
                label_map = label_map[wl_pad:wl_pad+w,
                                      hl_pad:hl_pad+h, dl_pad:dl_pad+d]

    return label_map.astype(np.uint8)


def remove_small_connected_object(npy_mask, area_least=10):
    from skimage import measure
    from skimage.morphology import label

    npy_mask[npy_mask != 0] = 1
    labeled_mask, num = label(npy_mask, return_num=True)
    print('Num of Connected Objects',num)
    if num == 2:
        print('No Postprocessing...')
        return npy_mask
    else:
        print('Postprocessing...')
        region_props = measure.regionprops(labeled_mask)

        res_mask = np.zeros_like(npy_mask)
        for i in range(1, num + 1):
            t_area = region_props[i - 1].area
            if t_area > area_least:
                res_mask[labeled_mask == i] = 1

    return res_mask

def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

	# 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    print(num_list,area_list)
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    print(num_list_sorted)
	# 去除面积较小的连通域
    if len(num_list_sorted) > 2:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[2:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    return label


def test_all_case_without_score(save_model_list,model_weights,model_name,base_dir, num_classes=14, patch_size=(80, 160, 160), json_path = None, test_save_path=None, TTA_flag = False):
    print("Testing begin")
    path = os.listdir(base_dir)
    _case_results = []
    for image_path in path:
        print('Processing',image_path)
        image = sitk.ReadImage(os.path.join(base_dir,image_path))
    #     if max(np.unique(sitk.GetArrayFromImage(image)) )> 256:
    #         print('Doning Preprocessing')
    #         image = Normalize(image, -750, 3000)
    #     image.SetSpacing((1.0,1.0,1.0))
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()

        image = sitk.Cast(image, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(image)
        image = normalized(image)
 
        prediction_all = test_single_case_v2(
            save_model_list,model_weights, image, patch_size, num_classes=43, do_mirroring=TTA_flag)

        prediction_all = postprocessing(prediction_all)

        pred_itk = sitk.GetImageFromArray(prediction_all.astype(np.uint8))

        pred_itk.SetOrigin(origin)
        pred_itk.SetSpacing(spacing)
        pred_itk.SetDirection(direction)
        sitk.WriteImage(pred_itk, test_save_path +
                        "/{}".format(image_path.replace('_0000.mha','.mha')))
        _case_results.append(process_case(image_path))
    save(json_path, _case_results)

    end = "Testing end"
    return end

def postprocessing(prediction):
    print("postprocessing by bC")
    label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
    pred_array_0 = np.uint8(np.zeros_like(prediction))
    for j, v in enumerate(label_list):
        pred_array_0[prediction == j + 1] = v

    unique = np.unique(pred_array_0)
    output_array = np.uint8(np.zeros_like(pred_array_0))
    for label in label_list:
        if label not in unique:
            continue
        pred_array_label = np.zeros_like(pred_array_0)
        pred_array_label[pred_array_0 == label] = 1
        itk_mask = sitk.GetImageFromArray(pred_array_label)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        output_mask = cc_filter.Execute(itk_mask)
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask)
        num_connected_label = cc_filter.GetObjectCount()
        np_output_mask = sitk.GetArrayFromImage(output_mask)
        if label == 7:
            max_area = 0
            max_label = 0
            for i in range(1, num_connected_label + 1):
                area = lss_filter.GetNumberOfPixels(i)
                if area > max_area:
                    max_area = area
                    max_label = i
            output_array[np_output_mask == max_label] = label
        else:
            if label in [3, 4]:
                area_thresh = 500
            elif label in [1]:
                area_thresh = 10000
            elif label in [2]:
                area_thresh = 5000
            elif label in [8]:
                area_thresh = 2000
            elif label in [5, 6, 9]:
                area_thresh = 1000
            elif label in [10, 11, 46]:
                area_thresh = 500
            elif label in [13, 14, 15, 21, 36, 37, 42, 43, 47]:
                area_thresh = 1500
            elif label in [48]:
                area_thresh = 3000
            else:
                area_thresh = 1000
            print(label, area_thresh)
            res_mask = np.zeros_like(np_output_mask)
            for i in range(1, num_connected_label + 1):
                area = lss_filter.GetNumberOfPixels(i)
                if area > area_thresh:
                    res_mask[np_output_mask == i] = 1
            output_array[res_mask == 1] = label

            

    return output_array



def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    # ravd = abs(metric.binary.ravd(pred, gt))
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # return np.array([dice, ravd, hd, asd])
    return np.array([dice])
