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
    mean = 342.14554
    sd = 208.45456
    percentile_99_5 = 1171.49951171875
    percentile_00_5 = -100.79910278320312

    # LowerBound,UpperBound = -750,3000
    # image[image < LowerBound] = LowerBound
    # image[image > UpperBound] = UpperBound
    # # Array = (Array  - np.mean(Array )) / np.std(Array )
    # image = (image.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)
    # image = (image * 255).astype(np.uint8)

    # mean, sd, percentile_99_5, percentile_00_5 = _compute_stats(voxels)
    image = np.clip(image, percentile_00_5, percentile_99_5)
    image = (image - mean) / sd
    return image


def test_single_case_v2(net, image, stride_xy, stride_z, patch_size, num_classes=1, do_mirroring = False):
    image = normalized(image)
    w, h, d = image.shape

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
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = steps[0][x]
        for y in range(0, sy):
            ys = steps[1][y]
            for z in range(0, sz):
                zs = steps[2][z]
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                # print('raynaryarnaranryarnararngranrary')
                # print(test_patch.shape)
                result_torch = torch.zeros([1, num_classes] + list(test_patch.shape[2:]),
                                            dtype=torch.float).cuda()
                test_patch = torch.from_numpy(test_patch).cuda()

####################################################################
                if do_mirroring:
                    mirror_idx = 8
                    mirror_axes = (0, 1, 2)
                    num_results = 2 ** len(mirror_axes)
                else:
                    mirror_idx = 1
                    mirror_axes = None
                    num_results = 1

                with autocast():
                    with torch.no_grad():
                        for m in range(mirror_idx):
                            if m == 0:
                                y = net(test_patch)[0]
                                y = torch.softmax(y, dim=1)

                                result_torch += 1 / num_results * y

                            if m == 1 and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, )))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4,))

                            if m == 2 and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, )))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3,))

                            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3)))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3))

                            if m == 4 and (0 in mirror_axes):
                                y = net(torch.flip(test_patch, (2, )))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (2, ))

                            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 2)))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 2))

                            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, 2)))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3, 2))

                            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3, 2)))[0]
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3, 2))

                # with autocast():
                #     with torch.no_grad():
                #         y1 = net(test_patch)
                #         # print(np.array(y1[0].cpu()).shape)
                #         # print(np.array(y1[1].cpu()).shape)
                #         # print(np.array(y1[2].cpu()).shape)
                #         # print(np.array(y1[0].cpu()).shape)
                #         # print(np.array(y1[1].cpu()).shape)
                #         # print(np.array(y1[2].cpu()).shape)
                #         # softmax = np.vstack(y1)
                #         # softmax_mean = np.mean(softmax, 0)
                #         # ensemble
                #         # y = torch.softmax(softmax_mean, dim=1)
                #         y = torch.softmax(y1, dim=1)


                y = result_torch.cpu().data.numpy()


                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        # score_map = score_map[:, wl_pad:wl_pad +
        #                       w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, do_mirroring = False):
    image = normalized(image)
    w, h, d = image.shape

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
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = steps[0][x]
        for y in range(0, sy):
            ys = steps[1][y]
            for z in range(0, sz):
                zs = steps[2][z]
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                # print('raynaryarnaranryarnararngranrary')
                # print(test_patch.shape)
                result_torch = torch.zeros([1, num_classes] + list(test_patch.shape[2:]),
                                            dtype=torch.float).cuda()
                test_patch = torch.from_numpy(test_patch).cuda()

####################################################################
                if do_mirroring:
                    mirror_idx = 8
                    mirror_axes = (0, 1, 2)
                    num_results = 2 ** len(mirror_axes)
                else:
                    mirror_idx = 1
                    mirror_axes = None
                    num_results = 1

                with autocast():
                    with torch.no_grad():
                        for m in range(mirror_idx):
                            if m == 0:
                                y = net(test_patch)
                                y = torch.softmax(y, dim=1)

                                result_torch += 1 / num_results * y

                            if m == 1 and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4,))

                            if m == 2 and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3,))

                            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3))

                            if m == 4 and (0 in mirror_axes):
                                y = net(torch.flip(test_patch, (2, )))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (2, ))

                            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 2))

                            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                                y = net(torch.flip(test_patch, (3, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (3, 2))

                            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                                y = net(torch.flip(test_patch, (4, 3, 2)))
                                y = torch.softmax(y, dim=1)
                                result_torch += 1 / num_results * torch.flip(y, (4, 3, 2))

                # with autocast():
                #     with torch.no_grad():
                #         y1 = net(test_patch)
                #         # print(np.array(y1[0].cpu()).shape)
                #         # print(np.array(y1[1].cpu()).shape)
                #         # print(np.array(y1[2].cpu()).shape)
                #         # print(np.array(y1[0].cpu()).shape)
                #         # print(np.array(y1[1].cpu()).shape)
                #         # print(np.array(y1[2].cpu()).shape)
                #         # softmax = np.vstack(y1)
                #         # softmax_mean = np.mean(softmax, 0)
                #         # ensemble
                #         # y = torch.softmax(softmax_mean, dim=1)
                #         y = torch.softmax(y1, dim=1)


                y = result_torch.cpu().data.numpy()


                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        # score_map = score_map[:, wl_pad:wl_pad +
        #                       w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def test_single_case_embedding(net1,net2,net3,net4,net5,net6, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

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

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net1(test_patch)
                    y2 = net2(test_patch)
                    y3 = net3(test_patch)
                    y4 = net4(test_patch)
                    y5 = net5(test_patch)
                    y6 = net6(test_patch)
                    # ensemble
                    a1 = torch.softmax(y1, dim=1)
                    a2 = torch.softmax(y2, dim=1)
                    a3 = torch.softmax(y3, dim=1)
                    a4 = torch.softmax(y4, dim=1)
                    a5 = torch.softmax(y5, dim=1)
                    a6 = torch.softmax(y6, dim=1)
                    print('a1_shape:')
                    print(a1.size)
                    print('a2_shape:')
                    print(a2.size)
                    print('a3_shape:')
                    print(a3.size)
                    print('a4_shape:')
                    print(a4.size)
                    print('a5_shape:')
                    print(a5.size)
                    print('a6_shape:')
                    print(a6.size)
                    y = (a1+a2+a3+a4+a5+a6)/6
                    print('y_shape:')
                    print(y.size)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1


    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)

def Normalize(Image, LowerBound, UpperBound):
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)

    Array[Array < LowerBound] = LowerBound
    Array[Array > UpperBound] = UpperBound
    # Array = (Array  - np.mean(Array )) / np.std(Array )
    Array = (Array.astype(np.float64) - LowerBound) / (UpperBound - LowerBound)
    Array = (Array * 255).astype(np.uint8)
    Image = sitk.GetImageFromArray(Array)
    Image.SetSpacing(Spacing)
    Image.SetOrigin(Origin)
    Image.SetDirection(Direction)
    return Image

def test_all_case_without_score1(net1,net2,net3,net4,bet5,net6,base_dir, num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None):
    print("Testing begin")
    path = os.listdir(base_dir)
    for image_path in path:
        image = sitk.ReadImage(os.path.join(base_dir,image_path))
        image = sitk.GetArrayFromImage(image)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()

        prediction = test_single_case_embedding(
            net1,net2,net3,net4,bet5,net6,image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))

        pred_itk.SetOrigin(origin)
        pred_itk.SetSpacing(spacing)
        pred_itk.SetDirection(direction)
        sitk.WriteImage(pred_itk, test_save_path +
                        "/{}_pred.nii.gz".format(image_path))

    end = "Testing end"
    return end

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


def test_all_case_without_score(net,model_name,base_dir, num_classes=14, patch_size=(64, 128, 128), stride_xy=32, stride_z=24, json_path = None, test_save_path=None, TTA_flag = False):
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

        image = sitk.GetArrayFromImage(image)
        if model_name == 'nnUNet':
            prediction = test_single_case(
                net,image, stride_xy, stride_z, patch_size, num_classes=num_classes, do_mirroring = TTA_flag)
        else:
            prediction = test_single_case_v2(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, do_mirroring=TTA_flag)

        # prediction = postprocessing(prediction)
        # prediction = remove_small_connected_object(prediction,area_least=100)
        pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))

        pred_itk.SetOrigin(origin)
        pred_itk.SetSpacing(spacing)
        pred_itk.SetDirection(direction)
        sitk.WriteImage(pred_itk, test_save_path +
                        "/{}".format(image_path.replace('_0000.nii','.nii')))
        _case_results.append(process_case(image_path))
    save(json_path, _case_results)

    end = "Testing end"
    return end

def postprocessing(prediction):
    label_value = [1]
    output = np.zeros_like(prediction)
    for i in label_value:
        label = np.zeros_like(prediction)
        label[np.where(prediction == i)] = 1
        # if i == 1 or i == 5 or i == 6 :
        #     label_i = connected_component_2(label)
        # else:
        #     label_i = connected_component(label)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     label_i = executor.map(connected_component,label)
        label_i = connected_component(label)
        output[np.where(label_i != 0)] = i
    return output


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
