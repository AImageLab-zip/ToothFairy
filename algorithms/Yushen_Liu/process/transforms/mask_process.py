
import cc3d
import fastremap
import numpy as np
from scipy.ndimage.interpolation import zoom
from typing import List, Optional, Sequence, Union

import cupy as cp
from cucim.skimage.measure import label
import scipy.ndimage.morphology as morphology


__all__ = ['crop_image_according_to_mask',
           'crop_image_according_to_bbox',
           'extract_bbox',
           'keep_multi_channel_cc_mask',
           'merge_keep_multi_channel_cc_mask',
           'keep_resample_multi_channel_cc_mask',
           'keep_single_channel_cc_mask',
           'cupy_keep_single_channel_cc_mask',
           'cupy_keep_multi_channel_cc_mask']


def smooth_mask(npy_mask, area_least=10, is_binary_close=False, morp_iters=3):
    npy_mask = npy_mask.astype(np.uint8)
    npy_mask = remove_small_connected_object(npy_mask, area_least)
    if is_binary_close:
        struct = morphology.generate_binary_structure(3, 1)
        npy_mask = morphology.binary_closing(npy_mask, structure=struct, iterations=morp_iters)
    npy_mask = npy_mask.astype(np.uint8)

    return npy_mask


def remove_small_connected_object(npy_mask, area_least=10):
    from skimage import measure
    from skimage.morphology import label

    npy_mask[npy_mask != 0] = 1
    labeled_mask, num = label(npy_mask, neighbors=4, background=0, return_num=True)
    region_props = measure.regionprops(labeled_mask)

    res_mask = np.zeros_like(npy_mask)
    for i in range(1, num + 1):
        t_area = region_props[i - 1].area
        if t_area > area_least:
            res_mask[labeled_mask == i] = 1

    return res_mask


def crop_image_according_to_mask(npy_image: np.ndarray, npy_mask: np.ndarray, margin: Optional[list] = None):
    """
    Cropping image and mask according to the bounding box of foreground where bounding box extends margin.
    :param npy_image: input image array.
    :param npy_mask: input mask array.
    :param margin: extend margin.
    :return:
        cropped image
        cropped mask
    """
    if margin is None:
        margin = [20, 20, 20]

    bbox = extract_bbox(npy_mask)
    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([npy_image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)

    crop_image = npy_image[
                 extend_bbox[0, 0]:extend_bbox[0, 1],
                 extend_bbox[1, 0]:extend_bbox[1, 1],
                 extend_bbox[2, 0]:extend_bbox[2, 1]]
    crop_mask = npy_mask[
                extend_bbox[0, 0]:extend_bbox[0, 1],
                extend_bbox[1, 0]:extend_bbox[1, 1],
                extend_bbox[2, 0]:extend_bbox[2, 1]]

    return crop_image, crop_mask


def crop_image_according_to_bbox(npy_image: np.ndarray, bbox: list, margin: Optional[list] = None):
    """
    Cropping image according to the bounding box of target where bounding box extends margin.
    :param npy_image: input image array
    :param bbox: target bounding box
    :param margin: extend margin.
    :return:
        cropped image
        extend bounding box
    """
    if margin is None:
        margin = [20, 20, 20]

    image_shape = npy_image.shape
    extend_bbox = [max(0, int(bbox[0]-margin[0])),
                   min(image_shape[0], int(bbox[1]+margin[0])),
                   max(0, int(bbox[2]-margin[1])),
                   min(image_shape[1], int(bbox[3]+margin[1])),
                   max(0, int(bbox[4]-margin[2])),
                   min(image_shape[2], int(bbox[5]+margin[2]))]
    crop_image = npy_image[extend_bbox[0]:extend_bbox[1],
                           extend_bbox[2]:extend_bbox[3],
                           extend_bbox[4]:extend_bbox[5]]

    return crop_image, extend_bbox


def convert_mask_2_one_hot(npy_mask: np.ndarray, label=None):
    """Convert mask label into one hot coding."""
    if label is None:
        label = [1]

    npy_masks = []
    for i_label in range(1, np.max(np.array(label)) + 1):
        mask_i = (npy_mask == i_label)
        npy_masks.append(mask_i)

    npy_mask_czyx = np.stack(npy_masks, axis=0)
    npy_mask_czyx = npy_mask_czyx.astype(np.uint8)

    return npy_mask_czyx


def extract_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Extract the bounding box of foreground from input mask.
    :param mask: input mask
    :return:
        the bounding box of target.
    """
    zz, yy, xx = np.where(mask > 0)

    bbox = np.array([[np.min(zz), np.max(zz)],
                     [np.min(yy), np.max(yy)],
                     [np.min(xx), np.max(xx)]])
    return bbox


def mapping_mask_to_raw_roi(mask: np.ndarray, bbox: list, target_shape: list) -> np.ndarray:
    out_mask = np.zeros(target_shape, np.uint8)
    out_mask[bbox[0]:bbox[1],
             bbox[2]:bbox[3],
             bbox[4]:bbox[5]] = mask

    return out_mask


def keep_multi_channel_cc_mask(masks: np.ndarray, label_num: List[int], area_least: int) -> np.ndarray:
    """
    Keep largest mask from multi-channel masks.
    :param masks:multi-channel masks.
    :param label_num:the number target of each channel mask.
    :param area_least: the lease area of connected region.
    :return:
        out_mask: target mask
    """
    mask_shape = masks.shape
    assert mask_shape[0] == len(label_num)

    out_mask = np.zeros(mask_shape[1:], np.uint8)
    for i in range(mask_shape[0]):
        t_mask = masks[i].copy()
        if np.sum(t_mask) < area_least:
            continue
        keep_single_channel_cc_mask(t_mask, label_num[i], area_least, out_mask, i+1)

    return out_mask


def merge_keep_multi_channel_cc_mask(masks: np.ndarray, label_num: List[int], area_least: int) -> np.ndarray:
    """
    Merge multi-channel mask, and keep largest targets.
    :param masks: multi-channel masks
    :param label_num: the number target of each channel mask.
    :param area_least: the lease area of connected region.
    :return:
        out_mask: target mask
    """
    mask_shape = masks.shape
    assert mask_shape[0] == len(label_num)

    merge_mask = np.zeros(mask_shape[1:], np.uint8)
    all_idx = []
    for i in range(mask_shape[0]):
        t_mask = masks[i].copy()
        if np.sum(t_mask) < area_least:
            continue
        merge_mask[t_mask != 0] = i+1
        all_idx.append(i+1)

    out_mask = np.zeros(mask_shape[1:], np.uint8)
    for idx in all_idx:
        t_mask = merge_mask.copy()
        t_mask = np.where(t_mask == idx, 1, 0)
        keep_single_channel_cc_mask(t_mask, label_num[idx-1], area_least, out_mask, idx)

    return out_mask


def keep_resample_multi_channel_cc_mask(masks: np.ndarray, label_num: List[int], area_least: int, target_shape: List[int], device=None) -> np.ndarray:
    """
    Keep largest target from multi-channel masks, and resample target mask to raw image size.
    :param masks: multi-channel masks.
    :param label_num: the number target of each channel mask.
    :param area_least: the least area of connected region.
    :param target_shape: the size of target mask.
    :param device: performing the mask resampling in device.
    :return:
        out_mask: target mask
    """
    mask_shape = masks.shape
    out_mask = np.zeros(mask_shape[1:], np.uint8)
    for i in range(mask_shape[0]):
        t_mask = masks[i].copy()
        if np.sum(t_mask) < area_least:
            continue
        keep_single_channel_cc_mask(t_mask, label_num[i], area_least, out_mask, i+1)

    scale = np.array(target_shape) / mask_shape[1:]
    out_mask = zoom(out_mask, scale, order=0)

    # out_mask = out_mask[np.newaxis, np.newaxis]
    # out_mask = torch.from_numpy(out_mask).float().to(device)
    # out_mask = F.interpolate(out_mask, size=target_shape, mode='nearest')
    # out_mask = out_mask.cpu().numpy().squeeze().astype(np.uint8)

    return out_mask


def keep_single_channel_cc_mask(mask: np.ndarray, k: int, area_least: int, out_mask: np.ndarray, out_label: int):
    """
    Keep topK target from single channel mask.
    :param mask: single channel mask
    :param k: top k
    :param area_least: the least area of connected region
    :param out_mask: output mask in place
    :param out_label: target label
    :return: None
    """
    labelled_mask = cc3d.connected_components(mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labelled_mask, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labelled_mask == int(candidates[i][0])] = out_label


def remove_small_cc(mask: np.ndarray, area_least: int, topk: int, out_mask: Optional[np.ndarray] = None) -> np.ndarray:
    labeled_mask = mask.copy()
    try:
        labeled_mask = cc3d.connected_components(labeled_mask, connectivity=26)
    except:
        return mask
    areas = {}
    for label, extracted in cc3d.each(labeled_mask, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    if out_mask is None:
        out_mask = np.zeros_like(mask)
    for i in range(min(topk, len(candidates))):
        if candidates[i][1] > area_least:
            coords = np.where(labeled_mask == int(candidates[i][0]))
            out_mask[coords] = mask[coords[0][0], coords[1][0], coords[2][0]]
        else:
            break

    return out_mask


def keep_topk_cc(mask: np.ndarray, area_least: int, topk: int, out_label: int, out_mask: Optional[np.ndarray] = None) -> np.ndarray:
    labeled_mask = mask.copy()
    labeled_mask = cc3d.connected_components(labeled_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labeled_mask, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    if out_mask is None:
        out_mask = np.zeros_like(mask)
    for i in range(min(topk, len(candidates))):
        if candidates[i][1] > area_least:
            coords = np.where(labeled_mask == int(candidates[i][0]))
            out_mask[coords] = out_label
        else:
            break

    return out_mask


def cupy_keep_multi_channel_cc_mask(mask: np.ndarray, applied_labels: Union[Sequence[int], int],
                                    labels_num: Union[Sequence[int], int], area_least: Union[Sequence[int], int],
                                    connectivity: int = 3):
    """
        Keep topK connected component in multi channel mask.
    :param mask: multi channel mask in one-hot encoding, shape must be (C, spatial_dim1[, spatial_dim2, ...]).
    :param applied_labels: Labels for applying the connected component analysis on.
    :param labels_num: Number of target in each channel mask.
    :param area_least: Least area of connected region.
    :param connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
    :return:
    """
    if isinstance(applied_labels, int):
        applied_labels = [i for i in range(1, applied_labels+1)]

    if isinstance(labels_num, int):
        labels_num = [labels_num] * len(applied_labels)

    if isinstance(area_least, int):
        area_least = [area_least] * len(applied_labels)

    mask_shape = mask.shape
    mask = cp.asarray(mask)
    out_mask = cp.zeros(shape=mask_shape[1:], dtype=mask.dtype)
    if max(applied_labels) > mask_shape[0]:
        raise ValueError(f'number of labels is larger than mask channel, '
                         f'labels number: {max(applied_labels)}, mask channel: {mask_shape[0]}')
    for idx, label in enumerate(applied_labels):
        foreground = mask[label-1, ...]
        if cp.sum(foreground) < area_least[idx]:
            continue
        cupy_keep_single_channel_cc_mask(foreground, out_mask, label, area_least[idx], labels_num[idx], connectivity)
    out_mask = cp.asnumpy(out_mask)

    return out_mask


def cupy_keep_single_channel_cc_mask(mask: cp.ndarray, out_mask: cp.ndarray, out_label: int = 1,
                                     area_least: int = 64, k: int = 1, connectivity: int = 3):
    """
        Keep topK connected component in single channel mask.
    :param mask: single channel mask, shape must be (spatial_dim1[, spatial_dim2, ...]).
    :param out_mask: out mask in place.
    :param out_label: target label.
    :param area_least: the least area of connected region.
    :param k: top k.
    :param connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
    :return:
    """
    mask, num = label(mask, return_num=True, connectivity=connectivity)
    component_sizes = cp.bincount(mask.ravel())[1:]
    component_sizes_dict = {}
    for i in range(num):
        component_sizes_dict[i + 1] = component_sizes[i]

    component_sizes_tuple = sorted(component_sizes_dict.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(k, num)):
        if component_sizes_tuple[i][1] > area_least:
            out_mask[mask == int(component_sizes_tuple[i][0])] = out_label
