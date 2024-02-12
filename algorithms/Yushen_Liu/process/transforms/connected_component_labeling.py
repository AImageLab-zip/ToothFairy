import cc3d
import fastremap
import numpy as np

from typing import List


class ConnectedComponentLabeling:
    """Connected component labelling"""
    merge_mask = None

    def __init__(self):
        pass

    @staticmethod
    def keep_multi_topk_target(mask: np.ndarray):
        pass

    @classmethod
    def extract_non_zeros_mask(cls, masks: np.ndarray, area_least: int) -> [list, List[np.ndarray]]:
        """
        Extract the non-zeros mask from multi-channel masks.
        :param masks: multi-channel masks.
        :param area_least: the least area of connected region.
        :return:
        """
        mask_shape = masks.shape
        merge_mask = np.zeros(mask_shape[1:], np.uint8)
        out_idx = []
        out_masks = []
        for i in range(mask_shape[0]):
            t_mask = masks[i].copy()
            if np.sum(t_mask) < area_least:
                continue
            merge_mask[t_mask != 0] = i + 1
            out_idx.append(i + 1)
            out_masks.append(t_mask)
        cls.merge_mask = merge_mask

        return out_idx, out_masks

    @staticmethod
    def keep_topk_target(mask: np.ndarray, k: int, area_least: int, out_mask: np.ndarray, out_label: int = 1) -> None:
        """Keep the topK largest connected region from single channel mask.
        :param mask: single channel mask.
        :param k: top k.
        :param area_least: the least area of connected region.
        :param out_mask: return target mask in place.
        :param out_label: target label.
        :return: None
        """
        labeled_mask = cc3d.connected_components(mask, connectivity=26)
        areas = {}
        for label, extracted in cc3d.each(labeled_mask, binary=True, in_place=True):
            areas[label] = fastremap.foreground(extracted)
        candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

        k = max(1, k)
        for i in range(min(k, len(candidates))):
            if candidates[i][1] > area_least:
                out_mask[labeled_mask == int(candidates[i][0])] = out_label


