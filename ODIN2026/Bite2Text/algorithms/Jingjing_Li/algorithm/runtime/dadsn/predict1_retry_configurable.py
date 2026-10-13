import os
import sys


import argparse
import gc
import hashlib
import heapq
import json
import traceback as tb
from pathlib import Path

import numpy as np
import open3d as o3d
import time
import trimesh
import torch

# 尽量减少相同输入下的 CUDA 随机差异。
# 某些第三方 CUDA 算子仍可能不是完全确定性的，
# 但 align 候选与 patch 随机序列已经被固定。
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
except ImportError as exc:
    raise ImportError(
        "本脚本需要 scipy。请先运行：pip install scipy"
    ) from exc

from model import DeformableTeethNet
from iops import align_scan


def parse_args():
    parser = argparse.ArgumentParser(
        "Two-pass tooth segmentation with geodesic ROI"
    )

    # 原模型参数
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--npoint", type=int, default=16000)
    parser.add_argument("--num_classes", type=int, default=17)
    parser.add_argument("--exp_name", type=str, default="All_align")
    parser.add_argument("--emb_dims", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--expand_times", type=float, default=1.0)

    # 两阶段推理参数
    parser.add_argument(
        "--gingiva_margin_mm",
        type=float,
        default=10.0,
        help=(
            "从第一遍可靠牙齿种子沿网格表面向外扩张的测地距离。"
            "建议先试 8、10、12 mm"
        ),
    )
    parser.add_argument(
        "--seed_confidence",
        type=float,
        default=0.45,
        help="第一遍分割中作为牙齿种子的最小类别概率",
    )
    parser.add_argument(
        "--fallback_seed_confidence",
        type=float,
        default=0.25,
        help="种子过少时自动降低到该概率阈值",
    )
    parser.add_argument(
        "--min_component_points",
        type=int,
        default=30,
        help="每个牙齿类别保留的连通分量最少顶点数",
    )
    parser.add_argument(
        "--max_components_per_label",
        type=int,
        default=2,
        help="每个牙齿类别最多保留几个主要连通分量",
    )
    parser.add_argument(
        "--max_component_diag_mm",
        type=float,
        default=28.0,
        help=(
            "牙齿种子连通分量包围盒对角线的软上限。"
            "过大的同标签分量可能是底座误标"
        ),
    )
    parser.add_argument(
        "--min_detected_tooth_classes",
        type=int,
        default=6,
        help="第一遍至少检测到多少个非零牙齿类别才允许按其裁剪",
    )
    parser.add_argument(
        "--min_roi_ratio",
        type=float,
        default=0.18,
        help="ROI 顶点比例过小时自动扩大 margin",
    )
    parser.add_argument(
        "--max_roi_ratio",
        type=float,
        default=0.92,
        help=(
            "ROI 超过该比例说明几乎没有裁剪。仍可运行第二遍，"
            "但终端会给出提示"
        ),
    )
    parser.add_argument(
        "--max_margin_mm",
        type=float,
        default=16.0,
        help="ROI 过小时允许自动扩张到的最大测地距离",
    )

    # 第一遍牙齿种子污染保护
    parser.add_argument(
        "--retry_pass1_seed",
        type=int,
        default=47,
        help="第一遍种子污染时，完整网格额外重跑所用随机种子",
    )
    parser.add_argument(
        "--retry_label_diag_mm",
        type=float,
        default=24.0,
        help=(
            "某一牙齿类别高置信度区域的 bbox 对角线超过该值时，"
            "认为可能向底座扩散并触发额外粗分割"
        ),
    )
    parser.add_argument(
        "--retry_label_ratio",
        type=float,
        default=0.10,
        help=(
            "单个牙齿类别占完整网格顶点比例超过该值时，"
            "认为可能发生大范围扩散"
        ),
    )
    parser.add_argument(
        "--retry_total_tooth_ratio",
        type=float,
        default=0.48,
        help="非零牙齿预测占比过高时触发额外粗分割",
    )
    parser.add_argument(
        "--core_probability_quantile",
        type=float,
        default=0.70,
        help=(
            "每个牙齿类别只用其较高概率部分寻找紧凑种子核心。"
            "0.70 表示优先使用该类别概率最高约 30% 的候选点"
        ),
    )
    parser.add_argument(
        "--core_density_radius_mm",
        type=float,
        default=3.0,
        help="寻找每个牙齿类别最密集核心中心时的局部半径",
    )
    parser.add_argument(
        "--core_keep_radius_mm",
        type=float,
        default=9.0,
        help=(
            "围绕每个牙齿类别最密集中心保留的种子半径。"
            "该半径只约束测地扩张的起点，不是最终牙龈 margin"
        ),
    )
    parser.add_argument(
        "--core_sample_limit",
        type=int,
        default=2500,
        help="寻找最密集核心时最多抽样多少个候选中心，控制速度",
    )
    parser.add_argument(
        "--core_min_points",
        type=int,
        default=20,
        help="单个牙齿紧凑核心至少包含的顶点数",
    )
    parser.add_argument(
        "--core_merge_center_mm",
        type=float,
        default=12.0,
        help=(
            "额外粗分割触发后，同一牙号两次紧凑核心中心距离小于该值，"
            "则合并两个核心"
        ),
    )
    parser.add_argument(
        "--single_run_core_min_confidence",
        type=float,
        default=0.45,
        help=(
            "两次粗分割只有一遍得到某牙号核心时，"
            "允许单独保留该核心所需的最小平均概率"
        ),
    )

    # 第一次分割后的牙冠种子过滤
    parser.add_argument(
        "--crown_reference_quantile",
        type=float,
        default=0.90,
        help=(
            "对每个牙号取该 Z 分位数，再用各牙号中位数估计全局牙冠高度"
        ),
    )
    parser.add_argument(
        "--crown_band_mm",
        type=float,
        default=14.0,
        help=(
            "牙齿种子只允许位于全局牙冠参考高度以下该距离以内。"
            "只限制测地种子，不直接裁最终牙龈"
        ),
    )
    parser.add_argument(
        "--class_top_z_quantile",
        type=float,
        default=0.35,
        help=(
            "每个牙号候选种子额外去掉其自身 Z 最低部分。"
            "0.35 表示优先使用该牙号较高的 65% 候选点"
        ),
    )
    parser.add_argument(
        "--arch_core_link_mm",
        type=float,
        default=22.0,
        help="不同牙号紧凑核心中心之间建立牙弓邻接的最大距离",
    )
    parser.add_argument(
        "--arch_cluster_min_classes",
        type=int,
        default=5,
        help=(
            "至少检测到该数量的牙号核心时，才启用牙弓簇分析"
        ),
    )
    parser.add_argument(
        "--arch_cluster_rescue_distance_mm",
        type=float,
        default=34.0,
        help=(
            "被主牙弓簇分开的后牙核心，距主牙弓最近中心不超过该值时，"
            "允许作为末端牙齿候选保留"
        ),
    )
    parser.add_argument(
        "--arch_cluster_rescue_z_mm",
        type=float,
        default=13.0,
        help=(
            "被分离牙齿核心与主牙弓核心中位 Z 的最大允许差值，"
            "用于排除明显位于低位底座的假核心"
        ),
    )
    parser.add_argument(
        "--arch_cluster_rescue_confidence",
        type=float,
        default=0.38,
        help="分离牙齿核心被召回所需的最低平均类别概率",
    )
    parser.add_argument(
        "--arch_cluster_rescue_max_groups",
        type=int,
        default=4,
        help="每个病例最多召回多少个被分离的小牙弓簇",
    )
    parser.add_argument(
        "--missing_core_rescue_confidence",
        type=float,
        default=0.34,
        help=(
            "某牙号只在一次粗分割中出现或被前面规则过滤时，"
            "重新召回其紧凑核心所需的最低置信度"
        ),
    )
    parser.add_argument(
        "--missing_core_rescue_distance_mm",
        type=float,
        default=34.0,
        help="漏牙紧凑核心距已保留牙弓核心的最大距离",
    )
    parser.add_argument(
        "--missing_core_rescue_z_mm",
        type=float,
        default=13.0,
        help="漏牙紧凑核心与已保留牙弓核心中位 Z 的最大差值",
    )
    parser.add_argument(
        "--missing_core_rescue_max",
        type=int,
        default=4,
        help="每个病例最多额外召回的牙齿核心数量",
    )
    # align_scan 多起点选择
    parser.add_argument(
        "--align_trials",
        type=int,
        default=4,
        help=(
            "对每个病例用多少个固定随机种子执行 align_scan。"
            "每个候选都会做一次 Pass1，自动选择质量最高者"
        ),
    )
    parser.add_argument(
        "--align_seed",
        type=int,
        default=0,
        help="第一个 align_scan 候选的基础种子",
    )
    parser.add_argument(
        "--align_seed_stride",
        type=int,
        default=1009,
        help="相邻 align_scan 候选种子的步长",
    )
    parser.add_argument(
        "--align_sample_num",
        type=int,
        default=8000,
        help=(
            "align_scan 的拟合采样点数。"
            "比原默认 2000 更稳定，但计算稍慢"
        ),
    )
    parser.add_argument(
        "--save_align_candidates",
        action="store_true",
        help="保存每个对齐候选的 Pass1 彩色 PLY，便于核对",
    )

    # 分割随机性控制
    parser.add_argument("--pass1_seed", type=int, default=11)
    parser.add_argument("--pass2_seed", type=int, default=29)

    # 上颌腭部保留
    parser.set_defaults(preserve_upper_palate=True)
    parser.add_argument(
        "--preserve_upper_palate",
        dest="preserve_upper_palate",
        action="store_true",
        help=(
            "上颌病例默认保留牙弓凸包内部的腭部区域，"
            "避免固定测地距离把中央腭部切掉"
        ),
    )
    parser.add_argument(
        "--no_preserve_upper_palate",
        dest="preserve_upper_palate",
        action="store_false",
        help="关闭上颌腭部保留",
    )
    parser.add_argument(
        "--palate_hull_scale",
        type=float,
        default=1.02,
        help=(
            "牙齿种子 XY 凸包相对中心的缩放系数。"
            "1.02 表示向外轻微扩张 2%%"
        ),
    )
    parser.add_argument(
        "--palate_depth_mm",
        type=float,
        default=18.0,
        help=(
            "从牙齿种子低位参考 Z 向下允许保留的最大深度。"
            "用于保留腭部，同时排除很低的厚底座"
        ),
    )
    parser.add_argument(
        "--palate_seed_z_quantile",
        type=float,
        default=0.10,
        help="估计牙齿种子低位参考 Z 所用分位数",
    )
    parser.add_argument(
        "--palate_min_seed_points",
        type=int,
        default=100,
        help="启用腭部凸包保留所需的最少可靠牙齿种子点数",
    )
    parser.add_argument(
        "--palate_min_detected_classes",
        type=int,
        default=6,
        help="启用腭部保留所需的最少牙齿类别数",
    )
    parser.add_argument(
        "--palate_max_added_ratio",
        type=float,
        default=0.35,
        help=(
            "腭部规则相对于完整网格最多额外加入的顶点比例，"
            "用于防止异常种子凸包保留过多底座"
        ),
    )

    # ROI 连通分量保护
    parser.add_argument(
        "--roi_component_min_vertices",
        type=int,
        default=300,
        help="ROI 独立连通块至少需要的顶点数",
    )
    parser.add_argument(
        "--roi_component_min_seed_points",
        type=int,
        default=10,
        help="ROI 独立连通块至少需要的可靠牙齿种子点数",
    )
    parser.add_argument(
        "--roi_component_vertex_ratio",
        type=float,
        default=0.03,
        help="相对最大 ROI 连通块的最小顶点比例",
    )
    parser.add_argument(
        "--roi_component_seed_ratio",
        type=float,
        default=0.05,
        help="相对最多种子连通块的最小种子比例",
    )
    parser.add_argument(
        "--roi_component_preserve_seed_fraction",
        type=float,
        default=0.90,
        help=(
            "过滤后至少保留的可靠种子比例；低于该值时会自动放宽，"
            "避免正常但拓扑断裂的牙弓被误删"
        ),
    )

    # 输出控制：默认保存 debug，使用 --no_debug 才关闭。
    parser.set_defaults(save_debug=True)
    parser.add_argument(
        "--save_debug",
        dest="save_debug",
        action="store_true",
        help="保存调试 PLY（默认已开启）",
    )
    parser.add_argument(
        "--no_debug",
        dest="save_debug",
        action="store_false",
        help="关闭调试 PLY 输出",
    )


    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("work/input"),
        help="Input mesh directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("work/output_two_pass_geodesic_recall_guard"),
        help="Segmentation output directory",
    )
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("work/debug_two_pass_geodesic_recall_guard"),
        help="Debug output directory",
    )

    return parser.parse_args()


def stable_case_seed(case_name, base_seed):
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    case_value = int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )
    return (case_value + int(base_seed)) % (2**32 - 1)


def get_random_index_patches(N, patch_size, expand=1):
    if N <= 0:
        raise ValueError("N 必须大于 0")

    if N <= patch_size:
        additional_points = patch_size - N

        if additional_points > 0:
            additional_indices = np.random.choice(
                N,
                additional_points,
                replace=True,
            )
            selected = np.concatenate(
                [np.arange(N), additional_indices]
            )
        else:
            selected = np.arange(N)

        return [selected]

    indices = np.arange(N)
    np.random.shuffle(indices)

    patches = []

    while len(indices) >= patch_size:
        patches.append(indices[:patch_size])
        indices = indices[patch_size:]

    if len(indices) > 0:
        additional_points = patch_size - len(indices)

        remaining_indices = np.setdiff1d(
            np.arange(N),
            indices,
            assume_unique=True,
        )

        # remaining_indices 理论上足够；增加保险处理。
        replace = len(remaining_indices) < additional_points

        additional_indices = np.random.choice(
            remaining_indices,
            additional_points,
            replace=replace,
        )

        patches.append(
            np.concatenate([indices, additional_indices])
        )

    expected = N // patch_size + int(N % patch_size > 0)
    assert len(patches) == expected

    expand_num = (
        int((expand - 1) * len(patches))
        if expand > 1
        else 0
    )

    for _ in range(expand_num):
        replace = N < patch_size
        patches.append(
            np.random.choice(
                N,
                patch_size,
                replace=replace,
            )
        )

    return patches


def softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exp_values = np.exp(shifted)
    denominator = np.sum(
        exp_values,
        axis=1,
        keepdims=True,
    )
    denominator = np.maximum(denominator, 1e-12)
    return exp_values / denominator


def run_segmentation(
    classifier,
    vertices,
    normals,
    device,
    num_classes,
    patch_point,
    expand_times,
    random_seed,
    stage_name,
):
    """
    在给定顶点/法向上执行一次完整 patch 推理。

    返回：
        labels: [N]
        probs:  [N, C]
        logits: [N, C]
        valid_mask: [N]
    """
    vertices = np.asarray(vertices)
    normals = np.asarray(normals)

    if vertices.shape != normals.shape:
        raise ValueError(
            f"{stage_name}: vertices/normals shape 不一致："
            f"{vertices.shape} vs {normals.shape}"
        )

    num_all = len(vertices)

    if num_all == 0:
        raise ValueError(f"{stage_name}: 输入点集为空")

    old_numpy_state = np.random.get_state()
    old_torch_state = torch.random.get_rng_state()

    if torch.cuda.is_available():
        old_cuda_states = torch.cuda.get_rng_state_all()
    else:
        old_cuda_states = None

    try:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        patch_indices = get_random_index_patches(
            num_all,
            patch_point,
            expand=expand_times,
        )

        pred_logits = np.zeros(
            (num_all, num_classes),
            dtype=np.float32,
        )
        pred_times = np.zeros(
            num_all,
            dtype=np.float32,
        )

        for selected_point_idxs in patch_indices:
            selected_point_idxs = np.asarray(
                selected_point_idxs,
                dtype=np.int64,
            )

            ori_points = vertices[selected_point_idxs]
            ori_normals = normals[selected_point_idxs]

            if (
                np.any(~np.isfinite(ori_points))
                or np.any(~np.isfinite(ori_normals))
            ):
                continue

            mean_val = ori_points.mean(axis=0)
            std_val = ori_points.std(axis=0)
            std_val[std_val < 1e-8] = 1.0

            normalized_points = (
                ori_points - mean_val
            ) / std_val

            points_np = np.concatenate(
                [normalized_points, ori_normals],
                axis=1,
            )

            if np.any(~np.isfinite(points_np)):
                continue

            points = torch.from_numpy(
                points_np
            ).float().to(device)

            points = points.unsqueeze(0).transpose(2, 1)

            seg_pred, _, _ = classifier(points)

            pred_val = (
                seg_pred
                .detach()
                .contiguous()
                .cpu()
                .numpy()[0]
            )

            if pred_val.ndim != 2:
                raise ValueError(
                    f"{stage_name}: 模型输出 shape 异常："
                    f"{pred_val.shape}"
                )

            # 兼容 [C, N] 输出。
            if (
                pred_val.shape[0] == num_classes
                and pred_val.shape[1] == len(selected_point_idxs)
            ):
                pred_val = pred_val.T

            expected_shape = (
                len(selected_point_idxs),
                num_classes,
            )

            if pred_val.shape != expected_shape:
                raise ValueError(
                    f"{stage_name}: 模型输出应为 {expected_shape}，"
                    f"实际为 {pred_val.shape}"
                )

            if np.any(~np.isfinite(pred_val)):
                continue

            pred_logits[selected_point_idxs] += pred_val
            pred_times[selected_point_idxs] += 1

        valid_mask = pred_times > 0
        valid_points = int(valid_mask.sum())

        print(
            f"  {stage_name} 有效点数: "
            f"{valid_points}/{num_all}"
        )

        if valid_points == 0:
            raise RuntimeError(
                f"{stage_name}: 没有有效预测点"
            )

        pred_times_safe = np.maximum(
            pred_times,
            1.0,
        )

        pred_logits = (
            pred_logits
            / pred_times_safe[:, None]
        )

        pred_logits = np.nan_to_num(
            pred_logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        probs = softmax_numpy(pred_logits)

        labels = np.zeros(
            num_all,
            dtype=np.int64,
        )

        labels[valid_mask] = np.argmax(
            probs[valid_mask],
            axis=1,
        )

        print(
            f"  {stage_name} 输出: "
            f"min={labels.min()}, "
            f"max={labels.max()}, "
            f"unique={len(np.unique(labels))}"
        )

        return labels, probs, pred_logits, valid_mask

    finally:
        np.random.set_state(old_numpy_state)
        torch.random.set_rng_state(old_torch_state)

        if (
            torch.cuda.is_available()
            and old_cuda_states is not None
        ):
            torch.cuda.set_rng_state_all(old_cuda_states)


def align_with_fixed_seed(
    mesh,
    case_name,
    base_seed,
    sample_num,
):
    """
    固定 align_scan 内部随机抽样，并显式增大采样数量。

    同一个病例、同一个 base_seed 会得到相同的 numpy 抽样序列。
    """
    old_numpy_state = np.random.get_state()
    seed = stable_case_seed(case_name, base_seed)

    try:
        np.random.seed(seed)
        aligned_mesh = align_scan(
            mesh,
            sample_num=sample_num,
        )
    finally:
        np.random.set_state(old_numpy_state)

    return aligned_mesh, seed


def build_mesh_graph(vertices, faces):
    """
    构建网格无向边图，边权为真实欧氏长度。
    网格坐标单位若为 mm，则测地距离同样为 mm。
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )

    # 无向边统一为小索引在前。
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    weights = np.linalg.norm(
        vertices[edges[:, 0]]
        - vertices[edges[:, 1]],
        axis=1,
    )

    valid = (
        np.isfinite(weights)
        & (weights > 1e-8)
    )

    edges = edges[valid]
    weights = weights[valid]

    row = np.concatenate(
        [edges[:, 0], edges[:, 1]]
    )
    col = np.concatenate(
        [edges[:, 1], edges[:, 0]]
    )
    data = np.concatenate(
        [weights, weights]
    )

    graph = csr_matrix(
        (data, (row, col)),
        shape=(len(vertices), len(vertices)),
    )

    return graph


def keep_reliable_label_components(
    vertices,
    graph,
    labels,
    probs,
    confidence_threshold,
    num_classes,
    min_component_points,
    max_components_per_label,
    max_component_diag_mm,
):
    """
    对第一遍预测的每个牙齿类别做连通分量清理。

    目的：
    - 去掉底座上的零散牙齿误标；
    - 保留每个牙齿类别的主要表面区域；
    - 最多允许两个主要分量，兼容局部断裂。
    """
    seed_mask = np.zeros(
        len(vertices),
        dtype=bool,
    )

    kept_class_count = 0
    class_stats = []

    for class_id in range(1, num_classes):
        candidate_mask = (
            (labels == class_id)
            & (probs[:, class_id] >= confidence_threshold)
        )

        candidate_indices = np.flatnonzero(
            candidate_mask
        )

        if len(candidate_indices) == 0:
            continue

        subgraph = graph[
            candidate_indices
        ][:, candidate_indices]

        component_count, component_labels = (
            connected_components(
                subgraph,
                directed=False,
                return_labels=True,
            )
        )

        component_sizes = np.bincount(
            component_labels,
            minlength=component_count,
        )

        order = np.argsort(
            component_sizes
        )[::-1]

        kept_components = 0
        kept_points = 0

        for component_id in order:
            component_size = int(
                component_sizes[component_id]
            )

            if component_size < min_component_points:
                continue

            local_mask = (
                component_labels == component_id
            )
            component_indices = candidate_indices[
                local_mask
            ]

            bbox_diag = float(
                np.linalg.norm(
                    np.ptp(
                        vertices[component_indices],
                        axis=0,
                    )
                )
            )

            # 这是软约束：
            # 若最大的分量超出上限，仍保留一个，防止整颗牙漏掉；
            # 后续第二个及以上的大分量则要求尺寸合理。
            if (
                bbox_diag > max_component_diag_mm
                and kept_components > 0
            ):
                continue

            seed_mask[component_indices] = True
            kept_components += 1
            kept_points += len(component_indices)

            if (
                kept_components
                >= max_components_per_label
            ):
                break

        if kept_points > 0:
            kept_class_count += 1
            class_stats.append(
                (
                    class_id,
                    kept_points,
                    kept_components,
                )
            )

    return seed_mask, kept_class_count, class_stats


def build_tooth_seed_mask(
    vertices,
    graph,
    labels,
    probs,
    args,
):
    """
    先用较严格概率阈值生成牙齿种子。
    若牙齿类别过少，则降低阈值再试一次。
    """
    thresholds = [
        args.seed_confidence,
        args.fallback_seed_confidence,
    ]

    final_result = None

    for threshold in thresholds:
        seed_mask, class_count, class_stats = (
            keep_reliable_label_components(
                vertices=vertices,
                graph=graph,
                labels=labels,
                probs=probs,
                confidence_threshold=threshold,
                num_classes=args.num_classes,
                min_component_points=(
                    args.min_component_points
                ),
                max_components_per_label=(
                    args.max_components_per_label
                ),
                max_component_diag_mm=(
                    args.max_component_diag_mm
                ),
            )
        )

        print(
            f"  牙齿种子阈值={threshold:.3f}: "
            f"类别数={class_count}, "
            f"点数={int(seed_mask.sum())}"
        )

        final_result = (
            seed_mask,
            class_count,
            class_stats,
            threshold,
        )

        if (
            class_count
            >= args.min_detected_tooth_classes
            and seed_mask.sum()
            >= args.min_component_points
        ):
            break

    return final_result



def analyze_pass1_contamination(
    vertices,
    labels,
    probs,
    args,
):
    """
    检查第一遍预测是否出现某一牙号向底座大面积扩散。

    只用于决定是否额外运行一次完整网格粗分割；
    不根据该检查直接删除任何几何。
    """
    vertex_count = len(vertices)
    reasons = []
    details = []

    tooth_mask = labels > 0
    total_tooth_ratio = float(
        tooth_mask.mean()
    )

    if total_tooth_ratio > args.retry_total_tooth_ratio:
        reasons.append(
            f"全部牙齿预测占比过高={total_tooth_ratio:.3f}"
        )

    for class_id in range(1, args.num_classes):
        class_mask = (
            (labels == class_id)
            & (
                probs[:, class_id]
                >= args.fallback_seed_confidence
            )
        )

        class_indices = np.flatnonzero(
            class_mask
        )

        if len(class_indices) < args.core_min_points:
            continue

        class_ratio = len(class_indices) / vertex_count
        bbox_diag = float(
            np.linalg.norm(
                np.ptp(
                    vertices[class_indices],
                    axis=0,
                )
            )
        )

        details.append(
            {
                "class_id": class_id,
                "points": int(len(class_indices)),
                "ratio": float(class_ratio),
                "bbox_diag": bbox_diag,
            }
        )

        if bbox_diag > args.retry_label_diag_mm:
            reasons.append(
                f"label {class_id} 扩展过大 "
                f"(diag={bbox_diag:.1f} mm)"
            )

        if class_ratio > args.retry_label_ratio:
            reasons.append(
                f"label {class_id} 点比例过高 "
                f"(ratio={class_ratio:.3f})"
            )

    # 去重，避免同一牙号产生多条完全相同的信息。
    reasons = list(dict.fromkeys(reasons))

    return {
        "suspicious": len(reasons) > 0,
        "reasons": reasons,
        "details": details,
        "total_tooth_ratio": total_tooth_ratio,
    }


def _densest_core_center(
    points,
    density_radius,
    sample_limit,
    random_seed,
):
    """
    在候选点中寻找局部密度最大的中心。

    牙齿真阳性通常集中在牙冠附近；
    向底座扩散的误标通常范围大而稀疏。
    """
    points = np.asarray(points)

    if len(points) == 0:
        raise ValueError("候选核心点为空")

    tree = cKDTree(points)

    if len(points) > sample_limit:
        rng = np.random.default_rng(
            random_seed
        )
        sample_indices = rng.choice(
            len(points),
            size=sample_limit,
            replace=False,
        )
    else:
        sample_indices = np.arange(
            len(points)
        )

    sample_points = points[
        sample_indices
    ]

    try:
        neighbor_counts = tree.query_ball_point(
            sample_points,
            r=density_radius,
            return_length=True,
        )
        neighbor_counts = np.asarray(
            neighbor_counts
        )
    except TypeError:
        neighbor_lists = tree.query_ball_point(
            sample_points,
            r=density_radius,
        )
        neighbor_counts = np.asarray(
            [len(items) for items in neighbor_lists]
        )

    best_local = int(
        np.argmax(neighbor_counts)
    )
    best_index = int(
        sample_indices[best_local]
    )

    return points[best_index], int(
        neighbor_counts[best_local]
    )



def estimate_crown_reference_z(
    vertices,
    labels,
    probs,
    args,
):
    """
    使用多个牙号的高位点，稳健估计牙冠参考 Z。

    align_scan 后，当前数据中 upper/lower 的牙冠均位于较高 Z。
    单个牙号即使向底座扩散，也不容易显著改变跨牙号中位数。
    """
    per_class_top_z = []

    for class_id in range(1, args.num_classes):
        mask = (
            (labels == class_id)
            & (
                probs[:, class_id]
                >= args.fallback_seed_confidence
            )
        )
        indices = np.flatnonzero(mask)

        if len(indices) < args.core_min_points:
            continue

        z_values = vertices[indices, 2]
        per_class_top_z.append(
            float(
                np.quantile(
                    z_values,
                    args.crown_reference_quantile,
                )
            )
        )

    if len(per_class_top_z) < 3:
        return None

    return float(
        np.median(per_class_top_z)
    )


def filter_seed_groups_by_arch_cluster(
    vertices,
    seed_groups,
    seed_stats,
    args,
):
    """
    对牙齿紧凑核心做牙弓空间一致性检查。

    旧版只保留最大的核心连通簇。这会误删：
    - 缺牙导致与主体之间间隔较大的末端磨牙；
    - 第一遍局部漏分后形成的左/右后牙小簇。

    新版：
    1. 主簇始终保留；
    2. 其他小簇若空间上接近主牙弓、Z 高度合理且置信度足够，
       则作为末端牙齿簇召回；
    3. 明显远离主牙弓或位于低位底座的簇仍删除。
    """
    if len(seed_groups) == 0:
        return seed_groups, seed_stats

    if len(seed_groups) != len(seed_stats):
        raise ValueError(
            f"seed_groups/seed_stats 数量不一致: "
            f"{len(seed_groups)} vs {len(seed_stats)}"
        )

    if len(seed_groups) < args.arch_cluster_min_classes:
        print("  牙号核心数量不足，不启用牙弓簇过滤")
        return seed_groups, seed_stats

    centers = np.asarray(
        [
            vertices[np.asarray(group, dtype=np.int64)].mean(axis=0)
            for group in seed_groups
        ],
        dtype=np.float64,
    )

    pairwise = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )

    adjacency = (
        (pairwise <= args.arch_core_link_mm)
        & (pairwise > 0)
    )

    graph = csr_matrix(adjacency.astype(np.uint8))

    component_count, component_labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    if component_count <= 1:
        print("  所有牙号核心位于同一牙弓簇")
        return seed_groups, seed_stats

    component_sizes = np.bincount(
        component_labels,
        minlength=component_count,
    )

    component_points = np.zeros(
        component_count,
        dtype=np.int64,
    )
    component_confidences = np.zeros(
        component_count,
        dtype=np.float64,
    )

    for group_index, group in enumerate(seed_groups):
        component_id = component_labels[group_index]
        component_points[component_id] += len(group)
        component_confidences[component_id] += float(
            seed_stats[group_index].get("confidence", 0.0)
        )

    component_confidences = component_confidences / np.maximum(
        component_sizes,
        1,
    )

    main_component = max(
        range(component_count),
        key=lambda cid: (
            int(component_sizes[cid]),
            int(component_points[cid]),
        ),
    )

    main_mask = component_labels == main_component
    main_centers = centers[main_mask]
    main_median_z = float(np.median(main_centers[:, 2]))

    keep_components = np.zeros(
        component_count,
        dtype=bool,
    )
    keep_components[main_component] = True

    rescue_candidates = []

    for component_id in range(component_count):
        if component_id == main_component:
            continue

        current_mask = component_labels == component_id
        current_centers = centers[current_mask]

        min_distance = float(
            np.min(
                np.linalg.norm(
                    current_centers[:, None, :]
                    - main_centers[None, :, :],
                    axis=2,
                )
            )
        )
        z_difference = float(
            abs(
                np.median(current_centers[:, 2])
                - main_median_z
            )
        )
        mean_confidence = float(
            component_confidences[component_id]
        )

        can_rescue = (
            min_distance
            <= args.arch_cluster_rescue_distance_mm
            and z_difference
            <= args.arch_cluster_rescue_z_mm
            and mean_confidence
            >= args.arch_cluster_rescue_confidence
        )

        rescue_candidates.append(
            (
                can_rescue,
                min_distance,
                z_difference,
                mean_confidence,
                int(component_sizes[component_id]),
                component_id,
            )
        )

    # 优先召回：置信度高、距离近、包含牙号多的小簇。
    rescue_candidates.sort(
        key=lambda item: (
            not item[0],
            -item[4],
            -item[3],
            item[1],
            item[2],
        )
    )

    rescued_count = 0

    for (
        can_rescue,
        min_distance,
        z_difference,
        mean_confidence,
        class_count,
        component_id,
    ) in rescue_candidates:
        status = "DROP"

        if (
            can_rescue
            and rescued_count
            < args.arch_cluster_rescue_max_groups
        ):
            keep_components[component_id] = True
            rescued_count += 1
            status = "RESCUE"

        class_ids = [
            int(seed_stats[index].get("class_id", -1))
            for index in np.flatnonzero(
                component_labels == component_id
            )
        ]

        print(
            f"  牙弓小簇 component={component_id}: "
            f"{status}, classes={class_ids}, "
            f"distance={min_distance:.2f} mm, "
            f"z_diff={z_difference:.2f} mm, "
            f"confidence={mean_confidence:.3f}"
        )

    keep_group_mask = keep_components[component_labels]

    filtered_groups = [
        group
        for group, keep
        in zip(seed_groups, keep_group_mask)
        if keep
    ]
    filtered_stats = [
        stat
        for stat, keep
        in zip(seed_stats, keep_group_mask)
        if keep
    ]

    print(
        f"  牙弓簇过滤/召回: {len(seed_groups)} -> "
        f"{len(filtered_groups)} 个牙号核心"
    )

    if len(filtered_groups) < args.min_detected_tooth_classes:
        print(
            "  ⚠️ 过滤后牙号过少，回退到过滤前全部紧凑核心"
        )
        return seed_groups, seed_stats

    return filtered_groups, filtered_stats


def extract_compact_seed_cores(
    vertices,
    labels,
    probs,
    args,
    case_seed,
    source_name,
    crown_reference_z=None,
):
    """
    为每个牙齿类别提取一个紧凑的高置信度核心。

    与旧版差别：
    - 不再把一个类别的整个最大连通分量都作为测地种子；
    - 即使该类别与底座连续相连，也只从最密集中心附近取有限半径；
    - 因而少量扩散不会把整块底座变成 ROI 起点。
    """
    vertices = np.asarray(vertices)
    result = {}

    for class_id in range(1, args.num_classes):
        predicted_mask = (
            labels == class_id
        )
        predicted_indices = np.flatnonzero(
            predicted_mask
        )

        if len(predicted_indices) < args.core_min_points:
            continue

        class_probabilities = probs[
            predicted_indices,
            class_id,
        ]

        # 先保证基础概率，再取类别内部高概率分位。
        quantile_threshold = float(
            np.quantile(
                class_probabilities,
                args.core_probability_quantile,
            )
        )
        probability_threshold = max(
            args.fallback_seed_confidence,
            quantile_threshold,
        )

        high_mask = (
            class_probabilities
            >= probability_threshold
        )
        high_indices = predicted_indices[
            high_mask
        ]

        if len(high_indices) < args.core_min_points:
            # 回退到该类别概率最高的若干点。
            order = np.argsort(
                class_probabilities
            )[::-1]
            keep_count = min(
                len(order),
                max(
                    args.core_min_points,
                    int(
                        np.ceil(
                            len(order)
                            * (1.0 - args.core_probability_quantile)
                        )
                    ),
                ),
            )
            high_indices = predicted_indices[
                order[:keep_count]
            ]

        if len(high_indices) < args.core_min_points:
            continue

        # ----------------------------------------------------------
        # 牙冠带过滤：
        # 只用该牙号较高 Z 的候选点作为测地种子来源。
        # 这一步只影响 ROI 种子，不会直接删除第二遍要保留的牙龈。
        # ----------------------------------------------------------
        class_z_values = vertices[
            high_indices,
            2,
        ]
        class_z_cut = float(
            np.quantile(
                class_z_values,
                args.class_top_z_quantile,
            )
        )

        if crown_reference_z is None:
            global_z_cut = -np.inf
        else:
            global_z_cut = (
                crown_reference_z
                - args.crown_band_mm
            )

        z_cut = max(
            class_z_cut,
            global_z_cut,
        )

        crown_indices = high_indices[
            vertices[high_indices, 2] >= z_cut
        ]

        if len(crown_indices) < args.core_min_points:
            # 不回退到低 Z 扩散区域，而是取该牙号 Z 最高的若干点。
            order = np.argsort(
                vertices[high_indices, 2]
            )[::-1]
            keep_count = min(
                len(order),
                max(
                    args.core_min_points,
                    int(
                        np.ceil(
                            len(order) * 0.20
                        )
                    ),
                ),
            )
            crown_indices = high_indices[
                order[:keep_count]
            ]

        if len(crown_indices) < args.core_min_points:
            continue

        high_indices = crown_indices

        center, density_count = (
            _densest_core_center(
                vertices[high_indices],
                density_radius=(
                    args.core_density_radius_mm
                ),
                sample_limit=args.core_sample_limit,
                random_seed=(
                    case_seed + class_id * 1009
                ),
            )
        )

        distances = np.linalg.norm(
            vertices[high_indices] - center,
            axis=1,
        )
        core_indices = high_indices[
            distances <= args.core_keep_radius_mm
        ]

        if len(core_indices) < args.core_min_points:
            nearest_order = np.argsort(
                distances
            )
            keep_count = min(
                len(nearest_order),
                args.core_min_points,
            )
            core_indices = high_indices[
                nearest_order[:keep_count]
            ]

        if len(core_indices) < args.core_min_points:
            continue

        core_center = np.average(
            vertices[core_indices],
            axis=0,
            weights=np.maximum(
                probs[core_indices, class_id],
                1e-6,
            ),
        )
        mean_confidence = float(
            probs[core_indices, class_id].mean()
        )
        bbox_diag = float(
            np.linalg.norm(
                np.ptp(
                    vertices[core_indices],
                    axis=0,
                )
            )
        )

        quality = float(
            mean_confidence
            * np.log1p(len(core_indices))
            / (1.0 + bbox_diag / 10.0)
        )

        result[class_id] = {
            "indices": np.unique(
                core_indices
            ),
            "center": core_center,
            "mean_confidence": mean_confidence,
            "bbox_diag": bbox_diag,
            "quality": quality,
            "density_count": density_count,
            "source": source_name,
        }

    return result


def merge_compact_seed_cores(
    first_cores,
    retry_cores,
    args,
):
    """
    合并两次完整网格粗分割得到的紧凑核心。

    - 两次中心接近：取并集，提高召回；
    - 两次中心相距很远：选择质量更高且更紧凑的一次；
    - 只有一次存在：要求平均概率较高才保留。
    """
    merged_mask_indices = []
    merged_stats = []
    all_classes = sorted(
        set(first_cores) | set(retry_cores)
    )

    for class_id in all_classes:
        first = first_cores.get(class_id)
        retry = retry_cores.get(class_id)

        if first is not None and retry is not None:
            center_distance = float(
                np.linalg.norm(
                    first["center"]
                    - retry["center"]
                )
            )

            if center_distance <= args.core_merge_center_mm:
                chosen_indices = np.unique(
                    np.concatenate(
                        [
                            first["indices"],
                            retry["indices"],
                        ]
                    )
                )
                decision = "merge"
                confidence = max(
                    first["mean_confidence"],
                    retry["mean_confidence"],
                )
            else:
                chosen = (
                    first
                    if first["quality"] >= retry["quality"]
                    else retry
                )
                chosen_indices = chosen["indices"]
                decision = (
                    "choose_first"
                    if chosen is first
                    else "choose_retry"
                )
                confidence = chosen[
                    "mean_confidence"
                ]

            merged_mask_indices.append(
                chosen_indices
            )
            merged_stats.append(
                {
                    "class_id": class_id,
                    "decision": decision,
                    "center_distance": center_distance,
                    "points": int(
                        len(chosen_indices)
                    ),
                    "confidence": float(
                        confidence
                    ),
                }
            )

        else:
            chosen = (
                first
                if first is not None
                else retry
            )

            if (
                chosen["mean_confidence"]
                < args.single_run_core_min_confidence
            ):
                print(
                    f"  暂不保留单次牙号核心 class={class_id}: "
                    f"confidence={chosen['mean_confidence']:.3f}"
                )
                continue

            merged_mask_indices.append(
                chosen["indices"]
            )
            merged_stats.append(
                {
                    "class_id": class_id,
                    "decision": "keep_single",
                    "center_distance": np.nan,
                    "points": int(
                        len(chosen["indices"])
                    ),
                    "confidence": float(
                        chosen["mean_confidence"]
                    ),
                }
            )

    return merged_mask_indices, merged_stats



def rescue_missing_tooth_cores(
    vertices,
    filtered_groups,
    filtered_stats,
    candidate_cores,
    args,
):
    """
    在牙弓簇过滤之后，召回被误删的真实末端牙齿核心。

    candidate_cores 是 Pass1/Pass1-Retry 每个牙号的最佳紧凑核心。
    只召回满足以下条件的缺失牙号：
    - 紧凑核心概率足够；
    - 核心尺寸不像大块底座；
    - 距已保留牙弓不远；
    - Z 高度与已保留牙齿一致。

    这一步提升 ROI 召回率，但不会把第一遍标签直接作为最终结果；
    第二遍模型仍会重新判断该区域。
    """
    if len(filtered_groups) == 0 or len(candidate_cores) == 0:
        return filtered_groups, filtered_stats

    accepted_classes = {
        int(stat["class_id"])
        for stat in filtered_stats
    }

    accepted_centers = np.asarray(
        [
            vertices[np.asarray(group, dtype=np.int64)].mean(axis=0)
            for group in filtered_groups
        ],
        dtype=np.float64,
    )
    accepted_median_z = float(
        np.median(accepted_centers[:, 2])
    )

    rescue_candidates = []

    for class_id, info in candidate_cores.items():
        if class_id in accepted_classes:
            continue

        center = np.asarray(
            info["center"],
            dtype=np.float64,
        )
        nearest_distance = float(
            np.min(
                np.linalg.norm(
                    accepted_centers - center,
                    axis=1,
                )
            )
        )
        z_difference = float(
            abs(center[2] - accepted_median_z)
        )
        confidence = float(
            info["mean_confidence"]
        )
        bbox_diag = float(
            info["bbox_diag"]
        )

        valid = (
            confidence
            >= args.missing_core_rescue_confidence
            and nearest_distance
            <= args.missing_core_rescue_distance_mm
            and z_difference
            <= args.missing_core_rescue_z_mm
            and bbox_diag
            <= args.max_component_diag_mm
        )

        rescue_candidates.append(
            (
                valid,
                confidence,
                nearest_distance,
                z_difference,
                bbox_diag,
                int(class_id),
                info,
            )
        )

    rescue_candidates.sort(
        key=lambda item: (
            not item[0],
            -item[1],
            item[2],
            item[3],
        )
    )

    rescued = 0

    for (
        valid,
        confidence,
        nearest_distance,
        z_difference,
        bbox_diag,
        class_id,
        info,
    ) in rescue_candidates:
        if (
            not valid
            or rescued >= args.missing_core_rescue_max
        ):
            continue

        filtered_groups.append(
            np.asarray(
                info["indices"],
                dtype=np.int64,
            )
        )
        filtered_stats.append(
            {
                "class_id": class_id,
                "decision": "missing_tooth_rescue",
                "center_distance": nearest_distance,
                "points": int(
                    len(info["indices"])
                ),
                "confidence": confidence,
            }
        )

        accepted_centers = np.vstack(
            [accepted_centers, info["center"]]
        )
        accepted_classes.add(class_id)
        rescued += 1

        print(
            f"  ✓ 召回漏牙核心 class={class_id}: "
            f"confidence={confidence:.3f}, "
            f"distance={nearest_distance:.2f} mm, "
            f"z_diff={z_difference:.2f} mm, "
            f"diag={bbox_diag:.2f} mm"
        )

    if rescued == 0:
        print("  漏牙核心召回: 无符合条件的候选")
    else:
        print(f"  漏牙核心召回数量: {rescued}")

    return filtered_groups, filtered_stats



def build_robust_tooth_seed_mask(
    classifier,
    vertices,
    normals,
    device,
    graph,
    labels_pass1,
    probs_pass1,
    case_name,
    args,
):
    """
    构造防扩散的牙齿测地种子。

    正常病例：
        Pass1 -> 每牙紧凑核心 -> ROI

    可疑病例：
        Pass1 -> 污染检测
              -> 在完整 mesh 上用另一随机种子重跑
              -> 两次每牙紧凑核心合并/择优
              -> ROI

    返回：
        seed_mask
        detected_class_count
        seed_stats
        retry_payload（未触发时为 None）
        contamination_report
    """
    contamination = analyze_pass1_contamination(
        vertices,
        labels_pass1,
        probs_pass1,
        args,
    )

    first_seed = stable_case_seed(
        case_name,
        args.pass1_seed,
    )

    crown_reference_pass1 = estimate_crown_reference_z(
        vertices,
        labels_pass1,
        probs_pass1,
        args,
    )
    print(
        "  Pass1 牙冠参考 Z: "
        + (
            f"{crown_reference_pass1:.3f}"
            if crown_reference_pass1 is not None
            else "不可用"
        )
    )

    first_cores = extract_compact_seed_cores(
        vertices=vertices,
        labels=labels_pass1,
        probs=probs_pass1,
        args=args,
        case_seed=first_seed,
        source_name="pass1",
        crown_reference_z=crown_reference_pass1,
    )

    retry_payload = None

    if contamination["suspicious"]:
        print("  ⚠️ 检测到第一遍牙齿标签可能向底座扩散：")
        for reason in contamination["reasons"]:
            print(f"    - {reason}")

        retry_seed = stable_case_seed(
            case_name,
            args.retry_pass1_seed,
        )

        (
            labels_retry,
            probs_retry,
            logits_retry,
            valid_retry,
        ) = run_segmentation(
            classifier=classifier,
            vertices=vertices,
            normals=normals,
            device=device,
            num_classes=args.num_classes,
            patch_point=args.npoint,
            expand_times=args.expand_times,
            random_seed=retry_seed,
            stage_name="Pass1-Retry 完整网格复核",
        )

        crown_reference_retry = estimate_crown_reference_z(
            vertices,
            labels_retry,
            probs_retry,
            args,
        )
        print(
            "  Pass1-Retry 牙冠参考 Z: "
            + (
                f"{crown_reference_retry:.3f}"
                if crown_reference_retry is not None
                else "不可用"
            )
        )

        retry_cores = extract_compact_seed_cores(
            vertices=vertices,
            labels=labels_retry,
            probs=probs_retry,
            args=args,
            case_seed=retry_seed,
            source_name="pass1_retry",
            crown_reference_z=crown_reference_retry,
        )

        merged_indices, merged_stats = (
            merge_compact_seed_cores(
                first_cores,
                retry_cores,
                args,
            )
        )

        retry_payload = {
            "labels": labels_retry,
            "probs": probs_retry,
            "logits": logits_retry,
            "valid": valid_retry,
            "seed": retry_seed,
            "merge_stats": merged_stats,
        }

        candidate_cores = {}

        for class_id in sorted(
            set(first_cores) | set(retry_cores)
        ):
            first_candidate = first_cores.get(class_id)
            retry_candidate = retry_cores.get(class_id)

            if first_candidate is None:
                candidate_cores[class_id] = retry_candidate
            elif retry_candidate is None:
                candidate_cores[class_id] = first_candidate
            else:
                candidate_cores[class_id] = (
                    first_candidate
                    if first_candidate["quality"]
                    >= retry_candidate["quality"]
                    else retry_candidate
                )

    else:
        print(
            "  第一遍种子未发现明显大范围扩散，"
            "使用每牙紧凑核心"
        )
        merged_indices = [
            info["indices"]
            for info in first_cores.values()
        ]
        merged_stats = [
            {
                "class_id": class_id,
                "decision": "pass1_compact_core",
                "center_distance": np.nan,
                "points": int(
                    len(info["indices"])
                ),
                "confidence": float(
                    info["mean_confidence"]
                ),
            }
            for class_id, info
            in sorted(first_cores.items())
        ]
        candidate_cores = dict(first_cores)

    merged_indices, merged_stats = (
        filter_seed_groups_by_arch_cluster(
            vertices=vertices,
            seed_groups=merged_indices,
            seed_stats=merged_stats,
            args=args,
        )
    )

    merged_indices, merged_stats = (
        rescue_missing_tooth_cores(
            vertices=vertices,
            filtered_groups=merged_indices,
            filtered_stats=merged_stats,
            candidate_cores=candidate_cores,
            args=args,
        )
    )

    seed_mask = np.zeros(
        len(vertices),
        dtype=bool,
    )

    for indices in merged_indices:
        seed_mask[
            np.asarray(
                indices,
                dtype=np.int64,
            )
        ] = True

    detected_class_count = sum(
        int(item["points"] > 0)
        for item in merged_stats
    )

    print(
        f"  紧凑牙齿种子: "
        f"类别数={detected_class_count}, "
        f"点数={int(seed_mask.sum())}"
    )

    return (
        seed_mask,
        detected_class_count,
        merged_stats,
        retry_payload,
        contamination,
    )


def multisource_dijkstra_fallback(
    graph,
    source_indices,
    limit,
):
    """
    在 scipy 版本不支持 min_only=True 时使用的多源 Dijkstra。
    """
    number_of_vertices = graph.shape[0]

    distances = np.full(
        number_of_vertices,
        np.inf,
        dtype=np.float64,
    )

    queue = []

    for source in source_indices:
        source = int(source)
        distances[source] = 0.0
        heapq.heappush(
            queue,
            (0.0, source),
        )

    indptr = graph.indptr
    indices = graph.indices
    data = graph.data

    while queue:
        current_distance, vertex = (
            heapq.heappop(queue)
        )

        if current_distance != distances[vertex]:
            continue

        if current_distance > limit:
            continue

        start = indptr[vertex]
        end = indptr[vertex + 1]

        for edge_index in range(start, end):
            neighbor = indices[edge_index]
            new_distance = (
                current_distance + data[edge_index]
            )

            if (
                new_distance < distances[neighbor]
                and new_distance <= limit
            ):
                distances[neighbor] = new_distance
                heapq.heappush(
                    queue,
                    (new_distance, neighbor),
                )

    return distances


def geodesic_distance_to_seeds(
    graph,
    seed_indices,
    limit,
):
    seed_indices = np.asarray(
        seed_indices,
        dtype=np.int64,
    )

    if len(seed_indices) == 0:
        raise ValueError("没有牙齿种子，无法计算测地距离")

    try:
        distances = dijkstra(
            graph,
            directed=False,
            indices=seed_indices,
            min_only=True,
            limit=limit,
            return_predecessors=False,
        )

        distances = np.asarray(
            distances
        ).reshape(-1)

    except TypeError:
        print(
            "  当前 scipy 不支持 min_only=True，"
            "使用 Python 多源 Dijkstra"
        )

        distances = multisource_dijkstra_fallback(
            graph,
            seed_indices,
            limit,
        )

    return distances


def build_geodesic_roi(
    graph,
    seed_mask,
    initial_margin,
    max_margin,
    min_roi_ratio,
    max_roi_ratio,
):
    """
    从牙齿种子沿网格表面扩张，保留牙齿和附近牙龈。

    ROI 过小时，每次增加 2 mm，直到达到安全比例或最大 margin。
    """
    seed_indices = np.flatnonzero(
        seed_mask
    )

    margin = float(initial_margin)

    while True:
        distances = geodesic_distance_to_seeds(
            graph,
            seed_indices,
            limit=margin,
        )

        roi_mask = np.isfinite(distances) & (
            distances <= margin
        )

        roi_ratio = float(
            roi_mask.mean()
        )

        print(
            f"  测地 ROI: margin={margin:.1f} mm, "
            f"保留={roi_mask.sum()}/{len(roi_mask)} "
            f"({roi_ratio * 100:.2f}%)"
        )

        if (
            roi_ratio >= min_roi_ratio
            or margin >= max_margin
        ):
            break

        margin = min(
            margin + 2.0,
            max_margin,
        )

    if roi_ratio > max_roi_ratio:
        print(
            f"  ⚠️ ROI 保留比例较高 "
            f"({roi_ratio * 100:.2f}%)，"
            "该病例底座可能较薄，或第一遍误标较多"
        )

    return roi_mask, distances, margin, roi_ratio




def is_upper_case(case_name):
    lower_name = case_name.lower()
    return (
        "upper" in lower_name
        or "_u" in lower_name
    )


def add_upper_palate_roi(
    vertices,
    seed_mask,
    roi_mask,
    detected_class_count,
    hull_scale,
    palate_depth_mm,
    seed_z_quantile,
    min_seed_points,
    min_detected_classes,
    max_added_ratio,
):
    """
    为上颌病例保留牙弓内部的中央腭部。

    原测地 ROI 只保留“距离牙齿表面若干毫米”的区域。
    上颌中央腭部可能距离两侧牙列超过 gingiva_margin_mm，
    因而会被错误删除。这里使用可靠牙齿种子的 XY 凸包描述
    牙弓内部区域，再通过 Z 下限排除明显更低的厚底座。

    该规则只增加 ROI，不改变第一遍标签。
    """
    vertices = np.asarray(vertices)
    seed_indices = np.flatnonzero(seed_mask)

    empty_mask = np.zeros(
        len(vertices),
        dtype=bool,
    )

    if (
        len(seed_indices) < min_seed_points
        or detected_class_count < min_detected_classes
    ):
        print(
            "  上颌腭部保留: 可靠种子或牙齿类别不足，跳过"
        )
        return roi_mask.copy(), empty_mask, {
            "applied": False,
            "reason": "insufficient_seed",
            "palate_z_min": np.nan,
            "inside_hull_points": 0,
            "added_points": 0,
        }

    seed_xy = vertices[
        seed_indices,
        :2,
    ]

    # 去除完全重复 XY 点，降低 QHull 数值问题。
    seed_xy = np.unique(
        np.round(seed_xy, decimals=5),
        axis=0,
    )

    if len(seed_xy) < 3:
        print(
            "  上颌腭部保留: XY 种子不足以构成凸包，跳过"
        )
        return roi_mask.copy(), empty_mask, {
            "applied": False,
            "reason": "degenerate_xy",
            "palate_z_min": np.nan,
            "inside_hull_points": 0,
            "added_points": 0,
        }

    try:
        hull = ConvexHull(seed_xy)
        hull_xy = seed_xy[
            hull.vertices
        ]

        hull_center = hull_xy.mean(
            axis=0
        )
        hull_xy_scaled = (
            hull_center
            + (
                hull_xy - hull_center
            ) * float(hull_scale)
        )

        hull_test = Delaunay(
            hull_xy_scaled
        )
        inside_hull = (
            hull_test.find_simplex(
                vertices[:, :2]
            )
            >= 0
        )

    except QhullError as exc:
        print(
            f"  上颌腭部保留: QHull 失败，跳过 ({exc})"
        )
        return roi_mask.copy(), empty_mask, {
            "applied": False,
            "reason": "qhull_error",
            "palate_z_min": np.nan,
            "inside_hull_points": 0,
            "added_points": 0,
        }

    seed_z_reference = float(
        np.quantile(
            vertices[seed_indices, 2],
            seed_z_quantile,
        )
    )
    palate_z_min = (
        seed_z_reference
        - float(palate_depth_mm)
    )

    palate_candidate = (
        inside_hull
        & (
            vertices[:, 2]
            >= palate_z_min
        )
    )

    added_mask = (
        palate_candidate
        & (~roi_mask)
    )

    max_added_points = int(
        np.floor(
            len(vertices)
            * float(max_added_ratio)
        )
    )

    # 安全保护：若异常凸包加入过多点，只保留候选中 Z 较高的部分。
    if (
        max_added_points > 0
        and added_mask.sum() > max_added_points
    ):
        added_indices = np.flatnonzero(
            added_mask
        )
        added_z = vertices[
            added_indices,
            2,
        ]

        keep_order = np.argsort(
            added_z
        )[::-1][
            :max_added_points
        ]

        safe_added_mask = np.zeros_like(
            added_mask
        )
        safe_added_mask[
            added_indices[keep_order]
        ] = True
        added_mask = safe_added_mask

        print(
            "  ⚠️ 上颌腭部候选过大，"
            f"按 Z 高位限制为 {max_added_points} 点"
        )

    roi_with_palate = (
        roi_mask
        | added_mask
    )

    print(
        "  上颌腭部保留:"
        f" hull内={int(inside_hull.sum())},"
        f" Z下限={palate_z_min:.3f},"
        f" 新增={int(added_mask.sum())} 点"
        f" ({added_mask.mean() * 100:.2f}%)"
    )

    stats = {
        "applied": True,
        "reason": "ok",
        "palate_z_min": float(
            palate_z_min
        ),
        "inside_hull_points": int(
            inside_hull.sum()
        ),
        "added_points": int(
            added_mask.sum()
        ),
    }

    return (
        roi_with_palate,
        added_mask,
        stats,
    )



def filter_geodesic_roi_components(
    graph,
    roi_mask,
    seed_mask,
    vertices,
    min_vertices,
    min_seed_points,
    relative_vertex_ratio,
    relative_seed_ratio,
    preserve_seed_fraction,
):
    """
    删除测地 ROI 中孤立的小连通块。

    这是针对如下情况：
        第一遍某牙号在一个远离牙弓的独立底座块上仍残留少量种子，
        导致该独立块被测地 ROI 保留。

    保留原则：
    1. 种子最多的主连通块始终保留；
    2. 其他块必须同时具有足够顶点和足够可靠种子；
    3. 若过滤后丢失太多可靠种子，则按种子数逐步放宽，
       保护拓扑本身断开的正常牙弓。
    """
    roi_indices = np.flatnonzero(roi_mask)

    if len(roi_indices) == 0:
        raise ValueError("ROI 为空，无法进行连通块过滤")

    roi_subgraph = graph[
        roi_indices
    ][:, roi_indices]

    component_count, component_labels = connected_components(
        roi_subgraph,
        directed=False,
        return_labels=True,
    )

    if component_count <= 1:
        print("  ROI 连通块过滤: 仅 1 个连通块，无需处理")
        stats = {
            "component_count_before": 1,
            "component_count_after": 1,
            "vertex_counts": np.asarray(
                [len(roi_indices)],
                dtype=np.int64,
            ),
            "seed_counts": np.asarray(
                [int(seed_mask[roi_indices].sum())],
                dtype=np.int64,
            ),
            "keep_components": np.asarray(
                [True],
                dtype=bool,
            ),
        }
        return roi_mask.copy(), stats

    vertex_counts = np.bincount(
        component_labels,
        minlength=component_count,
    ).astype(np.int64)

    seed_counts = np.zeros(
        component_count,
        dtype=np.int64,
    )

    component_centroids = np.zeros(
        (component_count, 3),
        dtype=np.float64,
    )

    for component_id in range(component_count):
        local_mask = (
            component_labels == component_id
        )
        full_indices = roi_indices[
            local_mask
        ]

        seed_counts[component_id] = int(
            seed_mask[full_indices].sum()
        )
        component_centroids[component_id] = (
            vertices[full_indices].mean(axis=0)
        )

    main_component = max(
        range(component_count),
        key=lambda cid: (
            int(seed_counts[cid]),
            int(vertex_counts[cid]),
        ),
    )

    max_vertex_count = int(
        vertex_counts.max()
    )
    max_seed_count = int(
        seed_counts.max()
    )

    vertex_threshold = max(
        int(min_vertices),
        int(
            np.ceil(
                max_vertex_count
                * float(relative_vertex_ratio)
            )
        ),
    )

    seed_threshold = max(
        int(min_seed_points),
        int(
            np.ceil(
                max_seed_count
                * float(relative_seed_ratio)
            )
        ),
    )

    keep_components = np.zeros(
        component_count,
        dtype=bool,
    )
    keep_components[main_component] = True

    for component_id in range(component_count):
        if component_id == main_component:
            continue

        if (
            vertex_counts[component_id]
            >= vertex_threshold
            and seed_counts[component_id]
            >= seed_threshold
        ):
            keep_components[component_id] = True

    total_seed_points = int(
        seed_mask[roi_indices].sum()
    )
    kept_seed_points = int(
        seed_counts[keep_components].sum()
    )

    if total_seed_points > 0:
        kept_seed_fraction = (
            kept_seed_points
            / total_seed_points
        )
    else:
        kept_seed_fraction = 1.0

    # 若严格过滤丢失了较多可靠种子，则按种子数由高到低恢复组件。
    if kept_seed_fraction < preserve_seed_fraction:
        candidate_order = np.argsort(
            seed_counts
        )[::-1]

        for component_id in candidate_order:
            if keep_components[component_id]:
                continue
            if seed_counts[component_id] <= 0:
                continue

            keep_components[component_id] = True
            kept_seed_points += int(
                seed_counts[component_id]
            )
            kept_seed_fraction = (
                kept_seed_points
                / max(total_seed_points, 1)
            )

            if (
                kept_seed_fraction
                >= preserve_seed_fraction
            ):
                break

    filtered_roi_mask = np.zeros_like(
        roi_mask,
        dtype=bool,
    )

    kept_local_mask = keep_components[
        component_labels
    ]
    filtered_roi_mask[
        roi_indices[kept_local_mask]
    ] = True

    print(
        f"  ROI 连通块过滤: "
        f"{component_count} -> "
        f"{int(keep_components.sum())} 个连通块"
    )
    print(
        f"    顶点阈值={vertex_threshold}, "
        f"种子阈值={seed_threshold}, "
        f"保留种子={kept_seed_points}/"
        f"{total_seed_points} "
        f"({kept_seed_fraction * 100:.2f}%)"
    )

    for component_id in range(component_count):
        status = (
            "KEEP"
            if keep_components[component_id]
            else "DROP"
        )
        centroid = component_centroids[
            component_id
        ]
        print(
            f"    component {component_id:02d}: "
            f"{status}, "
            f"vertices={vertex_counts[component_id]}, "
            f"seeds={seed_counts[component_id]}, "
            f"center=[{centroid[0]:.1f}, "
            f"{centroid[1]:.1f}, "
            f"{centroid[2]:.1f}]"
        )

    stats = {
        "component_count_before": int(
            component_count
        ),
        "component_count_after": int(
            keep_components.sum()
        ),
        "vertex_counts": vertex_counts,
        "seed_counts": seed_counts,
        "keep_components": keep_components,
    }

    return filtered_roi_mask, stats



def subset_mesh_from_vertex_roi(
    mesh,
    roi_mask,
):
    """
    只保留三个顶点都位于 ROI 内的三角面。

    牙龈保留量已经由 gingiva_margin_mm 控制，因此这里不再使用
    “任意一个顶点位于 ROI 就保留整张三角面”的宽松规则，
    避免把 ROI 外的小块、长三角面和尖刺重新带回第二遍输入。

    返回：
        cropped_mesh
        full_vertex_indices：cropped mesh 顶点对应完整 aligned mesh 的索引
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    keep_face_mask = np.all(
        roi_mask[faces],
        axis=1,
    )

    kept_faces_old = faces[
        keep_face_mask
    ]

    if len(kept_faces_old) == 0:
        raise ValueError("ROI 裁剪后没有三角面")

    full_vertex_indices = np.unique(
        kept_faces_old.reshape(-1)
    )

    old_to_new = np.full(
        len(vertices),
        -1,
        dtype=np.int64,
    )

    old_to_new[full_vertex_indices] = np.arange(
        len(full_vertex_indices),
        dtype=np.int64,
    )

    kept_faces_new = old_to_new[
        kept_faces_old
    ]

    cropped_mesh = trimesh.Trimesh(
        vertices=vertices[full_vertex_indices],
        faces=kept_faces_new,
        process=False,
    )

    return cropped_mesh, full_vertex_indices


def load_decoder(
    decoder_root,
    case_name,
    num_classes,
):
    lower_case = case_name.lower()

    if "upper" in lower_case or "_u" in lower_case:
        decoder_path = (
            decoder_root + "upper.json"
        )
    elif "lower" in lower_case or "_l" in lower_case:
        decoder_path = (
            decoder_root + "lower.json"
        )
    else:
        decoder_path = None

    if decoder_path is None:
        print(
            "  ⚠️ 无法判断上下颌，JSON 保留数字类别"
        )
        return {
            str(k): k
            for k in range(num_classes)
        }

    with open(
        decoder_path,
        "r",
        encoding="utf-8",
    ) as decoder_file:
        return json.load(
            decoder_file
        )["label"]


def labels_to_colors(
    labels,
    color_dict,
):
    colors = []

    for label in labels:
        key = str(int(label))
        colors.append(
            color_dict.get(
                key,
                [128, 128, 128],
            )
        )

    return (
        np.asarray(
            colors,
            dtype=np.float64,
        )
        / 255.0
    )


def save_colored_mesh(
    path,
    vertices,
    faces,
    labels,
    color_dict,
):
    vertices = np.asarray(
        vertices,
        dtype=np.float64,
    )
    faces = np.asarray(
        faces,
        dtype=np.int32,
    )
    labels = np.asarray(labels)

    if len(vertices) != len(labels):
        raise ValueError(
            f"保存 PLY 时顶点数与标签数不一致："
            f"{len(vertices)} vs {len(labels)}"
        )

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = (
        o3d.utility.Vector3dVector(vertices)
    )
    mesh_o3d.triangles = (
        o3d.utility.Vector3iVector(faces)
    )
    mesh_o3d.vertex_colors = (
        o3d.utility.Vector3dVector(
            labels_to_colors(
                labels,
                color_dict,
            )
        )
    )
    mesh_o3d.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(
        str(path),
        mesh_o3d,
        write_vertex_colors=True,
    )

    if not success:
        raise RuntimeError(
            f"PLY 保存失败：{path}"
        )


def save_roi_debug_mesh(
    path,
    mesh,
    seed_mask,
    roi_mask,
):
    """
    颜色：
        红色：可靠牙齿种子
        绿色：测地扩张后保留的牙龈/ROI
        深灰：将被排除的远端底座
    """
    vertex_count = len(mesh.vertices)

    colors = np.tile(
        np.asarray(
            [70, 70, 70],
            dtype=np.float64,
        ),
        (vertex_count, 1),
    )

    colors[roi_mask] = [
        0,
        220,
        0,
    ]

    colors[seed_mask] = [
        255,
        0,
        0,
    ]

    colors /= 255.0

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = (
        o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices).astype(
                np.float64
            )
        )
    )
    mesh_o3d.triangles = (
        o3d.utility.Vector3iVector(
            np.asarray(mesh.faces).astype(
                np.int32
            )
        )
    )
    mesh_o3d.vertex_colors = (
        o3d.utility.Vector3dVector(colors)
    )
    mesh_o3d.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        str(path),
        mesh_o3d,
        write_vertex_colors=True,
    )



def save_component_filter_debug_mesh(
    path,
    mesh,
    seed_mask,
    roi_before,
    roi_after,
):
    """
    颜色：
        红色：可靠牙齿种子
        绿色：连通块过滤后保留的最终 ROI
        橙色：原测地 ROI 中被连通块过滤删除的孤立块
        深灰：从未进入测地 ROI 的区域
    """
    vertex_count = len(mesh.vertices)

    colors = np.tile(
        np.asarray(
            [65, 65, 65],
            dtype=np.float64,
        ),
        (vertex_count, 1),
    )

    removed_component_mask = (
        roi_before
        & (~roi_after)
    )

    colors[removed_component_mask] = [
        255,
        145,
        0,
    ]
    colors[roi_after] = [
        0,
        220,
        0,
    ]
    colors[seed_mask] = [
        255,
        0,
        0,
    ]

    colors /= 255.0

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(
        np.asarray(
            mesh.vertices
        ).astype(np.float64)
    )
    mesh_o3d.triangles = o3d.utility.Vector3iVector(
        np.asarray(
            mesh.faces
        ).astype(np.int32)
    )
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        colors
    )
    mesh_o3d.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(
        str(path),
        mesh_o3d,
        write_vertex_colors=True,
    )

    if not success:
        raise RuntimeError(
            f"连通块 debug PLY 保存失败：{path}"
        )




def save_upper_palate_debug_mesh(
    path,
    mesh,
    seed_mask,
    geodesic_roi_mask,
    palate_added_mask,
    final_roi_mask,
):
    """
    颜色：
        红色：可靠牙齿种子
        绿色：原测地 ROI
        青色：上颌腭部规则额外加入的区域
        深灰：最终未保留区域
    """
    vertex_count = len(mesh.vertices)

    colors = np.tile(
        np.asarray(
            [65, 65, 65],
            dtype=np.float64,
        ),
        (vertex_count, 1),
    )

    colors[geodesic_roi_mask] = [
        0,
        210,
        0,
    ]
    colors[
        palate_added_mask
        & final_roi_mask
    ] = [
        0,
        210,
        255,
    ]
    colors[seed_mask] = [
        255,
        0,
        0,
    ]

    colors /= 255.0

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(
        np.asarray(
            mesh.vertices
        ).astype(np.float64)
    )
    mesh_o3d.triangles = o3d.utility.Vector3iVector(
        np.asarray(
            mesh.faces
        ).astype(np.int32)
    )
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        colors
    )
    mesh_o3d.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(
        str(path),
        mesh_o3d,
        write_vertex_colors=True,
    )

    if not success:
        raise RuntimeError(
            f"上颌腭部 debug PLY 保存失败：{path}"
        )




def score_alignment_candidate(
    vertices,
    labels,
    probs,
    args,
    case_seed,
):
    """
    用 Pass1 的分割结构评估 align_scan 候选。

    评分不要求 Pass1 完全正确，只判断它是否像一个合理的牙弓分割：
    - 检出的牙号数量；
    - 牙齿点置信度；
    - 每牙紧凑核心数量与质量；
    - 单个标签是否异常扩散；
    - 各牙齿核心是否形成空间连贯的牙弓。

    得分只用于同一个病例的多个对齐候选之间排序。
    """
    vertices = np.asarray(vertices)
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    contamination = analyze_pass1_contamination(
        vertices,
        labels,
        probs,
        args,
    )

    crown_reference_z = estimate_crown_reference_z(
        vertices,
        labels,
        probs,
        args,
    )

    compact_cores = extract_compact_seed_cores(
        vertices=vertices,
        labels=labels,
        probs=probs,
        args=args,
        case_seed=case_seed,
        source_name="align_candidate",
        crown_reference_z=crown_reference_z,
    )

    reliable_classes = []
    class_diags = []
    class_ratios = []
    class_confidences = []

    for class_id in range(1, args.num_classes):
        class_mask = (
            (labels == class_id)
            & (
                probs[:, class_id]
                >= max(
                    args.fallback_seed_confidence,
                    0.30,
                )
            )
        )
        indices = np.flatnonzero(class_mask)

        if len(indices) < args.core_min_points:
            continue

        reliable_classes.append(class_id)
        class_ratios.append(
            len(indices) / len(vertices)
        )
        class_diags.append(
            float(
                np.linalg.norm(
                    np.ptp(
                        vertices[indices],
                        axis=0,
                    )
                )
            )
        )
        class_confidences.append(
            float(
                probs[indices, class_id].mean()
            )
        )

    class_count = len(reliable_classes)
    core_count = len(compact_cores)

    tooth_mask = labels > 0
    tooth_ratio = float(tooth_mask.mean())

    if tooth_mask.any():
        predicted_confidence = probs[
            np.arange(len(labels)),
            labels,
        ]
        mean_tooth_confidence = float(
            predicted_confidence[
                tooth_mask
            ].mean()
        )
    else:
        mean_tooth_confidence = 0.0

    if class_confidences:
        median_class_confidence = float(
            np.median(class_confidences)
        )
    else:
        median_class_confidence = 0.0

    if compact_cores:
        core_qualities = np.asarray(
            [
                info["quality"]
                for info in compact_cores.values()
            ],
            dtype=np.float64,
        )
        median_core_quality = float(
            np.median(core_qualities)
        )

        core_centers = np.asarray(
            [
                info["center"]
                for info in compact_cores.values()
            ],
            dtype=np.float64,
        )

        if len(core_centers) >= 2:
            pairwise = np.linalg.norm(
                core_centers[:, None, :]
                - core_centers[None, :, :],
                axis=2,
            )
            pairwise += np.eye(
                len(core_centers)
            ) * 1e9
            nearest = pairwise.min(axis=1)
            isolated_core_count = int(
                np.sum(
                    nearest
                    > args.arch_core_link_mm
                )
            )
            median_nearest_distance = float(
                np.median(nearest)
            )
        else:
            isolated_core_count = 1
            median_nearest_distance = np.inf

    else:
        median_core_quality = 0.0
        isolated_core_count = 16
        median_nearest_distance = np.inf

    expansion_penalty = 0.0

    for diag in class_diags:
        expansion_penalty += max(
            diag - args.retry_label_diag_mm,
            0.0,
        ) / 6.0

    for ratio in class_ratios:
        expansion_penalty += 25.0 * max(
            ratio - args.retry_label_ratio,
            0.0,
        )

    # 牙号数量在 12~16 附近已经足够，不继续线性奖励，
    # 防止通过过度误标得到更高分。
    class_reward = min(
        class_count,
        14,
    )
    core_reward = min(
        core_count,
        14,
    )

    score = (
        5.0 * class_reward
        + 3.0 * core_reward
        + 24.0 * mean_tooth_confidence
        + 18.0 * median_class_confidence
        + 4.0 * median_core_quality
        - 7.0 * isolated_core_count
        - 5.0 * len(
            contamination["reasons"]
        )
        - 4.0 * expansion_penalty
        - 45.0 * max(
            tooth_ratio
            - args.retry_total_tooth_ratio,
            0.0,
        )
    )

    if class_count < args.min_detected_tooth_classes:
        score -= 35.0

    if core_count < args.min_detected_tooth_classes:
        score -= 25.0

    # 同病例多对齐候选中，漏掉多个牙号的候选通常不应胜出。
    score -= 4.0 * max(12 - class_count, 0)
    score -= 3.0 * max(12 - core_count, 0)

    details = {
        "score": float(score),
        "class_count": int(class_count),
        "core_count": int(core_count),
        "mean_tooth_confidence": float(
            mean_tooth_confidence
        ),
        "median_class_confidence": float(
            median_class_confidence
        ),
        "median_core_quality": float(
            median_core_quality
        ),
        "isolated_core_count": int(
            isolated_core_count
        ),
        "median_nearest_core_distance": float(
            median_nearest_distance
        ),
        "tooth_ratio": float(tooth_ratio),
        "contamination_reason_count": int(
            len(contamination["reasons"])
        ),
        "expansion_penalty": float(
            expansion_penalty
        ),
    }

    return float(score), details


def choose_best_alignment_and_pass1(
    mesh_raw,
    case_name,
    classifier,
    device,
    color_dict,
    debug_dir,
    args,
):
    """
    自动执行多次 align_scan + Pass1，并选择分割质量最高的候选。

    这相当于把“人工反复运行直到出现标准分割”变成：
        固定候选集合 -> 自动质量评分 -> 确定性选择最佳结果。

    返回的 Pass1 直接进入后续 ROI 流程，不会再重复运行。
    """
    if args.align_trials < 1:
        raise ValueError("--align_trials 必须 >= 1")

    pass1_seed = stable_case_seed(
        case_name,
        args.pass1_seed,
    )

    best = None
    candidate_records = []

    print(
        f"  开始多起点对齐: trials={args.align_trials}, "
        f"sample_num={args.align_sample_num}"
    )

    for trial_index in range(
        args.align_trials
    ):
        trial_base_seed = (
            args.align_seed
            + trial_index
            * args.align_seed_stride
        )

        try:
            mesh_candidate, align_seed = (
                align_with_fixed_seed(
                    mesh_raw.copy(),
                    case_name,
                    trial_base_seed,
                    sample_num=(
                        args.align_sample_num
                    ),
                )
            )

            vertices_candidate = np.asarray(
                mesh_candidate.vertices
            )
            normals_candidate = np.asarray(
                mesh_candidate.vertex_normals
            )

            if (
                len(vertices_candidate) == 0
                or vertices_candidate.shape
                != normals_candidate.shape
            ):
                raise ValueError(
                    "对齐后顶点或法向无效"
                )

            (
                labels_candidate,
                probs_candidate,
                logits_candidate,
                valid_candidate,
            ) = run_segmentation(
                classifier=classifier,
                vertices=vertices_candidate,
                normals=normals_candidate,
                device=device,
                num_classes=args.num_classes,
                patch_point=args.npoint,
                expand_times=args.expand_times,
                random_seed=pass1_seed,
                stage_name=(
                    f"Align候选 {trial_index + 1}/"
                    f"{args.align_trials} Pass1"
                ),
            )

            score, details = (
                score_alignment_candidate(
                    vertices=vertices_candidate,
                    labels=labels_candidate,
                    probs=probs_candidate,
                    args=args,
                    case_seed=(
                        align_seed
                        + pass1_seed
                    ),
                )
            )

            record = {
                "trial_index": int(
                    trial_index
                ),
                "align_seed": int(
                    align_seed
                ),
                **details,
            }
            candidate_records.append(record)

            print(
                f"  Align候选 {trial_index}: "
                f"seed={align_seed}, "
                f"score={score:.3f}, "
                f"classes={details['class_count']}, "
                f"cores={details['core_count']}, "
                f"confidence="
                f"{details['mean_tooth_confidence']:.3f}, "
                f"isolated="
                f"{details['isolated_core_count']}, "
                f"contamination="
                f"{details['contamination_reason_count']}"
            )

            if (
                args.save_debug
                and args.save_align_candidates
            ):
                safe_score = (
                    f"{score:.2f}"
                    .replace("-", "m")
                    .replace(".", "p")
                )
                save_colored_mesh(
                    debug_dir
                    / (
                        f"{case_name}_align_candidate_"
                        f"{trial_index:02d}_"
                        f"score_{safe_score}.ply"
                    ),
                    vertices_candidate,
                    np.asarray(
                        mesh_candidate.faces
                    ),
                    labels_candidate,
                    color_dict,
                )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "score": float(score),
                    "trial_index": int(
                        trial_index
                    ),
                    "align_seed": int(
                        align_seed
                    ),
                    "mesh": mesh_candidate,
                    "vertices": vertices_candidate,
                    "normals": normals_candidate,
                    "labels": labels_candidate,
                    "probs": probs_candidate,
                    "logits": logits_candidate,
                    "valid": valid_candidate,
                    "details": details,
                }

        except Exception as exc:
            print(
                f"  ⚠️ Align候选 {trial_index} 失败: "
                f"{exc}"
            )
            tb.print_exc()
            candidate_records.append(
                {
                    "trial_index": int(
                        trial_index
                    ),
                    "align_seed": None,
                    "score": -np.inf,
                    "error": str(exc),
                }
            )
            continue

    if best is None:
        raise RuntimeError(
            "所有 align_scan 候选均失败"
        )

    print(
        "  ✓ 选择最佳对齐: "
        f"trial={best['trial_index']}, "
        f"seed={best['align_seed']}, "
        f"score={best['score']:.3f}"
    )

    return (
        best["mesh"],
        best["align_seed"],
        pass1_seed,
        best["labels"],
        best["probs"],
        best["logits"],
        best["valid"],
        best["trial_index"],
        best["score"],
        candidate_records,
    )



def test(args):
    asset_root = Path(os.environ["DADSN_ASSET_ROOT"]).expanduser().resolve()
    decoder_root = str(asset_root / "label_c17" / "label_decoder_")

    with open(
        asset_root / "label_c17" / "color_dict.json",
        "r",
        encoding="utf-8",
    ) as color_file:
        color_dict = json.load(color_file)



    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.save_debug:
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        print(f"调试输出目录: {debug_dir}")
    else:
        print("调试输出已通过 --no_debug 关闭")

    file_list = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower()
        in {".stl", ".ply", ".obj"}
    )

    print(f"找到 {len(file_list)} 个网格文件")

    # GPU
    print("\n=== GPU 诊断信息 ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', '未设置')}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Grand Challenge GPU runtime requires CUDA, but no CUDA device is visible. "
            "Bite2Text does not fall back to CPU for DADSN retry."
        )
    device = torch.device("cuda:0")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 使用GPU: {device}")

    model_load_started = time.perf_counter()
    classifier = DeformableTeethNet(
        num_classes=args.num_classes,
        k=args.k,
        emb_dims=args.emb_dims,
    ).to(device)

    model_path = asset_root / "outputs" / args.exp_name / "models" / "best_model.pth"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    classifier.load_state_dict(
        checkpoint["state_dict"]
    )
    classifier.eval()
    print(f"B2T_MODEL_LOAD dadsn_retry count=1 seconds={time.perf_counter() - model_load_started:.6f}", flush=True)

    print(
        f"模型所在设备: "
        f"{next(classifier.parameters()).device}"
    )

    with torch.no_grad():
        for file_index, mesh_path in enumerate(
            file_list,
            start=1,
        ):
            case_name = mesh_path.stem

            print(
                f"\n[{file_index}/{len(file_list)}] "
                f"Processing {case_name}"
            )

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # 与原 predict 一致，默认 process=True。
                mesh_raw = trimesh.load(
                    str(mesh_path),
                    force="mesh",
                )

                if not isinstance(
                    mesh_raw,
                    trimesh.Trimesh,
                ):
                    print("  ✗ 不是有效 Trimesh，跳过")
                    continue

                if (
                    mesh_raw.vertices is None
                    or len(mesh_raw.vertices) < 100
                    or len(mesh_raw.faces) == 0
                ):
                    print("  ✗ 网格为空或过小，跳过")
                    continue

                raw_vertices = np.asarray(
                    mesh_raw.vertices
                ).copy()

                print(
                    f"  原始网格: "
                    f"{len(mesh_raw.vertices)} 顶点, "
                    f"{len(mesh_raw.faces)} 面"
                )

                # --------------------------------------------------
                # 多次 align_scan + Pass1，自动选择质量最高的对齐。
                # 选中的 Pass1 直接进入后续流程。
                # --------------------------------------------------
                try:
                    (
                        mesh_aligned,
                        align_seed,
                        pass1_seed,
                        labels_pass1,
                        probs_pass1,
                        logits_pass1,
                        valid_pass1,
                        chosen_align_trial,
                        chosen_align_score,
                        align_candidate_records,
                    ) = choose_best_alignment_and_pass1(
                        mesh_raw=mesh_raw,
                        case_name=case_name,
                        classifier=classifier,
                        device=device,
                        color_dict=color_dict,
                        debug_dir=debug_dir,
                        args=args,
                    )
                except Exception as exc:
                    print(
                        f"  ✗ 多起点对齐选择失败: {exc}"
                    )
                    tb.print_exc()
                    continue

                aligned_vertices = np.asarray(
                    mesh_aligned.vertices
                )
                aligned_normals = np.asarray(
                    mesh_aligned.vertex_normals
                )

                if (
                    len(aligned_vertices) == 0
                    or aligned_vertices.shape
                    != aligned_normals.shape
                ):
                    print(
                        "  ✗ 最佳对齐顶点/法向无效，跳过"
                    )
                    continue

                if (
                    len(aligned_vertices)
                    != len(raw_vertices)
                ):
                    print(
                        "  ⚠️ align_scan 改变了顶点数量，"
                        "raw 坐标映射将不可用"
                    )
                    raw_vertices = None

                print(
                    f"  最佳对齐后: "
                    f"{len(aligned_vertices)} 顶点, "
                    f"{len(mesh_aligned.faces)} 面, "
                    f"trial={chosen_align_trial}, "
                    f"seed={align_seed}, "
                    f"score={chosen_align_score:.3f}"
                )

                # --------------------------------------------------
                # 网格图 + 可靠牙齿种子
                # --------------------------------------------------
                print("  构建网格邻接图...")
                graph = build_mesh_graph(
                    aligned_vertices,
                    np.asarray(
                        mesh_aligned.faces
                    ),
                )

                (
                    seed_mask,
                    detected_class_count,
                    class_stats,
                    retry_payload,
                    contamination_report,
                ) = build_robust_tooth_seed_mask(
                    classifier=classifier,
                    vertices=aligned_vertices,
                    normals=aligned_normals,
                    device=device,
                    graph=graph,
                    labels_pass1=labels_pass1,
                    probs_pass1=probs_pass1,
                    case_name=case_name,
                    args=args,
                )

                used_confidence = args.seed_confidence

                if (
                    detected_class_count
                    < args.min_detected_tooth_classes
                    or seed_mask.sum()
                    < args.min_component_points
                ):
                    print(
                        "  ⚠️ 第一遍可靠牙齿种子不足，"
                        "本病例第二遍不裁剪，只更换随机种子重跑"
                    )

                    roi_mask = np.ones(
                        len(aligned_vertices),
                        dtype=bool,
                    )
                    geodesic_distances = np.zeros(
                        len(aligned_vertices),
                        dtype=np.float32,
                    )
                    used_margin = 0.0
                    roi_ratio = 1.0
                    crop_fallback = True

                else:
                    (
                        roi_mask,
                        geodesic_distances,
                        used_margin,
                        roi_ratio,
                    ) = build_geodesic_roi(
                        graph=graph,
                        seed_mask=seed_mask,
                        initial_margin=(
                            args.gingiva_margin_mm
                        ),
                        max_margin=args.max_margin_mm,
                        min_roi_ratio=args.min_roi_ratio,
                        max_roi_ratio=args.max_roi_ratio,
                    )
                    crop_fallback = False

                geodesic_roi_mask = (
                    roi_mask.copy()
                )

                palate_added_mask = np.zeros(
                    len(aligned_vertices),
                    dtype=bool,
                )
                palate_stats = {
                    "applied": False,
                    "reason": "not_upper_or_disabled",
                    "palate_z_min": np.nan,
                    "inside_hull_points": 0,
                    "added_points": 0,
                }

                if (
                    (not crop_fallback)
                    and args.preserve_upper_palate
                    and is_upper_case(case_name)
                ):
                    (
                        roi_mask,
                        palate_added_mask,
                        palate_stats,
                    ) = add_upper_palate_roi(
                        vertices=aligned_vertices,
                        seed_mask=seed_mask,
                        roi_mask=roi_mask,
                        detected_class_count=(
                            detected_class_count
                        ),
                        hull_scale=(
                            args.palate_hull_scale
                        ),
                        palate_depth_mm=(
                            args.palate_depth_mm
                        ),
                        seed_z_quantile=(
                            args.palate_seed_z_quantile
                        ),
                        min_seed_points=(
                            args.palate_min_seed_points
                        ),
                        min_detected_classes=(
                            args.palate_min_detected_classes
                        ),
                        max_added_ratio=(
                            args.palate_max_added_ratio
                        ),
                    )

                roi_mask_before_component_filter = (
                    roi_mask.copy()
                )

                if crop_fallback:
                    component_filter_stats = {
                        "component_count_before": 0,
                        "component_count_after": 0,
                        "vertex_counts": np.empty(
                            (0,),
                            dtype=np.int64,
                        ),
                        "seed_counts": np.empty(
                            (0,),
                            dtype=np.int64,
                        ),
                        "keep_components": np.empty(
                            (0,),
                            dtype=bool,
                        ),
                    }
                    print(
                        "  ROI 连通块过滤: fallback 模式，"
                        "保持完整网格"
                    )
                else:
                    (
                        roi_mask,
                        component_filter_stats,
                    ) = filter_geodesic_roi_components(
                        graph=graph,
                        roi_mask=roi_mask,
                        seed_mask=seed_mask,
                        vertices=aligned_vertices,
                        min_vertices=(
                            args.roi_component_min_vertices
                        ),
                        min_seed_points=(
                            args.roi_component_min_seed_points
                        ),
                        relative_vertex_ratio=(
                            args.roi_component_vertex_ratio
                        ),
                        relative_seed_ratio=(
                            args.roi_component_seed_ratio
                        ),
                        preserve_seed_fraction=(
                            args.roi_component_preserve_seed_fraction
                        ),
                    )

                    roi_ratio = float(
                        roi_mask.mean()
                    )

                # --------------------------------------------------
                # 构造牙齿 + 牙龈 ROI 子网格
                # --------------------------------------------------
                (
                    mesh_roi,
                    roi_full_indices,
                ) = subset_mesh_from_vertex_roi(
                    mesh_aligned,
                    roi_mask,
                )

                roi_vertices = np.asarray(
                    mesh_roi.vertices
                )
                roi_normals = np.asarray(
                    mesh_roi.vertex_normals
                )

                print(
                    f"  第二遍输入: "
                    f"{len(roi_vertices)} 顶点, "
                    f"{len(mesh_roi.faces)} 面 "
                    f"({len(roi_vertices) / len(aligned_vertices) * 100:.2f}% 顶点)"
                )

                # --------------------------------------------------
                # Pass 2：ROI 精分割
                # --------------------------------------------------
                pass2_seed = stable_case_seed(
                    case_name,
                    args.pass2_seed,
                )

                (
                    labels_pass2_roi,
                    probs_pass2_roi,
                    logits_pass2_roi,
                    valid_pass2,
                ) = run_segmentation(
                    classifier=classifier,
                    vertices=roi_vertices,
                    normals=roi_normals,
                    device=device,
                    num_classes=args.num_classes,
                    patch_point=args.npoint,
                    expand_times=args.expand_times,
                    random_seed=pass2_seed,
                    stage_name="Pass2 ROI 精分割",
                )

                # 映射回完整 aligned/raw 顶点。
                labels_pass2_full = np.zeros(
                    len(aligned_vertices),
                    dtype=np.int64,
                )
                labels_pass2_full[
                    roi_full_indices
                ] = labels_pass2_roi

                # --------------------------------------------------
                # 保存 NPY / NPZ
                # --------------------------------------------------
                np.save(
                    output_dir
                    / f"{case_name}_pass1.npy",
                    labels_pass1,
                )
                np.save(
                    output_dir
                    / f"{case_name}_pass2_roi.npy",
                    labels_pass2_roi,
                )
                np.save(
                    output_dir
                    / f"{case_name}.npy",
                    labels_pass2_full,
                )
                np.save(
                    output_dir
                    / f"{case_name}_roi_indices.npy",
                    roi_full_indices,
                )

                np.savez_compressed(
                    output_dir
                    / f"{case_name}_two_pass_data.npz",
                    pass1_labels=labels_pass1,
                    pass1_probs=probs_pass1,
                    pass2_roi_labels=labels_pass2_roi,
                    pass2_roi_probs=probs_pass2_roi,
                    pass2_full_labels=labels_pass2_full,
                    roi_full_indices=roi_full_indices,
                    tooth_seed_mask=seed_mask,
                    roi_mask=roi_mask,
                    roi_mask_before_component_filter=(
                        roi_mask_before_component_filter
                    ),
                    geodesic_roi_mask=(
                        geodesic_roi_mask
                    ),
                    palate_added_mask=(
                        palate_added_mask
                    ),
                    palate_preservation_applied=np.asarray(
                        palate_stats["applied"],
                        dtype=bool,
                    ),
                    palate_z_min=np.asarray(
                        palate_stats["palate_z_min"],
                        dtype=np.float32,
                    ),
                    palate_added_points=np.asarray(
                        palate_stats["added_points"],
                        dtype=np.int64,
                    ),
                    roi_component_vertex_counts=(
                        component_filter_stats[
                            "vertex_counts"
                        ]
                    ),
                    roi_component_seed_counts=(
                        component_filter_stats[
                            "seed_counts"
                        ]
                    ),
                    roi_component_keep_flags=(
                        component_filter_stats[
                            "keep_components"
                        ]
                    ),
                    geodesic_distances=(
                        geodesic_distances
                    ),
                    aligned_vertices=(
                        aligned_vertices
                    ),
                    raw_vertices=(
                        raw_vertices
                        if raw_vertices is not None
                        else np.empty(
                            (0, 3),
                            dtype=np.float32,
                        )
                    ),
                    used_margin_mm=np.asarray(
                        used_margin,
                        dtype=np.float32,
                    ),
                    roi_ratio=np.asarray(
                        roi_ratio,
                        dtype=np.float32,
                    ),
                    seed_confidence=np.asarray(
                        used_confidence,
                        dtype=np.float32,
                    ),
                    detected_tooth_classes=np.asarray(
                        detected_class_count,
                        dtype=np.int32,
                    ),
                    pass1_retry_used=np.asarray(
                        retry_payload is not None,
                        dtype=bool,
                    ),
                    pass1_retry_labels=(
                        retry_payload["labels"]
                        if retry_payload is not None
                        else np.empty(
                            (0,),
                            dtype=np.int64,
                        )
                    ),
                    pass1_retry_probs=(
                        retry_payload["probs"]
                        if retry_payload is not None
                        else np.empty(
                            (0, args.num_classes),
                            dtype=np.float32,
                        )
                    ),
                    contamination_detected=np.asarray(
                        contamination_report["suspicious"],
                        dtype=bool,
                    ),
                    contamination_total_tooth_ratio=np.asarray(
                        contamination_report["total_tooth_ratio"],
                        dtype=np.float32,
                    ),
                    crop_fallback=np.asarray(
                        crop_fallback,
                        dtype=bool,
                    ),
                    align_seed=np.asarray(
                        align_seed,
                        dtype=np.uint32,
                    ),
                    chosen_align_trial=np.asarray(
                        chosen_align_trial,
                        dtype=np.int32,
                    ),
                    chosen_align_score=np.asarray(
                        chosen_align_score,
                        dtype=np.float32,
                    ),
                    align_candidate_seeds=np.asarray(
                        [
                            (
                                -1
                                if item.get("align_seed")
                                is None
                                else item["align_seed"]
                            )
                            for item
                            in align_candidate_records
                        ],
                        dtype=np.int64,
                    ),
                    align_candidate_scores=np.asarray(
                        [
                            item.get(
                                "score",
                                -np.inf,
                            )
                            for item
                            in align_candidate_records
                        ],
                        dtype=np.float32,
                    ),
                    pass1_seed=np.asarray(
                        pass1_seed,
                        dtype=np.uint32,
                    ),
                    pass2_seed=np.asarray(
                        pass2_seed,
                        dtype=np.uint32,
                    ),
                )

                with open(
                    output_dir
                    / (
                        f"{case_name}_"
                        "alignment_selection.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as alignment_file:
                    json.dump(
                        {
                            "chosen_trial": int(
                                chosen_align_trial
                            ),
                            "chosen_seed": int(
                                align_seed
                            ),
                            "chosen_score": float(
                                chosen_align_score
                            ),
                            "candidates": (
                                align_candidate_records
                            ),
                        },
                        alignment_file,
                        indent=4,
                        ensure_ascii=False,
                    )

                print(
                    "  ✓ npy/npz 和对齐选择报告保存成功"
                )

                # --------------------------------------------------
                # 保存最终 JSON：与完整原 mesh 顶点长度一致
                # --------------------------------------------------
                try:
                    decoder = load_decoder(
                        decoder_root,
                        case_name,
                        args.num_classes,
                    )

                    result = [
                        decoder.get(
                            str(int(label)),
                            int(label),
                        )
                        for label in labels_pass2_full
                    ]

                    with open(
                        output_dir
                        / f"{case_name}.json",
                        "w",
                        encoding="utf-8",
                    ) as json_file:
                        json.dump(
                            result,
                            json_file,
                            indent=4,
                            ensure_ascii=False,
                        )

                    print("  ✓ json 保存成功")

                except Exception as exc:
                    print(
                        f"  ✗ json 保存失败: {exc}"
                    )
                    tb.print_exc()

                # --------------------------------------------------
                # 保存最终 PLY
                # --------------------------------------------------
                try:
                    save_colored_mesh(
                        output_dir
                        / f"{case_name}_pass2_roi.ply",
                        roi_vertices,
                        np.asarray(
                            mesh_roi.faces
                        ),
                        labels_pass2_roi,
                        color_dict,
                    )

                    save_colored_mesh(
                        output_dir
                        / f"{case_name}_pass2_full_aligned.ply",
                        aligned_vertices,
                        np.asarray(
                            mesh_aligned.faces
                        ),
                        labels_pass2_full,
                        color_dict,
                    )

                    if raw_vertices is not None:
                        save_colored_mesh(
                            output_dir
                            / f"{case_name}_pass2_full_raw.ply",
                            raw_vertices,
                            np.asarray(
                                mesh_raw.faces
                            ),
                            labels_pass2_full,
                            color_dict,
                        )

                    print("  ✓ 最终 PLY 保存成功")

                except Exception as exc:
                    print(
                        f"  ✗ 最终 PLY 保存失败: {exc}"
                    )
                    tb.print_exc()

                # --------------------------------------------------
                # 调试文件
                # --------------------------------------------------
                if args.save_debug:
                    try:
                        save_colored_mesh(
                            debug_dir
                            / f"{case_name}_pass1_full.ply",
                            aligned_vertices,
                            np.asarray(
                                mesh_aligned.faces
                            ),
                            labels_pass1,
                            color_dict,
                        )

                        save_roi_debug_mesh(
                            debug_dir
                            / f"{case_name}_seed_roi_debug.ply",
                            mesh_aligned,
                            seed_mask,
                            roi_mask,
                        )

                        save_component_filter_debug_mesh(
                            debug_dir
                            / (
                                f"{case_name}_"
                                "roi_component_filter_debug.ply"
                            ),
                            mesh_aligned,
                            seed_mask,
                            roi_mask_before_component_filter,
                            roi_mask,
                        )

                        if is_upper_case(case_name):
                            save_upper_palate_debug_mesh(
                                debug_dir
                                / (
                                    f"{case_name}_"
                                    "upper_palate_debug.ply"
                                ),
                                mesh_aligned,
                                seed_mask,
                                geodesic_roi_mask,
                                palate_added_mask,
                                roi_mask,
                            )

                        if retry_payload is not None:
                            save_colored_mesh(
                                debug_dir
                                / f"{case_name}_pass1_retry_full.ply",
                                aligned_vertices,
                                np.asarray(
                                    mesh_aligned.faces
                                ),
                                retry_payload["labels"],
                                color_dict,
                            )

                        print("  ✓ 调试 PLY 保存成功")

                    except Exception as exc:
                        print(
                            f"  ✗ 调试 PLY 保存失败: {exc}"
                        )
                        tb.print_exc()

                print(
                    "  ✓ 完成：Pass1 粗分割 -> 紧凑种子/QC -> "
                    "测地牙龈 ROI -> Pass2 精分割"
                )

                del (
                    mesh_raw,
                    mesh_aligned,
                    mesh_roi,
                    aligned_vertices,
                    aligned_normals,
                    roi_vertices,
                    roi_normals,
                    graph,
                    labels_pass1,
                    probs_pass1,
                    logits_pass1,
                    labels_pass2_roi,
                    probs_pass2_roi,
                    logits_pass2_roi,
                    labels_pass2_full,
                )

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                print(f"  ✗ 文件级错误: {exc}")
                tb.print_exc()
                continue

    print("\n🎉 Done!")


if __name__ == "__main__":
    arguments = parse_args()
    test(arguments)