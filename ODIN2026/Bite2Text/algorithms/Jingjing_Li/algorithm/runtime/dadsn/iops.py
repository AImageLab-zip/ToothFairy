#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yuxian Jiang
@Contact: yuxianjiang@sjtu.edu.cn
@Article: Morphology Prior Enhanced Teeth Segmentation for High Resolution Oral Scans
@Journal: Journal of Biomedical and Health Informatics (2025)
"""

import os
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import warnings

def get_rotation_matrix(vector_source, vector_target):    # rotate B to A

    # Calculate the rotation axis by taking the cross product of the two vectors
    rotation_axis = np.cross(vector_source, vector_target)

    # Calculate the angle between the two vectors
    norm_source = np.linalg.norm(vector_source)
    norm_target = np.linalg.norm(vector_target)
    
    # 【修复】避免除以0
    if norm_source < 1e-10 or norm_target < 1e-10:
        return np.eye(3)
    
    cos_angle = np.dot(vector_source, vector_target) / (norm_source * norm_target)
    # 【修复】限制 cos_angle 在 [-1, 1] 范围内，避免 arccos 出错
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    norm_axis = np.linalg.norm(rotation_axis)
    rotation_axis = rotation_axis / norm_axis if norm_axis != 0 else rotation_axis
    x, y, z = rotation_axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([
        [t*x**2 + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y**2 + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z**2 + c]
    ])

    return rotation_matrix

def parabola_3Dfitting(vertices, normals):
    # 【修复】输入验证
    if len(vertices) < 10:
        raise ValueError(f"顶点数太少: {len(vertices)}")
    
    # 【修复】检查是否退化
    bbox = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    if np.min(bbox) < 1e-6:
        raise ValueError("网格退化（所有点几乎共面）")
    
    init_normal = np.mean(np.asarray(normals), axis=0)
    norm_init = np.linalg.norm(init_normal)
    
    # 【修复】避免除以0
    if norm_init < 1e-10:
        raise ValueError("法线向量为零")
    
    init_normal /= norm_init
    init_rotation = get_rotation_matrix(np.array([0, 0, 1]), init_normal)
    init_rotation = R.from_matrix(init_rotation)
    init_angles = init_rotation.as_euler('xyz', degrees=True)
    init_offset = np.mean(np.asarray(vertices), axis=0)
    init_params = init_angles.tolist() + init_offset.tolist() + [1]

    def fitting_error(params, points):
        pitch, yaw, roll, x, y, z, a = params

        euler_angles = np.array([pitch, yaw, roll])
        rotation = R.from_euler('xyz', euler_angles, degrees=True)

        rot_z = rotation.apply(np.array([0, 0, 1]))
        A, B, C, D = rot_z[0], rot_z[1], rot_z[2], -np.dot(rot_z, np.array([x, y, z]))

        normal_length = np.sqrt(A ** 2 + B ** 2 + C ** 2)
        # 【修复】避免除以0
        if normal_length < 1e-10:
            return 1e10
        
        signed_distance_numerator = A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D
        signed_distances = signed_distance_numerator / normal_length
        projected_points = points - np.dot(signed_distances[:, None], np.array([[A, B, C]])) / normal_length

        trans_matrix = np.vstack([np.hstack([rotation.as_matrix(), np.array([[x], [y], [z]])]), np.array([[0, 0, 0, 1]])])
        
        # 【修复】检查矩阵是否可逆
        try:
            trans_matrix_inv = np.linalg.inv(trans_matrix)
        except np.linalg.LinAlgError:
            return 1e10
        
        trans_points = np.dot(trans_matrix_inv, np.vstack([projected_points.T, np.ones((1, points.shape[0]))]))
        trans_points = trans_points[:3, :].T

        distance_error = np.mean(signed_distances ** 2)
        parabola_error = np.mean((trans_points[:, 1] - a * trans_points[:, 0] ** 2) ** 2)

        center_distance_error = np.linalg.norm(np.mean(trans_points[:, :2], axis=0)) ** 2

        return distance_error * 10 + parabola_error + center_distance_error * 0.01

    # 【修复】用 try-except 包裹优化过程
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_params = minimize(
                fun=fitting_error,
                x0=init_params,
                args=(np.asarray(vertices)),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
        
        # 【修复】检查优化是否成功
        if not optimized_params.success:
            raise ValueError(f"优化失败: {optimized_params.message}")
        
        pitch, yaw, roll, x, y, z, a = optimized_params.x
        euler_angles = np.array([pitch, yaw, roll])
        rotation = R.from_euler('xyz', euler_angles, degrees=True)
        rot_z = rotation.apply(np.array([0, 0, 1]))
        
        if np.dot(rot_z, init_normal) < 0:
            rot_z = - rot_z
        rot_y = rotation.apply(np.array([0, -1, 0])) if a >= 0 else rotation.apply(np.array([0, 1, 0]))
        center = np.array([x, y, z])

        return center, rot_z, rot_y
        
    except Exception as e:
        raise ValueError(f"parabola_3Dfitting 失败: {str(e)}")

def align_scan(mesh, mask=None, sample_num=2000, target_orit=None, target_face=None):
    """
    Align the mesh to the standard orientation. center: [0, 0, 0], orit: [0, 0, -1], face: [1, 0, 0]
    :param mesh: Mesh
    :param mask: mask of vertices
    :param sample_num:
    :return: aligned mesh
    """
    vertices = mesh.vertices
    normals = np.asarray(mesh.vertex_normals)
    
    # 【修复】输入验证
    if len(vertices) < 100:
        print(f"  警告: 顶点数太少 ({len(vertices)})，跳过对齐")
        return mesh

    if mask is not None:
        mask = mask.astype(bool)
        vertices = vertices[mask]
        normals = normals[mask]

    if vertices.shape[0] > sample_num:
        indices = np.random.choice(vertices.shape[0], sample_num, replace=False)
        selected_vertices, selected_normals = vertices[indices], normals[indices]
    else:
        selected_vertices, selected_normals = vertices, normals

    # 【修复】用 try-except 包裹抛物线拟合
    try:
        mesh_center, mesh_orit, mesh_face = parabola_3Dfitting(selected_vertices, selected_normals)
    except Exception as e:
        print(f"  警告: parabola_3Dfitting 失败: {e}，跳过对齐")
        return mesh

    np.set_printoptions(precision=3, suppress=True)
    center_new = np.array([0, 0, 0])

    def loss_function(rotation_vector, x1, y1, x2, y2):
        r = R.from_rotvec(rotation_vector).as_matrix()
        rot_x1 = np.dot(r, x1)
        rot_x2 = np.dot(r, x2)
        loss = np.sum((rot_x1 - y1) ** 2) + np.sum((rot_x2 - y2) ** 2)
        return loss
    
    target_orit = np.array([0, 0, -1]) if target_orit is None else np.asarray(target_orit)
    target_face = np.array([1, 0, 0]) if target_face is None else np.asarray(target_face)

    initial_guess = [0, 0, 0]
    bounds = [(-np.pi, np.pi) for _ in range(3)]
    
    # 【修复】用 try-except 包裹旋转优化
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(fun=loss_function, x0=initial_guess,
                              args=(mesh_orit, target_orit, mesh_face, target_face),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000})
        
        rotation_vector_optimal = result.x
        rotation_matrix = R.from_rotvec(rotation_vector_optimal).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = center_new - np.dot(rotation_matrix, mesh_center)
        mesh.apply_transform(transform_matrix)
        
    except Exception as e:
        print(f"  警告: 旋转优化失败: {e}，跳过对齐")
        return mesh

    return mesh