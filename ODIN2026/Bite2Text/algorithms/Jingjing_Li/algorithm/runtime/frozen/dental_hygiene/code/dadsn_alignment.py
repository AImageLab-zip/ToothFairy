#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal DADSN scan alignment used by the frozen live segmenter."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def rotation_matrix_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    axis = np.cross(source, target)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-10 or target_norm < 1e-10:
        return np.eye(3)
    cosine = np.clip(np.dot(source, target) / (source_norm * target_norm), -1.0, 1.0)
    angle = np.arccos(cosine)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm if axis_norm else axis
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


def fit_parabola(vertices: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(vertices) < 10:
        raise ValueError("Too few vertices for scan alignment")
    if np.min(np.ptp(vertices, axis=0)) < 1e-6:
        raise ValueError("Degenerate mesh for scan alignment")
    initial_normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(initial_normal)
    if norm < 1e-10:
        raise ValueError("Zero mean normal for scan alignment")
    initial_normal /= norm
    initial_rotation = Rotation.from_matrix(
        rotation_matrix_between(np.asarray([0.0, 0.0, 1.0]), initial_normal)
    )
    initial_angles = initial_rotation.as_euler("xyz", degrees=True)
    initial_offset = np.mean(vertices, axis=0)
    initial = initial_angles.tolist() + initial_offset.tolist() + [1.0]

    def error(parameters: np.ndarray, points: np.ndarray) -> float:
        pitch, yaw, roll, x, y, z, coefficient = parameters
        rotation = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True)
        normal_vector = rotation.apply(np.asarray([0.0, 0.0, 1.0]))
        a, b, c = normal_vector
        d = -np.dot(normal_vector, np.asarray([x, y, z]))
        normal_length = np.linalg.norm(normal_vector)
        if normal_length < 1e-10:
            return 1e10
        signed = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_length
        projected = points - signed[:, None] * normal_vector[None, :] / normal_length
        transform = np.vstack(
            [
                np.hstack([rotation.as_matrix(), np.asarray([[x], [y], [z]])]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ]
        )
        try:
            transformed = np.linalg.inv(transform) @ np.vstack([projected.T, np.ones((1, len(points)))])
        except np.linalg.LinAlgError:
            return 1e10
        transformed = transformed[:3].T
        distance_error = np.mean(signed**2)
        parabola_error = np.mean((transformed[:, 1] - coefficient * transformed[:, 0] ** 2) ** 2)
        center_error = np.linalg.norm(np.mean(transformed[:, :2], axis=0)) ** 2
        return float(distance_error * 10.0 + parabola_error + center_error * 0.01)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized = minimize(
            error,
            initial,
            args=(vertices,),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-6},
        )
    if not optimized.success:
        raise ValueError(f"Scan alignment optimization failed: {optimized.message}")
    pitch, yaw, roll, x, y, z, coefficient = optimized.x
    rotation = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True)
    orientation = rotation.apply(np.asarray([0.0, 0.0, 1.0]))
    if np.dot(orientation, initial_normal) < 0:
        orientation = -orientation
    face = rotation.apply(np.asarray([0.0, -1.0, 0.0])) if coefficient >= 0 else rotation.apply(
        np.asarray([0.0, 1.0, 0.0])
    )
    return np.asarray([x, y, z]), orientation, face


def align_scan(mesh, sample_count: int = 2000):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if len(vertices) < 100:
        return mesh
    if len(vertices) > sample_count:
        indices = np.random.choice(len(vertices), sample_count, replace=False)
        selected_vertices = vertices[indices]
        selected_normals = normals[indices]
    else:
        selected_vertices = vertices
        selected_normals = normals
    center, orientation, face = fit_parabola(selected_vertices, selected_normals)

    target_orientation = np.asarray([0.0, 0.0, -1.0])
    target_face = np.asarray([1.0, 0.0, 0.0])

    def rotation_error(vector: np.ndarray) -> float:
        matrix = Rotation.from_rotvec(vector).as_matrix()
        return float(
            np.sum((matrix @ orientation - target_orientation) ** 2)
            + np.sum((matrix @ face - target_face) ** 2)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized = minimize(
            rotation_error,
            [0.0, 0.0, 0.0],
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * 3,
            options={"maxiter": 1000},
        )
    matrix = Rotation.from_rotvec(optimized.x).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = -matrix @ center
    mesh.apply_transform(transform)
    return mesh
