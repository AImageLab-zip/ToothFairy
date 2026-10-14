from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethinstseg import TeethInstSegDataModule
from teethland.data.datasets import TeethSegDataset
import teethland.data.transforms as T
from teethland.visualization import draw_point_clouds


class TeethInstFullDataModule(TeethInstSegDataModule):
    """Data module to load intraoral scans with teeth instances."""

    def __init__(
        self,
        uniform_density_voxel_size: List[int],
        **dm_cfg,
    ):
        super().__init__(uniform_density_voxel_size=uniform_density_voxel_size[1:], **dm_cfg)

        self.default_transforms = T.Compose(
            T.UniformDensityDownsample(uniform_density_voxel_size[0]),
            self.default_transforms,
        )

    def process_instances(
        self,
        instances: PointTensor,
    ) -> PointTensor:
        # determine mesh of background triangles
        bg_mask = instances.F == -1

        bg_triangles = self.triangles[torch.any(bg_mask[self.triangles], dim=-1)]
        
        vertex_mask = torch.zeros_like(bg_mask)
        vertex_mask[bg_triangles.flatten()] = True
        bg_vertices = instances.C[vertex_mask]
        
        vertex_map = torch.full((instances.C.shape[0],), -1).to(instances.F)
        vertex_map[vertex_mask] = torch.arange(bg_vertices.shape[0]).to(instances.F)
        bg_triangles = vertex_map[bg_triangles]

        bg_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(bg_vertices.cpu().numpy()),
            open3d.utility.Vector3iVector(bg_triangles.cpu().numpy()),
        )

        # cluster connected triangles of background mesh
        cluster_idxs, _, _ = bg_mesh.cluster_connected_triangles()
        cluster_idxs = torch.from_numpy(np.asarray(cluster_idxs)).to(instances.F)
        
        # add connected triangles surrounded by one instance to that instance
        vertex_idxs = torch.nonzero(vertex_mask)[:, 0]
        for idx in torch.unique(cluster_idxs):
            edges = torch.concatenate((
                bg_triangles[cluster_idxs == idx][:, [0, 1]],
                bg_triangles[cluster_idxs == idx][:, [0, 2]],
                bg_triangles[cluster_idxs == idx][:, [1, 2]],
            ))
            unique, counts = torch.unique(torch.sort(edges, dim=-1)[0], dim=0, return_counts=True)
            border_vertex_idxs = unique[counts == 1].flatten()

            labels = instances.F[vertex_idxs[border_vertex_idxs]]
            if labels.shape[0] == 0 or labels[0] == -1 or not torch.all(labels[0] == labels):
                continue

            instances.F[vertex_idxs[edges.flatten()]] = labels[0].item()

        # determine mesh of tooth triangles
        fg_mask = instances.F >= 0

        fg_triangles = self.triangles[torch.any(fg_mask[self.triangles], dim=-1)]
        
        vertex_mask = torch.zeros_like(fg_mask)
        vertex_mask[fg_triangles.flatten()] = True
        fg_vertices = instances.C[vertex_mask]
        
        vertex_map = torch.full((instances.C.shape[0],), -1).to(instances.F)
        vertex_map[vertex_mask] = torch.arange(fg_vertices.shape[0]).to(instances.F)
        fg_triangles = vertex_map[fg_triangles]

        fg_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(fg_vertices.cpu().numpy()),
            open3d.utility.Vector3iVector(fg_triangles.cpu().numpy()),
        )

        # cluster connected triangles of teeth
        cluster_idxs, _, cluster_areas = fg_mesh.cluster_connected_triangles()
        cluster_idxs = torch.from_numpy(np.asarray(cluster_idxs)).to(instances.F)
        cluster_areas = torch.from_numpy(np.asarray(cluster_areas)).to(instances.C)

        # remove small connected components
        vertex_idxs = torch.nonzero(vertex_mask)[:, 0]
        for idx in torch.unique(cluster_idxs):
            if cluster_areas[idx] >= 0.005:
                continue

            cluster_vertex_idxs = fg_triangles[cluster_idxs == idx].flatten()
            instances.F[vertex_idxs[cluster_vertex_idxs]] = -1
                
        return instances

    def process_landmarks(
        self,
        labels: PointTensor,
        landmarks: PointTensor,
    ) -> PointTensor:
        # apply inverse affine transformation to undo preprocessing
        affine = labels.cache['rot_matrix'] @ self.affine[0]
        landmarks_hom = torch.column_stack((
            landmarks.C, torch.ones_like(landmarks.C[:, 0]),
        ))
        landmarks_coords = (landmarks_hom @ torch.linalg.inv(affine.T))[:, :3]

        # project landmarks to scan surface
        mesh = open3d.io.read_triangle_mesh(str(self.root / self.scan_file))
        mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = open3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
        closest_points = scene.compute_closest_points(landmarks_coords.cpu().numpy())
        landmarks_coords = torch.from_numpy(closest_points['points'].numpy()).to(landmarks_coords)

        # separate mesial and distal landmarks
        mesial_and_distal = landmarks[landmarks.F[:, -1] == 0]
        mesial_mask = torch.zeros(0, device=labels.F.device, dtype=torch.int64)
        for b in range(landmarks.batch_size):
            label = labels.F[b]
            md = mesial_and_distal.batch(b).C

            if label % 10 in [1, 2]:  # incisor
                if label // 10 in [2, 4, 6, 8]:
                    is_mesial = md[:, 0] > labels.C[b, 0]
                elif label // 10 in [1, 3, 5, 7]:
                    is_mesial = md[:, 0] < labels.C[b, 0]
            elif label % 10 == 3:  # canine
                if label // 10 in [2, 4, 6, 8]:
                    scores = np.stack((
                        (md - labels.C[b])[:, :2].cpu().numpy() @ [-1, 1],  # topleft = distal
                        (md - labels.C[b])[:, :2].cpu().numpy() @ [1, -1],  # bottomright = mesial
                    ))
                elif label // 10 in [1, 3, 5, 7]:
                    scores = np.stack((
                        (md - labels.C[b])[:, :2].cpu().numpy() @ [1, 1],  # topright = distal
                        (md - labels.C[b])[:, :2].cpu().numpy() @ [-1, -1],  # bottomleft = mesial
                    ))
                
                is_mesial = torch.from_numpy(scores.argmax(0)).to(mesial_mask)
            elif label % 10 in [4, 5, 6, 7, 8]:  # premolar+molar
                is_mesial = md[:, 1] < labels.C[b, 1]

            mesial_mask = torch.cat((mesial_mask, is_mesial))

        # update classes of landmarks into separate mesial/distal
        md_idxs = torch.nonzero(landmarks.F[:, -1] == 0)[:, 0]
        landmarks_classes = landmarks.F[:, -1].clone()
        landmarks_classes[md_idxs[mesial_mask == -1]] = -2
        landmarks_classes[md_idxs[mesial_mask == 1]] = -1
        landmarks_classes += 1

        # save new landmark classes to PointTensor
        batch_counts = torch.bincount(
            landmarks.batch_indices[landmarks_classes >= 0],
            minlength=landmarks.batch_size,
        )
        landmarks = PointTensor(
            coordinates=landmarks_coords[landmarks_classes >= 0],
            features=torch.column_stack((
                landmarks.F[landmarks_classes >= 0, 0],
                landmarks_classes[landmarks_classes >= 0],
            )),
            batch_counts=batch_counts,
        )

        return landmarks

    def setup(self, stage: Optional[str]=None):
        if stage is None or stage == 'predict':
            files = self._files('predict', exclude=[])
            print('Total number of files:', len(files))

            self.pred_dataset = TeethSegDataset(
                stage='predict',
                root=self.root,
                files=files,
                clean=self.clean,
                transform=self.default_transforms,
            )

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        str,
        TensorType['B', torch.bool],
        TensorType['T', 3, torch.float32],
        TensorType['B', 4, 4, torch.float32],
        PointTensor,
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        is_lower = torch.stack(batch_dict['is_lower'])
        triangles = torch.cat(batch_dict['triangles'])
        affine = torch.stack(batch_dict['affine'])

        # collate input points and features
        point_counts = torch.stack(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            features=torch.cat(batch_dict['features']),
            batch_counts=point_counts,
        )

        x.cache['instseg_downsample_idxs'] = self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs_1'],
            batch_dict['ud_downsample_count_1'],
        )
        x.cache['landmarks_downsample_idxs'] = self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs_2'],
            batch_dict['ud_downsample_count_2'],
        )

        return scan_file, is_lower, triangles, affine, x

    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> PointTensor:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)
        self.triangles = batch[2].to(device)
        self.affine = batch[3].to(device)

        return batch[4].to(device)
    

class TeethMixedFullDataModule(TeethInstFullDataModule):

    @property
    def num_classes(self) -> int:
        return 12 if self.filter or not self.distinguish_upper_lower else 24

    def reorder_teeth(
        self,
        classes: PointTensor,
        side_mask: TensorType['N', torch.bool],
        dist_thresh_nonmolar: float=0.35,
        dist_thresh_molar: float=0.5,
    ) -> PointTensor:
        # fix teeth labels iteratively on one side
        side_idxs = torch.nonzero(side_mask)[:, 0]
        side_classes = classes[side_idxs]
        sort_cols = torch.column_stack((side_classes.F, side_classes.C[:, 1]))
        _, argsort1 = torch.sort(sort_cols[:, 1], stable=True)
        _, argsort0 = torch.sort(sort_cols[argsort1, 0], stable=True)
        argsort = argsort1[argsort0]
        batch_idxs = side_classes.batch_indices[argsort]
        coords = side_classes.C[argsort]
        for i in range(side_idxs.shape[0] - 1):
            if batch_idxs[i] != batch_idxs[i + 1]:
                continue

            i_ = side_idxs[argsort[i]]
            j_ = side_idxs[argsort[i + 1]]
            if classes.F[i_] < 2:
                continue
            
            dist_thresh = dist_thresh_nonmolar if classes.F[j_] < 5 else dist_thresh_molar
            dist = torch.sqrt(torch.sum((coords[i, :2] - coords[i + 1, :2])**2))
            dist_offset = classes.F[j_] - classes.F[i_]
            if dist / dist_thresh < 1 + dist_offset:
                continue
            
            classes.F[j_] = classes.F[i_] + (dist / dist_thresh).long()

        # add third molar if three molars are present on one side
        if (classes.F[side_idxs] >= 5).sum() == 3:
            m3_idx = classes.C[classes.F >= 5, 1].argmax()
            classes.F[torch.nonzero(classes.F >= 5)[m3_idx, 0]] = 7        

        return classes

    def teeth_classes_to_labels(
        self,
        classes: PointTensor,
    ) -> PointTensor:

        # determine tooth instances on the right side of the arch
        right_mask = torch.zeros(0, dtype=torch.bool, device=classes.C.device)
        for batch_idx in range(classes.batch_size):
            is_incisor = (classes.batch_indices == batch_idx) & torch.any(
                classes.F[:, None] == torch.tensor([[0, 1, 7, 8]]).to(classes.F),
            dim=1)
            is_incisor = torch.nonzero(is_incisor)[:, 0]
            is_incisor = is_incisor[torch.argsort(classes.C[is_incisor, 1])[:4]]

            if is_incisor.shape[0] <= 2:
                zero_x = 0.0
            else:
                weights = 1 - (classes.F[is_incisor] % 7) / 2
                zero_x = torch.sum(weights * classes.C[is_incisor, 0]) / torch.sum(weights)

            right = classes.batch(batch_idx).C[:, 0] < zero_x
            right_mask = torch.cat((right_mask, right))

        # make room for third molars
        labels = classes.clone()
        labels.F[labels.F >= 7] += 1

        # do reordering without primary/permanent distinction
        primary_mask = labels.F >= 8
        labels.F = labels.F % 8

        # fix duplicate classes on one side, introducing third molars
        labels = self.reorder_teeth(labels, right_mask)
        labels = self.reorder_teeth(labels, ~right_mask)

        # translate index label to FDI label
        if self.filter == 'lower':
            labels.F += 31 + 10 * right_mask + 40 * primary_mask
        else:
            if self.distinguish_upper_lower:
                raise NotImplementedError()
            else:
                labels.F = torch.clip(labels.F, 0, 7)
                labels.F += 40 * primary_mask * (labels.F < 5)
                labels.F += 20 * self.is_lower[classes.batch_indices]
                
            labels.F += 11 + 10 * right_mask
        
        return labels
