from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from teethland import PointTensor
from teethland.nn.modules.loss import lovasz_hinge_flat


class SpatialEmbeddingLoss(nn.Module):

    def __init__(
        self,
        learn_center: bool=True,
        learn_ellipsoid: bool=True,
        w_foreground: float=1.0,
        w_instance: float=1.0,
        w_smooth: float=10.0,
        w_seed: float=10.0,
    ):
        super().__init__()

        self.learn_center = learn_center
        self.n_sigma = 3 if learn_ellipsoid else 1

        self.w_foreground = w_foreground
        self.w_instance = w_instance
        self.w_smooth = w_smooth
        self.w_seed = w_seed

    def forward(
        self,
        pred_offsets: PointTensor,
        pred_sigmas: PointTensor,
        pred_seeds: PointTensor,
        targets: PointTensor,
    ):
        """
        'k' index represents instance
        'i' index represents point
        """

        loss = 0

        for b in range(targets.batch_size):

            spatial_emb = torch.tanh(pred_offsets.batch(b).F) + pred_offsets.batch(b).C
            sigma = pred_sigmas.batch(b).F
            seed_map = torch.sigmoid(pred_seeds.batch(b).F)

            # loss accumulators
            smooth_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instances = targets.batch(b).F

            # regress bg to zero
            bg_mask = instances == -1
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for k in instances.unique()[1:]:
                mask_k = instances == k  # 1 x d x h x w

                # predict center of attraction (\hat{C}_k)
                if self.learn_center:
                    center_k = spatial_emb[mask_k].mean(0)
                else:
                    center_k = self.xyzm[mask_k.expand_as(self.xyzm)].view(
                        3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1

                # calculate sigma
                sigmas_ki = sigma[mask_k]
                sigma_k = sigmas_ki.mean(0)

                # calculate smooth loss before exp
                smooth_loss = smooth_loss + torch.mean(
                    torch.pow(sigmas_ki - sigma_k.detach(), 2),
                )

                # exponential to effectively predict 1 / (2 * sigma_k**2)
                sigma_k = torch.exp(sigma_k * 10)

                # calculate gaussian
                probs_i = torch.exp(-1 * torch.sum(
                    sigma_k * torch.pow(spatial_emb - center_k, 2),
                    dim=1,
                ))

                # apply lovasz-hinge loss
                logits_i = 2 * probs_i - 1
                instance_loss = instance_loss + lovasz_hinge_flat(logits_i, mask_k)

                # seed loss
                seed_loss += self.w_foreground * torch.sum(
                    torch.pow(seed_map[mask_k] - probs_i[mask_k, None].detach(), 2),
                )

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                smooth_loss /= obj_count

            seed_loss = seed_loss / targets.batch_counts[b]

            loss += (
                self.w_instance * instance_loss
                + self.w_smooth * smooth_loss
                + self.w_seed * seed_loss
            )

        loss = loss / targets.batch_size

        return loss + pred_offsets.F.sum()*0


class IdentificationLoss(nn.Module):

    def __init__(
        self,
        w_ce: float=1.0,
        w_focal: float=1.0,
        w_homo: float=1.0,
        alpha: float=0.25,
        gamma: float=2.0,
    ):
        super().__init__()

        self.w_ce = w_ce
        self.w_focal = w_focal
        self.w_homo = w_homo
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        prototypes: PointTensor,
        classes: PointTensor,
        targets: PointTensor,
    ):
        ce_loss = F.cross_entropy(classes.F, targets.F, reduction='none', label_smoothing=0.01)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        homo_loss = torch.zeros_like(ce_loss[:0])
        for b in range(prototypes.batch_size):
            homo = torch.mean(prototypes.batch(b).F - prototypes.batch(b).F.mean(0))
            homo_loss = torch.cat((homo_loss, homo[None]))

        loss = (
            self.w_ce * ce_loss.mean()
            + self.w_focal * focal_loss.mean()
            + self.w_homo * homo_loss.mean()
        )

        return loss
    

class PoolingLandmarkLoss(nn.Module):

    def __init__(
        self,
        w_offsets: float=12.0,
        w_kobjs: float=1.0,
        w_homo: float=1.0,
    ):
        super().__init__()

        self.w_offsets = w_offsets
        self.w_kobjs = w_kobjs
        self.w_homo = w_homo

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        prototypes: PointTensor,
        landmarks: PointTensor,
        targets: PointTensor,
    ):
        landmarks = landmarks.F.reshape(landmarks.F.shape[0], -1, 4)
        coords = landmarks[..., :3]
        kobjs = landmarks[..., -1]

        kpt_mask = torch.any(~torch.isclose(targets.F, torch.zeros_like(targets.F)), axis=-1)

        d = torch.pow(coords - targets.F, 2).sum(-1)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        offsets_loss = kpt_loss_factor[:, None] * ((1 - torch.exp(-d)) * kpt_mask)

        kobj_loss = self.bce(kobjs, kpt_mask.float())
        
        homo_loss = torch.zeros(0).to(coords)
        for b in range(prototypes.batch_size):
            homo = torch.mean(prototypes.batch(b).F - prototypes.batch(b).F.mean(0))
            homo_loss = torch.cat((homo_loss, homo[None]))

        loss = (
            self.w_offsets * offsets_loss.mean()
            + self.w_kobjs * kobj_loss.mean()
            + self.w_homo * homo_loss.mean()
        )

        return loss


class OffsetLandmarkLoss(nn.Module):

    def __init__(
        self,
        dist_thresh: float=0.10,
        w_dist: float=0.05,
        w_chamfer: float=0.05,
        w_separation: float=0.0005,
    ):
        super().__init__()

        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')

        self.dist_thresh = dist_thresh
        self.w_dist = w_dist
        self.w_chamfer = w_chamfer
        self.w_separation = w_separation

    def forward(
        self,
        landmarks: PointTensor,
        targets: PointTensor,
        classes: Optional[List[int]],
    ):
        # make tensor from classes to select
        if classes is not None:
            classes = torch.tensor(classes).to(targets.F)
        else:
            classes = torch.unique(targets.F)

        # determine loss value of all cusp landmarks
        loss = 0
        for b in range(landmarks.batch_size):
            # ground-truth landmarks
            land_classes_k = targets.batch(b).F
            land_mask_k = torch.any(land_classes_k[None] == classes[:, None], axis=0)
            land_coords_k = targets.batch(b).C[land_mask_k]

            # determine predicted distances and landmarks
            dists_i = landmarks.batch(b).F[:, 0]
            coords_i = landmarks.batch(b).C + landmarks.batch(b).F[:, 1:]
            if land_coords_k.shape[0] == 0:
                dist_loss = self.smooth_l1(
                    dists_i, torch.full_like(dists_i, 2 * self.dist_thresh),
                )
                loss += self.w_dist * dist_loss
                continue

            # distance loss
            point_dists_ki = torch.linalg.norm(
                landmarks.batch(b).C[None] - land_coords_k[:, None], dim=-1,
            )
            min_dists = point_dists_ki.amin(dim=0)
            points_mask_i = min_dists < self.dist_thresh
            dist_loss = self.smooth_l1(
                dists_i, torch.clip(min_dists, 0, 2 * self.dist_thresh),
            )

            # masked chamfer distance loss
            land_dists_ki = torch.linalg.norm(
                coords_i[None] - land_coords_k[:, None], dim=-1,
            )
            pred2gt_dist = torch.sum(land_dists_ki.amin(dim=0)[points_mask_i] ** 2)
            gt2pred_dist = torch.sum(land_dists_ki.amin(dim=1) ** 2)
            chamfer_loss = pred2gt_dist + gt2pred_dist

            # masked separation loss
            if land_coords_k.shape[0] >= 2:
                neighbours = torch.argsort(point_dists_ki, dim=0)[:2]
                land_dists_pi = land_dists_ki[(neighbours, torch.arange(land_dists_ki.shape[1]))]
                separation = torch.sum(land_dists_pi[0, points_mask_i] / land_dists_pi[1, points_mask_i])
            else:
                separation = 0.0

            loss += (
                self.w_dist * dist_loss
                + self.w_chamfer * chamfer_loss
                + self.w_separation * separation
            )
        
        if landmarks.batch_size > 0:
            loss = loss / landmarks.batch_size

        return loss


class LandmarkLoss(nn.Module):

    def __init__(
        self,
        w_offsets: float=12.0,
        w_kobjs: float=1.0,
        w_homo: float=1.0,
        dist_thresh: float=0.12,
        w_dist: float=0.05,
        w_chamfer: float=0.05,
        w_separation: float=0.0005,
    ):
        super().__init__()

        self.pooling_loss = PoolingLandmarkLoss(w_offsets, w_kobjs, w_homo)
        self.points_loss = OffsetLandmarkLoss(dist_thresh, w_dist, w_chamfer, w_separation)
    
    def forward(
        self,
        prototypes: PointTensor,
        centroid_offsets: PointTensor,
        point_offsets: PointTensor,
        instances: PointTensor,
        landmarks: PointTensor,
    ):
        instances_loss = self.pooling_loss(prototypes, centroid_offsets, instances)
        points_loss = self.points_loss(point_offsets, landmarks, [5, 6])

        return instances_loss, points_loss
