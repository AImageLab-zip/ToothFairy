from typing import Any, List, Optional, Tuple, Union

import sklearn.cluster
import torch
from torch_scatter import scatter
from torchtyping import TensorType

from pointops import (
    ballQuery,
    farthestPointSampling,
    kNNQuery,
)

import teethland.cluster


class TensorCache(dict):
    """Implements cache to store intermediate tensors."""

    def __init__(self, device: torch.device):
        super().__init__()

        self.device = device

    def __setitem__(self, key: str, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), (
            f'value must be a torch.Tensor, got {type(value)}.'
        )
        # if not hasattr(self, 'device'):
        #     print(key, value)
        # assert self.device == value.device, (
        #     f'value must be on {self.device}, got {value.device}'
        # )
        super().__setitem__(key, value)

    def key(self, value: torch.Tensor) -> str:
        assert isinstance(value, torch.Tensor), (
            f'value must be a torch.Tensor, got {type(value)}.'
        )

        for k, v in self.items():
            if value.is_set_to(v):
                return k
        
        raise ValueError(f'{value} is not in cache.')


class PointTensor:
    """Implements abstraction of torch.Tensor for batched point cloud data."""

    def __init__(
        self,
        coordinates: TensorType['N', 3, torch.float32],
        features: Optional[TensorType['N', '...', Any]]=None,
        batch_counts: Optional[TensorType['B', Any]]=None,
        batch_indices: Optional[TensorType['N', Any]]=None,
    ):
        assert isinstance(coordinates, torch.Tensor), (
            f'coordinates must be a torch.Tensor, got {type(coordinates)}.'
        )
        assert coordinates.shape[1] == 3, (
            f'coordinates must have 3 columns, got {coordinates.shape[1]}.'
        )
        assert coordinates.dtype == torch.float32, (
            f'coordinates must have float32 dtype, got {coordinates.dtype}.'
        )

        assert features is None or isinstance(features, torch.Tensor), (
            f'features must be a torch.Tensor, got {type(features)}.'
        )
        assert features is None or coordinates.device == features.device, (
            'coordinates and features must be on same device, got '
            f'{coordinates.device} and {features.device}.'
        )
        assert features is None or coordinates.shape[0] == features.shape[0], (
            'coordinates and features must share first dimension, got '
            f'{coordinates.shape[0]} and {features.shape[0]}.'
        )

        if batch_counts is None:
            batch_counts = torch.tensor([coordinates.shape[0]])
        assert isinstance(batch_counts, torch.Tensor), (
            f'batch_counts must be a torch.Tensor, got {type(batch_counts)}.'
        )
        assert batch_counts.dtype in [torch.int32, torch.int64], (
            f'batch_counts must have int32/64 dtype, got {batch_counts.dtype}.'
        )
        assert batch_counts.dim() == 1, (
            f'batch_counts must be 1D, got {batch_counts.dim()} dimensions.'
        )
        assert coordinates.shape[0] == batch_counts.sum(), (
            'Sum of batch_counts must equal provided number of coordinates, '
            f'got {batch_counts.sum()} and {coordinates.shape[0]}'
        )

        if batch_indices is None:
            batch_indices = torch.arange(
                batch_counts.shape[0], device=batch_counts.device,
            )
            batch_indices = batch_indices.repeat_interleave(batch_counts)
        assert isinstance(batch_indices, torch.Tensor), (
            f'batch_indices must be a torch.Tensor, got {type(batch_indices)}.'
        )
        assert batch_indices.dim() == 1, (
            f'batch_indices must be 1D, got {batch_indices.dim()} dimensions.'
        )

        self._coordinates = coordinates
        self._features = features
        self._batch_counts = batch_counts.to(coordinates.device, torch.int32)
        self._batch_indices = batch_indices.to(coordinates.device, torch.int16)
        self.cache = TensorCache(coordinates.device)

    def clone(self):
        pt = PointTensor(
            coordinates=self._coordinates.clone(),
            batch_counts=self._batch_counts.clone(),
            batch_indices=self._batch_indices.clone(),
        )
        if self.has_features:
            pt.F = self.F.clone()

        for k, v in self.cache.items():
            pt.cache[k] = v.clone()

        return pt

    def to(self, *args, **kwargs):
        pt = PointTensor(
            coordinates=self._coordinates.to(*args, **kwargs),
            batch_counts=self._batch_counts.to(*args, **kwargs),
            batch_indices=self._batch_indices.to(*args, **kwargs),
        )
        if self.has_features:
            pt.F = self.F.to(*args, **kwargs)

        for k, v in self.cache.items():
            pt.cache[k] = v.to(*args, **kwargs)

        return pt

    def new_tensor(
        self, 
        coordinates: Optional[TensorType['N', 3, torch.float32]]=None,
        features: Optional[TensorType['N', '...', Any]]=None,
    ):
        if coordinates is None:
            coordinates = self._coordinates
        assert isinstance(coordinates, torch.Tensor), (
            f'coordinates must be a torch.Tensor, got {type(coordinates)}.'
        )
        assert self.num_points == coordinates.shape[0], (
            'Current and new coordinates must share first dimension, got '
            f'{self.num_points} and {coordinates.shape[0]}.'
        )
        assert features is None or isinstance(features, torch.Tensor), (
            f'features must be a torch.Tensor, got {type(features)}.'
        )
        assert features is None or self.num_points == features.shape[0], (
            'Current and new features must share first dimension, got '
            f'{self.num_points} and {features.shape[0]}.'
        )

        pt = PointTensor(
            coordinates=coordinates,
            features=features,
            batch_counts=self._batch_counts,
            batch_indices=self._batch_indices,
        )
        if coordinates.is_set_to(self._coordinates):
            pt.cache = self.cache

        return pt

    def __bool__(self) -> bool:
        return self.num_points > 0

    @property
    def has_features(self) -> bool:
        return self.F is not None

    def __iadd__(self, other):
        assert self._coordinates.is_set_to(other._coordinates), (
            'Addition only possible for PointTensor pair with equal coordinates'
        )
        assert self.has_features and other.has_features, (
            'Both operands must have features other than None.'
        )
        assert self.F.device == other.F.device, (
            'Both operands must be on same device, got '
            f'{self.F.device} and {other.F.device}.'
        )
        assert self.F.shape == other.F.shape, (
            'Shapes of features must match exactly, got '
            f'{self.F.shape} and {other.F.shape}.'
        )
        self.F += other.F

        return self

    def __add__(self, other):
        assert self._coordinates.is_set_to(other._coordinates), (
            'Addition only possible for PointTensor pair with equal coordinates'
        )
        assert self.has_features and other.has_features, (
            'Both operands must have features other than None.'
        )
        assert self.F.device == other.F.device, (
            'Both operands must be on same device, got '
            f'{self.F.device} and {other.F.device}.'
        )
        assert self.F.shape == other.F.shape, (
            'Shapes of features must match exactly, got '
            f'{self.F.shape} and {other.F.shape}.'
        )

        pt = PointTensor(
            coordinates=self._coordinates,
            features=self.F + other.F,
            batch_counts=self._batch_counts,
            batch_indices=self._batch_indices,
        )
        pt.cache = self.cache

        return pt

    @property
    def batch_counts(self) -> TensorType['B', torch.int64]:
        return self._batch_counts.long()

    @property
    def batch_indices(self) -> TensorType['N', torch.int64]:
        return self._batch_indices.long()

    @property
    def C(self) -> TensorType['N', 3, torch.float32]:
        return self._coordinates.detach()

    @property
    def F(self) -> Optional[TensorType['N', '...', Any]]:
        return self._features

    @F.setter
    def F(self, value: Optional[TensorType['N', '...', Any]]):
        assert value is None or isinstance(value, torch.Tensor), (
            f'value must be a torch.Tensor, got {type(value)}.'
        )
        assert value is None or self._coordinates.device == value.device, (
            'coordinates and value must be on same device, got '
            f'{self._coordinates.device} and {value.device}.'
        )
        assert value is None or self.num_points == value.shape[0], (
            'coordinates and value must share first dimension, got '
            f'{self.num_points} and {value.shape[0]}.'
        )

        self._features = value

    @property
    def batch_size(self) -> int:
        return self._batch_counts.shape[0]

    @property
    def num_points(self) -> int:
        return self._coordinates.shape[0]

    def __getitem__(
        self,
        index: Union[
            TensorType['N', torch.bool],
            TensorType['M', torch.int64],
        ],
    ):
        if not self:
            return self

        assert isinstance(index, torch.Tensor), (
            f'index must be torch.tensor, got {type(index)}.'
        )

        index = index.to(self._coordinates.device)
        if index.dtype == torch.bool:
            assert self.num_points == index.shape[0], (
                'PointTensor and torch.bool index must share first dimension, '
                f'got {self.num_points} and {index.shape[0]}.'
            )
        elif index.dtype == torch.int64:
            assert (
                index.numel() == 0
                or (0 <= index.amin() and index.amax() < self.num_points)
            ), f'Indices must be from [0, {self.num_points - 1}].'
            _, argsort = torch.sort(self._batch_indices[index], stable=True)
            index = index[argsort]
        else:
            raise ValueError(
                f'dtype of index must be bool or int64, got {index.dtype}.',
            )

        batch_counts = torch.bincount(
            self._batch_indices[index], minlength=self.batch_size,
        )

        pt = PointTensor(
            coordinates=self._coordinates[index],
            batch_counts=batch_counts,
        )
        if self.has_features:
            pt.F = self.F[index]

        return pt

    def batch(
        self,
        index: Union[
            int,
            TensorType['B', torch.int64],
        ],
    ):
        if not self:
            return self

        if isinstance(index, int):
            index = torch.tensor([index])

        assert isinstance(index, torch.Tensor), (
            f'index must be an int or torch.Tensor, got {type(index)}.'
        )
        assert (
            index.numel() == 0
            or (0 <= index.amin() and index.amax() < self.batch_size)
        ), f'Batch indices must be in [0, {self.batch_size - 1}].'

        index = index.unique().to(self._coordinates.device)
        mask = self._batch_indices.unsqueeze(-1) == index.unsqueeze(0)
        mask = torch.any(mask, dim=-1)

        pt = self[mask]
        pt._batch_counts = pt._batch_counts[index]
        _, pt._batch_indices = torch.unique_consecutive(pt.batch_indices, return_inverse=True)

        return pt

    def _fps(
        self,
        ratio: float=1.0,
        max_points: Optional[int]=None,
    ) -> Tuple[
        TensorType['M', torch.int64],
        TensorType['B', torch.int32],
    ]:
        if not self:
            sample_idxs, sample_batch_counts = torch.zeros(
                2, 0, dtype=torch.int64, device=self._coordinates.device,
            )

            return sample_idxs, sample_batch_counts.int()

        assert 0 <= ratio <= 1, (
            'ratio must be in [0, 1].'
        )
        assert max_points is None or 0 <= max_points, (
            'max_points must be at least 0.'
        )

        # get results from cache if they are already computed
        cache_keys = (
            f'sample_idxs__ratio={ratio}__max_points={max_points}',
            f'sample_batch_counts__ratio={ratio}__max_points={max_points}',
        )

        if all(k in self.cache for k in cache_keys):
            return tuple(self.cache[k] for k in cache_keys)

        # do farthest point sampling
        max_points = self.num_points if max_points is None else max_points
        sample_idxs, sample_batch_counts = farthestPointSampling(
            self._coordinates, self._batch_counts, ratio, max_points,
        )

        # save results in cache
        self.cache[cache_keys[0]] = sample_idxs
        self.cache[cache_keys[1]] = sample_batch_counts

        return sample_idxs, sample_batch_counts

    def farthest_point_sampling(
        self,
        ratio: float=1.0,
        max_points: Optional[int]=None,
    ) -> TensorType['M', torch.int64]:
        sample_idxs, _ = self._fps(ratio, max_points)

        return sample_idxs

    def downsample(self, ratio: float=0.25):
        downsample_idxs, downsample_batch_counts = self._fps(ratio)
        
        pt = PointTensor(
            coordinates=self._coordinates[downsample_idxs],
            batch_counts=downsample_batch_counts,
        )
        if self.has_features:
            pt.F = self.F[downsample_idxs]

        return pt

    def gridsample(
        self,
        voxel_size: float,
        start: float=0,
    ) -> TensorType['N', torch.int64]:
        if not self:
            return torch.zeros(
                0, dtype=torch.int64, device=self._coordinates.device,
            )

        start_coords = start + self._coordinates.amin(dim=0)

        coord_idxs = self._coordinates - start_coords
        coord_idxs /= voxel_size
        coord_idxs = coord_idxs.int()

        factors = coord_idxs.amax(dim=0) + 1
        factors = factors.cumprod(dim=-1, dtype=torch.int32)

        batched_coord_idxs = torch.column_stack(
            (coord_idxs, self._batch_indices),
        )
        batched_coord_idxs[:, 1:] *= factors

        voxel_idxs = batched_coord_idxs.sum(dim=1)
        
        return voxel_idxs

    def _cluster_cpu(
        self,
        max_neighbor_dist: float,
        min_points: int=1,
    ) -> TensorType['N', torch.int64]:
        max_range = (
            self._coordinates.amax(dim=0)
            - self._coordinates.amin(dim=0)
        ).amax()
        coordinates_np = torch.column_stack(
            (self._coordinates, 2 * max_range * self._batch_indices)
        ).cpu().numpy()

        dbscan = sklearn.cluster.DBSCAN(
            eps=max_neighbor_dist, min_samples=min_points, n_jobs=-1,
        )
        dbscan.fit(coordinates_np)
        cluster_idxs = torch.from_numpy(dbscan.labels_)
        cluster_idxs = cluster_idxs.to(self._coordinates.device)

        return cluster_idxs

    def cluster(
        self,
        max_neighbor_dist: float=0.04,
        min_points: int=1,
        weighted: bool=False,
        return_index: bool=False,
    ):
        if not self:
            return self

        # cluster points using DBSCAN
        if weighted:
            max_range = (
                self._coordinates.amax(dim=0)
                - self._coordinates.amin(dim=0)
            ).amax()
            coordinates = torch.column_stack(
                (self._coordinates, 2 * max_range * self._batch_indices)
            )
            dists = torch.linalg.norm(coordinates[None] - coordinates[:, None], axis=-1)
            cluster_idxs = teethland.cluster.wdbscan(
                dmatrix=dists,
                epsilon=max_neighbor_dist,
                mu=min_points,
                weights=self.F,
                noise=False,
            )
        else:
            cluster_idxs = self._cluster_cpu(max_neighbor_dist, min_points)

        if return_index:
            return cluster_idxs.long()

        # remove points identified as noise
        mask = cluster_idxs != -1
        coordinates = self._coordinates[mask]
        cluster_idxs = cluster_idxs[mask]
        batch_indices = self.batch_indices[mask]

        # compute cluster centroids by averaging
        cluster_idxs += (cluster_idxs.amax(dim=0) + 1) * batch_indices
        _, cluster_idxs = torch.unique(cluster_idxs, return_inverse=True)
        if weighted:
            coords_sum = scatter(
                src=coordinates * self.F[mask, None],
                index=cluster_idxs,
                dim=0,
                reduce='sum',
            )
            weights_sum = scatter(
                src=self.F[mask, None],
                index=cluster_idxs,
                dim=0,
                reduce='sum',
            )
            cluster_centroids = coords_sum / weights_sum
        else:
            cluster_centroids = scatter(
                src=coordinates,
                index=cluster_idxs,
                dim=0,
                reduce='mean',
            )

        # compute number of clusters per point cloud
        max_idxs = scatter(cluster_idxs + 1, batch_indices, dim_size=self.batch_size, reduce='max')
        min_idxs = scatter(cluster_idxs, batch_indices, dim_size=self.batch_size, reduce='min')

        if weighted:
            scores = scatter(
                src=self.F[mask],
                index=cluster_idxs,
                dim=0,
                reduce='max',
            )
            pt = PointTensor(
                coordinates=cluster_centroids,
                features=scores,
                batch_counts=max_idxs - min_idxs,
            )
        else:
            pt = PointTensor(
                coordinates=cluster_centroids,
                batch_counts=max_idxs - min_idxs,
            )
        
        return pt

    def _neighbors(
        self,
        other,
        k: Union[int, TensorType['M', torch.int64]],
        method: str,
        **kwargs,
    ) -> Union[
        Tuple[
            TensorType['M', 'k', torch.int64],
            TensorType['M', 'k', torch.float32],
        ],
        Tuple[
            TensorType['N', torch.int64],
            TensorType['N', torch.float32],
        ],
    ]:
        if not other:
            size = (0, k) if isinstance(k, int) else (0,)
            neighbor_idxs, sq_dists = torch.zeros(
                2, *size, device=self._coordinates.device,
            )

            return neighbor_idxs.long(), sq_dists

        ks = torch.full((other.num_points,), k) if isinstance(k, int) else k
        ks = ks.to(self._coordinates.device, torch.int32)

        min_points = min(self.num_points, self._batch_counts.amin())
        assert min_points >= ks.amax(), (
            f'{k} nearest neighbors were queried, but only '
            f'{min_points} points are available.'
        )
        assert self._batch_counts.shape[0] == other._batch_counts.shape[0], (
            'Both PointTensors must have same number of batch elements, got '
            f'{self._batch_counts.shape[0]} and {other._batch_counts.shape[0]}.'
        )

        if method == 'ball':
            method_fn = ballQuery
        elif method == 'knn':
            method_fn = kNNQuery
        else:
            raise ValueError(f'Nearest neighbors method unknown, got {method}.')

        neighbor_idxs, sq_dists = method_fn(
            self._coordinates,
            self._batch_counts,
            other._coordinates,
            other._batch_indices,
            ks,
            **kwargs,
        )

        if isinstance(k, int):
            neighbor_idxs = neighbor_idxs.reshape(-1, k)
            sq_dists = sq_dists.reshape(-1, k)

        return neighbor_idxs, sq_dists

    def neighbors(
        self,
        other=None,
        k: Union[int, TensorType['M', torch.int32]]=16,
        cache: bool=False,
        method: str='knn',
        **kwargs: dict,
    ) -> Union[
        Tuple[
            TensorType['M', 'k', torch.int64],
            TensorType['M', 'k', torch.float32],
        ],
        Tuple[
            TensorType['N', torch.int64],
            TensorType['N', torch.float32],
        ],
    ]:
        assert other is None or not cache, (
            'Cannot cache neighbor indices of PointTensor pair.'
        )

        cache_keys = ['__'.join([
            name,
            f'k={k}',
            f'method={method}',
            *[f'{k}={v}' for k, v in kwargs.items()],
        ]) for name in ['neighbor_idxs', 'sq_dists']]

        if all(key in self.cache for key in cache_keys):
            return tuple(self.cache[key] for key in cache_keys)

        other = self if other is None else other
        neighbor_idxs, sq_dists = self._neighbors(other, k, method, **kwargs)

        if other is self and cache:
            self.cache[cache_keys[0]] = neighbor_idxs
            self.cache[cache_keys[1]] = sq_dists

        if self._coordinates.requires_grad or other._coordinates.requires_grad:
            coords = (
                self._coordinates[neighbor_idxs]
                - (
                    other._coordinates.unsqueeze(1)
                    if neighbor_idxs.dim() == 2 else
                    other._coordinates.repeat_interleave(k, dim=0)
                )
            )
            sq_dists = torch.sum(coords ** 2, dim=-1)

        return neighbor_idxs, sq_dists

    def queryandgroup(self, other, k: int=16):
        assert self.has_features, 'self must have features other than None.'

        neighbor_idxs, _ = self._neighbors(other, k, 'knn')

        return other.new_tensor(
            features=self.F[neighbor_idxs],
        )

    def interpolate(self, other, k: int=3, dist_thresh: float=1e6, eps: float=1e-8):
        assert self.has_features, 'self must have features other than None.'
        assert self.F.dtype in [torch.int32, torch.int64, torch.float32, torch.float64], (
            f'self.F must have float32/64 or int32/64, got {self.F.dtype}.'
        )

        neighbor_idxs, sq_dists = self._neighbors(other, k, 'knn')
        weights = 1 / (sq_dists + eps)

        if self.F.dtype in [torch.float32, torch.float64]:
            weights = weights.to(self.F)
            features = torch.einsum(
                'Mk...,Mk,M->M...',
                self.F[neighbor_idxs],
                weights,
                1 / weights.sum(dim=1),
            )
            mask = torch.all(torch.sqrt(sq_dists) < dist_thresh, dim=1)
            features[~mask] = self.F.amin()
        elif self.F.dtype in [torch.int32, torch.int64]:
            weights = weights.float()
            features = torch.full((neighbor_idxs.shape[0],), self.F.amin()).to(self.F)
            max_interp = torch.zeros_like(features).float()
            for value in torch.unique(self.F)[1:]:
                interp = torch.einsum(
                    'Mk...,Mk,M->M...',
                    (self.F == value).float()[neighbor_idxs],
                    weights,
                    1 / weights.sum(dim=1),
                )
                features = torch.where(
                    (interp >= 0.5) & (interp > max_interp), value, features,
                )
                max_interp = torch.maximum(max_interp, interp)

        return other.new_tensor(features=features)

    def __repr__(self) -> str:
        coordinates = ''.join([
            'tensor(',
            f'shape={tuple(self._coordinates.shape)}, ',
            f'dtype={self._coordinates.dtype}',
            ')',
        ])

        if self.has_features:
            features = ''.join([
                'tensor(',
                f'shape={tuple(self.F.shape)}, ',
                f'dtype={self.F.dtype}',
                ')',
            ])
        else:
            features = 'None'

        if self.cache:
            cache = '\n'.join([
                '(',
                *[f'        \'{k}\',' for k in self.cache],
                '    )',
            ])
        else:
            cache = '()'

        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    coordinates={coordinates},',
            f'    features={features},',
            f'    batch_counts={self._batch_counts.cpu()},',
            f'    device=\'{self._coordinates.device}\',',
            f'    cache={cache},',
            ')',
        ])


def collate(
    tensors: List[PointTensor],
    batch_counts: TensorType['B', torch.int64],
    index: TensorType['N', torch.int64],
) -> PointTensor:
    coordinates = torch.cat([t._coordinates for t in tensors])
    
    pt = PointTensor(
        coordinates=coordinates[index],
        batch_counts=batch_counts,
    )

    if not all(t.has_features for t in tensors):
        return pt

    if len(set(t.F.dim() for t in tensors)) != 1:
        return pt

    features = torch.cat([t.F for t in tensors])
    pt.F = features[index]

    return pt


def cat(tensors: List[PointTensor]) -> PointTensor:
    batch_counts = [pt.batch_counts for pt in tensors]
    batch_counts = torch.sum(torch.stack(batch_counts), dim=0)

    batch_indices = torch.cat([pt.batch_indices for pt in tensors])
    _, index = torch.sort(batch_indices, stable=True)

    return collate(tensors, batch_counts, index)


def stack(tensors: List[PointTensor]) -> PointTensor:
    batch_counts = torch.cat([pt.batch_counts for pt in tensors])
    batch_counts = batch_counts[batch_counts > 0]

    index = torch.arange(torch.sum(batch_counts), device=batch_counts.device)
    
    return collate(tensors, batch_counts, index)
