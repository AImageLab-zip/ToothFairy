import copy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
import open3d
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.stats import truncnorm
from sklearn.decomposition import PCA
import torch
from torch_scatter import scatter_mean, scatter_min
from torchtyping import TensorType


class Compose:

    def __init__(
        self,
        *transforms: List[Callable[..., Dict[str, Any]]],
    ):
        self.transforms = transforms

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        for t in self.transforms:
            data_dict = t(**data_dict)
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            *[
                '    ' + repr(t).replace('\n', '\n    ') + ','
                for t in self.transforms
            ],
            ')',
        ])


class ToTensor:

    def __init__(
        self,
        bool_dtypes: List[np.dtype]=[bool, np.bool_],
        int_dtypes: List[np.dtype]=[int, np.int16, np.uint16, np.int32, np.int64],
        float_dtypes: List[np.dtype]=[float, np.float32, np.float64],
    ) -> None:
        self.bool_dtypes = bool_dtypes
        self.int_dtypes = int_dtypes
        self.float_dtypes = float_dtypes

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, TensorType[..., Any]]:
        for k, v in data_dict.items():
            dtype = v.dtype if isinstance(v, np.ndarray) else type(v)
            if dtype in self.bool_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.bool)
            elif dtype in self.int_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.int64)            
            elif dtype in self.float_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.float32)
            elif dtype == str:
                data_dict[k] = v
            else:
                raise ValueError(
                    'Expected a scalar or list or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class RandomZAxisRotate:

    def __init__(
        self,
        max_degrees: float=45,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max_angle = max_degrees / 180 * np.pi
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        angle = self.rng.uniform(-self.max_angle, self.max_angle)
        cosval, sinval = np.cos(angle), np.sin(angle)

        R = np.array([
            [cosval,  -sinval, 0],
            [sinval, cosval, 0],
            [0,       0,      1],
        ])
        data_dict['points'] = points @ R.T

        if 'normals' in data_dict:
            data_dict['normals'] = data_dict['normals'] @ R.T

        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] = data_dict['landmark_coords'] @ R.T
        if 'instance_landmarks' in data_dict:
            data_dict['instance_landmarks'] = data_dict['instance_landmarks'] @ R.T
        if 'instance_centroids' in data_dict:
            data_dict['instance_centroids'] = data_dict['instance_centroids'] @ R.T
        
        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_angle: {self.max_angle * 180 / np.pi} degrees,',
            ')',
        ])


class RandomScale(object):

    def __init__(
        self,
        low: float=0.95,
        high: float=1.05,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        scale = self.rng.uniform(self.low, self.high)
        data_dict['points'] = points * scale

        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] *= scale
        if 'instance_landmarks' in data_dict:
            data_dict['instance_landmarks'] *= scale
        if 'instance_centroids' in data_dict:
            data_dict['instance_centroids'] *= scale
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    low: {self.low},',
            f'    high: {self.high},',
            ')',
        ])


class RandomJitter(object):

    def __init__(
        self,
        sigma: float=0.005,
        clip: float=0.02,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.sigma = sigma
        self.clip = clip / sigma
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,        
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        jitter = truncnorm.rvs(
            -self.clip, self.clip,
            scale=self.sigma,
            size=points.shape,
            random_state=self.rng,
        )
        data_dict['points'] = points + jitter

        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    sigma: {self.sigma},',
            f'    clip: {self.clip},',
            ')',
        ])


class RandomAxisFlip(object):

    def __init__(
        self,
        axis: int=0,
        prob: float=0.5,
        landmark_flip_idxs: Optional[List[int]]=None,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.axis = axis
        self.prob = prob
        self.landmark_flip_idxs = landmark_flip_idxs
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.rng.random() < self.prob:
            coords = points[:, self.axis]
            points[:, self.axis] = coords.max() + coords.min() - coords

            if 'normals' in data_dict:
                data_dict['normals'][:, self.axis] *= -1

            if 'landmark_coords' in data_dict:
                landmarks = data_dict['landmark_coords'][:, self.axis]
                data_dict['landmark_coords'][:, self.axis] = (
                    coords.max() + coords.min() - landmarks
                )
                data_dict['landmark_classes'] = self.landmark_flip_idxs[data_dict['landmark_classes']]
                
            if 'instance_landmarks' in data_dict:
                landmarks_mask = np.any(data_dict['instance_landmarks'] != 0, axis=-1)
                landmarks = data_dict['instance_landmarks'][..., self.axis]
                data_dict['instance_landmarks'][..., self.axis] = np.where(
                    landmarks_mask, coords.max() + coords.min() - landmarks, 0.0,
                )
                flip_idxs = self.landmark_flip_idxs[:data_dict['instance_landmarks'].shape[1]]
                data_dict['instance_landmarks'] = data_dict['instance_landmarks'][:, flip_idxs]

            if 'instance_centroids' in data_dict:
                centroids = data_dict['instance_centroids'][:, self.axis]
                data_dict['instance_centroids'][:, self.axis] = (
                    coords.max() + coords.min() - centroids
                )

        data_dict['points'] = points

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    prob: {self.prob},',
            ')',
        ])


class PoseNormalize:

    def __call__(
        self,
        points: NDArray[Any],
        normals: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        pca = PCA()
        pca.fit(points)
        R = pca.components_
        
        # disallow reflections
        if np.linalg.det(R) < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]) @ R

        # rotate 180 degrees around y-axis if points are upside down
        if (normals @ R.T)[:, 2].mean() < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]) @ R

        # rotate 180 degrees around z-axis if points are flipped
        if (points[(points @ R.T)[:, 0] > 1] @ R.T).mean(0)[1] < 0:
            R = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]) @ R
        
        # rotate points to principal axes of decreasing explained variance
        data_dict['points'] = points @ R.T
        data_dict['normals'] = normals @ R.T
        
        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] = data_dict['landmark_coords'] @ R.T

        T = np.eye(4)
        T[:3, :3] = R
        data_dict['affine'] = T @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ZScoreNormalize:

    def __init__(
        self,
        mean: Optional[Union[float, ArrayLike]]=None,
        std: Optional[Union[float, ArrayLike]]=None,
    ) -> None:
        self.mean_ = mean
        self.std_ = std

    def mean(
        self,
        points: NDArray[Any],
    ) -> Union[float, ArrayLike]:
        if self.mean_ is None:
            return points.mean(axis=0)

        return self.mean_

    def std(
        self,
        points: NDArray[Any],
    ) -> Union[float, ArrayLike]:
        if self.std_ is None:
            return points.std(axis=0)

        return self.std_

    def affine(
        self,
        points: NDArray[Any],
    ) -> NDArray[Any]:
        trans = np.eye(4)
        trans[:3, 3] -= self.mean(points)

        scale = np.eye(4)
        scale[np.diag_indices(3)] /= self.std(points)

        return scale @ trans

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        affine = self.affine(points)
        points_hom = np.column_stack((points, np.ones_like(points[:, 0])))
        data_dict['points'] = (points_hom @ affine.T)[:, :3]

        if 'landmark_coords' in data_dict:
            landmarks = data_dict['landmark_coords']
            landmarks_hom = np.column_stack((landmarks, np.ones_like(landmarks[:, 0])))
            data_dict['landmark_coords'] = (landmarks_hom @ affine.T)[:, :3]

        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    mean={self.mean_},',
            f'    std={self.std_},',
            ')',
        ])


class XYZAsFeatures:

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['points'] = points

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], points), axis=-1,
            )
        else:
            data_dict['features'] = points

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class NormalAsFeatures:

    def __init__(
        self,
        eps: float=1e-8,
    ):
        self.eps = eps

    def __call__(
        self,
        normals: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + self.eps

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], normals), axis=-1,
            )
        else:
            data_dict['features'] = normals        
            
        data_dict['normals'] = normals

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class CentroidOffsetsAsFeatures:

    def __init__(
        self,
        eps: float=1e-8,
    ):
        self.eps = eps

    def __call__(
        self,
        points: NDArray[Any],
        centroids: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        diffs = points - centroids[:, None]

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], diffs), axis=-1,
            )
        else:
            data_dict['features'] = diffs        
            
        data_dict['points'] = points
        data_dict['centroids'] = centroids

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class UniformDensityDownsample:

    def __init__(
        self,
        voxel_size: float,
        inplace: bool=False,
    ) -> None:
        self.voxel_size = voxel_size
        self.inplace = inplace
    
    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ):
        pos_points = points - points.min(axis=0)
        discrete_coords = (pos_points / self.voxel_size).astype(int)

        voxel_centers = self.voxel_size * (discrete_coords + 0.5)
        sq_dists = np.sum((pos_points - voxel_centers) ** 2, axis=-1)

        factors = discrete_coords.max(axis=0) + 1
        factors = factors.cumprod() / factors
        vertex_voxel_idxs = np.sum(discrete_coords * factors, axis=-1)
        _, vertex_voxel_idxs = np.unique(
            vertex_voxel_idxs, return_inverse=True,
        )

        argmin = scatter_min(
            src=torch.from_numpy(sq_dists),
            index=torch.from_numpy(vertex_voxel_idxs),
        )[1].numpy()

        if 'ud_downsample_idxs' in data_dict:
            data_dict['ud_downsample_idxs_1'] = data_dict['ud_downsample_idxs'] 
            data_dict['ud_downsample_count_1'] = data_dict['ud_downsample_count']
            data_dict['ud_downsample_idxs_2'] = argmin
            data_dict['ud_downsample_count_2'] = argmin.shape[0]
        else:
            data_dict['ud_downsample_idxs'] = argmin
            data_dict['ud_downsample_count'] = argmin.shape[0]

        if not self.inplace:
            data_dict['points'] = points
            return data_dict

        data_dict['points'] = points[argmin]
        data_dict['point_count'] = argmin.shape[0]

        for key in ['features', 'labels', 'instances', 'normals']:
            if key not in data_dict:
                continue

            data_dict[key] = data_dict[key][argmin]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    voxel_size={self.voxel_size},',
            f'    inplace={self.inplace},',
            ')',
        ])



class BoundaryAwareDownsample(UniformDensityDownsample):

    def __init__(
        self,
        voxel_size: float,
        sample_ratio: float,
        min_points: int=10_000,
        inplace: bool=False,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        super().__init__(voxel_size, inplace)

        self.sample_ratio = sample_ratio
        self.min_points = min_points
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        confidences: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # first get sample of points with uniform density
        data_dict = super().__call__(**data_dict)
        sample_idxs = data_dict['ud_downsample_idxs']
        sample_count = data_dict['ud_downsample_count']

        # determine boundary-aware sample of remaining points
        rand_idxs = self.rng.choice(
            a=sample_count,
            size=max(int(sample_count * self.sample_ratio), self.min_points),
            replace=False,
            p=softmax(-np.abs(confidences[sample_idxs] / 4)),
        )

        data_dict['ba_downsample_idxs'] = sample_idxs[rand_idxs]
        data_dict['ba_downsample_count'] = rand_idxs.shape[0]

        if not self.inplace:
            data_dict['confidences'] = confidences
            return data_dict

        data_dict['confidences'] = confidences[sample_idxs][rand_idxs]
        data_dict['points'] = data_dict['points'][rand_idxs]
        data_dict['point_count'] = rand_idxs.shape[0]

        for key in ['features', 'labels', 'instances']:
            if key not in data_dict:
                continue

            data_dict[key] = data_dict[key][rand_idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    voxel_size={self.voxel_size},',
            f'    sample_ratio={self.sample_ratio},',
            f'    min_points={self.min_points},',
            f'    inplace={self.inplace},',
            ')',
        ])
    

class InstanceCentroids:

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'labels' not in data_dict:
            data_dict['points'] = points

            return data_dict

        labels, instances = data_dict['labels'], data_dict['instances']

        if labels.shape[0] < points.shape[0]:
            print('different shape!')
            diff = points.shape[0] - labels.shape[0]
            labels = np.concatenate((labels, [0]*diff))
            instances = np.concatenate((instances, [0]*diff))
            
            data_dict['labels'] = labels
            data_dict['instances'] = instances
        elif labels.shape[0] > points.shape[0]:
            print('different shape!')
            diff = labels.shape[0] - points.shape[0]
            labels = labels[:-diff]
            instances = instances[:-diff]
            
            data_dict['labels'] = labels
            data_dict['instances'] = instances

        instance_centroids = scatter_mean(
            src=torch.from_numpy(points),
            index=torch.from_numpy(instances),
            dim=0,
        ).numpy()

        instance_point_idxs = np.bincount(instances).cumsum() - 1
        instance_labels = labels[np.argsort(instances)[instance_point_idxs]]

        data_dict['points'] = points
        data_dict['instance_centroids'] = instance_centroids
        data_dict['instance_labels'] = instance_labels
        data_dict['instance_count'] = instance_labels.shape[0]

        return data_dict
    
    def __repr__(self) -> int:
        return self.__class__.__name__ + '()'


class MatchLandmarksAndTeeth:

    def __init__(
        self,
        move: float=0.04,
    ):
        from teethland.data.datasets import TeethLandDataset
        self.landmark_classes = TeethLandDataset.landmark_classes
        self.move = move

    def move_landmarks(
        self,
        landmark_coords: NDArray[Any],
        landmark_classes: NDArray[Any],
        points: NDArray[Any],
        labels: NDArray[Any],
        instances: NDArray[Any],
        **data_dict,
    ):
        moved_coords = landmark_coords.copy()

        mesial = landmark_classes == self.landmark_classes['Mesial']
        distal = landmark_classes == self.landmark_classes['Distal']

        # move mesial landmarks of central incisors
        mesial_coords = landmark_coords[mesial]
        dists = np.linalg.norm(mesial_coords[None] - mesial_coords[:, None], axis=-1)
        dists = np.where(dists > 0, dists, 1e6)
        if dists.min() < 4 * self.move:
            mesial_idxs = np.unravel_index(dists.argmin(), dists.shape)
            mesial_idxs = np.nonzero(mesial)[0][list(mesial_idxs)]
            mesial[mesial_idxs] = False
            if landmark_coords[mesial_idxs][:, 1].ptp() < 2 * self.move:
                if landmark_coords[mesial_idxs[0], 0] < landmark_coords[mesial_idxs[1], 0]:
                    moved_coords[mesial_idxs[0], 0] -= self.move
                    moved_coords[mesial_idxs[1], 0] += self.move
                else:
                    moved_coords[mesial_idxs[1], 0] -= self.move
                    moved_coords[mesial_idxs[0], 0] += self.move

        # move pairs of mesial and distal landmarks
        mesial_coords = moved_coords[mesial]
        distal_coords = moved_coords[distal]
        dists = np.linalg.norm(mesial_coords[:, None, :2] - distal_coords[None, :, :2], axis=-1)
        for mesial_idx, distal_idx in zip(*np.nonzero(dists < 5 * self.move)):
            diff = mesial_coords[mesial_idx, :2] - distal_coords[distal_idx, :2]
            diff /= np.linalg.norm(diff)

            mesial_idx = np.nonzero(mesial)[0][mesial_idx]
            distal_idx = np.nonzero(distal)[0][distal_idx]
            
            moved_coords[mesial_idx, :2] += self.move * diff
            moved_coords[distal_idx, :2] -= self.move * diff

        # move inner points of front elements away from center
        max_y = points[(labels > 0) & (labels % 10 <= 3), 1].max() - self.move
        for i, (coords, cls) in enumerate(zip(landmark_coords, landmark_classes)):
            if coords[1] > max_y or cls != self.landmark_classes['InnerPoint']:
                continue

            dists = np.linalg.norm(points - coords, axis=-1)
            min_dists = scatter_min(
                src=torch.from_numpy(dists),
                index=torch.from_numpy(instances),
                dim=0,
            )[0][1:].numpy()
            if (min_dists < self.move).sum() > 1:
                continue

            diff = coords[:2] / np.linalg.norm(coords[:2])
            moved_coords[i, :2] = coords[:2] + self.move * diff

        return moved_coords  

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'labels' not in data_dict:
            data_dict['points'] = points

            return data_dict
        
        # match each landmark to an instance after moving it inward
        moved_coords = self.move_landmarks(points=points, **data_dict)        
        tooth_points = np.where(data_dict['labels'][:, None] > 0, points, 1e6)
        dists = np.linalg.norm(
            tooth_points[None] - moved_coords[:, None],
        axis=-1)
        instance_idxs = data_dict['instances'][dists.argmin(1)]

        data_dict['points'] = points
        data_dict['landmark_instances'] = instance_idxs

        return data_dict
    
    def __repr__(self) -> int:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    move={self.move},',
            ')',
        ])
    

class StructureLandmarks:

    def __init__(
        self,
        include_cusps: bool=False,
        to_left_right: bool=False,
        separate_front_posterior: bool=True,
    ):
        from teethland.data.datasets import TeethLandDataset
        self.landmark_classes = TeethLandDataset.landmark_classes
        self.include_cusps = include_cusps
        self.to_left_right = to_left_right
        self.separate_front_posterior = separate_front_posterior

        assert not to_left_right or not separate_front_posterior

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'labels' not in data_dict:
            data_dict['points'] = points

            return data_dict
        
        centroids = data_dict['instance_centroids']
        land_coords = data_dict['landmark_coords']
        land_classes = data_dict['landmark_classes']
        land_instances = data_dict['landmark_instances']
        
        num_landmarks = 5 + 2 * self.separate_front_posterior + 5 * self.include_cusps
        out = np.zeros((centroids.shape[0], num_landmarks, 3))
        for i in range(centroids.shape[0]):
            instance_mask = land_instances == i
            fdi = data_dict['instance_labels'][i]

            # no landmarks for this instance
            if not np.any(instance_mask):
                if i > 0: 
                    print(data_dict['scan_file'], fdi, 'no landmarks')
                continue
            
            # process the non-cusp landmarks
            for k, v in self.landmark_classes.items():
                if k == 'Cusp':
                    continue
                
                num_landmarks = (land_classes[instance_mask] == v).sum()
                if num_landmarks != 1:
                    print(data_dict['scan_file'], fdi, k, num_landmarks)
                    if num_landmarks == 0: continue

                out[i, v] = land_coords[instance_mask & (land_classes == v)][-1]

            # have more consistent mesial/distal landmarks as left-to-right landmarks
            if self.to_left_right and fdi // 10 in [1, 4]:
                mesial, distal = [self.landmark_classes[k] for k in ['Mesial', 'Distal']]
                out[i, [mesial, distal]] = out[i, [distal, mesial]]
            if self.separate_front_posterior and (fdi % 10) <= 3:
                mesial, distal = [self.landmark_classes[k] for k in ['Mesial', 'Distal']]
                if fdi // 10 in [1, 4]:
                    left, right = distal, mesial
                else:
                    left, right = mesial, distal
                    
                out[i, [-2, -1]] = out[i, [left, right]]
                out[i, [left, right]] = 0

            # process the cusp landmarks
            cusp_class = self.landmark_classes['Cusp']
            cusps = land_classes[instance_mask] == cusp_class
            if not self.include_cusps or cusps.sum() == 0:  # incisors, canines
                continue
            
            # remove 6th cusp if it is present
            cusp_coords = land_coords[instance_mask][cusps] - centroids[i]
            if cusps.sum() == 6:  # two extra cusps
                print(data_dict['scan_file'], fdi, '6 cusps')
                dists = np.linalg.norm(cusp_coords[None] - cusp_coords[:, None], axis=-1)
                dists = np.where(dists > 0, dists, 1e6)
                pair = np.unravel_index(dists.argmin(), dists.shape)
                remove_idx = pair[0] if cusp_coords[pair[0], 2] < cusp_coords[pair[1], 2] else pair[1]
                cusps[np.nonzero(cusps)[0][remove_idx]] = False
                cusp_coords = np.concatenate((cusp_coords[:remove_idx], cusp_coords[remove_idx + 1:]))

            if (fdi % 10) in [2, 3, 4, 5] and cusps.sum() in [1, 2]:  # premolars
                scores = np.stack((
                    cusp_coords[:, :2] @ [1, 0],  # left
                    cusp_coords[:, :2] @ [-1, 0],  # right
                ))

                dir_idxs, cusp_idxs = linear_sum_assignment(scores, maximize=True)
                for dir_idx, cusp_idx in zip(dir_idxs, cusp_idxs):
                    out[i, cusp_class + dir_idx] = land_coords[instance_mask][cusps][cusp_idx]
            elif (fdi % 10) in [6, 7, 8] or cusps.sum() >= 3:  # molars
                assert cusps.sum() in [1, 2, 3, 4, 5], f'Can only have 1 to 5 cusps for a molar, found {cusps.sum()}!'

                scores = np.stack((
                    cusp_coords[:, :2] @ [1, 1],  # top left
                    cusp_coords[:, :2] @ [-1, 1],  # top right
                    cusp_coords[:, :2] @ [1, -1],  # bottom left
                    cusp_coords[:, :2] @ [-1, -1],  # bottom right
                    cusp_coords[:, :2] @ [0, 0],  # extra
                ))
                
                dir_idxs, cusp_idxs = linear_sum_assignment(scores, maximize=True)
                for dir_idx, cusp_idx in zip(dir_idxs, cusp_idxs):
                    out[i, cusp_class + dir_idx] = land_coords[instance_mask][cusps][cusp_idx]
            else:
                raise ValueError('Could not process cusps')
            
            if (fdi // 10) in [2, 3]:
                out[i, [cusp_class, cusp_class + 1]] = out[i, [cusp_class + 1, cusp_class]]
                out[i, [cusp_class + 2, cusp_class + 3]] = out[i, [cusp_class + 3, cusp_class + 2]]
            
        data_dict['points'] = points
        data_dict['instance_landmarks'] = out

        return data_dict
    
    def __repr__(self) -> int:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    include_cusps={self.include_cusps},',
            f'    to_left_right={self.to_left_right},',
            f'    separate_front_posterior={self.separate_front_posterior},',
            ')',
        ])
    

class GenerateProposals:

    def __init__(
        self,
        proposal_points: int,
        max_proposals: int,
        rng: Optional[np.random.Generator]=None,
    ):
        self.proposal_points = proposal_points
        self.max_proposals = max_proposals
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'instances' not in data_dict:
            data_dict['points'] = points

            return data_dict
        
        unique_instances = np.unique(data_dict['instances'])[1:]
        instance_idxs = np.sort(self.rng.choice(
            unique_instances,
            size=min(self.max_proposals, unique_instances.shape[0]),
            replace=False,
        ))

        centroids = data_dict['instance_centroids'][instance_idxs]
        if 'instance_landmarks' in data_dict:
            instance_landmarks = data_dict['instance_landmarks'][instance_idxs]
            landmark_mask = np.any(instance_landmarks != 0, axis=-1, keepdims=True)
            instance_landmarks = np.where(landmark_mask, instance_landmarks - centroids[:, None], 0.0)
            data_dict['instance_landmarks'] = instance_landmarks
        
        if 'landmark_coords' in data_dict:
            coords = data_dict['landmark_coords']
            classes = data_dict['landmark_classes']
            instances = data_dict['landmark_instances']
            landmark_mask = np.any(instances[None] == instance_idxs[:, None], axis=0)
            landmark_idxs = np.nonzero(landmark_mask)[0][np.argsort(instances[landmark_mask])]

            instance_map = np.full((instance_idxs.max() + 1,), -1)
            instance_map[instance_idxs] = np.arange(instance_idxs.shape[0])
            instances = instance_map[instances[landmark_idxs]]

            landmarks = np.column_stack((
                coords[landmark_idxs],
                classes[landmark_idxs],
                instances,
            ))
            data_dict['landmarks'] = landmarks

        dists = np.linalg.norm(points[None] - centroids[:, None], axis=-1)
        point_idxs = np.argsort(dists, axis=1)[:, :self.proposal_points]
        fg_masks = data_dict['instances'][point_idxs] == instance_idxs[:, None]
        assert np.all(np.any(fg_masks, 1))

        data_dict['points'] = points[point_idxs]
        data_dict['normals'] = data_dict['normals'][point_idxs]
        data_dict['labels'] = fg_masks.astype(int)
        data_dict['centroids'] = centroids
        data_dict['point_count'] = np.array([self.proposal_points]).repeat(centroids.shape[0])
        data_dict['instance_count'] = centroids.shape[0]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    proposal_points={self.proposal_points},',
            f'    max_proposals={self.max_proposals},',
            ')',
        ])
