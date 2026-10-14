import json
from pathlib import Path
from typing import Any, Callable, Dict, Union

import numpy as np
from numpy.typing import NDArray
import pymeshlab

from teethland.data.datasets.base import MeshDataset
import teethland.data.transforms as T


class TeethSegDataset(MeshDataset):
    """Dataset to load intraoral scans with teeth segmentations."""

    MEAN = None  # [2.0356, -0.6506, -90.0502]
    STD = 17.3281  # mm

    def __init__(
        self,
        clean: bool,
        pre_transform: Callable[..., Dict[str, Any]]=dict,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.ZScoreNormalize(self.MEAN, self.STD),
            T.PoseNormalize() if clean else dict,
            T.InstanceCentroids(),
            pre_transform,
        )

        super().__init__(pre_transform=pre_transform, **kwargs)

        self.clean = clean

    def load_jaw(self, file: Path) -> str:
        try:
            if 'lower' in file.stem.lower():
                return 'lower'
            elif 'upper' in file.stem.lower():
                return 'upper'
            jaw = file.stem.split('_')[1]
        except Exception:
            with open(self.root / file, 'r') as f:
                jaw = f.readline()[2:-1]

        return jaw

    def load_scan(
        self,
        file: Path,
    ) -> Dict[str, Union[bool, int, NDArray[Any]]]:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.root / file))
        ms.meshing_remove_duplicate_vertices()
        # if self.clean:
        #     ms.meshing_repair_non_manifold_edges()
        #     ms.meshing_close_holes(maxholesize=130)  # ~20mm boundary

        mesh = ms.current_mesh()
        mesh.compact()

        return {
            'scan_file': file.as_posix(),
            'is_lower': self.load_jaw(file) in ['lower', 'mandible'],
            'points': mesh.vertex_matrix(),
            'triangles': mesh.face_matrix(),
            'normals': mesh.vertex_normal_matrix(),
            'point_count': mesh.vertex_number(),
            'triangle_count': mesh.face_number(),
        }

    def load_annotation(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.int64]]:
        with open(self.root / file, 'rb') as f:
            annotation = json.load(f)

        instances = np.array(annotation['instances'])
        _, instances, counts = np.unique(instances, return_inverse=True, return_counts=True)

        labels = np.array(annotation['labels'])
        instances[(counts < 20)[instances]] = 0
        labels[(counts < 20)[instances]] = 0

        return {
            **(
                {}
                if 'confidences' not in annotation else
                {'confidences': np.array(annotation['confidences'])}
            ),
            'labels': labels,
            'instances': instances,
        }


class TeethLandDataset(TeethSegDataset):
    """Dataset to load intraoral scans with teeth segmentations and landmarks."""

    landmark_classes = {
        'Mesial': 0,
        'Distal': 1,
        'FacialPoint': 2,
        'OuterPoint': 3,
        'InnerPoint': 4,
        'Cusp': 5,
    }

    def __init__(
        self,
        seg_root: Path,
        landmarks_root: Path,
        include_cusps: bool,
        to_left_right: bool,
        separate_front_posterior: bool,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            root=seg_root,
            pre_transform=T.Compose(
                T.MatchLandmarksAndTeeth(),
                T.StructureLandmarks(
                    include_cusps=include_cusps,
                    to_left_right=to_left_right,
                    separate_front_posterior=separate_front_posterior,
                ),
            ),
            **kwargs,
        )
        
        self.landmarks_root = landmarks_root

    def load_annotation(
        self,
        seg_file: Path,
        landmark_file: Path,
    ) -> Dict[str, NDArray[np.int64]]:
        out_dict = super().load_annotation(seg_file)

        with open(self.landmarks_root / landmark_file, 'rb') as f:
            landmark_annotation = json.load(f)

        landmark_coords, landmark_classes = [], []
        for landmark in landmark_annotation['objects']:
            landmark_classes.append(self.landmark_classes[landmark['class']])
            landmark_coords.append(landmark['coord'])

        return {
            **out_dict,
            'landmark_coords': np.array(landmark_coords),
            'landmark_classes': np.array(landmark_classes),
        }
