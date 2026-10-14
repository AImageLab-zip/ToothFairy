from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethinstseg import TeethInstSegDataModule
from teethland.data.datasets import TeethLandDataset
import teethland.data.transforms as T


class TeethMixedSegDataModule(TeethInstSegDataModule):
    """Implements data module that loads meshes and segmentations with mixed dentitions."""

    @property
    def num_classes(self) -> int:
        return 12 if self.filter or not self.distinguish_upper_lower else 24
    
    def _files(
        self,
        stage: str,
        exclude: List[str]=[],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        return super()._files(stage, exclude)
    
    def teeth_labels_to_classes(
        self,
        labels: Union[
            NDArray[np.int64],
            TensorType['N', torch.int64],
        ]
    ) -> Union[
        NDArray[np.int64],
        TensorType['N', torch.int64],
    ]:
        if isinstance(labels, np.ndarray):
            classes = labels.copy()
        elif isinstance(labels, torch.Tensor):
            classes = labels.clone()
        else:
            raise ValueError(
                f'Expected np.ndarray or torch.Tensor, got {type(labels)}.',
            )

        classes[(51 <= labels) & (labels <= 55)] -= 44
        classes[(11 <= labels) & (labels <= 17)] -= 11
        classes[labels == 18] = 6
        classes[(61 <= labels) & (labels <= 65)] -= 54
        classes[(21 <= labels) & (labels <= 27)] -= 21
        classes[labels == 28] = 6

        if self.filter == 'lower' or not self.distinguish_upper_lower:
            classes[(71 <= labels) & (labels <= 75)] -= 64
            classes[(31 <= labels) & (labels <= 37)] -= 31
            classes[labels == 38] = 6
            classes[(81 <= labels) & (labels <= 85)] -= 74
            classes[(41 <= labels) & (labels <= 47)] -= 41
            classes[labels == 48] = 6
        else:
            classes[(71 <= labels) & (labels <= 75)] -= 52
            classes[(31 <= labels) & (labels <= 37)] -= 19
            classes[labels == 38] = 18
            classes[(81 <= labels) & (labels <= 85)] -= 62
            classes[(41 <= labels) & (labels <= 47)] -= 29
            classes[labels == 48] = 18
        
        return classes
