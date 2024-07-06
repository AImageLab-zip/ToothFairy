from pathlib import Path
import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np
import torchio as tio

from torch.utils.data import DataLoader

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class Toothfairy2_algorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/oral-pharyngeal-segmentation/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image):
        input_array = sitk.GetArrayFromImage(input_image)

        input_tensor = torch.from_numpy(input_array.astype(np.float32))
        input_tensor = input_tensor[None, ...].to(get_default_device())

        output = input_tensor.squeeze(0)
        output = (output > 1500).int()
        output = output.detach().cpu().numpy().squeeze().astype(np.uint8)
        output = sitk.GetImageFromArray(output)

        return output


if __name__ == "__main__":
    Toothfairy2_algorithm().process()
