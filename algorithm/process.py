from pathlib import Path
import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

import PosPadUNet3D


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class SimpleNet(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        avg = x.double().mean()
        return torch.where(x > avg, 1, 0)

class Toothfairy_algorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/inferior-alveolar-canal/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        self._output_path.mkdir(parents=True)

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image):
        input_array = sitk.GetArrayFromImage(input_image)

        input_tensor = torch.from_numpy(input_array.astype(np.float32))
        input_tensor = input_tensor[None, ...].to(get_default_device())

        net = PosPadUNet3D.PosPadUNet3D(1, [10, 10, 10], 1)
        net = net.to(get_default_device())

        input_tensor = PosPadUNet3D.preprocessing(input_tensor)
        output = net(input_tensor)

        output = output.detach().cpu().numpy().squeeze().astype(np.uint8)
        output = np.where(output > 0.5, 1, 0)
        output = sitk.GetImageFromArray(output)

        return output


if __name__ == "__main__":
    Toothfairy_algorithm().process()
