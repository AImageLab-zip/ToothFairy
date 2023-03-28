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

        input_tensor = torch.from_numpy(input_array.astype(np.int32))
        # print(f'input_tensor: {input_tensor.max()}, {input_tensor.min()}')
        net = SimpleNet()
        net = net.to(get_default_device())
        output = net(input_tensor)
        # print(f'output: {output.max()}, {output.min()}')
        output = output.detach().cpu().numpy()
        output = sitk.GetImageFromArray(output.astype(input_array.dtype))
        output.SetOrigin(input_image.GetOrigin())
        output.SetSpacing(input_image.GetSpacing())

        return output


if __name__ == "__main__":
    Toothfairy_algorithm().process()
