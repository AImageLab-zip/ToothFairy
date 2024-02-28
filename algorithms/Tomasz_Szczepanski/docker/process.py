from pathlib import Path
import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np
import os

from src.arg_parser import args

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from src.inference_LQ_coarse import inference_LQ
from src.inference_HQ import inference_HQ
from src.cuda_stats import setup_cuda


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("using GPU...")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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
        input_array = input_array.astype(np.float32)
        np.save("/opt/app/to_inference.npy", input_array)

        gcr = inference_LQ("/opt/app/to_inference.npy", args=args, device=get_default_device())
        gcr_numpy = gcr.squeeze()
        np.save("/opt/app/gcr.npy", gcr_numpy)


        output = inference_HQ("/opt/app/to_inference.npy", "/opt/app/gcr.npy", args=args, device=get_default_device())
        print(f"Output tensor of shape: {output.shape}")

        output = output.squeeze()
        output = np.where(output > 0.5, 1, 0).astype(np.uint8)

        output = sitk.GetImageFromArray(output)

        return output


if __name__ == "__main__":
    
    # CUDA 
    # TODO ONLY DEBUG!!! TURN OFF FOR SUBMISSION
    # setup_cuda(args.gpu_frac, num_threads=8, device=args.device, visible_devices=args.visible_devices,
    #            use_cuda_with_id=args.cuda_device_id)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    # if device.type == "cuda":
    #     device_name = torch.cuda.get_device_name(int(args.cuda_device_id))

    #reproducibility
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    
    # seeds
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    # speedup
    torch.backends.cudnn.benchmark = False
    Toothfairy_algorithm().process()
