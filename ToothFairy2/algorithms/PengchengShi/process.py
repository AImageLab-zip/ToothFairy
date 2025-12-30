from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import os

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    
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
    
        imagesTs_path = "/opt/app/imagesTs"
        os.makedirs(imagesTs_path, exist_ok=True)
        main_nii_path = os.path.join(imagesTs_path, "inference_case1_0000.nii.gz")
        sitk.WriteImage(input_image, main_nii_path, useCompression=True)
        
        nnUNet_results = "/opt/app/nnUNet/nnUNet_results"
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset112_ToothFairy2_CT_480/nnUNetTrainer_NexToU_Hierarchical_BTI_ToothFairy48_NoMirroring__nnUNetPlans__3d_fullres_patch32'),
            use_folds=("all",0,1),
            checkpoint_name='checkpoint_final.pth',
        )

        # predict a single numpy array
        img, props = SimpleITKIO().read_images(["/opt/app/imagesTs/inference_case1_0000.nii.gz"])
        output = predictor.predict_single_npy_array(img, props, None, None, False)
        output = output.astype(np.uint8)
        
        image = sitk.GetImageFromArray(output)
        image.SetDirection(props['sitk_stuff']['direction'])
        image.SetOrigin(props['sitk_stuff']['origin'])
        image.SetSpacing(props['sitk_stuff']['spacing'])
        return image

if __name__ == "__main__":
    Toothfairy2_algorithm().process()
