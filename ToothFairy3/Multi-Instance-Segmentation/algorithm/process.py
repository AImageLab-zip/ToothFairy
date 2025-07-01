from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, Tuple

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def your_oral_pharyngeal_segmentation_algorithm(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Simple example algorithm using a single linear layer with random weights
    
    Args:
        input_tensor: Preprocessed CBCT volume tensor [1, H, W, D]
        
    Returns:
        Segmentation mask as numpy array
    """
    # Remove batch dimension for processing
    volume = input_tensor.squeeze(0)  # Remove batch dimension: [H, W, D]
    
    # Return zeros with same shape as original volume (without batch dimension)
    return np.zeros_like(volume.cpu().numpy(), dtype=np.uint8)


class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
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
        
        # Create output directory if it doesn't exist
        if not self._output_path.exists():
            self._output_path.mkdir(parents=True)
        
        # Create metadata output directory
        self.metadata_output_path = Path('/output/metadata/')
        if not self.metadata_output_path.exists():
            self.metadata_output_path.mkdir(parents=True)
        
        # Initialize device
        self.device = get_default_device()
        print(f"Using device: {self.device}")

    def save_instance_metadata(self, metadata: Dict, image_name: str):
        """
        Save instance metadata to JSON file
        
        Args:
            metadata: Instance metadata dictionary
            image_name: Name of the input image (without extension)
        """
        metadata_file = self.metadata_output_path / f"{image_name}_instances.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image) -> sitk.Image:
        input_array = sitk.GetArrayFromImage(input_image)
        
        # Basic preprocessing
        input_array = np.clip(input_array.astype(np.float32), 0, 2100) / 2100.0
        
        input_tensor = torch.from_numpy(input_array)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        output_array = your_oral_pharyngeal_segmentation_algorithm(input_tensor)
        
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        
        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()
