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


def your_multi_instance_segmentation_algorithm(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Simple example algorithm using a single linear layer with random weights
    
    Args:
        input_tensor: Preprocessed CBCT volume tensor [1, H, W, D]
        
    Returns:
        Tuple of (segmentation_mask, metadata)
    """
    
    # Get tensor shape
    batch_size, height, width, depth = input_tensor.shape
    
    # Flatten the volume for linear layer
    flattened = input_tensor.view(batch_size, -1)  # [1, H*W*D]
    
    # Simple linear layer with random weights (7 classes just for example)
    # I definetly suggest to compress the labels to be contiguous and use the correct number of classes
    num_classes = 7
    linear_layer = nn.Linear(flattened.shape[1], num_classes)
    
    # Generate random output logits
    with torch.no_grad():
        logits = linear_layer(flattened)
        logits_volume = logits.unsqueeze(-1).repeat(1, 1, height * width * depth)
        logits_volume = logits_volume.permute(0, 2, 1)
        predictions = torch.argmax(logits_volume, dim=-1)
        output_volume = predictions.view(height, width, depth)
    
    output_array = output_volume.cpu().numpy().astype(np.uint8)
    
    return output_array


class ToothFairy3_MultiInstanceSegmentation(SegmentationAlgorithm):
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
        
        output_array = your_multi_instance_segmentation_algorithm(input_tensor)
        
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        
        return output_image


if __name__ == "__main__":
    ToothFairy3_MultiInstanceSegmentation().process()
