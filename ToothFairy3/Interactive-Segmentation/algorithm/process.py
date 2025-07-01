from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import json
import glob
import os
from typing import Dict, Tuple

from evalutils import SegmentationAlgorithm


def get_default_device():
    """Set device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def your_oral_pharyngeal_segmentation_algorithm(input_tensor: torch.Tensor, clicks_data: Dict) -> np.ndarray:
    """
    Simple example algorithm using a single linear layer with random weights

    Args:
        input_tensor: Preprocessed CBCT volume tensor [1, H, W, D]

    Returns:
        Segmentation mask as numpy array
    """
    # Remove batch dimension for processing
    volume = input_tensor.squeeze(0)  # Remove batch dimension: [H, W, D]
    print(f"Received {int(len(clicks_data.get('points', [])) / 2)} clicks for each IAC.")

    # Return zeros with same shape as original volume (without batch dimension)
    return np.zeros_like(volume.cpu().numpy(), dtype=np.uint8)


class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            input_path=Path('/input/images/cbct/'),
            output_path=Path('/output/images/iac-segmentation/'),
            validators={},
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

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_ = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        sitk.WriteImage(segmented_, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }


    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image, input_image_file_path: str = None) -> sitk.Image:
        input_array = sitk.GetArrayFromImage(input_image)
        # === Load and parse the JSON clicks file ===

        filename = Path(input_image_file_path).name

        if filename.endswith(".nii.gz"):
            base = filename[:-7]  # remove '.nii.gz' (7 chars)
        elif filename.endswith(".mha"):
            base = filename[:-4]  # remove '.mha' (4 chars)
        else:
            raise ValueError("Unsupported file extension")

        parts = base.split('_')
        input_json_clicks = f"/input/iac_clicks_{parts[0]}_{parts[-1]}.json"
        if not os.path.isfile(input_json_clicks):
            input_json_clicks = f"/input/iac_clicks_{base}.json"
        if not os.path.isfile(input_json_clicks):
            # Look for exactly one JSON file in /input/ that has the keyword "clicks"
            json_files = [f for f in glob.glob("/input/*.json") if "clicks" in f]
            print(json_files)
            if len(json_files) == 1:
                input_json_clicks = json_files[0]
                print(f"Using single JSON file found: {input_json_clicks}")
            else:
                raise RuntimeError(f"Could not find clicks JSON file at '{input_json_clicks}', "
                                   f"and found {len(json_files)} JSON files in /input/: {json_files}")

        try:
            with open(input_json_clicks, 'r') as f:
                clicks_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON clicks file '{input_json_clicks}': {e}")



        # Basic preprocessing
        input_array = np.clip(input_array.astype(np.float32), 0, 2100) / 2100.0

        input_tensor = torch.from_numpy(input_array)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        output_array = your_oral_pharyngeal_segmentation_algorithm(input_tensor, clicks_data)

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)

        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()
