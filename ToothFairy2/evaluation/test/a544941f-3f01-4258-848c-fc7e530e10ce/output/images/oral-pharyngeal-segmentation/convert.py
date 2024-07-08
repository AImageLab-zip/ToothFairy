import numpy as np
import SimpleITK as sitk

# Function to create and save random segmentation map
def create_random_segmentation_map(input_array, output_filename):
    # Create a random segmentation map with values between 0 and 40
    segmentation_map = np.random.randint(0, 41, size=input_array.shape, dtype=np.uint8)

    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(segmentation_map)

    # Save the SimpleITK image as a .mha file
    sitk.WriteImage(sitk_image, output_filename)

# Example usage
input_array = np.random.rand(1, 11, 22, 33)  # Replace with your own input array
output_filename = ['b247e7a6-9217-4162-b22d-9e048a49b1a3.mha', 'c74f059a-ce44-4a60-a2fb-32bca924afc2.mha', '6a9d11b3-9334-40d4-a0b0-a52add69c3f4.mha', 'e690fc76-d513-480b-8682-122c5647de0d.mha']

for fn in output_filename:
    create_random_segmentation_map(input_array, fn)
    print(f"Segmentation map saved as {fn}")

