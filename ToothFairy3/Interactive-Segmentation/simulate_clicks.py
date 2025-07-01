import SimpleITK as sitk
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import json


def save_heatmap_volume(image, clicks, mask, save_path, sigma=3):
    """Save MHA/NIfTI files with Gaussian heatmaps for clicks and a binary mask, preserving metadata."""
    array = sitk.GetArrayFromImage(image)

    # Create heatmap for clicks
    heatmap = np.zeros_like(array, dtype=np.float32)
    for x, y, z in clicks:
        heatmap[x, y, z] = 1

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Convert mask to SimpleITK format
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(image)

    # Convert heatmap to SimpleITK format
    heatmap_image = sitk.GetImageFromArray(heatmap)
    heatmap_image.CopyInformation(image)

    # Save both images
    sitk.WriteImage(heatmap_image, save_path.replace(".mha", "_heatmap.nii.gz"))
    sitk.WriteImage(mask_image, save_path.replace(".mha", "_mask.nii.gz"))

def sample_clicks_from_mask(mask, num_clicks=5, noise_level=3, x_noise=5):
    """Sample click points from a binary mask with less uniform X sampling and centered Y, Z."""
    indices = np.argwhere(mask)
    import cc3d

    num_components = cc3d.connected_components(mask, connectivity=26).max()

    if indices.size == 0:
        return None  # No valid clicks found


    x_coords = indices[:, 0]  # Axial slices
    y_coords = indices[:, 1]  # Y-coordinates
    z_coords = indices[:, 2]  # Z-coordinates

    # Find min/max X, but start slightly later (0-5 slices offset)
    x_min, x_max = x_coords.min(), x_coords.max()
    x_start = np.random.randint(x_min, x_min + 6)
    x_end = np.random.randint(x_max - 5, x_max + 1)

    # Define valid X-values where the mask has valid points
    valid_x = [el for el in np.arange(x_min, x_max + 1) if np.any(mask[el])]


    # Sample X-values with approximate uniformity while ensuring validity
    sampled_x = np.linspace(x_start, x_end, num=num_clicks, dtype=int)
    sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

    # Add noise and ensure X values remain within valid_x
    sampled_x += np.random.randint(-x_noise, x_noise + 1, size=num_clicks)
    sampled_x = np.clip(sampled_x, x_start, x_end)
    sampled_x = np.array([valid_x[np.abs(np.array(valid_x) - x).argmin()] for x in sampled_x])

    # Ensure first and last clicks are exactly at the adjusted start and end points
    sampled_x[0], sampled_x[-1] = x_start, x_end
    if x_start not in valid_x:
        sampled_x[0] = x_min
    if x_end not in valid_x:
        sampled_x[-1] = x_max

    for x in sampled_x:
        assert x in valid_x # check if sampled slices contain a label


    click_points = []
    for x in sampled_x:
        valid_points = indices[x_coords == x]  # Get all (Y, Z) for this X slice
        assert valid_points.size > 0
        # Compute center of Y and Z for this slice
        y_center = np.median(valid_points[:, 1]).astype(int)
        z_center = np.median(valid_points[:, 2]).astype(int)

        # Add small noise while keeping in bounds
        y = np.clip(y_center + np.random.randint(-noise_level, noise_level + 1), y_coords.min(), y_coords.max())
        z = np.clip(z_center + np.random.randint(-noise_level, noise_level + 1), z_coords.min(), z_coords.max())

        # Add small noise while keeping in bounds
        for _ in range(10):  # Try 10 times to find a valid (y, z)
            y = np.clip(y_center + np.random.randint(-noise_level, noise_level + 1), y_coords.min(), y_coords.max())
            z = np.clip(z_center + np.random.randint(-noise_level, noise_level + 1), z_coords.min(), z_coords.max())
            if mask[x, y, z]:
                break
        else:
            # If no valid point is sampled with perturbation, sample one randomly
            y, z = valid_points[np.random.randint(valid_points.shape[0]), 1:]
            print('[WARNING] Using a uniform point since center perturbation always led to an invalid point...')
        click_points.append([int(el) for el in [x, y, z]])

        assert mask[x, y, z] # make sure simulated click is in IAC
    return click_points


def read_mha(file_path):
    """Read an MHA file and extract bounding boxes for left and right IAC."""
    image = sitk.ReadImage(file_path)
    np_image = sitk.GetArrayFromImage(image)  # Shape: (depth, height, width)

    left_IAC = (np_image == 3)
    right_IAC = (np_image == 4)

    clicks_left_IAC = sample_clicks_from_mask(left_IAC)
    clicks_right_IAC = sample_clicks_from_mask(right_IAC)


    return clicks_left_IAC, clicks_right_IAC, image, left_IAC, right_IAC

def convert_json_format_to_gc(json_dict):
    json_dict_gc = {
        "version": {"major": 1, "minor": 0},
        "type": "Multiple points",
        "points": []
        }
    for left_IAC_point in json_dict['Left_IAC']:
        json_dict_gc["points"].append({"point": left_IAC_point, "name": "Left_IAC"})
    for right_IAC_point in json_dict['Right_IAC']:
        json_dict_gc["points"].append({"point": right_IAC_point, "name": "Right_IAC"})
    return json_dict_gc

def create_archive_items_0_to_5_clicks(json_dict, gc_archive_output, archive_name):
    archive_path = os.path.join(gc_archive_output, archive_name)
    os.makedirs(archive_path, exist_ok=True)

    for i in range(6):
        new_json_dict_gc = {
            "version": {"major": 1, "minor": 0},
            "type": "Multiple points",
            "points": []
        }

        for j in range(i):
            new_json_dict_gc['points'].append(json_dict['points'][j])
        for j in range(i):
            new_json_dict_gc['points'].append(json_dict['points'][j + 5])

        filename = f"{i}_clicks.json"
        filepath = os.path.join(archive_path, filename)

        with open(filepath, "w") as f:
            json.dump(new_json_dict_gc, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_label", required=True, help="Path to the mha label")
    parser.add_argument("--debug_output", required=False, help="Output path for click visualization for debugging")
    parser.add_argument("--gc_archive_output", required=False, help="Output path for Grand Challenge archives (0-5) clicks that will be uploaded as cases to GC in this format")
    parser.add_argument("--json_output", required=True, help="Output path for JSON containing all clicks")

    args = parser.parse_args()

    suffix = '.nii.gz' if '.nii.gz' in args.input_label else '.mha'

    os.makedirs(args.json_output, exist_ok=True)

    print(f'Computing Clicks for {args.input_label}...')
    clicks_left_IAC, clicks_right_IAC, image, left_IAC, right_IAC = read_mha(args.input_label)
    json_dict = {'Left_IAC': clicks_left_IAC, 'Right_IAC': clicks_right_IAC}
    json_dict = convert_json_format_to_gc(json_dict)



    if args.debug_output is not None:
        os.makedirs(args.debug_output, exist_ok=True)
        save_heatmap_volume(image, clicks_left_IAC, left_IAC, os.path.join(args.debug_output, f"{os.path.basename(args.input_label).split(suffix)[0]}_left_IAC.mha"))
        save_heatmap_volume(image, clicks_right_IAC, right_IAC, os.path.join(args.debug_output, f"{os.path.basename(args.input_label).split(suffix)[0]}_right_IAC.mha"))
    if not os.path.exists(args.json_output):
        os.mkdir(args.json_output)
    with open(os.path.join(args.json_output, f"{os.path.basename(args.input_label).split(suffix)[0]}_clicks.json"),'w') as f:
        json.dump(json_dict, f, indent=2)
    if args.gc_archive_output is not None:
        create_archive_items_0_to_5_clicks(json_dict, args.gc_archive_output, os.path.basename(args.input_label).split(suffix)[0])




if __name__ == "__main__":
    main()
