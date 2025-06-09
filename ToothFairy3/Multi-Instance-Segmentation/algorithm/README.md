# ToothFairy3 Multi-Instance-Segmentation Algorithm Template

This directory contains the algorithm template for the Multi-Instance-Segmentation track of ToothFairy3 challenge.

## Files Overview

- `process.py`: Main algorithm implementation
- `Dockerfile`: Docker container configuration  
- `requirements.txt`: Python dependencies
- `build.ps1` / `build.sh`: Build scripts for Windows/Linux
- `test.ps1` / `test.sh`: Test scripts for local validation
- `export.ps1` / `export.sh`: Export scripts for submission

## Implementation Guide

### 1. Algorithm Implementation (`process.py`)

The main algorithm class `ToothFairy3_MultiInstanceSegmentation` inherits from `evalutils.SegmentationAlgorithm`. Key methods to implement:

- `predict()`: Main prediction method that processes CBCT images
- `your_multi_instance_segmentation_algorithm()`: Your core multi-instance segmentation logic
- `separate_instances()`: Connected component analysis for instance separation
- `assign_anatomical_labels()`: Assign FDI tooth numbers based on spatial position

### 2. Instance Labeling Convention

Instances are labeled using the formula: `tooth_class * 1000 + instance_id`

Examples:
- Upper Right First Molar (16), Instance 1: `16001`
- Lower Left Canine (33), Instance 1: `33001`
- Multiple instances of same tooth type get sequential IDs

### 3. Output Format

The algorithm produces two outputs:

#### Segmentation Image
- 3D label map with instance-specific labels
- Background pixels: `0`
- Instance pixels: `tooth_class * 1000 + instance_id`

#### Metadata JSON
```json
{
  "16001": {
    "tooth_class": 16,
    "tooth_name": "Upper Right First Molar",
    "instance_id": 1,
    "centroid": [x, y, z],
    "volume": 12345,
    "confidence": 0.95
  }
}
```

### 4. FDI Tooth Numbering

The algorithm uses the FDI (World Dental Federation) numbering system:
- **Quadrant 1 (Upper Right)**: 11-18
- **Quadrant 2 (Upper Left)**: 21-28  
- **Quadrant 3 (Lower Left)**: 31-38
- **Quadrant 4 (Lower Right)**: 41-48

### 5. Dependencies

Add your required packages to `requirements.txt`. Recommended packages:
- `connected-components-3d`: Fast 3D connected components
- `scipy`: Image processing and spatial algorithms
- `scikit-image`: Additional image processing tools
- `monai`: Medical image analysis toolkit

### 6. Model Architecture Considerations

For multi-instance segmentation, consider:
- **Instance Segmentation Networks**: Mask R-CNN, YOLACT, etc.
- **Panoptic Segmentation**: Combined semantic + instance segmentation
- **Point-based Methods**: PointNet++ for 3D point cloud processing
- **Graph-based Methods**: Model spatial relationships between teeth

## Building and Testing

### Build the Docker image:
```powershell
# Windows
.\build.ps1

# Linux/macOS
./build.sh
```

### Test locally:
```powershell  
# Windows
.\test.ps1

# Linux/macOS
./test.sh
```

### Export for submission:
```powershell
# Windows
.\export.ps1

# Linux/macOS  
./export.sh
```

## Implementation Tips

1. **Spatial Consistency**: Use anatomical knowledge to validate instance assignments
2. **Post-processing**: Apply morphological operations to clean up instance boundaries
3. **Multi-scale Analysis**: Process at different resolutions for speed and accuracy
4. **Anatomical Constraints**: Enforce tooth presence/absence patterns
5. **Uncertainty Estimation**: Provide confidence scores for each instance

## Common Challenges

1. **Adjacent Teeth Separation**: Distinguishing between touching or overlapping teeth
2. **Missing Teeth**: Handling extracted or congenitally missing teeth
3. **Dental Work**: Correctly identifying and labeling restorations vs natural teeth
4. **Anatomical Variation**: Handling individual differences in tooth shape and position
5. **Partial Volumes**: Managing teeth that are partially outside the imaging volume

## Evaluation Criteria

Your algorithm will be evaluated on:
- **Instance Detection**: Precision/Recall of correctly identified instances
- **Instance Segmentation**: Per-instance Dice coefficient and Hausdorff distance
- **Anatomical Consistency**: Correctness of FDI numbering assignments
- **Spatial Relationships**: Proper modeling of anatomical constraints

## Submission

Submit the exported Docker image through the Grand-Challenge platform. Ensure your algorithm:
- Produces valid instance labels and metadata
- Handles variable numbers of teeth correctly
- Runs within computational limits
- Maintains anatomical consistency

For more information, visit the [challenge website](https://toothfairy3.grand-challenge.org/).
