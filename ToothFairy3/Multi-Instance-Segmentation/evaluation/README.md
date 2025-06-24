# ToothFairy3 Multi-Instance-Segmentation Evaluation

This is the evaluation code for the ToothFairy3 Multi-Instance-Segmentation track. The challenge focuses on identifying and separately segmenting multiple instances of similar anatomical structures within CBCT volumes.

## Multi-Instance Segmentation Evaluation

The evaluation script handles:
- **Instance Detection**: Matching predicted instances to ground truth using Hungarian algorithm
- **Instance Segmentation**: Computing Dice and HD95 for matched instance pairs
- **Detection Metrics**: True positives, false positives, and false negatives per structure class

## Label Format

Multi-instance labels use the format: `Structure class ID * 1000 + Instance ID`

Examples:
- Tooth 16 instance 1: `16001`
- Tooth 16 instance 2: `16002` 
- Tooth 21 instance 1: `21001`

Non-instance structures use standard labels (1-7 for jawbones, canals, etc.)

## Metrics Computed

### Segmentation Metrics
- **Dice Coefficient**: Per-class and overall averages
- **Hausdorff Distance 95%**: Boundary accuracy measure

### Detection Metrics  
- **True Positives (TP)**: Correctly detected instances
- **False Positives (FP)**: Incorrectly detected instances
- **False Negatives (FN)**: Missing instances

### Aggregated Metrics
- **Average All**: Mean across all structure classes
- **Average Teeth**: Mean across tooth instances only

## Usage

The evaluation runs automatically on the Grand-Challenge platform. For local testing:

```bash
python evaluation.py
```

Input files expected:
- `/input/predictions.json`: Prediction mappings
- Prediction images in multi-instance format
- Ground truth images in `/opt/app/ground-truth/`

Output:
- `/output/metrics.json`: Complete evaluation results

## Instance Matching Algorithm

Instances are matched using the Hungarian algorithm based on centroid distances. This ensures optimal pairing between predicted and ground truth instances for fair evaluation.
