import json
from pathlib import Path
from typing import Dict
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# Simple replacement for medpy.metric.binary.hd95
def hd95(pred, gt):
    """Simple HD95 implementation."""
    # Get surface points using edge detection
    pred_edges = pred ^ binary_erosion(pred)
    gt_edges = gt ^ binary_erosion(gt)
    
    pred_surface = np.argwhere(pred_edges)
    gt_surface = np.argwhere(gt_edges)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float(np.linalg.norm(pred.shape))
    
    # Compute distances
    distances_pred_to_gt = cdist(pred_surface, gt_surface, metric='euclidean')
    min_distances_pred = np.min(distances_pred_to_gt, axis=1)
    
    distances_gt_to_pred = cdist(gt_surface, pred_surface, metric='euclidean')
    min_distances_gt = np.min(distances_gt_to_pred, axis=1)
    
    all_distances = np.concatenate([min_distances_pred, min_distances_gt])
    return float(np.percentile(all_distances, 95))

class binary:
    @staticmethod
    def hd95(pred, gt):
        return hd95(pred, gt)

LABELS = {
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
    "Upper Right Central Incisor": 11,
    "Upper Right Lateral Incisor": 12,
    "Upper Right Canine": 13,
    "Upper Right First Premolar": 14,
    "Upper Right Second Premolar": 15,
    "Upper Right First Molar": 16,
    "Upper Right Second Molar": 17,
    "Upper Right Third Molar (Wisdom Tooth)": 18,
    "Upper Left Central Incisor": 21,
    "Upper Left Lateral Incisor": 22,
    "Upper Left Canine": 23,
    "Upper Left First Premolar": 24,
    "Upper Left Second Premolar": 25,
    "Upper Left First Molar": 26,
    "Upper Left Second Molar": 27,
    "Upper Left Third Molar (Wisdom Tooth)": 28,
    "Lower Left Central Incisor": 31,
    "Lower Left Lateral Incisor": 32,
    "Lower Left Canine": 33,
    "Lower Left First Premolar": 34,
    "Lower Left Second Premolar": 35,
    "Lower Left First Molar": 36,
    "Lower Left Second Molar": 37,
    "Lower Left Third Molar (Wisdom Tooth)": 38,
    "Lower Right Central Incisor": 41,
    "Lower Right Lateral Incisor": 42,
    "Lower Right Canine": 43,
    "Lower Right First Premolar": 44,
    "Lower Right Second Premolar": 45,
    "Lower Right First Molar": 46,
    "Lower Right Second Molar": 47,
    "Lower Right Third Molar (Wisdom Tooth)": 48,
    "Left Mandibular Incisive Canal": 103,
    "Right Mandibular Incisive Canal": 104,
    "Lingual Canal": 105,
    "Pulp": 150  # Combined pulp class
}

# Pulp labels that should be converted to class 150
PULP_LABELS = [111, 112, 113, 114, 115, 116, 117, 118,
               121, 122, 123, 124, 125, 126, 127, 128,
               131, 132, 133, 134, 135, 136, 137, 138,
               141, 142, 143, 144, 145, 146, 147, 148]

class SimpleITKLoader:
    """Simple replacement for evalutils.io.SimpleITKLoader"""
    def load_image(self, path):
        return sitk.ReadImage(str(path))

def fix_filename_format(filename: str) -> str:
    if len(filename) < 4: 
        raise Exception()
    if filename[-4:] != '.mha' and filename[-4] == '.':
        raise Exception()
    if filename[-4:] != '.mha':
        filename += '.mha'
    return filename

def load_predictions_json(fname: Path):
    """Load predictions JSON for oral-pharyngeal segmentation evaluation."""
    with open(fname, "r") as f:
        entries = json.load(f)

    # Check if this is the new evalutils format (results.json)
    if (len(entries) > 0 and 
        'outputs' in entries[0] and 
        'inputs' in entries[0] and 
        'pk' not in entries[0]):
        print(f"Converting {fname} format to predictions format...")
        return convert_results_to_predictions(entries)
    
    # Original predictions.json format
    mapping = {}
    for e in entries:
        pk = e['pk']
        input_entry = e['inputs'][0]
        output_entry = e['outputs'][0]        # Update path for oral-pharyngeal segmentation output
        m_key = f"/input/{pk}/output/images/oral-pharyngeal-segmentation/{output_entry['image']['pk']}.mha"
        m_value = f"/opt/ml/input/data/ground_truth/{input_entry['image']['name']}"
        mapping[m_key] = m_value

    return mapping

def compute_ground_truth_filename(input_filename):
    """
    Compute the ground truth filename based on the input filename.
    
    Examples:
    - ToothFairy3P_381_0000.nii -> ToothFairy3P_381.mha (grand-challenge)
    - ToothFairy3P_077_0000.mha -> ToothFairy3P_077.mha (grand-challenge)
    - ToothFairy3S_0042_0000.nii -> ToothFairy3S_0042.mha (grand-challenge)
    - ToothFairy3P_077.mha -> ToothFairy3P_077.mha (test with same files as GT)
    - 5f29221f-88b9-43a8-af27-c8ea49eba32f.mha -> 5f29221f-88b9-43a8-af27-c8ea49eba32f.mha
    """
    from pathlib import Path
    
    input_path = Path(input_filename)
    name_without_ext = input_path.stem
    
    # Remove .nii extension if present (for .nii.gz files)
    if name_without_ext.endswith('.nii'):
        name_without_ext = name_without_ext[:-4]
    
    # Check for grand-challenge pattern: ends with _0000 
    if name_without_ext.endswith('_0000'):
        # Grand-challenge files: ToothFairy3P_381_0000 -> ToothFairy3P_381
        gt_base_name = name_without_ext[:-5]  # Remove '_0000'
        gt_filename = f"{gt_base_name}.mha"
    else:
        # For files that don't have '_0000' suffix, keep the same filename
        # but ensure it has .mha extension (ground truth is always .mha)
        if input_filename.endswith('.mha'):
            gt_filename = input_filename
        else:
            gt_filename = f"{name_without_ext}.mha"
    
    return gt_filename

def convert_results_to_predictions(results_entries):
    """Convert evalutils results.json format to predictions format for evaluation."""
    import uuid
    
    mapping = {}
    for i, entry in enumerate(results_entries):
        if not entry.get('outputs') or not entry.get('inputs'):
            continue
            
        # Generate a unique pk for this entry
        pk = str(uuid.uuid4())
        
        # Extract filenames
        output_filename = entry['outputs'][0]['filename']
        input_filename = entry['inputs'][0]['filename'] 

        # Compute ground truth filename based on input filename
        gt_filename = compute_ground_truth_filename(input_filename)

        m_key = f"/input/images/oral-pharyngeal-segmentation/{output_filename}"
        m_value = f"/opt/ml/input/data/ground_truth/{gt_filename}"
        mapping[m_key] = m_value

    return mapping

def convert_pulp_labels(label_map):
    """Convert all pulp labels to a single pulp class (150)."""
    converted = label_map.copy()
    for pulp_label in PULP_LABELS:
        converted[label_map == pulp_label] = 150
    return converted

def mean(l):
    if len(l) == 0:
        return 0
    return sum(l)/len(l)

def compute_binary_dice(pred, gt):
    """Compute Dice coefficient for binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    
    if union == 0:
        return 1.0  # Both masks are empty
    
    return 2.0 * intersection / union

def compute_binary_hd95(pred, gt):
    """Compute HD95 for binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if not pred.any() and not gt.any():
        return 0.0  # Both masks are empty
    if not pred.any() or not gt.any():
        return float(np.linalg.norm(pred.shape))  # One mask is empty
    
    return binary.hd95(pred, gt)

def compute_multiclass_dice_and_hd95(pred, gt):
    """Compute multiclass Dice and HD95 metrics."""
    # Convert pulp labels
    pred = convert_pulp_labels(pred)
    gt = convert_pulp_labels(gt)
    
    dice_per_class = {}
    hd_per_class = {}
    
    for label_name, label_id in LABELS.items():
        binary_class_pred = pred == label_id
        binary_class_label = gt == label_id
        dice = compute_binary_dice(binary_class_pred, binary_class_label)
        hd = compute_binary_hd95(binary_class_pred, binary_class_label)
        dice_per_class[label_name] = dice
        hd_per_class[label_name] = hd

    dice_per_class['average'] = mean(dice_per_class.values())
    hd_per_class['average'] = mean(hd_per_class.values())
    return dice_per_class, hd_per_class

class ToothfairyOralPharyngealEvaluation():
    def __init__(self,):
        # Try to load predictions.json first, fallback to results.json
        predictions_file = Path('/input/predictions.json')
        results_file = Path('/input/results.json')
        
        if predictions_file.exists():
            self.mapping = load_predictions_json(predictions_file)
        elif results_file.exists():
            self.mapping = load_predictions_json(results_file)
        else:
            raise FileNotFoundError("Neither predictions.json nor results.json found in /input/")
        
        self.loader = SimpleITKLoader()
        self.case_results = pd.DataFrame()
        self.aggregates = {
            "mean",
            "std", 
            "min",
            "max",
            "25%",
            "50%",
            "75%",
            "count",
            "uniq",
            "freq",
        }
    
    def find_image_file(self, file_path):
        """Find image file, checking for .mha first, then .nii.gz fallback."""
        path_obj = Path(file_path)
        
        # If the file exists as specified, return it
        if path_obj.exists():
            return str(path_obj)
        
        # Try .mha extension first
        mha_path = path_obj.with_suffix('.mha')
        if mha_path.exists():
            print(f"Found .mha file: {mha_path}")
            return str(mha_path)
        
        # Fallback to .nii.gz
        nii_gz_path = path_obj.with_suffix('.nii.gz')
        if nii_gz_path.exists():
            print(f"Found .nii.gz file: {nii_gz_path}")
            return str(nii_gz_path)
        
        # If neither exists, try replacing .mha with .nii.gz in the original path
        if str(path_obj).endswith('.mha'):
            nii_gz_fallback = str(path_obj).replace('.mha', '.nii.gz')
            if Path(nii_gz_fallback).exists():
                print(f"Found .nii.gz fallback: {nii_gz_fallback}")
                return nii_gz_fallback
        
        # If nothing found, return original path (will fail later with clear error)
        print(f"No file found for: {path_obj}")
        return str(path_obj)

    def find_ground_truth_file(self, filename):
        """
        Find ground truth file in the extracted tarball location.
        The tarball is extracted to /opt/ml/input/data/ground_truth/ at runtime.
        """
        # Primary location where grand-challenge extracts the tarball
        gt_dir = Path('/opt/ml/input/data/ground_truth')
        
        # Fallback locations for testing/development
        fallback_dirs = [
            Path('/opt/app/ground-truth'),  # Original location
            Path('./ground-truth'),         # Local testing
            Path('./test/ground-truth'),    # Test directory
            Path('../test/ground-truth'),   # Test directory up one level
        ]
        
        all_dirs = [gt_dir] + fallback_dirs
        
        for dir_path in all_dirs:
            if not dir_path.exists():
                continue
                
            # Try exact filename first
            exact_path = dir_path / filename
            if exact_path.exists():
                print(f"Found ground truth file: {exact_path}")
                return str(exact_path)
            
            # Try without extension and add .mha
            base_name = Path(filename).stem
            if base_name.endswith('.nii'):
                base_name = base_name[:-4]  # Remove .nii part
            
            mha_path = dir_path / f"{base_name}.mha"
            if mha_path.exists():
                print(f"Found ground truth .mha file: {mha_path}")
                return str(mha_path)
            
            # Try with .nii.gz extension
            nii_path = dir_path / f"{base_name}.nii.gz"
            if nii_path.exists():
                print(f"Found ground truth .nii.gz file: {nii_path}")
                return str(nii_path)
          # If not found, return the expected path for better error messages
        expected_path = gt_dir / filename
        print(f"Ground truth file not found in any location. Expected: {expected_path}")
        return str(expected_path)

    def evaluate(self,):
        for k in self.mapping.keys():
            score = self.score_case(k)
            self.case_results = pd.concat([self.case_results, pd.DataFrame([score])], ignore_index=True)

        aggregate_results = {}
        for col in self.case_results.columns:
            aggregate_results[col] = self.aggregate_series(
                series=self.case_results[col]
            )

        with open('/output/metrics.json', "w") as f:
            f.write(json.dumps({
                "case": self.case_results.to_dict(),
                "aggregates": aggregate_results,
            }))

    def score_case(self, case):
        pred_path = self.find_image_file(case)
        
        # Extract just the filename from the mapped ground truth path
        gt_filename = Path(self.mapping[case]).name
        gt_path = self.find_ground_truth_file(gt_filename)
        
        print(f"Evaluating: {pred_path} vs {gt_path}")
        
        pred = self.loader.load_image(pred_path)
        gt = self.loader.load_image(gt_path)

        pred = sitk.GetArrayFromImage(pred).squeeze()
        gt = sitk.GetArrayFromImage(gt).squeeze()

        dice, hd95 = compute_multiclass_dice_and_hd95(pred, gt)

        metrics_dict = {
            'Dice Average': dice['average'],
            'HD95 Average': hd95['average'],
            'pred_fname': pred_path,
            'gt_fname': gt_path,
        }

        return metrics_dict

    def aggregate_series(self, *, series: pd.Series) -> Dict:
        summary = series.describe()
        valid_keys = [a for a in self.aggregates if a in summary]
        
        summary = summary[valid_keys]
        return {
            key: float(value) if isinstance(value, (int, float, np.number)) else str(value)
            for key, value in summary.items()
        }

if __name__ == "__main__":
    ToothfairyOralPharyngealEvaluation().evaluate()
