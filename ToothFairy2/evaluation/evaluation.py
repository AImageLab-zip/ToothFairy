import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import Dict
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from medpy.metric import binary

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
    "Lower Right Third Molar (Wisdom Tooth)": 48
}

from evalutils.io import SimpleITKLoader
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

def fix_filename_format(filename: str) -> str:
    if len(filename) < 4: raise Exception()

    if filename[-4:] != '.mha' and filename[-4] == '.':
        raise Exception()

    if filename[-4:] != '.mha':
        filename += '.mha'
    return filename

def load_predictions_json(fname: Path):
    with open(fname, "r") as f:
        entries = json.load(f)

    mapping = {}
    for e in entries:
        pk = e['pk']
        input_entry = e['inputs'][0]
        output_entry = e['outputs'][0]
        m_key = f"/input/{pk}/output/images/oral-pharyngeal-segmentation/{output_entry['image']['pk']}.mha"
        m_value = f"/opt/app/ground-truth/{input_entry['image']['name']}"
        mapping[m_key] = m_value

    return mapping


def mean(l):
    if len(l) == 0:
        return 0
    return sum(l)/len(l)


def compute_binary_dice(pred, label):
    addition = pred.sum() + label.sum()
    if addition == 0:
        return 1.0
    return 2. * np.logical_and(pred, label).sum() / addition


def compute_binary_hd95(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return np.linalg.norm(pred.shape)
    return binary.hd95(pred, gt)


def compute_multiclass_dice_and_hd95(pred, label):

    dice_per_class = {}
    hd_per_class = {}

    for label_name, label_id in LABELS.items():
        print(f'{label_name=}')
        binary_class_pred = pred == label_id
        binary_class_label = label == label_id
        dice = compute_binary_dice(binary_class_pred, binary_class_label)
        hd = compute_binary_hd95(binary_class_pred, binary_class_label)
        dice_per_class[label_name] = dice
        hd_per_class[label_name] = hd

    dice_per_class['average'] = mean(dice_per_class.values())
    hd_per_class['average'] = mean(hd_per_class.values())
    return dice_per_class, hd_per_class



class ToothfairyEvaluation():
    def __init__(self,):
        self.mapping = load_predictions_json(Path('/input/predictions.json'))
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

    def evaluate(self,):
        for k in self.mapping.keys():
            score = self.score_case(k)
            self.case_results = self.case_results.append(
                score, ignore_index=True
            )

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
        pred = self.loader.load_image(case)
        gt = self.loader.load_image(self.mapping[case])

        pred = sitk.GetArrayFromImage(pred).squeeze()
        gt = sitk.GetArrayFromImage(gt).squeeze()

        dice, hd95 = compute_multiclass_dice_and_hd95(gt, pred)

        metrics_dict = {
            'DiceCoefficient': dice['average'],
            'HausdorffDistance95': hd95['average'],
            'pred_fname': case,
            'gt_fname': self.mapping[case],
        }

        for label_name in LABELS.keys():
            metrics_dict[f'Dice {label_name}'] = dice[label_name]
            metrics_dict[f'HD95 {label_name}'] = hd95[label_name]
        return metrics_dict

    def aggregate_series(self, *, series: pd.Series) -> Dict:
        summary = series.describe()
        valid_keys = [a for a in self.aggregates if a in summary]

        series_summary = {}

        for k in valid_keys:
            value = summary[k]

            # % in keys could cause problems when looking up values later
            key = k.replace("%", "pc")

            try:
                json.dumps(value)
            except TypeError:
                value = int(value)

            series_summary[key] = value

        return series_summary

if __name__ == "__main__":
    ToothfairyEvaluation().evaluate()
