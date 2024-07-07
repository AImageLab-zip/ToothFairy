import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import Dict
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from medpy.metric import binary

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
        print(f'{m_key=}')
        m_value = f"/opt/app/ground-truth/{input_entry['image']['name']}"
        mapping[m_key] = m_value

    return mapping

def compute_dice(pred, label):
    print(f'{pred.shape=}')
    print(f'{label.shape=}')
    addition = pred.sum() + label.sum()
    if addition == 0:
        return 1.0
    return 2. * np.logical_and(pred, label).sum() / addition

def compute_hd95(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    return binary.hd95(pred, gt)

def compute_cldice(gt, pred):
    """
    Code adapted from: https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py
    """
    s_gt = skeletonize(gt)
    s_pred = skeletonize(pred)
    t_prec = np.sum(gt*s_pred)/np.sum(s_pred) if np.sum(s_pred) > 0 else 1
    t_rec = np.sum(pred*s_gt)/np.sum(s_gt) if np.sum(s_gt) > 0 else 1
    cldice = 0 if t_prec+t_rec == 0 else 2*t_prec*t_rec/(t_prec+t_rec)
    return cldice


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

        print(f'{case=}')
        print(f'{self.mapping[case]=}')

        pred = sitk.GetArrayFromImage(pred).squeeze()
        gt = sitk.GetArrayFromImage(gt).squeeze()



        dice = compute_dice(gt, pred)
        hd95 = compute_hd95(gt, pred)
        cldice = compute_cldice(gt, pred)

        return {
            'DiceCoefficient': dice,
            'HausdorffDistance95': hd95,
            'clDice': cldice,
            'pred_fname': case,
            'gt_fname': self.mapping[case],
        }

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
