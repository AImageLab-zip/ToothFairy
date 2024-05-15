import SimpleITK as sitk
import numpy as np
import scipy
import json
from pathlib import Path
import os
from typing import Dict
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize

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
        m_key = f"/input/{pk}/output/images/inferior-alveolar-canal/{output_entry['image']['pk']}.mha"
        m_value = f"/opt/app/ground-truth/{input_entry['image']['name']}"
        mapping[m_key] = m_value

    return mapping

def compute_dice(gt, pred):
    overlap_measure = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measure.SetNumberOfThreads(1)
    overlap_measure.Execute(gt, pred)
    return overlap_measure.GetDiceCoefficient()

def compute_hd95(gt, pred):
    # gt.SetSpacing(np.array([1, 1, 1]).astype(np.float64))
    # pred.SetSpacing(np.array([1, 1, 1]).astype(np.float64))

    signed_distance_map = sitk.SignedMaurerDistanceMap(
        gt, squaredDistance=False, useImageSpacing=True
    )

    ref_distance_map = sitk.Abs(signed_distance_map)
    ref_surface = sitk.LabelContour(gt, fullyConnected=True)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())


    signed_distance_map_pred = sitk.SignedMaurerDistanceMap( pred, squaredDistance=False, useImageSpacing=True)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    seg_surface = sitk.LabelContour(pred > 0.5, fullyConnected=True)

    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)

    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(seg_surface > 0.5)

    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.percentile(all_surface_distances, 95)

def compute_cldice(gt, pred):
    """
    Code adapted from: https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py
    """
    if isinstance(gt, sitk.Image):
        gt = sitk.GetArrayFromImage(gt)
    if isinstance(pred, sitk.Image):
        pred = sitk.GetArrayFromImage(pred)
    s_gt = skeletonize(gt)
    s_pred = skeletonize(pred)
    t_prec = np.sum(gt*s_pred)/np.sum(s_pred) if np.sum(s_pred) > 0 else 1
    t_rec = np.sum(pred*s_gt)/np.sum(s_gt) if np.sum(s_gt) > 0 else 1
    cldice = 0 if t_prec+t_rec == 0 else 2*t_prec*t_rec/(t_prec+t_rec)
    return cldice

def betti_number(img: np.array):
    """
    calculates the Betti number B0, B1, and B2 for a 3D or 2D img
    from the Euler characteristic number
    """

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1
    N26 = 3

    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)

    # make sure the image is binary with
    assert set(np.unique(padded)).issubset({0, 1})

    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1

    # get the label connected regions for foreground
    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    # get the label connected regions for background
    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    # NOTE: need to substract 1 from b2
    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    # print(f"Betti number: b0 = {b0}, b1 = {b1}, b2 = {b2}")

    return [b0, b1]

def compute_betti_errors(gt, pred):
    """
    Betti error calculation adapted from the TopCoW evaluation script
    https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py
    """
    if isinstance(gt, sitk.Image):
        gt = sitk.GetArrayFromImage(gt)
    if isinstance(pred, sitk.Image):
        pred = sitk.GetArrayFromImage(pred)
    gt_betti_number = betti_number(gt)
    pred_betti_number = betti_number(pred)
    betti_0_error = abs(gt_betti_number[0] - pred_betti_number[0])
    betti_1_error = abs(gt_betti_number[1] - pred_betti_number[1])
    return betti_0_error, betti_1_error

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

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        dice = compute_dice(gt, pred)
        hd95 = compute_hd95(gt, pred)
        cldice = compute_cldice(gt, pred)
        betti_0, betti_1 = compute_betti_errors(gt, pred)
        return {
            'DiceCoefficient': dice,
            'HausdorffDistance95': hd95,
            'clDice': cldice,
            'Betti_0': betti_0,
            'Betti_1': betti_1,
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
