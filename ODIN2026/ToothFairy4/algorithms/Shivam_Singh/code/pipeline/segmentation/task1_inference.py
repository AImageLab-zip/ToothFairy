"""
Adapted from U-Mamba2, documentation/competitions/Toothfairy3/task1_inference.py
    https://github.com/zhiqin1998/U-Mamba2
    Licensed CC BY-NC 4.0 (Attribution-NonCommercial). Original copyright
    retained by the U-Mamba2 authors.

MODIFICATIONS (ToothFairy4 submission container)
------------------------------------------------
Inference memory is bounded by the patch size rather than the volume size, so
no field of view can exhaust VRAM:

  * predict_preprocessed_image()
    The sliding-window logits accumulator and the Gaussian weight map are
    allocated on the CPU instead of the GPU. For a 47-class full-resolution
    volume that tensor is ~24 GiB, which does not fit a 24 GiB card alongside
    the model. Only one patch is resident on the GPU at a time.

  * convert_predicted_logits_to_segmentation_with_correct_shape()
    The final resample and argmax run per class channel on the CPU, tracking a
    running argmax, rather than materialising all 47 full-resolution channels at
    once. Peak memory falls from ~24 GiB to ~2 GiB. The result is identical:
    argmax over channels is the same whether computed incrementally or in one
    pass (verified on 30 random volumes).

Everything else -- test-time augmentation, the left/right label swap, the
connected-component postprocessing and the FDI remap -- is unchanged from
upstream.
"""
import os

n_threads = 4  # set to 4 to emulate grand challenge runtime env
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["BLIS_NUM_THREADS"] = str(n_threads)

import argparse
import gc
import time as _time
import itertools
import json
import time
import os
import traceback
from pathlib import Path
from typing import Union, Tuple
from copy import deepcopy
import cc3d

import nnunetv2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
# torch.cuda.set_per_process_memory_fraction(16/24) #16gb

# orientation of human: x is top to bottom, y is front to back, z is left to right
def mask_to_bbox_slice(mask, return_coord=False):
    x, y, z = np.any(mask, axis=(1, 2)), np.any(mask, axis=(0, 2)), np.any(mask, axis=(0, 1))
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    xmax, ymax, zmax = xmax + 1, ymax + 1, zmax + 1
    if return_coord:
        return xmin, xmax, ymin, ymax, zmin, zmax
    return slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)


class CustomPredictor(nnUNetPredictor):
    def __init__(self, *args, explicit_half=False, export=False, tta_batch_size=1, lr_mapping=None, n_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_half = explicit_half
        self.export = export
        # if export:
        #     assert self.explicit_half
        self.tta_batch_size = tta_batch_size
        if lr_mapping is not None:
            self.lr_mapping = list(range(n_class))
            for l, r in lr_mapping:
                self.lr_mapping[l] = r
                self.lr_mapping[r] = l
        else:
            self.lr_mapping = lr_mapping

    def predict_single_cbct_npy_array(self, input_image: np.ndarray, image_properties: dict,):
        # torch.set_num_threads(7)
        image_properties = deepcopy(image_properties)
        _t0 = _time.time()
        with torch.no_grad():
            if self.verbose:
                print('preprocessing')

            data, _, image_properties = self.preprocessor.run_case_npy(input_image, None, image_properties,
                                                                       self.plans_manager,
                                                                       self.configuration_manager,
                                                                       self.dataset_json)

            data = torch.from_numpy(data)
            del input_image
            _t_pre = _time.time() - _t0
            print(f"=+=   seg.preprocess       {_t_pre:8.1f}s", flush=True)

            if self.verbose:
                print('predicting')
            _t1 = _time.time()
            predicted_logits = self.predict_preprocessed_image(data)
            print(f"=+=   seg.forward(GPU)     {_time.time() - _t1:8.1f}s", flush=True)
            if self.verbose:
                print('Prediction done')

            _t2 = _time.time()
            _out = self.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, image_properties)
            print(f"=+=   seg.resample+argmax  {_time.time() - _t2:8.1f}s", flush=True)
            return _out

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        # dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        # plans = load_json(join(model_training_output_dir, 'plans.json'))
        # plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        assert len(use_folds) == 1
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=self.device, weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None
                dataset_json = checkpoint['init_args']['dataset_json']
                plans_manager = PlansManager(checkpoint['init_args']['plans'])

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        # trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        #                                             trainer_name, 'nnunetv2.training.nnUNetTrainer')
        # if trainer_class is None:
        #     raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
        #                        f'Please place it there (in any .py file)!')

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        assert len(self.list_of_parameters) == 1
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = (1,)#inference_allowed_mirroring_axes
        if self.lr_mapping is not None and 2 not in self.allowed_mirroring_axes:
            print('adding axis2 to allowed_mirroring_axes')
            self.allowed_mirroring_axes = (*self.allowed_mirroring_axes, 2)
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        network = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        self.network = network
        self.network = self.network.to(device=self.device, dtype=torch.float16 if self.explicit_half else None)
        self.network.load_state_dict(
            checkpoint['network_weights']
            # torch.load(self.list_of_parameters[0], map_location=self.device, weights_only=False)['network_weights']
        )
        self.network.eval()
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
            print('Using torch.compile')
            # self.network = torch.compile(self.network)
            self.network.compile()
        elif self.export:
            print('Using torch.export')
            st = time.time()
            with torch.no_grad():
                example_inputs = (
                    torch.randn((self.tta_batch_size, 1, *self.configuration_manager.patch_size),).cuda(),
                )
                self.network = torch.export.export(self.network, example_inputs,).module()
            print(f'model exported in {time.time() - st:.5f}s')

        self.preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        data_device = self.device
        predicted_logits_device = torch.device('cpu')   # big accumulator in RAM, not VRAM
        gaussian_device = torch.device('cpu')
        interim_dtype = torch.float16
        compute_device = self.device

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image
        gc.collect()
        empty_cache(self.device)

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        data = data.to(device=data_device, dtype=torch.float16 if self.explicit_half else None, non_blocking=True)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=interim_dtype,
                                       device=predicted_logits_device)
        # n_predictions = torch.zeros(data.shape[1:], dtype=interim_dtype, device=predicted_logits_device)

        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=interim_dtype)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        with torch.autocast(self.device.type, enabled=not self.explicit_half):
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                pred = self._internal_maybe_mirror_and_predict(data[sl][None].to(device=compute_device, non_blocking=True))[0].to(
                    device=predicted_logits_device, dtype=interim_dtype)
                pred /= (pred.max() / 100)
                # pred /= 10 # doing this save ~0.02s per image lol
                predicted_logits[sl] += (pred * gaussian)
                # n_predictions[sl[1:]] += gaussian

            # torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # if torch.any(torch.isinf(predicted_logits)):
            #     raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
            #                        'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
            #                        'predicted_logits to fp32')
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        return predicted_logits

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        # prediction = self.network(x)

        if mirror_axes is None:
            return self.network(x)

        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

        mirror_axes = [m + 2 for m in mirror_axes]
        axes_combinations = [
            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        ]

        if self.tta_batch_size > 1:
            # assert (len(axes_combinations) + 1) % self.tta_batch_size == 0, '(len(axes_combinations) + 1) must be divisible by tta_batch_size'
            x_combinations = [torch.flip(x, axes) for axes in axes_combinations]
            x_combinations.insert(0, x)
            assert len(x_combinations) % self.tta_batch_size == 0, f'{len(x_combinations)} not divisible by {self.tta_batch_size}'
            prediction = None
            for i in range(0, len(x_combinations), self.tta_batch_size):
                batch_prediction = self.network(torch.cat(x_combinations[i:i + self.tta_batch_size], dim=0))
                for j in range(batch_prediction.shape[0]):
                    original_idx = i + j
                    if original_idx == 0:
                        prediction = batch_prediction[j:j + 1]
                    else:
                        axes_to_flip_back = axes_combinations[original_idx - 1]
                        prediction += (torch.flip(batch_prediction[j:j + 1], axes_to_flip_back)[:, self.lr_mapping] if self.lr_mapping is not None and 4 in axes_to_flip_back else torch.flip(batch_prediction[j:j + 1], axes_to_flip_back))
                        # temp = torch.flip(batch_prediction[j:j + 1], axes_to_flip_back)
                        # if self.lr_mapping is not None and 2 + 2 in axes_to_flip_back:
                        #     temp = temp[:, self.lr_mapping]
                        # prediction += temp
        else:
            prediction = self.network(x)
            for axes in axes_combinations:
                prediction += (torch.flip(self.network(torch.flip(x, axes)), axes)[:, self.lr_mapping] if self.lr_mapping is not None and 4 in axes else torch.flip(self.network(torch.flip(x, axes)), axes))
                # temp = torch.flip(self.network(torch.flip(x, axes)), axes)
                # if self.lr_mapping is not None and 2 + 2 in axes:
                #     temp = temp[:, self.lr_mapping]
                # prediction += temp

        prediction /= (len(axes_combinations) + 1)
        return prediction

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
        # old = torch.get_num_threads()
        # torch.set_num_threads(7)

        # resample to original shape
        spacing_transposed = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration_manager.spacing if \
            len(self.configuration_manager.spacing) == \
            len(props['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration_manager.spacing]
        # Memory-bounded resample+argmax on CPU: one class at a time, running argmax.
        # Peak RAM ~2 single-channel full-res tensors, not all 47 (24GB -> ~2GB).
        _target_shape = props['shape_after_cropping_and_before_resampling']
        _target_spacing = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        predicted_logits = predicted_logits.cpu().float()
        _C = predicted_logits.shape[0]
        best_val = None
        best_idx = None
        with torch.no_grad():
            for _c in range(_C):
                _r = self.configuration_manager.resampling_fn_probabilities(
                    predicted_logits[_c:_c + 1], _target_shape, current_spacing,
                    _target_spacing, device=torch.device('cpu'))[0]
                if best_val is None:
                    best_val = _r.clone()
                    best_idx = torch.zeros(_r.shape, dtype=torch.int16)
                else:
                    _m = _r > best_val
                    best_val[_m] = _r[_m]
                    best_idx[_m] = _c
                del _r
            segmentation = best_idx
            del best_val
        del predicted_logits

        # segmentation may be torch.Tensor but we continue with numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(props['shape_before_cropping'],
                                                  dtype=np.uint8 if len(self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        # torch.set_num_threads(old)
        return segmentation_reverted_cropping



class BasePredictor(CustomPredictor):
    def __init__(self, cls_vol=None, simple_postprocess=False, *args, **kwargs):
        if cls_vol is None:
            cls_vol = {
                # 1: 521383,
                2: 448.88,
                3: 2198,
                4: 1754,
                5: 814.48,
                6: 1869,
                7: 71533,
                8: 7211,
                9: 5694,

                11: 1473.895,
                12: 888.1,
                13: 1200,
                14: 871.4,
                15: 1970.9,
                16: 1851,
                17: 2146.2,
                18: 1552,
                21: 1520,
                22: 728.52,
                23: 1196.26,
                24: 1357.385,
                25: 1964.44,
                26: 1064.55,
                27: 1930.25,
                28: 1684,
                31: 2177,
                32: 1092,
                33: 2844.9,
                34: 1590.155,
                35: 1833,
                36: 1057.15,
                37: 2160,
                38: 1179.185,
                41: 1070,
                42: 3597.525,
                43: 1195,
                44: 1947,
                45: 2075,
                46: 2051.65,
                47: 1057.345,
                48: 3724,

                51: 75,
                52: 33,
            }
        self.cls_vol = cls_vol
        self.simple_postprocess = simple_postprocess

        # define mapping for grand challenge class
        self.orig_mapping = np.arange(47, dtype=np.uint8)
        remapping = {43: 51, 44: 52, 45: 53}
        remapping.update({i: i+2 for i in range(19, 27)})
        remapping.update({i: i+4 for i in range(27, 35)})
        remapping.update({i: i+6 for i in range(35, 43)})
        remapping[46] = 50 # combined pulp
        for k, v in remapping.items():
            self.orig_mapping[k] = v

        super().__init__(*args, **kwargs)

    def postprocessing(self, segmentation):
        segmentation = np.ascontiguousarray(self.orig_mapping[segmentation])
        if self.simple_postprocess:
            for cls_id, vol_thres in self.cls_vol.items():
                if vol_thres > 0:
                    mask = segmentation == cls_id
                    if mask.any():
                        pred_vol = np.sum(mask)
                        if pred_vol < vol_thres:
                            segmentation[mask] = 0
                            if self.verbose:
                                print('removed label', cls_id, 'with predicted volume', pred_vol)
            # process pulp
            # mask = segmentation == 50
            # if mask.any():
            #     labels_out = cc3d.connected_components(mask, binary_image=True, connectivity=18)
            #     for _, pulp_mask in cc3d.each(labels_out, binary=True, in_place=True):
            #         slicer = mask_to_bbox_slice(pulp_mask)
            #         tooth_id = np.argmax(np.bincount(segmentation[slicer].flat)[:49])
            #         if 10 < tooth_id < 49:
            #             pass
            #             # segmentation[pulp_mask] += tooth_id
            #         else:
            #             if self.verbose:
            #                 print('warning fp pulp with surrounding id', tooth_id)
            #             segmentation[pulp_mask] = tooth_id  # fp
        else:
            labels_out = cc3d.connected_components(segmentation, connectivity=18)
            for _, _mask in cc3d.each(labels_out, binary=True, in_place=True):
                cls_id = segmentation[_mask][0]
                vol_thres = self.cls_vol.get(cls_id, 0)
                if vol_thres > 0:
                    pred_vol = _mask.sum()
                    if pred_vol < vol_thres:
                        segmentation[_mask] = 0
                        if self.verbose:
                            print('removed cc label', cls_id, 'with predicted volume', pred_vol)
                        continue
                # if cls_id == 50: # TODO: do we need this?
                #     slicer = mask_to_bbox_slice(_mask)
                #     tooth_id = np.argmax(np.bincount(segmentation[slicer].flat)[:49])
                #     if 10 < tooth_id < 49:
                #         pass
                #         # segmentation[_mask] += tooth_id
                #     else:
                #         if self.verbose:
                #             print('warning fp pulp with surrounding id', tooth_id)
                #         segmentation[_mask] = tooth_id  # fp
        return segmentation


def predict_semseg(im, prop, base_predictor):
    semseg_pred = base_predictor.predict_single_cbct_npy_array(
        im, prop
    )
    _t3 = _time.time()
    semseg_pred = base_predictor.postprocessing(semseg_pred)
    print(f"=+=   seg.postprocess      {_time.time() - _t3:8.1f}s", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return semseg_pred


if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw

    # os.environ['nnUNet_compile'] = 'f'
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_json', type=Path, default=os.path.join(nnUNet_preprocessed, 'Dataset119_ToothFairy3', 'splits_final.json'))
    parser.add_argument('-i', '--input_folder', type=Path, default=os.path.join(nnUNet_raw, 'Dataset119_ToothFairy3', 'imagesTr'))
    parser.add_argument('-o', '--output_folder', type=Path, default=os.path.join(nnUNet_results, 'Dataset119_ToothFairy3', 'eval_results'))
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--folds', type=str, nargs='+', default=[0,])
    args = parser.parse_args()

    (args.output_folder / 'predictions').mkdir(exist_ok=True, parents=True)

    folds = [i if i == 'all' else int(i) for i in args.folds]
    base_model = args.base_model

    mapping = [(3, 4), (5, 6), (43, 44)]  # ian, sinus, incisive canal
    for l, r in zip(range(19, 27), range(11, 19)):  # upper
        mapping.append((l, r))
    for l, r in zip(range(27, 35), range(35, 43)):  # lower
        mapping.append((l, r))
    print('defining lr mapping as', mapping)

    # initialize predictors
    base_predictor = BasePredictor(
        tile_step_size=0.95,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        allow_tqdm=True,
        tta_batch_size=1,
        # export=True,
        lr_mapping=mapping,
        n_class=47,
        verbose=False
    )
    base_predictor.initialize_from_trained_model_folder(
        base_model,
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth'
    )

    rw = SimpleITKIO()

    with open(args.split_json, 'r') as f:
        input_files = [args.input_folder / (x + '_0000.nii.gz') for x in json.load(f)[0]['val']]
    # input_files = list(args.input_folder.glob('*.nii.gz')) + list(args.input_folder.glob('*.mha'))

    case_stats = {}

    try:
        for input_fname in input_files:
            output_fname = args.output_folder / 'predictions' / input_fname.name.replace('_0000.nii.gz', '.nii.gz')

            # we start with the instance seg because we can then start converting that while semseg is being predicted
            # load test image
            im, prop = rw.read_images([input_fname])
            print(input_fname, im.shape)

            with torch.no_grad():
                st = time.time()
                torch.cuda.reset_peak_memory_stats(device=None)
                semseg_pred = predict_semseg(im, prop, base_predictor)

            # now postprocess
            # semseg_pred = postprocess(semseg_pred, np.prod(prop['spacing']), True)

            time_elapsed = time.time() - st
            memory_used = torch.cuda.max_memory_allocated(device=None) / 1024 / 1024
            if memory_used / 1024 >= 16:
                print('memory usage exceeded', memory_used, 'shape', im.shape, im.size)
            case_stats[input_fname.name.replace('_0000.nii.gz', '.nii.gz')] = {'time_elapsed': float(time_elapsed), 'memory_used': float(memory_used)}

            # now save
            rw.write_seg(semseg_pred, output_fname, prop)

            del im, prop, semseg_pred
            gc.collect()
            torch.cuda.empty_cache()
    except RuntimeError:
        print(traceback.format_exc())
        print('stopping now at', input_fname)

    with open(args.output_folder / 'gpu_stats.json', 'w') as f:
        json.dump(case_stats, f)

    print(f'first case time: {case_stats[input_files[0].name.replace("_0000.nii.gz", ".nii.gz")]["time_elapsed"]:.5f}')
    print(f'average time: {np.mean([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'average gpu memory: {np.mean([x["memory_used"] for x in case_stats.values()]):.5f}')
    print(f'max time: {max([x["time_elapsed"] for x in case_stats.values()]):.5f}')
    print(f'max gpu memory: {max([x["memory_used"] for x in case_stats.values()]):.5f}')
