import time

# TORCH
import torch
from monai.data import set_track_meta, decollate_batch, Dataset, DataLoader
from monai.inferers import sliding_window_inference

# external modules
from src.data_augmentation import Transforms
from src.models.resunet import ResUNet

import numpy as np


def inference_LQ(img_numpy_path, args, device):
    print("Starting inference LQ...")
    start_time_testing = time.time()
    full_prediction = None
    for checkpoint_path in args.checkpoint_lq:

        if args.model_name_lq == "ResUnet18":
            model = ResUNet(spatial_dims=3, in_channels=args.in_channels_lq, out_channels=args.classes, act='relu',
                            norm=args.norm,
                            backbone_name='resnet18', bias=False, big_decoder=True)
        elif args.model_name_lq == "ResUnet50":
            model = ResUNet(spatial_dims=3, in_channels=args.in_channels_lq, out_channels=args.classes, act='relu',
                            norm=args.norm,
                            backbone_name='resnet50', bias=False)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        trans = Transforms(args, device, inference_mode="LQ")
        set_track_meta(True)

        dataset = Dataset([{"image": img_numpy_path}], trans.val_transform_lq)
        loader = DataLoader(dataset, batch_size=1)
        data = next(iter(loader))

        data["pred"] = sliding_window_inference(data["image"], roi_size=args.patch_size_lq, sw_batch_size=8,
                                                predictor=model,
                                                overlap=0.6, sw_device=device,
                                                device=device, mode=args.inference_mode, sigma_scale=0.125,
                                                padding_mode='constant', cval=0, progress=True)

        val_pred = [trans.val_invert_transform_lq(seg_pred)["pred"] for seg_pred in decollate_batch(data)]

        if full_prediction is None:
            full_prediction = val_pred[0].detach().cpu().numpy()
        else:
            full_prediction = full_prediction+val_pred[0].detach().cpu().numpy()

    full_prediction = full_prediction/len(args.checkpoint_lq)
    test_time = time.time() - start_time_testing
    print(f"Finished inference LQ: {test_time:.2f}s")
    return full_prediction
