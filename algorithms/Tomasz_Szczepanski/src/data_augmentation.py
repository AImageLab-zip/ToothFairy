import torch

from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice
)

from monai.transforms import (
    EnsureChannelFirstD,
    AddChannelD,
    CropForegroundD,
    ConcatItemsD,
    EnsureTypeD,
    GaussianSmoothD,
    LoadImageD,
    OrientationD,
    ResizeD,
    RandAdjustContrastD,
    RandSpatialCropD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandSpatialCropSamplesD,
    RandGaussianSmoothD,
    RandZoomD,
    RandFlipD,
    RandRotate90D,
    RandRotateD,
    RandAffineD,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
    SpacingD,
    SpatialPadD,
    ThresholdIntensityD,
    ToDeviceD,
    Invertd,
)
# from src.custom_augmentation import CropForegroundFixedD

class Transforms():
    def __init__(self,
                 args,
                 device : str = 'cpu'
                ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.class_treshold = args.classes if args.classes == 1 else args.classes-1
        keys = args.keys
        spatial_pad_size = args.spatial_pad_size

        if len(keys)==2:
            self.train_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NumpyReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    # OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ScaleIntensityRangeD(keys="image",
                        a_min=0,
                        a_max=args.houndsfield_clip,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                    ResizeWithPadOrCropD(keys=keys, spatial_size=spatial_pad_size, method = 'symmetric', mode='constant', constant_values=0),
                    #######################################
                    #GEOMETRIC - RANDOM - DATA AUGMENTATION
                    # RandRotate90D(keys=keys, prob=0.5, spatial_axes=(1,2)),
                    RandFlipD(keys=keys, prob=0.5, spatial_axis=(1,2)),
                    RandRotateD(keys=keys, range_x=0.15, range_y=0.15, range_z=0.15, mode=("bilinear", "nearest"), prob=0.5),
                    RandAffineD(keys=keys, prob=1.0, translate_range=(0.1, 0.1, 0.1), scale_range=[(-0.1, 0.1),(-0.1, 0.1),(-0.1, 0.1)], mode=("bilinear", "nearest"), padding_mode='zeros'),
                    # CropForegroundFixedD(keys=keys,
                    #                     source_key="label",
                    #                     select_fn=lambda x: x > 0,
                    #                     margin=args.spatial_crop_margin,
                    #                     spatial_size=args.spatial_crop_size,
                    #                     mode='constant',
                    #                     return_coords=True,
                    #                     constant_values=(-1000, 0)),
                    ##image
                    ##label
                    # ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    # ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    # RandSpatialCropD(keys=keys,
                    #                     roi_size=args.patch_size,
                    #                     random_center=True,
                    #                     random_size=False),
                    #######################################
                    #INTENSITY - RANDOM - DATA AUGMENTATION
                    RandAdjustContrastD(keys="image",
                                        gamma=(0.5, 2.0),
                                        prob=0.25),
                    RandShiftIntensityD(keys="image", offsets=0.25, prob=0.5),
                    RandScaleIntensityD(keys="image", factors=0.2, prob=0.5),
                    #FINAL CHECK
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    ToDeviceD(keys=keys, device=device)
                ]
            )
        #keys : 'input': image, gcr (probability-map-sigmoid: float <0,1>), 'labels': 'dense'
        if len(keys)==3:
            self.train_transform_2ch = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NumpyReader'),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    # OrientationD(keys=keys, axcodes="RAS"),
                    #GEOMETRIC - NON-RANDOM - PREPROCESSING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "bilinear", "nearest")),
                    ResizeWithPadOrCropD(keys=keys, spatial_size=args.spatial_pad_size, method = 'symmetric', mode='constant', constant_values=0),
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ScaleIntensityRangeD(keys="image",
                        a_min=0,
                        a_max=args.houndsfield_clip,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True),
                    # only for pretraining 
                    GaussianSmoothD(keys='gcr', sigma=1.0),
                    #######################################
                    #GEOMETRIC - RANDOM - DATA AUGMENTATION
                    RandSpatialCropSamplesD(keys=keys, roi_size=args.patch_size, random_center=True, random_size=False, num_samples=args.crop_num_samples),
                    # only for pretraining 
                    RandGaussianSmoothD(keys='gcr', prob=0.25),
                    # RandSpatialCropD(keys=keys, roi_size=args.patch_size, random_center=True, random_size=False),
                    RandFlipD(keys=keys, prob=0.5, spatial_axis=(1,2)),
                    RandRotateD(keys=keys, range_x=0.15, range_y=0.15, range_z=0.15, mode=("bilinear", "bilinear", "nearest"), prob=0.5),
                    RandAffineD(keys=keys, prob=1.0, translate_range=(0.1, 0.1, 0.1), scale_range=[(-0.1, 0.1),(-0.1, 0.1),(-0.1, 0.1)], mode=("bilinear", "bilinear", "nearest"), padding_mode='zeros'),
                    ##image
                    ##label
                    # ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    # ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    # ToDeviceD(keys=keys, device=device),
                    # RandSpatialCropD(keys=keys,
                    #                     roi_size=args.patch_size,
                    #                     random_center=True,
                    #                     random_size=False),
                    #######################################
                    #INTENSITY - RANDOM - DATA AUGMENTATION
                    RandAdjustContrastD(keys="image",
                                        gamma=(0.5, 2.0),
                                        prob=0.25),
                    RandShiftIntensityD(keys="image", offsets=0.25, prob=0.5),
                    RandScaleIntensityD(keys="image", factors=0.2, prob=0.5),
                    #CONCAT gcr - global context reference with image
                    ConcatItemsD(keys=["image", "gcr"], name="image", dim=0),
                    #FINAL CHECK
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    ToDeviceD(keys=keys, device=device)
                ]
            )

        if len(keys)==2:
            self.val_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NumpyReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    # OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor"),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                    # CropForegroundFixedD(keys=keys,
                    #                     source_key="label",
                    #                     select_fn=lambda x: x > 0,
                    #                     margin=args.spatial_crop_margin,
                    #                     spatial_size=args.spatial_crop_size,
                    #                     mode='constant',
                    #                     return_coords=True,
                    #                     constant_values=(-1000, 0)),
                    #NON-RANDOM - perform on GPU
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ##image
                    ScaleIntensityRangeD(keys="image",
                                        a_min=0,
                                        a_max=args.houndsfield_clip,
                                        b_min=0.0,
                                        b_max=1.0,
                                        clip=True),
                    ##label
                    # ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    # ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    EnsureTypeD(keys=keys, data_type="tensor"),
                    ToDeviceD(keys=keys, device=device)
                ]
            )
            self.val_invert_transform = Compose([
                Invertd(
                    keys=["pred", "dense"],
                    transform=self.val_transform,
                    orig_keys=["dense", "dense"],
                    to_tensor=True,
                )
            ])
        if len(keys)==3:
            self.val_transform_2ch = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NumpyReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    # OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    #GEOMETRIC - NON-RANDOM - PREPROCESSING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "bilinear", "nearest")),
                    ResizeWithPadOrCropD(keys=keys, spatial_size=args.spatial_pad_size, method = 'symmetric', mode='constant', constant_values=0),
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ScaleIntensityRangeD(keys="image",
                        a_min=0,
                        a_max=args.houndsfield_clip,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True),
                    GaussianSmoothD(keys='gcr', sigma=1.0),
                    #CONCAT
                    ConcatItemsD(keys=["image", "gcr"], name="image", dim=0),
                    #######################################
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    ToDeviceD(keys=keys, device=device)
                ]
            )

            self.val_invert_transform_2ch = Compose([
                Invertd(
                    keys=["pred", "dense"],
                    transform=self.val_transform_2ch,
                    orig_keys=["dense", "dense"],
                    to_tensor=True,
                )
            ])

        self.binarize_transform = ThresholdIntensityD(keys="label", above=False, threshold=1, cval=1)

        if args.classes > 1:
            self.post_pred = Compose([Activations(softmax=True, dim=0),
                                      AsDiscrete(argmax=True,
                                                 dim=0,
                                                 keepdim=True),
                                      ToDevice(device=device)
                                    ])
            self.post_pred_labels = Compose([AsDiscrete(argmax=False,
                                                        to_onehot=args.classes,
                                                        dim=0),
                                             ToDevice(device=device)
                                            ])
        elif args.classes == 1:
            self.post_pred = Compose([
                                    # Activations(sigmoid=True),
                                      AsDiscrete(threshold=0.5)],
                                      ToDevice(device=device))

# GPU accelerated morphological dillation and errosion - torch based
def dilation2d(image : torch.tensor, kernel : torch.tensor, border_type: str = 'constant', border_value: int = 0):

    _, _, se_h, se_w = kernel.shape
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(image, pad_margin, mode=border_type, value=border_value)
    out = torch.nn.functional.conv2d(volume_pad, kernel, padding=0).to(torch.int)
    dilation_out = torch.clamp(out,0,1)
    return dilation_out


def dilation3d(volume : torch.tensor, kernel : torch.tensor = torch.ones((1,1,3,3,3)), border_type: str = 'constant', border_value: int = 0):

    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape

    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(volume, pad_margin, mode=border_type, value=border_value)
    out = torch.nn.functional.conv3d(volume_pad, kernel, padding=0)
    dilation_out = torch.clamp(out,0,1)
    return dilation_out


def erosion2d(image : torch.tensor, kernel : torch.tensor, border_type: str = 'constant', border_value: int = 0):

    _, _, se_h, se_w = kernel.shape
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(image, pad_margin, mode=border_type, value=border_value)
    if torch.is_tensor(kernel):
        bias=-kernel.sum().unsqueeze(0)
    else:
        bias=torch.tensor(-kernel.sum()).unsqueeze(0)
    out = torch.nn.functional.conv2d(volume_pad, kernel, padding=0, bias=bias).to(torch.int)
    erosion_out = torch.add(torch.clamp(out,-1,0),1)
    return erosion_out


def erosion3d(volume : torch.tensor, kernel : torch.tensor = torch.ones((1,1,3,3,3)), border_type: str = 'constant', border_value: int = 0):

    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape

    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(volume, pad_margin, mode=border_type, value=border_value)
    if torch.is_tensor(kernel):
        bias=-kernel.sum().unsqueeze(0)
    else:
        bias=torch.tensor(-kernel.sum()).unsqueeze(0)
    out = torch.nn.functional.conv3d(volume_pad, kernel, padding=0, stride=1, bias=bias)
    erosion_out = torch.add(torch.clamp(out,-1,0),1)
    return erosion_out
