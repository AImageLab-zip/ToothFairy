from monai.transforms import (
    Compose
)

from monai.transforms import (
    EnsureChannelFirstD,
    ConcatItemsD,
    EnsureTypeD,
    GaussianSmoothD,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
    SpacingD,
    ToDeviceD,
    Invertd,
    LoadImageD
)


class Transforms:
    def __init__(self,
                 args,
                 device: str = 'cpu',
                 inference_mode: str = "LQ"
                 ) -> None:

        self.pixdim_hq = (args.pixdim_hq,) * 3
        self.pixdim_lq = (args.pixdim_lq,) * 3
        keys_lq = args.keys_lq
        keys_hq = args.keys_hq

        if inference_mode == "LQ":
            self.val_transform_lq = Compose(
                [
                    # INITAL SETUP
                    LoadImageD(keys=keys_lq, reader="NumpyReader"),
                    EnsureChannelFirstD(keys=keys_lq, channel_dim='no_channel'),
                    ToDeviceD(keys=keys_lq, device=device),
                    EnsureTypeD(keys=keys_lq, data_type="tensor"),
                    # GEOMETRIC - NON-RANDOM - PREPROCESING
                    ResizeWithPadOrCropD(keys=keys_lq, spatial_size=(192, 512, 512), method='symmetric',
                                         mode='constant', constant_values=0),
                    SpacingD(keys=keys_lq, pixdim=self.pixdim_lq, mode=("bilinear")),
                    # INTENSITY - NON-RANDOM - PREPROCESING
                    ##image
                    ScaleIntensityRangeD(keys="image",
                                         a_min=0,
                                         a_max=args.houndsfield_clip,
                                         b_min=0.0,
                                         b_max=1.0,
                                         clip=True),
                    ##label
                    EnsureTypeD(keys=keys_lq, data_type="tensor"),
                    ToDeviceD(keys=keys_lq, device=device)
                ]
            )
            self.val_invert_transform_lq = Compose([
                Invertd(
                    keys=["pred"],
                    transform=self.val_transform_lq,
                    orig_keys=["image"],
                    to_tensor=True,
                )
            ])

        else:
            self.val_transform_hq = Compose(
                [
                    # INITAL SETUP
                    LoadImageD(keys=keys_hq, reader="NumpyReader"),
                    EnsureChannelFirstD(keys=keys_hq, channel_dim='no_channel'),
                    ToDeviceD(keys=keys_hq, device=device),
                    EnsureTypeD(keys=keys_hq, data_type="tensor", device=device),
                    # GEOMETRIC - NON-RANDOM - PREPROCESSING
                    SpacingD(keys=keys_hq, pixdim=self.pixdim_hq, mode=("bilinear", "bilinear")),
                    ResizeWithPadOrCropD(keys=keys_hq, spatial_size=args.spatial_pad_size_hq, method='symmetric',
                                         mode='constant', constant_values=0),
                    # INTENSITY - NON-RANDOM - PREPROCESING
                    ScaleIntensityRangeD(keys="image",
                                         a_min=0,
                                         a_max=args.houndsfield_clip,
                                         b_min=0.0,
                                         b_max=1.0,
                                         clip=True),
                    # CONCAT
                    ConcatItemsD(keys=["image", "gcr"], name="image", dim=0),
                    #######################################
                    EnsureTypeD(keys=keys_hq, data_type="tensor", device=device)
                ]
            )

            self.val_invert_transform_hq = Compose([
                Invertd(
                    keys=["pred"],
                    transform=self.val_transform_hq,
                    orig_keys=["gcr"],
                    to_tensor=True,
                )
            ])
