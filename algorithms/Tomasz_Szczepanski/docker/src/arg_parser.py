import argparse

parser = argparse.ArgumentParser(description="Training Segmentation Network on CBCT Dataset.")
parser.add_argument("--checkpoint_hq",
                    nargs='+',
                    default=["checkpoints/HQ/model-ResUnet18-fold-0_current_best_val.pt",
                             "checkpoints/HQ/model-ResUnet18-fold-1_current_best_val.pt",
                             "checkpoints/HQ/model-ResUnet18-fold-2_current_best_val.pt",
                             "checkpoints/HQ/model-ResUnet18-fold-3_current_best_val.pt",
                             "checkpoints/HQ/model-ResUnet18-fold-4_current_best_val.pt",],
                    help="Path to the checkpoint HQ.")
parser.add_argument("--checkpoint_lq",
                    nargs='+',
                    default=["checkpoints/LQ/model-ResUnet18-fold-0_current_best_val.pt",
                             "checkpoints/LQ/model-ResUnet18-fold-1_current_best_val.pt",
                             "checkpoints/LQ/model-ResUnet18-fold-2_current_best_val.pt",
                             "checkpoints/LQ/model-ResUnet18-fold-3_current_best_val.pt",
                             "checkpoints/LQ/model-ResUnet18-fold-4_current_best_val.pt",],
                    help="Path to the checkpoint LQ.")
parser.add_argument("--model_name_lq",
                    type=str,
                    default="ResUnet18",
                    choices=["ResUnet18", "ResUnet50"],
                    help="Type of the model.")
parser.add_argument("--model_name_hq",
                    type=str,
                    default="ResUnet18",
                    choices=["ResUnet18", "ResUnet50"],
                    help="Type of the model.")
parser.add_argument("--in_channels_hq",
                    type=int,
                    default=2,
                    help="Number of channels of model input HQ.")
parser.add_argument("--in_channels_lq",
                    type=int,
                    default=1,
                    help="Number of channels of model input LQ.")
parser.add_argument("--classes",
                    type=int,
                    default=1,
                    help="Number of classes in the dataset - background included.")
parser.add_argument("--keys_hq",
                    type=dict,
                    default=['image', 'gcr'],
                    help="All keys to perform monai augmentation pipeline.")
parser.add_argument("--keys_lq",
                    type=dict,
                    default=['image'],
                    help="All keys to perform monai augmentation pipeline.")
parser.add_argument("--patch_size_hq",
                    type=tuple,
                    default=(128, 128, 128),
                    help="Patch size.")
parser.add_argument("--patch_size_lq",
                    type=tuple,
                    default=(96, 256, 256),
                    help="Patch size.")
parser.add_argument("--spatial_pad_size_hq",
                    type=tuple,
                    default=(168, 280, 360),
                    help="Spatial size for crop or pad.")
parser.add_argument("--spatial_pad_size_lq",
                    type=tuple,
                    default=(96, 256, 256),
                    help="Spatial size for crop or pad.")
parser.add_argument("--spatial_crop_margin",
                    type=tuple,
                    default=(32, 32, 32),
                    help="Spatial size margin for foreground crop based on labels position.")
parser.add_argument("--padding_size",
                    type=tuple,
                    default=(256, 256, 256),
                    help="Padding size - symetrical with zeros.")
parser.add_argument("--norm",
                    type=str,
                    default='batch',
                    choices=['instance', 'batch', 'group'],
                    help="Type of norm for model.")
parser.add_argument("--pixdim_hq",
                    type=float,
                    default=1.0,
                    help="Pixels dimensions for spacing preprocessing when using PersistentDataset.")
parser.add_argument("--pixdim_lq",
                    type=float,
                    default=2.0,
                    help="Pixels dimensions for spacing preprocessing when using PersistentDataset.")
parser.add_argument("--houndsfield_clip",
                    type=int,
                    default=2100,
                    help="Upper value of hounsfield unit for data intensity scaling - values above will be clipped.")
parser.add_argument("--gpu_frac",
                    type=float,
                    default=0.2,
                    help="Fraction of GPU memory available for training. Default 100% memory")
parser.add_argument("--use_scaler",
                    type=bool,
                    default=True,
                    help="Use amp scaler with bfloat16 to speed up training.")
parser.add_argument("--device",
                    type=str,
                    default="cuda",
                    help="Device used for training.")
parser.add_argument("--visible_devices",
                    type=str,
                    default="0,1",
                    help="Devices visible within system.")
parser.add_argument("--cuda_device_id",
                    type=int,
                    default=1,
                    help="Visible CUDA GPU index to be used for training.")
parser.add_argument("--inference_mode",
                    type=str,
                    default='gaussian',
                    choices=["constant", "gaussian"],
                    help="Use amsgrad adam.")
parser.add_argument("--use_residuals",
                    type=int,
                    default=0,
                    choices=[0, 1],
                    help="Type of the model.")
args = parser.parse_args()
