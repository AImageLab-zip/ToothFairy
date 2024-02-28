import argparse

parser = argparse.ArgumentParser(description="Training Segmentation Network on CBCT Dataset.")

parser.add_argument("--data",
                    type=str,
                    default="./ToothFairy/Dataset",
                    help="Path to the data.")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default='./Checkpoints/ToothFairy',
                    help="Directory for model checkpoints")
parser.add_argument("--inference_checkpoint",
                    type=str,
                    default='./Checkpoints/ToothFairy/ResUNet/LQ_SYNTH2_ResUnet18_tfloat32/model-ResUnet18-fold-0_current_best_test.pt',
                    help="Path to checkpoint for inference")
parser.add_argument("--inference_save_output",
                    type=bool,
                    default=True,
                    help="Save output predictions")
parser.add_argument("--inference_save_name",
                    type=str,
                    default="gcr_resunet",
                    help="Save output predictions as gt_{inference_save_name}.npy.")
parser.add_argument("--cache_dir",
                    type=str,
                    default="./CachedDatasets/ToothFairy",
                    help="Path to the root dir of cache for PersistentDataset.")
parser.add_argument("--clear_cache",
                    type=bool,
                    default=False,
                    help="Delete train and validation cache before training.")
parser.add_argument("--model_name",
                    type=str, 
                    default="ResUnet18",
                    choices=["ResUnet18", "ResUnet50"],
                    help="Type of the model.")
parser.add_argument("--big_decoder",
                    type=bool, 
                    default="True",
                    help="If should use the deep decoder or a shallow one.")
parser.add_argument("--use_residuals",
                    type=int, 
                    default=0,
                    choices=[0,1],
                    help="Type of the model.")
parser.add_argument("--in_channels",
                    type=int,
                    default=1,
                    help="Number of channels of model input.")
parser.add_argument("--scheduler_name",
                    type=str,
                    default="warmup",
                    choices=["annealing", "warmup", "warmup_restarts", "plateau"],
                    help="Type of the scheduler.")
parser.add_argument("--warmup_steps",
                    type=int,
                    default=1,
                    help="Number of steps of gradual lr increase.")
parser.add_argument("--scheduler_gamma",
                    type=float,
                    default=0.1,
                    help="Gamma for max lr scheduler with restarts.")
parser.add_argument("--binary_warmup",
                    type=int,
                    default=0,
                    help="Only binary loss to this epoch number.")
parser.add_argument("--ce_warmup",
                    type=int,
                    default=0,
                    help="CE and binary loss to this epoch number.")
parser.add_argument("--classes",
                    type=int,
                    default=1,
                    help="Number of classes in the dataset - background included.")
parser.add_argument("--loss_name",
                    type=str,
                    default="DiceLoss",
                    choices=["DiceCELoss", "DiceLoss", "UnifiedFocalLoss", "WasserDiceCELoss", "MSE"],
                    help="Type of the loss - by name.")
parser.add_argument("--weighted_ce",
                    type=bool,
                    default=True,
                    help="Use weights for cross entropy loss.")
parser.add_argument("--inter_quarter_penalty",
                    type=bool,
                    default=True,
                    help="Use penalties between quarters for gwdl loss.")
parser.add_argument("--loss_auto",
                    type=bool,
                    default=True,
                    help="Auto choice of loss based on number of classes.")
parser.add_argument("--include_background",
                    type=bool,
                    default=True,
                    help="If include background in loss citerion and metric, works when more than 1 class.")
parser.add_argument("--split",
                    type=int,
                    default=5,
                    help="Number of splits for kfold.") 
parser.add_argument("--use_json_splits",
                    type=bool,
                    default=False,
                    help="Use predefined splits from json file.") 
parser.add_argument("--keys",
                    type=dict,
                    default=['image', 'dense'], 
                    help="All keys to perform monai augmentation pipeline.")
parser.add_argument("--seed",
                    type=int,
                    default=-1,
                    help="Seed value - if minus one - will not set determnistic training")
parser.add_argument("--epochs",
                    type=int,
                    default=301,
                    help="Number of epochs.")
parser.add_argument("--patch_size",
                    type=tuple,
                    default=(96,256,256),
                    help="Patch size.")
parser.add_argument("--spatial_crop_size",
                    type=tuple,
                    default=(224, 192, 160),
                    help="Spatial size for foreground crop based on labels position.")
parser.add_argument("--spatial_pad_size",
                    type=tuple,
                    default=(96, 256, 256),
                    help="Spatial size for foreground crop based on labels position.")
parser.add_argument("--spatial_crop_margin",
                    type=tuple,
                    default=(32, 32, 32),
                    help="Spatial size margin for foreground crop based on labels position.")
parser.add_argument("--padding_size",
                    type=tuple,
                    default=(256, 256, 256),
                    help="Padding size - symetrical with zeros.")
parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="Number of items for dataloader batch.")
parser.add_argument("--norm",
                    type=str,
                    default='batch',
                    choices=['instance', 'batch', 'group'],
                    help="Type of norm for model.")
parser.add_argument("--batch_size_val",
                    type=int,
                    default=1,
                    help="Number of items for dataloader batch.")
parser.add_argument("--pixdim",
                    type=float,
                    default=2.0,
                    help="Pixels dimensions for spacing preprocessing when using PersistentDataset.")   
parser.add_argument("--houndsfield_clip",
                    type=int,
                    default=2100,
                    help="Upper value of hounsfield unit for data intensity scaling - values above will be clipped.")            
parser.add_argument("--num_workers",
                    type=int,
                    default=0,
                    help="Number of processes loading data for dataloader batch.")
parser.add_argument("--num_threads",
                    type=int,
                    default=24,
                    help="Number of threads loading data for dataloader batch.")
parser.add_argument("--pin_memory",
                    type=bool,
                    default=False,
                    help="Number of threads loading data for dataloader batch.")
parser.add_argument("--feature_size",
                    type=int,
                    default=48,
                    help="Number of Transformer's features.")
parser.add_argument("--n_features",
                    type=int,
                    default=32,
                    help="Initial number of feature maps for UNET.")
parser.add_argument("--unet_depth",
                    type=int,
                    default=5,
                    help="Number of depth blocks for UNET architecture.")
parser.add_argument("--lr",
                    type=float,
                    default=1e-3,
                    help="Initial learning rate value.")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-5,
                    help="Number of weight decay")
parser.add_argument("--background_weight",
                    type=float,
                    default=0.1,
                    help="Background weights for ce loss.")
parser.add_argument("--step_lr",
                    type=int,
                    default=300,
                    help="Learning rate step value")
parser.add_argument("--lr_gamma",
                    type=float,
                    default=0.1,
                    help="Step learning rate factor")
parser.add_argument("--optimizer",
                    type=str,
                    default="AdamW",
                    choices=["Adam", "AdamW", "SGD"],
                    help="Type of optimizer.")
parser.add_argument("--adam_ams",
                    type=bool,
                    default=False,
                    help="Use amsgrad adam.")
parser.add_argument("--adam_eps",
                    type=float,
                    default=1e-8,
                    help="Numerical stability for adam")
parser.add_argument("--inference_mode",
                    type=str,
                    default='constant',
                    choices=["constant","gaussian"],
                    help="Use amsgrad adam.")
parser.add_argument("--activation_checkpoints",
                    type=bool,
                    default=False,
                    help="Use checkpoints for gradient calculation.")
parser.add_argument("--gradient_accumulation",
                    type=int,
                    default=1,
                    help="Number of interations to accumlate.")
parser.add_argument("--parallel",
                    type=bool,
                    default=False,
                    help="Parallel learning.")
parser.add_argument("--gpu_frac",
                    type=float,
                    default=1.0,
                    help="Fraction of GPU memory available for training. Default 100% memory")
parser.add_argument("--continue_training",
                    type=bool,
                    default=False,
                    help="if continue training from checkpoint provided in trained_model")
parser.add_argument("--trained_model",
                    type=str,
                    default="/home/tf/Checkpoints/ToothFairy/EffUNet/LQ_SYNTH2_FP32/model-efficientnet-b5-fold-0_current_best_train.pt",
                    help="Path to the trained model state dictionary")
parser.add_argument("--use_scaler",
                    type=bool,
                    default=False,
                    help="Use amp scaler with bfloat16 to speed up training.")
parser.add_argument("--autocast_dtype",
                    type=str,
                    default="tfloat32",
                    choices=['float16','float32','float64', 'bfloat16', 'tfloat32'],
                    help="autocast data type")
parser.add_argument("--grad_clip", 
                    type=bool,
                    default=True,
                    help="gradient clip")
parser.add_argument("--max_grad_norm", 
                    default=1.0, 
                    type=float, 
                    help="maximum gradient norm")
parser.add_argument("--save_interval",
                    type=int,
                    default=100,
                    help="Epochs interval between model saves") 
parser.add_argument("--save_optimiser_interval",
                    type=int,
                    default=500,
                    help="Epochs interval between model saves with optimiser if save_optimizer is used") 
parser.add_argument("--save_optimizer",
                    type=bool,
                    default=True,
                    help="Save optimizer dict together with model to continue training.")
parser.add_argument("--start_epoch",
                    type=int,
                    default=0,
                    help="Epoch that model starts training.")
parser.add_argument("--comet",
                    type=bool,
                    default=True,
                    help="Log training with comet ml.")
parser.add_argument("--print_config",
                    type=bool,
                    default=False,
                    help="Print monai config.")
parser.add_argument("--tags",
                    type=str,
                    default='LQ#RES18#SYNTH2#FP32#KFOLD1_5#BIGDECODER',
                    help="Tags to easier track experiments in comet ml separated by #")
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
                    default=0,
                    help="Visible CUDA GPU index to be used for training.")
parser.add_argument("--validation_interval",
                    type=int,
                    default=25,#25
                    help="Epochs interval between validations.")
parser.add_argument("--test_interval",
                    type=int,
                    default=25,#25
                    help="Epochs interval between testset evaluations.")
parser.add_argument("--metrics_interval",
                    type=int,
                    default=5,
                    help="Epochs interval between metrics calculation.")
#INTERVALS
parser.add_argument("--log_batch_interval",
                    type=int,
                    default=10,
                    help="Batch interval of console print of model loss per epoch.")
parser.add_argument("--log_slice_interval",
                    type=int,
                    default=10,#10
                    help="Number of epochs interval to log predictions slices - X,Y,Z.")
parser.add_argument("--log_3d_scene_interval_training",
                    type=int,
                    default=50,#50
                    help="Number of epochs interval to log 3d pyvista scene on training sample.")
parser.add_argument("--log_3d_scene_interval_validation",
                    type=int,
                    default=50,#50
                    help="Number of epochs interval to log 3d pyvista scene on validation sample.")
parser.add_argument("--log_3d_scene_interval_test",
                    type=int,
                    default=50,#50
                    help="Number of epochs interval to log 3d pyvista scene on testset sample.")
#LOGGER FLAG
parser.add_argument("--is_log_image",
                    type=bool,
                    default=True,
                    help="if log results image to logger")
parser.add_argument("--is_log_3d",
                    type=bool,
                    default=True,
                    help="if log 3d images with pyvista")
args = parser.parse_args()
