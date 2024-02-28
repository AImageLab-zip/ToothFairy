import time
import glob
import os
import warnings
import json
import numpy as np
from comet_ml import Experiment
from src.arg_parser_gcr_finetune import args
from natsort import natsorted

tuple2str = lambda a: '_'.join([str(x) for x in list(a)]) 
model_folder_name = "RES"
data_type = "SYNTH2" if args.use_synthetic else "DENSE-GT"
args.cache_dir = os.path.join(args.cache_dir, f"HQ_{data_type}_{model_folder_name}_2CH_{args.autocast_dtype}_{tuple2str(args.patch_size)}_finetune")

if not os.path.exists(args.cache_dir):
    os.makedirs(os.path.join(args.cache_dir, 'train'))
    os.makedirs(os.path.join(args.cache_dir, 'val'))
    os.makedirs(os.path.join(args.cache_dir, 'test'))

if args.clear_cache:
    print("Clearning cache...")
    train_cache = glob.glob(os.path.join(args.cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(args.cache_dir, 'val/*.pt'))
    test_cache = glob.glob(os.path.join(args.cache_dir, 'test/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(f"Cleared cache in dir: {args.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")

if args.comet:
    experiment = Experiment(
        api_key="",
        project_name="toothfairy",
        workspace=""
    )
    tags = args.tags.split('#')
    tags += [args.model_name]
    experiment.add_tags(tags)
    experiment.log_asset('src/arg_parse_gcr_finetune.py')
    name = experiment.get_name()
    experiment.set_name(name + '_' + args.tags)
else:
    from src.dummy_logger import DummyExperiment
    experiment = DummyExperiment()

# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler
from torch.nn import MSELoss, BCEWithLogitsLoss

#MONAI modules
from monai.networks.utils import one_hot
from monai.losses import DiceLoss
#segmentation
from monai.metrics import HausdorffDistanceMetric, MeanIoU, DiceMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader, decollate_batch
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
if args.print_config:
    print_config()

from sklearn.model_selection import KFold

#external modules
from src.cuda_stats import setup_cuda
from src.log_image import Logger
from src.scheduler import CosineAnnealingWarmupRestarts
from src.data_augmentation_fine import Transforms
from src.models.resunet import ResUNet

#config
if args.seed != -1:
    set_determinism(seed=args.seed)
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )
experiment.log_parameters(vars(args))

# use amp to accelerate training
scaler = None
if args.use_scaler:
    TORCH_DTYPES = {
    'bfloat16': torch.bfloat16, 
    'float16': torch.float16,     
    'float32': torch.float32,
    'tfloat32' : torch.float32, # backend flags
    'float64': torch.float64
    }
    if args.autocast_dtype == "tfloat32":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        args.use_scaler = False # scaller is not necessary for FP32, backend is already executed in TF32
    scaler = torch.cuda.amp.GradScaler()
    autocast_d_type=TORCH_DTYPES[args.autocast_dtype]
    if autocast_d_type == torch.bfloat16:
        os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"
        
#full reproducibility
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
#seeds
torch.manual_seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
#speedup
torch.backends.cudnn.benchmark = False
# precision for nn.modules : eg. nn.conv3d
torch.backends.cudnn.allow_tf32 = True
# precision for linear algebra - eg. interpolations and elastic transforms
torch.backends.cuda.matmul.allow_tf32 = True
# detect gradient errors - debug cuda C code
torch.autograd.set_detect_anomaly(False)

#LOGGER
log = Logger(args.classes, args.is_log_3d)

#CUDA
setup_cuda(args.gpu_frac, num_threads=args.num_threads, device=args.device, visible_devices=args.visible_devices, use_cuda_with_id=args.cuda_device_id)
if args.device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.cuda_device_id))

#TRANSFORMS
trans = Transforms(args, device)
set_track_meta(True)

#MONAI dataset
if args.use_json_splits:
    import json
    if "Maxillo" in args.data:
        #LOAD PATHS FRO MAXILLO DATA
        image_paths = natsorted(glob.glob(os.path.join(args.data, "SPARSE", '*', "data.npy"), recursive=False))
        dense_paths = natsorted(glob.glob(os.path.join(args.data, "DENSE", '*', "gt_alpha.npy"), recursive=False))
        sparse_paths = natsorted(glob.glob(os.path.join(args.data, "SPARSE", '*', "gt_sparse.npy"), recursive=False))
        print(f"Found: {len(image_paths)} scans, {len(dense_paths)} dense labels, {len(sparse_paths)} sparse labels.")
        # APPLY SPLITS
        with open(os.path.join(args.data, 'splits.json')) as splits_file:
            json_splits = json.load(splits_file)
        with open(os.path.join(args.data, 'low_quality_labels.json')) as splits_file:
            json_broken_labels = json.load(splits_file)
        low_quality_labels = json_broken_labels["broken"]+json_broken_labels["short"]
        patients_dense = [pth.split('/')[-2] for idx, pth in enumerate(dense_paths)]
        image_paths = [path for path in image_paths if path.split('/')[-2] in patients_dense]
        patinets_dense_dict = {pid: idx for idx, pid in enumerate(patients_dense)}
        train_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['train'] if patient_id not in low_quality_labels]
        val_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['val']]
        test_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['test']]
        print(f"Using data split based on splits.json: {len(train_split_ids)} train, {len(val_split_ids)} val, {len(test_split_ids)} test.")
    if "ToothFairy" in args.data:
        if not args.use_synthetic:
            # '*' is a wildcard for the patient id
            image_paths = natsorted(glob.glob(os.path.join(args.data, '*', "data.npy"), recursive=False))
            dense_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_alpha.npy"), recursive=False))
            gcr_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_gcr_resunet_soft.npy"), recursive=False))
            sparse_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_sparse.npy"), recursive=False))
            print(f"Found: {len(image_paths)} scans, {len(dense_paths)} dense labels, {len(sparse_paths)} sparse labels, {len(gcr_paths)} gcr paths.")
            # APPLY SPLITS
            with open(os.path.join(args.data.replace('/Dataset', '/'), 'splits.json')) as splits_file:
                json_splits = json.load(splits_file)
            #get patient ids with gt_alpha dense labels
            patients_dense = [pth.split('/')[-2] for idx, pth in enumerate(dense_paths)]
            image_paths = [path for path in image_paths if path.split('/')[-2] in patients_dense]
            patinets_dense_dict = {pid: idx for idx, pid in enumerate(patients_dense)}
            
            train_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['train']]
            val_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['val']]
            test_split_ids = [patinets_dense_dict[patient_id] for patient_id in json_splits['test']]
            print(f"Using gt data split based on splits.json: {len(train_split_ids)} train, {len(val_split_ids)} val, {len(test_split_ids)} test.")
        else:
            with open(os.path.join(args.data.replace('/Dataset', '/'), 'splits.json')) as splits_file:
                json_splits = json.load(splits_file)
            synthethic_patients = json_splits["synthetic"]
            # '*' is a wildcard for the patient id
            image_paths = natsorted(glob.glob(os.path.join(args.data, '*', "data.npy"), recursive=False))
            image_paths = [path for path in image_paths if path.split('/')[-2] in synthethic_patients]
            synthetic_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_synthetic2.npy"), recursive=False))
            dense_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_alpha.npy"), recursive=False))
            sparse_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_sparse.npy"), recursive=False))
            
            print(f"Found: {len(image_paths)} scans, {len(dense_paths)} dense labels, {len(sparse_paths)} sparse labels, {len(synthetic_paths)} synthethic dense labels.")
            
            from sklearn.model_selection import train_test_split
            data_ids = list(range(len(synthetic_paths)))
            train_split_ids, test_ids = train_test_split(data_ids, test_size=0.15, shuffle=True, random_state=48)
            val_split_ids, test_split_ids = train_test_split(test_ids, test_size=0.7, shuffle=True, random_state=48)
else:
    # '*' is a wildcard for the patient id
    image_paths = natsorted(glob.glob(os.path.join(args.data, '*', "data.npy"), recursive=False))
    dense_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_alpha.npy"), recursive=False))
    with open(os.path.join(args.data.replace('/Dataset', '/'), 'splits.json')) as splits_file:
        json_splits = json.load(splits_file)
    #get patient ids with gt_alpha dense labels
    patients_dense = [pth.split('/')[-2] for idx, pth in enumerate(dense_paths)]
    image_paths = [path for path in image_paths if path.split('/')[-2] in patients_dense]
    #genrated with LQ
    gcr_paths = natsorted(glob.glob(os.path.join(args.data, '*', "gt_gcr_resunet_soft_3f.npy"), recursive=False))
    print(f"Found: {len(image_paths)} scans, {len(dense_paths)} dense labels, {len(gcr_paths)} gcr paths.")
    
# keys image, gcr (for now dense is copied), dense. GCR = global context reference
if len(args.keys) == 3:
    if args.use_synthetic:
        data_dicts_list = [{args.keys[0]: image, args.keys[1]: gcr_synthetic, args.keys[2]: dense_synthetic} for (image, gcr_synthetic, dense_synthetic) in zip(image_paths, synthetic_paths, synthetic_paths)]
    else:
        data_dicts_list = [{args.keys[0]: image, args.keys[1]: gcr, args.keys[2]: dense} for (image, gcr, dense) in zip(image_paths, gcr_paths, dense_paths)]
else:
    data_dicts_list = [{args.keys[0]: image, args.keys[1]: dense} for (image, dense) in zip(image_paths, dense_paths)]
        
train_dataset = PersistentDataset(data_dicts_list, trans.train_transform_2ch, cache_dir=os.path.join(args.cache_dir, 'train'))
val_dataset = PersistentDataset(data_dicts_list, trans.val_transform_2ch, cache_dir=os.path.join(args.cache_dir, 'val'))
test_dataset = PersistentDataset(data_dicts_list, trans.val_transform_2ch, cache_dir=os.path.join(args.cache_dir, 'test'))

kfold = KFold(n_splits=args.split, shuffle=True, random_state=42)

if args.loss_name == "DiceLoss" and args.classes == 1:
    criterion = DiceLoss(sigmoid=False)
elif args.loss_name == "MSE":
    criterion = MSELoss()

criterion_mse = MSELoss()

## TRAINING_STEP
def training_step(batch_idx, train_data, args):
    
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output = model(train_data["image"])
        loss = criterion(output, train_data["dense"].long())
        
    if args.use_scaler:
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss.backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    #PREDICTION
    pred = trans.post_pred(output).long()
    
    #METRICS
    #calculate metrics for last batch every epoch, and every 10th epoch for the whole data loader  
    if (epoch+1) %args.metrics_interval == 0 or (batch_idx+1) == len(train_loader):
        #segmentation 
        if (batch_idx+1) == len(train_loader):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for func in seg_metrics:
                    func(y_pred=pred, y=train_data['dense'].long())
        else:
            for func in seg_metrics[:-1]:
                func(y_pred=pred, y=train_data['dense'].long())
        
        if (epoch+1)%args.metrics_interval==0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg_metrics[2](y_pred=pred, y=train_data['dense'].long())
        
        if (batch_idx+1) == len(train_loader):
            # calculate HD95 only for last batch, because it is very long for all data
            # catch warnings for HD95 nan/inf distance
        
            seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
            train_dice_cum.append(seg_metric_results[0], count=len(train_loader))
            train_jac_cum.append(seg_metric_results[1], count=len(train_loader))
            train_hd95_cum.append(seg_metric_results[2], count=len(train_loader))

    if (batch_idx+1) % args.log_batch_interval == 0:
        print(" ", end="")
        print(f"Batch: {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}.")
    if (batch_idx+1) == len(train_loader):
        avg_loss = train_loss_cum.aggregate().mean().item()
        print(f" ** Batch: {batch_idx + 1}/{len(train_loader)} - Avg.loss: {avg_loss:.4f}.")
        if (epoch+1) % args.metrics_interval == 0:
            print(f" __Metrics__(Whole data loader):\n"
                  f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, HD95: {seg_metric_results[2]:.4f}.")
        else:
            print(f" __Metrics__(Last batch):\n"
                  f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, HD95: {seg_metric_results[2]:.4f}.")
    
    #log running average for loss
    batch_size = train_data["image"].shape[0]
    train_loss_cum.append(loss.item(), count=batch_size)
    
    # log visual results to comet.ml
    if (args.is_log_image or args.is_log_3d) and batch_idx == 9:
        if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0:
            #seg
            pred_np = pred[0].squeeze().detach().cpu().numpy()
            label_np = train_data["dense"][0].long().squeeze().detach().cpu().numpy()

            if (epoch+1) % args.log_slice_interval == 0 and args.is_log_image:
                image = train_data["image"][0].squeeze().detach().cpu().numpy()
                #seg
                #canal 0 is image, canal 1 is gcr
                image_log_out = log.log_image(pred_np, label_np, image[0])
                experiment.log_image(image_log_out, name=f'train_img_{(epoch+1):04}_{batch_idx+1:02}')
                # gcr
                image_log_out = log.log_image(pred_np, label_np, image[1])
                experiment.log_image(image_log_out, name=f'train_gcr_{(epoch+1):04}_{batch_idx+1:02}')
            if (epoch+1) % args.log_3d_scene_interval_training == 0 and args.is_log_3d:
                scene_log_out = log.log_3dscene_comp(pred_np, label_np, args.classes, scene_size=512)
                experiment.log_image(scene_log_out, name=f'train_scene_{(epoch+1):04}_{batch_idx+1:02}')

### VALIDATION STEP ###
def validation_step(batch_idx, val_data, args):
    if val_data["image"].device != device:
        val_data["image"] = val_data["image"].to(device)
        val_data["gcr"] = val_data["gcr"].to(device)
        val_data["dense"] = val_data["dense"].to(device)
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output = sliding_window_inference(val_data["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                            device=device, mode=args.inference_mode, sigma_scale=0.125, padding_mode='constant', cval=0, progress=True)
        
        val_pred = [(seg_pred > 0.5).long() for seg_pred in decollate_batch(output)]
        val_seg_label = [i for i in decollate_batch(val_data["dense"])]

        #segmentation 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for func in seg_metrics:
                func(y_pred=val_pred, y=val_seg_label)

        if (batch_idx+1) == len(val_loader):
            seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
            
            #log running average for metrics
            val_dice_cum.append(seg_metric_results[0])
            val_jac_cum.append(seg_metric_results[1])
            val_hd95_cum.append(seg_metric_results[2])
            
            print(f" Validation metrics:\n"
                  f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, HD95: {seg_metric_results[2]:.4f}.")
                
        if (args.is_log_image or args.is_log_3d) and batch_idx == 0:
            if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_validation == 0:
                pred_seg_np = val_pred[0].squeeze().detach().cpu().float().numpy()
                gt_seg_np = val_data['dense'][0].squeeze().detach().cpu().float().numpy()    
                if (epoch+1) % args.log_slice_interval == 0 and args.is_log_image:
                    image = val_data["image"][0].squeeze().detach().cpu().float().numpy()        
                    image_log_out = log.log_image(pred_seg_np, gt_seg_np, image[0])
                    experiment.log_image(image_log_out, name=f'val_img_{(epoch+1):04}_{batch_idx+1:02}')
                if (epoch+1) % args.log_3d_scene_interval_validation == 0 and args.is_log_3d:
                    scene_log_out = log.log_3dscene_comp(pred_seg_np, gt_seg_np, args.classes, scene_size=1024)
                    experiment.log_image(scene_log_out, name=f'val_scene_{(epoch+1):04}_{batch_idx+1:02}')

### TEST STEP ###
def test_step(batch_idx, test_data, args):
    if test_data["image"].device != device:
       test_data["image"] = test_data["image"].to(device)
       test_data["gcr"] = test_data["gcr"].to(device)
       test_data["dense"] = test_data["dense"].to(device)
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output = sliding_window_inference(test_data["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                            device=device, mode=args.inference_mode, sigma_scale=0.125, padding_mode='constant', cval=0, progress=True)
        test_pred = [(seg_pred > 0.5).long() for seg_pred in decollate_batch(output)]
        test_seg_label = [i for i in decollate_batch(test_data["dense"])]
        
        #segmentation 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for func in seg_metrics:
                func(y_pred=test_pred, y=test_seg_label)

        if (batch_idx+1) == len(test_loader):
            seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
      
            #log running average for metrics
            test_dice_cum.append(seg_metric_results[0])
            test_jac_cum.append(seg_metric_results[1])
            test_hd95_cum.append(seg_metric_results[2])
           
            print(f" Test metrics:\n"
                  f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, HD95: {seg_metric_results[2]:.4f}.")
                
        if (args.is_log_image or args.is_log_3d) and batch_idx == 0:
            if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_test == 0:
                #seg
                pred_seg_np = test_pred[0].squeeze().detach().cpu().float().numpy()
                gt_seg_np = test_data['dense'][0].squeeze().detach().cpu().float().numpy()    
                if (epoch+1) % args.log_slice_interval == 0 and args.is_log_image:
                    image = test_data["image"][0].squeeze().detach().cpu().float().numpy()        
                    image_log_out = log.log_image(pred_seg_np, gt_seg_np, image[0])
                    experiment.log_image(image_log_out, name=f'test_img_{(epoch+1):04}_{batch_idx+1:02}')
                if (epoch+1) % args.log_3d_scene_interval_test == 0 and args.is_log_3d:
                    scene_log_out = log.log_3dscene_comp(pred_seg_np, gt_seg_np, args.classes, scene_size=1024)
                    experiment.log_image(scene_log_out, name=f'test_scene_{(epoch+1):04}_{batch_idx+1:02}')


### CROSS VALIDATION LOOP ###
print("--------------------")
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    #PREDEFINED SPLITS - overwrite k-fold indices
    if args.use_json_splits:
        train_ids = train_split_ids
        val_ids = val_split_ids
        test_ids = test_split_ids
        print(f"Using data split based on splits.json: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test.")
        if fold == 1:
            break
    else:
        test_ids = None
        print(f"Using kfold data split: {len(train_ids)} train, {len(val_ids)} val.")
        print(f"FOLD {fold}")
        print("-------------------")
        # if fold != args.split - 1:
        #     continue
        
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    if test_ids is not None:
        test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_subsampler)
    val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=val_subsampler)
    if test_ids is not None:
        test_loader = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=test_subsampler)
                        
    #MODEL INIT
    if  args.model_name == "ResUnet18":
        model = ResUNet(spatial_dims=3, in_channels=args.in_channels, out_channels=args.classes, act='relu', norm=args.norm, backbone_name='resnet18', bias=False, big_decoder=args.big_decoder)
    elif args.model_name == "ResUnet50":
        model = ResUNet(spatial_dims=3, in_channels=args.in_channels, out_channels=args.classes, act='relu', norm=args.norm, backbone_name='resnet50', bias=False, big_decoder=args.big_decoder)
    else:
        raise NotImplementedError(f"There is no implementation of: {args.model_name}")

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    
    #LOAD MODEL STATE DICTS to continue training    
    if args.finetune_training:
        def set_bn_eval(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        model_dict = torch.load(args.trained_model, map_location=device)
        model.load_state_dict(model_dict['model_state_dict'], strict=True)
        for p in model.parameters():
            p.requires_grad = False
        #set parameters for tuning - deepest layer of encoder and deepest layer of decoder    
        for p in model.encoder_blocks[-1].parameters():
            p.requires_grad = True
        for p in model.decoder_blocks[-1].parameters():
            p.requires_grad = True
        # it should be cvalled after model.train    
        # model.apply(set_bn_eval)
        print(f'Loaded model for fine tuning.')
        
        # Optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams, eps=args.adam_eps)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")
    
    # Scheduler
    if args.scheduler_name == 'annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
    elif args.scheduler_name == 'warmup':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, warmup_multiplier=0.01, t_total=args.epochs)
    elif args.scheduler_name == "warmup_restarts":
        scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_steps=args.warmup_steps, first_cycle_steps=int(args.epochs * args.first_cycle), cycle_mult=args.cycle_mult, gamma=args.scheduler_gamma, max_lr=args.lr, min_lr=1e-6) 
    elif args.scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=50, threshold=1e-3, min_lr=1e-6, verbose=False)

    #METRICS
    reduction='mean_batch'
    include_background=True
    seg_metrics = [
            DiceMetric(include_background=include_background, reduction=reduction),
            MeanIoU(include_background=include_background, reduction=reduction),
            HausdorffDistanceMetric(include_background=include_background, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=True, reduction=reduction)
            ]

    treshold_value = 1e-2
    treshold = nn.Threshold(treshold_value, 0)

    #RUNNING_AVERAGES
    #training loss
    train_loss_cum = CumulativeAverage()
    training_loss_cms = [train_loss_cum]
    #training metrics
    train_dice_cum = CumulativeAverage()
    train_jac_cum = CumulativeAverage()
    train_hd95_cum = CumulativeAverage()
    training_metrics_cms = [train_dice_cum, train_jac_cum, train_hd95_cum]
    #validation metrics
    val_dice_cum = CumulativeAverage()
    val_jac_cum = CumulativeAverage()
    val_hd95_cum = CumulativeAverage()

    val_metrics_cms = [val_dice_cum, val_jac_cum, val_hd95_cum]
    #test metrics
    if test_ids is not None:
        test_dice_cum = CumulativeAverage()
        test_jac_cum = CumulativeAverage()   
        test_hd95_cum = CumulativeAverage()
        test_metrics_cms = [test_dice_cum, test_jac_cum, test_hd95_cum]
    
    with experiment.train():

        best_dice_score = 0.0
        best_hd_score = np.inf
        best_dice_val_score = 0.0
        best_hd_val_score = np.inf
        best_dice_test_score = 0.0
        accum_iter = args.gradient_accumulation
        args.validation_interval = 30
        
        for epoch in range(args.start_epoch, args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            
            model.train()
            model.apply(set_bn_eval)
            epoch_time=0.0
            if epoch > 150:
                args.validation_interval = 15
            
            # finetuning update
            if (epoch+1) == args.unfreeze2_epoch:
                for p in model.encoder_blocks[-2].parameters():
                    p.requires_grad = True
                for p in model.decoder_blocks[-2].parameters():
                    p.requires_grad = True
            
            # finetuning update 2
            if (epoch+1) == args.unfreeze3_epoch:
                for p in model.encoder_blocks[-3].parameters():
                    p.requires_grad = True
                for p in model.decoder_blocks[-3].parameters():
                    p.requires_grad = True
            
            for batch_idx, train_data in enumerate(train_loader):
                training_step(batch_idx, train_data, args)   
            epoch_time=time.time() - start_time_epoch

            #RESET METRICS for training
            _ = [func.reset() for func in seg_metrics]
            
            #VALIDATION & TEST

            model.eval()
            with torch.no_grad():
                #validation dataset
                if (epoch+1) % args.validation_interval == 0:
                    print("Starting validating...")
                    start_time_validation = time.time()
                    for batch_idx, val_data in enumerate(val_loader):
                        validation_step(batch_idx, val_data, args)
                    val_time=time.time() - start_time_validation
                    print( f"Validation time: {val_time:.2f}s")

                #RESET METRICS for validation
                _ = [func.reset() for func in seg_metrics]

                #test dataset
                if test_ids is not None:
                    if (epoch+1) % args.test_interval == 0:
                        print("Starting testing...")
                        start_time_testing = time.time()
                        for batch_idx, test_data in enumerate(test_loader):
                            test_step(batch_idx, test_data, args)
                        val_time=time.time() - start_time_testing
                        print( f"Testing time: {val_time:.2f}s")
                    
                    #RESET METRICS for test
                    _ = [func.reset() for func in seg_metrics]

                #AGGREGATE RUNNING AVERAGES
                train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
                train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
                val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]
                if test_ids is not None:
                    test_metrics_agg = [cum.aggregate() for cum in test_metrics_cms]
                
                #reset running averages
                _ = [cum.reset() for cum in training_loss_cms]
                _ = [cum.reset() for cum in training_metrics_cms]
                _ = [cum.reset() for cum in val_metrics_cms]
                if test_ids is not None:
                    _ = [cum.reset() for cum in test_metrics_cms]

                #LOG METRICS TO COMET
                if args.scheduler_name == "plateau":
                    scheduler.step(train_loss_agg[0].item())
                    experiment.log_metric("lr_rate", scheduler._last_lr[0], epoch=epoch)
                elif args.scheduler_name == "warmup_restarts":
                    scheduler.step()
                    experiment.log_metric("lr_rate", scheduler.get_lr(), epoch=epoch)
                else:
                    scheduler.step()
                    experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
                    
                experiment.log_current_epoch(epoch)
                #loss
                experiment.log_metric("train_loss", train_loss_agg[0], epoch=epoch)
                #train metrics
                experiment.log_metric("train_dice", train_metrics_agg[0], epoch=epoch)
                experiment.log_metric("train_jac", train_metrics_agg[1], epoch=epoch)
                experiment.log_metric("train_hd95", train_metrics_agg[2], epoch=epoch)
                #val metrics
                experiment.log_metric("val_dice", val_metrics_agg[0], epoch=epoch)
                experiment.log_metric("val_jac", val_metrics_agg[1], epoch=epoch)
                experiment.log_metric("val_hd95", val_metrics_agg[2], epoch=epoch)
                #test metrics
                if test_ids is not None:
                    experiment.log_metric("test_dice", test_metrics_agg[0], epoch=epoch)
                    experiment.log_metric("test_jac", test_metrics_agg[1], epoch=epoch)
                    experiment.log_metric("test_hd95", test_metrics_agg[2], epoch=epoch)

                # CHECKPOINTS SAVE
                model_folder_name = "ResUNet"
                data_type = "SYNTH" if args.use_synthetic else "DENSE_GT"
                directory = os.path.join(args.checkpoint_dir, model_folder_name, f"HQ_{data_type}2_{args.model_name}_FP32_2restarts_nearest_shuffle")
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save best TRAIN model - dice
                if best_dice_score < train_metrics_agg[0] and (epoch+1) %args.metrics_interval == 0:
                    save_path = f"{directory}/model-{args.model_name}-fold-{fold}_current_best_dice_train.pt"
                    #save optimiser and scheduler to allow continue training
                    torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler' : scheduler.state_dict(),
                            'model_val_dice': train_metrics_agg[0],
                            'model_val_jac': train_metrics_agg[1]
                            }, save_path)
                    best_dice_score = train_metrics_agg[0]
                    print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                
                 # save best TRAIN model - HD95
                if best_hd_score > train_metrics_agg[2] and (epoch+1) %args.metrics_interval == 0 and train_metrics_agg[2] != 0:
                    save_path = f"{directory}/model-{args.model_name}-fold-{fold}_current_best_hd95_train.pt"
                    #save optimiser and scheduler to allow continue training
                    torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler' : scheduler.state_dict(),
                            'model_val_dice': train_metrics_agg[0],
                            'model_val_jac': train_metrics_agg[1]
                            }, save_path)
                    best_hd_score = train_metrics_agg[2]
                    print(f"Current best train hd95 score {best_hd_score:.2f}. Model saved!")
                                
                # save best VALIDATION score - dice
                if best_dice_val_score < val_metrics_agg[0]:
                    save_path = f"{directory}/model-{args.model_name}-fold-{fold}_current_best_val.pt"
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_jac': val_metrics_agg[1],
                        'model_val_hd95': val_metrics_agg[2], 
                        }, save_path)
                    best_dice_val_score = val_metrics_agg[0]
                    print(f"Current best validation dice score {best_dice_val_score:.4f} - hd95: {val_metrics_agg[2]:.2f}. Model saved!")
                
                # save best VALIDATION score - HD95
                if best_hd_val_score > val_metrics_agg[2] and val_metrics_agg[2] != 0:
                    save_path = f"{directory}/model-{args.model_name}-fold-{fold}_current_best_hd95.pt"
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_jac': val_metrics_agg[1],
                        'model_val_hd95': val_metrics_agg[2] 
                        }, save_path)
                    best_hd_val_score = val_metrics_agg[2]
                    print(f"Current best validation h95 score {best_hd_val_score:.2f} - dice: {val_metrics_agg[0]:.4f}. Model saved!")
                
                # save best TEST score
                if test_ids is not None:
                    if best_dice_test_score < test_metrics_agg[0]:
                        save_path = f"{directory}/model-{args.model_name}-fold-{fold}_current_best_test.pt"
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_val_dice': test_metrics_agg[0],
                            'model_val_jac': test_metrics_agg[1] 
                            }, save_path)
                        best_dice_test_score = test_metrics_agg[0]
                        print(f"Current best test dice score {best_dice_test_score:.4f}. Model saved!")

                #save based on SAVE INTERVAL
                if epoch % args.save_interval == 0 and epoch != 0:
                    save_path = f"{directory}/model-{args.model_name}-fold-{fold}_train_{train_metrics_agg[0]:.4f}_{train_metrics_agg[2]:.2f}_epoch_{(epoch+1):04}.pt"
                    #save based on optimiser save interval
                    if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler' : scheduler.state_dict(),
                            'model_train_dice': train_metrics_agg[0],
                            'model_train_jac': train_metrics_agg[1],
                            'model_val_dice': val_metrics_agg[0],
                            'model_val_jac': val_metrics_agg[1]
                            }, save_path)
                        print("Saved optimizer and scheduler state dictionaries.")
                    else:
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_train_dice': train_metrics_agg[0],
                            'model_train_jac': train_metrics_agg[1],
                            'model_val_dice': val_metrics_agg[0],
                            'model_val_jac': val_metrics_agg[1]
                            }, save_path)
                    print(f"Interval model saved! - train_dice: {train_metrics_agg[0]:.4f}, train_hd95: {train_metrics_agg[2]:.2f}.")

            print(f"Epoch: {epoch+1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
        print(f"Training finished!")
