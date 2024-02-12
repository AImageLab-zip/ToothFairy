

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from glob import glob
from torch.nn import MSELoss
import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data_function import MedData_val,MedData_finetune
import monai
from torch.utils.data import DataLoader
import configs_loader as cfg
from os.path import join
import torchio as tio
import sys
from tqdm import tqdm
import json
import SimpleITK as sitk
from loss_function import DiceLoss
from torch.optim.lr_scheduler import StepLR
from functools import reduce
def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a['patch_size'] 
def compute_dice(pred, gt):
    intersection = torch.sum(pred * gt)
    sum_pred = torch.sum(pred)
    sum_gt = torch.sum(gt)
    dice = (2.0 * intersection + 1e-5) / (sum_pred + sum_gt + 1e-5)
    return dice

def transform(patchsize, query_points, occupancy,location):
    patch_size = torch.tensor(patchsize).unsqueeze(0)

    masks = [
        query_points[:,:,0] >=location[:,0],
        query_points[:,:,0] < location[:,3],
        query_points[:,:,1] >=location[:,1],
        query_points[:,:,1] < location[:,4],
        query_points[:,:,2] >=location[:,2],
        query_points[:,:,2] < location[:,5]
    ]
    
    mask  = reduce(np.logical_and , masks)

    query_points = query_points[mask]
    occupancy = occupancy[mask]
    if len(query_points.shape) < 3:
        query_points = query_points.unsqueeze(0)
        occupancy = occupancy.unsqueeze(0)
    query_points[:,:,0], query_points[:,:,1], query_points[:,:,2] = query_points[:,:,0]- location[:,0], query_points[:,:,1] -location[:,1], query_points[:,:,2] - location[:,2]
    centers = patch_size/2
    query_points -= centers
    query_points /= patch_size
    temp = query_points.clone()
    query_points[:,:,0] , query_points[:,:,2]  = temp[:,:,2],  temp[:,:,0]
    query_points = query_points*2
    return query_points, occupancy
    


    

if __name__ == "__main__":
    cfg = cfg.get_config()
    exp_path = join( os.path.dirname(sys.argv[0]) , cfg.exp_name) ## experiment path
    processed_data =  './processed_data' ## processed data path
    patch_size = load_json(cfg.plan_path)
    patch_size= (128,128,128)
    log_file = join(exp_path,'log.txt')
    os.makedirs(exp_path , exist_ok=True)
    os.makedirs(join(exp_path, 'checkpoints'),exist_ok=True)
    os.makedirs(join(exp_path, 'logs'), exist_ok= True)

    patch_overlap = 32,32,32






    train_dataset = MedData_trainImplictSeg(join( processed_data,'imagesTr' ), join(processed_data , 'occupancy'), patch_size)

    train_loader = DataLoader(train_dataset.queue_dataset, batch_size = 1,shuffle = True, drop_last = True, num_workers=0)

    val_dataset = MedData_val(join( processed_data,'imagesTr' ) , join( processed_data, 'labelsTr' ))


                                          
      
    print(f'Begin Training {cfg.network}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from test_UNet import UNet3D
    model = UNet3D(in_channels= 1, out_channels= 2, init_features=32).to(device)
        



    CE, DICE =torch.nn.CrossEntropyLoss() , DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    scheduler = StepLR(optimizer, step_size = 5, gamma= 0.98)
    start_epoch = 1
    if cfg.checkpoint is not None:
        ckpt = torch.load(join(exp_path,'checkpoints/',cfg.checkpoint))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        with open(log_file,'a+') as f:
            f.write(f"load checkpoint from {join(exp_path,'checkpoints/',cfg.checkpoint)} \n")
            f.close
    val_interval = 5
    best_metric =0
    best_metric_epoch = -1
    epoch_loss_values = list()
    epochs = 200

    writer = SummaryWriter(join(exp_path,'./logs/'))
    for epoch in range(start_epoch , epochs):
        print("-" * 10)
        print(f"epoch {epoch}/{epochs}")
        model.train()
        
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            query_points =[]
            occupancy = []
            images, points, occupancy,location = batch_data['image']['data'].to(device), batch_data['points'],batch_data['occupancy'], batch_data['location'] 

            query_points , occupancy = transform(patch_size , points, occupancy,location)
            query_points , occupancy = query_points.to(device), occupancy.to(device)
            optimizer.zero_grad()
            outputs = model(images, query_points)
            ce_loss = CE(outputs, occupancy.long().unsqueeze(-1).unsqueeze(-1))
            dice_loss = DICE(outputs, occupancy.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            loss = ce_loss + dice_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_loader) 
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        print('end')
        with open(log_file,'a+') as f:
            f.write(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, lr : {optimizer.param_groups[0]['lr']} \n")
            f.close


        scheduler.step()
        if epoch  % val_interval == 0:
            os.makedirs('./checkpoints', exist_ok= True)
            torch.save({ 'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, join( exp_path, f"checkpoints/checkpoint-{epoch}.pth"))
            
            model.eval()
            metric_sum = 0

            for subj in val_dataset.subjects:
                
                grid_sampler = tio.inference.GridSampler(
                    subj,
                    patch_size,
                    patch_overlap,
                    padding_mode= 'edge'
                )
                patch_loader = torch.utils.data.DataLoader(grid_sampler , batch_size = 1, num_workers = 2)
                aggregator = tio.inference.GridAggregator(grid_sampler)
                with torch.no_grad():
                    for patches_batch in tqdm(patch_loader):
                        images= patches_batch['image']['data'].type(torch.FloatTensor).to(device)
                        locations = patches_batch[tio.LOCATION]
                        outputs = model(images)
                        aggregator.add_batch(outputs,locations)
                    output_tensor = aggregator.get_output_tensor()
                gt = subj['label']['data']
                output_tensor = torch.softmax(output_tensor , 0)
                output_tensor = output_tensor.argmax(dim = 0)


                metric_sum += compute_dice(output_tensor, gt.squeeze())
                torch.cuda.empty_cache()

            metric = metric_sum / len(val_dataset.subjects)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                with open(log_file,'a+') as f:
                    f.write("current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {} \n".format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                    f.close
            writer.add_scalar("val_mean_dice", metric, epoch + 1)


    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()