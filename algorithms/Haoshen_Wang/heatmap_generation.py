import torchio as tio
import os
from scipy.spatial import cKDTree as KDTree
import numpy as np
import torch
def heatmap_generation(path):
    file_num = os.path.basename(path).split('_')[0]
    target_path = 'nerve-002.nii.gz'
    kernel_size = 7
    sigma = kernel_size/7
    half_size = int((kernel_size - 1)/2)
    img = tio.LabelMap(path)
    skeleton = img[tio.DATA].squeeze().numpy()

    skeleton_points = np.where(skeleton == 1)
    skeleton_points = np.concatenate((skeleton_points[0][:,np.newaxis],skeleton_points[1][:,np.newaxis],skeleton_points[2][:,np.newaxis]),axis=-1)
    shape = skeleton.shape
    kdtree = KDTree(skeleton_points)

    heat_map = np.zeros_like(skeleton, dtype= np.float64)
    intensity = 200
    points_set = np.empty((0,3))
    for i in range(len(skeleton_points)):
        center = skeleton_points[i]
        x_min, y_min, z_min = max(center[0]-half_size , 0) , max(center[1]-half_size , 0) , max(center[2]-half_size , 0)
        x_max, y_max, z_max = min(center[0]+half_size , shape[0]-1), min(center[1]+half_size, shape[1]-1) , min(center[2]+half_size, shape[2]-1)
        x , y , z = np.arange(x_min, x_max+1), np.arange(y_min, y_max + 1), np.arange(z_min, z_max + 1)
        X , Y , Z = np.meshgrid( x, y , z ,indexing='ij')
        points = np.concatenate((X[:,:,:,np.newaxis],Y[:,:,:,np.newaxis],Z[:,:,:,np.newaxis]),axis = -1).reshape(-1, 3)
        points_set = np.concatenate((points_set,points), axis= 0 )
    points = np.unique(points_set , axis = 0)
    distances, _ = kdtree.query(points)
    gaussian_kernel = np.exp(-(distances)/(2*sigma**2))*intensity
    gaussian_kernel = gaussian_kernel[:,np.newaxis]
    points = np.concatenate((points , gaussian_kernel), axis=-1)

    heat_map[points[:,0].astype(int),points[:,1].astype(int), points[:,2].astype(int)] = points[:,3]

    tio.ScalarImage(tensor = torch.tensor(heat_map).unsqueeze(0), affine = img.affine).save(target_path)