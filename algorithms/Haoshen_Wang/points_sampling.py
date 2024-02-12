import torchio as tio
import os
import open3d as o3d
import numpy as np
from os.path import join

def sampling_occupancy(self,sigma, path):
    if sigma == 0:
        sample_num = 100000
    elif sigma == 0.3:
        sample_num = 10000000
    elif sigma == 5:
        sample_num = 1000000
    elif sigma == 10:
        sample_num = 100000
    out_path = 'occupancy'
    file_name = os.path.splitext(os.path.basename(path))[0].split('-')[-1]
    input_file = path
    out_file = join(out_path,f'{file_name}_occ.npz')
    mesh = o3d.io.read_triangle_mesh(input_file)
    if sigma !=0:
        pcd = mesh.sample_points_uniformly(sample_num)
        points = np.asarray(pcd.points)
        query_points =points +  sigma*np.random.randn(sample_num, 3)
    else:
        shape = tio.LabelMap(f'processed_LABEL/segmentation-{file_name}.nii.gz').shape[1:]
        query_points = np.random.uniform(low = 0 , high = shape, size= (sample_num,3))
    query_points = o3d.core.Tensor(query_points,dtype = o3d.core.Dtype.Float32)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    occupancy = scene.compute_occupancy(query_points)
    occupancy = occupancy.numpy()
    np.savez(out_file, points=query_points.numpy(), occupancy = occupancy)
    print('Finished: {}'.format(path))