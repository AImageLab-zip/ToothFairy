import json
from pathlib import Path

import numpy as np
import open3d.visualization
import pymeshlab
import open3d
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/CBCT/fusion/stls')
    verbose = False

    out_dir = root.parent / 'IOS teeth'
    out_dir.mkdir(exist_ok=True)

    ms = pymeshlab.MeshSet()
    for mesh_file in tqdm(list(root.glob('*.stl'))):
        patient_id = mesh_file.name.split('_')[0]
        
        patient_dir = out_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        seg_file = mesh_file.with_suffix('.json')

        ms.load_new_mesh(str(mesh_file))
        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        with open(seg_file, 'r') as f:
            seg_dict = json.load(f)
        labels = np.array(seg_dict['labels'])

        for label in np.unique(labels)[1:]:
            fg_mask = labels == label

            label_triangles = triangles[np.any(fg_mask[triangles], axis=-1)]

            label_vertex_idxs = np.unique(label_triangles.flatten())
            label_vertices = vertices[label_vertex_idxs]
            
            vertex_map = np.full((labels.shape[0],), -1)
            vertex_map[label_vertex_idxs] = np.arange(label_vertex_idxs.shape[0])

            label_triangles = vertex_map[label_triangles]

            mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(label_vertices),
                triangles=open3d.utility.Vector3iVector(label_triangles),
            )
            mesh.compute_vertex_normals()
            if verbose:
                open3d.visualization.draw_geometries([mesh])
            open3d.io.write_triangle_mesh(
                str(out_dir / patient_dir / f'{label}.stl'),
                mesh,
            )
