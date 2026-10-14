from functools import partial
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import pymeshlab
from tqdm import tqdm


def mesh_means(
    pre_transform: Callable[..., Dict[str, Any]],
    file: Path,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    int,
]:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file))
    mesh = ms.current_mesh()
    mesh.compact()

    vertices = mesh.vertex_matrix()
    normals = mesh.vertex_normal_matrix()
    vertex_count = mesh.vertex_number()

    data_dict = pre_transform(
        points=vertices, normals=normals,
    )
    vertices = data_dict['points']

    means = vertices.mean(axis=0)
    sq_means = (vertices ** 2).mean(axis=0)

    return means, sq_means, vertex_count


def mesh_means_stds(
    files: List[Path],
    pre_transform: Callable[..., Dict[str, Any]]=dict,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    total_means, total_sq_means = np.zeros((2, 3))
    total_count = 0

    p = Pool(cpu_count())
    means_iter = p.imap(partial(mesh_means, pre_transform), files)
    for means, sq_means, count in tqdm(means_iter, total=len(files)):
        total_count += count
        total_means += (count / total_count) * (means - total_means)
        total_sq_means += (count / total_count) * (sq_means - total_sq_means)

    total_stds = np.sqrt(total_sq_means - total_means ** 2)
    
    return total_means, total_stds


if __name__ == '__main__':
    root = '/home/mka3dlab/Documents/3dteethseg'
    files = list(Path(root).glob('**/*.obj'))
    
    means, stds = mesh_means_stds(files)
    print('Means:', means)
    print('STDs:', stds)
