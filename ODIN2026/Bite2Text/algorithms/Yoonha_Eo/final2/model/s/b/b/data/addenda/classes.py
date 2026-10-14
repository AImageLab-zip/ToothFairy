import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas
import pymeshlab
import torch
from torchtyping import TensorType
from tqdm import tqdm

import teethland
from teethland import PointTensor
from teethland.visualization import draw_point_clouds


def files(root: Path) -> Tuple[List[Path], List[Path], Path]:
    mesh_files = sorted(root.glob('**/*.obj'))
    scan_ids = [f.stem for f in mesh_files]
    _, unique_idxs = np.unique(scan_ids, return_index=True)
    mesh_files = [mesh_files[i] for i in unique_idxs]

    ann_files = sorted(root.glob('**/*.json'))
    scan_ids = [f.stem for f in ann_files]
    _, unique_idxs = np.unique(scan_ids, return_index=True)
    ann_files = [ann_files[i] for i in unique_idxs]
    
    addenda_file = root / 'addenda' / 'IOS addenda.csv'

    return mesh_files, ann_files, addenda_file


def load_addendum_dicts(addenda_file: Path) -> Dict[str, Dict[int, int]]:
    df = pandas.read_csv(addenda_file)
    df = df[df['Dict correction'].notna()]

    add_dicts = {}
    for _, row in df.iterrows():
        scan_id = row['file_name']

        add_dict = row['Dict correction']
        add_dict = [s for s in re.split(r'[\s:,\{\}]', add_dict) if s]
        add_dict = {
            int(add_dict[i]): int(add_dict[i + 1])
            for i in range(0, len(add_dict), 2)
        }

        add_dicts[scan_id] = add_dict

    return add_dicts


def load_source(
    mesh_files: List[Path],
    ann_files: List[Path],
    scan_id: str,
) -> Tuple[
    TensorType['N', 3, torch.float32],
    Dict[str, Any],
]:
    # load mesh from storage
    mesh_file = [f for f in mesh_files if f.stem == scan_id][0]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_file))
    mesh = ms.current_mesh()
    mesh.compact()
    vertices = torch.tensor(mesh.vertex_matrix()).float()

    # load class annotations from storage
    ann_file = [f for f in ann_files if f.stem == scan_id][0]
    with open(ann_file, 'rb') as f:
        ann = json.load(f)
    
    return vertices, ann


def load_addendum(
    labels: TensorType['N', torch.int64],
    add_dict: Dict[int, int],
) -> TensorType['N', torch.int64]:
    add_classes = torch.arange(49)
    for key, value in add_dict.items():
        add_classes[key] = value

    add_labels = add_classes[labels]

    return add_labels


def process_addenda(root: Path) -> List[PointTensor]:
    mesh_files, ann_files, addenda_file = files(root)

    add_dicts = load_addendum_dicts(addenda_file)

    pt_pairs = []
    for scan_id, add_dict in tqdm(add_dicts.items(), total=len(add_dicts)):
        # load original annotation
        vertices, ann = load_source(mesh_files, ann_files, scan_id)
        labels = torch.tensor(ann['labels'])

        # load addendum annotation
        add_labels = load_addendum(labels, add_dict)

        # save new annotation to storage
        ann['labels'] = add_labels.tolist()
        with open(root / 'addenda' / f'{scan_id}.json', 'w') as f:
            json.dump(ann, f)

        # process batch before visualization
        source_pt = PointTensor(
            coordinates=vertices,
            features=labels,
        )
        add_pt = PointTensor(
            coordinates=vertices,
            features=add_labels,
        )
        pt_pairs.append(miccai.stack([source_pt, add_pt]))

    return pt_pairs


if __name__ == '__main__':
    root = Path('/home/mka3dlab/Documents/3dteethseg')
    pt_pairs = process_addenda(root)

    # visualize source and addendum annotations
    for pt_pair in tqdm(pt_pairs):
        pt_pair.F %= 10
        draw_point_clouds(pt_pair)
