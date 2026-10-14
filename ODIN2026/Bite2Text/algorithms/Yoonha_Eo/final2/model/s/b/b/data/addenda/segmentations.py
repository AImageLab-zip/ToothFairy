import json
from pathlib import Path
import re
from typing import List, Tuple

from bs4 import BeautifulSoup
import torch
from torchtyping import TensorType
from tqdm import tqdm

import teethland
from teethland import PointTensor
from teethland.data.addenda.classes import load_source
from teethland.visualization import draw_point_clouds


def load_addendum(
    add_file: Path,
) -> Tuple[
    TensorType['N', 3, torch.float32],
    List[str],
    TensorType['N', torch.int64],
]:
    with open(add_file, 'r') as f:
        markup = f.read()

    soup = BeautifulSoup(markup, 'lxml')

    lines = soup.regions.get_text(strip=True).split('\n')
    fdi_labels = [l.split(',')[1] for l in lines]

    lines = soup.v.get_text(strip=True).split('\n')
    data = torch.tensor([[float(s) for s in l.split(',')] for l in lines])
    vertices = data[:, 1:-1]
    label_idxs = data[:, -1].long()

    return vertices, fdi_labels, label_idxs


def project_labels(
    source_pt: PointTensor,
    add_pt: PointTensor,
    fdi_labels: List[str],
) -> Tuple[
    TensorType['N', torch.int64],
    TensorType['N', torch.int64],
]:
    # mesh that incorporates the addendum
    neighbor_idxs, _ = source_pt.neighbors(add_pt, k=1)
    neighbor_idxs = neighbor_idxs.squeeze(-1)
    labels, instance_idxs = source_pt.F.clone().T
    for i, label in enumerate(fdi_labels):
        if re.match(r'^\d\d$', label) is not None:
            label = int(label)
            instance_idx = instance_idxs.amax() + 1

            # remove old annotation
            label_mask = labels == label
            if torch.any(label_mask):
                instance_idx = instance_idxs[label_mask][0]
                labels[label_mask] = 0
                instance_idxs[label_mask] = 0
        elif re.match(r'^\d\d\-\d$', label) is not None:
            label = int(label[:2])
            instance_idx = instance_idxs.amax() + 1 
        else:
            raise ValueError(f'Unknown FDI label: {label}')
        
        labels[neighbor_idxs[add_pt.F == i]] = label
        instance_idxs[neighbor_idxs[add_pt.F == i]] = instance_idx

    return labels, instance_idxs


def process_addenda(root: Path) -> List[PointTensor]:
    mesh_files = sorted(root.glob('**/*.obj'))
    ann_files = sorted(root.glob('**/*.json'))[-len(mesh_files):]
    add_files = list(root.glob('**/*.mas'))

    pt_pairs = []
    for add_file in tqdm(add_files):
        scan_id = add_file.stem

        # load original annotated mesh as PointTensor
        vertices, ann = load_source(mesh_files, ann_files, scan_id)
        labels = torch.tensor(ann['labels'])
        instance_idxs = torch.tensor(ann['instances'])

        source_pt = PointTensor(
            coordinates=vertices.cuda(),
            features=torch.column_stack((labels, instance_idxs)).cuda(),
        )

        # load addendum annotation mesh as PointTensor
        add_vertices, fdi_labels, label_idxs = load_addendum(add_file)

        add_pt = PointTensor(
            coordinates=add_vertices.cuda(),
            features=label_idxs.cuda(),
        )

        # project addendum annotation to source mesh
        add_labels, add_instance_idxs = project_labels(
            source_pt, add_pt, fdi_labels,
        )

        # save new JSON file
        ann['labels'] = add_labels.tolist()
        ann['instances'] = add_instance_idxs.tolist()
        with open(root / 'addenda' / f'{scan_id}.json', 'w') as f:
            json.dump(ann, f)

        # save PointTensors for visualization
        source_pt.F = labels.cuda()
        add_pt = source_pt.new_tensor(features=add_labels)
        pt_pairs.append(miccai.stack([source_pt, add_pt]))

    return pt_pairs


if __name__ == '__main__':
    root = Path('/home/mka3dlab/Documents/3dteethseg')
    pt_pairs = process_addenda(root)

    # visualize source and addendum point clouds
    for pt_pair in tqdm(pt_pairs):
        draw_point_clouds(pt_pair)
