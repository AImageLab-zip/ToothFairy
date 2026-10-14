from collections import defaultdict
import json
from pathlib import Path

import pymeshlab
import numpy as np
from tqdm import tqdm

import teethland.data.transforms as T

landmark_classes = {
    'Mesial': 0,
    'Distal': 1,
    'FacialPoint': 2,
    'OuterPoint': 3,
    'InnerPoint': 4,
    'Cusp': 5,
}


def get_case_data(landmarks_file: Path):
    with open(landmarks_file, 'r') as f:
        landmarks = json.load(f)['objects']
    
    land_coords = np.array([land['coord'] for land in landmarks])
    land_classes = np.array([landmark_classes[land['class']] for land in landmarks])


    stem = landmarks_file.stem.split('__')[0]
    seg_file = next(seg_root.glob(f'**/{stem}.json'))
    with open(seg_file, 'r') as f:
        ann = json.load(f)

    labels = np.array(ann['labels'])
    instances = np.array(ann['instances'])

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(next(seg_root.glob(f'**/{stem}.obj'))))
    points = ms.current_mesh().vertex_matrix()

    data_dict = {
        'points': points,
        'labels': labels,
        'instances': instances,
        'landmark_coords': land_coords,
        'landmark_classes': land_classes,
    }
    data_dict = matching(**data_dict)
    land_instances = data_dict['landmark_instances']

    return labels, instances, land_coords, land_classes, land_instances


if __name__ == '__main__':
    seg_root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/lower_upper')
    landmarks_root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/3DTeethLand_landmarks_train')

    matching = T.MatchLandmarksAndTeeth(move = 0.04 * 17.3281)

    fdi_ratio = defaultdict(list)

    for landmarks_file in tqdm(list(landmarks_root.glob('**/*.json'))):
        labels, instances, land_coords, land_classes, land_instances = get_case_data(landmarks_file)
        fdis, min_cusp_dists, min_inter_dists = [], [], []
        for i in np.unique(land_instances):
            coords = land_coords[land_instances == i]
            classes = land_classes[land_instances == i]

            mesial_coords = coords[classes == landmark_classes['Mesial']]
            distal_coords = coords[classes == landmark_classes['Distal']]
            if mesial_coords.shape[0] == 0 or distal_coords.shape[0] == 0:
                continue

            diff = distal_coords[0] - mesial_coords[0]
            width = np.abs(diff[0])
            height = np.abs(diff[1])

            fdi = labels[instances == i][0]
            fdi_ratio[fdi].append(width / height)

            if classes.max() < 5:
                continue

            class_pairs = classes[None] == classes[:, None]
            cusps = (classes[None] == 5) | (classes[:, None] == 5)
            dists = np.linalg.norm(coords[None] - coords[:, None], axis=-1)
            dists[class_pairs] = 1e6
            dists[cusps] = 1e6
            min_inter_dists.append(dists.min())
            fdis.append(fdi)

            dists = np.linalg.norm(coords[classes == 5][None] - coords[classes == 5][:, None], axis=-1)
            if dists.shape[0] <= 1:
                min_cusp_dists.append(1e6)
                continue
            min_cusp_dists.append(dists[np.triu_indices_from(dists, 1)].min())



        if not min_cusp_dists:
            print(landmarks_file.name, 'N/A')
            continue

        print(
            landmarks_file.name,
            min(min_cusp_dists),
            fdis[np.argmin(min_cusp_dists)],
            min(min_inter_dists),
            fdis[np.argmin(min_inter_dists)],
        )
            
    k = 3
