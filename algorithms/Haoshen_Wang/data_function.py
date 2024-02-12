import torch
import os
import torchio as tio
from glob import glob
from preprocess import CustomTransform
from preprocess import Resample
import numpy as np
from os.path import join

class MedData_finetune(torch.utils.data.Dataset):
    def __init__(self, images_dir,  points_dir,patch_size):


        queue_length = 24
        samples_per_volume = 2
        self.images = sorted(glob(os.path.join(images_dir, "volume*.nii.gz")))

        self.images = [self.images[0]] #-15

 
        self.subjects = []
        self.query_points = []
        self.occupancy = []
        self.noise =[0, 0.3, 5, 10]
        

        for img in self.images:
            file_num = os.path.basename(img).split('.')[0].split('-')[-1]
            for noise in self.noise:
                p=np.load(join(points_dir , f'{file_num}_boundary_{str(noise)}_samples.npz'))
                self.query_points.append(p['points'])
                self.occupancy.append(p['occupancy'])
            self.query_points  = np.concatenate(self.query_points, axis= 0)
            self.occupancy = np.concatenate(self.occupancy , axis= 0 )

            subject = tio.Subject(
                image = tio.ScalarImage(img),
                points = self.query_points,
                occupancy = self.occupancy
            )
            self.subjects.append(subject)


        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)


        self.queue_dataset = tio.Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            tio.UniformSampler(patch_size),
            num_workers= 2,
        )

    


class MedData_val(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images = sorted(glob(os.path.join(images_dir, "volume*.nii.gz")))[-15:] #-15
        self.labels = sorted(glob(os.path.join(labels_dir, 'segmentation*.nii.gz')))[-15:]


        self.subjects = []
        for (img, lab) in zip(self.images, self.labels):
                subject = tio.Subject(
                    image=tio.ScalarImage(img),
                    label = tio.LabelMap(lab),
                )
                self.subjects.append(subject)
        self.val_set = tio.SubjectsDataset(self.subjects, transform=None)

