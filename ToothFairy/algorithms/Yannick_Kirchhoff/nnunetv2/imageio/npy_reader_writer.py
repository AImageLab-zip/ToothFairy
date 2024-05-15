#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class NumpyIO(BaseReaderWriter):
    supported_file_endings = [
        '.npy',
        '.npz'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]], key="data") -> Tuple[np.ndarray, dict]:
        images = []
        for f in image_fnames:
            if f.endswith(".npy"):
                image = np.load(f)
            elif f.endswith(".npz"):
                with np.load(f) as file:
                    image = file[key]
            assert len(image.shape) == 3, 'only 3d images are supported by NumpyIO'
            images.append(image[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'spacing': [1, 1, 1]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,), "seg")

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        seg = seg.astype(np.uint8)
        if output_fname.endswith(".npy"):
            np.save(output_fname, seg)
        elif output_fname.endswith(".npz"):
            np.savez(output_fname, seg=seg)