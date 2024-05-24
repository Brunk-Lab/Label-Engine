import cv2
import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat

from engine.core.data.base import ISDataset
from engine.core.data.sample import DSample


class ShanghaiTechDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        split='train',
        part='A',
        **kwargs
    ) -> None:
        super(ShanghaiTechDataset, self).__init__(**kwargs)
        self.root = Path(dataset_path)

        self._images_path = self.root / f'part_{part}' / f'{split}_data' / 'images'
        self._coords_path = self.root / f'part_{part}' / f'{split}_data' / 'gt'

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        coords_path = str(self._coords_path / f'GT_{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # generate mask based on coordinates
        coords = loadmat(coords_path)
        coords = len(np.load(coords_path))
        mask = None
        

        return DSample(image, mask, objects_ids=[1], coords=coords, sample_id=index)
