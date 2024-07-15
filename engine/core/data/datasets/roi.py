import cv2
import numpy as np
from pathlib import Path

from engine.core.data.base import ISDataset
from engine.core.data.sample import DSample


class ROIDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        split='train',
        celline=None,
        date='0609_2024',
        **kwargs
    ) -> None:
        super(ROIDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        
        self._images_path = self.dataset_path / 'images'
        self._masks_path = self.dataset_path / 'masks'
        self._coords_path = self.dataset_path / 'coords'
        self._ROIs_path = self.dataset_path / 'ROIs'

        self.split = split
        assert split in ('train', 'val', 'test')
        if split == 'train':
            imlist_file = self.dataset_path / 'datasets' / f'train_{date}.txt'
        elif split == 'val':
            imlist_file = self.dataset_path / 'datasets' / f'val_{date}.txt'
        else:
            imlist_file = self.dataset_path / 'datasets' / f'test_{date}.txt'

        self.dataset_samples = []
        f = open(imlist_file, 'r')
        for line in f:
            im_name = line.split('.')[0]
            if celline is not None:
                if not im_name.startswith(celline):
                    continue

            self.dataset_samples.append(im_name)

    def get_sample(self, index) -> DSample:
        """
        Get a sample from index.

        Parameters:
        - index: int, the index of the sample

        Returns:
        - DSample: containing image, mask, objects_ids, and coordinates 
        """
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        coords_path = str(self._coords_path / f'{image_name}.npy')
        coords = len(np.load(coords_path))

        # mask_path = str(self._masks_path / f'{image_name}.png')
        # mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)

        roi_path = str(self._ROIs_path / f'{image_name}.png')
        roi = cv2.imread(roi_path)[:, :, 0].astype(np.int32)
        mask = roi

        return DSample(image, mask, objects_ids=[255], coords=coords, 
                       sample_id=index)