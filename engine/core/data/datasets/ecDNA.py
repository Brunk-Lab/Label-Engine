import cv2
import numpy as np
from pathlib import Path

from engine.core.data.base import ISDataset
from engine.core.data.sample import DSample


class ecDNADataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        split='train',
        celline=None,
        **kwargs
    ) -> None:
        super(ecDNADataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        
        self._images_path = self.dataset_path / 'images'
        self._masks_path = self.dataset_path / 'masks'
        self._coords_path = self.dataset_path / 'coords'

        self.split = split
        assert split in ('train', 'val', 'test')
        if split == 'train':
            imlist_file = self.dataset_path / 'datasets' / 'train_0422_2024.txt'
        elif split == 'val':
            imlist_file = self.dataset_path / 'datasets' / 'val_0422_2024.txt'
        else:
            imlist_file = self.dataset_path / 'datasets' / 'test_0422_2024.txt'

        self.dataset_samples = []
        f = open(imlist_file, 'r')
        for line in f:
            im_name = line.split('.')[0]
            if celline is not None:
                if not im_name.startswith(celline):
                    continue

            self.dataset_samples.append(im_name)

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.png')
        mask_path = str(self._masks_path / f'{image_name}.png')
        coords_path = str(self._coords_path / f'{image_name}.npy')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # generate mask based on coordinates
        # TO BE DONE
        mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        mask[mask == 128] = 0
        mask[mask > 128] = 1
        
        coords = len(np.load(coords_path))

        return DSample(image, mask, objects_ids=[1], coords=coords, sample_id=index)
