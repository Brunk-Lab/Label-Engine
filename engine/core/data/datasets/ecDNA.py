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
        **kwargs
    ) -> None:
        super(ecDNADataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        

    def get_sample(self, index) -> DSample:
        pass
