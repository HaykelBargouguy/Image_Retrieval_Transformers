import os
import numpy as np
from torchvision import datasets
from .base_dataset import BaseDataset


class Cub200Dataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(Cub200Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"training", "test"}

    def set_paths_and_labels(self):

        self.split_path = self.split+"_set"

        dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'training_set'))

        
        paths = np.array([a for (a, b) in dataset.imgs])
        
        labels = np.array([b for (a, b) in dataset.imgs])
        sorted_lb = list(sorted(set(labels)))
       
        set_labels = set(sorted_lb)
        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)
                


