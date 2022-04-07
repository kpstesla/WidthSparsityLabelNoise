import torch
from torch.utils.data import Dataset
import numpy as np


class MislabelledDataset(Dataset):
    def __init__(self, dataset, mislabel_ratio=0, num_classes=10, cache=True, transform=None, target_transform=None,
                 asym=False):
        """
        Mislabelled Dataset wrapper.  Returns items as (x, fake_label, real_label, index).

        :param dataset: Dataset to wrap.
        :param mislabel_ratio: Noise to add to dataset.
        :param num_classes: Number of distinct labels in the dataset
        :param cache: True to cache the dataset in memory (as a python list)
        """
        self.dataset = dataset
        self.mislabel_ratio = mislabel_ratio
        self.num_classes = num_classes
        self.cache = cache
        self.fake_labels = []
        self.real_labels = []
        self.x_cache = []
        self.transform = transform
        self.target_transform = target_transform

        # permutation list for asymmetric noise
        class_list = np.arange(num_classes)
        permu_list = np.random.permutation(class_list)
        while True:
            if np.any(class_list == permu_list):
                permu_list = np.random.permutation(class_list)
            else:
                break

        # get labels, potentially cache x, and generate fake labels
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            if self.cache:
                self.x_cache.append(x)
            if np.random.random() < float(self.mislabel_ratio):
                if asym:
                    self.fake_labels.append(permu_list[y])
                else:
                    self.fake_labels.append(np.random.choice(num_classes))
            else:
                self.fake_labels.append(y)
            self.real_labels.append(y)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        # get x from cache
        if self.cache:
            x = self.x_cache[ind]
        else:
            x, _ = self.dataset[ind]
        # apply transforms
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y_fake = self.target_transform(self.fake_labels[ind])
            y_real = self.target_transform(self.real_labels[ind])
        else:
            y_fake = self.fake_labels[ind]
            y_real = self.real_labels[ind]
        return x, y_fake, y_real, ind


