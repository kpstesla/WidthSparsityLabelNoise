import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


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
        if self.cache:
            print(f"Caching...")
        if self.cache or self.mislabel_ratio > 0:
            for i in tqdm(range(len(self.dataset))):
                x, y = self.dataset[i]
                if self.cache:
                    self.x_cache.append(x)
                if np.random.random() < float(self.mislabel_ratio):
                    if asym:
                        self.fake_labels.append(permu_list[y])
                    else:
                        new_target = np.random.choice(num_classes)
                        while new_target == y:
                            new_target = np.random.choice(num_classes)
                        self.fake_labels.append(new_target)
                else:
                    self.fake_labels.append(y)
                self.real_labels.append(y)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        # if we cached, then get x from cache
        if self.cache or self.mislabel_ratio > 0:
            y_real = self.real_labels[ind]
            y_fake = self.fake_labels[ind]
        if self.cache:
            x = self.x_cache[ind]
        else:
            x, y = self.dataset[ind]
            if not self.mislabel_ratio > 0:
                y_fake = y
                y_real = y

        # transforms
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y_real = self.target_transform(y_real)
            y_fake = self.target_transform(y_fake)

        # return
        return x, y_fake, y_real, ind



