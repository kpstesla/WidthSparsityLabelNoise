"""
Filename: webvision_datasets.py
Author: Kyle Whitecross
Description: Contains pytorch datasets for webvision and imagenet respectively.
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image


class ImagenetDataset(Dataset):
    def __init__(self, data_root, num_classes=50, train=True, transform=None, target_transform=None):
        """
        Imagenet dataset.  Root dir must be organized as

        root/
         - synsets.txt
         - train/
         - val/

        where train and val contain folders corresponding to each synset in synsets with each folder containing the
        respective datapoints.

        :param data_root: root directory for data
        :param num_classes: number of classes to load
        :param train: True to get training data.  False for validation data.
        :param transform: function to apply to each X.
        :param target_transform: function to apply to each y.
        """

        self.data_root = data_root
        self.img_folder = os.path.join(data_root, 'train/' if train else 'val/')
        self.num_classes = num_classes
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.img_paths = []
        self.labels = []

        with open(os.path.join(self.data_root, 'synsets.txt')) as f:
            lines = f.readlines()[:num_classes]
            for i, line in enumerate(lines):
                synset = line.split()[0]
                class_path = os.path.join(self.img_folder, synset)
                imgs = os.listdir(class_path)
                for img in imgs:
                    self.img_paths.append(os.path.join(class_path, img))
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        x = Image.open(self.img_paths[ind]).convert('RGB')
        label = self.labels[ind]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return x, label
