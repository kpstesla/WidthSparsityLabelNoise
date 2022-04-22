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
    def __init__(self, data_root, num_classes=50, class_inds=None, train=True, transform=None, target_transform=None):
        """
        Imagenet dataset.  Root dir must be organized as

        root/
         - imagenet/
          - synsets.txt
          - train/
          - val/

        where train and val contain folders corresponding to each synset in synsets with each folder containing the
        respective datapoints.

        :param data_root: root directory for data
        :param num_classes: number of classes to load
        :param class_inds: indices of classes to load (same as for webvision)
        :param train: True to get training data.  False for validation data.
        :param transform: function to apply to each X.
        :param target_transform: function to apply to each y.
        """

        self.data_root = os.path.join(data_root, 'imagenet')
        self.img_folder = os.path.join(self.data_root, 'train/' if train else 'val/')
        self.num_classes = num_classes
        self.class_inds = class_inds
        self.label_mapping = None
        self.rev_label_mapping = None
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.img_paths = []
        self.labels = []

        assert(len(class_inds) == num_classes)

        if class_inds is not None:
            new_labels = np.arange(num_classes)
            self.label_mapping = {class_inds[i]: new_labels[i] for i in range(num_classes)}
            self.rev_label_mapping = {new_labels[i]: class_inds[i] for i in range(num_classes)}

        with open(os.path.join(self.data_root, 'synsets.txt')) as f:
            if class_inds is not None:
                lines = np.array(f.readlines())[self.class_inds]
            else:
                lines = f.readlines()[:num_classes]
            for i, line in enumerate(lines):
                synset = line.split()[0]
                class_path = os.path.join(self.img_folder, synset)
                imgs = os.listdir(class_path)
                for img in imgs:
                    self.img_paths.append(os.path.join(class_path, img))
                    self.labels.append(i) # this should work for class_inds I think

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


class WebvisionDataset(Dataset):
    def __init__(self, data_root, num_classes=50, train=True, include_flickr=False, class_inds=None, transform=None,
                 target_transform=None):
        """
        Webvision Dataset.  Root dir must be organized as

        root/
         - webvision
          - info/
           - val_filelist.txt
           - train_filelist_google.txt
           - train_filelist_flickr.txt
           - val_images_256/
         - google/
         - flickr/

        :param data_root: filepath of data root
        :param num_classes: number of classes to use
        :param train: true for training data, false for validation data
        :param include_flickr: true to include flickr examples, false to exclude them
        :param class_inds: if not None, then the first num_classes are used. If a list of indices, then those indices
            are selected from the webvision dataset and mapped to 0-50 in the order they are passed in.  num_classes
            must match the length of this list
        :param transform: transform to apply to images
        :param target_transform: transform to apply to labels
        """
        self.data_root = os.path.join(data_root, 'webvision')
        self.num_classes = num_classes
        self.class_inds = class_inds
        self.label_mapping = None
        self.rev_label_mapping = None
        self.transform = transform
        self.target_transform = target_transform
        self.img_paths = []
        self.labels = []

        # make sure num classes is correct
        if class_inds is not None:
            if num_classes != len(class_inds):
                raise AssertionError("num_classes not equal to length of class inds")

        # make sure label mapping is created
        if class_inds is not None:
            new_labels = np.arange(num_classes)
            self.label_mapping = {self.class_inds[i]: new_labels[i] for i in range(num_classes)}
            self.rev_label_mapping = {new_labels[i]: self.class_inds[i] for i in range(num_classes)}

        # get path class mappings
        if train:
            with open(os.path.join(self.data_root, 'info', 'train_filelist_google.txt')) as f:
                lines = f.readlines()
            if include_flickr:
                with open(os.path.join(self.data_root, 'info', 'train_filelist_flickr.txt')) as f:
                    lines += f.readlines()
        else:
            with open(os.path.join(self.data_root, 'info', 'val_filelist.txt')) as f:
                lines = f.readlines()

        # populate img_paths and labels
        for line in lines:
            s = line.split()
            path, label = s[0], int(s[1])
            # only include if label is in the first num_classes
            if label < self.num_classes or class_inds is not None and label in class_inds:
                if not train:
                    path = os.path.join('val_images_256/', path)
                path = os.path.join(self.data_root, path)

                self.img_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        path, label = self.img_paths[ind], self.labels[ind]
        img = Image.open(path).convert('RGB')

        # potentially remap classes
        if self.class_inds is not None:
            label = self.label_mapping[label]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

