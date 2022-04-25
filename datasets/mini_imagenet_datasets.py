import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class NoisyMiniImagenet(Dataset):
    def __init__(self, data_root, train=True, transform=None, target_transform=None):
        """
        The noisy mini imagenet dataset.  root dir must be organized as

        root/
         - mini_imagenet/
          - clean_samp/
          - val/
          - noisy/

        :param data_root: the root dir
        :param train: True to load training data, false to load validation data
        :param transform: transform to be applied to images
        :param target_transform: transform to be applied to labels
        """

        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []
        self.paths = []

        if train:
            # load noisy data
            for imdir in os.listdir(os.path.join(self.data_root, "noisy")):
                # read label
                with open(os.path.join(self.data_root, 'noisy/', imdir, 'label.txt')) as f:
                    label = int(f.readlines()[0])
                # create img path
                path = os.path.join(self.data_root, 'noisy/', imdir, f'{imdir}.jpg')
                self.labels.append(label)
                self.paths.append(path)
            # load clean data
            for imdir in os.listdir(os.path.join(self.data_root, "clean_samp")):
                # read label
                with open(os.path.join(self.data_root, 'clean_samp/', imdir, 'label.txt')) as f:
                    label = int(f.readlines()[0])
                # create img path
                path = os.path.join(self.data_root, 'clean_samp/', imdir, f'{imdir}.jpg')
                self.labels.append(label)
                self.paths.append(path)
        else:
            # load val data
            for imdir in os.listdir(os.path.join(self.data_root, "val")):
                # read label
                with open(os.path.join(self.data_root, 'val/', imdir, 'label.txt')) as f:
                    label = int(f.readlines()[0])
                # create img path
                path = os.path.join(self.data_root, 'val/', imdir, f'{imdir}.jpg')
                self.labels.append(label)
                self.paths.append(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        path, label = self.paths[ind], self.labels[ind]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

