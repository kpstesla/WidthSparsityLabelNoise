from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class NoisyCaltech256(Dataset):
    def __init__(self, root_dir: str, mislabel_ratio: float = 0.0, n_test: int = 25, train=True,
                 transform=None, target_transform=None):
        self.data_root = os.path.join(root_dir, 'caltech256')
        self.transform = transform
        self.target_transform = target_transform
        self.mislabel_ratio = mislabel_ratio
        self.img_paths = []
        self.labels = []
        for cls, cls_path in os.listdir(self.data_root):
            cls_full_path = os.path.join(self.data_root, cls_path)
            if train:
                for img_path in os.listdir(cls_full_path)[n_test:]:
                    self.img_paths.append(os.path.join(cls_full_path, img_path))
                    if np.random.rand() < mislabel_ratio:
                        self.labels.append(np.random.randint(0, 257))
                    else:
                        self.labels.append(cls)
            else:
                for img_path in os.listdir(cls_full_path)[:n_test]:
                    self.img_paths.append(os.path.join(cls_full_path, img_path))
                    self.labels.append(cls)

        print(f"Noisy Caltech 256. Train: {train}, mislabel_ratio: {mislabel_ratio}, size: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        path, label = self.img_paths[item], self.labels[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
