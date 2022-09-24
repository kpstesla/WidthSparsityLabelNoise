from argparse import ArgumentParser
import sys
import yaml
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from datasets import MislabelledDataset, WebvisionDataset, ImagenetDataset, NoisyMiniImagenet, StanfordCarsRed80, \
    NoisyCaltech256
from torch.utils.data import Subset
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance


DEFAULT_CONFIG = "exps/template.yaml"


def parse_args_with_config():
    # try and parse config first
    parser = ArgumentParser(description="Dynamic config based argument parser")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    known_args, _ = parser.parse_known_args()

    # open config
    f = open(known_args.config)
    config = yaml.safe_load(f)
    f.close()

    # add config options to arg parser
    parser_config = ArgumentParser(description="Dynamic config based argument parser")
    parser_config.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    for key, val in config.items():
        if type(val) is str:
            val = val.lower()
        if type(val) is bool:
            parser_config.add_argument(f"--{key.lower()}", action='store_true', default=val)
            parser_config.add_argument(f"--disable_{key.lower()}", action="store_true", default=False)
        else:
            parser_config.add_argument(f"--{key.lower()}", default=val, type=type(val))

    # parse arguments with new arg parser
    args = parser_config.parse_args()

    # use 'disable_' arguments to optionally disable boolean arguments
    for key, val in args.__dict__.items():
        if 'disable_' in key:
            suffix = key.split('disable_')[-1]
            if args.__dict__[suffix] and val:
                args.__dict__[suffix] = False
    return args


def load_datasets(args):
    """
    Loads and sets up the 2 or 3 datasets.

    :param args: commandline/config arguments
    :return: num_classes, train_dataset, val_dataset, val_dataset_2
    """

    # set up data augmentations and normalizations
    cifar_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize(args.cifar_img_size)
    ])
    cifar_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    im_web_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    nmi_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize(299)
    ])
    web_train_aug = transforms.Compose([
        transforms.RandomCrop(227),
        transforms.Resize(args.webvision_img_size),
        transforms.RandomHorizontalFlip()
    ])
    web_test = transforms.Compose([
        transforms.CenterCrop(227),
        transforms.Resize(args.webvision_img_size),
    ])
    im_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.Resize(args.webvision_img_size),
    ])


    # caltech transforms.  Copied from
    # https://github.com/TropComplique/image-classification-caltech-256/blob/master/training_utils/data_utils.py

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }

    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    # train data augmentation on the fly
    caltech_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # validation data is already resized
    caltech_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # select dataset to load
    val_2_dataset = None
    if args.dataset.lower() == "cifar10":
        train_data = CIFAR10(args.data_root, train=True, transform=cifar_normalize, download=True)
        val_data = CIFAR10(args.data_root, train=False, transform=cifar_normalize, download=True)
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
        print("Loading CIFAR10 training set...")
        train_dataset = MislabelledDataset(train_data, mislabel_ratio=args.mislabel_ratio, num_classes=10,
                                           cache=args.cache, transform=cifar_aug, asym=args.asym)
        print("Loading CIFAR10 validation set...")
        val_dataset = MislabelledDataset(val_data, mislabel_ratio=0, num_classes=10, cache=args.cache)
        num_classes = 10
    elif args.dataset.lower() == "cifar100":
        train_data = CIFAR100(args.data_root, train=True, transform=cifar_normalize, download=True)
        val_data = CIFAR100(args.data_root, train=False, transform=cifar_normalize, download=True)
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
        print("Loading CIFAR100 training set...")
        train_dataset = MislabelledDataset(train_data, mislabel_ratio=args.mislabel_ratio, num_classes=100,
                                           cache=args.cache, transform=cifar_aug, asym=args.asym)
        print("Loading CIFAR100 validation set...")
        val_dataset = MislabelledDataset(val_data, mislabel_ratio=0, num_classes=100, cache=args.cache)
        num_classes = 100
    elif args.dataset.lower() == "webvision":
        num_classes = 50
        inds = None
        if args.webvision_custom_inds:
            print("Loading custom indices")
            inds = np.load(args.custom_inds_path)
            num_classes = len(inds)
        train_data = WebvisionDataset(args.data_root, num_classes, train=True, class_inds=inds,
                                      transform=im_web_normalize)
        val_data = WebvisionDataset(args.data_root, num_classes, train=False, class_inds=inds,
                                    transform=transforms.Compose([im_web_normalize, web_test]))
        val_2_data = ImagenetDataset(args.data_root, num_classes, train=False, class_inds=inds,
                                     transform=transforms.Compose([im_web_normalize, im_test]))
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            val_2_sample = np.random.choice(len(val_2_data), int(args.subset_size * len(val_2_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
            val_2_data = Subset(val_2_data, val_2_sample)
        print("Loading MiniWebvision training set...")
        train_dataset = MislabelledDataset(train_data, num_classes=num_classes, cache=args.cache,
                                           transform=web_train_aug)
        print("Loading MiniWebvision validation set...")
        val_dataset = MislabelledDataset(val_data, num_classes=num_classes, cache=args.cache)
        print("Loading Imagenet validation set...")
        val_2_dataset = MislabelledDataset(val_2_data, num_classes=num_classes, cache=args.cache)
    elif args.dataset.lower() == "noisyminiimagenet":
        num_classes = 100
        train_data = NoisyMiniImagenet(args.data_root, train=True, transform=nmi_normalize)
        val_data = NoisyMiniImagenet(args.data_root, train=False,
                                     transform=transforms.Compose([nmi_normalize, im_test]))
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
        print("Loading NoisyMiniImagenet training set...")
        train_dataset = MislabelledDataset(train_data, num_classes=num_classes, cache=args.cache,
                                           transform=web_train_aug)
        print("Loading NoisyMiniImagenet validation set...")
        val_dataset = MislabelledDataset(val_data, num_classes=num_classes, cache=args.cache)
    elif args.dataset.lower() == "stanfordcarsred80":
        num_classes = 196
        train_data = StanfordCarsRed80(args.data_root, train=True, transform=nmi_normalize)
        val_data = StanfordCarsRed80(args.data_root, train=False,
                                     transform=transforms.Compose([nmi_normalize, im_test]))
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
        print("Loading StanfordCarsRed80 training set...")
        train_dataset = MislabelledDataset(train_data, num_classes=num_classes, cache=args.cache,
                                           transform=web_train_aug)
        print("Loading StanfordCarsRed80 validation set...")
        val_dataset = MislabelledDataset(val_data, num_classes=num_classes, cache=args.cache)
    elif args.dataset.lower() == "caltech256":
        num_classes = 257
        train_data = NoisyCaltech256(args.data_root, mislabel_ratio=0.0, train=True,
                                     transform=caltech_train)
        val_data = NoisyCaltech256(args.data_root, mislabel_ratio=0.0, train=False, transform=caltech_val)
        if args.subset:
            train_sample = np.random.choice(len(train_data), int(args.subset_size * len(train_data)), replace=False)
            val_sample = np.random.choice(len(val_data), int(args.subset_size * len(val_data)), replace=False)
            train_data = Subset(train_data, train_sample)
            val_data = Subset(val_data, val_sample)
        train_dataset = MislabelledDataset(train_data, num_classes=num_classes, cache=False,
                                           mislabel_ratio=args.mislabel_ratio)
        val_dataset = MislabelledDataset(val_data, num_classes=num_classes, cache=False, mislabel_ratio=0)
    else:
        raise NotImplementedError

    return num_classes, train_dataset, val_dataset, val_2_dataset


def save_model(path, model, num_classes, args):
    model_dict = {
        "model": args.model,
        "state": model.state_dict(),
        "num_classes": num_classes,
        "width": args.width
    }
    torch.save(model_dict, path)


class CSEWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        """
        CSE wrapper for elr compatibility.

        :param kwargs: kwargs to be passed to torch.nn.CrossEntropyLoss
        """
        super(CSEWrapper, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, outputs, labels, indices):
        return self.loss(outputs, labels)


def get_prunable_params(model):
    """
    returns a list of all of the conv linear layers along with their weights.

    :param model: the model to get the prunable parameters of.
    :return: a list of tuples of the form [(module, 'weight'), ...]
    """
    prunable = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prunable.append((m, 'weight'))
    return prunable
