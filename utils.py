from argparse import ArgumentParser
import sys
import yaml
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from datasets import MislabelledDataset, WebvisionDataset, ImagenetDataset
import torch
import torch.nn as nn

DEFAULT_CONFIG = "exps/template.yaml"


def parse_args_with_config(args_str=None):
    # try and parse config first
    parser = ArgumentParser(description="Dynamic config based argument parser")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    if args_str is None:
        args_str = sys.argv
    known_args, _ = parser.parse_known_args(args_str)

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
            parser_config.add_argument(f"--disable_{key.lower()}", action="store_true", default=not val)
        else:
            parser_config.add_argument(f"--{key.lower()}", default=val)

    # parse arguments with new arg parser
    args = parser_config.parse_args(args_str)

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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    im_web_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    web_train_aug = transforms.Compose([
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip()
    ])
    web_test = transforms.Compose([
        transforms.CenterCrop(227)
    ])
    im_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227)
    ])

    # select dataset to load
    val_2_dataset = None
    if args.dataset.lower() == "cifar10":
        train_data = CIFAR10(args.data_root, train=True, transform=cifar_normalize, download=True)
        val_data = CIFAR10(args.data_root, train=False, transform=cifar_normalize, download=True)
        train_dataset = MislabelledDataset(train_data, mislabel_ratio=args.mislabel_ratio, num_classes=10,
                                           cache=args.cache, transform=cifar_aug)
        val_dataset = MislabelledDataset(val_data, mislabel_ratio=0, num_classes=10, cache=args.cache)
        num_classes = 10
    elif args.dataset.lower() == "cifar100":
        train_data = CIFAR100(args.data_root, train=True, transform=cifar_normalize, download=True)
        val_data = CIFAR100(args.data_root, train=False, transform=cifar_normalize, download=True)
        train_dataset = MislabelledDataset(train_data, mislabel_ratio=args.mislabel_ratio, num_classes=100,
                                           cache=args.cache, transform=cifar_aug)
        val_dataset = MislabelledDataset(val_data, mislabel_ratio=0, num_classes=100, cache=args.cache)
        num_classes = 100
    elif args.dataset.lower() == "webvision":
        train_data = WebvisionDataset(args.data_root, 50, train=True, transform=im_web_normalize)
        val_data = WebvisionDataset(args.data_root, 50, train=False,
                                    transform=transforms.Compose([im_web_normalize, web_test]))
        val_2_data = ImagenetDataset(args.data_root, 50, train=False,
                                     transform=transforms.Compose([im_web_normalize, im_test]))
        train_dataset = MislabelledDataset(train_data, num_classes=50, cache=args.cache, transform=web_train_aug)
        val_dataset = MislabelledDataset(val_data, num_classes=50, cache=args.cache)
        val_2_dataset = MislabelledDataset(val_2_data, num_classes=50, cache=args.cache)
        num_classes = 50
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
