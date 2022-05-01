'''
Filename: main.py
Author: Kyle Whitecross
Description: Main training script for various models, datasets, and configurations.
'''

from utils import parse_args_with_config, load_datasets, get_prunable_params, save_model, CSEWrapper
from robust import ELRLoss
from train import train, train_mixup, validate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch
import numpy as np
import os
import time
import models


def main(args):
    # get device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print(f"Running on gpu {args.gpu}")
    else:
        device = None

    # get logging/save path
    save_path = os.path.join(args.save_dir, args.run_name + "_" + str(int(time.time())))
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print(f"Warning: directory {save_path} already exists!")

    # tensorboard logging
    writer = SummaryWriter(save_path)

    # load datasets
    num_classes, train_dataset, val_dataset, val_dataset_2 = load_datasets(args)

    # optionally take a random subset of the data (for speed)
    '''
    if args.subset:
        train_size = int(len(train_dataset) * args.subset_size)
        train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), train_size, replace=False))
        val_size = int(len(val_dataset) * args.subset_size)
        val_dataset = Subset(val_dataset, np.random.choice(len(val_dataset), val_size, replace=False))
        if val_dataset_2 is not None:
            val_size_2 = int(len(val_dataset_2) * args.subset_size)
            val_dataset_2 = Subset(val_dataset_2, np.random.choice(len(val_dataset_2), val_size, replace=False))
    '''

    # setup dataloaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)

    # optionally setup mixup loader
    mixup_loader = None
    if args.mixup:
        mixup_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)

    # optionally setup val_2_loader
    val_loader_2 = None
    if val_dataset_2 is not None:
        val_loader_2 = DataLoader(val_dataset_2, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)

    # optionally load model
    if args.load_model:
        # load checkpoint
        # f = open(args.load_model_path)
        model_save = torch.load(args.load_model_path, map_location='cpu')

        # init blank model
        model = models.__dict__[model_save['model']](model_save['width'], model_save['num_classes'])

        # set up model as pruned model
        prune.global_unstructured(get_prunable_params(model), prune.RandomUnstructured, amount=0.0)

        # load state dict
        model.load_state_dict(model_save['state'])
    else:
        model = models.__dict__[args.model](args.width, num_classes)

        # prune the model
        prune.global_unstructured(get_prunable_params(model), prune.RandomUnstructured, amount=(1 - args.density))

    if args.dataparallel:
        model = nn.DataParallel(model)
        device = f'cuda:{model.device_ids[0]}'
        model = model.to(device)
    else:
        if device is not None:
            model = model.to(device)

    # save initial model
    save_model(os.path.join(save_path, 'initial_model.pt'), model, num_classes, args)

    # set up loss functions
    val_criterion = CSEWrapper()
    if args.elr:
        train_criterion = ELRLoss(len(train_dataset), num_classes, args.elr_lambda, args.elr_beta, device=device)
    else:
        train_criterion = val_criterion

    # set up optimizer
    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), args.learning_rate, args.momentum, weight_decay=args.l2_reg)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
    else:
        raise NotImplementedError()

    # set up lr scheduler
    lr_scheduler = MultiStepLR(optimizer, args.lr_milestones, args.lr_gamma)

    # train loop stats
    val_acc = 0
    val_acc_2 = 0
    val_loss = 0
    val_loss_2 = 0
    best_val_acc = 0
    best_val_acc_2 = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_accs = []
    val_losses_2 = []
    val_accs_2 = []
    best_val_accs_2 = []

    # train loop
    for e in range(args.epochs):
        # train
        model.train()
        if args.mixup:
            train_loss, train_acc = train_mixup(model, train_criterion, optimizer, train_loader, mixup_loader,
                                                 args.mixup_strength, e, device)
        else:
            train_loss, train_acc = train(model, train_criterion, optimizer, train_loader, e, device)

        # validate
        if e % args.eval_every == 0:
            model.eval()
            val_loss, val_acc = validate(model, val_criterion, val_loader, e, device)
            if val_loader_2 is not None:
                val_loss_2, val_acc_2 = validate(model, val_criterion, val_loader_2, e, device)

        # step lr
        lr_scheduler.step()

        # store data
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        if e % args.eval_every == 0:
            best_val_acc = max(best_val_acc, val_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            best_val_accs.append(best_val_acc)
            if val_loader_2 is not None:
                best_val_acc_2 = max(best_val_acc_2, val_acc_2)
                val_losses_2.append(val_loss_2)
                val_accs_2.append(val_acc_2)
                best_val_accs_2.append(best_val_acc_2)

        # tensorboard log stats
        writer.add_scalar("loss/train", train_loss, e)
        writer.add_scalar("loss/val", val_loss, e)
        writer.add_scalar("acc/train", train_acc, e)
        writer.add_scalar("acc/val", val_acc, e)
        writer.add_scalar("acc/val_best", best_val_acc, e)
        writer.add_scalar("optim/lr", lr_scheduler.get_last_lr()[0], e)
        if val_loader_2 is not None:
            writer.add_scalar("loss/val_2", val_loss_2, e)
            writer.add_scalar("acc/val_2", val_acc_2, e)
            writer.add_scalar("acc/val_2_best", best_val_acc_2, e)

    # log hyperparams
    writer.add_hparams({
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_gamma": args.lr_gamma,
        "n_milestones": len(args.lr_milestones),
        "momentum": args.momentum,
        "l2_reg": args.l2_reg,
        "epochs": args.epochs,
        "width": args.width,
        "density": args.density,
        "subset": int(args.subset),
        "subset_size": args.subset_size,
        "cifar_img_size": args.cifar_img_size,
        "webvision_img_size": args.webvision_img_size,
        "elr": int(args.elr),
        "elr_lambda": args.elr_lambda,
        "elr_beta": args.elr_beta,
        "mixup": int(args.mixup),
        "mixup_strength": args.mixup_strength,
        "num_classes": num_classes
    }, {
        "best_acc": best_val_acc
    })
    writer.flush()
    writer.close()

    # save model
    save_model(os.path.join(save_path, "final_model.pt"), model, num_classes, args)

    # set up return dict
    ret = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_accs": best_val_accs
    }
    if val_loader_2 is not None:
        ret["val_losses_2"] = val_losses_2
        ret["val_accs_2"] = val_accs_2
        ret["best_val_accs_2"] = best_val_accs_2

    # save return dict
    torch.save(ret, os.path.join(save_path, "results.pt"))

    # done!
    return ret



if __name__ == "__main__":
    args = parse_args_with_config()
    main(args)

