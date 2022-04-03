'''
Filename: train.py
Author: Kyle Whitecross
Description: Various single-epoch training functions.
'''

from tqdm import tqdm
import torch
import numpy as np


def train(model, criterion, optimizer, train_loader, epoch, device=None):
    """
    Trains a model for a single epoch.
    :param model: The model.  Must be in train mode and on the proper device.
    :param criterion: The loss function.
    :param optimizer: The optimizer
    :param train_loader: The training dataset dataloader
    :param epoch: the current epoch (used for tqdm)
    :param device: the device to move the data.  If none, no data moving will occur
    :return: train loss, train acc
    """
    # stats
    running_loss = 0
    running_correct = 0
    running_examples = 0
    loop = tqdm(train_loader, desc=f"Train {epoch}", total=len(train_loader))

    # preclear grads
    optimizer.zero_grad()

    # iterate through loader
    for batch_x, batch_y, batch_real, batch_ind in loop:
        # potentially move data
        if device is not None:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

        # forward pass
        out = model.forward(batch_x)
        loss = criterion(out, batch_y, batch_ind).mean()

        # backward pass
        loss.backward()
        optimizer.step()

        # collect stats
        batch_size = len(batch_y)
        running_loss += loss.item() * batch_size
        n_correct = torch.sum(out.argmax(dim=-1) == batch_y).item()
        running_correct += n_correct
        running_examples += batch_size

        # clear grads
        optimizer.zero_grad()

        # update tqdm postfix
        loop.set_postfix({
            "loss": f"{running_loss / running_examples : .03f}",
            "acc": f"{running_correct / running_examples : .03f}"
        })

    return running_loss / running_examples, running_correct / running_examples


def train_mixup(model, criterion, optimizer, train_loader, mixup_loader, mixup_strength, epoch, device=None):
    """
    Trains a model for a single epoch using Mixup.

    :param model: The model.  Must be in train mode and on the proper device.
    :param criterion: The loss function.
    :param optimizer: The optimizer
    :param train_loader: The training dataset dataloader
    :param mixup_loader: A copy of the mixup loader.
    :param mixup_strength: The strength of the mixup.
    :param epoch: the current epoch (used for tqdm)
    :param device: the device to move the data.  If none, no data moving will occur
    :return: train loss, train acc
    """
    # stats
    running_loss = 0
    running_correct = 0
    running_examples = 0
    loop = tqdm(train_loader, desc=f"Train {epoch}", total=len(train_loader))
    mixup_iter = iter(mixup_loader)

    # preclear grads
    optimizer.zero_grad()

    # iterate through loader
    for batch_x, batch_y, batch_real, batch_ind in loop:
        # potentially move data
        if device is not None:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

        # get mixup data
        mix_x, mix_y, mix_real, mix_ind = next(mixup_iter)

        # potentially move data
        if device is not None:
            mix_x = mix_x.to(device)
            mix_y = mix_y.to(device)

        # mixup
        beta = np.random.beta(mixup_strength, mixup_strength)
        batch_x = batch_x * beta + mix_x * (1 - beta)

        # forward pass
        out = model.forward(batch_x)
        loss = beta * criterion(out, batch_y, batch_ind).mean() + (1 - beta) * criterion(out, mix_y, mix_ind).mean()

        # backward pass
        loss.backward()
        optimizer.step()

        # collect stats
        batch_size = len(batch_y)
        running_loss += loss.item() * batch_size
        if beta > 0.5:
            n_correct = torch.sum(out.argmax(dim=-1) == batch_y).item()
        else:
            n_correct = torch.sum(out.argmax(dim=-1) == mix_y).item()
        running_correct += n_correct
        running_examples += batch_size

        # clear grads
        optimizer.zero_grad()

        # update tqdm postfix
        loop.set_postfix({
            "loss": f"{running_loss / running_examples : .03f}",
            "acc": f"{running_correct / running_examples : .03f}"
        })

    return running_loss / running_examples, running_correct / running_examples


def validate(model, criterion, test_loader, epoch, device=None):
    """
    Computes the accuracy and loss without updating the model.
    :param model: The model.  Must be in the proper mode and on the proper device.
    :param criterion: The loss function.
    :param test_loader: The test loader.
    :param epoch: The current epoch (for tqdm)
    :param device: The device to move the batches to.  If None no data movement will occur.
    :param classwise: If true, will return test_acc as a vector of size num_classes
    :param num_classes: The number of classes (only needed if classwise is True).
    :return: test_loss, test_acc
    """

    # stats
    running_loss = 0
    running_correct = 0
    running_examples = 0
    loop = tqdm(test_loader, desc=f"Valid {epoch}", total=len(test_loader))

    # no grads
    with torch.no_grad():
        # iterate through loader
        for batch_x, batch_y, batch_real, batch_ind in loop:
            # optionally move data
            if device is not None:
                batch_x = batch_x.to(device)
                batch_real = batch_real.to(device)

            # forward pass
            out = model.forward(batch_x)
            loss = criterion(out, batch_real, batch_ind).mean()

            # collect stats
            batch_size = len(batch_y)
            running_loss += loss.item() * batch_size
            n_correct = torch.sum(out.argmax(dim=-1) == batch_real).item()
            running_correct += n_correct
            running_examples += batch_size

            # set tqdm postfix
            loop.set_postfix({
                "loss": f"{running_loss / running_examples : .03f}",
                "acc": f"{running_correct / running_examples : .03f}"
            })

    return running_loss / running_examples, running_correct / running_examples
