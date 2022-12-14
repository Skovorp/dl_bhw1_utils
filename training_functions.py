#!g1.1

from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import wandb
import pandas as pd


def rand_bbox(W, H, lmd):
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py?plain=1#L279
    cut_rat = np.sqrt(1 - lmd)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    new_lmd = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    return bbx1, bby1, bbx2, bby2, new_lmd


def test(model, loader):
    loss_sum = 0
    acc_sum = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)

        loss_sum += nn.functional.cross_entropy(pred, target, reduction='sum').item()
        acc_sum += (pred.argmax(1) == target).sum().item()
    return loss_sum / len(loader.dataset), acc_sum / len(loader.dataset)


def train_epoch_simple(model, optimizer, train_loader, label_smoothing):
    loss_sum = 0
    acc_sum = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = nn.functional.cross_entropy(pred, target, label_smoothing=label_smoothing)

        loss_sum += loss.item() * target.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (pred.argmax(1) == target).sum().item()
        acc_sum += acc
        # acc_log.append(acc.item())

    # return loss_log, acc_log,
    return loss_sum / len(train_loader.dataset), acc_sum / len(train_loader.dataset)


def train_epoch_cutmix(model, optimizer, train_loader, alpha, aug_possibility, label_smoothing):
    loss_sum = 0
    acc_sum = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        coinflip = np.random.rand()
        if coinflip < aug_possibility:
            lmd = np.random.beta(alpha, alpha)
            pair_indexes = np.random.permutation(data.shape[0])
            bbx1, bby1, bbx2, bby2, lmd = rand_bbox(data.shape[2], data.shape[3], lmd)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[pair_indexes, :, bbx1:bbx2, bby1:bby2]
            target_b = target[pair_indexes]

            pred = model(data)
            loss = nn.functional.cross_entropy(pred, target, label_smoothing=label_smoothing) * lmd + \
                   nn.functional.cross_entropy(pred, target_b, label_smoothing=label_smoothing) * (1 - lmd)
        else:
            pred = model(data)
            loss = nn.functional.cross_entropy(pred, target, label_smoothing=label_smoothing)

        loss_sum += loss.item() * target.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if coinflip < aug_possibility:
            acc = torch.logical_or(pred.argmax(1) == target, pred.argmax(1) == target_b).sum().item()
        else:
            acc = (pred.argmax(1) == target).sum().item()
        acc_sum += acc

    return loss_sum / len(train_loader.dataset), acc_sum / len(train_loader.dataset)


def train_epoch_mixup(model, optimizer, train_loader, alpha, aug_possibility, label_smoothing):
    loss_sum = 0
    acc_sum = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        coinflip = np.random.rand()
        if coinflip < aug_possibility:
            lmd = np.random.beta(alpha, alpha)
            pair_indexes = np.random.permutation(data.shape[0])
            data = data * lmd + data[pair_indexes] * (1 - lmd)
            target_b = target[pair_indexes]

            pred = model(data)
            loss = nn.functional.cross_entropy(pred, target, label_smoothing=label_smoothing) * lmd + \
                   nn.functional.cross_entropy(pred, target_b, label_smoothing=label_smoothing) * (1 - lmd)
        else:
            pred = model(data)
            loss = nn.functional.cross_entropy(pred, target, label_smoothing=label_smoothing)

        loss_sum += loss.item() * target.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if coinflip < aug_possibility:
            acc = torch.logical_or(pred.argmax(1) == target, pred.argmax(1) == target_b).sum().item()
        else:
            acc = (pred.argmax(1) == target).sum().item()
        acc_sum += acc

    return loss_sum / len(train_loader.dataset), acc_sum / len(train_loader.dataset)


def train(model, optimizer, scheduler, n_epochs, train_loader, val_loader, alpha, augmentation_type, aug_possibility, label_smoothing, pass_acc=False):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []
    if augmentation_type not in ('simple', 'cutmix', 'mixup'):
        raise Exception(f"Bad augmentation_type: {augmentation_type}. Correct are: ('simple', 'cutmix', 'mixup')")

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        print(f"lr: {optimizer.param_groups[0]['lr']}")

        if augmentation_type == 'simple':
            train_loss, train_acc = train_epoch_simple(model, optimizer, train_loader, label_smoothing)
        elif augmentation_type == 'cutmix':
            train_loss, train_acc = train_epoch_cutmix(model, optimizer, train_loader, alpha, aug_possibility, label_smoothing)
        elif augmentation_type == 'mixup':
            train_loss, train_acc = train_epoch_mixup(model, optimizer, train_loader, alpha, aug_possibility, label_smoothing)

        val_loss, val_acc = test(model, val_loader)

        # train_loss_log.extend(train_loss)
        # train_acc_log.extend(train_acc)

        # val_loss_log.append(val_loss)
        # val_acc_log.append(val_acc)

        print(f" train loss: {train_loss}, train acc: {train_acc}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")
        wandb.log({"train_loss": train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

        if scheduler is not None:
            if pass_acc:
                scheduler.step(val_acc)
            else:
                scheduler.step()

    # return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def predict_test(model, loader):
    res = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(1).tolist()
        res.extend(pred)
    return res


def generate_csv(labels, path):
    df = pd.DataFrame({
        'Id': [f'test_{i:05}.jpg' for i in range(len(labels))],
        'Label': labels
    })
    df.to_csv(f'{path}/labels_test.csv', index=False)
