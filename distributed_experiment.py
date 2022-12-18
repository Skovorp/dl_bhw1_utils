from __future__ import print_function, division
import os
import torch
import pandas as pd
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import wandb
from torch import optim
import os
import pandas as pd
from datetime import datetime
import traceback

from torch import optim, nn
import torch


from training_functions import train, predict_test, generate_csv
from build_dataset import get_my_data


def remote_experiment(trial):
    torch.manual_seed(3407)
    np.random.seed(10)

    name = f'remote_experiment_{datetime.now().strftime("%H_%M")}'
    model_name = 'resnet18'

    optuna_params = {
        'pick_dropout': trial.suggest_float('dropout', 0, 1),
        'pick_wd': trial.suggest_float('dropout', 0, 1),
        'pick_alpha': trial.suggest_float('alpha', 0, 4),
        'pick_gamma': trial.suggest_float('gamma', 0.6, 0.95),
        'pick_label_smoothing': trial.suggest_float('label_smoothing', 0, 1),
    }

    dropout = optuna_params['pick_dropout']

    net = torch.hub.load('pytorch/vision:v0.10.0', model_name, num_classes=200)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    if dropout > 0:
        if model_name[:6] == 'resnet':
            print("set dropout")
            net.fc = nn.Sequential(nn.Dropout(p=dropout), net.fc)
        elif model_name == 'mobilenet_v2':
            net.classifier[0].p = dropout
        else:
            raise Exception('Can\'t add dropout')

    optimizer = optim.AdamW(net.parameters(), lr=0.002, weight_decay=optuna_params['pick_wd'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=optuna_params['pick_gamma'])

    run_config = {
        'model_name': model_name,
        'transform_set_train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),

        'transform_set_test': transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'batch_size_train': 32,
        'batch_size_test': 32,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'optimizer_params': optimizer.__dict__,
        'sheduler_params': scheduler.__dict__,
        'epochs': 30,
        'alpha': optuna_params['pick_alpha'],
        'augmentation_type': 'cutmix',
        'aug_possibility': 1,
        'label_smoothing': optuna_params['pick_label_smoothing'],
        'dropout': dropout,
    }

    train_loader, val_loader, test_loader = get_my_data(
        '/kaggle/working/bhw1-dataset/trainval',
        '/kaggle/working/bhw1-dataset/test',
        '/kaggle/working/bhw1-dataset/labels.csv',
        run_config['batch_size_train'], run_config['batch_size_test'],
        run_config['transform_set_train'], run_config['transform_set_test'])

    run = wandb.init(project="dl_big_hw1", entity="sposiboh", config=run_config, name=name)
    val_acc = 0
    try:
        val_acc = train(net, run_config['optimizer'], run_config['scheduler'], run_config['epochs'], train_loader, val_loader,
              run_config['alpha'], run_config['augmentation_type'], run_config['aug_possibility'],
              run_config['label_smoothing'], run)
    except:
        print(traceback.format_exc())
        print("FAILED TRAINING")
        val_acc = 0
    finally:
        wandb.finish()
    return val_acc
