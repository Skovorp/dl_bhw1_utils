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

from torch import optim
import torch
from training_functions import train, predict_test, generate_csv
from build_dataset import get_my_data


def experiment_alpa_resnet(alpha):
    name = f'resnet_alpha_{alpha}'
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False, num_classes=200)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

    run_config = {
        'transform_set_train': transforms.Compose([
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'transform_set_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'batch_size_train': 128,
        'batch_size_test': 256,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'sheduler_params': scheduler.__dict__,
        'epochs': 150,
        'alpha': 4
    }

    train_loader, val_loader, test_loader = get_my_data(
        '/home/jupyter/mnt/datasets/bhw1/trainval/trainval',
        '/home/jupyter/mnt/datasets/bhw1/test/test',
        '/home/jupyter/mnt/datasets/bhw1/labels.csv',
        run_config['batch_size_train'], run_config['batch_size_test'],
        run_config['transform_set_train'], run_config['transform_set_test'])

    wandb.init(project="dl_big_hw1", entity="sposiboh", config=run_config, name=name)
    train(net, run_config['optimizer'], run_config['scheduler'], run_config['epochs'], train_loader, val_loader,
          alpha=run_config['alpha'])
    wandb.finish()

    os.mkdir(name)
    torch.save(net.state_dict(), f'{name}/{name}_model_weights.pth')
    out = predict_test(net, test_loader)
    generate_csv(out, name)


