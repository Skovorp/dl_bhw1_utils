#!g1.1

from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, img_dir, file_inds, labels_csv=None, transform_set=None):
        if labels_csv:
            self.labels = pd.read_csv(labels_csv)
        else:
            self.labels = None
        self.img_dir = img_dir
        self.transform_set = transform_set
        self.file_inds = file_inds

    def __len__(self):
        return self.file_inds.shape[0]

    def __getitem__(self, idx):
        file_ind = self.file_inds[idx]
        if self.labels is None:
            img_name = self.img_dir + f'/test_{file_ind:05}.jpg'
            label = 0
        else:
            img_name = self.img_dir + f'/trainval_{file_ind:05}.jpg'
            label = self.labels.iloc[file_ind, 1]
        image = Image.open(img_name).convert('RGB')
        if self.transform_set:
            image = self.transform_set(image)
        return image, label


def files_in_dir(dir):
    _, _, files = next(os.walk(dir))
    return len(files)

def get_my_data(path_to_train, path_to_test, path_to_labels, batch_size_train, batch_size_test, transform_trainval, transform_test):
    torch.manual_seed(0)
    np.random.seed(0)

    trainval_size = files_in_dir(path_to_train)
    test_size = files_in_dir(path_to_test)

    trainval_inds = np.arange(trainval_size)
    np.random.shuffle(trainval_inds)
    # train_inds = trainval_inds[:trainval_inds.shape[0] // 10 * 8]
    # val_inds = trainval_inds[trainval_inds.shape[0] // 10 * 8:]
    train_inds = trainval_inds[:trainval_inds.shape[0] // 10]
    val_inds = trainval_inds[trainval_inds.shape[0] // 10 : trainval_inds.shape[0] // 10 * 2]

    trainset = MyDataset(path_to_train, train_inds, path_to_labels, transform_trainval)
    valset = MyDataset(path_to_train, val_inds, path_to_labels, transform_test)
    testset = MyDataset(path_to_test, test_inds, None, transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_test,
                                                shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                                shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader