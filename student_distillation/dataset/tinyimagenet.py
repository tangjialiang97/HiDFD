from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
from torchvision import transforms
from dataset.tinyimagenet_switch import TinyImageNet200
"""
mean = {
    'tiny-imagenet': (0.485, 0.456, 0.406),
}

std = {
    'tiny-imagenet': (0.229, 0.224, 0.225),
}
"""

labels_t = []
image_names = []
with open('/opt/data/private/data/tiny-imagenet-200/wnids.txt') as wnid:
    for line in wnid:
        labels_t.append(line.strip('\n'))
for label in labels_t:
    txt_path = '/opt/data/private/data/tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append(line.strip('\n').split('\t')[0])
    image_names.append(image_name)
labels = np.arange(200)

val_labels_t = []
val_labels = []
val_names = []
with open('/opt/data/private/data/tiny-imagenet-200/val/val_annotations.txt') as txt:
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_t)):
        if val_labels_t[i] == labels_t[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)

def get_data_folder():

    data_folder = '/opt/data/private/data/tiny-imagenet-200'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class tinyimagenetInstance(datasets.CIFAR100):
    """tiny-imagenetInstance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_tinyimagenet_dataloaders(opt):
    """
    tiny-imagenet
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ROOT = '/opt/data/private/data/'
    trainset = TinyImageNet200(root=ROOT, train=True,
                               transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.batch_size)

    testset = TinyImageNet200(root=ROOT, train=False,
                              transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.batch_size)
    return train_loader, val_loader
