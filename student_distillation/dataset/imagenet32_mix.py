"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
import torch
from collections import Counter

def get_data_folder(opt):
    """
    return the path to store the data
    """
    data_folder = opt.data_path

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_i32_dataloader(opt):
    """
    Data Loader for imagenet
    """

    normalize = transforms.Normalize(mean=[0.480, 0.457, 0.409],
                                     std=[0.275, 0.271, 0.281])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


    train_set_real = ImageFolderInstance(opt.train_folder_real, transform=train_transform)
    train_set_fake = ImageFolderInstance(opt.train_folder_fake, transform=train_transform)

    indices_real = torch.randperm(len(train_set_real)).tolist()[:opt.real_num]

    if opt.repeated_num == None:
        train_set_real = torch.utils.data.Subset(train_set_real, indices_real)
    else:
        indices_real = indices_real * opt.repeated_num
        train_set_real = torch.utils.data.Subset(train_set_real, indices_real)
    indices_fake = torch.randperm(len(train_set_fake)).tolist()[:opt.fake_num]
    train_set_fake = torch.utils.data.Subset(train_set_fake, indices_fake)

    print('Load %d real images, %d fake images' % (len(train_set_real), len(train_set_fake)))

    train_set_mix = torch.utils.data.ConcatDataset([train_set_real, train_set_fake])
    train_loader = DataLoader(train_set_mix,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              pin_memory=True)



    return train_loader, train_set_mix