"""
get data loaders
"""
from __future__ import print_function

from torch.utils.data import DataLoader
from torchvision import datasets
import torch



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


def get_i32_dataloader(transform=None, dataset='C10_select', train_num=None, select_dir=None):
    """
    Data Loader for imagenet
    """

    if dataset == 'C10_select':
        root = '****/data/cifar/cifar10_select'
    elif dataset == 'C100_select':
        root = '****/data/cifar/cifar100_select'

    if select_dir is not None:
        root = select_dir

    print('root:', root)
    train_set = ImageFolderInstance(root, transform=transform)
    if train_num is not None:
        import random
        random.seed(0)
        indices = torch.randperm(len(train_set)).tolist()[:train_num]
        train_set = torch.utils.data.Subset(train_set, indices)




    return train_set
