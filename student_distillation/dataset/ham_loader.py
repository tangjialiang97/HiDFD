import os
import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

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

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
                      for val in image_list]
    return images

class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

def han_loader(path='/opt/data/private/data/HAM/HAM_CLS', batch_size=16, num_workers=1, pin_memory=False, split_ratio=0.8):
    image_list = open(os.path.join(path, "image_list.txt")).readlines()
    random.seed(42)  # 使用任意整数作为种子
    random.shuffle(image_list)
    split_point = int(split_ratio * len(image_list))

    # 分割为训练集和测试集
    list_train = image_list[:split_point]
    list_test = image_list[split_point:]

    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    train_set = ImageList(list_train,
                                 transform=transforms_train)
    test_set = ImageList(list_test,
                                transform=transforms_test)
    print('Load %d train images, %d test images' % (len(train_set), len(test_set)))
    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def get_ham_dataloaders(opt):
    """
    Data Loader for imagenet
    """

    # data_folder = get_data_folder(opt)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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