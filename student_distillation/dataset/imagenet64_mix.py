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


def get_i64_dataloader(opt):
    """
    Data Loader for imagenet
    """

    # data_folder = get_data_folder(opt)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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


def train_loader_select_student(student, data_train_loader, train_set, opt):

    problist = []
    predlist = []
    classwise_acc = torch.zeros((opt.num_classes),).cuda()
    p_cutoff_cls = torch.zeros((opt.num_classes),).cuda()

    student.eval()
    for i, (inputs, labels, _) in enumerate(data_train_loader):
        inputs = inputs.cuda()
        outputs = student(inputs)
        pseudo_label = torch.softmax(outputs.detach(), dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        problist.append(max_probs)
        predlist.append(max_idx)
    problist = torch.cat(problist, dim=0)
    predlist = torch.cat(predlist, dim=0)

    pseudo_counter = Counter(predlist.tolist())
    for i in range(opt.num_classes):
        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
        p_cutoff_cls[i] = opt.p_cutoff * (classwise_acc[i] / (2. - classwise_acc[i]))

    pos_idx = []

    for i in range(len(problist)):
        p_cutoff = p_cutoff_cls[predlist[i]]
        if problist[i] > p_cutoff:
            pos_idx.append(i)

    data_train_select = torch.utils.data.Subset(train_set, pos_idx)
    print('Select_sampes %d' % (len(data_train_select)))
    trainloader_select = torch.utils.data.DataLoader(data_train_select, batch_size=opt.batch_size, shuffle=False,
                                                     num_workers=opt.num_workers)


    return trainloader_select

def train_loader_select_teacher(teacher, data_train_loader, train_set, opt):

    problist = []

    teacher.eval()
    for i, (inputs, labels, _) in enumerate(data_train_loader):
        inputs = inputs.cuda()
        outputs = teacher(inputs)
        pseudo_label = torch.softmax(outputs.detach(), dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        problist.append(max_probs)

    problist = torch.cat(problist, dim=0)


    pos_idx = []
    for i in range(len(problist)):
        if problist[i] > opt.p_cutoff:
            pos_idx.append(i)

    data_train_select = torch.utils.data.Subset(train_set, pos_idx)
    print('Select_sampes %d' % (len(data_train_select)))
    trainloader_select = torch.utils.data.DataLoader(data_train_select, batch_size=opt.batch_size, shuffle=False,
                                                     num_workers=opt.num_workers)


    return trainloader_select

def train_loader_select(teacher, data_train_loader, opt):
    value = []
    index = 0
    celoss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    teacher.eval()
    for i, (inputs, labels, _) in enumerate(data_train_loader):
        inputs = inputs.cuda()
        outputs = teacher(inputs)
        pred = outputs.data.max(1)[1]
        loss = celoss(outputs, pred)
        value.append(loss.detach().clone())
        index += inputs.shape[0]

    values = torch.cat(value, dim=0)




    positive_index = values.topk(opt.num_select, largest=False)[1]

    positive_index = positive_index.tolist()


    data_folder = get_data_folder(opt)

    normalize = transforms.Normalize(mean=[0.480, 0.457, 0.409],
                                     std=[0.275, 0.271, 0.281])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = data_folder

    train_set = ImageFolderInstance(train_folder, transform=train_transform)

    data_train_select = torch.utils.data.Subset(train_set, positive_index)
    trainloader_select = torch.utils.data.DataLoader(data_train_select, batch_size=opt.batch_size, shuffle=True,
                                                     num_workers=opt.num_workers)

    return trainloader_select
