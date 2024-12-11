"""
get data loaders
"""
from __future__ import print_function
from torchvision.datasets import ImageFolder
import _pickle


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class I32_dataset(Dataset):
    def __init__(self, data_path, transform):

        self.transform = transform
        train_folder = data_path

        train_dic = unpickle(train_folder)
        train_data = train_dic['train']['data']
        train_target = train_dic['train']['target']
        data_num = len(train_data)


        train_data = train_data.reshape((data_num, -1, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))


        self.train_data = train_data
        self.train_target = train_target




    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_target[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target



    def __len__(self):
        return len(self.train_data)


def get_cinic_dataloaders(opt):
    train_dir = '/opt/data/private/data/CINIC/train'
    val_dir = '/opt/data/private/data/CINIC/valid'
    test_dir = '/opt/data/private/data/CINIC/test'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    data_train = ImageFolder(train_dir, transform=train_transform)
    data_val = ImageFolder(val_dir, transform=test_transform)

    # train_data = I32_dataset(train_dir, train_transform)
    # val_data = I32_dataset(val_dir, transform)

    trainloader = DataLoader(
        dataset=data_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)
    valloader = DataLoader(
        dataset=data_val,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    return trainloader, valloader


