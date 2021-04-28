# -*- coding: utf-8 -*-
import torch
from torch.backends import cudnn

import os
import random
from option import get_option
from trainer import Trainer
from utils import save_option
# import data_loader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    if not torch.cuda.is_available() and option.cuda:
        option.cuda = False

    if option.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark
    if option.train_baseline:
        option.is_train = True


def main():
    option = get_option()
    backend_setting(option)
    trainer = Trainer(option)

#    custom_loader = data_loader.WholeDataLoader(option)
#    trainval_loader = torch.utils.data.DataLoader(custom_loader,
#                                                  batch_size=option.batch_size,
#                                                  shuffle=True,
#                                                  num_workers=option.num_workers)
    download_root = '/data'

    mnist_transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5,), (1.0,))
    ])

    train_dataset = datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
    valid_dataset = datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True)

    train_loader = data.DataLoader(dataset=train_dataset, 
                              batch_size=option.batch_size,
                              shuffle=True,
                              num_workers=option.num_workers)

    valid_loader = data.DataLoader(dataset=valid_dataset, 
                              batch_size=option.batch_size,
                              shuffle=True,
                              num_workers=option.num_workers)

    if option.is_train:
        save_option(option)
        trainer.train(train_loader)
    else:
        trainer._validate(valid_loader)

        pass

if __name__ == '__main__': main()
