import sys, json
sys.path.append('/home/dkkim/workspace/MLVU/biases-are-features')
# -*- coding: utf-8 -*-
import torch
from torch.backends import cudnn

import os
import random

from torchvision.transforms.transforms import ToPILImage
from option_dk import get_option
from trainer_dk import Trainer
from utils import save_option
# import data_loader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

from dataset.oxford_pet import load_annotation, Databasket

def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    # if not torch.cuda.is_available() and option.cuda:
    #     option.cuda = False
    if option.cuda:
        print(f'cuda setting on {option.CUDA_VISIBLE_DEVICES}')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = option.CUDA_VISIBLE_DEVICES
        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark
    if option.train_baseline:
        option.is_train = True


def main():
    option = get_option()
    backend_setting(option)
    trainer = Trainer(option)

    train_transforms = transforms.Compose([#transforms.ToPILImage(),
                                # transforms.Resize(256),
                                #  transforms.CenterCrop(224),
                                #  transforms.ToTensor(),
                                #  transforms.Normalize(
                                #      mean=[0.485, 0.456, 0.406],
                                #      std=[0.229, 0.224, 0.225])
                                transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop((224)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
                                 ])
    valid_transforms = transforms.Compose([#transforms.ToPILImage(),
                                # transforms.Resize(256),
                                # transforms.CenterCrop(224),
                                # transforms.ToTensor(),
                                # transforms.Normalize(
                                #     mean=[0.485, 0.456, 0.406],
                                #     std=[0.229, 0.224, 0.225])
                                transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop((224)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
                                ])

    trainval_data = load_annotation(txt_path='/home/dkkim/workspace/MLVU/biases-are-features/oxford_pet/annotations/trainval.txt', 
                                    img_dir='/home/dkkim/workspace/MLVU/biases-are-features/oxford_pet/images',
                                    label_col=2)
    test_data = load_annotation(txt_path='/home/dkkim/workspace/MLVU/biases-are-features/oxford_pet/annotations/test.txt', 
                                img_dir='/home/dkkim/workspace/MLVU/biases-are-features/oxford_pet/images',
                                label_col=2)
    db = Databasket(trainval_data, option.n_class)
    # train_dataset, valid_dataset = db.gen_dataset_split(val_split=0.2, train_transforms= train_transforms, val_transforms=valid_transforms)
    train_dataset = db.gen_dataset(transforms=train_transforms)
    db = Databasket(test_data, option.n_class)
    test_dataset = db.gen_dataset(transforms=valid_transforms)
    # print(len(train_dataset), len(valid_dataset), len(test_dataset))
    print(len(train_dataset), len(test_dataset))

    train_loader = data.DataLoader(dataset=train_dataset, 
                              batch_size=option.batch_size,
                              shuffle=True,
                              num_workers=option.num_workers)

    # valid_loader = data.DataLoader(dataset=valid_dataset, 
    #                           batch_size=option.batch_size,
    #                           shuffle=True,
    #                           num_workers=option.num_workers)
    
    test_loader = data.DataLoader(dataset=test_dataset, 
                              batch_size=option.batch_size,
                              shuffle=True,
                              num_workers=option.num_workers)

    if option.is_train:
        save_option(option)
        trainer.train(train_loader, val_loader=test_loader)
    else:
        trainer._validate(valid_loader)
        pass

if __name__ == '__main__': main()
