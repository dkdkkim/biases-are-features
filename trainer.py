# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import model

import time
import os
import math
import sys

from tqdm import tqdm

from utils import logger_setting


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)



class Trainer(object):
    def __init__(self, option):
        self.option = option

        self._build_model()
        self._set_optimizer()
        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)

    def _build_model(self):
        self.net = model.convnet(num_classes=self.option.n_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()


    def _set_optimizer(self):
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)


    @staticmethod
    def _weights_init_xavier(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)            

    def _initialization(self):
        self.net.apply(self._weights_init_xavier)

        if self.option.is_train and self.option.use_pretrain:
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("no prtrained model")


    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
        else:
            self.net.eval()


    def _train_step(self, data_loader, step):
        
        loss_sum = 0.
        for i, (images,labels) in enumerate(tqdm(data_loader)):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            pred_label = self.net(images)

            loss = self.loss(pred_label, torch.squeeze(labels))
            loss_sum += loss
            loss.backward()
            self.optim.step()

        msg = f"[TRAIN] LOSS : {loss_sum/len(data_loader)}"
        self.logger.info(msg)


    def _validate(self, data_loader):
        self._mode_setting(is_train=False)
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()
        else:
            print("No trained model")
            sys.exit()

        num_test = 10000

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        for i, (images,labels) in enumerate(tqdm(data_loader)):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            pred_label = self.net(images)


            loss = self.loss(pred_label, torch.squeeze(labels))
            
            batch_size = images.shape[0]
            total_num_correct += self._num_correct(pred_label,labels,topk=1).data
            total_loss += loss.data*batch_size
            total_num_test += batch_size
               
        avg_loss = total_loss/total_num_test
        avg_acc = total_num_correct/total_num_test
        msg = f"[EVALUATION] LOSS  {avg_loss}, ACCURACY : {avg_acc}"
        self.logger.info(msg)


    def _num_correct(self,outputs,labels,topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct


    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy


    def _save_model(self, step):
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, os.path.join(self.option.save_dir,self.option.exp_name, f'checkpoint_step_{step}.pth'))
        print('checkpoint saved. step : %d'%step)


    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])


    def train(self, train_loader, val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            if self.option.train_baseline:
                self._train_step_baseline(train_loader, step)
            else:
                self._train_step(train_loader,step)
            self.scheduler.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):
                if val_loader is not None:
                    self._validate(step, val_loader)
                self._save_model(step)


    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)
