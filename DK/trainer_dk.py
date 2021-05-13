# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.models as models

import model

import time
import os
import math
import sys

from tqdm import tqdm

from utils import logger_setting
# from model_baf_vgg19 import vgg19 as vgg19_baf
from model_vgg import vgg19 as vgg19_baf

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)

class AdaptiveConcatPool2d(torch.nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = torch.nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = torch.nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class FinetuneModel():
    def __init__(self) -> None:
        self.bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    
    def freeze_all(self, model_params):
        for param in model_params:
            param.requires_grad = False

    def create_head(self, nf, nc, bn_final=False, single_head=False):
        "Model head that takes in 'nf' features and outputs 'nc' classes"
        pool = AdaptiveConcatPool2d()
        layers = [pool, torch.nn.Flatten()]
        if single_head:
            layers += self.head_blocks(nf, 0.5, nc)
        else:
            layers += self.head_blocks(nf, 0.25, 512, torch.nn.ReLU(inplace=True))
            layers += self.head_blocks(512, 0.5, nc)
        
        if bn_final:
            layers.append(nn.BatchNorm1d(nc, momentum=0.01))
        
        return nn.Sequential(*layers)

    def head_blocks(self, in_dim, p, out_dim, activation=None):
        "Basic Linear block"
        layers = [
            nn.BatchNorm1d(in_dim),
            nn.Dropout(p),
            nn.Linear(in_dim, out_dim)
        ]
        
        if activation is not None:
            layers.append(activation)
            
        return layers     

    def requires_grad(self, layer):
        "Determines whether 'layer' requires gradients"
        ps = list(layer.parameters())
        if not ps: return None
        return ps[0].requires_grad

    def cnn_model(self, model, nc, hidden, single_head=False, bn_final=False, init=torch.nn.init.kaiming_normal_):
        "Creates a model using a pretrained 'model' and appends a new head to it with 'nc' outputs"
        
        # remove dense and freeze everything
        if single_head:
            body = nn.Sequential(*list(model.children())[:-1])
        else:
            body = nn.Sequential(*list(model.children())[:-2])
        head = self.create_head(hidden, nc, bn_final, single_head)
        
        model = torch.nn.Sequential(body, head)
        
        # freeze the resnet34 base of the model
        self.freeze_all(model[0].parameters())
        
        # initialize the weights of the head
        for child in model[1].children():
            if isinstance(child, torch.nn.Module) and (not isinstance(child, self.bn_types)) and self.requires_grad(child): 
                init(child.weight)
        
        return model                            



class Trainer(object):
    def __init__(self, option, steps_per_epoch):
        self.option = option

        self._build_model()
        self._set_optimizer(steps_per_epoch)
        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)

    def _build_model(self):
        if self.option.model == 'cnn':
            self.net = model.convnet(num_classes=self.option.n_class)
        elif self.option.model == 'resnet18':
            if self.option.use_pretrain:
                resnet =  models.resnet18(pretrained = True)
                resnet.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(resnet, self.option.n_class, 1024, bn_final=True)
            else:
                self.net = models.resnet18(pretrained = False, num_classes=self.option.n_class)
            # self.net = models.resnet18(pretrained = True)
            # self.net.fc = nn.Linear(512, self.option.n_class)
        
        elif self.option.model == 'resnet101':
            self.net = models.resnet101(pretrained = False, num_classes=self.option.n_class)


        elif self.option.model == 'vgg19':
            if self.option.use_pretrain:
                vgg19 =  models.vgg19(pretrained = True)
                vgg19.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(vgg19, self.option.n_class, 1024, bn_final=True)
            else:
                self.net = models.vgg19(pretrained = False, num_classes=self.option.n_class)
        
        elif self.option.model == 'mobilenet':
            if self.option.use_pretrain:
                mobilenet =  models.mobilenet_v2(pretrained = True)
                mobilenet.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(mobilenet, self.option.n_class, 2560, bn_final=True, single_head=True)
            else:
                self.net = models.vgg19(pretrained = False, num_classes=self.option.n_class)

        elif self.option.model == 'googlenet':
            if self.option.use_pretrain:
                googlenet =  models.googlenet(pretrained = True)
                googlenet.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(googlenet, self.option.n_class, 2048, bn_final=True)
            else:
                self.net = models.googlenet(pretrained = False, num_classes=self.option.n_class)

        elif self.option.model == 'alexnet':
            if self.option.use_pretrain:
                alexnet =  models.alexnet(pretrained = True)
                alexnet.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(alexnet, self.option.n_class, 512, bn_final=True)
            else:
                self.net = models.alexnet(pretrained = False, num_classes=self.option.n_class)
        
        elif self.option.model == 'vgg19_baf':
            from model_baf_vgg19 import vgg19 as vgg19_baf

            if self.option.use_pretrain:
                vgg19_baf = vgg19_baf(pretrained = True)
                vgg19_baf.eval()
                ft = FinetuneModel()
                self.net = ft.cnn_model(vgg19_baf, self.option.n_class, 512, bn_final=True)
            else:
                self.net = vgg19_baf(pretrained = False, num_classes=self.option.n_class)

        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()


    def _set_optimizer(self, steps):
        # self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        # self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, weight_decay=self.option.weight_decay)
        
        self.optim = optim.Adam(self.net.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)
        
        # self.optim = optim.Adam(self.net.parameters(), lr=1e-7,  weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=5e-2, pct_start=0.3, steps_per_epoch=steps, epochs=self.option.max_step)
        
        if self.option.use_pretrain:
            self.optim = optim.Adam(self.net.parameters(), lr=1e-7,  weight_decay=1e-5)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=5e-2, pct_start=0.3, steps_per_epoch=steps, epochs=self.option.max_step)


    @staticmethod
    def _weights_init_xavier(m):
        classname = m.__class__.__name__
        if classname == 'BasicConv2d' or classname == 'ConvBNReLU':
            pass
        elif classname.find('Conv') != -1:
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
        total_num_correct = 0.
        total_num_test = 0.
        for i, (images,labels) in enumerate(tqdm(data_loader)):
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            pred_label = self.net(images)

            total_num_correct += self._num_correct(pred_label,labels,topk=1).data
            batch_size = images.shape[0]
            total_num_test += batch_size

            loss = self.loss(pred_label, torch.squeeze(labels))
            loss_sum += loss
            loss.backward()
            self.optim.step()
        avg_acc = total_num_correct/total_num_test
        msg = f"[TRAIN] LOSS  {loss_sum/len(data_loader)}, ACCURACY : {avg_acc}"

        self.logger.info(msg)


    def _validate(self, data_loader, step=0):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print('not in training process')
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
        with torch.no_grad():
            for i, (images,labels) in enumerate(tqdm(data_loader)):
                
                images = self._get_variable(images)
                labels = self._get_variable(labels)

                # self.optim.zero_grad()
                pred_label = self.net(images)

                loss = self.loss(pred_label, torch.squeeze(labels))
                
                batch_size = images.shape[0]
                total_num_correct += self._num_correct(pred_label,labels,topk=1).data
                total_loss += loss.data*batch_size
                total_num_test += batch_size
               
        avg_loss = total_loss/total_num_test
        avg_acc = total_num_correct/total_num_test
        msg = f"[EVALUATION] step {step}, LOSS {avg_loss}, ACCURACY : {avg_acc}"
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
        if not self.option.use_pretrain:
            self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            self._mode_setting(is_train=True)

            if self.option.train_baseline:
                self._train_step_baseline(train_loader, step)
            else:
                self._train_step(train_loader,step)
            self.scheduler.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):
                if val_loader is not None:
                    self._validate(val_loader, step)
                self._save_model(step)


    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

if __name__ == '__main__': main()
