import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class BAF_VGG(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(BAF_VGG, self).__init__()
        self.block1 = make_layers(3,[64, 64, 'M'], True)
        self.block2 = make_layers(64, [128, 128, 'M'], True)
        self.block3 = make_layers(128, [256, 256, 256, 256, 'M'], True)
        self.block4 = make_layers(256, [512, 512, 512, 512, 'M'], True)
        self.block5 = make_layers(512, [512, 512, 512, 512, 'M'], True)
        
        self.trans1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(True),
        )
        self.trans2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(True),
        )
        self.trans3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(True),
        )
        self.trans4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
        )
        self.trans5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 5 * 5, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        out1 = self.avgpool(x)
        out1 = self.trans1(out1) # 1x1x1024
        x = self.block2(x)
        out2 = self.avgpool(x)
        out2 = self.trans2(out2)
        x = self.block3(x)
        out3 = self.avgpool(x)
        out3 = self.trans3(out3)
        x = self.block4(x)
        out4 = self.avgpool(x)
        out4 = self.trans4(out4)
        x = self.block5(x)
        out5 = self.avgpool(x)
        out5 = self.trans5(x)

        pad1 = torch.zeros_like(out1)
        pad2 = torch.zeros_like(out2)
        pad3 = torch.zeros_like(out3)
        pad4 = torch.zeros_like(out4)
        pad5 = torch.zeros_like(out5)
        
        conv_concat_all = torch.cat((out1, out2, out3, out4, out5), axis=1)
        conv_concat_1 = torch.cat((pad1, out2, out3, out4, out5), axis=1)
        conv_concat_2 = torch.cat((out1, pad2, out3, out4, out5), axis=1)
        conv_concat_3 = torch.cat((out1, out2, pad3, out4, out5), axis=1)
        conv_concat_4 = torch.cat((out1, out2, out3, pad4, out5), axis=1)
        conv_concat_5 = torch.cat((out1, out2, out3, out4, pad5), axis=1)
        conv_pjt_1 = conv_concat_all - torch.matmul(torch.matmul(torch.matmul(conv_concat_1, torch.inverse(torch.matmul(
            torch.transpose(conv_concat_1,0,1),conv_concat_1))),torch.transpose(conv_concat_1,0,1)), conv_concat_all)
        conv_pjt_2 = conv_concat_all - torch.matmul(torch.matmul(torch.matmul(conv_concat_2, torch.inverse(torch.matmul(
            torch.transpose(conv_concat_2,0,1),conv_concat_2))),torch.transpose(conv_concat_2,0,1)), conv_concat_all)
        conv_pjt_3 = conv_concat_all - torch.matmul(torch.matmul(torch.matmul(conv_concat_3, torch.inverse(torch.matmul(
            torch.transpose(conv_concat_3,0,1),conv_concat_3))),torch.transpose(conv_concat_3,0,1)), conv_concat_all)
        conv_pjt_4 = conv_concat_all - torch.matmul(torch.matmul(torch.matmul(conv_concat_4, torch.inverse(torch.matmul(
            torch.transpose(conv_concat_4,0,1),conv_concat_4))),torch.transpose(conv_concat_4,0,1)), conv_concat_all)
        conv_pjt_5 = conv_concat_all - torch.matmul(torch.matmul(torch.matmul(conv_concat_5, torch.inverse(torch.matmul(
            torch.transpose(conv_concat_5,0,1),conv_concat_5))),torch.transpose(conv_concat_5,0,1)), conv_concat_all)
        
        conv_pjt_concat = torch.cat((conv_pjt_1, conv_pjt_2, conv_pjt_3, conv_pjt_4, conv_pjt_5), axis=1)
        # y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),
        #                       y_conv_H, transpose_b=True), y_conv_loss)
        
        # x = torch.flatten(out, 1)
        x = self.classifier(conv_pjt_concat)
        return x



class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
        ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(in_ch: int, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = in_ch
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = BAF_VGG(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', True, pretrained, progress, **kwargs)
