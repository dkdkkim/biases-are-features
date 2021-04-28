import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class convnet(nn.Module):
    def __init__(self,num_classes=10):
        super(convnet,self).__init__()
#        self.bn0     = nn.BatchNorm2d(3)
        self.conv1   = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
#        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        feat_out = x  
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7

        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0),-1)
        y_low = self.fc(feat_low)

        return y_low

