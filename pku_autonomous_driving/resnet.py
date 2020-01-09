from torch import nn
import torch.nn.functional as F
import torchvision


def forward(self, x):
    conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
    conv1 = F.max_pool2d(conv1, 3, stride=2, padding=1)

    feats4 = self.layer1(conv1)
    feats8 = self.layer2(feats4)
    feats16 = self.layer3(feats8)
    feats32 = self.layer4(feats16)

    return feats4, feats8, feats16, feats32


torchvision.models.resnet.ResNet.forward = forward

from torchvision.models.resnet import *
