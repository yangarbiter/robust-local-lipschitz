from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wideresnet import *
from .resnet import resnet50, resnet152, resnet101, resnet50_drop50, resnet50_drop20

def ResNet101(n_classes, n_channels):
    return resnet101(pretrained=False, n_channels=n_channels, num_classes=n_classes)

def ResNet50_drop20(n_classes, n_channels):
    return resnet50_drop20(pretrained=False, n_channels=n_channels, num_classes=n_classes)

def ResNet50_drop50(n_classes, n_channels):
    return resnet50_drop50(pretrained=False, n_channels=n_channels, num_classes=n_classes)

def ResNet50(n_classes, n_channels):
    return resnet50(pretrained=False, n_channels=n_channels, num_classes=n_classes)

def ResNet152(n_classes, n_channels):
    return resnet152(pretrained=False, n_channels=n_channels, num_classes=n_classes)

class CNN001(nn.Module):
    def __init__(self, n_classes, n_channels=None):
        super(CNN001, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

class CNN002(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_classes, drop=0.5, n_channels=1):
        super(CNN002, self).__init__()

        self.num_channels = n_channels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

class CNN003(CNN002):
    def __init__(self, n_classes, drop=0.5, n_channels=1):
        super().__init__(n_classes=n_classes, drop=0.5, n_channels=1)
        self.gamma_var = nn.Parameter(torch.ones(1), requires_grad=True)

class MLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x

class LargeMLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv2(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv2, self).__init__()
        self.hidden = nn.Linear(n_features[0], 384)
        self.hidden2 = nn.Linear(384, 384)
        self.hidden3 = nn.Linear(384, 384)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x
