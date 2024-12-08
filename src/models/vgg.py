from enum import Enum
import torch
import torch.nn as nn


class VGGVersion(Enum):
    VGG11 = "vgg11"
    VGG13 = "vgg13"


class VGG_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG_BLOCK, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VGG_CLASSIFIER(nn.Module):
    def __init__(self, num_classes):
        super(VGG_CLASSIFIER, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# Custom VGG13_BN model no pretrained model
class VGG13(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG13, self).__init__()
        self.sequential = nn.Sequential(
            VGG_BLOCK(3, 64),
            VGG_BLOCK(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(64, 128),
            VGG_BLOCK(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(128, 256),
            VGG_BLOCK(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(256, 512),
            VGG_BLOCK(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(512, 512),
            VGG_BLOCK(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = VGG_CLASSIFIER(num_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Custom VGG11_BN model no pretrained model
class VGG11(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG11, self).__init__()
        self.sequential = nn.Sequential(
            VGG_BLOCK(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(128, 256),
            VGG_BLOCK(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(256, 512),
            VGG_BLOCK(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG_BLOCK(512, 512),
            VGG_BLOCK(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = VGG_CLASSIFIER(num_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_vgg(num_classes, model=VGGVersion.VGG13):
    if model == VGGVersion.VGG13:
        return VGG13(num_classes)
    elif model == VGGVersion.VGG11:
        return VGG11(num_classes)
    else:
        raise ValueError(f"Model {model} not found")
