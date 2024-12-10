from enum import Enum
import torch
import torch.nn as nn
from torchvision.models import vgg11, vgg13


class VGGPretrainedVersion(Enum):
    VGG11 = "vgg11"
    VGG13 = "vgg13"


def get_vgg_pretrained(num_classes, model=VGGPretrainedVersion.VGG13, pretrained=True):
    if model == VGGPretrainedVersion.VGG11:
        return VGG11(num_classes, pretrained)
    elif model == VGGPretrainedVersion.VGG13:
        return VGG13(num_classes, pretrained)
    else:
        raise ValueError(f"Invalid VGG version: {model}")


# VGG11 Pretrained model
class VGG11(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(VGG11, self).__init__()
        self.model = vgg11(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


# VGG13 Pretrained model
class VGG13(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(VGG13, self).__init__()
        self.model = vgg13(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
