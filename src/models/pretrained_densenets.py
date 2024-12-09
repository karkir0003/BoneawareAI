import torch.nn as nn
from torchvision.models import densenet169, densenet121
from enum import Enum

class PretrainedDenseNetVersion(Enum):
    DENSENET_121 = "densenet121"
    DENSENET_169 = "densenet169"


class PretrainedDenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        variant: PretrainedDenseNetVersion = PretrainedDenseNetVersion.DENSENET_169,
    ):
        super(PretrainedDenseNet, self).__init__()

        if variant == PretrainedDenseNetVersion.DENSENET_169:
            self.model = densenet169(pretrained=pretrained)
        elif variant == PretrainedDenseNetVersion.DENSENET_121:
            self.model = densenet121(pretrained=pretrained)
        else:
            raise ValueError(
                f"Invalid variant: {variant}. Choose from {list(PretrainedDenseNetVersion)}"
            )

        for param in self.model.parameters():
            param.requires_grad = False

        final_fc_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(final_fc_in_features, num_classes)

    def forward(self, x):
        preds = self.model(x)
        return preds.squeeze(-1)