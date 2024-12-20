import torch.nn as nn
from torchvision.models import resnet18, resnet34

"""
Fine tuning ResNet model for use in Bone Abnormality Classification. 

Sources: 
@misc{
    Chilamkurthy, 
    title={Transfer learning for computer vision tutorial¶}, 
    url={https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html}, 
    journal={Transfer Learning for Computer Vision Tutorial - PyTorch Tutorials 2.5.0+cu124 documentation}, 
    publisher={PyTorch}, 
    author={Chilamkurthy, Sasank}
}

 
"""

from enum import Enum


class ResNetVersion(Enum):
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"


class ResNet(nn.Module):
    def __init__(
        self,
        num_labels: int,
        pretrained: bool = False,
        variant: ResNetVersion = ResNetVersion.RESNET_18,
    ):
        """
        Generate fine tuned resnet model

        Args:
            num_labels (int): How many labels are in the dataset (this helps finetune the model)
            pretrained (bool): Should pretrained weights be used. Default to False
            variant (ResNetVersion): Which variant of Resnet. Default to Resnet 18
        """
        super(ResNet, self).__init__()
        # Initialize the model based on the variant
        if variant == ResNetVersion.RESNET_18:
            self.model = resnet18(pretrained=pretrained)
        elif variant == ResNetVersion.RESNET_34:
            self.model = resnet34(pretrained=pretrained)
        else:
            raise ValueError(
                f"Invalid variant: {variant}. Choose from {list(ResNetVersion)}"
            )

        # freeze the parameters for every layer in resnet model
        for param in self.model.parameters():
            param.requires_grad = False

        # replace final FC layer
        final_fc_in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(final_fc_in_features, num_labels)

    def forward(self, x):
        preds = self.model(x)
        return preds.squeeze(-1)  # Squeeze to get shape [batch_size] (logits)
