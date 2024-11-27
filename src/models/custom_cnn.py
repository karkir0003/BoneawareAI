import torch
import torch.nn as nn


class CustomCNN1(nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=1):
        """
        Custom CNN for binary classification.

        Args:
            dropout_rate (float): Dropout rate after GAP.
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super(CustomCNN1, self).__init__()

        # Initial convolution layers (stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Mid convolution layers
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Final convolution layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classification layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.middle(x)
        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # Squeeze for binary classification logits
