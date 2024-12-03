import torch
import torch.nn as nn
import torch.optim as optim


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


class BodyPartCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        """
        Lightweight CNN model for individual body parts.

        Args:
            input_channels (int): Number of input channels (default 3 for RGB).
            num_classes (int): Number of output classes (default 1 for binary classification).
        """
        super(BodyPartCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downscale by 2

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downscale by 2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)  # Binary classification logits


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        y = x.view(batch_size, channels, -1).mean(dim=2)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch_size, channels, 1, 1)
        return x * y


class CustomCNNWithAttention(nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.middle = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SqueezeExcitation(128),
            nn.MaxPool2d(2, 2)
        )
        self.final = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SqueezeExcitation(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.middle(x)
        x = self.final(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)
