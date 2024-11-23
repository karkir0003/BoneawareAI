import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate, dropout_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        in_channels = 2 * growth_rate

        # Initial Convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # DenseBlocks with Transition Layers
        blocks = []
        for i in range(num_blocks):
            blocks.append(DenseBlock(num_layers_per_block, in_channels, growth_rate))
            in_channels += num_layers_per_block * growth_rate
            if i != num_blocks - 1:  # No transition after the last block
                out_channels = int(in_channels * reduction)
                blocks.append(TransitionLayer(in_channels, out_channels))
                in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # Classification Layer (output one logit for binary classification)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)  # num_classes = 1 for binary classification
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # Squeeze to get shape [batch_size] (logits)