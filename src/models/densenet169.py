import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.batchnorm = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

        init.xavier_uniform_(self.conv.weight, gain=init.calculate_gain("relu"))

    def forward(self, x):
        conv_out = self.conv(x)
        batchnorm_out = self.batchnorm(conv_out)
        out = self.relu(batchnorm_out)
        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(2)

        init.xavier_uniform_(self.conv.weight, gain=init.calculate_gain("relu"))

    def forward(self, x):
        conv_out = self.conv(x)
        batchnorm_out = self.batchnorm(conv_out)
        relu_out = F.relu(batchnorm_out)
        out = self.pool(relu_out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(input_channels + i * growth_rate, growth_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, num_classes=1):
        super(DenseNet169, self).__init__()

        self.first_conv = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.first_batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.first_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init.xavier_uniform_(self.first_conv.weight, gain=init.calculate_gain("relu"))

        self.first_block = DenseBlock(6, 64, 32)
        self.first_transition = TransitionLayer(64 + 6 * 32, 128)

        self.second_block = DenseBlock(12, 128, 32)
        self.second_transition = TransitionLayer(128 + 12 * 32, 256)

        self.third_block = DenseBlock(32, 256, 32)
        self.third_transition = TransitionLayer(256 + 32 * 32, 512)

        self.fourth_block = DenseBlock(32, 512, 32)

        self.final_layer = nn.Linear(512 + 32 * 32, num_classes)

        init.xavier_uniform_(self.final_layer.weight, gain=init.calculate_gain("relu"))

    def forward(self, x):
        first_pool_out = self.first_pool(
            F.relu(self.first_batchnorm(self.first_conv(x)))
        )

        first_block_out = self.first_block(first_pool_out)
        first_transition_out = self.first_transition(first_block_out)

        second_block_out = self.second_block(first_transition_out)
        second_transition_out = self.second_transition(second_block_out)

        third_block_out = self.third_block(second_transition_out)
        third_transition_out = self.third_transition(third_block_out)

        fourth_block_out = self.fourth_block(third_transition_out)

        temp_out = F.adaptive_avg_pool2d(fourth_block_out, (1, 1))
        temp_out = torch.flatten(temp_out, 1)

        out = self.final_layer(temp_out)
        return out
