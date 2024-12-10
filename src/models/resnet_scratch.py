import torch.nn as nn


"""
Residual Neural Network Architecture as per the paper "Deep Residual Learning for Image Recognition"

Citations:

@misc{1512.03385,
Author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
Title = {Deep Residual Learning for Image Recognition},
Year = {2015},
Eprint = {arXiv:1512.03385},
}


@misc{Ruiz_2019, 
title={Understanding and visualizing ResNets}, 
url={https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8},
journal={Medium}, 
publisher={Towards Data Science}, 
author={Ruiz, Pablo}, 
year={2019}, 
month={Apr}
} 

@misc{1502.03167,
Author = {Sergey Ioffe and Christian Szegedy},
Title = {Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
Year = {2015},
Eprint = {arXiv:1502.03167},
}
"""
from enum import Enum

class MyResNetVersion(Enum):
    RESNET_18 = 'resnet18'
    RESNET_34 = 'resnet34'

class InitialConvLayer(nn.Module):
    def __init__(self):
        """
        Initial Conv Layer for ResNet. Note that after each conv layer, the paper applies Batch Norm followed by ReLU.
        Reasoning the authors mention comes from the BatchNorm paper (Batch Normalization: Accelerating Deep Network
        Training by Reducing Internal Covariate Shift)
        
        Flow:
          * Input dims (224x224), which is compatible with our dataset
          * Conv Layer: 7x7 kernel, stride 2, 64 channels, padding = 3
          * BatchNorm2d
          * ReLU activation
          * Output dims: 112x112
          
          * MaxPool2d - 3x3 kernel, stride 2, padding = 1
          * Output dims: 56 x 56
        """
        super(InitialConvLayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) #input image has 3 channels
        self.batch_norm_one = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        self.max_pool_one = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        conv = self.conv_1(x)
        normed = self.batch_norm_one(conv)
        relud = self.relu_1(normed)
        output = self.max_pool_one(relud)
        return output 

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Basic block implemented as per the ResNet paper. 
        
        This block implements two convolutional layers with a residual connection, meaning
        that the input will be added to output of convolutional layer and then a ReLU is applied 
        to that total
        
        This residual connection helps improve gradient flow by allowing model to learn to use the learned
        feature representations from convolutions or ignore
        
        Flow:
          * Input
          * Conv Layer: 3x3 kernel, stride 1, padding = 1
          * Batch Norm 
          * ReLU
          * Conv Layer: 3x3 kernel, stride 1, padding = 1
          * Batch Norm
          * Input + output  (AKA "identity projection residual connection")
          * ReLU

        Args:
            input_channels (int): Number of channels C in input (input is N x C x H x W)
            output_channels (int): Number of channels in output
        """
        super(BasicBlock, self).__init__()
        # first conv block
        self.first_conv_block = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.first_bn = nn.BatchNorm2d(output_channels)
        self.first_relu = nn.ReLU()
        
        # second conv block 
        self.second_conv_block = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.second_bn = nn.BatchNorm2d(output_channels)
        
        # relu for adding nonlinearity to residual connection
        self.second_relu = nn.ReLU()
    
    def forward(self, x):
        # run first conv block
        first_conv = self.first_conv_block(x)
        first_block_result = self.first_relu(self.first_bn(first_conv))
        
        # run second conv block
        second_conv =  self.second_conv_block(first_block_result)
        second_block_result = self.second_bn(second_conv)
        
        # append identity residual connection
        result = x + second_block_result
        result = self.second_relu(result)
        return result

class ProjectionLayerBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Basic block projection layer. This layer helps handle adding 
        residual connection when number of channels increases from one basic block 
        to another
        
        The ResNet paper offers a few ways to handle implementation of the residual connection:
        * Zero padding the input
        * 1x1 convolution (followed by batch norm)
        
        For simplicity, we will go with the 1x1 convolution followed by batch norm. 
        
        NOTE: The projection layer for Basic Block also does downsampling to reduce output dimension, but the 
        1x1 convolution can be thought of like an elementwise dot product pixel by pixel
        
        Flow:
          Path 1:
          * Conv Layer: 3x3 kernel, stride = 2, padding = 1
          * Batch Norm 2d
          * ReLU
          * Conv Layer: 3x3 kernel, stride = 1, padding = 1
          * Batch Norm 2d
          
          Path 2: 
          * Conv Layer: 1x1 kernel, stride = 2, padding = 1
          * Batch Norm 2d
          
          Path 1 + Path 2 (same as basic block, but instead of doing x + F(x), we do proj(x) + F(x))
          
          ReLU
        
        Args:
            input_channels (int): Number of channels C in input (input is N x C x H x W)
            output_channels (int): Number of channels in output
        """
        super(ProjectionLayerBasicBlock, self).__init__()
        
        # projection shortcut block (notice the downsampling happening here)
        self.proj_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2)
        self.proj_bn = nn.BatchNorm2d(output_channels)
        
        # downsampling block
        self.downsample_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size = 3, stride = 2, padding = 1)
        self.batch_norm_one = nn.BatchNorm2d(output_channels)
        self.relu_one = nn.ReLU()
        
        # second block
        self.second_conv_block = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.second_bn = nn.BatchNorm2d(output_channels)
        
        # relu for adding nonlinearity to residual connection
        self.second_relu = nn.ReLU()
    
    def forward(self, x):
        # run basic block route
        
        first_block_result = self.relu_one(self.batch_norm_one(self.downsample_conv(x)))
        
        basic_block_result = self.second_bn(self.second_conv_block(first_block_result))
        
        # projection shortcut
        projection_result = self.proj_bn(self.proj_conv(x))
        
        result = projection_result + basic_block_result
        
        result = self.second_relu(result)
        
        return result

class FinalLayer(nn.Module):
    def __init__(self, num_channels, num_classes):
        """
        After chaining several basic blocks, we need to convert output into a dimension corresponding to num classes
        
        Flow:
        * Avg Pool 2d (use adaptive avg pool 2d since this function will auto-determine the kernel size given output size)
        * Fully Connected Layer to transform result into N x num_classes
        """
        super(FinalLayer, self).__init__()
        # the table in ResNet paper shows avg pool layer should give output size 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(num_channels, num_classes)
    
    def forward(self, x):
        pooled = self.avg_pool(x) #(N, C, 1, 1)
        pooled = self.flatten(pooled)
        return self.fc(pooled)

class MyResNet18(nn.Module):
    def __init__(self, num_classes):
        """
        ResNet 18 arch:
        * 1 Initial Layer
        * 2 basic block layers (64 channel)
        * 1 projection basic block (128 channel)
        * 1 basic block layers (128 channel)
        * 1 projection basic block (256 channel)
        * 1 basic block layers (256 channel)
        * 1 projection basic block (512 channel)
        * 1 basic block layers (512 channel)
        * 1 final layer 
        
        Args
          num_classes (int): Number of classes to predict (for binary, set as 1)
        """
        super(MyResNet18, self).__init__()
        self.layers = []
        self.layers.append(InitialConvLayer())
        
        # 2 basic block 64 channel
        for i in range(2):
            self.layers.append(BasicBlock(64, 64))
        
        # 64 channel --> 128 channel projection
        self.layers.append(ProjectionLayerBasicBlock(64, 128))
        
        # 1 basic block 128 channel
        self.layers.append(BasicBlock(128, 128))
        
        # 128 channel --> 256 channel projection
        self.layers.append(ProjectionLayerBasicBlock(128, 256))
        
        # 1 basic block 256 channel
        self.layers.append(BasicBlock(256, 256))
        
        # 256 channel --> 512 channel projection
        self.layers.append(ProjectionLayerBasicBlock(256, 512))
        
        # 1 basic block 512 channel
        self.layers.append(BasicBlock(512, 512))
        
        # final layer
        self.layers.append(FinalLayer(512, num_classes))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        preds = self.layers(x)
        return preds.squeeze(-1)
        

class MyResNet34(nn.Module):
    def __init__(self, num_classes):
        """
        ResNet 34 arch:
        * 1 Initial Layer
        * 3 basic block layers (64 channel)
        * 1 projection basic block (128 channel)
        * 3 basic block layers (128 channel)
        * 1 projection basic block (256 channel)
        * 5 basic block layers (256 channel)
        * 1 projection basic block (512 channel)
        * 2 basic block layers (512 channel)
        * 1 final layer 
        
        Args
          num_classes (int): Number of classes to predict (for binary, set as 1)
        """
        super(MyResNet34, self).__init__()
        self.layers = []
        self.layers.append(InitialConvLayer())
        
        # 3 basic block 64 channel
        for i in range(3):
            self.layers.append(BasicBlock(64, 64))
        
        # 64 channel --> 128 channel projection
        self.layers.append(ProjectionLayerBasicBlock(64, 128))
        
        # 3 basic block 128 channel
        for i in range(3):
            self.layers.append(BasicBlock(128, 128))
        
        # 128 channel --> 256 channel projection
        self.layers.append(ProjectionLayerBasicBlock(128, 256))
        
        # 5 basic block 256 channel
        for i in range(5):
            self.layers.append(BasicBlock(256, 256))
        
        # 256 channel --> 512 channel projection
        self.layers.append(ProjectionLayerBasicBlock(256, 512))
        
        # 2 basic block 512 channel
        for i in range(2):
            self.layers.append(BasicBlock(512, 512))
        
        # final layer
        self.layers.append(FinalLayer(512, num_classes))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        preds = self.layers(x)
        return preds.squeeze(-1)

def init_weights(model: nn.Module):
    """
    Initializes the weights of the model using Xavier initialization.
    Applies Xavier initialization to Conv2d and Linear layers in the model's layers and building blocks.
    
    Args:
        model (nn.Module): The model whose weights are to be initialized.
    """
    # A helper function to initialize weights recursively
    def initialize_layer_weights(layer):
        if isinstance(layer, nn.Conv2d):
            # Apply Xavier uniform initialization to Conv2d layers
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            # Apply Xavier uniform initialization to Linear layers
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    # Traverse the model layers recursively to apply weight initialization
    def traverse_modules(modules):
        for module in modules:
            if isinstance(module, nn.Module):
                # Initialize weights for the current module if it's a Conv2d or Linear layer
                initialize_layer_weights(module)
                # If the module contains other layers, recursively initialize them
                if isinstance(module, nn.Sequential):
                    traverse_modules(module)
                elif hasattr(module, 'children') and module.children():
                    traverse_modules(module.children())
    
    # Apply the weight initialization to all layers in the model
    traverse_modules(model.layers)
    

def get_resnet(num_classes, variant=MyResNetVersion.RESNET_18):
    if variant == MyResNetVersion.RESNET_18:
        return MyResNet18(num_classes)
    elif variant == MyResNetVersion.RESNET_34:
        return MyResNet34(num_classes)
    else:
        raise ValueError(f"Invalid variant: {variant}. Choose from {list(MyaResNetVersion)}")