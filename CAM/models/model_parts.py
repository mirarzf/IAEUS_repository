""" Parts of the EfficientNet model """

import torch
import torch.nn as nn
import torch.nn.functional as F


## ResNet parts 

class ResidualBlock(nn.Module): 
    """resultinglayer = f(previouslayer) + layer"""
    def __init__(self, in_channels, out_channels, padding = 0): 
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels: 
            first_stride = 2 
            self.downsampled_shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=first_stride), 
                nn.BatchNorm2d(in_channels)
            )
        else: 
            first_stride = 1
            self.downsampled_shortcut = None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=padding)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)
    
    def forward(self, x): 
        out = self.conv1(x)
        out = self.normalization(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.normalization(out)
        if self.downsampled_shortcut == None: 
            out += x 
        else: 
            out += self.downsampled_shortcut(x)
        out = self.relu(out)

        return out 

    
class BottleneckResidualBlock(nn.Module): 
    """resultinglayer = f(previouslayer) + layer"""
    def __init__(self, in_channels, out_channels, padding = 0): 
        super(BottleneckResidualBlock, self).__init__()
        if in_channels != out_channels: 
            first_stride = 2 
            self.downsampled_shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=first_stride), 
                nn.BatchNorm2d(in_channels)
            )
        else: 
            first_stride = 1
            self.downsampled_shortcut = None
        mid_channels = out_channels // 2 
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=first_stride, padding=padding)
        self.normalization1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=padding)
        self.normalization3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x): 
        out = self.conv1(x)
        out = self.normalization1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.normalization3(out)
        if self.downsampled_shortcut == None: 
            out += x 
        else: 
            out += self.downsampled_shortcut(x)
        out = self.relu(out)

        return out 


## EfficientNet parts 

class ConvBNRelu6(nn.Module): 
    """convolution => [BN] => ReLU6"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, norm_layer=None):
        super(ConvBNRelu6, self).__init__()
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.convbnrelu6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x): 
        return self.convbnrelu6(x)
